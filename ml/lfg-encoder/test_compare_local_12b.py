from __future__ import annotations

import http.server
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import compare_local_12b as comparison


class ComparisonAdapterTests(unittest.TestCase):
    def test_output_schema_is_dynamic_unique_and_allows_empty_negative(self):
        schema = comparison.output_schema(["build.software", "verify.outcome"])
        self.assertEqual(schema["properties"]["labels"]["items"]["enum"],
                         ["build.software", "verify.outcome"])
        self.assertTrue(schema["properties"]["labels"]["uniqueItems"])
        self.assertEqual(comparison.parse_prediction('{"labels":[],"abstain":false}', True, "stop",
                                                     {"build.software"}), ("answered", []))

    def test_prediction_outcomes_preserve_abstentions_malformed_and_truncation(self):
        allowed = {"build.software", "verify.outcome"}
        cases = [
            ('{"labels":["build.software"],"abstain":false}', True, "stop", "answered", ["build.software"]),
            ('{"labels":[],"abstain":true}', True, "stop", "abstain", None),
            ('{"labels":[],"abstain":false}', True, "stop", "answered", []),
            ('{"labels":["build.software","build.software"],"abstain":false}', True, "stop", "invalid_schema", None),
            ('{"labels":["other"],"abstain":false}', True, "stop", "invalid_schema", None),
            ('{"labels":["build.software"],"abstain":false}', True, "length", "truncated", None),
            ('{"labels":["build.software"],"abstain":false}', False, "length", "incomplete", None),
            ('{"refusal":"cannot classify"}', True, "stop", "refusal", None),
            ('not-json', True, "stop", "malformed", None),
        ]
        for raw, done, reason, expected_status, expected_labels in cases:
            with self.subTest(status=expected_status, reason=reason):
                self.assertEqual(comparison.parse_prediction(raw, done, reason, allowed),
                                 (expected_status, expected_labels))

    def test_loopback_endpoint_rejects_dns_public_https_and_other_paths(self):
        for endpoint in ("http://localhost:11434/api/chat", "http://example.com:11434/api/chat",
                         "https://127.0.0.1:11434/api/chat", "http://127.0.0.1:11434/redirect"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                comparison._loopback_endpoint(endpoint)
        self.assertEqual(comparison._loopback_endpoint("http://127.0.0.1:11434/api/chat"),
                         ("127.0.0.1", 11434, "/api/chat"))

    def test_redirect_is_not_followed(self):
        seen = []

        class Redirect(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append(self.path)
                self.send_response(307)
                self.send_header("Location", "http://127.0.0.1:1/other")
                self.end_headers()

            def log_message(self, fmt, *args):
                return

        server = http.server.HTTPServer(("127.0.0.1", 0), Redirect)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        config = {"endpoint_url": f"http://127.0.0.1:{server.server_port}/api/chat",
                  "timeout_seconds": 2.0, "max_request_bytes": 1024, "max_response_bytes": 1024}
        try:
            with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1",
                                               "HTTPS_PROXY": "http://127.0.0.1:1"}):
                with self.assertRaisesRegex(ValueError, "redirect refused"):
                    comparison._http_call(config, b"{}")
            self.assertEqual(seen, ["/api/chat"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_total_deadline_applies_to_slow_drip_body(self):
        class Slow(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "20")
                self.end_headers()
                self.wfile.write(b"{")
                self.wfile.flush()
                time.sleep(0.25)
                try:
                    self.wfile.write(b"x" * 19)
                    self.wfile.flush()
                except OSError:
                    pass

            def log_message(self, fmt, *args):
                return

        server = http.server.HTTPServer(("127.0.0.1", 0), Slow)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        config = {"endpoint_url": f"http://127.0.0.1:{server.server_port}/api/chat",
                  "timeout_seconds": 0.08, "max_request_bytes": 1024, "max_response_bytes": 1024}
        try:
            with self.assertRaises(TimeoutError):
                comparison._http_call(config, b"{}")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_request_contains_only_configured_schema_and_screened_fixture(self):
        config = {"system_prompt": "Choose from {labels}; abstain when unsure. Text: {text}",
                  "model_name": comparison.PINNED_MODEL, "keep_alive": "5m",
                  "num_ctx": 4096, "num_predict": 64, "temperature": 0.0, "seed": 1,
                  "top_k": 40, "top_p": 0.9, "repeat_penalty": 1.0}
        schema = comparison.output_schema(["build.software"])
        raw = comparison._schema_request(config, schema, "already-screened fixture")
        request = json.loads(raw)
        self.assertEqual(request["model"], comparison.PINNED_MODEL)
        self.assertEqual(request["format"], schema)
        self.assertIn("build.software", request["messages"][0]["content"])
        self.assertIn("already-screened fixture", request["messages"][0]["content"])
        self.assertEqual(request["options"], {"temperature": 0.0, "num_ctx": 4096, "num_predict": 64,
                                               "seed": 1, "top_k": 40, "top_p": 0.9, "repeat_penalty": 1.0})

    def test_runtime_inventory_is_read_from_ps_endpoint_and_checks_identity(self):
        class Inventory(http.server.BaseHTTPRequestHandler):
            paths = []

            def do_GET(self):
                self.paths.append(self.path)
                body = json.dumps({"models": [{"name": comparison.PINNED_MODEL,
                                                "digest": "sha256:" + "a" * 64,
                                                "size": comparison.PINNED_MODEL_BYTES + 1024}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args):
                return

        server = http.server.HTTPServer(("127.0.0.1", 0), Inventory)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        config = {"endpoint_url": f"http://127.0.0.1:{server.server_port}/api/chat",
                  "timeout_seconds": 2.0, "max_request_bytes": 1024,
                  "max_response_bytes": 4096, "model_name": comparison.PINNED_MODEL}
        artifact = {"manifest_sha256": "a" * 64, "model_layer_bytes": comparison.PINNED_MODEL_BYTES}
        try:
            result = comparison.verify_loaded_model(config, artifact)
            self.assertTrue(result["identity_match"])
            self.assertEqual(Inventory.paths, ["/api/ps"])
            self.assertIn("not weight layer size", result["reported_size_scope"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_constant_targets_follow_configured_abstention_or_majority_policy(self):
        import numpy as np
        y_train = np.array([[1, 0], [1, 0], [0, 0]])
        known = np.array([[True, False], [True, False], [True, False]])
        abstain = comparison._target_scoring_policy(y_train, known, ["variable", "unsupported"], "abstain")
        self.assertEqual(abstain["variable"]["mode"], "model_prediction")
        self.assertEqual(abstain["unsupported"]["mode"], "unsupported_abstain")
        positive_train = np.ones((3, 1), dtype=int)
        all_positive = comparison._target_scoring_policy(positive_train, np.ones_like(positive_train, dtype=bool),
                                                         ["always-positive"], "abstain")
        self.assertEqual(all_positive["always-positive"]["mode"], "constant_target_abstain")
        all_negative = comparison._target_scoring_policy(y_train[:, 1:2], np.ones((3, 1), dtype=bool),
                                                         ["always-negative"], "majority_class")
        self.assertEqual(all_negative["always-negative"]["mode"], "constant_target_majority_fallback")
        self.assertEqual(all_negative["always-negative"]["constant_train_value"], 0)

    def test_worker_recomputes_text_identity_and_partial_label_masks(self):
        record_id = "a" * 64
        text = "screened local task"
        known = [True, False]
        binding = {"labels": ["build.software", "verify.outcome"],
                   "validation_record_id_sha256": [record_id],
                   "validation_target_mask_sha256": comparison._mask_digest_from_hashes([record_id], [known])}
        payload = {"schema_version": "lfg.local-12b-worker-input.v1", "binding": binding,
                   "max_request_bytes": 1000,
                   "records": [{"id_sha256": record_id, "text": text,
                                "text_sha256": comparison.sha(text.encode()),
                                "target_values": [1, None], "target_known": known}]}
        config = {"max_request_bytes": 1000}
        labels, records = comparison._validate_worker_payload(payload, config)
        self.assertEqual(labels, binding["labels"])
        self.assertEqual(records[0]["target_values"], [1, None])

        changed_text = json.loads(json.dumps(payload))
        changed_text["records"][0]["text"] = "different"
        with self.assertRaisesRegex(ValueError, "text, target, or digest"):
            comparison._validate_worker_payload(changed_text, config)
        changed_mask = json.loads(json.dumps(payload))
        changed_mask["records"][0]["target_known"] = [True, True]
        changed_mask["records"][0]["target_values"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "target observation mask"):
            comparison._validate_worker_payload(changed_mask, config)

    def test_worker_rejects_unattested_negative_or_bad_id_digest(self):
        record_id = "b" * 64
        text = "fixture"
        known = [False]
        payload = {"schema_version": "lfg.local-12b-worker-input.v1",
                   "binding": {"labels": ["build.software"], "validation_record_id_sha256": [record_id],
                               "validation_target_mask_sha256": comparison._mask_digest_from_hashes([record_id], [known])},
                   "max_request_bytes": 200,
                   "records": [{"id_sha256": record_id, "text": text,
                                "text_sha256": comparison.sha(text.encode()),
                                "target_values": [None], "target_known": known}]}
        self.assertEqual(comparison._validate_worker_payload(payload, {"max_request_bytes": 200})[0],
                         ["build.software"])
        bad = json.loads(json.dumps(payload))
        bad["records"][0]["id_sha256"] = "private-id"
        with self.assertRaisesRegex(ValueError, "text, target, or digest"):
            comparison._validate_worker_payload(bad, {"max_request_bytes": 200})

    def test_metrics_match_classical_observed_target_semantics(self):
        import numpy as np
        import classifier_comparison as classical
        y_true = np.array([[1, 0], [0, 0], [1, 0]])
        y_pred = np.array([[1, -1], [1, 0], [-1, 1]])
        known = np.array([[True, False], [True, True], [True, True]])
        expected = classical._metrics(y_true, y_pred, known, ["build.software", "verify.outcome"])
        actual = comparison._compute_metrics(y_true.tolist(), y_pred.tolist(), known.tolist(),
                                             ["build.software", "verify.outcome"])
        for field in ("subset_accuracy_on_fully_scored", "scored_label_predictions", "target_coverage",
                      "prediction_coverage_on_known_targets", "macro_precision", "macro_recall", "macro_f1",
                      "micro_precision_on_covered", "micro_recall_on_covered", "micro_f1_on_covered"):
            self.assertEqual(actual[field], expected[field], field)

    def test_config_accepts_inert_extension_fields(self):
        config = {
            "schema_version": comparison.SCHEMA, "revision": "fixture",
            "baseline_report": "unused", "baseline_report_sha256": "a" * 64,
            "classical_config": "unused", "classical_config_sha256": "b" * 64,
            "acquisition_receipt": "unused", "acquisition_receipt_sha256": "c" * 64,
            "remote_acquisition_receipt": "/home/hades/receipt.json",
            "model_store": "/tmp/model", "model_name": comparison.PINNED_MODEL,
            "server_binary_sha256": "d" * 64, "server_version": "0.30.11",
            "endpoint_url": "http://127.0.0.1:11434/api/chat", "keep_alive": "10m",
            "max_report_bytes": 10000, "max_baseline_report_bytes": 10000,
            "max_source_bytes": 10000, "max_total_source_bytes": 20000,
            "max_manifest_bytes": 10000, "max_screened_file_bytes": 10000,
            "max_policy_bytes": 10000, "max_ledger_bytes": 10000,
            "max_model_manifest_bytes": 10000, "max_model_blob_bytes": comparison.PINNED_MODEL_BYTES,
            "hash_chunk_bytes": 65536, "max_server_binary_bytes": 100000000,
            "min_memory_available_bytes": 1000000000, "min_disk_free_bytes": 1000000000,
            "min_gpu_free_bytes": 0, "gpu_memory_policy": "shared_unified_memory_unavailable",
            "num_ctx": 4096, "num_predict": 32, "timeout_seconds": 10.5,
            "max_request_bytes": 10000, "max_response_bytes": 10000, "max_payload_bytes": 20000,
            "system_prompt": "labels {labels}; text {text}", "prompt_revision": "v1",
            "seed": 1, "temperature": 0.0, "top_k": 40, "top_p": 0.9,
            "repeat_penalty": 1.0, "remote_host": "clyffy-03", "remote_user": "hades",
            "remote_uid": 1000, "remote_workdir": "/home/hades/.local/share/lfg-local-12b",
            "remote_model_store": "/home/hades/.local/share/lfg-local-12b-models",
            "remote_adapter_path": "/home/hades/.local/share/lfg-local-12b/compare_local_12b.py",
            "remote_python_path": "/usr/bin/python3", "server_binary_path": "/usr/local/bin/ollama",
            "runtime_port": 11435, "ssh_port": 22, "worker_start_timeout_seconds": 900,
            "resource_memory_max_bytes": 51539607552, "resource_cpu_quota_percent": 200,
            "max_server_binary_bytes": 100000000, "max_payload_bytes": 20000,
            "operator_extension": {"future_field": "ignored"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            loaded, _ = comparison.load_config(path)
        self.assertEqual(loaded["operator_extension"], {"future_field": "ignored"})

    def test_complete_aggregate_assembly_accepts_real_worker_result_shape(self):
        import numpy as np
        label = "build.software"
        train_rows = [{"id": "train-1"}, {"id": "train-2"}]
        validation_rows = [{"id": "dev-1"}, {"id": "dev-2"}]
        y_train = np.array([[1], [0]])
        train_known = np.array([[True], [True]])
        y_validation = np.array([[1], [0]])
        validation_known = np.array([[True], [True]])
        validation_ids = [comparison.sha(row["id"].encode()) for row in validation_rows]
        baseline = {"code_sha256": "1" * 64,
                    "data": {"validation_record_id_sha256": sorted(validation_ids),
                             "plan_sha256": "2" * 64, "review_ledger_sha256": "3" * 64,
                             "rejected_rows_excluded": {"train": 0, "validation": 0}},
                    "task": {"validation_support": {label: {"positive": 1, "negative": 1}}},
                    "algorithms": {name: {"metrics": {"macro_f1": 0.5}} for name in
                                   ("logistic_regression", "knn", "decision_tree")}}
        classical_config = {"target_field": "intents", "selected_labels": [label],
                            "constant_target_behavior": "abstain", "rejected_label_handling": "exclude"}
        corpus = {"taxonomy_sha256": "4" * 64, "manifest_sha256": "5" * 64,
                  "train_sha256": "6" * 64, "validation_sha256": "7" * 64,
                  "policy_sha256": "8" * 64, "screen_code_sha256": "9" * 64}
        config = {"baseline_report_sha256": "a" * 64, "server_binary_sha256": "b" * 64,
                  "server_version": "0.30.11", "prompt_revision": "fixture-prompt",
                  "system_prompt": "labels {labels}; text {text}", "num_ctx": 4096,
                  "num_predict": 64, "temperature": 0.0, "seed": 17, "top_k": 40,
                  "top_p": 0.9, "repeat_penalty": 1.0, "keep_alive": "10m",
                  "max_request_bytes": 500000, "max_response_bytes": 1000000,
                  "timeout_seconds": 30, "resource_memory_max_bytes": 51539607552,
                  "resource_cpu_quota_percent": 200}
        payload = {"records": [{"id_sha256": item} for item in validation_ids]}
        result = {"outcomes": ["answered", "abstain"], "predictions": [[1], [None]],
                  "timing_ms": [12.0, 18.0], "aborted": False, "abort_reason": None,
                  "artifact": {"manifest_sha256": "5" * 64},
                  "loaded_model": {"identity_match": True},
                  "resources_after": {"effective_memory_available_bytes": 40_000_000_000},
                  "server_cleanup_verified": True}
        ready = {"warmup": {"status": "completed"},
                 "resources_before": {"effective_memory_available_bytes": 45_000_000_000}}
        host = {"hostname": "clyffy-03", "architecture": "aarch64", "memory_available_bytes": 100,
                "home_disk_free_bytes": 100, "gpu_inventory": None, "ollama_service_pid": "2324",
                "ollama_version": "ollama version is 0.30.11", "loaded_model_count": 0}
        report = comparison._assemble_report(
            config, "c" * 64, {"sources": {"upstream_model_revision": comparison.PINNED_UPSTREAM_REVISION},
                               "provenance": {}}, "d" * 64, baseline, classical_config,
            train_rows, validation_rows, y_train, train_known, y_validation, validation_known,
            corpus, "e" * 64, payload, b"fixture-payload", ready, result, host)
        self.assertEqual(report["provenance"]["plan_sha256"], "2" * 64)
        self.assertEqual(report["inference"]["outcome_counts"], {"abstain": 1, "answered": 1})
        self.assertEqual(report["metrics"]["labels"][label]["tp"], 1)
        self.assertEqual(report["metrics"]["prediction_coverage_on_known_targets"], 0.5)
        self.assertEqual(report["inference"]["worker_lifecycle"]["unknown_or_ambiguous"], 0)
        self.assertIsNone(report["resources"]["per_run_cpu_time_usec"])
        self.assertIsNone(report["resources"]["per_run_peak_memory_bytes"])
        self.assertEqual(report["model"]["inference_device"],
                         "unknown; runtime response did not report actual CPU/GPU placement")
        self.assertNotIn("records", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
