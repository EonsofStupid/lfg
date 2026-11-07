from __future__ import annotations

import http.server
import json
import os
import shlex
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import compare_local_12b as comparison

TEST_MODEL = "fixture/model:development-tag"


class ComparisonAdapterTests(unittest.TestCase):
    def test_remote_snapshot_fails_closed_when_models_are_loaded(self):
        config = {"server_binary_path": "/opt/ollama", "remote_home": "/home/lfg",
                  "remote_system_service": "ollama", "remote_systemctl_path": "/usr/bin/systemctl",
                  "remote_nvidia_smi_path": "/usr/bin/nvidia-smi",
                  "remote_expected_hostname": "worker", "remote_uid": 1000}
        snapshot = {"hostname": "worker", "architecture": "x86_64", "uid": 1000,
                    "netns_inode": 10, "memory_available_bytes": 100,
                    "home_disk_free_bytes": 100, "loaded_model_count": 1}
        result = mock.Mock(returncode=0, stdout=json.dumps(snapshot).encode("utf-8"))
        with mock.patch.object(comparison, "_ssh_run", return_value=result):
            with self.assertRaisesRegex(ValueError, "safe coexistence is not established"):
                comparison._remote_snapshot(config)

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
                self.rfile.read(int(self.headers["Content-Length"]))
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

    def test_total_deadline_applies_to_slow_drip_headers(self):
        class SlowHeaders(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.connection.sendall(b"HTTP/1.1 200 OK\r\n")
                self.connection.sendall(b"Content-Type: application/json\r\n")
                time.sleep(0.06)
                try:
                    self.connection.sendall(b"Content-Length: 2\r\n")
                    time.sleep(0.06)
                    self.connection.sendall(b"\r\n{}")
                except OSError:
                    pass

            def log_message(self, fmt, *args):
                return

        server = http.server.HTTPServer(("127.0.0.1", 0), SlowHeaders)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        config = {"endpoint_url": f"http://127.0.0.1:{server.server_port}/api/chat",
                  "timeout_seconds": 0.1, "max_request_bytes": 1024, "max_response_bytes": 1024}
        started = time.monotonic()
        try:
            with self.assertRaises(TimeoutError):
                comparison._http_call(config, b"{}")
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_request_contains_only_configured_schema_and_screened_fixture(self):
        config = {"system_prompt": "Choose from {labels}; abstain when unsure. Text: {text}",
                  "model_name": TEST_MODEL, "keep_alive": "5m",
                  "num_ctx": 4096, "num_predict": 64, "temperature": 0.0, "seed": 1,
                  "top_k": 40, "top_p": 0.9, "repeat_penalty": 1.0}
        schema = comparison.output_schema(["build.software"])
        raw = comparison._schema_request(config, schema, "already-screened fixture")
        request = json.loads(raw)
        self.assertEqual(request["model"], TEST_MODEL)
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
                body = json.dumps({"models": [{"name": TEST_MODEL,
                                                "digest": "sha256:" + "a" * 64,
                                                "size": 1200}]}).encode()
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
                  "max_response_bytes": 4096, "model_name": TEST_MODEL}
        artifact = {"manifest_sha256": "a" * 64, "model_layer_bytes": 1200}
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

    def test_model_responses_are_not_replaced_by_constant_target_fallback(self):
        predictions = comparison._response_prediction_matrix([[1], [None], [None]], ["only-label"], 3)
        self.assertEqual(predictions, [[1], [-1], [-1]])
        outcomes = ["answered", "timeout_unknown_worker_lifecycle", "not_attempted_after_abort"]
        self.assertEqual(comparison._http_requests_started(outcomes), 2)

    def test_attempted_timing_excludes_not_started_zeroes(self):
        summary = comparison._stats([12.0, None, 18.0, None])
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["missing_count"], 2)
        self.assertEqual(summary["median_ms"], 15.0)

    def test_final_timing_list_must_match_started_and_not_started_progress(self):
        outcomes = ["answered", "not_attempted_after_abort"]
        progress = [{"index": 0, "outcome": "answered", "timing_ms": 12.0},
                    {"index": 1, "outcome": "not_attempted_after_abort", "timing_ms": None}]
        comparison._validate_final_timings(outcomes, [12.0, None], progress, 2)
        for timings in ([13.0, None], [12.0], [float("nan"), None], [12.0, 0.0]):
            with self.subTest(timings=timings), self.assertRaisesRegex(ValueError, "final timing"):
                comparison._validate_final_timings(outcomes, list(timings), progress, 2)

    def test_missing_baseline_review_and_history_are_reported_unknown(self):
        review = comparison._baseline_review_provenance({"data": {}})
        self.assertIsNone(review["label_review_state"])
        self.assertIsNone(review["all_rows_marked_verified"])
        self.assertEqual(review["validation_group_use_claim"], "unknown")
        self.assertIsNone(review["validation_group_use_claim_verified"])
        self.assertEqual(review["validation_independence_claim"], "unknown; no baseline claim supplied")
        self.assertIsNone(review["validation_independence_verified"])
        self.assertTrue(any("history is unknown" in item for item in
                            comparison._review_history_limitations(review)))

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
            "model_store": "/tmp/model", "model_name": TEST_MODEL,
            "server_binary_sha256": "d" * 64, "server_version": "0.30.11",
            "endpoint_url": "http://127.0.0.1:11434/api/chat", "keep_alive": "10m",
            "max_report_bytes": 10000, "max_baseline_report_bytes": 10000,
            "max_source_bytes": 10000, "max_total_source_bytes": 20000,
            "max_manifest_bytes": 10000, "max_screened_file_bytes": 10000,
            "max_policy_bytes": 10000, "max_ledger_bytes": 10000,
            "max_model_manifest_bytes": 10000, "max_model_blob_bytes": 999999,
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
            "expected_upstream_revision": "1" * 40,
            "expected_registry_manifest_url": "https://registry.ollama.ai/v2/fixture/model/manifests/development-tag",
            "expected_manifest_sha256": "e" * 64, "expected_model_blob_sha256": "f" * 64,
            "expected_model_blob_bytes": 900000,
            "expected_license_name": "Fixture License", "expected_quantization": "fixture format",
            "remote_home": "/home/hades", "remote_expected_hostname": "clyffy-03",
            "remote_expected_architecture": "aarch64", "remote_system_service": "ollama.service",
            "remote_systemctl_path": "/usr/bin/systemctl", "remote_sudo_path": "/usr/bin/sudo",
            "remote_systemd_run_path": "/usr/bin/systemd-run", "remote_unshare_path": "/usr/bin/unshare",
            "remote_bash_path": "/usr/bin/bash", "remote_ip_path": "/usr/sbin/ip",
            "remote_setpriv_path": "/usr/bin/setpriv", "remote_true_path": "/usr/bin/true",
            "remote_nvidia_smi_path": "/usr/bin/nvidia-smi", "remote_scope_tasks_max": 512,
            "worker_ready_wait_margin_seconds": 90, "artifact_install_timeout_seconds": 600,
            "batch_overhead_margin_seconds": 120,
            "operator_extension": {"future_field": "ignored"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            loaded, _ = comparison.load_config(path)
        self.assertEqual(loaded["operator_extension"], {"future_field": "ignored"})
        self.assertEqual(loaded["model_name"], TEST_MODEL)
        self.assertEqual(loaded["expected_manifest_sha256"], "e" * 64)
        self.assertEqual(loaded["remote_scope_tasks_max"], 512)

        config.update({
            "model_name": "team/model:version-2-q8_0",
            "expected_upstream_revision": "2" * 64,
            "expected_registry_manifest_url": "https://registry.ollama.ai/v2/team/model/manifests/version-2-q8_0",
            "expected_manifest_sha256": "1" * 64,
            "expected_model_blob_sha256": "2" * 64,
            "expected_model_blob_bytes": 900000,
            "expected_license_name": "Alternate License",
            "expected_quantization": "Q8_0",
            "resource_cpu_quota_percent": 2500,
            "remote_home": "/srv/worker home",
            "remote_acquisition_receipt": "/srv/worker home/receipt file.json",
            "remote_workdir": "/srv/worker home/runs",
            "remote_model_store": "/srv/worker home/models",
            "remote_adapter_path": "/srv/worker home/runs/compare_local_12b.py",
            "remote_python_path": "/opt/python with space/bin/python3",
            "server_binary_path": "/opt/ollama suite/bin/ollama",
            "remote_expected_hostname": "example-host",
            "remote_expected_architecture": "aarch64",
            "remote_system_service": "ollama-alt.service",
            "remote_scope_tasks_max": 768,
            "worker_ready_wait_margin_seconds": 75,
            "artifact_install_timeout_seconds": 450,
            "batch_overhead_margin_seconds": 30,
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alternate-config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            alternate, _ = comparison.load_config(path)
        self.assertEqual(alternate["model_name"], "team/model:version-2-q8_0")
        self.assertEqual(alternate["expected_manifest_sha256"], "1" * 64)
        self.assertEqual(alternate["expected_model_blob_sha256"], "2" * 64)
        self.assertEqual(alternate["expected_quantization"], "Q8_0")
        self.assertEqual(alternate["resource_cpu_quota_percent"], 2500)
        self.assertEqual(alternate["remote_scope_tasks_max"], 768)
        self.assertEqual(alternate["worker_ready_wait_margin_seconds"], 75)
        unsafe_timeout = dict(config)
        unsafe_timeout["batch_overhead_margin_seconds"] = 10**30
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsafe-timeout.json"
            path.write_text(json.dumps(unsafe_timeout), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safe platform timeout arithmetic"):
                comparison.load_config(path)

        for field, value in (("remote_host", "host;touch /tmp/pwn"),
                             ("remote_user", "worker$(id)")):
            unsafe = dict(config)
            unsafe[field] = value
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "unsafe-config.json"
                path.write_text(json.dumps(unsafe), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "SSH user or host"):
                    comparison.load_config(path)

    def test_remote_worker_command_quotes_space_paths_and_rejects_metacharacters(self):
        config = {"remote_sudo_path": "/usr/bin/sudo", "remote_systemd_run_path": "/usr/bin/systemd-run",
                  "remote_unshare_path": "/usr/bin/unshare", "remote_bash_path": "/usr/bin/bash",
                  "remote_ip_path": "/usr/sbin/ip", "remote_setpriv_path": "/usr/bin/setpriv",
                  "remote_python_path": "/opt/python with space/bin/python3", "remote_uid": 1001,
                  "remote_user": "worker", "remote_host": "example-host", "remote_home": "/srv/worker home",
                  "resource_memory_max_bytes": 4096, "resource_cpu_quota_percent": 2500,
                  "remote_scope_tasks_max": 768}
        command = comparison._worker_remote_command(
            config, "/srv/worker home/runs/compare script.py", "/srv/worker home/runs/config file.json",
            "lfg12b-" + "a" * 20 + ".scope")
        outer = shlex.split(command)
        inner = shlex.split(outer[-1])
        self.assertIn("/opt/python with space/bin/python3", inner)
        self.assertIn("/srv/worker home/runs/compare script.py", inner)
        self.assertIn("TasksMax=768", command)
        unsafe = {**config, "remote_python_path": "/usr/bin/python3;touch /tmp/pwn"}
        with self.assertRaisesRegex(ValueError, "unsafe remote executable"):
            comparison._worker_remote_command(unsafe, "/srv/worker home/adapter.py",
                                              "/srv/worker home/config.json", "lfg12b-" + "b" * 20 + ".scope")

    def test_ssh_script_command_quotes_interpreter_and_rejects_destination_injection(self):
        config = {"remote_python_path": "/opt/python with space/bin/python3", "remote_user": "worker",
                  "remote_host": "example-host", "ssh_port": 2222}
        with mock.patch.object(comparison.subprocess, "run") as run_mock:
            comparison._ssh_run(config, "print('fixture')")
        command = run_mock.call_args.args[0][-1]
        self.assertIn("'/opt/python with space/bin/python3'", command)
        self.assertIn("-I", command)
        unsafe = {**config, "remote_host": "example-host;touch /tmp/pwn"}
        with mock.patch.object(comparison.subprocess, "run") as run_mock:
            with self.assertRaisesRegex(ValueError, "unsafe SSH destination"):
                comparison._ssh_run(unsafe, "print('fixture')")
            run_mock.assert_not_called()

    def test_acquisition_identity_uses_configured_alternate_pins(self):
        receipt = {"schema": comparison.RECEIPT_SCHEMA, "outcome": "verified_downloaded_not_loaded",
                   "manifest": {"sha256": "1" * 64, "bytes": 80, "file": "manifest.json"},
                   "objects": [{"digest": "sha256:" + "2" * 64, "bytes": 90,
                                "sha256_verified": True}],
                   "sources": {"upstream_model_revision": "3" * 40,
                               "registry_manifest": "https://registry.ollama.ai/v2/team/model/manifests/tag"},
                   "provenance": {}, "staging_locator": "private fixture locator"}
        raw = json.dumps(receipt, separators=(",", ":")).encode()
        config = {"acquisition_receipt_sha256": comparison.sha(raw), "max_report_bytes": 10000,
                  "expected_manifest_sha256": "1" * 64, "expected_model_blob_sha256": "2" * 64,
                  "expected_model_blob_bytes": 90, "expected_upstream_revision": "3" * 40,
                  "expected_registry_manifest_url": "https://registry.ollama.ai/v2/team/model/manifests/tag"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            path.write_bytes(raw)
            config["acquisition_receipt"] = str(path)
            loaded, digest = comparison.verify_acquisition(config)
        self.assertEqual(digest, comparison.sha(raw))
        self.assertEqual(loaded["manifest"]["sha256"], "1" * 64)
        config["expected_model_blob_sha256"] = "4" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            path.write_bytes(raw)
            config["acquisition_receipt"] = str(path)
            with self.assertRaisesRegex(ValueError, "configured verified model layer"):
                comparison.verify_acquisition(config)

    def test_complete_aggregate_assembly_accepts_real_worker_result_shape(self):
        import numpy as np
        label = "build.software"
        train_rows = [{"id": "train-1"}, {"id": "train-2"}]
        validation_rows = [{"id": "dev-1"}, {"id": "dev-2"}, {"id": "dev-3"}]
        y_train = np.array([[0], [0]])
        train_known = np.array([[True], [True]])
        y_validation = np.array([[1], [0], [0]])
        validation_known = np.array([[True], [True], [True]])
        validation_ids = [comparison.sha(row["id"].encode()) for row in validation_rows]
        train_review_state = {"rows": 2, "target_array_nonempty": 2, "target_array_empty": 0,
                              "target_array_missing_or_null": 0, "verified_true": 2, "verified_false": 0,
                              "labels_pending_true": 0, "labels_pending_false": 2, "labels_pending_absent": 0,
                              "target_completeness_true": 2, "target_completeness_false": 0,
                              "target_completeness_absent": 0}
        validation_review_state = {"rows": 3, "target_array_nonempty": 3, "target_array_empty": 0,
                                   "target_array_missing_or_null": 0, "verified_true": 3, "verified_false": 0,
                                   "labels_pending_true": 0, "labels_pending_false": 3, "labels_pending_absent": 0,
                                   "target_completeness_true": 3, "target_completeness_false": 0,
                                   "target_completeness_absent": 0}
        baseline = {"code_sha256": "1" * 64,
                    "data": {"validation_record_id_sha256": sorted(validation_ids),
                             "plan_sha256": "2" * 64, "review_ledger_sha256": "3" * 64,
                             "rejected_rows_excluded": {"train": 0, "validation": 0},
                             "label_review_state": {"train": train_review_state,
                                                    "validation": validation_review_state},
                             "validation_group_use_claim": "reported_unused",
                             "validation_group_use_claim_verified": False,
                             "independence_claim": "project/group/family and screened-text separation checked against the selected plan; source independence remains unverified"},
                    "task": {"validation_support": {label: {"positive": 1, "negative": 2}}},
                    "algorithms": {name: {"metrics": {"macro_f1": 0.5}} for name in
                                   ("logistic_regression", "knn", "decision_tree")}}
        classical_config = {"target_field": "intents", "selected_labels": [label],
                            "constant_target_behavior": "majority_class", "rejected_label_handling": "exclude"}
        corpus = {"taxonomy_sha256": "4" * 64, "manifest_sha256": "5" * 64,
                  "train_sha256": "6" * 64, "validation_sha256": "7" * 64,
                  "policy_sha256": "8" * 64, "screen_code_sha256": "9" * 64}
        config = {"baseline_report_sha256": "a" * 64, "server_binary_sha256": "b" * 64,
                  "server_version": "0.30.11", "prompt_revision": "fixture-prompt",
                  "model_name": TEST_MODEL, "expected_quantization": "fixture Q4",
                  "expected_license_name": "Fixture License", "worker_code_sha256": "9" * 64,
                  "comparator_git_revision": None,
                  "system_prompt": "labels {labels}; text {text}", "num_ctx": 4096,
                  "num_predict": 64, "temperature": 0.0, "seed": 17, "top_k": 40,
                  "top_p": 0.9, "repeat_penalty": 1.0, "keep_alive": "10m",
                  "max_request_bytes": 500000, "max_response_bytes": 1000000,
                  "timeout_seconds": 30, "resource_memory_max_bytes": 51539607552,
                  "resource_cpu_quota_percent": 200}
        payload = {"records": [{"id_sha256": item} for item in validation_ids]}
        result = {"outcomes": ["answered", "http_error", "not_attempted_after_abort"],
                  "predictions": [[1], [None], [None]],
                  "timing_ms": [12.0, 18.0, None], "aborted": True, "abort_reason": "fixture abort",
                  "artifact": {"manifest_sha256": "5" * 64, "model_layer_bytes": 12345},
                  "loaded_model": {"identity_match": True},
                  "resources_after": {"effective_memory_available_bytes": 40_000_000_000,
                                      "cgroup_memory_available_bytes": 12_000_000,
                                      "gpu_attribution": "device-wide memory readings only"},
                  "server_cleanup_verified": True}
        ready = {"warmup": {"status": "completed"},
                 "resources_before": {"effective_memory_available_bytes": 45_000_000_000,
                                      "cgroup_memory_available_bytes": 15_000_000,
                                      "gpu_attribution": "device memory query unavailable"}}
        host = {"hostname": "clyffy-03", "architecture": "aarch64", "memory_available_bytes": 100,
                "home_disk_free_bytes": 100, "gpu_inventory": None, "ollama_service_pid": "2324",
                "ollama_version": "ollama version is 0.30.11", "loaded_model_count": 0}
        report = comparison._assemble_report(
            config, "c" * 64, {"sources": {"upstream_model_revision": "1" * 40},
                               "provenance": {}}, "d" * 64, baseline, classical_config,
            train_rows, validation_rows, y_train, train_known, y_validation, validation_known,
            corpus, "e" * 64, payload, b"fixture-payload", ready, result, host, host)
        self.assertEqual(report["provenance"]["plan_sha256"], "2" * 64)
        self.assertEqual(report["inference"]["outcome_counts"],
                         {"answered": 1, "http_error": 1, "not_attempted_after_abort": 1})
        self.assertEqual(report["metrics"]["labels"][label]["tp"], 1)
        self.assertEqual(report["metrics"]["labels"][label]["tn"], 0)
        self.assertEqual(report["metrics"]["labels"][label]["unknown_predictions"], 2)
        self.assertEqual(report["metrics"]["prediction_coverage_on_known_targets"], 1 / 3)
        self.assertEqual(report["inference"]["worker_lifecycle"]["unknown_or_ambiguous"], 1)
        self.assertEqual(report["inference"]["worker_lifecycle"]["not_started"], 1)
        self.assertEqual(report["inference"]["timing_ms"]["count"], 2)
        self.assertEqual(report["inference"]["timing_ms"]["missing_count"], 1)
        self.assertEqual(report["task"]["target_scoring_policy"][label]["mode"],
                         "constant_target_majority_fallback")
        self.assertEqual(report["provenance"]["validation_group_use_claim"], "reported_unused")
        self.assertFalse(report["provenance"]["validation_group_use_claim_verified"])
        self.assertTrue(any("reports validation groups unused" in item for item in report["limitations"]))
        self.assertIsNone(report["provenance"]["validation_independence_verified"])
        self.assertTrue(report["provenance"]["all_rows_marked_verified"])
        self.assertEqual(report["provenance"]["label_review_state"]["validation"]["verified_true"], 3)
        self.assertIn("did not collect per-run cgroup", report["resources"]["cgroup_usage_scope"])
        self.assertEqual(report["resources"]["cgroup_memory_available_bytes_before"], 15_000_000)
        self.assertEqual(report["resources"]["gpu_memory_attribution"]["before"],
                         "device memory query unavailable")
        self.assertNotIn("GB10", json.dumps(report))
        self.assertNotIn("labels_unverified", report["provenance"])
        self.assertIsNone(report["resources"]["per_run_cpu_time_usec"])
        self.assertIsNone(report["resources"]["per_run_peak_memory_bytes"])
        self.assertEqual(report["model"]["inference_device"],
                         "unknown; runtime response did not report actual CPU/GPU placement")
        self.assertNotIn("after_existing_service_unchanged", report["host_state"])
        self.assertTrue(report["host_state"]["sampled_service_comparisons"]["ollama_service_pid_matches"])
        self.assertEqual(report["comparator"]["code_sha256"], "9" * 64)
        self.assertNotIn("records", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
