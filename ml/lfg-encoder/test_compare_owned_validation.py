"""Contract tests for the matched owned evaluator; fixtures contain no private prompts."""
from __future__ import annotations

import http.client
import json
import os
import shlex
import subprocess
import sys
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import compare_owned_validation as compare


class _Response:
    def __init__(self, status: int, body: bytes = b"{}", headers=()):
        self.status = status
        self._body = body
        self._headers = list(headers)

    def getheaders(self):
        return self._headers

    def read(self, size=-1):
        return self._body if size < 0 else self._body[:size]


class _Connection:
    def __init__(self, host, port, timeout):
        self.host, self.port, self.timeout = host, port, timeout
        self.headers = None
        self.response = _Response(200, b'{"ok":true}', [("Cache-Control", "no-store")])

    def request(self, method, path, body=None, headers=None):
        self.headers = headers

    def getresponse(self):
        return self.response

    def close(self):
        pass


class OwnedMappingTests(unittest.TestCase):
    def response(self, ids=None, *, primary=None, fallback=None, truncated=False,
                 config_digest="a" * 64, session="matched-1"):
        ids = ["build.software"] if ids is None else ids
        if primary is None:
            primary = ids[0] if ids else "respond.general"
        if fallback is None:
            fallback = not ids
        return {
            "schemaVersion": "lfg.assessment.v1", "sessionId": session, "revision": 1,
            "configRevision": "test-1", "configDigest": config_digest,
            "prompt": {"scoringTruncated": truncated},
            "intent": {"all": [{"id": value} for value in ids], "primary": primary, "fallback": fallback},
        }

    def test_exact_owned_positive_is_unknown_without_checkpoint_taxonomy(self):
        self.assertEqual(compare._owned_prediction("ok", {"intent": "build.software"}, False, "build.software"),
                         (None, "unknown_unmapped_taxonomy"))

    def test_only_exact_supported_positive_counts_and_other_intents_stay_unknown(self):
        self.assertEqual(compare._owned_prediction("ok", {"intent": "build.software"}, True, "build.software"),
                         (1, "exact_positive"))
        self.assertEqual(compare._owned_prediction("ok", {"intent": "respond.general"}, True, "build.software"),
                         (None, "unknown_non_target_intent"))
        self.assertEqual(compare._owned_prediction("ok", {"intent": "software development"}, True, "build.software"),
                         (None, "unknown_non_target_intent"))

    def test_non_ok_malformed_missing_and_truncated_owned_outputs_abstain(self):
        for status in ("truncated", "timeout", "malformed_output", "incomplete_output", "inference_error", "unavailable"):
            with self.subTest(status=status):
                self.assertEqual(compare._owned_prediction(status, {"intent": "build.software"}, True, "build.software"),
                                 (None, status))
        self.assertEqual(compare._owned_prediction("ok", None, True, "build.software"), (None, "malformed_contract"))
        self.assertEqual(compare._owned_prediction("ok", {"intent": 4}, True, "build.software"), (None, "malformed_contract"))

    def test_owned_response_requires_pinned_worker_lifecycle_measurements(self):
        complete = {
            "schemaVersion": "lfg.owned-run-measurement.v1", "status": "ok",
            "workerStarted": True, "workerCompletedSuccessfully": True,
        }
        self.assertEqual(compare._owned_lifecycle_complete("ok", complete), (True, "completed"))
        self.assertEqual(compare._owned_lifecycle_complete("ok", None),
                         (False, "unknown_lifecycle_measurement"))
        self.assertEqual(compare._owned_lifecycle_complete("ok", {**complete, "status": "timeout"}),
                         (False, "unknown_lifecycle_measurement"))
        self.assertEqual(compare._owned_lifecycle_complete("timeout", {**complete, "status": "timeout"}),
                         (False, "owned_timeout"))
        self.assertEqual(compare._owned_lifecycle_complete(
            "inference_error", {**complete, "status": "inference_error", "workerCompletedSuccessfully": False}),
            (False, "owned_worker_incomplete"))

    def test_deterministic_exhaustive_rule_hit_and_empty_fallback(self):
        allowed = ["build.software", "research.web"]
        positive, status, identity = compare._deterministic_prediction(
            self.response(), allowed, "respond.general", "build.software", "a" * 64, "b" * 64, "test-1", "matched-1")
        self.assertEqual((positive, status), (1, "rule_hit_positive"))
        self.assertEqual(identity, {"config_revision": "test-1", "assessment_sha256": "a" * 64,
                                    "rules_sha256": "b" * 64})
        negative, status, _ = compare._deterministic_prediction(
            self.response(ids=[], primary="respond.general", fallback=True), allowed,
            "respond.general", "build.software", "a" * 64, "b" * 64, "test-1", "matched-1")
        self.assertEqual((negative, status), (0, "rule_hit_negative"))

    def test_deterministic_invalid_partial_unknown_or_unstable_results_abstain(self):
        base = self.response()
        cases = [
            (None, "unknown_invalid_response"),
            ({**base, "prompt": {"scoringTruncated": True}}, "unknown_invalid_or_truncated"),
            ({**base, "configDigest": "c" * 64}, "unknown_config_identity_drift"),
            ({**base, "sessionId": "other"}, "unknown_invalid_or_truncated"),
            ({**base, "intent": {"all": [{"id": "unconfigured"}], "primary": "unconfigured", "fallback": False}},
             "unknown_invalid_rule_list"),
            ({**base, "intent": {"all": [], "primary": "respond.general", "fallback": False}},
             "unknown_invalid_fallback"),
            ({**base, "intent": {"all": [{"id": "build.software"}], "primary": "research.web", "fallback": False}},
             "unknown_invalid_primary"),
        ]
        for value, expected in cases:
            with self.subTest(expected=expected):
                prediction, status, _ = compare._deterministic_prediction(
                    value, ["build.software", "research.web"], "respond.general", "build.software",
                    "a" * 64, "b" * 64, "test-1", "matched-1")
                self.assertIsNone(prediction)
                self.assertEqual(status, expected)

    def test_deterministic_revision_identity_drift_is_not_a_plain_abstention(self):
        base = self.response()
        for field, changed in (("configDigest", "c" * 64), ("configRevision", "new-revision")):
            value = {**base, field: changed}
            prediction, status, identity = compare._deterministic_prediction(
                value, ["build.software", "research.web"], "respond.general", "build.software",
                "a" * 64, "b" * 64, "test-1", "matched-1")
            self.assertIsNone(prediction)
            self.assertEqual(status, "unknown_config_identity_drift")
            self.assertIsNone(identity)

    def test_unknown_target_masks_do_not_become_negatives_and_zero_coverage_is_null(self):
        result = compare._metrics([1, 0, 0], [True, False, True], [1, 0, None])
        self.assertEqual(result["known_target_cells"], 2)
        self.assertEqual(result["covered_known_target_cells"], 1)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (1, 0, 0, 0))
        empty = compare._metrics([0, 0], [False, False], [None, None])
        self.assertEqual(empty["coverage"], None)
        self.assertEqual(empty["precision"], None)
        self.assertEqual(empty["recall"], None)
        self.assertEqual(empty["f1"], None)
        fp_only = compare._metrics([0], [True], [1])
        self.assertEqual((fp_only["tp"], fp_only["fp"], fp_only["fn"], fp_only["f1"]), (0, 1, 0, 0.0))
        fp_and_fn = compare._metrics([1, 0], [True, True], [0, 1])
        self.assertEqual((fp_and_fn["tp"], fp_and_fn["fp"], fp_and_fn["fn"], fp_and_fn["f1"]), (0, 1, 1, 0.0))


class OwnedResponseDiagnosticsTests(unittest.TestCase):
    def test_production_and_empty_payloads_bind_effective_diagnostic_boundaries(self):
        class Matrix:
            def __init__(self, value):
                self.value = value
            def tolist(self):
                return self.value

        config = {
            "classical_report_sha256": "a" * 64, "classical_config_sha256": "1" * 64,
            "comparison_12b_report_sha256": "b" * 64, "comparison_12b_config_sha256": "2" * 64,
            "private_source_root": "/private/root", "service_binary": "/private/root/server",
            "service_binary_sha256": "c" * 64, "service_config_template": "/private/root/service.json",
            "service_config_template_sha256": "d" * 64, "metadata_directory": "/private/root/metadata",
            "expected_model_sha256": "e" * 64, "expected_framing_sha256": "f" * 64,
            "expected_cached_manifest_sha256": "1" * 64, "expected_runtime_config_sha256": "2" * 64,
            "expected_training_manifest_sha256": "3" * 64, "coexisting_service_unit": "lfg.service",
            "resource_cpu_quota_percent": 100, "resource_memory_max_bytes": 1000,
            "min_memory_available_bytes": 1, "min_disk_free_bytes": 1,
            "per_request_deadline_seconds": 10, "total_runtime_deadline_seconds": 100,
            "max_response_bytes": 10000, "max_payload_bytes": 100000,
            "output_text_utf8_byte_bucket_boundaries": [32, 128],
        }
        bound = {
            "validation_rows": [{"id": "fixture-row", "text": "fixture input"}],
            "y_validation": Matrix([[1]]), "target_known": Matrix([[True]]),
            "target_label": "build.software", "classical_config_sha256": "4" * 64,
            "comparison_12b_config_sha256": "5" * 64, "validation_id_set_sha256": "6" * 64,
            "target_mask_sha256": "7" * 64, "classical_report": {"data": {"validation_sha256": "8" * 64}},
            "deterministic_assessment_sha256": "9" * 64, "deterministic_rule_ids": ["build.software"],
            "deterministic_rules_sha256": "a" * 64, "deterministic_fallback_id": "respond.general",
            "classical_config": {"selected_labels": ["build.software"], "max_text_bytes": 128},
        }
        production = json.loads(compare._make_payload(config, bound, "a" * 32))
        empty = json.loads(compare._empty_probe_payload(
            config, {"selected_labels": ["build.software"], "target_field": "intents"}, "b" * 32))
        expected = [32, 128]
        self.assertEqual(production["diagnostics"]["output_text_utf8_byte_bucket_boundaries"], expected)
        self.assertEqual(empty["diagnostics"]["output_text_utf8_byte_bucket_boundaries"], expected)
        self.assertEqual(compare._payload_output_text_byte_bucket_boundaries(production), expected)
        self.assertEqual(compare._payload_output_text_byte_bucket_boundaries(empty), expected)
        self.assertEqual(len(production["records"]), 1)
        self.assertEqual(empty["records"], [])

    def test_generated_worker_counts_only_allowlisted_diagnostics_and_never_emits_sentinels(self):
        namespace = {"__name__": "generated_worker_test"}
        exec(compare._remote_worker_source(), namespace)
        boundaries = [2, 4, 8]
        counts = namespace["_new_owned_response_diagnostics"](boundaries)
        responses = [
            ("stop_sequence", '{"intent":"private-sentinel-field"}', {"outputTextUtf8Bytes": 3}),
            ("byte_limit", "[]", {"outputTextUtf8Bytes": 4}),
            ("context_limit", "invalid-private-sentinel", {"outputTextUtf8Bytes": 8}),
            ("unrecognized-private-stop", None, {}),
            (None, 4, {"outputTextUtf8Bytes": -1}),
            (17, "é", {"outputTextUtf8Bytes": 2}),
        ]
        for stop_reason, output_text, measurement in responses:
            namespace["_record_owned_response_diagnostics"](
                stop_reason, output_text, measurement, boundaries, counts)
        self.assertEqual(counts["accepted_responses"], 6)
        self.assertEqual(counts["stop_reason_counts"], {
            "stop_sequence": 1, "byte_limit": 1, "context_limit": 1, "other": 1, "unavailable": 2})
        self.assertEqual(counts["output_text_shape_counts"], {
            "valid_json_object": 1, "valid_json_non_object": 1, "invalid_json": 2,
            "unavailable_or_invalid_type": 2})
        self.assertEqual(counts["output_text_utf8_byte_bucket_counts"], [0, 2, 1, 1])
        self.assertEqual(counts["output_text_utf8_bytes_unavailable_or_invalid"], 2)
        serialized = json.dumps(counts, sort_keys=True)
        for sentinel in ("private-sentinel-field", "invalid-private-sentinel", "unrecognized-private-stop", "intent"):
            self.assertNotIn(sentinel, serialized)
        compare._validate_owned_response_diagnostics(counts, 6, boundaries, 6, 6)

    def test_bucket_boundaries_are_bounded_strictly_increasing_integers(self):
        self.assertEqual(compare._validated_output_text_byte_bucket_boundaries(None), [256, 1024, 4096, 16384])
        self.assertEqual(compare._validated_output_text_byte_bucket_boundaries([4, 8]), [4, 8])
        for invalid in ([], [0], [4, 4], [8, 4], [True], [-1], [1 << 63], [1] * 17, "4,8"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                compare._validated_output_text_byte_bucket_boundaries(invalid)

    def test_untrusted_or_malformed_worker_diagnostics_are_unknown_or_rejected(self):
        failed = {"rows_total": 2, "rows_attempted": None, "aborted": True,
                  "supervisor_failure": True, "abort_reason": "supervisor_timeout",
                  "owned_response_diagnostics": {"private-sentinel": 99}}
        self.assertIs(compare._validate_worker_aggregate(failed, 2), failed)
        self.assertIsNone(failed["owned_response_diagnostics"])

        valid = compare._new_owned_response_diagnostics([4])
        valid["accepted_responses"] = 1
        valid["stop_reason_counts"]["other"] = 1
        valid["output_text_shape_counts"]["invalid_json"] = 1
        valid["output_text_utf8_bytes_unavailable_or_invalid"] = 1
        aggregate = {"rows_total": 2, "rows_attempted": 1, "owned_requests_started": 1,
                     "owned_response_status_counts": {"ok": 1},
                     "owned_response_diagnostics": valid, "aborted": True}
        self.assertIs(compare._validate_worker_aggregate(aggregate, 2, [4]), aggregate)
        for mutate in (
            lambda d: d["stop_reason_counts"].update({"private-sentinel": 1}),
            lambda d: d["output_text_shape_counts"].update({"invalid_json": True}),
            lambda d: d.update({"accepted_responses": -1}),
            lambda d: d["output_text_utf8_byte_bucket_counts"].__setitem__(0, 2),
        ):
            bad = json.loads(json.dumps(aggregate))
            mutate(bad["owned_response_diagnostics"])
            with self.assertRaises(ValueError):
                compare._validate_worker_aggregate(bad, 2, [4])

    def test_lifecycle_incomplete_accepted_response_is_still_diagnosed_once(self):
        namespace = {"__name__": "generated_worker_test"}
        exec(compare._remote_worker_source(), namespace)
        boundaries = [4]
        counts = namespace["_new_owned_response_diagnostics"](boundaries)
        measurement = {"schemaVersion": "lfg.owned-run-measurement.v1", "status": "ok",
                       "workerStarted": True, "workerCompletedSuccessfully": False,
                       "outputTextUtf8Bytes": 7}
        namespace["_record_owned_response_diagnostics"](
            "byte_limit", '{"field":"secret-value"}', measurement, boundaries, counts)
        prediction = namespace["_owned_terminal_prediction"](
            "ok", {"intent": "secret-value"}, measurement, False, "build.software")
        self.assertEqual(prediction, (None, "owned_worker_incomplete", False))
        self.assertEqual(counts["accepted_responses"], 1)
        self.assertEqual(counts["stop_reason_counts"]["byte_limit"], 1)
        self.assertEqual(counts["output_text_shape_counts"]["valid_json_object"], 1)
        self.assertEqual(counts["output_text_utf8_byte_bucket_counts"], [0, 1])
        self.assertNotIn("secret-value", json.dumps(counts))

    def test_report_exposes_aggregates_and_public_boundaries_only(self):
        boundaries = [4, 8]
        diagnostics = compare._new_owned_response_diagnostics(boundaries)
        diagnostics["accepted_responses"] = 1
        diagnostics["stop_reason_counts"]["stop_sequence"] = 1
        diagnostics["output_text_shape_counts"]["valid_json_object"] = 1
        diagnostics["output_text_utf8_byte_bucket_counts"][1] = 1
        result = {"rows_total": 1, "rows_attempted": 1, "owned_requests_started": 1,
                  "owned_response_status_counts": {"incomplete_output": 1},
                  "owned_response_diagnostics": diagnostics, "aborted": False,
                  "owned_metrics": {"coverage": None}, "deterministic_metrics": {},
                  "owned_mapping": {"supported": False}, "owned_mapping_outcome_counts": {},
                  "owned_lifecycle_outcome_counts": {"owned_worker_incomplete": 1},
                  "deterministic_rule_status_counts": {}, "owned_engine_identity": {},
                  "owned_status_limits": {}}
        bound = {
            "validation_rows": [{}], "classical_report": {
                "task": {"validation_support": {}}, "algorithms": {},
                "data": {"plan_sha256": "a" * 64, "review_ledger_sha256": "b" * 64}},
            "comparison_12b_report": {"metrics": {}},
            "comparison_12b_config": {"remote_host": "baseline-host", "remote_user": "baseline-user"},
            "classical_config": {"target_field": "intents", "selected_labels": ["build.software"]},
            "corpus": {"taxonomy_sha256": "c" * 64, "manifest_sha256": "d" * 64,
                       "policy_sha256": "e" * 64, "screen_code_sha256": "f" * 64,
                       "train_sha256": "1" * 64, "validation_sha256": "2" * 64},
            "validation_id_set_sha256": "3" * 64, "target_mask_sha256": "4" * 64,
            "classical_config_sha256": "5" * 64, "comparison_12b_config_sha256": "6" * 64,
            "review_provenance": {}, "deterministic_files_sha256": {},
            "deterministic_assessment_sha256": "7" * 64, "deterministic_rules_sha256": "8" * 64,
        }
        config = {"output_text_utf8_byte_bucket_boundaries": boundaries,
                  "classical_report_sha256": "9" * 64, "classical_config_sha256": "a" * 64,
                  "comparison_12b_report_sha256": "b" * 64, "comparison_12b_config_sha256": "c" * 64,
                  "remote_host": "fixture", "remote_user": "worker",
                  "service_config_template_sha256": "d" * 64,
                  "per_request_deadline_seconds": 10}
        report = compare._report(config, "e" * 64, bound, result, b"{}", "f" * 64, Path("unused"))
        self.assertEqual(report["owned"]["response_diagnostics"]["accepted_responses"], 1)
        self.assertEqual(report["owned"]["response_diagnostics"]["output_text_utf8_byte_bucket_boundaries"], boundaries)
        self.assertEqual(report["owned"]["status_counts"], {"incomplete_output": 1})
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("secret-value", serialized)
        self.assertNotIn("outputText", serialized)

    def test_generated_worker_main_counts_multiple_rows_and_terminal_failure_safely(self):
        import io
        from types import SimpleNamespace

        def run_worker(owned_responses):
            namespace = {"__name__": "generated_worker_test"}
            exec(compare._remote_worker_source(), namespace)
            boundaries = [8, 32]
            expected_hashes = {"model": "a" * 64, "framing": "b" * 64,
                               "cached": "c" * 64, "runtime": "d" * 64, "training": "e" * 64}
            rows = []
            for index, text in enumerate(("fixture row one", "fixture row two")):
                row_id = str(index + 1)
                rows.append({"record_id_sha256": compare.sha(row_id.encode()), "text": text,
                             "text_sha256": compare.sha(text.encode()), "target_known": True,
                             "target_value": index % 2})
            ids_digest = compare.sha(json.dumps(sorted(row["record_id_sha256"] for row in rows),
                                                  separators=(",", ":")).encode())
            masks = [[row["record_id_sha256"], [row["target_known"]]] for row in rows]
            mask_digest = compare.sha(json.dumps(masks, separators=(",", ":")).encode())
            payload = {"schema_version": "lfg.owned-matched-worker-input.v1", "run_id": "f" * 32,
                       "binding": {"rows": 2, "validation_record_id_set_sha256": ids_digest,
                                   "target_mask_sha256": mask_digest, "target_label": "build.software",
                                   "rule_ids": ["build.software"], "fallback_id": "respond.general",
                                   "assessment_sha256": "1" * 64, "rules_sha256": "2" * 64},
                       "remote": {}, "diagnostics": {"output_text_utf8_byte_bucket_boundaries": boundaries},
                       "records": rows, "outer_netns_inode": 11}

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "private"
                root.mkdir(mode=0o700)
                metadata = root / "metadata"
                metadata.mkdir(mode=0o700)
                binary = root / "server"
                binary.write_bytes(b"fixture binary")
                binary.chmod(0o700)
                framing = metadata / "framing.json"
                runtime = metadata / "runtime.json"
                training = metadata / "training.json"
                for path, content in ((framing, b'{"system":"fixture"}'),
                                      (runtime, b'{"fixture":true}'),
                                      (training, b'{"fixture":true}')):
                    path.write_bytes(content)
                    path.chmod(0o600)
                service = {"schemaVersion": "lfg.owned-assessment-service.v1", "enabled": True,
                           "backend": "cached_onnx", "artifactMode": "embedded",
                           "expectedModelSha256": expected_hashes["model"],
                           "expectedFramingSha256": expected_hashes["framing"],
                           "expectedCachedManifestSha256": expected_hashes["cached"],
                           "maxConcurrent": 1, "workerThreads": 1, "timeoutMs": 500,
                           "maxRequestBytes": 1000, "maxContextBytes": 2000, "maxGeneratedBytes": 3000,
                           "maxCachedGraphBytes": 4000, "maxCacheStateBytesPerRequest": 5000,
                           "requiredOutputFields": [{"name": "intent", "valueType": "string", "nonEmpty": True}]}
                template = root / "service.json"
                template.write_text(json.dumps(service), encoding="utf-8")
                template.chmod(0o600)
                run_directory = root / "run"
                run_directory.mkdir(mode=0o700)
                expected_hashes.update({"model": compare.sha(binary.read_bytes()),
                                        "framing": compare.sha(framing.read_bytes()),
                                        "runtime": compare.sha(runtime.read_bytes()),
                                        "training": compare.sha(training.read_bytes())})
                service.update({"expectedModelSha256": expected_hashes["model"],
                                "expectedFramingSha256": expected_hashes["framing"],
                                "expectedCachedManifestSha256": expected_hashes["cached"]})
                template.write_text(json.dumps(service), encoding="utf-8")
                template.chmod(0o600)
                remote = {"private_source_root": str(root), "service_binary": str(binary),
                          "service_binary_sha256": expected_hashes["model"],
                          "service_config_template": str(template),
                          "service_config_template_sha256": compare.sha(template.read_bytes()),
                          "metadata_directory": str(metadata), "expected_model_sha256": expected_hashes["model"],
                          "expected_framing_sha256": expected_hashes["framing"],
                          "expected_cached_manifest_sha256": expected_hashes["cached"],
                          "expected_runtime_config_sha256": expected_hashes["runtime"],
                          "expected_training_manifest_sha256": expected_hashes["training"],
                          "per_request_deadline_seconds": 10, "total_runtime_deadline_seconds": 100,
                          "max_response_bytes": 10000}
                payload["remote"] = remote
                payload["run_directory"] = str(run_directory)
                engine = {"id": "fixture-engine", "modelIdentityKind": "source_weights_sha256",
                          "modelSha256": expected_hashes["model"],
                          "sourceWeightsSha256": expected_hashes["model"],
                          "cachedManifestSha256": expected_hashes["cached"],
                          "framingSha256": expected_hashes["framing"],
                          "runtimeConfigSha256": expected_hashes["runtime"],
                          "trainingManifestSha256": expected_hashes["training"]}
                limits = {"backend": "cached_onnx", "maxRequestBytes": service["maxRequestBytes"],
                          "maxContextBytes": service["maxContextBytes"], "maxGeneratedBytes": service["maxGeneratedBytes"],
                          "maxCachedGraphBytes": service["maxCachedGraphBytes"],
                          "maxCacheStateBytesPerRequest": service["maxCacheStateBytesPerRequest"],
                          "timeoutMs": service["timeoutMs"], "maxConcurrent": 1, "workerThreads": 1}
                caps = {"assessmentSha256": payload["binding"]["assessment_sha256"],
                        "rulesSha256": payload["binding"]["rules_sha256"],
                        "assessmentRevision": "fixture", "rulesRevision": "fixture"}

                def response(status="ok", stop_reason="stop_sequence", output_text='{"intent":"sentinel-output"}',
                             *, completed=True, byte_count=24):
                    return {"schemaVersion": "lfg.owned-assessment.v1", "status": status,
                            "engine": engine, "stopReason": stop_reason, "outputText": output_text,
                            "output": {"intent": "sentinel-output"},
                            "measurement": {"schemaVersion": "lfg.owned-run-measurement.v1",
                                            "status": status, "workerStarted": True,
                                            "workerCompletedSuccessfully": completed,
                                            "outputTextUtf8Bytes": byte_count}}

                response_queue = list(owned_responses(response))
                def http(_port, path, _method="GET", _body=None, token=None, **_kwargs):
                    headers = {"cache-control": "no-store"}
                    if path.endswith("/api/lfg/assess/owned/status"):
                        if token is None:
                            return 401, headers, b"{}"
                        return 200, headers, json.dumps({"status": "ready", "enabled": True,
                                                        "engine": engine, "limits": limits}).encode()
                    if path == "/api/lfg/observations/capabilities":
                        return 200, headers, json.dumps(caps).encode()
                    if path == "/api/lfg/assess":
                        return 200, headers, b"{}"
                    if path == "/api/lfg/assess/owned":
                        next_response = response_queue.pop(0)
                        if isinstance(next_response, BaseException):
                            raise next_response
                        return 200, headers, json.dumps(next_response).encode()
                    raise AssertionError(path)

                class Child:
                    def poll(self):
                        return None

                real_stat = namespace["os"].stat
                def stat(path, *args, **kwargs):
                    if path == "/proc/self/ns/net":
                        return SimpleNamespace(st_ino=22)
                    return real_stat(path, *args, **kwargs)
                real_path_stat = Path.stat
                def secure_path_stat(path, *args, **kwargs):
                    result = real_path_stat(path, *args, **kwargs)
                    values = list(result)
                    values[0] = result.st_mode & ~0o077
                    values[4] = 1000
                    return os.stat_result(values)

                with patch.object(namespace["sys"], "stdin", io.StringIO(json.dumps(payload))), \
                        patch.object(namespace["os"], "stat", side_effect=stat), \
                        patch.object(namespace["os"], "geteuid", return_value=1000, create=True), \
                        patch.object(namespace["pathlib"].Path, "stat", new=secure_path_stat), \
                        patch.object(namespace["subprocess"], "Popen", return_value=Child()), \
                        patch.dict(namespace, {"_http_loopback": http,
                                               "_deterministic_prediction": lambda *_args: (None, "unknown", None),
                                               "route_isolated": lambda: None,
                                               "_terminate_process_group": lambda _child: {"verified": True, "exit_code": 0},
                                               "_loopback_port_closed": lambda _port: True,
                                               "resource_sample": lambda: {"cgroup": None}}):
                    return namespace["main"]()

        def response_factory(response):
            first = response()
            incomplete = response(stop_reason="context_limit", completed=False, byte_count=40)
            return [first, incomplete]

        lifecycle = run_worker(response_factory)
        self.assertEqual(lifecycle["owned_response_diagnostics"]["accepted_responses"], 2)
        self.assertEqual(lifecycle["owned_response_status_counts"], {"ok": 2})
        self.assertEqual(lifecycle["owned_lifecycle_outcome_counts"], {"completed": 1, "owned_worker_incomplete": 1})
        self.assertEqual(lifecycle["owned_response_diagnostics"]["stop_reason_counts"]["context_limit"], 1)
        self.assertEqual(sum(lifecycle["owned_response_diagnostics"]["stop_reason_counts"].values()), 2)
        self.assertEqual(sum(lifecycle["owned_response_diagnostics"]["output_text_shape_counts"].values()), 2)
        self.assertEqual(sum(lifecycle["owned_response_diagnostics"]["output_text_utf8_byte_bucket_counts"])
                         + lifecycle["owned_response_diagnostics"]["output_text_utf8_bytes_unavailable_or_invalid"], 2)
        self.assertEqual(lifecycle["owned_metrics"]["coverage"], 0.0)
        self.assertTrue(lifecycle["aborted"])
        self.assertNotIn("sentinel-output", json.dumps(lifecycle))

        def no_terminal_factory(response):
            return [response(), TimeoutError("private-sentinel-exception")]

        terminal = run_worker(no_terminal_factory)
        self.assertEqual(terminal["owned_response_diagnostics"]["accepted_responses"], 1)
        self.assertEqual(terminal["owned_response_status_counts"], {"ok": 1, "no_terminal_response": 1})
        self.assertEqual(terminal["owned_response_diagnostics"]["stop_reason_counts"]["stop_sequence"], 1)
        self.assertEqual(sum(terminal["owned_response_diagnostics"]["output_text_shape_counts"].values()), 1)
        self.assertEqual(sum(terminal["owned_response_diagnostics"]["output_text_utf8_byte_bucket_counts"])
                         + terminal["owned_response_diagnostics"]["output_text_utf8_bytes_unavailable_or_invalid"], 1)
        self.assertTrue(terminal["aborted"])
        self.assertNotIn("private-sentinel-exception", json.dumps(terminal))
        self.assertNotIn("sentinel-output", json.dumps(terminal))


class BoundaryTests(unittest.TestCase):
    def test_unloaded_unit_cleanup_requires_prior_observation_and_empty_namespace(self):
        missing = {"LoadState": "not-found"}
        cg = "/user.slice/lfg-owned-matched-0123456789abcdef01.service"
        self.assertTrue(compare._verified_stop(missing, 5, True, cg, 123, True, True))
        self.assertFalse(compare._verified_stop(missing, 5, True, "", 123, True, True))
        self.assertFalse(compare._verified_stop(missing, 5, False, cg, 123, True, True))
        self.assertFalse(compare._verified_stop(missing, 5, True, cg, 123, None, True))
        self.assertFalse(compare._verified_stop(missing, 5, True, cg, 123, True, None))
        loaded = {"LoadState": "loaded", "ActiveState": "inactive", "MainPID": "0"}
        self.assertFalse(compare._verified_stop(loaded, 5, True, cg, 123, True, True))
        self.assertTrue(compare._verified_stop(loaded, 0, True, cg, 123, True, True))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux file locking required")
    def test_launch_stop_lock_and_cancel_prevent_late_start(self):
        nonce = "0123456789abcdef0123456789abcdef"
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / ("lfg-owned-matched-" + nonce[:18]); work.mkdir(mode=0o700)
            static = {"nonce": nonce, "run_directory": str(work), "cleanup_deadline_seconds": 3}
            raw = json.dumps(static).encode(); static_path = work / "static.json"
            static_path.write_bytes(raw); static_path.chmod(0o600)
            source = compare._file_control_source().rsplit("if __name__=='__main__':", 1)[0]
            harness = '''
def dispatch(action,work,nonce,static):
 with (work/'events').open('a') as f: f.write(action+'-begin\\n')
 if action=='start':
  deadline=time.monotonic()+2
  while not (work/'launch-cancel.json').exists() and time.monotonic()<deadline: time.sleep(.01)
  with (work/'events').open('a') as f: f.write('start-end\\n')
main()
'''
            script = work / "test-control.py"; script.write_text(source + harness)
            def command(action):
                return [sys.executable, str(script), action, str(work), nonce, compare.sha(raw)]
            start = subprocess.Popen(command("start"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 3
            while not (work / "events").exists() and time.monotonic() < deadline: time.sleep(.01)
            stop = subprocess.Popen(command("stop"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.assertEqual(start.wait(timeout=5), 0)
            self.assertEqual(stop.wait(timeout=5), 0)
            self.assertEqual((work / "events").read_text().splitlines(), ["start-begin", "start-end", "stop-begin"])
            self.assertTrue((work / "launch-cancel.json").exists())
            late = subprocess.run(command("start"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            self.assertNotEqual(late.returncode, 0)
            self.assertEqual((work / "events").read_text().splitlines(), ["start-begin", "start-end", "stop-begin"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux no-replace rename required")
    def test_concurrent_published_files_are_single_link_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            done = threading.Event()
            failures = []
            def consume():
                seen = set()
                while not done.is_set() or len(seen) < 100:
                    for path in root.glob("ready-*"):
                        if path in seen: continue
                        try:
                            self.assertEqual(compare._read_private_file(path, 64), b"complete")
                        except Exception as error:
                            failures.append(error); return
                        seen.add(path)
                    if done.is_set() and len(seen) < 100: time.sleep(.001)
            reader = threading.Thread(target=consume)
            reader.start()
            for index in range(100):
                compare._publish_private_file(root / ("ready-" + str(index)), b"complete", 64)
            done.set(); reader.join(5)
            self.assertFalse(reader.is_alive())
            self.assertFalse(failures)
            source = root / "partial"; source.write_bytes(b"replacement"); source.chmod(0o600)
            with self.assertRaises(FileExistsError):
                compare._rename_noreplace(source, root / "ready-0")
            self.assertEqual(source.read_bytes(), b"replacement")
            self.assertEqual((root / "ready-0").read_bytes(), b"complete")

    def test_probe_envelope_has_no_records_or_private_source_text(self):
        config = {name: "fixture" for name in (
            "classical_report_sha256", "classical_config_sha256", "comparison_12b_report_sha256",
            "comparison_12b_config_sha256", "private_source_root", "service_binary", "service_binary_sha256",
            "service_config_template", "metadata_directory", "expected_model_sha256", "expected_framing_sha256",
            "service_config_template_sha256",
            "expected_cached_manifest_sha256", "expected_runtime_config_sha256", "expected_training_manifest_sha256",
            "coexisting_service_unit", "resource_cpu_quota_percent", "resource_memory_max_bytes",
            "min_memory_available_bytes", "min_disk_free_bytes", "per_request_deadline_seconds",
            "total_runtime_deadline_seconds", "max_response_bytes")}
        raw = compare._empty_probe_payload(config, {"selected_labels": ["build.software"],
                                                    "target_field": "intents"}, "a" * 32)
        probe = json.loads(raw)
        self.assertEqual(probe["records"], [])
        self.assertEqual(probe["binding"]["rows"], 0)
        self.assertEqual(probe["diagnostics"]["output_text_utf8_byte_bucket_boundaries"],
                         list(compare.DEFAULT_OUTPUT_TEXT_BYTE_BUCKET_BOUNDARIES))
        self.assertNotIn('"text"', raw.decode())

    def test_current_empty_probe_and_exact_code_digests_are_required(self):
        config_digest = "a" * 64
        worker_sha = compare.sha(compare._file_worker_source().encode())
        control_sha = compare.sha(compare._file_control_source().encode())
        result = {"rows_total": 0, "rows_attempted": 0, "deterministic_requests_started": 0,
                  "owned_requests_started": 0, "aborted": False, "cleanup_verified": True,
                  "coexisting_service_snapshots_match": True, "remote_workspace_retained": False,
                  "owned_response_status_counts": {},
                  "owned_response_diagnostics": compare._new_owned_response_diagnostics(
                      list(compare.DEFAULT_OUTPUT_TEXT_BYTE_BUCKET_BOUNDARIES))}
        probe = {"schema_version": "lfg.owned-file-probe.v1", "config_sha256": config_digest,
                 "adapter_sha256": compare.sha(Path(compare.__file__).read_bytes()),
                 "worker_code_sha256": worker_sha, "control_code_sha256": control_sha,
                 "passed": True, "private_rows": 0, "result": result}
        raw = json.dumps(probe).encode()
        compare._validate_probe(raw, config_digest)
        probe["control_code_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "no-input probe"):
            compare._validate_probe(json.dumps(probe).encode(), config_digest)
        probe["control_code_sha256"] = control_sha
        probe["result"]["owned_requests_started"] = 1
        with self.assertRaisesRegex(ValueError, "probe result"):
            compare._validate_probe(json.dumps(probe).encode(), config_digest)

    def test_file_coordinator_gates_private_transfer_and_handles_partial_timeout(self):
        nonce = "0123456789abcdef0123456789abcdef"
        unit = "lfg-owned-matched-" + nonce[:18] + ".service"
        cg = "/user.slice/user-1000.slice/" + unit
        boundaries = list(compare.DEFAULT_OUTPUT_TEXT_BYTE_BUCKET_BOUNDARIES)
        payload = json.dumps({"run_id": nonce, "binding": {"rows": 0}, "remote": {},
                              "diagnostics": {"output_text_utf8_byte_bucket_boundaries": boundaries},
                              "records": []},
                             sort_keys=True, separators=(",", ":")).encode()
        config = {"private_source_root": "/home/hades/private", "remote_python_path": "/usr/bin/python3",
                  "systemd_run_path": "/usr/bin/systemd-run", "remote_user": "hades", "remote_host": "example",
                  "ssh_port": 22, "file_preflight_deadline_seconds": 2, "file_transfer_deadline_seconds": 2,
                  "file_cleanup_deadline_seconds": 2, "max_payload_bytes": 100000, "max_response_bytes": 100000,
                  "total_runtime_deadline_seconds": 5, "resource_cpu_quota_percent": 100,
                  "per_request_deadline_seconds": 1,
                  "file_max_binary_bytes": 100000, "file_max_metadata_bytes": 100000,
                  "file_min_gpu_free_bytes": 0,
                  "expected_remote_architecture": "aarch64",
                  "resource_memory_max_bytes": 1000, "coexisting_service_unit": "lfg.service",
                  "coexisting_listener_port": 33333, "min_memory_available_bytes": 1, "min_disk_free_bytes": 1}
        worker_source, control_source = compare._file_worker_source(), compare._file_control_source()
        for scenario in ("success", "preflight_exit", "transfer_timeout"):
            events = []
            def remote(argv, timeout, limit):
                cmd = shlex.split(argv[-1])
                if "-c" in cmd:
                    if "work.mkdir" in cmd[cmd.index("-c") + 1]:
                        return json.dumps({"uid": 1000, "gid": 1000, "groups": [1000], "outer_netns_inode": 1}).encode()
                    return b""
                action = cmd[3]; events.append(action)
                if action == "start": reply = {"started": True}
                elif action == "check":
                    if scenario == "preflight_exit": raise RuntimeError("not ready")
                    reply = {"ready": True, "cgroup": cg, "netns_inode": 2}
                elif action == "state": reply = {"active": False}
                elif action == "promote": reply = {"promoted": True}
                elif action == "result": reply = {"ready": True, "result": {
                    "rows_total": 0, "rows_attempted": 0, "aborted": False,
                    "owned_requests_started": 0,
                    "owned_response_status_counts": {},
                    "owned_response_diagnostics": compare._new_owned_response_diagnostics(boundaries),
                    "worker_payload_sha256": compare.sha(payload), "cleanup_verified": True}}
                elif action == "stop": reply = {"verified": scenario != "preflight_exit",
                                                 "coexisting_service_snapshots_match": True}
                elif action == "clean": reply = {"removed": True}
                else: raise AssertionError(action)
                return json.dumps(reply).encode()
            def transfer(argv, **kwargs):
                if argv[-1].endswith("/payload.partial"):
                    events.append("payload_transfer")
                    if scenario == "transfer_timeout": raise subprocess.TimeoutExpired(argv, 2)
                return subprocess.CompletedProcess(argv, 0)
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(compare, "_file_worker_source", return_value=worker_source), \
                    patch.object(compare, "_file_control_source", return_value=control_source), \
                    patch.object(compare, "_bounded_command", side_effect=remote), \
                    patch.object(compare.subprocess, "run", side_effect=transfer):
                result = compare._file_evaluate(config, payload, nonce, Path(directory))
            if scenario == "success":
                self.assertLess(events.index("check"), events.index("payload_transfer"))
                self.assertLess(events.index("payload_transfer"), events.index("promote"))
                self.assertTrue(result["cleanup_verified"])
                self.assertFalse(result["remote_workspace_retained"])
            elif scenario == "preflight_exit":
                self.assertNotIn("payload_transfer", events)
                self.assertEqual(result["payload_transfer_state"], "not_transferred_before_preflight")
                self.assertTrue(result["remote_workspace_retained"])
            else:
                self.assertNotIn("promote", events)
                self.assertEqual(result["payload_transfer_state"], "unknown_after_file_transfer")

    def test_command_output_and_stall_are_bounded(self):
        with self.assertRaises(TimeoutError):
            compare._bounded_command([sys.executable, "-c", "print('x'*20000)"], 3, 64)
        with self.assertRaises(TimeoutError):
            compare._bounded_command([sys.executable, "-c", "import time; time.sleep(5)"], 1, 64)
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            compare._bounded_command([sys.executable, "-c",
                "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(8)']); time.sleep(8)"], 1, 64)
        self.assertLess(time.monotonic() - start, 7)

    def test_host_preflight_fails_for_listener_pid_and_namespace_drift(self):
        nonce = "0123456789abcdef0123456789abcdef"
        unit = "lfg-owned-matched-" + nonce[:18] + ".service"
        cg = "/user.slice/user-1000.slice/" + unit
        marker = {"schema_version": "lfg.file-preflight.v1", "nonce": nonce, "unit": unit,
                  "uid": 1000, "gid": 1000, "groups": [27, 1000], "cgroup": cg,
                  "netns_inode": 2, "outer_netns_inode": 1, "port": 33333,
                  "worker_pid": 10, "server_pid": 11}
        props = {"ActiveState": "active", "KillMode": "control-group", "MainPID": "9", "ControlGroup": cg}
        status = {"Uid": "1000 1000 1000 1000", "Gid": "1000 1000 1000 1000",
                  "Groups": "27 1000", "NoNewPrivs": "1"}
        status.update({name: "0" for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")})
        args = [marker, props, cg, status, status, cg, cg, 2, 2, 1, nonce, 1000, 1000, [27, 1000], False]
        compare._validate_host_preflight(*args)
        for index, bad in ((14, True), (7, 1), (8, 3), (2, cg + "/child")):
            changed = list(args); changed[index] = bad
            with self.subTest(index=index), self.assertRaises(ValueError):
                compare._validate_host_preflight(*changed)
        changed = list(args); changed[0] = dict(marker, nonce="f" * 32)
        with self.assertRaises(ValueError):
            compare._validate_host_preflight(*changed)

    def test_file_control_source_loads_without_side_effects(self):
        namespace = {"__name__": "file_control_test"}
        exec(compile(compare._file_control_source(), "generated-file-control", "exec"), namespace)
        self.assertTrue(callable(namespace["preflight"]))

    def test_file_worker_executes_definitions_and_attests_before_payload_read(self):
        source = compare._file_worker_source()
        namespace = {"__name__": "file_worker_test"}
        exec(compile(source, "generated-file-worker", "exec"), namespace)
        body = source[source.index("def main():", source.index("def receive_payload")):]
        self.assertLess(body.index("file_preflight(static,child,port)"),
                        body.index("received=receive_payload(static)"))
        self.assertLess(body.index("service template digest mismatch"), body.index("child=subprocess.Popen"))
        self.assertNotIn("json.load(sys.stdin)", body)
        self.assertTrue(callable(namespace["file_preflight"]))

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW") and hasattr(os, "geteuid"), "POSIX file controls required")
    def test_private_file_publication_refuses_overwrite_symlinks_and_large_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "ready"
            compare._publish_private_file(target, b"safe", 4)
            self.assertEqual(compare._read_private_file(target, 4), b"safe")
            with self.assertRaises(FileExistsError):
                compare._publish_private_file(target, b"new", 4)
            self.assertEqual(target.read_bytes(), b"safe")
            with self.assertRaises(ValueError):
                compare._read_private_file(target, 3)
            link = Path(directory) / "link"
            link.symlink_to(target)
            with self.assertRaises(OSError):
                compare._read_private_file(link, 4)

    def test_process_preflight_checks_all_identity_and_capability_sets(self):
        status = {"Uid": "1000 1000 1000 1000", "Gid": "1000 1000 1000 1000",
                  "Groups": "1000 27", "NoNewPrivs": "1"}
        status.update({name: "0000000000000000" for name in
                       ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")})
        compare._validate_process_isolation(status, 1000, 1000, [27, 1000])
        for field in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
            bad = dict(status, **{field: "1"})
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare._validate_process_isolation(bad, 1000, 1000, [27, 1000])
        for field, value in (("Uid", "1000 1000 0 1000"), ("Gid", "1000 1000 1000 0"),
                             ("Groups", "1000"), ("NoNewPrivs", "0")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare._validate_process_isolation(dict(status, **{field: value}), 1000, 1000, [27, 1000])

    def test_ipv6_routes_and_cgroup_paths_are_checked_exactly(self):
        links = [{"ifname": "lo", "flags": ["LOOPBACK", "UP"]}]
        compare._validate_network_isolation(links, [], [])
        with self.assertRaises(ValueError):
            compare._validate_network_isolation(links, [], [{"dst": "::/0"}])
        with self.assertRaises(ValueError):
            compare._validate_network_isolation(links + [{"ifname": "eth0"}], [], [])
        unit = "lfg-owned-matched-0123456789abcdef01.service"
        compare._validate_unit_cgroup("/user.slice/user-1000.slice/" + unit, unit)
        for path in ("user.slice/" + unit, "/user.slice//" + unit,
                     "/user.slice/../" + unit, "/user.slice/" + unit + "/child"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                compare._validate_unit_cgroup(path, unit)

    def test_12b_payload_binding_includes_text_and_response_identity(self):
        payload = {"records": [{"text": "fixture", "target_known": [True]}]}
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False) + "\n").encode()
        ids = ["b" * 64, "a" * 64]
        identity = compare.sha(json.dumps(sorted(ids), separators=(",", ":")).encode())
        report = {"inference": {"payload_sha256": compare.sha(raw),
                                "response_record_id_set_sha256": identity}}
        compare._validate_12b_payload_binding(payload, report, ids)
        with self.assertRaisesRegex(ValueError, "input bytes"):
            compare._validate_12b_payload_binding({"records": []}, report, ids)
        with self.assertRaisesRegex(ValueError, "row identity"):
            compare._validate_12b_payload_binding(payload, report, ["c" * 64])

    def test_rejected_transport_cannot_start_a_remote_process(self):
        with patch.object(compare.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "transport disabled"):
                compare._ssh_worker({}, b"private input", "0" * 32)
        popen.assert_not_called()

    def test_12b_history_claims_preserve_false_and_unknown_for_both_schemas(self):
        baseline_unknown = {"data": {}}
        unknown_claims = compare.comparison_12b._baseline_review_provenance(baseline_unknown)
        unknown = compare._validate_12b_review_provenance(
            "lfg.local-12b-comparison-config.v2", unknown_claims, baseline_unknown)
        self.assertEqual(unknown["validation_group_use_claim"], "unknown")
        self.assertIsNone(unknown["validation_independence_verified"])

        baseline_false = {"data": {"validation_group_use_claim": "reported_unused",
                                    "validation_group_use_claim_verified": False,
                                    "independence_claim": "reported only",
                                    "validation_independence_verified": False}}
        expected = compare.comparison_12b._baseline_review_provenance(baseline_false)
        self.assertEqual(compare._validate_12b_review_provenance(
            "lfg.local-12b-comparison-config.v2", expected, baseline_false), expected)
        wrong = {**expected, "validation_independence_verified": None}
        with self.assertRaisesRegex(ValueError, "review/history"):
            compare._validate_12b_review_provenance(
                "lfg.local-12b-comparison-config.v2", wrong, baseline_false)

        historical = compare._validate_12b_review_provenance(
            "lfg.local-12b-comparison-config.v1",
            {"labels_unverified": True, "validation_independence_verified": False}, baseline_unknown)
        self.assertIs(historical["validation_independence_verified"], False)
        with self.assertRaises(ValueError):
            compare._validate_12b_review_provenance(
                "lfg.local-12b-comparison-config.v1", {"labels_unverified": True}, baseline_unknown)

    @unittest.skipUnless(hasattr(signal, "setitimer"), "strict SIGALRM deadline requires POSIX")
    def test_http_total_deadline_stops_slow_header_drips(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class DripHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                for chunk in (b"HTTP/1.1 200 OK\r\n", b"Cache-Control: no-store\r\n", b"Content-Length: 2\r\n", b"\r\n", b"ok"):
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except OSError:
                        break
                    time.sleep(0.04)

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), DripHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            with self.assertRaises(TimeoutError):
                compare._http_loopback(server.server_port, "/", timeout=0.12)
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_ssh_remote_command_quotes_paths_and_rejects_injection(self):
        import shlex
        command = compare._ssh_remote_command("/opt/Python Runtime/python3", "print('safe')")
        self.assertEqual(shlex.split(command), ["/opt/Python Runtime/python3", "-I", "-c", "print('safe')"])
        with self.assertRaises(ValueError):
            compare._ssh_remote_command("/usr/bin/python3;touch /tmp/pwned", "pass")
        self.assertEqual(compare._ssh_destination("hades", "clyffy-02"), "hades@clyffy-02")
        with self.assertRaises(ValueError):
            compare._ssh_destination("hades;id", "clyffy-02")
        cleanup = compare._remote_cleanup_script(
            "lfg-owned-matched-0123456789abcdef01.scope", "/home/hades")
        compile(cleanup, "bounded-unit-cleanup", "exec")
        self.assertIn("systemctl','--user','stop',unit", cleanup)
        with self.assertRaises(ValueError):
            compare._remote_cleanup_script("lfg-owned-cached-assessment.service", "/home/hades")

    def test_dirty_or_untracked_adapter_does_not_claim_head_revision(self):
        tracked = subprocess.CompletedProcess([], 0, stdout="compare_owned_validation.py\n")
        dirty = subprocess.CompletedProcess([], 0, stdout=" M compare_owned_validation.py\n")
        with patch.object(compare.subprocess, "run", side_effect=[tracked, dirty]):
            self.assertIsNone(compare._git_revision())
        untracked = subprocess.CalledProcessError(1, ["git", "ls-files"])
        with patch.object(compare.subprocess, "run", side_effect=untracked):
            self.assertIsNone(compare._git_revision())
        clean = subprocess.CompletedProcess([], 0, stdout="")
        head = subprocess.CompletedProcess([], 0, stdout="a" * 40 + "\n")
        with patch.object(compare.subprocess, "run", side_effect=[tracked, clean, head]):
            self.assertEqual(compare._git_revision(), "a" * 40)

    def test_generated_worker_executes_multiple_terminal_rows_and_abstains_incomplete(self):
        namespace = {"__name__": "generated_worker_test"}
        exec(compare._remote_worker_source(), namespace)
        self.assertIn("sys", namespace)
        self.assertTrue(callable(namespace["_owned_terminal_prediction"]))
        measurement = {"schemaVersion": "lfg.owned-run-measurement.v1", "status": "ok",
                       "workerStarted": True, "workerCompletedSuccessfully": True}
        status_counts, lifecycle_counts = {}, {}
        completed = [namespace["_owned_terminal_prediction"](
            "ok", {"intent": "build.software"}, measurement, True, "build.software",
            status_counts, lifecycle_counts)
            for _ in range(3)]
        self.assertEqual(completed, [(1, "exact_positive", True)] * 3)
        self.assertEqual(status_counts, {"ok": 3})
        self.assertEqual(lifecycle_counts, {"completed": 3})
        incomplete = namespace["_owned_terminal_prediction"](
            "ok", {"intent": "build.software"},
            {**measurement, "workerCompletedSuccessfully": False}, True, "build.software")
        self.assertEqual(incomplete, (None, "owned_worker_incomplete", False))

    def test_generated_worker_recomputes_payload_row_and_target_mask_hashes(self):
        import hashlib
        namespace = {"__name__": "generated_worker_test"}
        exec(compare._remote_worker_source(), namespace)
        rows = []
        for identity, text, known, value in (("a" * 64, "row one", True, 1),
                                             ("b" * 64, "row two", False, None)):
            rows.append({"record_id_sha256": identity, "text": text,
                         "text_sha256": compare.sha(text.encode()),
                         "target_known": known, "target_value": value})
        id_set_sha = compare.sha(json.dumps(sorted(row["record_id_sha256"] for row in rows),
                                               separators=(",", ":")).encode())
        mask = [[row["record_id_sha256"], [row["target_known"]]] for row in rows]
        mask_sha = compare.sha(json.dumps(mask, separators=(",", ":")).encode())
        payload = {"schema_version": "fixture", "run_id": "f" * 32,
                   "binding": {"rows": 2, "validation_record_id_set_sha256": id_set_sha,
                               "target_mask_sha256": mask_sha},
                   "diagnostics": {"output_text_utf8_byte_bucket_boundaries": [8, 16]}, "records": rows,
                   "outer_netns_inode": 123, "run_directory": "/private/run"}
        evidence = namespace["verify_payload_evidence"](payload)
        original = {key: value for key, value in payload.items()
                    if key not in ("outer_netns_inode", "run_directory")}
        self.assertEqual(evidence["payload_sha256"], compare.sha(json.dumps(
            original, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()))
        self.assertEqual(evidence["target_mask_sha256"], mask_sha)
        changed = json.loads(json.dumps(payload))
        changed["records"][1]["target_known"] = True
        changed["records"][1]["target_value"] = 0
        with self.assertRaisesRegex(RuntimeError, "target or row identity digest"):
            namespace["verify_payload_evidence"](changed)

    def test_missing_lifecycle_measurement_counts_one_response_and_one_lifecycle_failure(self):
        status_counts = {}
        lifecycle_counts = {}
        result = compare._owned_terminal_prediction("ok", {"intent": "build.software"}, None,
                                                    True, "build.software", status_counts, lifecycle_counts)
        self.assertEqual(result, (None, "unknown_lifecycle_measurement", False))
        self.assertEqual(status_counts, {"ok": 1})
        self.assertEqual(lifecycle_counts, {"unknown_lifecycle_measurement": 1})

    def test_partial_remote_aggregate_requires_explicit_aborted_count(self):
        empty_diagnostics = compare._new_owned_response_diagnostics(
            list(compare.DEFAULT_OUTPUT_TEXT_BYTE_BUCKET_BOUNDARIES))
        full = {"rows_total": 11, "rows_attempted": 11, "owned_requests_started": 0,
                "owned_response_status_counts": {}, "owned_response_diagnostics": empty_diagnostics,
                "aborted": False}
        partial = {"rows_total": 11, "rows_attempted": 4, "owned_requests_started": 0,
                   "owned_response_status_counts": {}, "owned_response_diagnostics": empty_diagnostics,
                   "aborted": True}
        full_but_aborted = {"rows_total": 11, "rows_attempted": 11, "owned_requests_started": 0,
                            "owned_response_status_counts": {}, "owned_response_diagnostics": empty_diagnostics,
                            "aborted": True}
        self.assertIs(compare._validate_worker_aggregate(full, 11), full)
        self.assertIs(compare._validate_worker_aggregate(partial, 11), partial)
        self.assertIs(compare._validate_worker_aggregate(full_but_aborted, 11), full_but_aborted)
        ambiguous = {"rows_total": 11, "rows_attempted": None, "aborted": True,
                     "supervisor_failure": True, "abort_reason": "supervisor_outer_timeout"}
        self.assertIs(compare._validate_worker_aggregate(ambiguous, 11), ambiguous)
        for malformed in (
            {"rows_total": 11, "rows_attempted": 4, "aborted": False},
            {"rows_total": 11, "rows_attempted": 12, "aborted": True},
        ):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                compare._validate_worker_aggregate(malformed, 11)

    def test_remote_payload_digest_must_match_exact_canonical_bytes(self):
        payload = b'{"records":[],"schema_version":"fixture"}'
        digest = compare.sha(payload)
        self.assertEqual(compare._validate_payload_identity(payload, digest), digest)
        with self.assertRaisesRegex(ValueError, "payload identity"):
            compare._validate_payload_identity(payload, "0" * 64)

    def test_mocked_ssh_timeout_survives_full_run_as_unknown_report(self):
        import subprocess
        payload_object = {"schema_version": "fixture", "binding": {"rows": 2},
                          "records": [{"target_known": True}, {"target_known": False}]}
        payload = json.dumps(payload_object, sort_keys=True, separators=(",", ":")).encode()
        config = {
            "remote_host": "clyffy-02", "remote_user": "hades", "remote_home": "/home/hades",
            "ssh_port": 22, "remote_python_path": "/usr/bin/python3", "systemd_run_path": "/usr/bin/systemd-run",
            "private_source_root": "/home/hades/private", "coexisting_service_unit": "lfg.service",
            "coexisting_listener_port": 43127, "host_snapshot_deadline_seconds": 10,
            "min_memory_available_bytes": 1, "min_disk_free_bytes": 1,
            "resource_cpu_quota_percent": 100, "resource_memory_max_bytes": 1024,
            "total_runtime_deadline_seconds": 100, "max_response_bytes": 50000,
            "max_payload_bytes": 1000000, "report_max_bytes": 100000,
            "per_request_deadline_seconds": 10,
            "classical_report_sha256": "a" * 64, "comparison_12b_report_sha256": "b" * 64,
            "service_config_template_sha256": "c" * 64,
        }
        calls = []

        baseline = {"algorithms": {name: {"metrics": None} for name in
                    ("logistic_regression", "knn", "decision_tree")},
                    "task": {"validation_support": {"build.software": {"positive": 1}}},
                    "data": {"plan_sha256": "1" * 64, "review_ledger_sha256": "2" * 64}}
        bound = {
            "comparison_12b_config": {"remote_host": "clyffy-03", "remote_user": "hades"},
            "classical_report": baseline,
            "comparison_12b_report": {"metrics": {"f1": None}},
            "classical_config": {"selected_labels": ["build.software"], "target_field": "intents"},
            "corpus": {"taxonomy_sha256": "3" * 64, "manifest_sha256": "4" * 64,
                       "policy_sha256": "5" * 64, "screen_code_sha256": "6" * 64,
                       "train_sha256": "7" * 64, "validation_sha256": "8" * 64},
            "validation_rows": [{}, {}], "validation_id_set_sha256": "9" * 64,
            "target_mask_sha256": "a" * 64, "classical_config_sha256": "b" * 64,
            "comparison_12b_config_sha256": "c" * 64,
            "review_provenance": {"validation_independence_verified": None},
            "deterministic_files_sha256": {}, "deterministic_assessment_sha256": "d" * 64,
            "deterministic_rules_sha256": "e" * 64,
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report"
            output.mkdir()
            with patch.object(compare, "load_config", return_value=(config, "f" * 64)), \
                    patch.object(compare, "_bind_inputs", return_value=bound), \
                    patch.object(compare, "_make_payload", return_value=payload), \
                    patch.object(compare, "_private_report_directory", return_value=output), \
                    patch.object(compare, "_git_revision", return_value=None), \
                    patch.object(compare, "_ssh_worker", return_value={
                        **compare._unknown_execution(payload_object, "supervisor_client_timeout",
                                                     "unknown_after_client_timeout"),
                        "abort_reason": "supervisor_client_timeout",
                        "remote_workspace_status": "unknown_after_client_timeout"}):
                result = compare.run(Path("unused-config"), output)
            saved = json.loads((output / "owned-matched-validation.json").read_text())
        self.assertIsNone(result["owned_coverage"])
        self.assertIsNone(saved["execution"]["rows_attempted"])
        self.assertEqual(saved["execution"]["abort_reason"], "supervisor_client_timeout")
        self.assertIsNone(saved["execution"]["remote_payload_sha256"])
        self.assertEqual(saved["provenance"]["payload_transfer_state"], "unknown_after_client_timeout")
        self.assertIsNone(saved["execution"]["remote_workspace_retained"])
        self.assertEqual(saved["baseline"]["comparison_12b_host"], "clyffy-03")
        self.assertEqual(saved["execution"]["owned_host"], "clyffy-02")
        self.assertEqual(saved["execution"]["remote_workspace_status"], "unknown_after_client_timeout")
        self.assertEqual(saved["provenance"]["remote_persistence_state"],
                         "unknown because supervisor or transfer did not return a verified worker aggregate")

    def test_supervisor_timeout_cleanup_is_verified_or_workspace_is_retained(self):
        import subprocess
        source = compare._remote_supervisor_body()
        namespace = {"__name__": "supervisor_test"}
        exec(source, namespace)
        with tempfile.TemporaryDirectory() as temp:
            vanished_cgroup = Path(temp) / "observed-cgroup"
            vanished_cgroup.mkdir()
            (vanished_cgroup / "cgroup.procs").write_text("123\n", encoding="utf-8")
            observed = {"unit_observed": True, "cgroup_observed": True,
                        "cgroup_path": str(vanished_cgroup)}
            (vanished_cgroup / "cgroup.procs").write_text("", encoding="utf-8")
            (vanished_cgroup / "cgroup.procs").unlink()
            vanished_cgroup.rmdir()

            def run_released(args, **_kwargs):
                if "stop" in args:
                    return type("R", (), {"returncode": 5, "stdout": ""})()
                if "show" in args:
                    return type("R", (), {"returncode": 0, "stdout":
                        "LoadState=not-found\nActiveState=inactive\nMainPID=0\nControlGroup=\n"})()
                raise AssertionError(args)

            with patch.object(namespace["os"], "getuid", return_value=1000, create=True), \
                    patch.object(namespace["subprocess"], "run", side_effect=run_released):
                cleanup = namespace["stop_and_verify"](
                    "lfg-owned-matched-test.service", 2, "/home/hades", observed)
            self.assertTrue(cleanup["verified"])
            self.assertEqual(cleanup["unit_presence"], "observed_then_released")
            self.assertTrue(cleanup["cgroup_released_after_observation"])
            self.assertIsNone(cleanup["private_namespace_listener_absent"])

            # A missing unit without prior observation is not proof that the
            # generated service never started or has stopped cleanly.
            with patch.object(namespace["os"], "getuid", return_value=1000, create=True), \
                    patch.object(namespace["subprocess"], "run", side_effect=run_released):
                unknown = namespace["stop_and_verify"](
                    "lfg-owned-matched-test.service", 2, "/home/hades", None)
            self.assertFalse(unknown["verified"])
            self.assertEqual(unknown["unit_presence"], "unknown")
            self.assertIsNone(unknown["unit_inactive"])

        def run_active_after_timeout(args, **_kwargs):
            if "stop" in args:
                raise subprocess.TimeoutExpired(args, 2)
            if "show" in args:
                return type("R", (), {"returncode": 0, "stdout":
                    "LoadState=loaded\nActiveState=active\nMainPID=123\nControlGroup=/missing/group\n"})()
            raise AssertionError(args)

        with patch.object(namespace["os"], "getuid", return_value=1000, create=True), \
                patch.object(namespace["subprocess"], "run", side_effect=run_active_after_timeout):
            cleanup = namespace["stop_and_verify"](
                "lfg-owned-matched-test.service", 2, "/home/hades", None)
        self.assertFalse(cleanup["verified"])
        self.assertFalse(cleanup["cgroup_observed_prior"])

        absent = {"verified": False, "unit_presence": "unknown", "cgroup_observed_prior": False,
                  "unit_inactive": None, "cgroup_released_after_observation": None}
        failure = namespace["failure_result"](
            {"binding": {"rows": 1}, "records": [{"target_known": True}]},
            "supervisor_worker_error", absent, "lfg-owned-matched-test.service", "/private/run",
            {"returncode": 218, "stderr_bytes": 0,
             "observed_unit": {"load_state": "loaded", "active_state": "failed",
                               "cgroup_observed": True, "task_count": 0}})
        self.assertTrue(failure["remote_workspace_retained"])
        self.assertEqual(failure["rows_attempted"], None)
        self.assertIsNone(failure["owned_metrics"]["coverage"])
        self.assertEqual(failure["unit_exit_code"], 218)
        self.assertEqual(failure["unit_stderr_bytes"], 0)
        self.assertEqual(failure["unit_observed_while_running"], {
            "load_state": "loaded", "active_state": "failed", "cgroup_observed": True,
            "task_count": 0})
        self.assertNotIn("stderr", failure)
        self.assertIs(compare._validate_worker_aggregate(failure, 1), failure)

    def test_supervisor_requires_worker_payload_and_target_mask_receipt(self):
        namespace = {"__name__": "supervisor_test"}
        exec(compare._remote_supervisor_body(), namespace)
        payload = {"binding": {"rows": 1, "validation_record_id_set_sha256": "a" * 64,
                               "target_mask_sha256": "b" * 64}, "records": []}
        digest = compare.sha(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                        separators=(",", ":"), allow_nan=False).encode())
        worker = {"worker_payload_sha256": digest,
                  "worker_validation_record_id_set_sha256": "a" * 64,
                  "worker_target_mask_sha256": "b" * 64}
        valid, expected, observed = namespace["verify_worker_identity"](payload, worker)
        self.assertTrue(valid)
        self.assertEqual(expected, digest)
        self.assertEqual(observed["target_mask_sha256"], "b" * 64)
        worker["worker_target_mask_sha256"] = "c" * 64
        self.assertFalse(namespace["verify_worker_identity"](payload, worker)[0])

    def test_generated_supervisor_launch_loop_observes_active_cgroup_and_bounds_output(self):
        import io
        namespace = {"__name__": "supervisor_test"}
        exec(compare._remote_supervisor_body(), namespace)
        payload = b"safe-test-payload"
        marker = {"unit_observed": True, "cgroup_observed": True,
                  "cgroup_path": "/sys/fs/cgroup/user.slice/test.service"}

        class FakeChild:
            def __init__(self, argv, **kwargs):
                self.argv = argv
                self.stdin = io.BytesIO()
                self.stdout = kwargs["stdout"]
                self.stderr = kwargs["stderr"]
                self.stdout.write(b'{"rows_attempted":1}')
                self.poll_count = 0
                self.returncode = 0

            def poll(self):
                self.poll_count += 1
                return None if self.poll_count == 1 else 0

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(namespace["os"], "getuid", return_value=1000, create=True), \
                patch.object(namespace["subprocess"], "Popen", FakeChild), \
                patch.dict(namespace, {"observe_unit": lambda _unit, _home: marker}):
            result = namespace["run_unit"](["systemd-run"], payload, 5, "lfg-owned-matched-test.service",
                                            "/home/hades", Path(directory), 1024)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"], b'{"rows_attempted":1}')
        self.assertEqual(result["observed_unit"], marker)
        self.assertFalse(result["timed_out"])

    def test_http_is_loopback_authenticated_bounded_and_does_not_follow_redirects(self):
        with patch.object(http.client, "HTTPConnection", _Connection):
            status, headers, body = compare._http_loopback(
                43210, "/api/lfg/assess/owned/status", token="fresh-secret", timeout=3, max_bytes=32)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"ok": True})

        instance = _Connection("127.0.0.1", 43210, 3)
        with patch.object(http.client, "HTTPConnection", return_value=instance):
            compare._http_loopback(43210, "/api/lfg/assess/owned", "POST", b"{}", "fresh-secret")
        self.assertEqual(instance.host, "127.0.0.1")
        self.assertEqual(instance.headers["X-LFG-Session-Token"], "fresh-secret")

        redirect = _Connection("127.0.0.1", 43210, 3)
        redirect.response = _Response(302, headers=[("Location", "http://outside.invalid")])
        with patch.object(http.client, "HTTPConnection", return_value=redirect):
            with self.assertRaisesRegex(ValueError, "redirects are forbidden"):
                compare._http_loopback(43210, "/api/lfg/assess/owned")

        oversized = _Connection("127.0.0.1", 43210, 3)
        oversized.response = _Response(200, b"x" * 20)
        with patch.object(http.client, "HTTPConnection", return_value=oversized):
            with self.assertRaisesRegex(ValueError, "exceeds configured limit"):
                compare._http_loopback(43210, "/status", max_bytes=8)

    def test_remote_worker_and_supervisor_compile_and_pin_isolation_and_lifecycle(self):
        import base64
        import inspect
        worker = compare._remote_worker_source()
        compile(worker, "remote-owned-worker", "exec")
        supervisor = compare._remote_supervisor_body()
        compile(supervisor, "remote-owned-supervisor", "exec")
        launcher = compare._remote_scope_launcher_body()
        compile(launcher, "remote-owned-scope-launcher", "exec")
        self.assertIn("unshare','--net", launcher)
        self.assertIn("setpriv --reuid", launcher)
        self.assertIn("--no-new-privs", launcher)
        self.assertIn("sys.stdin.buffer.read(req['max_payload']+1)", launcher)
        self.assertLess(launcher.index("sys.stdout.buffer.write(ready)"),
                        launcher.index("sys.stdin.buffer.read(req['max_payload']+1)"))
        self.assertIn("unit.removesuffix('.scope')", supervisor)
        self.assertIn("os.killpg(child.pid", worker)
        self.assertIn("worker did not enter a distinct network namespace", worker)
        self.assertIn("redirects are forbidden", worker)

    def test_scope_preflight_requires_privilege_drop_and_distinct_private_netns(self):
        preflight = {"type": "preflight", "ready": True, "username": "hades", "uid": 1000,
                     "gid": 1000, "scope_unit": "lfg-owned-matched-0123456789abcdef01.scope",
                     "outer_netns_inode": 10, "netns_inode": 11, "interfaces": ["lo"],
                     "ipv4_routes": [], "ipv6_routes": [], "capabilities_empty": True,
                     "no_new_privs": True, "owner_groups_verified": True,
                     "scope_cgroup_observed": True}
        self.assertEqual(compare._validate_scope_preflight(preflight, "hades",
                                                          "0123456789abcdef0123456789abcdef"), preflight)
        for key, invalid in (("netns_inode", 10), ("capabilities_empty", False),
                             ("no_new_privs", False), ("ipv4_routes", ["default"])):
            bad = dict(preflight); bad[key] = invalid
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "preflight"):
                compare._validate_scope_preflight(bad, "hades", "0123456789abcdef0123456789abcdef")

    def test_config_accepts_extension_fields_without_assigning_them_behavior(self):
        fields = {
            "schema_version": compare.SCHEMA, "revision": "fixture",
            "classical_report": "C:/private/report.json", "classical_report_sha256": "a" * 64,
            "classical_config": "C:/private/classical.json", "classical_config_sha256": "b" * 64,
            "comparison_12b_report": "C:/private/12b.json", "comparison_12b_report_sha256": "c" * 64,
            "comparison_12b_config": "C:/private/12b-config.json", "comparison_12b_config_sha256": "d" * 64,
            "remote_host": "clyffy-02", "remote_user": "hades", "remote_home": "/home/hades", "ssh_port": 22,
            "remote_python_path": "/usr/bin/python3", "systemd_run_path": "/usr/bin/systemd-run",
            "private_source_root": "/home/hades/private/fixture", "service_binary": "/home/hades/private/fixture/lfg-server",
            "service_binary_sha256": "e" * 64, "service_config_template": "/home/hades/private/fixture/service.json",
            "service_config_template_sha256": "a" * 64,
            "metadata_directory": "/home/hades/private/fixture/metadata", "expected_model_sha256": "f" * 64,
            "expected_framing_sha256": "1" * 64, "expected_cached_manifest_sha256": "2" * 64,
            "expected_runtime_config_sha256": "3" * 64, "expected_training_manifest_sha256": "4" * 64,
            "coexisting_service_unit": "lfg-owned.service", "coexisting_listener_port": 43127,
            "resource_cpu_quota_percent": 400,
            "resource_memory_max_bytes": 8_589_934_592, "min_memory_available_bytes": 1,
            "min_disk_free_bytes": 1, "max_source_bytes": 100, "max_total_source_bytes": 100,
            "max_manifest_bytes": 100, "max_screened_file_bytes": 100, "max_policy_bytes": 100,
            "max_ledger_bytes": 100, "max_config_bytes": 100, "max_payload_bytes": 10000,
            "max_response_bytes": 10000, "max_report_bytes": 1_000_000, "per_request_deadline_seconds": 615,
            "total_runtime_deadline_seconds": 8000, "report_max_bytes": 1_000_000,
            "host_snapshot_deadline_seconds": 30, "file_preflight_deadline_seconds": 60,
            "file_transfer_deadline_seconds": 30, "file_cleanup_deadline_seconds": 30,
            "file_max_binary_bytes": 1000000000, "file_max_metadata_bytes": 1000000,
            "file_min_gpu_free_bytes": 0,
            "output_text_utf8_byte_bucket_boundaries": [4, 8],
            "expected_remote_architecture": "aarch64",
            "extension": {"ignored": True},
        }
        if os.name == "posix":
            for name in ("classical_report", "classical_config", "comparison_12b_report", "comparison_12b_config"):
                fields[name] = "/private/fixture.json"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(fields), encoding="utf-8")
            loaded, digest = compare.load_config(path)
            self.assertEqual(loaded["output_text_utf8_byte_bucket_boundaries"], [4, 8])
            for invalid in ([4, 4], [True], [-1], None, [1] * 17):
                fields["output_text_utf8_byte_bucket_boundaries"] = invalid
                path.write_text(json.dumps(fields), encoding="utf-8")
                with self.subTest(boundaries=invalid), self.assertRaisesRegex(ValueError, "output_text_utf8"):
                    compare.load_config(path)
            fields["output_text_utf8_byte_bucket_boundaries"] = [4, 8]
            fields["expected_remote_architecture"] = "../invalid"
            path.write_text(json.dumps(fields), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "architecture"):
                compare.load_config(path)
            fields["expected_remote_architecture"] = "aarch64"
        self.assertEqual(loaded["extension"], {"ignored": True})
        self.assertEqual(loaded["output_text_utf8_byte_bucket_boundaries"], [4, 8])
        self.assertEqual(digest, compare.sha(json.dumps(fields).encode()))


if __name__ == "__main__":
    unittest.main()
