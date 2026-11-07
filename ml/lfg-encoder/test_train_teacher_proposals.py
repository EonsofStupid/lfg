from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import train_teacher_proposals as adapter
import masked_supervision as masked


def row(row_id="a" * 64):
    columns = [{"field": "intents", "label": "build.software"},
               {"field": "risks", "label": "sensitive-data"}]
    text = "PRIVATE_SENTINEL user prompt"
    return {"schema_version": adapter.MASKED_ROW_SCHEMA, "id": row_id, "split": "train",
            "family": "1" * 64, "text": text, "text_sha256": adapter._sha(text.encode()),
            "original_text_sha256": "2" * 64,
            "screening": {"policy_sha256": "3" * 64, "redaction_counts": {"email": 1},
                          "text_changed": True},
            "source_identity": {"artifact_path": "source-0001.jsonl", "artifact_sha256": "4" * 64,
                "project_claim_sha256": "5" * 64, "project_group_claim_sha256": "6" * 64,
                "record_id_sha256": row_id, "source_claim_sha256": "7" * 64},
            "columns": columns, "targets": [None, 0], "known_mask": [False, True],
            "weights": [0, 1], "target_origins": ["unknown", "reviewed"],
            "target_reviewer_claims": [None, "8" * 64], "verified_source_claim": False,
            "partial_review_provenance": {"extra": "never transfer"}}


def binding():
    return {"manifest_sha256": "a" * 64, "train_sha256": "b" * 64,
            "taxonomy_sha256": "c" * 64, "config_sha256": "d" * 64,
            "proposal_config_sha256": "e" * 64, "comparison_config_sha256": "f" * 64,
            "adapter_code_sha256": "0" * 64, "comparator_code_sha256": "9" * 64}


def worker_expectation(zero=False):
    return {"payload_binding": binding(), "zero_row_probe": zero, "max_rows": 3,
            "selected_columns": row()["columns"], "export_manifest_sha256": "a" * 64,
            "train_sha256": "b" * 64, "taxonomy_sha256": "c" * 64,
            "export_config_sha256": "d" * 64, "screening_policy_sha256": "3" * 64}


class TrainOnlySnapshotTests(unittest.TestCase):
    def _export(self, root: Path, manifest_obj: dict):
        export = root / "export"
        export.mkdir()
        config_raw = b'{"schema_version":"x"}\n'
        train_raw = b'{"one":"train only"}\n'
        manifest_obj = {**manifest_obj, "config_sha256": adapter._sha(config_raw),
                        "taxonomy_sha256": "c" * 64, "screening_policy_sha256": "3" * 64,
                        "screening_code_sha256": "4" * 64, "selected_columns": row()["columns"],
                        "output_files": manifest_obj["output_files"]}
        (export / "config.json").write_bytes(config_raw)
        (export / "train.jsonl").write_bytes(train_raw)
        raw = (json.dumps(manifest_obj, sort_keys=True) + "\n").encode()
        (export / "manifest.json").write_bytes(raw)
        cfg = {"export_dir": str(export), "export_manifest_sha256": adapter._sha(raw),
               "export_config_sha256": adapter._sha(config_raw), "export_train_sha256": adapter._sha(train_raw),
               "max_export_bytes": len(train_raw), "taxonomy_sha256": "c" * 64,
               "screening_policy_sha256": "3" * 64, "screening_code_sha256": "4" * 64,
               "selected_columns": row()["columns"], "selection": {"row_ids": []}}
        return export, cfg, manifest_obj, config_raw, train_raw

    def test_matching_digest_manifest_with_validation_split_aborts_before_any_payload_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp, cfg, manifest, config_raw, train_raw = self._export(Path(tmp), {
                "output_files": {"train.jsonl": {"bytes": len(b'{"one":"train only"}\n'),
                    "sha256": adapter._sha(b'{"one":"train only"}\n')},
                    "validation.jsonl": {"bytes": 1, "sha256": "0" * 64}}})
            read_paths = []
            real_read = adapter._read
            def recording_read(path, *args):
                read_paths.append(Path(path).name)
                return Path(path).read_bytes()
            class FakeMasked:
                @staticmethod
                def _read_bounded(path, limit, _what):
                    raw = Path(path).read_bytes()
                    if len(raw) > limit:
                        raise ValueError("too large")
                    return raw
            with mock.patch.object(adapter, "_private_dir", return_value=True), \
                 mock.patch.object(adapter, "_private_file", return_value=True), \
                 mock.patch.object(adapter, "_read", side_effect=recording_read), \
                 mock.patch.object(adapter, "_masked", return_value=FakeMasked()):
                with self.assertRaises(adapter.ProposalError):
                    adapter._preflight_export(cfg)
            self.assertEqual(read_paths, ["manifest.json"])
            self.assertFalse((exp / "validation.jsonl").exists())

    def test_manifest_swap_after_pin_uses_only_original_train_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_raw = b'{"one":"train only"}\n'
            exp, cfg, manifest, config_raw, _ = self._export(Path(tmp), {
                "output_files": {"train.jsonl": {"bytes": len(train_raw), "sha256": adapter._sha(train_raw)}}})
            replacement = {**manifest, "output_files": {**manifest["output_files"],
                "validation.jsonl": {"bytes": 9, "sha256": "0" * 64}}}
            source_reads = []
            real_read = adapter._read
            first = True
            def swap_after_manifest(path, *args):
                nonlocal first
                path = Path(path)
                source_reads.append(path.name)
                raw = real_read(path, *args)
                if path.name == "manifest.json" and first:
                    first = False
                    (exp / "manifest.json").write_text(json.dumps(replacement), encoding="utf-8")
                return raw
            loader_inventory = []
            class FakeMasked:
                @staticmethod
                def _read_bounded(path, limit, _what):
                    raw = Path(path).read_bytes()
                    if len(raw) > limit:
                        raise ValueError("too large")
                    return raw
                @staticmethod
                def _verify_owner_private_access(*_a, **_k):
                    return None
                @staticmethod
                def _secure_private_path(*_a, **_k):
                    return None
                @staticmethod
                def load_export(snapshot, train_only=False):
                    loader_inventory.extend(p.name for p in snapshot.iterdir())
                    self.assertEqual(train_only, True)
                    self.assertEqual(json.loads((snapshot / "manifest.json").read_text())["output_files"].keys(), {"train.jsonl"})
                    return manifest, {"train": [row()]}
            with mock.patch.object(adapter, "_private_dir", return_value=True), \
                 mock.patch.object(adapter, "_private_file", return_value=True), \
                 mock.patch.object(adapter, "_read", side_effect=swap_after_manifest), \
                 mock.patch.object(adapter, "_masked", return_value=FakeMasked()):
                loaded, rows, got_binding = adapter._preflight_export(cfg)
            self.assertEqual(loaded, manifest)
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(source_reads), {"manifest.json", "config.json", "train.jsonl"})
            self.assertEqual(source_reads.count("manifest.json"), 1)
            self.assertNotIn("validation.jsonl", loader_inventory)
            self.assertEqual(got_binding["manifest_sha256"], cfg["export_manifest_sha256"])

    def test_real_train_only_snapshot_loader_accepts_owner_private_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            columns = [{"field": "intents", "label": "build.software"}]
            text = "local sanitized fixture"
            row_id = "a" * 64
            raw_row = {
                "schema_version": masked.ROW_SCHEMA, "id": row_id, "split": "train",
                "family": "b" * 64, "text": text,
                "text_sha256": adapter._sha(text.encode("utf-8")),
                "source_identity": {"artifact_path": "source.jsonl", "artifact_sha256": "c" * 64,
                    "project_claim_sha256": "d" * 64, "project_group_claim_sha256": "e" * 64,
                    "record_id_sha256": row_id, "source_claim_sha256": "f" * 64},
                "columns": columns, "targets": [None], "known_mask": [False], "weights": [0],
                "target_origins": ["unknown"], "target_reviewer_claims": [None],
            }
            train_raw = (json.dumps(raw_row, sort_keys=True) + "\n").encode("utf-8")
            export_config = {
                "schema_version": masked.CONFIG_SCHEMA, "workspace": str(root / "workspace"),
                "output": str(root / "export"), "screening_policy": str(root / "policy.json"),
                "guide": None, "selected_labels": {field: ["label"] for field in masked.FIELDS},
                "allow_unreviewed_proposals": True, "unreviewed_complete_array_fields": [],
                "rejected_record_policy": "exclude", "stale_retracted_event_policy": "omit_record",
                "weights": {"default": 1.0, "per_label": {}},
                "limits": {key: 100_000 for key in masked.LIMIT_FIELDS},
                "near_duplicate": {"similarity": 0.98},
            }
            config_raw = (json.dumps(export_config, sort_keys=True) + "\n").encode("utf-8")
            export = adapter._private_temp_directory("lfg-teacher-test-export")
            try:
                adapter._write_new(export / "config.json", config_raw)
                adapter._write_new(export / "train.jsonl", train_raw)
                manifest = {"schema_version": masked.EXPORT_SCHEMA,
                    "config_sha256": adapter._sha(config_raw), "selected_columns": columns,
                    "taxonomy_sha256": "1" * 64, "screening_policy_sha256": "2" * 64,
                    "screening_code_sha256": "3" * 64,
                    "output_files": {"train.jsonl": {"bytes": len(train_raw),
                        "sha256": adapter._sha(train_raw)}}}
                manifest_raw = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
                adapter._write_new(export / "manifest.json", manifest_raw)
                cfg = {"export_dir": str(export), "export_manifest_sha256": adapter._sha(manifest_raw),
                    "export_config_sha256": adapter._sha(config_raw),
                    "export_train_sha256": adapter._sha(train_raw), "max_export_bytes": len(train_raw),
                    "taxonomy_sha256": "1" * 64, "screening_policy_sha256": "2" * 64,
                    "screening_code_sha256": "3" * 64, "selected_columns": columns,
                    "selection": {"row_ids": []}}
                loaded, rows, _binding = adapter._preflight_export(cfg)
                self.assertEqual(loaded, manifest)
                self.assertEqual([item["id"] for item in rows], [row_id])
                self.assertEqual({p.name for p in export.iterdir()},
                                 {"config.json", "manifest.json", "train.jsonl"})
            finally:
                import shutil
                shutil.rmtree(export)


class OwnerPrivateTempTests(unittest.TestCase):
    def test_uuid_collision_preserves_preexisting_empty_directory(self):
        token = "a" * 32
        prefix = "lfg-teacher-collision"
        with tempfile.TemporaryDirectory(prefix="lfg-teacher-collision-case-") as base:
            existing = Path(base) / f"{prefix}-{token}"
            existing.mkdir()
            with mock.patch.object(adapter.tempfile, "gettempdir", return_value=base), \
                 mock.patch.object(adapter.uuid, "uuid4", return_value=SimpleNamespace(hex=token)):
                with self.assertRaises(adapter.ProposalError):
                    adapter._private_temp_directory(prefix)
            self.assertTrue(existing.is_dir())
            self.assertEqual(list(existing.iterdir()), [])

    def test_windows_temp_directory_and_file_have_real_owner_only_acl_before_and_after_write(self):
        if os.name != "nt":
            self.skipTest("native Windows ACL verification requires Windows")
        directory = adapter._private_temp_directory("lfg-teacher-acl-e2e")
        try:
            masked._verify_owner_private_access(directory, is_directory=True)
            path = directory / "worker-config.json"
            # _write_new establishes and verifies the strict ACL before payload bytes.
            adapter._write_new(path, b'{"fixture":"safe"}\n')
            masked._verify_owner_private_access(path, is_directory=False)
            self.assertEqual(path.read_bytes(), b'{"fixture":"safe"}\n')
        finally:
            import shutil
            shutil.rmtree(directory)

    @unittest.skipIf(os.name == "nt", "POSIX mode assertions are platform-specific")
    def test_posix_temp_directory_and_file_use_owner_only_modes(self):
        directory = adapter._private_temp_directory("lfg-teacher-acl-posix")
        try:
            self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
            masked._verify_owner_private_access(directory, is_directory=True)
            path = directory / "snapshot.json"
            adapter._write_new(path, b"fixture\n")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            masked._verify_owner_private_access(path, is_directory=False)
        finally:
            import shutil
            shutil.rmtree(directory)

    def test_acl_setup_failure_writes_no_payload_and_removes_only_empty_task_paths(self):
        with tempfile.TemporaryDirectory() as base, \
             mock.patch.object(adapter.tempfile, "gettempdir", return_value=base), \
             mock.patch.object(masked, "_secure_private_path", side_effect=ValueError("private sentinel")):
            with self.assertRaises(adapter.ProposalError):
                adapter._private_temp_directory("lfg-teacher-acl-fail")
            self.assertEqual(list(Path(base).iterdir()), [])
        with tempfile.TemporaryDirectory() as base:
            directory = adapter._private_temp_directory("lfg-teacher-acl-file-fail")
            path = directory / "secret.json"
            with mock.patch.object(masked, "_secure_private_path", side_effect=ValueError("private sentinel")):
                with self.assertRaises(ValueError):
                    adapter._write_new(path, b"must-not-be-written")
            self.assertFalse(path.exists())
            import shutil
            shutil.rmtree(directory)


class PublicationTests(unittest.TestCase):
    def _publish(self, root: Path, status="complete"):
        parent = root / "private-output"
        parent.mkdir()
        final = parent / "run-1"
        manifest = {"schema_version": adapter.REPORT_SCHEMA, "status": status,
                    "promotion_eligible": False}
        with mock.patch.object(adapter, "_private_dir", return_value=True), \
             mock.patch.object(adapter, "_outside_git", return_value=True), \
             mock.patch.object(adapter.comparator, "_private_report_directory",
                               side_effect=lambda path: path.mkdir(mode=0o700) or path):
            return parent, final, adapter._atomic_private_publish(final, [{"status": "answered"}], manifest)

    def test_atomic_publish_preserves_complete_and_aborted_status(self):
        for status in ("complete", "partial_aborted_ambiguous"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                parent, final, (published, _sha) = self._publish(Path(tmp), status)
                self.assertEqual(published, final)
                self.assertEqual(json.loads((final / "manifest.json").read_text())["status"], status)
                self.assertFalse(list(parent.glob("*.partial-*")))

    def test_publication_interruption_never_leaves_complete_marker_in_partial_directory(self):
        real_write = adapter._write_new
        for fail_at in (1, 2, 3):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp) / "private-output"
                parent.mkdir()
                final = parent / "run-1"
                writes = 0
                def interrupted_write(path, raw):
                    nonlocal writes
                    writes += 1
                    real_write(path, raw)
                    if writes == fail_at:
                        raise OSError("injected interruption")
                with mock.patch.object(adapter, "_private_dir", return_value=True), \
                     mock.patch.object(adapter, "_outside_git", return_value=True), \
                     mock.patch.object(adapter.comparator, "_private_report_directory",
                                       side_effect=lambda path: path.mkdir(mode=0o700) or path), \
                     mock.patch.object(adapter, "_write_new", side_effect=interrupted_write):
                    with self.assertRaises(OSError):
                        adapter._atomic_private_publish(final, [{"status": "answered"}],
                            {"schema_version": adapter.REPORT_SCHEMA, "status": "complete"})
                dirs = list(parent.iterdir())
                self.assertEqual(len(dirs), 0 if fail_at == 1 else 1)
                if not dirs:
                    self.assertFalse(final.exists())
                    continue
                marker = dirs[0] / "manifest.json"
                if marker.exists():
                    self.assertEqual(json.loads(marker.read_text())["status"], "incomplete")
                self.assertFalse((dirs[0] / ".manifest-complete.tmp").exists())

    def test_rename_and_final_replace_interruptions_leave_incomplete_manifest(self):
        for point in ("rename", "replace"):
            with self.subTest(point=point), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp) / "private-output"
                parent.mkdir()
                final = parent / "run-1"
                patcher = mock.patch.object(adapter.os, "rename", side_effect=OSError("interrupt")) if point == "rename" else \
                          mock.patch.object(adapter.os, "replace", side_effect=OSError("interrupt"))
                with mock.patch.object(adapter, "_private_dir", return_value=True), \
                     mock.patch.object(adapter, "_outside_git", return_value=True), \
                     mock.patch.object(adapter.comparator, "_private_report_directory",
                                       side_effect=lambda path: path.mkdir(mode=0o700) or path), patcher:
                    with self.assertRaises(OSError):
                        adapter._atomic_private_publish(final, [{"status": "answered"}],
                            {"schema_version": adapter.REPORT_SCHEMA, "status": "partial_aborted_ambiguous"})
                dirs = list(parent.iterdir())
                self.assertEqual(len(dirs), 1)
                marker = dirs[0] / "manifest.json"
                self.assertTrue(marker.exists())
                self.assertEqual(json.loads(marker.read_text())["status"], "incomplete")
                self.assertFalse((dirs[0] / ".manifest-complete.tmp").exists())


class WorkerProtocolTests(unittest.TestCase):
    def test_controller_acknowledges_only_after_outside_denial_then_waits_for_verified_ready(self):
        order = []
        runtime = {"boundary": "same_uid_private_network_namespace_without_nonloopback_routes",
                   "network_namespace_inode": 22, "server_pid": 50,
                   "server_binary_sha256": "c" * 64, "server_version": "v1"}
        control = {"phase": "control_ready", "control_nonce": "1" * 32,
                   "network_namespace_inode": 22, "server_pid": 50, "port": 11434,
                   "receipt_sha256": "e" * 64, "server_binary_sha256": "c" * 64,
                   "server_version": "v1", "worker_code_sha256": "a" * 64,
                   "worker_config_sha256": "b" * 64, "runtime_boundary": runtime}
        ready = {"phase": "ready_for_data", "network_namespace_inode": 22,
                 "server_pid": 50, "port": 11434, "receipt_sha256": "e" * 64,
                 "server_binary_sha256": "c" * 64, "server_version": "v1",
                 "worker_code_sha256": "a" * 64, "worker_config_sha256": "b" * 64,
                 "runtime_boundary": runtime, "model_identity": {"reported_manifest_sha256": "d" * 64},
                 "model_manifest_sha256": "d" * 64, "model_layer_bytes": 100,
                 "artifact_layer_bytes": 100, "warmup": {"status": "completed"}}

        class Writer:
            def __init__(self):
                self.data = bytearray()
            def write(self, value):
                self.data.extend(value)
                order.append("ack")
            def flush(self):
                pass

        process = SimpleNamespace(stdin=Writer())
        def read_line(*_args):
            order.append("control" if len(order) == 0 else "ready")
            return control if len(order) == 1 else ready
        comp = {"worker_start_timeout_seconds": 5, "worker_ready_wait_margin_seconds": 1,
                "timeout_seconds": 10, "server_binary_sha256": "c" * 64,
                "server_version": "v1", "runtime_port": 11434}
        with mock.patch.object(adapter, "_read_worker_line", side_effect=read_line), \
             mock.patch.object(adapter.comparator, "_remote_port_closed",
                               side_effect=lambda _c: order.append("outside_denied") or True):
            got = adapter._worker_warmup_handshake(process, object(), comp,
                {"netns_inode": 11}, "a" * 64, "b" * 64, "e" * 64)
        self.assertIs(got, ready)
        self.assertEqual(order, ["control", "outside_denied", "ack", "ready"])
        ack = json.loads(bytes(process.stdin.data).decode().strip())
        self.assertEqual(ack["phase"], "warmup_ack")
        self.assertEqual(ack["control_nonce"], control["control_nonce"])

    def test_controller_denial_sends_no_warmup_ack(self):
        control = {"phase": "control_ready", "control_nonce": "1" * 32,
                   "network_namespace_inode": 22, "server_pid": 50, "port": 11434,
                   "receipt_sha256": "e" * 64, "server_binary_sha256": "c" * 64,
                   "server_version": "v1", "worker_code_sha256": "a" * 64,
                   "worker_config_sha256": "b" * 64,
                   "runtime_boundary": {"boundary": "same_uid_private_network_namespace_without_nonloopback_routes",
                       "network_namespace_inode": 22, "server_pid": 50,
                       "server_binary_sha256": "c" * 64, "server_version": "v1"}}
        writer = SimpleNamespace(data=bytearray(), write=lambda value: None, flush=lambda: None)
        process = SimpleNamespace(stdin=writer)
        comp = {"worker_start_timeout_seconds": 5, "worker_ready_wait_margin_seconds": 1,
                "timeout_seconds": 10, "server_binary_sha256": "c" * 64,
                "server_version": "v1", "runtime_port": 11434}
        with mock.patch.object(adapter, "_read_worker_line", return_value=control), \
             mock.patch.object(adapter.comparator, "_remote_port_closed", return_value=False):
            with self.assertRaises(adapter.ProposalError):
                adapter._worker_warmup_handshake(process, object(), comp,
                    {"netns_inode": 11}, "a" * 64, "b" * 64, "e" * 64)
        self.assertEqual(writer.data, bytearray())

    def test_zero_probe_to_actual_worker_requires_exact_artifact_and_model_identity(self):
        host = {"hostname": "worker", "architecture": "x86_64"}
        ready = {"model_manifest_sha256": "d" * 64, "model_layer_bytes": 100,
                 "artifact_layer_bytes": 100, "server_binary_sha256": "c" * 64,
                 "server_version": "v1"}
        probe = {"zero_row_probe": True, "cleanup_verified": True,
                 "outside_listener_denied": True, "persistent_service_unchanged": True,
                 "proposal_config_sha256": "a" * 64, "comparison_config_sha256": "b" * 64,
                 "adapter_code_sha256": "1" * 64, "comparator_code_sha256": "2" * 64,
                 "receipt_sha256": "e" * 64, "host_before": dict(host),
                 "model_identity": {"reported_manifest_sha256": "d" * 64},
                 "model_layer_bytes": 100, "artifact": {"model_layer_bytes": 100},
                 "runtime_boundary": {"server_binary_sha256": "c" * 64, "server_version": "v1"}}
        adapter._validate_zero_probe_match(probe, ready, host, "a" * 64, "b" * 64,
                                           "1" * 64, "2" * 64, "e" * 64)
        changed = json.loads(json.dumps(probe))
        changed["artifact"]["model_layer_bytes"] = 101
        with self.assertRaises(adapter.ProposalError):
            adapter._validate_zero_probe_match(changed, ready, host, "a" * 64, "b" * 64,
                                               "1" * 64, "2" * 64, "e" * 64)
        changed = json.loads(json.dumps(probe))
        changed["model_identity"]["reported_manifest_sha256"] = "f" * 64
        with self.assertRaises(adapter.ProposalError):
            adapter._validate_zero_probe_match(changed, ready, host, "a" * 64, "b" * 64,
                                               "1" * 64, "2" * 64, "e" * 64)

    def test_isolated_python_bootstrap_imports_digest_pinned_sibling_module(self):
        adapter_path = Path(adapter.__file__).resolve()
        comparator_path = Path(adapter.comparator.__file__).resolve()
        config = {"worker_code_sha256": adapter._sha(adapter_path.read_bytes()),
                  "comparator_code_sha256": adapter._sha(comparator_path.read_bytes())}
        raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "worker-config.json"
            config_path.write_bytes(raw)
            result = subprocess.run([sys.executable, "-I", "-c", adapter._isolated_worker_bootstrap(),
                                     str(adapter_path), str(config_path), adapter._sha(raw)],
                                    capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Teacher proposal task failed", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_isolated_bootstrap_rejects_tampered_config_adapter_and_comparator(self):
        original_adapter = Path(adapter.__file__).read_bytes()
        original_comparator = Path(adapter.comparator.__file__).read_bytes()
        for target in ("config", "adapter", "comparator"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                a_path, q_path, c_path = root / "train_teacher_proposals.py", root / "compare_local_12b.py", root / "config.json"
                a_path.write_bytes(original_adapter)
                q_path.write_bytes(original_comparator)
                config = {"worker_code_sha256": adapter._sha(original_adapter),
                          "comparator_code_sha256": adapter._sha(original_comparator)}
                config_raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
                c_path.write_bytes(config_raw)
                expected = adapter._sha(config_raw)
                if target == "config":
                    c_path.write_bytes(config_raw + b" ")
                elif target == "adapter":
                    a_path.write_bytes(original_adapter + b"\n# changed")
                else:
                    q_path.write_bytes(original_comparator + b"\n# changed")
                result = subprocess.run([sys.executable, "-I", "-c", adapter._isolated_worker_bootstrap(),
                                         str(a_path), str(c_path), expected],
                                        capture_output=True, text=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Teacher proposal task failed", result.stderr)

    def test_payload_whitelists_and_validates_only_pinned_train_fields(self):
        safe_row = adapter._payload_row(row())
        self.assertNotIn("partial_review_provenance", safe_row)
        payload = {"schema_version": adapter.PAYLOAD_SCHEMA, "binding": binding(),
                   "records": [safe_row], "zero_row_probe": False}
        records, columns = adapter._validate_worker_payload(payload, worker_expectation())
        self.assertEqual(len(records), 1)
        self.assertEqual(columns, row()["columns"])
        payload["records"][0]["split"] = "validation"
        with self.assertRaises(adapter.ProposalError):
            adapter._validate_worker_payload(payload, worker_expectation())

    def test_worker_loop_aborts_after_ambiguous_failure_and_emits_no_raw_content(self):
        rows = [row(), row("1" * 64)]
        config = {"system_prompt": "Choose {labels}. {text}", "temperature": 0,
                  "model_name": "local:tag", "keep_alive": "1m", "num_ctx": 1024, "num_predict": 20, "seed": 1, "top_k": 1,
                  "top_p": 1, "repeat_penalty": 1}
        calls = []
        def infer(*_):
            calls.append(True)
            raise TimeoutError("PRIVATE_SENTINEL timeout detail")
        progress, results, aborted, reason = adapter._worker_loop(
            rows, rows[0]["columns"], config, infer, False, 8192)
        self.assertEqual(len(calls), 1)
        self.assertTrue(aborted)
        self.assertEqual(reason, "timeout_or_worker_lifecycle_ambiguous")
        self.assertEqual([x["status"] for x in results], ["timeout", "not_attempted_after_abort"])
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(progress + results))
        self.assertNotIn("partial_review_provenance", adapter._payload_row(rows[0]))
        self.assertEqual(adapter._validate_progress_event(progress[0], 0, rows[0], rows[0]["columns"], False), results[0])
        malformed_elapsed = json.loads(json.dumps(progress[0]))
        malformed_elapsed["elapsed_ms"] = 1.0
        malformed_elapsed["result"]["elapsed_ms"] = 1.0
        with self.assertRaises(adapter.ProposalError):
            adapter._validate_progress_event(malformed_elapsed, 0, rows[0], rows[0]["columns"], False)
        self.assertEqual(adapter._validate_progress_result(progress[0], 0, rows[0], rows[0]["columns"], False), results[0])
        changed = json.loads(json.dumps(progress[0]))
        changed["result"]["source_target_origins"][0] = "fabricated"
        with self.assertRaises(adapter.ProposalError):
            adapter._validate_progress_result(changed, 0, rows[0], rows[0]["columns"], False)

    def test_exhaustiveness_is_explicit_and_conflicts_remain_claims(self):
        value = row()
        value["targets"] = [1, None]
        value["known_mask"] = [True, False]
        config = {"system_prompt": "{labels} {text}", "temperature": 0,
                  "model_name": "local:tag", "keep_alive": "1m", "num_ctx": 1024, "num_predict": 20, "seed": 1, "top_k": 1,
                  "top_p": 1, "repeat_penalty": 1}
        _, output, _, _ = adapter._worker_loop([value], value["columns"], config,
            lambda *_: ("answered", ["risks:sensitive-data"], 1.0), True, 8192)
        self.assertEqual(output[0]["proposal_states"]["intents:build.software"], "proposed_negative")
        self.assertEqual(output[0]["conflict_labels"], ["intents:build.software"])

    def test_generated_remote_scripts_compile_and_check_owner_before_restrictive_mode(self):
        captured = []
        comp = {"remote_home": "/home/lfg", "remote_workdir": "/home/lfg/.cache/task",
                "remote_uid": 1234}
        with mock.patch.object(adapter.comparator, "_ssh_run", side_effect=lambda c, s, **k: captured.append(s) or mock.Mock(returncode=0, stdout=b'{"verified":true,"files":1}')):
            adapter._remote_create_workdir(comp, "/home/lfg/.cache/task/run-" + "a" * 32)
            adapter._remote_verify_files(comp, {"workdir": "/home/lfg/.cache/task/run-" + "a" * 32},
                {"/home/lfg/.cache/task/run-" + "a" * 32 + "/x.py": ("a" * 64, 4)})
        for script in captured:
            compile(script, "<remote-script>", "exec")
        self.assertIn("st.st_uid!=uid", captured[1])
        self.assertIn("stat.S_ISREG(st.st_mode)", captured[1])
        cleanup_cfg = {"remote_home": "/home/lfg", "remote_workdir": "/home/lfg/.cache/task",
                       "remote_uid": 1234, "remote_sudo_path": "/usr/bin/sudo",
                       "remote_systemctl_path": "/usr/bin/systemctl", "remote_host": "worker.test",
                       "remote_user": "lfg", "remote_python_path": "/usr/bin/python3", "ssh_port": 22}
        with mock.patch.object(adapter.comparator, "_ssh_run", side_effect=lambda c, s, **k: captured.append(s) or mock.Mock(returncode=0, stdout=b"")):
            adapter._remote_cleanup_worker(cleanup_cfg, "/home/lfg/.cache/task/run-" + "a" * 32,
                                            "lfg12b-" + "a" * 20 + ".scope")
        compile(captured[-1], "<cleanup-script>", "exec")
        self.assertIn("systemctl,'is-active','--quiet'", captured[-1])
        self.assertIn("state.returncode not in (3,4)", captured[-1])

    def test_generated_worker_entry_runs_real_loop_and_emits_bounded_protocol(self):
        source_row = row()
        payload = {"schema_version": adapter.PAYLOAD_SCHEMA, "binding": binding(),
                   "records": [adapter._payload_row(source_row)], "zero_row_probe": False}
        raw_payload = (json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":")) + "\n").encode()
        expected = worker_expectation()
        expected.update({"max_prompt_bytes": 8192, "max_protocol_bytes": 100_000,
                         "assume_exhaustive_labels": False})
        config = {"worker_code_sha256": adapter._sha(Path(adapter.__file__).read_bytes()),
                  "system_prompt": "Choose {labels}. {text}", "model_name": "local:tag",
                  "keep_alive": "1m", "temperature": 0, "num_ctx": 1024,
                  "num_predict": 20, "seed": 1, "top_k": 1, "top_p": 1,
                  "repeat_penalty": 1,
                  "remote_uid": 42, "endpoint_url": "http://127.0.0.1:11434/api/chat",
                  "model_store": "/tmp/models", "server_binary_path": "/opt/ollama",
                  "server_version": "test-version", "worker_start_timeout_seconds": 3,
                  "max_payload_bytes": 100_000, "max_response_bytes": 100_000,
                  "teacher_binding": expected}

        class Server:
            pid = 456
            returncode = None
            def __init__(self, *_args, **_kwargs):
                self.returncode = None
            def poll(self):
                return self.returncode
            def wait(self, timeout=None):
                self.returncode = -15
                return self.returncode

        stdout = io.StringIO()
        boundary = {"boundary": "same_uid_private_network_namespace_without_nonloopback_routes",
                    "network_namespace_inode": 123, "server_pid": 456, "server_binary_sha256": "8" * 64,
                    "server_version": "test-version"}
        artifact = {"model_layer_bytes": 100}
        model_identity = {"reported_manifest_sha256": "9" * 64}
        lifecycle = []

        class FakeInput:
            def __init__(self):
                self.reads = 0
            def readline(self, limit=-1):
                self.reads += 1
                last = json.loads(stdout.getvalue().splitlines()[-1])
                if self.reads == 1:
                    self.assert_phase(last, "control_ready")
                    return (json.dumps({"phase": "warmup_ack", "control_nonce": last["control_nonce"],
                        "network_namespace_inode": last["network_namespace_inode"], "port": last["port"]}) + "\n").encode()
                self.assert_phase(last, "ready_for_data")
                return raw_payload
            @staticmethod
            def assert_phase(event, phase):
                if event.get("phase") != phase:
                    raise AssertionError(f"expected {phase} before stdin read, got {event.get('phase')}")

        fake_stdin = SimpleNamespace(buffer=FakeInput())
        def warmup(_cfg):
            lifecycle.append("warmup")
            return {"status": "completed"}
        def loaded(_cfg, _artifact):
            self.assertEqual(lifecycle, ["warmup"])
            lifecycle.append("verify_loaded_model")
            return model_identity
        with mock.patch.object(adapter.comparator, "load_config", return_value=(config, "f" * 64)), \
             mock.patch.object(adapter.comparator, "verify_acquisition", return_value=({}, "e" * 64)), \
             mock.patch.object(adapter.sys, "platform", "linux"), \
             mock.patch.object(adapter.comparator, "_route_is_private", return_value=True), \
             mock.patch.object(adapter.os, "geteuid", return_value=42, create=True), \
             mock.patch.object(adapter.comparator, "resource_snapshot", return_value={"cpu": 1}), \
             mock.patch.object(adapter.comparator, "verify_installed_artifact", return_value=artifact), \
             mock.patch.object(adapter.comparator, "_http_call", return_value=(200, b'{"version":"test-version"}')), \
             mock.patch.object(adapter.time, "sleep"), \
             mock.patch.object(adapter.comparator, "verify_loaded_model", side_effect=loaded), \
             mock.patch.object(adapter.comparator, "verify_runtime_boundary", side_effect=lambda *_: boundary), \
             mock.patch.object(adapter.subprocess, "Popen", side_effect=Server), \
             mock.patch.object(adapter.comparator, "_warmup", side_effect=warmup), \
             mock.patch.object(adapter.comparator, "_prediction_http", return_value=("answered", ["intents:build.software"], 3.0)), \
             mock.patch.object(adapter.comparator, "_descendant_pids", return_value=[]), \
             mock.patch.object(adapter.os, "killpg", create=True), \
             mock.patch.object(adapter.sys, "stdin", fake_stdin), \
             mock.patch.object(adapter.sys, "stdout", stdout):
            adapter._worker(Path("/private/worker-config.json"), "f" * 64)
            success_lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual([line["phase"] for line in success_lines],
                         ["control_ready", "ready_for_data", "progress", "result"])
        self.assertEqual(success_lines[2]["result"]["proposal_states"]["intents:build.software"], "proposed_positive")
        self.assertTrue(success_lines[3]["cleanup_verified"])
        self.assertEqual(success_lines[1]["model_layer_bytes"], artifact["model_layer_bytes"])
        self.assertEqual(lifecycle, ["warmup", "verify_loaded_model"])
        self.assertNotIn("PRIVATE_SENTINEL", stdout.getvalue())


class CoordinatorEntryTests(unittest.TestCase):
    def cfg_and_comp(self, root: Path):
        cfg = {"schema_version": adapter.SCHEMA, "revision": "test-v1",
               "export_dir": str(root / "private-export"), "export_manifest_sha256": "a" * 64,
               "comparison_config": str(root / "comparison.json"), "output_dir": str(root / "output"),
               "selection": {"row_ids": []}, "max_rows": 2, "assume_exhaustive_labels": False,
               "max_output_bytes": 100_000, "max_prompt_bytes": 8192, "max_protocol_bytes": 1_000_000,
               "max_export_bytes": 100_000, "export_config_sha256": "b" * 64,
               "export_train_sha256": "c" * 64, "taxonomy_sha256": "d" * 64,
               "selected_columns": row()["columns"], "screening_policy_sha256": "3" * 64,
               "screening_code_sha256": "4" * 64}
        comp = {"max_manifest_bytes": 100_000}
        (root / "comparison.json").write_text("{}", encoding="utf-8")
        return cfg, comp

    def test_actual_run_requires_zero_probe_then_enters_actual_worker(self):
        r = row()
        manifest = {"selected_columns": r["columns"]}
        probe = {"zero_row_probe": True, "host_before": {"hostname": "host"}}
        with tempfile.TemporaryDirectory() as tmp:
            cfg, comp = self.cfg_and_comp(Path(tmp))
            calls = []
            def worker(*args, **kwargs):
                calls.append((args, kwargs))
                return probe if len(calls) == 1 else {"actual": True}
            with mock.patch.object(adapter, "load_config", return_value=(cfg, "e" * 64)), \
                 mock.patch.object(adapter.comparator, "load_config", return_value=(comp, "f" * 64)), \
                 mock.patch.object(adapter, "_private_file", return_value=True), \
                 mock.patch.object(adapter.comparator, "verify_acquisition", return_value=({}, "9" * 64)), \
                 mock.patch.object(adapter, "_preflight_export", return_value=(manifest, [r], {"train_sha256": "c" * 64})), \
                 mock.patch.object(adapter, "_check_result_bounds"), \
                 mock.patch.object(adapter, "_remote_identity_snapshot", return_value={"hostname": "host"}), \
                 mock.patch.object(adapter, "_run_transient_worker", side_effect=worker):
                result = adapter.run(Path(tmp) / "config.json")
            self.assertEqual(result, {"actual": True})
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[0][0][9])
            self.assertFalse(calls[1][0][9])
            self.assertEqual(calls[1][1]["expected_probe"], probe)

    def test_host_hostname_and_architecture_checked_before_any_payload_work(self):
        comp = {"remote_expected_hostname": "expected", "remote_expected_architecture": "x86",
                "min_memory_available_bytes": 1, "min_disk_free_bytes": 1}
        snapshot = {"hostname": "unexpected", "architecture": "x86", "memory_available_bytes": 10,
                    "home_disk_free_bytes": 10, "loaded_model_count": 0}
        with mock.patch.object(adapter.comparator, "_remote_snapshot", return_value=snapshot), \
             mock.patch.object(adapter.comparator, "_remote_port_closed", side_effect=AssertionError("too late")):
            with self.assertRaises(adapter.ProposalError):
                adapter._run_transient_worker({}, "x", comp, "y", {}, "z", None, None, None, True)

    def test_zero_probe_entry_does_not_preflight_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, comp = self.cfg_and_comp(Path(tmp))
            sentinel = {"zero_row_probe": True}
            with mock.patch.object(adapter, "load_config", return_value=(cfg, "e" * 64)), \
                 mock.patch.object(adapter.comparator, "load_config", return_value=(comp, "f" * 64)), \
                 mock.patch.object(adapter, "_private_file", return_value=True), \
                 mock.patch.object(adapter.comparator, "verify_acquisition", return_value=({}, "9" * 64)), \
                 mock.patch.object(adapter, "_remote_identity_snapshot", return_value={"hostname": "host"}), \
                 mock.patch.object(adapter, "_preflight_export", side_effect=AssertionError("train export opened")), \
                 mock.patch.object(adapter, "_run_transient_worker", return_value=sentinel) as worker:
                result = adapter.run(Path(tmp) / "config.json", zero_rows=True)
            self.assertEqual(result, sentinel)
            self.assertTrue(worker.call_args.args[9])

    def test_main_emits_only_fixed_sanitized_diagnostic_for_controller_failure(self):
        stderr = io.StringIO()
        sentinel = "PRIVATE_SENTINEL C:\\private\\prompt --token=secret"
        with mock.patch.object(sys, "argv", ["train_teacher_proposals.py", "--config", sentinel,
                                               "--zero-row-probe"]), \
             mock.patch.object(adapter, "load_config", side_effect=adapter.ProposalError(sentinel)), \
             contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                adapter.main()
        self.assertIn("Teacher proposal task failed", str(raised.exception))
        lines = stderr.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), {
            "schema_version": adapter.DIAGNOSTIC_SCHEMA,
            "phase": "config_validation",
            "category": "validation_or_protocol_failure",
            "cleanup": "not_started",
        })
        self.assertNotIn(sentinel, stderr.getvalue())

    def test_main_distinguishes_remote_directory_and_local_temp_config_failures(self):
        comp = {"max_manifest_bytes": 100_000, "min_memory_available_bytes": 1,
            "min_disk_free_bytes": 1, "resource_memory_max_bytes": 1,
            "resource_cpu_quota_percent": 100, "remote_scope_tasks_max": 1,
            "remote_sudo_path": "/usr/bin/sudo", "remote_systemd_run_path": "/usr/bin/systemd-run",
            "remote_true_path": "/usr/bin/true", "remote_home": "/home/owner",
            "remote_workdir": "/home/owner/work", "ssh_port": 22, "remote_user": "owner",
            "remote_host": "worker.test", "remote_expected_hostname": "worker",
            "remote_expected_architecture": "x86_64"}
        snapshot = {"memory_available_bytes": 10, "home_disk_free_bytes": 10,
            "loaded_model_count": 0, "hostname": "worker", "architecture": "x86_64"}
        paths = {"workdir": "/home/owner/work/run-x", "teacher": "/home/owner/work/run-x/a.py",
            "comparator": "/home/owner/work/run-x/b.py", "config": "/home/owner/work/run-x/c.json"}
        sentinel = "PRIVATE_SENTINEL C:\\private\\prompt --token=secret"
        real_private_temp_directory = adapter._private_temp_directory

        for failure, expected_phase in (("remote", "remote_task_directory"),
                                        ("temp", "local_temp_directory"),
                                        ("config", "local_config_file")):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                cfg, _ = self.cfg_and_comp(Path(tmp))
                stderr = io.StringIO()
                def create_remote(*_args):
                    if failure == "remote":
                        raise adapter.ProposalError(sentinel)
                def create_local_temp(*_args):
                    if failure == "temp":
                        raise adapter.ProposalError(sentinel)
                    return real_private_temp_directory("lfg-teacher-diagnostic-test")
                def write_config(*_args):
                    if failure == "config":
                        raise adapter.ProposalError(sentinel)
                with mock.patch.object(sys, "argv", ["train_teacher_proposals.py", "--config",
                        str(Path(tmp) / "private-config.json"), "--zero-row-probe"]), \
                     mock.patch.object(adapter, "load_config", return_value=(cfg, "e" * 64)), \
                     mock.patch.object(adapter.comparator, "load_config", return_value=(comp, "f" * 64)), \
                     mock.patch.object(adapter, "_private_file", return_value=True), \
                     mock.patch.object(adapter.comparator, "verify_acquisition", return_value=({}, "9" * 64)), \
                     mock.patch.object(adapter, "_remote_identity_snapshot", return_value=snapshot), \
                     mock.patch.object(adapter.comparator, "_remote_port_closed", return_value=True), \
                     mock.patch.object(adapter.comparator, "_ssh_run", return_value=SimpleNamespace(returncode=0)), \
                     mock.patch.object(adapter, "_remote_stage_paths", return_value=paths), \
                     mock.patch.object(adapter, "_worker_config", return_value={}), \
                     mock.patch.object(adapter, "_remote_create_workdir", side_effect=create_remote), \
                     mock.patch.object(adapter, "_private_temp_directory", side_effect=create_local_temp), \
                     mock.patch.object(adapter, "_write_new", side_effect=write_config), \
                     mock.patch.object(adapter, "_remote_cleanup_worker"), \
                     contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit):
                        adapter.main()
                records = [json.loads(line) for line in stderr.getvalue().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["phase"], expected_phase)
                self.assertEqual(records[0]["cleanup"], "succeeded")
                self.assertNotIn(sentinel, stderr.getvalue())

    def test_worker_failure_reports_fixed_worker_phase_to_controller_protocol(self):
        stderr, stdout = io.StringIO(), io.StringIO()
        sentinel = "PRIVATE_SENTINEL /private/config prompt"
        with mock.patch.object(sys, "argv", ["train_teacher_proposals.py", "--worker", "--config",
                                               "/private/config", "--config-sha256", "a" * 64]), \
             mock.patch.object(adapter.comparator, "load_config", side_effect=adapter.ProposalError(sentinel)), \
             contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit):
                adapter.main()
        diagnostic = json.loads(stdout.getvalue().splitlines()[0])
        self.assertEqual(diagnostic["phase"], "worker_diagnostic")
        self.assertEqual(diagnostic["worker_phase"], "worker_config_validation")
        self.assertNotIn(sentinel, stdout.getvalue() + stderr.getvalue())

    def test_cleanup_failure_does_not_replace_primary_controller_error(self):
        primary = adapter.ProposalError("primary private sentinel")
        comp = {"min_memory_available_bytes": 1, "min_disk_free_bytes": 1,
                "resource_memory_max_bytes": 1, "resource_cpu_quota_percent": 100,
                "remote_scope_tasks_max": 1, "remote_sudo_path": "/usr/bin/sudo",
                "remote_systemd_run_path": "/usr/bin/systemd-run", "remote_true_path": "/usr/bin/true",
                "remote_home": "/home/owner", "remote_workdir": "/home/owner/work", "ssh_port": 22,
                "remote_user": "owner", "remote_host": "worker.test"}
        snapshot = {"memory_available_bytes": 10, "home_disk_free_bytes": 10,
                    "loaded_model_count": 0, "hostname": "worker", "architecture": "x86_64"}
        cfg = {"export_manifest_sha256": "a" * 64, "export_train_sha256": "b" * 64,
               "taxonomy_sha256": "c" * 64, "export_config_sha256": "d" * 64,
               "selected_columns": [], "max_rows": 1, "screening_policy_sha256": "e" * 64,
               "screening_code_sha256": "f" * 64, "max_prompt_bytes": 10,
               "max_output_bytes": 10, "max_protocol_bytes": 100,
               "assume_exhaustive_labels": False}
        paths = {"workdir": "/home/owner/work/run-x", "teacher": "/home/owner/work/run-x/a.py",
                 "comparator": "/home/owner/work/run-x/b.py", "config": "/home/owner/work/run-x/c.json"}
        with mock.patch.object(adapter, "_remote_identity_snapshot", return_value=snapshot), \
             mock.patch.object(adapter.comparator, "_remote_port_closed", return_value=True), \
             mock.patch.object(adapter.comparator, "_ssh_run", return_value=SimpleNamespace(returncode=0)), \
             mock.patch.object(adapter, "_remote_stage_paths", return_value=paths), \
             mock.patch.object(adapter, "_worker_config", return_value={}), \
             mock.patch.object(adapter, "_remote_create_workdir"), \
             mock.patch.object(adapter, "_private_temp_directory", side_effect=primary), \
             mock.patch.object(adapter, "_remote_cleanup_worker", side_effect=OSError("cleanup secret")):
            with self.assertRaises(adapter.ProposalError) as raised:
                adapter._run_transient_worker(cfg, "x", comp, "y", {}, "z", None, None, None, True)
        self.assertIs(raised.exception, primary)
        self.assertEqual(adapter._DIAGNOSTIC_CLEANUP, "failed")

    def test_worker_diagnostic_protocol_accepts_only_allowlisted_stage_values(self):
        safe = {"phase": "worker_diagnostic", "schema_version": adapter.DIAGNOSTIC_SCHEMA,
                "worker_phase": "worker_warmup", "category": "timeout", "cleanup": "succeeded"}
        lines = __import__("queue").Queue()
        lines.put(json.dumps(safe))
        with self.assertRaises(adapter.WorkerDiagnosticError) as raised:
            adapter._read_worker_line(None, lines, 0)
        record = adapter._diagnostic_record(raised.exception)
        self.assertEqual(record["worker_phase"], "worker_warmup")
        self.assertEqual(record["worker_cleanup"], "succeeded")
        safe["worker_phase"] = "worker_warmup PRIVATE_SENTINEL"
        lines = __import__("queue").Queue()
        lines.put(json.dumps(safe))
        with self.assertRaises(adapter.ProposalError):
            adapter._read_worker_line(None, lines, 0)


if __name__ == "__main__":
    unittest.main()
