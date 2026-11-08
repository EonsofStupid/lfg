from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
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
             mock.patch.object(adapter.comparator, "_verify_private_report_directory", return_value=True), \
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

    def test_completed_and_tampered_destinations_are_validated_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, final, (_published, _digest) = self._publish(Path(tmp))
            manifest = {"schema_version": adapter.REPORT_SCHEMA, "status": "complete",
                        "promotion_eligible": False}
            with mock.patch.object(adapter, "_private_dir", return_value=True), \
                 mock.patch.object(adapter, "_private_file", return_value=True), \
                 mock.patch.object(adapter, "_outside_git", return_value=True), \
                 mock.patch.object(adapter.comparator, "_verify_private_report_directory", return_value=True):
                before = {p.name: p.read_bytes() for p in final.iterdir()}
                with self.assertRaises(adapter.ProposalError):
                    adapter._atomic_private_publish(final, [{"status": "different"}], manifest)
                self.assertEqual({p.name: p.read_bytes() for p in final.iterdir()}, before)
                (final / "proposals.jsonl").write_bytes(b"tampered\n")
                tampered = {p.name: p.read_bytes() for p in final.iterdir()}
                with self.assertRaises(adapter.ProposalError):
                    adapter._atomic_private_publish(final, [{"status": "answered"}], manifest)
                self.assertEqual({p.name: p.read_bytes() for p in final.iterdir()}, tampered)

    def test_complete_manifest_claim_count_outcome_and_binding_tampering_fails_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "private-output"
            parent.mkdir()
            final = parent / "run-1"
            manifest = {"schema_version": adapter.REPORT_SCHEMA, "status": "complete",
                "promotion_eligible": False,
                "claims": {"labels_independently_verified": False,
                           "teacher_suggestions_are_unreviewed": True},
                "counts": {"answered_rows": 1, "selected_rows": 1},
                "abort": {"aborted": False, "reason": None},
                "uncertainty": {"teacher_labels_are_claims": True},
                "provenance": {"progress_store_events_sha256": "e" * 64,
                               "terminal_worker_result_verified": True,
                               "selected_row_ids_sha256": "f" * 64}}
            rows = [{"row_id": "synthetic-row", "status": "answered"}]
            mocks = (mock.patch.object(adapter, "_private_dir", return_value=True),
                     mock.patch.object(adapter, "_private_file", return_value=True),
                     mock.patch.object(adapter, "_outside_git", return_value=True),
                     mock.patch.object(adapter.comparator, "_verify_private_report_directory", return_value=True),
                     mock.patch.object(adapter.comparator, "_private_report_directory",
                                       side_effect=lambda path: path.mkdir(mode=0o700) or path))
            with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4]:
                adapter._atomic_private_publish(final, rows, manifest)
                saved = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
                mutations = {
                    "claim": lambda value: value["claims"].__setitem__("labels_independently_verified", True),
                    "count": lambda value: value["counts"].__setitem__("answered_rows", 999),
                    "status": lambda value: value.__setitem__("status", "partial_aborted_ambiguous"),
                    "abort": lambda value: value["abort"].__setitem__("reason", "changed"),
                    "uncertainty": lambda value: value["uncertainty"].__setitem__("teacher_labels_are_claims", False),
                    "terminal_binding": lambda value: value["provenance"].__setitem__("progress_store_events_sha256", "0" * 64),
                    "row_binding": lambda value: value["provenance"].__setitem__("selected_row_ids_sha256", "0" * 64),
                }
                path = final / "manifest.json"
                for label, mutate in mutations.items():
                    with self.subTest(field=label):
                        candidate = json.loads(json.dumps(saved))
                        mutate(candidate)
                        path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                        changed = {p.name: p.read_bytes() for p in final.iterdir()}
                        with self.assertRaises(adapter.ProposalError):
                            adapter._atomic_private_publish(final, rows, manifest)
                        self.assertEqual({p.name: p.read_bytes() for p in final.iterdir()}, changed)
                path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def test_pending_manifest_tampering_fails_closed_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "private-output"
            parent.mkdir()
            final = parent / "run-1"
            rows = [{"row_id": "synthetic-row", "status": "answered"}]
            payload = b'{"row_id":"synthetic-row","status":"answered"}\n'
            manifest = {"schema_version": adapter.REPORT_SCHEMA, "status": "complete",
                "promotion_eligible": False,
                "claims": {"labels_independently_verified": False},
                "counts": {"answered_rows": 1},
                "abort": {"aborted": False, "reason": None},
                "uncertainty": {"teacher_labels_are_claims": True},
                "provenance": {"progress_store_events_sha256": "e" * 64,
                               "selected_row_ids_sha256": "f" * 64,
                               "terminal_worker_result_verified": True}}
            publication_id = adapter._sha(adapter._canonical_json({
                "proposal_sha256": adapter._sha(payload), "progress_events_sha256": "e" * 64,
                "status": "complete"}).rstrip(b"\n"))
            manifest = {**manifest, "publication_id": publication_id,
                "proposals_sha256": adapter._sha(payload), "proposals_bytes": len(payload),
                "rows_written": len(rows)}
            partial = parent / f".{final.name}.partial-{publication_id}"
            partial.mkdir(mode=0o700)
            marker = {"schema_version": adapter.REPORT_SCHEMA, "status": "incomplete",
                      "promotion_eligible": False, "publication_id": publication_id}
            (partial / "manifest.json").write_bytes(adapter._canonical_json(marker))
            (partial / "proposals.jsonl").write_bytes(payload)
            pending = partial / f".manifest-complete-{publication_id}-interrupted.tmp"
            expected_pending = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            with mock.patch.object(adapter, "_private_dir", return_value=True), \
                 mock.patch.object(adapter, "_private_file", return_value=True), \
                 mock.patch.object(adapter, "_outside_git", return_value=True), \
                 mock.patch.object(adapter.comparator, "_verify_private_report_directory", return_value=True):
                expected_input = {k: v for k, v in manifest.items()
                    if k not in {"publication_id", "proposals_sha256", "proposals_bytes", "rows_written"}}
                mutations = {
                    "claim": lambda value: value["claims"].__setitem__("labels_independently_verified", True),
                    "count": lambda value: value["counts"].__setitem__("answered_rows", 999),
                    "status": lambda value: value.__setitem__("status", "partial_aborted_ambiguous"),
                    "abort": lambda value: value["abort"].__setitem__("reason", "changed"),
                    "uncertainty": lambda value: value["uncertainty"].__setitem__("teacher_labels_are_claims", False),
                    "terminal_binding": lambda value: value["provenance"].__setitem__("progress_store_events_sha256", "0" * 64),
                    "row_binding": lambda value: value["provenance"].__setitem__("selected_row_ids_sha256", "0" * 64),
                }
                for label, mutate in mutations.items():
                    with self.subTest(field=label):
                        candidate = json.loads(expected_pending)
                        mutate(candidate)
                        pending.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                        before = {p.name: p.read_bytes() for p in partial.iterdir()}
                        with self.assertRaises(adapter.ProposalError):
                            adapter._atomic_private_publish(final, rows, expected_input)
                        self.assertEqual({p.name: p.read_bytes() for p in partial.iterdir()}, before)

    def test_revision_reservation_publishes_linked_report_without_changing_prior_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masked._secure_private_path(root, is_directory=True)
            masked._verify_owner_private_access(root, is_directory=True)
            parent = root / "private-output"
            parent.mkdir(mode=0o700)
            masked._secure_private_path(parent, is_directory=True)
            base_output = parent / "run-1"
            source = row("a" * 64)
            with mock.patch.object(adapter, "_request", return_value=b"fixture"):
                event = adapter._worker_loop([source], source["columns"], {},
                    lambda *_: ("timeout", None, None), False, 8192)[0][0]
            binding = {"schema_version": adapter.PROGRESS_STORE_SCHEMA,
                       "proposal_config_identity_sha256": "1" * 64,
                       "selected_row_ids": [source["id"]]}
            progress_store = parent / ".run-1.teacher-progress"
            first_rows = [event["result"]]
            first_manifest = {"schema_version": adapter.REPORT_SCHEMA,
                "status": "partial_aborted_ambiguous", "promotion_eligible": False,
                "claims": {"teacher_suggestions_are_unreviewed": True,
                    "heldout_accessed": False, "labels_independently_verified": False},
                "counts": {"selected_rows": 1, "statuses": {"timeout": 1}},
                "provenance": {"progress_store_events_sha256": "e" * 64,
                    "progress_store_id_sha256": adapter._sha(adapter._canonical_json(binding).rstrip(b"\n")),
                    "export_manifest_sha256": "1" * 64,
                    "model_manifest_sha256": "4" * 64,
                    "terminal_worker_result_verified": False}}

            real_private_report_directory = adapter.comparator._private_report_directory
            def private_report_dir(path):
                return real_private_report_directory(path)

            with mock.patch.object(adapter, "_outside_git", return_value=True), \
                 mock.patch.object(adapter, "_private_dir", return_value=True), \
                 mock.patch.object(adapter, "_private_file", return_value=True), \
                 mock.patch.object(adapter.comparator, "_verify_private_report_directory", return_value=True), \
                 mock.patch.object(adapter.comparator, "_private_report_directory", side_effect=private_report_dir):
                adapter._progress_store_create(progress_store, binding, 1_000_000, "2" * 64)
                attempt = "a" * 32
                adapter._progress_store_record_attempt(progress_store, attempt, [0], binding)
                adapter._checkpoint_worker_event(progress_store, attempt, 0, 0, source, event,
                    source["columns"], False, 1_000_000)
                events_sha = adapter._sha((progress_store / "events.jsonl").read_bytes())
                first_manifest["provenance"]["progress_store_events_sha256"] = events_sha
                adapter._progress_store_write_publication_receipt(progress_store, binding,
                    first_rows, first_manifest, 1_000_000)
                adapter._atomic_private_publish(base_output, first_rows, first_manifest)
                old_manifest = (base_output / "manifest.json").read_bytes()
                old_rows = (base_output / "proposals.jsonl").read_bytes()
                original_store = {item.name: item.read_bytes() for item in progress_store.iterdir()}
                for field, mutate in (
                    ("claims", lambda value: value["claims"].__setitem__("heldout_accessed", True)),
                    ("counts", lambda value: value["counts"].__setitem__("selected_rows", 99)),
                    ("provenance", lambda value: value["provenance"].__setitem__("model_manifest_sha256", "f" * 64)),
                    ("source", lambda value: value["provenance"].__setitem__("export_manifest_sha256", "f" * 64)),
                ):
                    candidate = json.loads(old_manifest)
                    mutate(candidate)
                    (base_output / "manifest.json").write_text(
                        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    with self.subTest(parent_field=field), self.assertRaises(adapter.ProposalError):
                        adapter._progress_revision_reserve(progress_store, base_output,
                            binding, 1_000_000)
                    self.assertEqual({item.name: item.read_bytes() for item in progress_store.iterdir()},
                                     original_store)
                    self.assertEqual(list(progress_store.glob("revision-*.json")), [])
                    (base_output / "manifest.json").write_bytes(old_manifest)
                changed_rows = [{**first_rows[0], "status": "answered"}]
                changed_payload = (json.dumps(changed_rows[0], sort_keys=True, ensure_ascii=False,
                    separators=(",", ":")) + "\n").encode("utf-8")
                (base_output / "proposals.jsonl").write_bytes(changed_payload)
                unrelated = json.loads(old_manifest)
                unrelated["proposals_sha256"] = adapter._sha(changed_payload)
                unrelated["proposals_bytes"] = len(changed_payload)
                unrelated["rows_written"] = 1
                unrelated["publication_id"] = adapter._sha(adapter._canonical_json({
                    "proposal_sha256": adapter._sha(changed_payload),
                    "progress_events_sha256": events_sha,
                    "status": unrelated["status"]}).rstrip(b"\n"))
                (base_output / "manifest.json").write_text(
                    json.dumps(unrelated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                with self.assertRaises(adapter.ProposalError):
                    adapter._progress_revision_reserve(progress_store, base_output, binding, 1_000_000)
                self.assertEqual({item.name: item.read_bytes() for item in progress_store.iterdir()},
                                 original_store)
                (base_output / "manifest.json").write_bytes(old_manifest)
                (base_output / "proposals.jsonl").write_bytes(old_rows)
                revision_script = ("import json, os, sys; from pathlib import Path; "
                    "sys.path.insert(0, sys.argv[1]); import train_teacher_proposals as a; "
                    "real = a._write_new; "
                    "a._write_new = lambda path, raw: os._exit(73) if Path(path).name == "
                    "'revision-reservation.json' else real(path, raw); "
                    "a._progress_revision_reserve(Path(sys.argv[2]), Path(sys.argv[3]), "
                    "json.loads(sys.argv[4]), 1000000)")
                child = subprocess.run([sys.executable, "-c", revision_script,
                    str(Path(adapter.__file__).parent), str(progress_store), str(base_output),
                    json.dumps(binding)], cwd=str(Path(adapter.__file__).parent),
                    capture_output=True, text=True, timeout=20)
                self.assertEqual(child.returncode, 73, child.stderr)
                revision = adapter._progress_revision_state(progress_store, base_output, binding)
                self.assertIsNotNone(revision)
                self.assertFalse(revision["published"])
                self.assertFalse((revision["target"] / "revision-reservation.json").exists())
                reservation = revision["target"] / "revision-reservation.json"
                foreign = revision["target"] / "foreign-entry"
                foreign.write_bytes(b"preserve")
                with self.assertRaisesRegex(adapter.ProposalError, "foreign data"):
                    adapter._progress_revision_state(progress_store, base_output, binding)
                foreign.unlink()
                revision = adapter._progress_revision_reserve(progress_store, base_output,
                    binding, 1_000_000)
                saved_reservation = reservation.read_bytes()
                reservation.write_bytes(b"tampered")
                with self.assertRaises(adapter.ProposalError):
                    adapter._progress_revision_state(progress_store, base_output, binding)
                reservation.write_bytes(saved_reservation)
                self.assertIsNotNone(revision)
                self.assertFalse(revision["published"])
                self.assertTrue(revision["target"].is_dir())
                self.assertTrue((revision["target"] / "revision-reservation.json").is_file())
                revised_rows = [{"row_id": "synthetic-row", "status": "answered"}]
                revised_manifest = {"schema_version": adapter.REPORT_SCHEMA,
                    "status": "complete", "promotion_eligible": False,
                    "provenance": {"progress_store_events_sha256": "f" * 64,
                        "terminal_worker_result_verified": True,
                        "publication_revision": revision["reference"]}}
                published, _digest = adapter._atomic_private_publish(revision["target"],
                    revised_rows, revised_manifest)
                saved = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["provenance"]["publication_revision"], revision["reference"])
                self.assertEqual((base_output / "manifest.json").read_bytes(), old_manifest)
                self.assertEqual((base_output / "proposals.jsonl").read_bytes(), old_rows)
                once = {p.name: p.read_bytes() for p in published.iterdir()}
                again, _same_digest = adapter._atomic_private_publish(revision["target"],
                    revised_rows, revised_manifest)
                self.assertEqual(again, published)
                self.assertEqual({p.name: p.read_bytes() for p in published.iterdir()}, once)
                state = adapter._progress_revision_state(progress_store, base_output, binding)
                self.assertTrue(state["published"])
                self.assertEqual(state["target"], published)

    def test_terminal_partial_parent_is_authenticated_against_its_historical_event_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masked._secure_private_path(root, is_directory=True)
            masked._verify_owner_private_access(root, is_directory=True)
            parent = root / "private-output"
            parent.mkdir(mode=0o700)
            masked._secure_private_path(parent, is_directory=True)
            base_output = parent / "run-1"
            source = row("c" * 64)
            with mock.patch.object(adapter, "_request", return_value=b"fixture"):
                timeout_event = adapter._worker_loop([source], source["columns"], {},
                    lambda *_: ("timeout", None, None), False, 8192)[0][0]
                answered_event = adapter._worker_loop([source], source["columns"], {},
                    lambda *_: ("answered", ["intents:build.software"], 1.0), False, 8192)[0][0]
            binding = {"schema_version": adapter.PROGRESS_STORE_SCHEMA,
                "selected_row_ids": [source["id"]], "proposal_config_identity_sha256": "1" * 64}
            store = parent / ".run-1.teacher-progress"
            adapter._progress_store_create(store, binding, 1_000_000, "2" * 64)
            attempt = "c" * 32
            adapter._progress_store_record_attempt(store, attempt, [0], binding)
            adapter._checkpoint_worker_event(store, attempt, 0, 0, source, timeout_event,
                source["columns"], False, 1_000_000)
            events_sha = adapter._sha((store / "events.jsonl").read_bytes())
            terminal_report = {"schema_version": adapter.REPORT_SCHEMA,
                "status": "partial_aborted_ambiguous", "promotion_eligible": False,
                "provenance": {"progress_store_events_sha256": events_sha,
                    "progress_store_id_sha256": adapter._sha(adapter._canonical_json(binding).rstrip(b"\n")),
                    "terminal_worker_result_verified": True}}
            adapter._progress_store_write_terminal(store, attempt, binding, events_sha,
                terminal_report, 1_000_000)
            with mock.patch.object(adapter, "_outside_git", return_value=True), \
                 mock.patch.object(adapter, "_private_dir", return_value=True), \
                 mock.patch.object(adapter, "_private_file", return_value=True), \
                 mock.patch.object(adapter.comparator, "_verify_private_report_directory", return_value=True), \
                 mock.patch.object(adapter.comparator, "_private_report_directory",
                                   side_effect=lambda path: path.mkdir(mode=0o700) or path):
                adapter._atomic_private_publish(base_output, [timeout_event["result"]], terminal_report)
                # Append a later retry event; the parent remains bound to the original terminal prefix.
                retry = "d" * 32
                adapter._progress_store_record_attempt(store, retry, [0], binding)
                adapter._checkpoint_worker_event(store, retry, 0, 0, source, answered_event,
                    source["columns"], False, 1_000_000)
                revision = adapter._progress_revision_reserve(store, base_output, binding, 1_000_000)
            self.assertEqual(revision["reference"]["parent_output_name"], base_output.name)
            self.assertEqual(revision["reference"]["parent_manifest_sha256"],
                adapter._sha((base_output / "manifest.json").read_bytes()))
            self.assertNotEqual(adapter._sha((store / "events.jsonl").read_bytes()), events_sha)

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
                     mock.patch.object(adapter.comparator, "_verify_private_report_directory", return_value=True), \
                     mock.patch.object(adapter.comparator, "_private_report_directory",
                                       side_effect=lambda path: path.mkdir(mode=0o700) or path), \
                     mock.patch.object(adapter, "_write_new", side_effect=interrupted_write):
                    with self.assertRaises(OSError):
                        adapter._atomic_private_publish(final, [{"status": "answered"}],
                            {"schema_version": adapter.REPORT_SCHEMA, "status": "complete"})
                dirs = list(parent.iterdir())
                self.assertEqual(len(dirs), 1)
                marker = dirs[0] / "manifest.json"
                if marker.exists():
                    self.assertEqual(json.loads(marker.read_text())["status"], "incomplete")
                self.assertFalse((dirs[0] / ".manifest-complete.tmp").exists())

    def test_publication_recovers_after_directory_rename_before_manifest_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "private-output"
            parent.mkdir()
            final = parent / "run-1"
            manifest = {"schema_version": adapter.REPORT_SCHEMA, "status": "complete",
                        "promotion_eligible": False}
            real_replace = adapter.os.replace
            def interrupted_replace(src, dst):
                if Path(dst).name == "manifest.json" and Path(src).name.startswith(".manifest-complete-"):
                    raise OSError("simulated interruption after rename")
                real_replace(src, dst)
            with mock.patch.object(adapter, "_private_dir", return_value=True), \
                 mock.patch.object(adapter, "_private_file", return_value=True), \
                 mock.patch.object(adapter, "_outside_git", return_value=True), \
                 mock.patch.object(adapter.comparator, "_verify_private_report_directory", return_value=True), \
                 mock.patch.object(adapter.comparator, "_private_report_directory",
                                   side_effect=lambda path: path.mkdir(mode=0o700) or path), \
                 mock.patch.object(adapter.os, "replace", side_effect=interrupted_replace):
                with self.assertRaises(OSError):
                    adapter._atomic_private_publish(final, [{"status": "answered"}], manifest)
            self.assertEqual(json.loads((final / "manifest.json").read_text())["status"], "incomplete")
            self.assertFalse((final / ".manifest-complete.tmp").exists())
            with mock.patch.object(adapter, "_private_dir", return_value=True), \
                 mock.patch.object(adapter, "_private_file", return_value=True), \
                 mock.patch.object(adapter, "_outside_git", return_value=True), \
                 mock.patch.object(adapter.comparator, "_verify_private_report_directory", return_value=True):
                path, _digest = adapter._atomic_private_publish(final, [{"status": "answered"}], manifest)
            self.assertEqual(path, final)
            self.assertEqual(json.loads((final / "manifest.json").read_text())["status"], "complete")
            self.assertFalse((final / ".manifest-complete.tmp").exists())

    def test_subprocess_death_at_publication_boundaries_recovers_only_matching_bytes(self):
        script = r'''import os,sys
from pathlib import Path
import train_teacher_proposals as a
parent=Path(sys.argv[1]); final=parent/'run'; stage=sys.argv[2]
manifest={'schema_version':a.REPORT_SCHEMA,'status':'complete','promotion_eligible':False,
          'created_at_epoch_seconds':1,
          'provenance':{'progress_store_events_sha256':'e'*64,'export_manifest_sha256':'f'*64}}
rows=[{'status':'answered'}]
if stage in ('marker','proposal_temp','during_marker','during_proposal_temp'):
 real=a._write_new
 def write(path,raw):
  is_marker=path.name.startswith('.manifest-incomplete-')
  is_proposal=path.name.startswith('.proposals-')
  if ((stage=='during_marker' and is_marker) or (stage=='during_proposal_temp' and is_proposal)):
   real(path,raw[:max(1,len(raw)//2)]); os._exit(71)
  real(path,raw)
  if ((stage=='marker' and is_marker) or (stage=='proposal_temp' and is_proposal)): os._exit(71)
 a._write_new=write
elif stage=='rename':
 real=os.rename
 def rename(src,dst):
  real(src,dst)
  if Path(dst)==final: os._exit(71)
 os.rename=rename
elif stage=='replace':
 real=os.replace
 def replace(src,dst):
  real(src,dst)
  if Path(dst)==final/'manifest.json' and Path(src).name.startswith('.manifest-complete-'):
   os._exit(71)
 os.replace=replace
a._atomic_private_publish(final,rows,manifest)
raise SystemExit(72)
'''
        for stage in ("during_marker", "marker", "during_proposal_temp", "proposal_temp", "rename", "replace"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp)
                masked._secure_private_path(parent, is_directory=True)
                masked._verify_owner_private_access(parent, is_directory=True)
                result = subprocess.run([sys.executable, "-c", script, str(parent), stage],
                    cwd=str(Path(adapter.__file__).parent), capture_output=True, text=True, timeout=40)
                self.assertEqual(result.returncode, 71, result.stderr)
                final = parent / "run"
                resume_manifest = {"schema_version": adapter.REPORT_SCHEMA, "status": "complete",
                    "promotion_eligible": False, "created_at_epoch_seconds": 1,
                    "provenance": {"progress_store_events_sha256": "e" * 64,
                                   "export_manifest_sha256": "f" * 64}}
                path, digest = adapter._atomic_private_publish(final,
                    [{"status": "answered"}],
                    resume_manifest)
                self.assertEqual(path, final)
                saved = json.loads((final / "manifest.json").read_text())
                self.assertEqual(saved["publication_id"], adapter._sha(adapter._canonical_json({
                    "proposal_sha256": adapter._sha(b'{"status":"answered"}\n'),
                    "progress_events_sha256": "e" * 64, "status": "complete"}).rstrip(b"\n")))
                if stage in ("rename", "replace"):
                    self.assertEqual(saved["created_at_epoch_seconds"], 1)
                    self.assertNotIn("pre_data_zero_row_probe", saved)
                self.assertEqual((final / "proposals.jsonl").read_bytes(), b'{"status":"answered"}\n')
                retained_once = {p.name: p.read_bytes() for p in final.iterdir()}
                path, second_digest = adapter._atomic_private_publish(final,
                    [{"status": "answered"}], resume_manifest)
                self.assertEqual(path, final)
                self.assertEqual(second_digest, digest)
                self.assertEqual({p.name: p.read_bytes() for p in final.iterdir()}, retained_once)


class OutputLeaseTests(unittest.TestCase):
    def test_kernel_lease_rejects_concurrent_owner_and_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "out"
            with mock.patch.object(adapter, "_private_file", return_value=True), \
                 mock.patch.object(adapter, "_private_dir", return_value=True):
                with adapter._output_lease(final):
                    with self.assertRaisesRegex(adapter.ProposalError, "another teacher run"):
                        with adapter._output_lease(final):
                            pass
                with adapter._output_lease(final):
                    self.assertTrue((Path(tmp) / ".out.teacher-progress.lock").exists())

    def test_native_child_process_cannot_take_held_output_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "out"
            script = ("import sys\nfrom pathlib import Path\nfrom unittest import mock\n"
                      "import train_teacher_proposals as a\n"
                      "mock.patch.object(a, '_private_file', return_value=True).start()\n"
                      "mock.patch.object(a, '_private_dir', return_value=True).start()\n"
                      "try:\n    with a._output_lease(Path(sys.argv[1])): raise SystemExit(3)\n"
                      "except a.ProposalError as e:\n    raise SystemExit(0 if 'another teacher run' in str(e) else 4)\n")
            with mock.patch.object(adapter, "_private_file", return_value=True), \
                 mock.patch.object(adapter, "_private_dir", return_value=True):
                with adapter._output_lease(final):
                    for _ in range(2):
                        result = subprocess.run([sys.executable, "-c", script, str(final)],
                            cwd=str(Path(adapter.__file__).parent), capture_output=True, text=True, timeout=15)
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fresh_native_private_parent_secures_lease_before_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            masked._secure_private_path(parent, is_directory=True)
            masked._verify_owner_private_access(parent, is_directory=True)
            final = parent / "out"
            with adapter._output_lease(final):
                lock_path = parent / ".out.teacher-progress.lock"
                self.assertTrue(adapter._private_file(lock_path))
                self.assertEqual(lock_path.stat().st_size, 1)

    def test_native_owner_process_crash_releases_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            masked._secure_private_path(parent, is_directory=True)
            masked._verify_owner_private_access(parent, is_directory=True)
            final = parent / "out"
            script = ("import os,sys\nfrom pathlib import Path\n"
                      "import train_teacher_proposals as a\n"
                      "with a._output_lease(Path(sys.argv[1])): os._exit(0)\n")
            result = subprocess.run([sys.executable, "-c", script, str(final)],
                cwd=str(Path(adapter.__file__).parent), capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            with adapter._output_lease(final):
                self.assertTrue(adapter._private_file(parent / ".out.teacher-progress.lock"))


class DurableProgressTests(unittest.TestCase):
    def _event(self, status="answered"):
        source = row()
        positives = ["intents:build.software"] if status == "answered" else None
        elapsed = 1.0 if status in {"answered", "abstain", "malformed", "invalid_schema", "refusal", "truncated"} else None
        with mock.patch.object(adapter, "_request", return_value=b"fixture"):
            events, _results, _aborted, _reason = adapter._worker_loop(
                [source], source["columns"], {}, lambda _request, _labels: (status, positives, elapsed),
                False, 8192)
        return source, events[0]

    def _store(self, root: Path, source: dict, attempt="a" * 32):
        masked._secure_private_path(root, is_directory=True)
        masked._verify_owner_private_access(root, is_directory=True)
        store = root / ".output.teacher-progress"
        binding = {"schema_version": adapter.PROGRESS_STORE_SCHEMA,
                   "selected_row_ids": [source["id"]],
                   "selected_row_order_sha256": adapter._sha(adapter._canonical_json([source["id"]])),
                   "selected_source_identities_sha256": "1" * 64,
                   "proposal_config_sha256": "2" * 64,
                   "model_manifest_sha256": "3" * 64,
                   "prompt_contract_sha256": "4" * 64,
                   "server_binary_sha256": "5" * 64,
                   "request_policy_sha256": "6" * 64}
        cfg = {"output_dir": str(root / "output"),
               "max_protocol_bytes": 1_000_000, "max_progress_store_bytes": 100_000_000,
               "selected_columns": source["columns"],
               "assume_exhaustive_labels": False}
        adapter._progress_store_create(store, binding)
        adapter._progress_store_record_attempt(store, attempt, [0], binding)
        return store, binding, cfg, attempt

    def test_validated_row_event_is_durable_and_resume_reuses_single_ordered_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, event = self._event()
            store, binding, cfg, attempt = self._store(root, source)
            adapter._checkpoint_worker_event(store, attempt, 0, 0, source, event,
                source["columns"], False, 1_000_000)
            records, latest = adapter._progress_store_read(store, binding, [source], cfg)
            self.assertEqual(len(records), 1)
            self.assertEqual(latest, {0: event["result"]})
            self.assertEqual(adapter._progress_pending_indices([source], latest), [])

    def test_ambiguous_and_not_attempted_events_remain_retryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, event = self._event("timeout")
            latest = {0: event["result"]}
            self.assertEqual(adapter._progress_pending_indices([source], latest), [0])
            source, event = self._event("not_attempted_after_abort")
            self.assertEqual(adapter._progress_pending_indices([source], {0: event["result"]}), [0])

    def test_capacity_can_only_increase_and_does_not_change_proposal_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _event = self._event()
            store, _binding, _cfg, _attempt = self._store(Path(tmp), source)
            self.assertEqual(adapter._progress_capacity_records(store, 100_000_000), 100_000_000)
            self.assertEqual(adapter._progress_capacity_records(store, 120_000_000), 120_000_000)
            with self.assertRaisesRegex(adapter.ProposalError, "cannot be decreased"):
                adapter._progress_capacity_records(store, 110_000_000)
            self.assertEqual(adapter._proposal_config_identity_sha({"x": 1, "max_progress_store_bytes": 5}),
                             adapter._proposal_config_identity_sha({"x": 1, "max_progress_store_bytes": 50}))

    def test_mismatched_resume_does_not_mutate_requested_capacity_increase(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _event = self._event()
            store, binding, cfg, _attempt = self._store(Path(tmp), source)
            before = {p.name: p.read_bytes() for p in store.iterdir()}
            cfg["max_progress_store_bytes"] = 200_000_000
            mismatch = {**binding, "proposal_config_sha256": "9" * 64}
            with self.assertRaisesRegex(adapter.ProposalError, "identity or selected row order"):
                adapter._progress_store_read(store, mismatch, [source], cfg)
            after = {p.name: p.read_bytes() for p in store.iterdir()}
            self.assertEqual(after, before)

    def test_worker_event_protocol_bound_includes_checkpoint_envelope(self):
        source = row()
        low, high = 0, adapter.PROGRESS_EVENT_MAX_BYTES
        found = None
        while low <= high:
            size = (low + high) // 2
            candidate = row()
            candidate["source_identity"]["artifact_path"] = "x" * size
            try:
                adapter._check_result_bounds([candidate], candidate["columns"],
                    2 * adapter.PROGRESS_EVENT_MAX_BYTES,
                    adapter.PROGRESS_EVENT_MAX_BYTES)
                low = size + 1
            except adapter.ProposalError as exc:
                if "serialized checkpoint event exceeds stored-record limit" in str(exc):
                    found = candidate
                    high = size - 1
                else:
                    high = size - 1
        self.assertIsNotNone(found, "expected a worker-line-valid but envelope-oversized source")
        with self.assertRaisesRegex(adapter.ProposalError, "stored-record limit"):
            adapter._check_result_bounds([found], found["columns"],
                2 * adapter.PROGRESS_EVENT_MAX_BYTES, adapter.PROGRESS_EVENT_MAX_BYTES)

    def test_progress_ack_is_bound_to_attempt_index_and_exact_durable_event(self):
        event = {"phase": "progress", "index": 2, "result": {"status": "answered"}}
        ack = adapter._progress_ack("a" * 32, 2, event)
        adapter._validate_progress_ack(ack, "a" * 32, 2, event)
        for bad in ({**ack, "index": 3}, {**ack, "attempt_id": "b" * 32},
                    {**ack, "event_sha256": "0" * 64}):
            with self.subTest(bad=bad), self.assertRaises(adapter.ProposalError):
                adapter._validate_progress_ack(bad, "a" * 32, 2, event)

    def test_two_row_worker_waits_for_controller_fsync_and_valid_ack(self):
        rows = [row(), row("1" * 64)]
        attempt = "a" * 32
        for stop_at in ("before_append", "during_append", "after_append_before_ack", "bad_ack", "valid_ack"):
            with self.subTest(stop_at=stop_at), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                masked._secure_private_path(root, is_directory=True)
                masked._verify_owner_private_access(root, is_directory=True)
                store = root / ".output.teacher-progress"
                binding = {"schema_version": adapter.PROGRESS_STORE_SCHEMA,
                    "selected_row_ids": [item["id"] for item in rows],
                    "selected_row_order_sha256": adapter._sha(adapter._canonical_json([item["id"] for item in rows])),
                    "selected_source_identities_sha256": "1" * 64,
                    "proposal_config_sha256": "2" * 64, "model_manifest_sha256": "3" * 64,
                    "prompt_contract_sha256": "4" * 64, "server_binary_sha256": "5" * 64,
                    "request_policy_sha256": "6" * 64}
                adapter._progress_store_create(store, binding)
                adapter._progress_store_record_attempt(store, attempt, [0, 1], binding)
                inferred = []
                def infer(*_args):
                    inferred.append(len(inferred))
                    return "answered", ["intents:build.software"], 1.0
                def controller_ack(event):
                    index = event["index"]
                    if stop_at == "before_append":
                        raise OSError("controller died before append")
                    if stop_at == "during_append":
                        real_write = os.write
                        def partial_write(fd, raw):
                            real_write(fd, raw[:max(1, len(raw) // 2)])
                            raise OSError("controller died during append")
                        with mock.patch.object(adapter.os, "write", side_effect=partial_write):
                            adapter._checkpoint_worker_event(store, attempt, index, index,
                                rows[index], event, rows[index]["columns"], False, 100_000_000)
                        return
                    adapter._checkpoint_worker_event(store, attempt, index, index,
                        rows[index], event, rows[index]["columns"], False, 100_000_000)
                    if stop_at == "after_append_before_ack":
                        raise BrokenPipeError("controller died before ACK")
                    ack = adapter._progress_ack(attempt, index, event)
                    if stop_at == "bad_ack" and index == 0:
                        ack = {**ack, "index": 1}
                    adapter._validate_progress_ack(ack, attempt, index, event)
                with mock.patch.object(adapter, "_request", return_value=b"synthetic"):
                    if stop_at in {"before_append", "during_append", "after_append_before_ack", "bad_ack"}:
                        with self.assertRaises((OSError, BrokenPipeError, adapter.ProposalError)):
                            adapter._worker_loop(rows, rows[0]["columns"], {}, infer, False, 8192,
                                                 progress_emit=controller_ack)
                        self.assertEqual(len(inferred), 1)
                    else:
                        _progress, _outputs, aborted, _reason = adapter._worker_loop(
                            rows, rows[0]["columns"], {}, infer, False, 8192,
                            progress_emit=controller_ack)
                        self.assertFalse(aborted)
                        self.assertEqual(len(inferred), 2)
                raw_events = (store / "events.jsonl").read_bytes()
                if stop_at == "before_append":
                    self.assertEqual(raw_events, b"")
                elif stop_at == "during_append":
                    self.assertTrue(raw_events)
                    self.assertFalse(raw_events.endswith(b"\n"))
                else:
                    self.assertTrue(raw_events.endswith(b"\n"))
                    self.assertEqual(len(raw_events.splitlines()), 1 if stop_at != "valid_ack" else 2)
                    records = [json.loads(line) for line in raw_events.splitlines()]
                    first_record = records[0]
                    self.assertEqual(first_record["index"], 0)
                    latest = {record["index"]: record["event"]["result"] for record in records}
                    self.assertEqual(adapter._progress_pending_indices(rows, latest),
                                     [1] if stop_at != "valid_ack" else [])
                if stop_at in {"before_append", "during_append"}:
                    # A request may have reached the teacher; absent/truncated
                    # persistence is never represented as an exactly-once result.
                    self.assertEqual(adapter._progress_pending_indices(rows, {}), [0, 1])

    def test_resume_replaces_retryable_prior_event_without_duplicate_output_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, uncertain = self._event("timeout")
            store, binding, cfg, first_attempt = self._store(root, source)
            adapter._progress_store_append(store, first_attempt, 0, uncertain, 1_000_000)
            _source, answered = self._event("answered")
            second_attempt = "b" * 32
            adapter._progress_store_record_attempt(store, second_attempt, [0], binding)
            adapter._checkpoint_worker_event(store, second_attempt, 0, 0, source, answered,
                source["columns"], False, 1_000_000)
            _records, latest = adapter._progress_store_read(store, binding, [source], cfg)
            output_rows = [latest[index] for index in range(len([source]))]
            self.assertEqual(len(output_rows), 1)
            self.assertEqual(output_rows[0]["status"], "answered")
            self.assertEqual(adapter._progress_pending_indices([source], latest), [])

    def test_checkpoint_rejects_duplicate_attempt_index_corruption_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, event = self._event()
            store, binding, cfg, attempt = self._store(root, source)
            adapter._checkpoint_worker_event(store, attempt, 0, 0, source, event,
                source["columns"], False, 1_000_000)
            adapter._progress_store_append(store, attempt, 0, event, 1_000_000)
            with self.assertRaisesRegex(adapter.ProposalError, "duplicate attempt indices"):
                adapter._progress_store_read(store, binding, [source], cfg)
        for truncate in (True, False):
            with self.subTest(truncate=truncate), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, event = self._event()
                store, binding, cfg, attempt = self._store(root, source)
                adapter._checkpoint_worker_event(store, attempt, 0, 0, source, event,
                    source["columns"], False, 1_000_000)
                path = store / "events.jsonl"
                raw = path.read_bytes()
                path.write_bytes(raw[:-1] if truncate else raw.replace(b"record_sha256", b"record_sha25", 1))
                with self.assertRaises(adapter.ProposalError):
                    adapter._progress_store_read(store, binding, [source], cfg)

    def test_checkpoint_rejects_changed_selected_order_and_changed_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _event = self._event()
            store, binding, cfg, _attempt = self._store(root, source)
            changed_binding = {**binding, "selected_row_order_sha256": "0" * 64}
            with self.assertRaisesRegex(adapter.ProposalError, "identity or selected row order"):
                adapter._progress_store_read(store, changed_binding, [source], cfg)
            for field in ("selected_source_identities_sha256", "proposal_config_sha256",
                          "model_manifest_sha256", "prompt_contract_sha256",
                          "server_binary_sha256", "request_policy_sha256"):
                with self.subTest(field=field):
                    changed_binding = {**binding, field: "f" * 64}
                    with self.assertRaisesRegex(adapter.ProposalError, "identity or selected row order"):
                        adapter._progress_store_read(store, changed_binding, [source], cfg)

    def test_terminal_record_tracks_durable_event_snapshot_for_publication_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, event = self._event()
            store, binding, cfg, attempt = self._store(root, source)
            adapter._checkpoint_worker_event(store, attempt, 0, 0, source, event,
                source["columns"], False, 1_000_000)
            events_sha = adapter._sha((store / "events.jsonl").read_bytes())
            report = {"schema_version": adapter.REPORT_SCHEMA,
                      "provenance": {"progress_store_events_sha256": events_sha,
                                     "terminal_worker_result_verified": True}}
            adapter._progress_store_write_terminal(store, attempt, binding, events_sha, report)
            terminal = adapter._progress_store_matching_terminal(store, binding, events_sha)
            self.assertEqual(terminal["output_manifest"], report)
            (store / f"terminal-{attempt}.json").write_bytes(b'{"truncated":')
            with self.assertRaises(adapter.ProposalError):
                adapter._progress_store_read(store, binding, [source], cfg)

    def test_publication_recovery_reuses_matching_terminal_without_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masked._secure_private_path(root, is_directory=True)
            masked._verify_owner_private_access(root, is_directory=True)
            source, event = self._event()
            cfg = {"output_dir": str(root / "output"), "taxonomy_sha256": "d" * 64,
                   "selected_columns": source["columns"], "assume_exhaustive_labels": False,
                   "max_protocol_bytes": 1_000_000, "max_output_bytes": 1_000_000,
                   "max_progress_store_bytes": 100_000_000}
            comp = {"model_name": "fixture/model", "server_binary_sha256": "a" * 64,
                    "server_version": "fixture-server", "system_prompt": "{labels}\n{text}",
                    "prompt_revision": "fixture-v1", "timeout_seconds": 1,
                    "max_request_bytes": 1000, "max_response_bytes": 1000,
                    "num_ctx": 256, "num_predict": 64, "temperature": 0.0,
                    "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0,
                    "resource_memory_max_bytes": 1000, "resource_cpu_quota_percent": 50,
                    "remote_scope_tasks_max": 10, "min_memory_available_bytes": 1,
                    "min_disk_free_bytes": 1, "min_gpu_free_bytes": 0,
                    "max_manifest_bytes": 1000}
            source_binding = {"manifest_sha256": "1" * 64, "config_sha256": "2" * 64,
                              "train_sha256": "3" * 64, "train_bytes": 10}
            stable_runtime = {"boundary": "same_uid_private_network_namespace_without_nonloopback_routes",
                "server_binary_sha256": "a" * 64, "server_version": "fixture-server"}
            original_probe = {"zero_row_probe": True, "rows_sent": 0, "train_payload_opened": False,
                "worker_ready": True, "outside_listener_denied": True, "cleanup_verified": True,
                "persistent_service_unchanged": True,
                "runtime_boundary": {**stable_runtime, "server_pid": 11, "network_namespace_inode": 111,
                    "process_tree_rss_bytes": 100, "process_tree_rss_scope":
                    "server and observed descendants; summed RSS can double-count shared pages"},
                "model_identity": {"reported_manifest_sha256": "4" * 64},
                "artifact": {"model_layer_bytes": 100}, "receipt_sha256": "9" * 64,
                "model_layer_bytes": 100, "host_before": {"hostname": "worker", "architecture": "x86_64"},
                "proposal_config_sha256": "5" * 64,
                "proposal_config_identity_sha256": adapter._proposal_config_identity_sha(cfg),
                "comparison_config_sha256": "6" * 64,
                "adapter_code_sha256": adapter._sha(Path(adapter.__file__).read_bytes()),
                "comparator_code_sha256": adapter._sha(Path(adapter.comparator.__file__).read_bytes())}
            probe = {**original_probe, "runtime_boundary": {**stable_runtime, "server_pid": 22,
                "network_namespace_inode": 222, "process_tree_rss_bytes": 120,
                "process_tree_rss_scope": stable_runtime.get("process_tree_rss_scope",
                    "server and observed descendants; summed RSS can double-count shared pages")},
                "host_before": {"hostname": "worker", "architecture": "x86_64"},
                "resources_before": {"available": 9}, "resources_after": {"available": 8}}
            checkpoint_binding = adapter._progress_binding(cfg, "5" * 64, comp, "6" * 64,
                [source], source_binding, adapter._sha(Path(adapter.__file__).read_bytes()),
                adapter._sha(Path(adapter.comparator.__file__).read_bytes()), probe)
            store = adapter._progress_store_path(Path(cfg["output_dir"]))
            adapter._progress_store_create(store, checkpoint_binding,
                cfg["max_progress_store_bytes"], proposal_config_sha256="5" * 64)
            attempt = "a" * 32
            adapter._progress_store_record_attempt(store, attempt, [0], checkpoint_binding)
            adapter._checkpoint_worker_event(store, attempt, 0, 0, source, event,
                source["columns"], False, 1_000_000)
            events_sha = adapter._sha((store / "events.jsonl").read_bytes())
            adapter_sha = adapter._sha(Path(adapter.__file__).read_bytes())
            comparator_sha = adapter._sha(Path(adapter.comparator.__file__).read_bytes())
            selected_ids = [source["id"]]
            labels = [f"{column['field']}:{column['label']}" for column in source["columns"]]
            states = event["result"]["proposal_states"]
            coverage = dict(sorted(Counter(f"{label}:{state}" for label, state in states.items()).items()))
            source_lineage = {"source_artifacts": [], "source_artifacts_sha256": {},
                "source_artifact_inventory_sha256": adapter._sha(b"{}"),
                "screening_policy_sha256": None, "screening_code_sha256": None,
                "taxonomy_sha256": None, "taxonomy_loader_sha256": None,
                "export_config_sha256": None, "workspace_plan_sha256": None,
                "review_ledger_sha256": None, "partial_sidecar_sha256": None,
                "label_guide_sha256": None, "target_policy": None,
                "selected_columns": source["columns"], "counts": None, "promotion_eligible": False}
            report = {"schema_version": adapter.REPORT_SCHEMA, "created_at_epoch_seconds": 1,
                "status": "complete", "promotion_eligible": False,
                "claims": {"teacher_suggestions_are_unreviewed": True, "heldout_accessed": False,
                    "labels_independently_verified": False, "screening_independently_verified": False,
                    "promotion_eligible": False},
                "provenance": {"proposal_config_sha256": "5" * 64,
                    "proposal_config_origin_sha256": "5" * 64,
                    "proposal_config_identity_sha256": adapter._proposal_config_identity_sha(cfg),
                    "publication_revision": None,
                    "comparison_config_sha256": "6" * 64, "adapter_code_sha256": adapter_sha,
                    "comparator_code_sha256": comparator_sha, "worker_config_sha256": "7" * 64,
                    "payload_sha256": "8" * 64, "export_manifest_sha256": "1" * 64,
                    "export_config_sha256": "2" * 64, "train_file_sha256": "3" * 64,
                    "train_file_bytes": 10, "taxonomy_sha256": "d" * 64,
                    "source_lineage": source_lineage, "selected_row_count": 1,
                    "selected_row_ids_sha256": adapter._sha(json.dumps(selected_ids, separators=(",", ":")).encode()),
                    "selected_text_digests_sha256": adapter._sha(json.dumps([source["text_sha256"]], separators=(",", ":")).encode()),
                    "target_mask_sha256": adapter._sha(json.dumps([source["known_mask"]], separators=(",", ":")).encode()),
                    "proposal_schema_sha256": adapter._sha(json.dumps(adapter.comparator.output_schema(labels), sort_keys=True, separators=(",", ":")).encode()),
                    "prompt_contract_sha256": checkpoint_binding["prompt_contract_sha256"],
                    "progress_store_schema": adapter.PROGRESS_STORE_SCHEMA,
                    "progress_store_id_sha256": adapter._sha(adapter._canonical_json(checkpoint_binding).rstrip(b"\n")),
                    "progress_store_events_sha256": events_sha, "progress_resumed": False,
                    "terminal_worker_result_verified": True, "model_name": comp["model_name"],
                    "model_manifest_sha256": checkpoint_binding["model_manifest_sha256"],
                    "model_layer_bytes": probe["model_layer_bytes"], "receipt_sha256": probe["receipt_sha256"],
                    "server_binary_sha256": comp["server_binary_sha256"], "server_version": comp["server_version"],
                    "runtime_boundary": {**stable_runtime, "server_pid": 11,
                        "network_namespace_inode": 111, "process_tree_rss_bytes": 130,
                        "process_tree_rss_scope": "server and observed descendants; summed RSS can double-count shared pages"},
                    "request_policy": {k: comp[k] for k in ("timeout_seconds", "max_request_bytes", "max_response_bytes",
                        "num_ctx", "num_predict", "temperature", "top_k", "top_p", "repeat_penalty")},
                    "resource_policy": {k: comp[k] for k in ("resource_memory_max_bytes", "resource_cpu_quota_percent",
                        "remote_scope_tasks_max", "min_memory_available_bytes", "min_disk_free_bytes", "min_gpu_free_bytes")}},
                "counts": {"selected_rows": 1, "statuses": {"answered": 1}, "answered_rows": 1,
                    "abstentions": 0, "failures": 0, "not_attempted_after_abort": 0,
                    "per_label_and_state": coverage},
                "pre_data_zero_row_probe": original_probe,
                "abort": {"aborted": False, "reason": None},
                "resources": {"before": {}, "after": {}},
                "uncertainty": {"source_and_review_claims_independently_verified": False,
                    "teacher_labels_are_claims": True, "omitted_labels_are_negative": False,
                    "selection_claim": "configured train-only subset; no heldout data opened"}}
            adapter._progress_store_write_terminal(store, attempt, checkpoint_binding,
                                                   events_sha, report)
            cfg["max_progress_store_bytes"] = 110_000_000
            probe = {**probe, "proposal_config_sha256": "7" * 64,
                     "runtime_boundary": {**stable_runtime, "server_pid": 44,
                         "network_namespace_inode": 444, "process_tree_rss_bytes": 140,
                         "process_tree_rss_scope": "server and observed descendants; summed RSS can double-count shared pages"}}
            with mock.patch.object(adapter, "_outside_git", return_value=True), \
                 mock.patch.object(adapter, "_private_dir", return_value=True), \
                 mock.patch.object(adapter, "_reverify_export"), \
                 mock.patch.object(adapter.comparator, "_verify_private_report_directory", return_value=True), \
                 mock.patch.object(adapter.comparator, "_private_report_directory",
                                   side_effect=lambda path: path.mkdir(mode=0o700) or path):
                result = adapter._resume_published_progress(cfg, "7" * 64, comp, "6" * 64,
                    {"selected_columns": source["columns"]}, [source], source_binding, probe)
                published = Path(result["output_directory"])
                before_repeat = (published / "manifest.json").read_bytes()
                saved_report = json.loads(before_repeat)
                original_probe_saved = saved_report["pre_data_zero_row_probe"]
                probe = {**probe, "runtime_boundary": {**stable_runtime, "server_pid": 55,
                    "network_namespace_inode": 555, "process_tree_rss_bytes": 150,
                    "process_tree_rss_scope": "server and observed descendants; summed RSS can double-count shared pages"}}
                repeated = adapter._resume_published_progress(cfg, "7" * 64, comp, "6" * 64,
                    {"selected_columns": source["columns"]}, [source], source_binding, probe)
            self.assertEqual(result["rows_completed"], 1)
            self.assertEqual(repeated["manifest_sha256"], result["manifest_sha256"])
            self.assertEqual((published / "manifest.json").read_bytes(), before_repeat)
            proposals = (published / "proposals.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(proposals), 1)
            saved = json.loads((published / "manifest.json").read_text())
            self.assertEqual(saved["counts"], report["counts"])
            self.assertNotIn("publication_recovered", saved["provenance"])
            self.assertEqual(saved["pre_data_zero_row_probe"], original_probe_saved)
            self.assertEqual(saved["created_at_epoch_seconds"], 1)
            self.assertEqual(saved["provenance"]["proposal_config_sha256"], "5" * 64)
            capacity_records = [json.loads(line) for line in (store / "capacity.jsonl").read_text().splitlines()]
            self.assertEqual([item["capacity_bytes"] for item in capacity_records], [100_000_000, 110_000_000])
            self.assertEqual([item["config_sha256"] for item in capacity_records], ["5" * 64, "7" * 64])
            self.assertEqual(len(list(store.glob("recovery-probe-*.json"))), 2)

    def test_rows_without_terminal_result_publish_only_as_ambiguous_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masked._secure_private_path(root, is_directory=True)
            masked._verify_owner_private_access(root, is_directory=True)
            source, event = self._event()
            cfg = {"output_dir": str(root / "output"), "taxonomy_sha256": "d" * 64,
                   "selected_columns": source["columns"], "assume_exhaustive_labels": False,
                   "max_protocol_bytes": 1_000_000, "max_output_bytes": 1_000_000,
                   "max_progress_store_bytes": 100_000_000}
            comp = {"model_name": "fixture/model", "server_binary_sha256": "a" * 64,
                    "server_version": "fixture-server", "system_prompt": "{labels}\n{text}",
                    "prompt_revision": "fixture-v1", "timeout_seconds": 1,
                    "max_request_bytes": 1000, "max_response_bytes": 1000,
                    "num_ctx": 256, "num_predict": 64, "temperature": 0.0,
                    "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0,
                    "resource_memory_max_bytes": 1000, "resource_cpu_quota_percent": 50,
                    "remote_scope_tasks_max": 10, "min_memory_available_bytes": 1,
                    "min_disk_free_bytes": 1, "min_gpu_free_bytes": 0,
                    "max_manifest_bytes": 1000}
            source_binding = {"manifest_sha256": "1" * 64, "config_sha256": "2" * 64,
                              "train_sha256": "3" * 64, "train_bytes": 10}
            probe = {"zero_row_probe": True,
                     "proposal_config_identity_sha256": adapter._proposal_config_identity_sha(cfg),
                     "model_identity": {"reported_manifest_sha256": "4" * 64},
                     "runtime_boundary": {"boundary": "same_uid_private_network_namespace_without_nonloopback_routes",
                         "server_binary_sha256": "a" * 64, "server_version": "fixture-server",
                         "server_pid": 22, "network_namespace_inode": 222,
                         "process_tree_rss_bytes": 120,
                         "process_tree_rss_scope": "server and observed descendants; summed RSS can double-count shared pages"},
                     "resources_before": {}, "resources_after": {}}
            checkpoint_binding = adapter._progress_binding(cfg, "5" * 64, comp, "6" * 64,
                [source], source_binding, adapter._sha(Path(adapter.__file__).read_bytes()),
                adapter._sha(Path(adapter.comparator.__file__).read_bytes()), probe)
            store = adapter._progress_store_path(Path(cfg["output_dir"]))
            adapter._progress_store_create(store, checkpoint_binding)
            attempt = "a" * 32
            adapter._progress_store_record_attempt(store, attempt, [0], checkpoint_binding)
            adapter._checkpoint_worker_event(store, attempt, 0, 0, source, event,
                source["columns"], False, 1_000_000)
            with mock.patch.object(adapter, "_outside_git", return_value=True), \
                 mock.patch.object(adapter, "_private_dir", return_value=True), \
                 mock.patch.object(adapter, "_reverify_export"), \
                 mock.patch.object(adapter.comparator, "_verify_private_report_directory", return_value=True), \
                 mock.patch.object(adapter.comparator, "_private_report_directory",
                                   side_effect=lambda path: path.mkdir(mode=0o700) or path):
                result = adapter._resume_published_progress(cfg, "5" * 64, comp, "6" * 64,
                    {"selected_columns": source["columns"]}, [source], source_binding, probe)
                saved_before = {p.name: p.read_bytes() for p in Path(result["output_directory"]).iterdir()}
                changed_probe = {**probe, "resources_before": {"observed": "later"},
                                 "resources_after": {"observed": "later"}}
                repeated = adapter._resume_published_progress(cfg, "5" * 64, comp, "6" * 64,
                    {"selected_columns": source["columns"]}, [source], source_binding, changed_probe)
            published = Path(result["output_directory"])
            saved_manifest = json.loads((published / "manifest.json").read_text())
            self.assertEqual(saved_manifest["status"], "partial_aborted_ambiguous")
            self.assertFalse(saved_manifest["provenance"]["terminal_worker_result_verified"])
            self.assertNotIn("pre_data_zero_row_probe", saved_manifest)
            self.assertNotIn("resources", saved_manifest)
            self.assertEqual(repeated["manifest_sha256"], result["manifest_sha256"])
            self.assertEqual({p.name: p.read_bytes() for p in published.iterdir()}, saved_before)

    def test_two_row_controller_recovers_empty_and_prefix_only_events_without_losing_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masked._secure_private_path(root, is_directory=True)
            masked._verify_owner_private_access(root, is_directory=True)
            rows = [row("a" * 64), row("b" * 64)]
            cfg = {"output_dir": str(root / "output"), "taxonomy_sha256": "d" * 64,
                "selected_columns": rows[0]["columns"], "assume_exhaustive_labels": False,
                "max_protocol_bytes": 1_000_000, "max_output_bytes": 1_000_000,
                "max_progress_store_bytes": 100_000_000}
            comp = {"model_name": "fixture/model", "server_binary_sha256": "a" * 64,
                "server_version": "fixture-server", "system_prompt": "{labels}\\n{text}",
                "prompt_revision": "fixture-v1", "timeout_seconds": 1,
                "max_request_bytes": 1000, "max_response_bytes": 1000,
                "num_ctx": 256, "num_predict": 64, "temperature": 0.0,
                "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0,
                "resource_memory_max_bytes": 1000, "resource_cpu_quota_percent": 50,
                "remote_scope_tasks_max": 10, "min_memory_available_bytes": 1,
                "min_disk_free_bytes": 1, "min_gpu_free_bytes": 0,
                "max_manifest_bytes": 1000}
            source_binding = {"manifest_sha256": "1" * 64, "config_sha256": "2" * 64,
                "train_sha256": "3" * 64, "train_bytes": 10}
            stable = {"boundary": "same_uid_private_network_namespace_without_nonloopback_routes",
                "server_binary_sha256": "a" * 64, "server_version": "fixture-server"}
            def probe(pid, netns):
                return {"zero_row_probe": True,
                    "proposal_config_identity_sha256": adapter._proposal_config_identity_sha(cfg),
                    "model_identity": {"reported_manifest_sha256": "4" * 64},
                    "runtime_boundary": {**stable, "server_pid": pid,
                        "network_namespace_inode": netns, "process_tree_rss_bytes": 100,
                        "process_tree_rss_scope": "server and observed descendants; summed RSS can double-count shared pages"},
                    "resources_before": {}, "resources_after": {}}
            original_probe = probe(20, 220)
            checkpoint_binding = adapter._progress_binding(cfg, "5" * 64, comp, "6" * 64,
                rows, source_binding, adapter._sha(Path(adapter.__file__).read_bytes()),
                adapter._sha(Path(adapter.comparator.__file__).read_bytes()), original_probe)
            store = adapter._progress_store_path(Path(cfg["output_dir"]))
            adapter._progress_store_create(store, checkpoint_binding,
                cfg["max_progress_store_bytes"], "5" * 64)
            inferred = []
            def make_answer(source):
                inferred.append(source["id"])
                with mock.patch.object(adapter, "_request", return_value=b"fixture"):
                    events, _rows, _aborted, _reason = adapter._worker_loop([source],
                        source["columns"], {},
                        lambda *_: ("answered", ["intents:build.software"], 1.0),
                        False, 8192)
                return events[0]

            first_attempt = "a" * 32
            with mock.patch.object(adapter, "_outside_git", return_value=True), \
                 mock.patch.object(adapter, "_private_dir", return_value=True), \
                 mock.patch.object(adapter, "_reverify_export"), \
                 mock.patch.object(adapter.comparator, "_verify_private_report_directory", return_value=True), \
                 mock.patch.object(adapter.comparator, "_private_report_directory",
                                   side_effect=lambda path: path.mkdir(mode=0o700) or path):
                # Interruption before the first event must leave both rows pending.
                self.assertIsNone(adapter._resume_published_progress(cfg, "5" * 64, comp, "6" * 64,
                    {"selected_columns": rows[0]["columns"]}, rows, source_binding, probe(21, 221)))
                self.assertEqual(adapter._progress_pending_indices(rows, {}), [0, 1])
                first_event = make_answer(rows[0])
                adapter._progress_store_record_attempt(store, first_attempt, [0], checkpoint_binding)
                adapter._checkpoint_worker_event(store, first_attempt, 0, 0, rows[0], first_event,
                    rows[0]["columns"], False, cfg["max_progress_store_bytes"])
                # A durable prefix with only row zero also resumes without indexing row one.
                self.assertIsNone(adapter._resume_published_progress(cfg, "5" * 64, comp, "6" * 64,
                    {"selected_columns": rows[0]["columns"]}, rows, source_binding, probe(22, 222)))
                _records, latest = adapter._progress_store_read(store, checkpoint_binding, rows, cfg)
                self.assertEqual(adapter._progress_pending_indices(rows, latest), [1])
                inferred.clear()
                second_event = make_answer(rows[1])
                second_attempt = "b" * 32
                adapter._progress_store_record_attempt(store, second_attempt, [1], checkpoint_binding)
                adapter._checkpoint_worker_event(store, second_attempt, 0, 1, rows[1], second_event,
                    rows[1]["columns"], False, cfg["max_progress_store_bytes"])
                result = adapter._resume_published_progress(cfg, "5" * 64, comp, "6" * 64,
                    {"selected_columns": rows[0]["columns"]}, rows, source_binding, probe(23, 223))
            self.assertEqual(inferred, [rows[1]["id"]])
            self.assertEqual(result["rows_completed"], 2)
            self.assertEqual(result["status_counts"], {"answered": 2})
            published = Path(result["output_directory"])
            saved_rows = [json.loads(line) for line in (published / "proposals.jsonl").read_text().splitlines()]
            self.assertEqual([item["row_id"] for item in saved_rows], [source["id"] for source in rows])
            saved = json.loads((published / "manifest.json").read_text())
            self.assertEqual(saved["counts"]["statuses"], {"answered": 2})
            self.assertFalse(saved["provenance"]["terminal_worker_result_verified"])
            attempts = [json.loads(path.read_text()) for path in sorted(store.glob("attempt-*.json"))]
            self.assertEqual([item["indices"] for item in attempts], [[0], [1]])
            events = [json.loads(line) for line in (store / "events.jsonl").read_text().splitlines()]
            self.assertEqual([item["index"] for item in events], [0, 1])

    def test_checkpoint_write_failure_retains_partial_evidence_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, event = self._event()
            store, binding, cfg, attempt = self._store(root, source)
            real_write = adapter.os.write
            calls = 0
            def partial_then_fail(fd, data):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return real_write(fd, data[:12])
                raise OSError("injected checkpoint interruption")
            with mock.patch.object(adapter.os, "write", side_effect=partial_then_fail):
                with self.assertRaises(OSError):
                    adapter._checkpoint_worker_event(store, attempt, 0, 0, source, event,
                        source["columns"], False, 1_000_000)
            self.assertTrue((store / "events.jsonl").exists())
            with self.assertRaises(adapter.ProposalError):
                adapter._progress_store_read(store, binding, [source], cfg)

    def test_rename_and_final_replace_interruptions_leave_incomplete_manifest(self):
        for point in ("rename", "replace"):
            with self.subTest(point=point), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp) / "private-output"
                parent.mkdir()
                final = parent / "run-1"
                if point == "rename":
                    patcher = mock.patch.object(adapter.os, "rename", side_effect=OSError("interrupt"))
                else:
                    real_replace = adapter.os.replace
                    def interrupt_manifest_replace(src, dst):
                        if Path(dst).name == "manifest.json" and Path(src).name.startswith(".manifest-complete-"):
                            raise OSError("interrupt")
                        real_replace(src, dst)
                    patcher = mock.patch.object(adapter.os, "replace", side_effect=interrupt_manifest_replace)
                with mock.patch.object(adapter, "_private_dir", return_value=True), \
                     mock.patch.object(adapter, "_outside_git", return_value=True), \
                     mock.patch.object(adapter.comparator, "_verify_private_report_directory", return_value=True), \
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
                self.assertTrue(list(dirs[0].glob(".manifest-complete-*.tmp")))

    def test_worst_legal_answered_output_is_preflighted_and_actual_publish_bound_is_enforced(self):
        source = row()
        labels = [f"{column['field']}:{column['label']}" for column in source["columns"]]
        with mock.patch.object(adapter, "_request", return_value=b"fixture"):
            _events, result_rows, _aborted, _reason = adapter._worker_loop(
                [source], source["columns"], {},
                lambda *_: ("answered", labels, 1.0), True, 8192)
        actual_payload = (json.dumps(result_rows[0], sort_keys=True, ensure_ascii=False,
            separators=(",", ":")) + "\n").encode("utf-8")
        bound = adapter._check_result_bounds([source], source["columns"], 1_000_000, 1_000_000)
        self.assertGreaterEqual(bound["max_output_bytes"], len(actual_payload))
        with self.assertRaises(adapter.ProposalError):
            adapter._check_result_bounds([source], source["columns"],
                len(actual_payload) - 1, 1_000_000)

        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "private-output"
            parent.mkdir()
            final = parent / "run-1"
            with mock.patch.object(adapter, "_private_dir", return_value=True), \
                 mock.patch.object(adapter, "_outside_git", return_value=True):
                with self.assertRaisesRegex(adapter.ProposalError, "encoded proposals exceed"):
                    adapter._atomic_private_publish(final, result_rows,
                        {"schema_version": adapter.REPORT_SCHEMA, "status": "complete"},
                        len(actual_payload) - 1)
            self.assertFalse(final.exists())
            self.assertEqual(list(parent.iterdir()), [])


class WorkerProtocolTests(unittest.TestCase):
    def test_worker_streams_each_row_event_before_starting_next_inference(self):
        first = row("a" * 64)
        second = row("b" * 64)
        streamed = []
        calls = 0
        def infer(_request, _labels):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.assertEqual(len(streamed), 1)
            return "answered", ["intents:build.software"], float(calls)
        with mock.patch.object(adapter, "_request", return_value=b"fixture"):
            progress, _results, _aborted, _reason = adapter._worker_loop(
                [first, second], first["columns"], {}, infer, False, 8192,
                progress_emit=streamed.append)
        self.assertEqual(streamed, progress)
        self.assertEqual([event["index"] for event in streamed], [0, 1])

    def test_post_worker_source_and_publication_phases_are_fixed_diagnostics(self):
        for phase in ("source_recheck", "publication"):
            with self.subTest(phase=phase):
                adapter._diagnostic_phase(phase)
                record = adapter._diagnostic_record(adapter.ProposalError("private sentinel"))
                self.assertEqual(record["phase"], phase)
                self.assertNotIn("sentinel", json.dumps(record))

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
                   "records": [safe_row], "zero_row_probe": False, "attempt_id": "a" * 32}
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
        source_row_two = row("1" * 64)
        payload = {"schema_version": adapter.PAYLOAD_SCHEMA, "binding": binding(),
                   "records": [adapter._payload_row(source_row), adapter._payload_row(source_row_two)], "zero_row_probe": False,
                   "attempt_id": "a" * 32}
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
                if self.reads == 2:
                    self.assert_phase(last, "ready_for_data")
                    return raw_payload
                self.assert_phase(last, "progress")
                return (json.dumps(adapter._progress_ack("a" * 32, last["index"], last)) + "\n").encode()
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
                         ["control_ready", "ready_for_data", "progress", "progress", "result"])
        self.assertEqual([line["index"] for line in success_lines if line["phase"] == "progress"], [0, 1])
        self.assertEqual(success_lines[2]["result"]["proposal_states"]["intents:build.software"], "proposed_positive")
        self.assertTrue(success_lines[4]["cleanup_verified"])
        self.assertEqual(success_lines[1]["model_layer_bytes"], artifact["model_layer_bytes"])
        self.assertEqual(lifecycle, ["warmup", "verify_loaded_model"])
        self.assertNotIn("PRIVATE_SENTINEL", stdout.getvalue())


class CoordinatorEntryTests(unittest.TestCase):
    def cfg_and_comp(self, root: Path):
        masked._secure_private_path(root, is_directory=True)
        masked._verify_owner_private_access(root, is_directory=True)
        cfg = {"schema_version": adapter.SCHEMA, "revision": "test-v1",
               "export_dir": str(root / "private-export"), "export_manifest_sha256": "a" * 64,
               "comparison_config": str(root / "comparison.json"), "output_dir": str(root / "output"),
               "selection": {"row_ids": []}, "max_rows": 2, "assume_exhaustive_labels": False,
               "max_output_bytes": 100_000, "max_prompt_bytes": 8192, "max_protocol_bytes": 1_000_000,
               "max_progress_store_bytes": 100_000_000,
               "max_export_bytes": 100_000, "export_config_sha256": "b" * 64,
               "export_train_sha256": "c" * 64, "taxonomy_sha256": "d" * 64,
               "selected_columns": row()["columns"], "screening_policy_sha256": "3" * 64,
               "screening_code_sha256": "4" * 64}
        comp = {"max_manifest_bytes": 100_000}
        (root / "comparison.json").write_text("{}", encoding="utf-8")
        return cfg, comp

    def _full_fake_coordinator(self, scenario: str, bad_model_identity: bool = False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "owner-private"
            root.mkdir()
            masked._secure_private_path(root, is_directory=True)
            masked._verify_owner_private_access(root, is_directory=True)
            export = root / "export"
            export.mkdir()
            masked._secure_private_path(export, is_directory=True)
            columns = row()["columns"]
            rows = [row("a" * 64), row("b" * 64)]
            train_raw = b"".join((json.dumps(item, sort_keys=True, ensure_ascii=False,
                separators=(",", ":")) + "\n").encode("utf-8") for item in rows)
            export_config_raw = b'{"schema_version":"synthetic-export-config"}\n'
            export_manifest = {"schema_version": masked.EXPORT_SCHEMA,
                "config_sha256": adapter._sha(export_config_raw),
                "taxonomy_sha256": "c" * 64, "screening_policy_sha256": "3" * 64,
                "screening_code_sha256": "4" * 64, "selected_columns": columns,
                "output_files": {"train.jsonl": {"sha256": adapter._sha(train_raw),
                    "bytes": len(train_raw)}}}
            manifest_raw = (json.dumps(export_manifest, sort_keys=True) + "\n").encode()
            for name, raw in (("config.json", export_config_raw), ("manifest.json", manifest_raw),
                              ("train.jsonl", train_raw)):
                path = export / name
                path.write_bytes(raw)
                masked._secure_private_path(path, is_directory=False)
                masked._verify_owner_private_access(path, is_directory=False)

            output_parent = root / "output-parent"
            output_parent.mkdir()
            masked._secure_private_path(output_parent, is_directory=True)
            masked._verify_owner_private_access(output_parent, is_directory=True)
            comparison_path = root / "comparison.json"
            comparison_path.write_text("{}\n", encoding="utf-8")
            masked._secure_private_path(comparison_path, is_directory=False)
            masked._verify_owner_private_access(comparison_path, is_directory=False)
            cfg_sha, comp_sha, receipt_sha = "5" * 64, "6" * 64, "9" * 64
            cfg = {"schema_version": adapter.SCHEMA, "revision": "fake-coordinator-v1",
                "export_dir": str(export), "export_manifest_sha256": adapter._sha(manifest_raw),
                "export_config_sha256": adapter._sha(export_config_raw),
                "export_train_sha256": adapter._sha(train_raw), "max_export_bytes": len(train_raw),
                "taxonomy_sha256": "c" * 64, "screening_policy_sha256": "3" * 64,
                "screening_code_sha256": "4" * 64, "selected_columns": columns,
                "selection": {"row_ids": [item["id"] for item in rows], "max_rows": 2},
                "max_rows": 2, "max_prompt_bytes": 8192, "max_output_bytes": 1_000_000,
                "max_protocol_bytes": 1_000_000, "max_progress_store_bytes": 100_000_000,
                "max_manifest_bytes": 1_000_000, "assume_exhaustive_labels": False,
                "comparison_config": str(comparison_path), "output_dir": str(output_parent / "run")}
            comp = {"model_name": "fixture/model", "server_binary_sha256": "a" * 64,
                "server_version": "fixture-server", "expected_manifest_sha256": "4" * 64,
                "remote_acquisition_receipt": {"synthetic": True}, "runtime_port": 11434,
                "remote_host": "worker.invalid", "remote_user": "teacher", "ssh_port": 22,
                "remote_home": "/home/teacher", "remote_workdir": "/home/teacher/tasks",
                "remote_uid": 1000, "remote_sudo_path": "/usr/bin/sudo",
                "remote_systemd_run_path": "/usr/bin/systemd-run", "remote_true_path": "/usr/bin/true",
                "remote_unshare_path": "/usr/bin/unshare", "remote_bash_path": "/usr/bin/bash",
                "remote_ip_path": "/usr/sbin/ip", "remote_setpriv_path": "/usr/bin/setpriv",
                "remote_python_path": "/usr/bin/python3", "remote_systemctl_path": "/usr/bin/systemctl",
                "remote_expected_hostname": "fake-worker", "remote_expected_architecture": "x86_64",
                "resource_memory_max_bytes": 10_000, "resource_cpu_quota_percent": 50,
                "remote_scope_tasks_max": 10, "min_memory_available_bytes": 1,
                "min_disk_free_bytes": 1, "min_gpu_free_bytes": 0,
                "worker_start_timeout_seconds": 1, "worker_ready_wait_margin_seconds": 1,
                "timeout_seconds": 1, "batch_overhead_margin_seconds": 1,
                "max_request_bytes": 1000, "max_response_bytes": 1000, "max_payload_bytes": 1_000_000,
                "max_manifest_bytes": 1_000_000,
                "num_ctx": 256, "num_predict": 64, "temperature": 0.0,
                "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0,
                "system_prompt": "{labels} {text}", "prompt_revision": "fake-v1"}
            host = {"hostname": "fake-worker", "architecture": "x86_64", "netns_inode": 900,
                "memory_available_bytes": 1_000_000, "home_disk_free_bytes": 1_000_000,
                "loaded_model_count": 0, "ollama_service_pid": 12, "ollama_version": "fixture-server"}
            inference_ids: list[str] = []
            ack_log: list[tuple[int, int, int]] = []
            captured_payloads: list[tuple[int, list[str], bool]] = []
            processes = []
            cleanup_calls = []
            real_run = subprocess.run

            class FakeMasked:
                @staticmethod
                def _read_bounded(path, limit, what):
                    return masked._read_bounded(path, limit, what)
                @staticmethod
                def _secure_private_path(path, is_directory):
                    return masked._secure_private_path(path, is_directory)
                @staticmethod
                def _verify_owner_private_access(path, is_directory):
                    return masked._verify_owner_private_access(path, is_directory)
                @staticmethod
                def load_export(snapshot, train_only=False):
                    loaded_manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
                    loaded_rows = [json.loads(line) for line in
                        (snapshot / "train.jsonl").read_text(encoding="utf-8").splitlines()]
                    return loaded_manifest, {"train": loaded_rows}

            class FakeInput:
                def __init__(self, process):
                    self.process = process
                    self.writes = []
                    self.closed = False
                def write(self, value):
                    raw = bytes(value)
                    self.writes.append(raw)
                    try:
                        item = json.loads(raw)
                    except (UnicodeError, json.JSONDecodeError):
                        return len(raw)
                    if item.get("schema_version") == adapter.PAYLOAD_SCHEMA:
                        self.process.payload = item
                        self.process.payload_raw = raw
                        captured_payloads.append((self.process.index,
                            [record["id"] for record in item["records"]], item["zero_row_probe"]))
                    elif item.get("phase") == "ack":
                        event = self.process.expected_events[item["index"]]
                        adapter._validate_progress_ack(item, self.process.payload["attempt_id"],
                                                       item["index"], event)
                        events_path = adapter._progress_store_path(Path(cfg["output_dir"]), allow_final=True) / "events.jsonl"
                        durable_count = len(events_path.read_bytes().splitlines())
                        expected_count = self.process.initial_event_count + len(self.process.delivered_events)
                        if durable_count != expected_count:
                            raise AssertionError("ACK preceded durable event persistence")
                        self.process.acks.append(item["index"])
                        ack_log.append((self.process.index, item["index"], durable_count))
                    return len(raw)
                def flush(self):
                    return None
                def close(self):
                    self.closed = True

            class FakeProcess:
                def __init__(self, index):
                    self.index = index
                    self.stdin = FakeInput(self)
                    self.returncode = None
                    self.read_count = 0
                    self.payload = None
                    self.payload_raw = None
                    self.ready = None
                    self.events = []
                    self.expected_events = {}
                    self.delivered_events = []
                    self.acks = []
                    self.initial_event_count = None
                    self.worker_config_sha = None
                    self.aborted = False
                    self.abort_reason = None
                    self.outputs = []
                    processes.append(self)
                def poll(self):
                    return self.returncode
                def kill(self):
                    self.returncode = -9
                def wait(self, timeout=None):
                    if self.returncode is None:
                        self.returncode = 0
                    return self.returncode

            def make_answer(record):
                def infer(_request, _labels):
                    inference_ids.append(record["id"])
                    return "answered", ["intents:build.software"], 1.0
                with mock.patch.object(adapter, "_request", return_value=b"synthetic-request"):
                    event = adapter._worker_loop([record], columns, {}, infer, False, 8192)[0][0]
                return event

            def read_worker_line(process, _lines, _timeout):
                ordinal = process.read_count
                process.read_count += 1
                if ordinal == 0:
                    pid, netns = 500 + process.index, 2000 + process.index
                    runtime = {"boundary": "same_uid_private_network_namespace_without_nonloopback_routes",
                        "network_namespace_inode": netns, "server_pid": pid,
                        "server_binary_sha256": comp["server_binary_sha256"],
                        "server_version": comp["server_version"], "process_tree_rss_bytes": 100,
                        "process_tree_rss_scope": "server and observed descendants; summed RSS can double-count shared pages"}
                    process.runtime = runtime
                    return {"phase": "control_ready", "control_nonce": f"{process.index + 1:032x}",
                        "network_namespace_inode": netns, "server_pid": pid, "port": comp["runtime_port"],
                        "receipt_sha256": receipt_sha, "server_binary_sha256": comp["server_binary_sha256"],
                        "server_version": comp["server_version"],
                        "worker_code_sha256": adapter._sha(Path(adapter.__file__).read_bytes()),
                        "worker_config_sha256": process.worker_config_sha, "runtime_boundary": runtime}
                if ordinal == 1:
                    manifest_sha = "f" * 64 if bad_model_identity and process.index == 1 else "4" * 64
                    layer_bytes = 101 if bad_model_identity and process.index == 1 else 100
                    process.ready = {"phase": "ready_for_data", "network_namespace_inode": process.runtime["network_namespace_inode"],
                        "server_pid": process.runtime["server_pid"], "port": comp["runtime_port"],
                        "model_manifest_sha256": manifest_sha, "model_layer_bytes": layer_bytes,
                        "artifact_layer_bytes": layer_bytes, "receipt_sha256": receipt_sha,
                        "server_binary_sha256": comp["server_binary_sha256"], "server_version": comp["server_version"],
                        "worker_code_sha256": adapter._sha(Path(adapter.__file__).read_bytes()),
                        "worker_config_sha256": process.worker_config_sha,
                        "runtime_boundary": process.runtime,
                        "model_identity": {"reported_manifest_sha256": manifest_sha},
                        "warmup": {"status": "completed"}, "resources_before": {"available": 9}}
                    return process.ready
                if process.payload is None:
                    raise AssertionError("controller requested a worker result before sending a payload")
                if process.payload["zero_row_probe"]:
                    return {"phase": "result", "schema_version": adapter.WORKER_RESULT_SCHEMA,
                        "payload_sha256": adapter._sha(process.payload_raw), "rows_received": 0,
                        "statuses": [], "progress_sha256": adapter._sha(b""), "aborted": False,
                        "abort_reason": None, "resources_after": {"available": 8},
                        "warmup": {"status": "completed"}, "artifact": {"model_layer_bytes": 100},
                        "model_identity": {"reported_manifest_sha256": "4" * 64},
                        "model_layer_bytes": 100, "receipt_sha256": receipt_sha,
                        "runtime_boundary": process.runtime, "worker_config_sha256": process.worker_config_sha,
                        "cleanup_verified": True}
                if process.initial_event_count is None:
                    store_events = adapter._progress_store_path(Path(cfg["output_dir"]), allow_final=True) / "events.jsonl"
                    process.initial_event_count = len(store_events.read_bytes().splitlines())
                event_ordinal = ordinal - 2
                records = process.payload["records"]
                if scenario == "prefix" and process.index == 1:
                    if event_ordinal == 0:
                        event = make_answer(records[0])
                    else:
                        if process.acks != [0]:
                            raise AssertionError("second row was requested before first row ACK")
                        raise TimeoutError("synthetic process interruption after first durable event")
                elif scenario == "abort" and process.index == 1:
                    if not process.events:
                        infer_count = 0
                        def timeout_first(_request, _labels):
                            nonlocal infer_count
                            infer_count += 1
                            inference_ids.append(records[infer_count - 1]["id"])
                            if infer_count == 1:
                                raise TimeoutError("synthetic teacher timeout")
                            return "answered", ["intents:build.software"], 1.0
                        with mock.patch.object(adapter, "_request", return_value=b"synthetic-request"):
                            events, outputs, aborted, reason = adapter._worker_loop(records, columns, {},
                                timeout_first, False, 8192)
                        process.events = events
                        process.outputs = outputs
                        process.aborted = aborted
                        process.abort_reason = reason
                        process.expected_events = {item["index"]: item for item in events}
                    if event_ordinal < len(process.events):
                        if event_ordinal and process.acks != list(range(event_ordinal)):
                            raise AssertionError("abort continuation event arrived before prior ACK")
                        event = process.events[event_ordinal]
                    else:
                        return make_result(process)
                else:
                    if event_ordinal < len(records):
                        if event_ordinal and process.acks != list(range(event_ordinal)):
                            raise AssertionError("next synthetic inference began before prior ACK")
                        event = make_answer(records[event_ordinal])
                        event["index"] = event_ordinal
                    else:
                        return make_result(process)
                process.expected_events[event["index"]] = event
                process.delivered_events.append(event)
                return event

            def make_result(process):
                events_raw = b"".join((json.dumps(event, sort_keys=True, ensure_ascii=False,
                    separators=(",", ":")) + "\n").encode("utf-8") for event in process.delivered_events)
                statuses = [event["status"] for event in process.delivered_events]
                return {"phase": "result", "schema_version": adapter.WORKER_RESULT_SCHEMA,
                    "payload_sha256": adapter._sha(process.payload_raw),
                    "rows_received": len(process.payload["records"]), "statuses": statuses,
                    "progress_sha256": adapter._sha(events_raw), "aborted": process.aborted,
                    "abort_reason": process.abort_reason, "resources_after": {"available": 8},
                    "warmup": {"status": "completed"}, "artifact": {"model_layer_bytes": process.ready["artifact_layer_bytes"]},
                    "model_identity": process.ready["model_identity"],
                    "model_layer_bytes": process.ready["model_layer_bytes"],
                    "receipt_sha256": receipt_sha, "runtime_boundary": process.runtime,
                    "worker_config_sha256": process.worker_config_sha, "cleanup_verified": True}

            def queue_protocol(_process):
                lines = adapter.queue.Queue()
                lines.put(None)
                return lines

            def fake_subprocess_run(argv, *args, **kwargs):
                if isinstance(argv, (list, tuple)) and argv and argv[0] == "scp":
                    local = Path(argv[-2])
                    if local.name == "worker-config.json":
                        for process in processes:
                            if process.worker_config_sha is None:
                                process.worker_config_sha = adapter._sha(local.read_bytes())
                        # The scp happens before its associated Popen; retain for the next worker.
                        transport.pending_worker_config_sha = adapter._sha(local.read_bytes())
                    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
                return real_run(argv, *args, **kwargs)

            class FakeTransport:
                pending_worker_config_sha = None
                def popen(self, *_args, **_kwargs):
                    process = FakeProcess(len(processes))
                    process.worker_config_sha = self.pending_worker_config_sha
                    self.pending_worker_config_sha = None
                    return process

            transport = FakeTransport()
            class FakeMaskedExport(FakeMasked):
                pass

            def perform_run(resume):
                return adapter._run_unleased(comparison_path, resume=resume)

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(adapter, "_masked", return_value=FakeMaskedExport()))
                stack.enter_context(mock.patch.object(adapter, "load_config", return_value=(cfg, cfg_sha)))
                stack.enter_context(mock.patch.object(adapter.comparator, "load_config", return_value=(comp, comp_sha)))
                stack.enter_context(mock.patch.object(adapter.comparator, "verify_acquisition",
                    return_value=({"synthetic": True}, receipt_sha)))
                stack.enter_context(mock.patch.object(adapter, "_remote_identity_snapshot", return_value=host))
                stack.enter_context(mock.patch.object(adapter.comparator, "_remote_snapshot", return_value=host))
                stack.enter_context(mock.patch.object(adapter.comparator, "_remote_port_closed", return_value=True))
                stack.enter_context(mock.patch.object(adapter.comparator, "_remote_install",
                    return_value={"receipt_sha256": receipt_sha}))
                stack.enter_context(mock.patch.object(adapter.comparator, "_ssh_run",
                    return_value=SimpleNamespace(returncode=0)))
                stack.enter_context(mock.patch.object(adapter, "_remote_create_workdir"))
                stack.enter_context(mock.patch.object(adapter, "_remote_verify_files"))
                stack.enter_context(mock.patch.object(adapter, "_remote_cleanup_worker",
                    side_effect=lambda *_: cleanup_calls.append("cleaned")))
                fake_subprocess = SimpleNamespace(run=fake_subprocess_run, Popen=transport.popen,
                    PIPE=subprocess.PIPE, DEVNULL=subprocess.DEVNULL,
                    TimeoutExpired=subprocess.TimeoutExpired, SubprocessError=subprocess.SubprocessError)
                stack.enter_context(mock.patch.object(adapter, "subprocess", fake_subprocess))
                stack.enter_context(mock.patch.object(adapter.comparator, "_queue_protocol", side_effect=queue_protocol))
                stack.enter_context(mock.patch.object(adapter, "_read_worker_line", side_effect=read_worker_line))
                stack.enter_context(mock.patch.object(adapter, "_reverify_export", wraps=adapter._reverify_export))

                if bad_model_identity:
                    with self.assertRaises(adapter.ProposalError):
                        perform_run(False)
                    self.assertEqual(len(processes), 2)
                    self.assertEqual(captured_payloads, [(0, [], True)])
                    self.assertFalse(Path(cfg["output_dir"]).exists())
                    self.assertFalse(adapter._progress_store_path(Path(cfg["output_dir"])).exists())
                    self.assertNotIn("PRIVATE_SENTINEL", json.dumps(captured_payloads))
                    return

                if scenario == "prefix":
                    with self.assertRaises(TimeoutError):
                        perform_run(False)
                    store = adapter._progress_store_path(Path(cfg["output_dir"]), allow_final=True)
                    first_events = [json.loads(line) for line in (store / "events.jsonl").read_text().splitlines()]
                    self.assertEqual([event["index"] for event in first_events], [0])
                    self.assertEqual(list(store.glob("terminal-*.json")), [])
                    self.assertFalse(Path(cfg["output_dir"]).exists())
                    result = perform_run(True)
                    self.assertEqual(result["status_counts"], {"answered": 2})
                    self.assertEqual(captured_payloads,
                        [(0, [], True), (1, [rows[0]["id"], rows[1]["id"]], False),
                         (2, [], True), (3, [rows[1]["id"]], False)])
                    final = Path(result["output_directory"])
                    saved = json.loads((final / "manifest.json").read_text())
                    proposals = [json.loads(line) for line in (final / "proposals.jsonl").read_text().splitlines()]
                    self.assertEqual(saved["status"], "complete")
                    self.assertTrue(saved["provenance"]["terminal_worker_result_verified"])
                    self.assertEqual([item["row_id"] for item in proposals], [item["id"] for item in rows])
                    self.assertEqual(saved["counts"]["statuses"], {"answered": 2})
                    self.assertEqual(proposals[0]["proposal_states"]["intents:build.software"], "proposed_positive")
                    self.assertEqual(proposals[0]["proposal_states"]["risks:sensitive-data"], "unknown")
                    self.assertEqual([item[1] for item in ack_log], [0, 0])
                    self.assertEqual([item[2] for item in ack_log], [1, 2])
                    attempts = [json.loads(path.read_text()) for path in store.glob("attempt-*.json")]
                    self.assertEqual(sorted(item["indices"] for item in attempts), [[0, 1], [1]])
                    event_records = [json.loads(line) for line in (store / "events.jsonl").read_text().splitlines()]
                    self.assertEqual([item["index"] for item in event_records], [0, 1])
                    self.assertEqual(len(list(store.glob("terminal-*.json"))), 1)
                    self.assertEqual(len(cleanup_calls), 4)
                    self.assertEqual(len({(process.runtime["server_pid"], process.runtime["network_namespace_inode"])
                                          for process in (processes[0], processes[2])}), 2)
                    self.assertNotIn("PRIVATE_SENTINEL", (final / "manifest.json").read_text()
                        + (final / "proposals.jsonl").read_text())
                else:
                    partial_result = perform_run(False)
                    partial = Path(partial_result["output_directory"])
                    old_manifest = (partial / "manifest.json").read_bytes()
                    old_proposals = (partial / "proposals.jsonl").read_bytes()
                    self.assertEqual(json.loads(old_manifest)["status"], "partial_aborted_ambiguous")
                    self.assertTrue(json.loads(old_manifest)["provenance"]["terminal_worker_result_verified"])
                    revision_result = perform_run(True)
                    revision = Path(revision_result["output_directory"])
                    linked = json.loads((revision / "manifest.json").read_text())
                    self.assertNotEqual(revision, partial)
                    self.assertEqual(linked["status"], "complete")
                    self.assertTrue(linked["provenance"]["terminal_worker_result_verified"])
                    self.assertEqual(linked["provenance"]["publication_revision"]["parent_manifest_sha256"],
                                     adapter._sha(old_manifest))
                    self.assertEqual((partial / "manifest.json").read_bytes(), old_manifest)
                    self.assertEqual((partial / "proposals.jsonl").read_bytes(), old_proposals)
                    self.assertEqual(captured_payloads,
                        [(0, [], True), (1, [rows[0]["id"], rows[1]["id"]], False),
                         (2, [], True), (3, [rows[0]["id"], rows[1]["id"]], False)])
                    revision_attempts = [json.loads(path.read_text()) for path in
                        adapter._progress_store_path(Path(cfg["output_dir"]), allow_final=True).glob("attempt-*.json")]
                    self.assertEqual(sorted(item["indices"] for item in revision_attempts), [[0, 1], [0, 1]])
                    self.assertEqual([event["row_id"] for event in
                        [json.loads(line) for line in (revision / "proposals.jsonl").read_text().splitlines()]],
                        [item["id"] for item in rows])
                    self.assertNotIn("PRIVATE_SENTINEL", (revision / "manifest.json").read_text()
                        + (revision / "proposals.jsonl").read_text())

    def test_full_unleased_prefix_interruption_resume_uses_actual_transient_worker(self):
        self._full_fake_coordinator("prefix")

    def test_full_unleased_abort_resumes_to_linked_immutable_revision(self):
        self._full_fake_coordinator("abort")

    def test_full_unleased_model_identity_mismatch_sends_no_train_payload(self):
        self._full_fake_coordinator("prefix", bad_model_identity=True)

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
