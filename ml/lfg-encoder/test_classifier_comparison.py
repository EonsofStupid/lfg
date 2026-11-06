from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import classifier_comparison as comparison
import screen_private_corpus


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class ComparisonTests(unittest.TestCase):
    def _corpus(self, root: Path, include_test: bool = False, same_project_across_splits: bool = False,
                extra_screened_validation: bool = False, fully_unknown_train_row: bool = False):
        root.mkdir(parents=True)
        tax = {"schema_version": "lfg.encoder-evaluation-taxonomy.v1",
               "intents": ["build", "verify"], "capabilities": ["write"], "risks": ["sensitive"]}
        (root / "taxonomy.json").write_bytes(_encoded(tax))
        policy = {"schema_version": screen_private_corpus.POLICY_VERSION, "revision": "test-policy-v1",
                  "limits": {key: 100000 for key in screen_private_corpus.LIMITS},
                  "rules": [{"id": "test-secret", "pattern": "SCREEN-SECRET-123", "replacement": "[REDACTED]"}]}
        policy_path = root / "policy.json"
        policy_raw = _encoded(policy)
        policy_path.write_bytes(policy_raw)
        policy, rules, policy_sha = screen_private_corpus.load_policy_bytes(policy_raw, len(policy_raw))
        rows = {
            "train": [
                {"id": "a", "family": "fa", "text": "screened train text alpha", "intents": ["build", "verify"],
                 "capabilities": ["write"], "risks": [], "verified": False, "source": {"project": "train-project"}},
                {"id": "b", "family": "fb", "text": "SCREEN-SECRET-123 train text beta", "intents": ["build"],
                 "capabilities": ["write"], "risks": [], "verified": False, "source": {"project": "train-project"}},
            ],
            "validation": [
                {"id": "c", "family": "fc", "text": "screened validation unique", "intents": ["verify"],
                 "capabilities": [], "risks": [], "verified": False, "source": {"project": "validation-project"}},
            ],
        }
        if fully_unknown_train_row:
            rows["train"].append({"id": "unknown", "family": "funknown", "text": "screened train unknown",
                                  "capabilities": [], "risks": [], "verified": False,
                                  "source": {"project": "train-project"}})
        sources, output_digests, output_counts, source_counts = [], {}, {}, {}
        matched_records = 0
        for split in ("train", "validation"):
            source_bytes, filtered = bytearray(), []
            group = f"{split}-group"
            for item in rows[split]:
                if same_project_across_splits and split == "validation":
                    item["source"]["project"] = "train-project"
                line = _encoded(item)
                source_bytes.extend(line + b"\n")
                screened_text, matches = screen_private_corpus._screen(item["text"], rules, policy["limits"])
                filtered.append({"schema_version": "lfg.screened-text-candidate.v1", "id": item["id"],
                                 "family": item["family"], "project": item["source"]["project"],
                                 "project_group": group, "split": split, "text": item["text"],
                                 "candidate_file_sha256": "a" * 64,
                                 "source_record_sha256": hashlib.sha256(line).hexdigest(),
                                 "source_text_sha256": hashlib.sha256(item["text"].encode()).hexdigest(),
                                 "screened_text_sha256": hashlib.sha256(screened_text.encode()).hexdigest(),
                                 "screen_policy_sha256": policy_sha, "redaction_counts": dict(matches),
                                 "review_status": "unreviewed", "redaction_complete": False})
                filtered[-1]["text"] = screened_text
                matched_records += int(bool(matches))
            source_file = root / f"source-{split}.jsonl"
            source_file.write_bytes(source_bytes)
            source_counts[split] = len(rows[split])
            source_sha = hashlib.sha256(source_bytes).hexdigest()
            for record in filtered:
                record["candidate_file_sha256"] = source_sha
            sources.append({"project": rows[split][0]["source"]["project"], "project_group": group, "split": split,
                            "path": source_file.name})
            screened_bytes = b"".join(_encoded(x) + b"\n" for x in filtered)
            if extra_screened_validation and split == "validation":
                extra = dict(filtered[0], id="extra", source_record_sha256="d" * 64)
                screened_bytes += _encoded(extra) + b"\n"
                filtered.append(extra)
            (root / f"{split}.jsonl").write_bytes(screened_bytes)
            output_digests[split] = hashlib.sha256(screened_bytes).hexdigest()
            output_counts[split] = len(filtered)
        if include_test:
            sources.append({"project": "heldout", "project_group": "heldout", "split": "test",
                            "path": "must-not-be-opened.jsonl"})
        plan = {"schema_version": "lfg.encoder-development-corpus-plan.v2", "review_ledger": None,
                "taxonomy": "taxonomy.json", "sources": sources}
        plan_raw = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        plan_path = root / "plan.json"
        plan_path.write_bytes(plan_raw)
        ledger_rows = [{"schema_version": "lfg.encoder-review.v1", "id": item["id"], "decision": "pending"}
                       for split_rows in rows.values() for item in split_rows]
        (root / "reviews.jsonl").write_bytes(b"".join(_encoded(item) + b"\n" for item in ledger_rows))
        manifest = {"schema_version": "lfg.private-corpus-screen-manifest.v1",
                    "plan_sha256": hashlib.sha256(plan_raw).hexdigest(), "policy_sha256": policy_sha,
                    "screen_code_sha256": comparison.sha(Path(screen_private_corpus.__file__).read_bytes()),
                    "helper_sha256": {Path(module.__file__).name: comparison.sha(Path(module.__file__).read_bytes())
                                      for module in (screen_private_corpus.assessment_distill,
                                                     screen_private_corpus.corpus_audit,
                                                     screen_private_corpus.review_corpus)},
                    "source_snapshots": [{"sha256": hashlib.sha256((root / s["path"]).read_bytes()).hexdigest(),
                                          "records": source_counts[s["split"]]} for s in sources if s["split"] != "test"],
                    "counts": output_counts, "matched_records": matched_records,
                    "claims": {"redaction_complete": False, "human_reviewed": False,
                               "original_source_verified": False, "independent_split_verified": False,
                               "promotion_eligible": False}, "outputs": output_digests}
        (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return plan_path, policy_path

    def _config(self, corpus: Path, plan: Path, policy: Path, **extra):
        return {"schema_version": comparison.SCHEMA, "revision": "test-1", "screened_corpus": str(corpus),
                "label_plan": str(plan), "screen_policy": str(policy),
                "review_ledger": str(corpus / "reviews.jsonl"), "max_ledger_bytes": 65536,
                "target_field": "intents", "selected_labels": ["build", "verify"],
                "seed": 17, "max_rows": 100, "max_text_bytes": 1024, "max_source_bytes": 4096,
                "max_total_source_bytes": 8192,
                "max_manifest_bytes": 4096, "max_screened_file_bytes": 8192, "max_features": 100,
                "knn_neighbors": 1, "tree_max_depth": None, "tfidf_ngram_range": [1, 1],
                "logistic_c": 1.0, "logistic_max_iter": 200, "knn_weights": "uniform", "knn_p": 2,
                "tree_criterion": "gini", "near_duplicate_similarity": 0.9,
                "max_near_duplicate_pairs": 100, "max_near_duplicate_work": 1_000_000,
                "max_policy_bytes": 65536, "constant_target_behavior": "abstain",
                "validation_group_use_claim": "unknown", "rejected_label_handling": "exclude", **extra}

    def test_load_config_accepts_inert_extension_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(self._config(Path(tmp), Path(tmp), Path(tmp), extension_note={"future": True})))
            config, digest = comparison.load_config(path, 65536)
            self.assertEqual(config["extension_note"], {"future": True})
            self.assertEqual(len(digest), 64)

    def test_test_source_rejected_before_source_file_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screened"
            plan, policy = self._corpus(root, include_test=True)
            with self.assertRaisesRegex(ValueError, "non-development source"):
                self._load(root, plan, policy)

    def test_complete_multilabel_arrays_preserved_and_missing_rejected(self):
        taxonomy = {"intents": {"build", "verify"}}
        rows = [{"intents": ["build", "verify"], "labels_pending": False, "verified": False},
                {"intents": [], "labels_pending": False, "verified": False}]
        encoder, matrix, observed = comparison._matrix(rows, "intents", ["build", "verify"], taxonomy)
        self.assertEqual(list(encoder.classes_), ["build", "verify"])
        self.assertEqual(matrix.tolist(), [[1, 1], [0, 0]])
        self.assertEqual(observed.tolist(), [[True, True], [False, False]])
        _, explicit_empty, explicit_observed = comparison._matrix(
            [{"intents": [], "verified": False, "target_completeness": {"intents": True}}],
            "intents", ["build"], taxonomy)
        self.assertEqual(explicit_empty.tolist(), [[0]])
        self.assertEqual(explicit_observed.tolist(), [[True]])
        _, missing, missing_observed = comparison._matrix([{"verified": False}], "intents", ["build"], taxonomy)
        self.assertEqual(missing.tolist(), [[0]])
        self.assertEqual(missing_observed.tolist(), [[False]])

    def test_false_completeness_keeps_positives_and_omitted_labels_unknown(self):
        taxonomy = {"intents": {"build", "verify"}}
        row = {"intents": ["build"], "target_completeness": {"intents": False}, "verified": False}
        _, matrix, known = comparison._matrix([row], "intents", ["build", "verify"], taxonomy)
        self.assertEqual(matrix.tolist(), [[1, 0]])
        self.assertEqual(known.tolist(), [[True, False]])

    def test_labels_pending_integer_is_rejected_and_verified_state_is_preserved(self):
        taxonomy = {"intents": {"build"}}
        with self.assertRaisesRegex(ValueError, "labels_pending"):
            comparison._matrix([{"intents": ["build"], "verified": False, "labels_pending": 1}],
                               "intents", ["build"], taxonomy)
        self.assertEqual(comparison._label_review_state(
            [{"intents": ["build"], "verified": True, "labels_pending": False}], "intents")["verified_true"], 1)
        state = comparison._label_review_state(
            [{"intents": ["build"], "labels_pending": False}, {"intents": ["build"]}], "intents")
        self.assertEqual(state["labels_pending_false"], 1)
        self.assertEqual(state["labels_pending_absent"], 1)

    def test_review_ledger_must_bind_exact_selected_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screened"
            plan, policy = self._corpus(root)
            ledger = root / "reviews.jsonl"
            lines = ledger.read_bytes().splitlines(keepends=True)
            ledger.write_bytes(b"".join(lines[:-1]))
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(self._config(root, plan, policy)))
            with self.assertRaisesRegex(ValueError, "cover exactly"):
                comparison.run(config_path, Path(tmp) / "report")

    def test_macro_precision_counts_predicted_only_labels(self):
        import numpy as np
        true = np.array([[1, 0], [1, 0]])
        pred = np.array([[1, 1], [1, 1]])
        known = np.ones_like(true, dtype=bool)
        metrics = comparison._metrics(true, pred, known, ["observed", "predicted-only"])
        self.assertEqual(metrics["macro_precision"], 0.5)
        self.assertEqual(metrics["macro_precision_supported_labels"], 1.0)

    def test_unscored_metrics_are_null_and_measured_false_positive_remains_zero(self):
        import numpy as np
        true = np.array([[0]])
        unknown_prediction = np.array([[-1]])
        known = np.array([[True]])
        unscored = comparison._metrics(true, unknown_prediction, known, ["negative-only"])
        self.assertEqual(unscored["scored_label_predictions"], 0)
        self.assertIsNone(unscored["labels"]["negative-only"]["precision"])
        self.assertIsNone(unscored["labels"]["negative-only"]["recall"])
        self.assertIsNone(unscored["labels"]["negative-only"]["f1"])
        self.assertIsNone(unscored["macro_precision"])
        self.assertEqual(unscored["macro_metric_label_coverage"]["precision"],
                         {"defined_labels": 0, "total_labels": 1})
        measured_fp = comparison._metrics(true, np.array([[1]]), known, ["negative-only"])
        self.assertEqual(measured_fp["labels"]["negative-only"]["precision"], 0.0)
        self.assertIsNone(measured_fp["labels"]["negative-only"]["recall"])
        self.assertEqual(measured_fp["labels"]["negative-only"]["f1"], 0.0)

    def test_rejected_labels_are_excluded_while_pending_labels_remain_fit_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screened"
            plan, policy = self._corpus(root)
            ledger = root / "reviews.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            for row in rows:
                if row["id"] == "b":
                    row["decision"] = "rejected"
            ledger.write_bytes(b"".join(_encoded(row) + b"\n" for row in rows))
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(self._config(root, plan, policy, selected_labels=["build"])))
            report = comparison.run(config_path, Path(tmp) / "report")
            self.assertEqual(report["data"]["review_ledger_decisions"],
                             {"pending": 2, "approved": 0, "rejected": 1})
            self.assertEqual(report["data"]["rejected_rows_excluded"], {"train": 1, "validation": 0})
            self.assertEqual(report["data"]["rejected_label_handling"], "exclude")
            for result in report["algorithms"].values():
                self.assertEqual(result["fit_rows"], 1)
                self.assertEqual(result["training_label_support"]["build"]["positive"], 1)
                self.assertEqual(result["training_label_support"]["build"]["unknown_rows"], 1)

    def test_constant_target_fallback_is_consistent_for_all_positive_and_negative(self):
        import numpy as np
        for value in (0, 1):
            training = np.full((3, 1), value, dtype=int)
            known = np.ones_like(training, dtype=bool)
            variable, constant, unsupported = comparison._target_partition(training, known)
            self.assertEqual(variable, [])
            self.assertEqual(constant, [0])
            self.assertEqual(unsupported, [])
            abstain = comparison._constant_fallbacks(training, known, ["x"], 2, "abstain")
            fallback = comparison._constant_fallbacks(training, known, ["x"], 2, "majority_class")
            self.assertEqual(abstain.tolist(), [[-1], [-1]])
            self.assertEqual(fallback.tolist(), [[value], [value]])

    def test_constant_target_majority_behavior_is_shared_by_all_estimators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screened"
            plan, policy = self._corpus(root)
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(self._config(root, plan, policy, selected_labels=["build"],
                                                            constant_target_behavior="majority_class")))
            output = Path(tmp) / "report"
            report = comparison.run(config_path, output)
            for result in report["algorithms"].values():
                self.assertEqual(result["status"], "constant_target_fallback_only")
                self.assertEqual(result["metrics"]["labels"]["build"]["predicted_positive"], 1)
                self.assertEqual(result["metrics"]["labels"]["build"]["fp"], 1)

    def test_abstains_on_zero_variance_target_in_normal_model_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screened"
            plan, policy = self._corpus(root)
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(self._config(root, plan, policy)))
            report = comparison.run(config_path, Path(tmp) / "report")
            for result in report["algorithms"].values():
                metric = result["metrics"]["labels"]["build"]
                self.assertEqual(metric["unknown_predictions"], 1)
                self.assertEqual(metric["predicted_positive"], 0)

    def test_report_path_rejects_git_worktree_even_when_ignored(self):
        repo = Path(__file__).resolve().parents[2]
        with self.assertRaisesRegex(ValueError, "outside every Git worktree"):
            comparison.require_private_report_path(repo / "target" / "ignored-private-report.json")

    def test_same_project_cannot_cross_splits_even_with_consistent_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screened"
            plan, policy = self._corpus(root, same_project_across_splits=True)
            with self.assertRaisesRegex(ValueError, "project is assigned"):
                self._load(root, plan, policy)

    def test_unmatched_screened_row_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screened"
            plan, policy = self._corpus(root, extra_screened_validation=True)
            with self.assertRaisesRegex(ValueError, "sets do not match exactly"):
                self._load(root, plan, policy)

    def test_screened_text_must_match_recomputed_policy_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screened"
            plan, policy = self._corpus(root)
            rows = [json.loads(line) for line in (root / "validation.jsonl").read_text().splitlines()]
            rows[0]["text"] = "forged screened text"
            rows[0]["screened_text_sha256"] = hashlib.sha256(rows[0]["text"].encode()).hexdigest()
            raw = b"".join(_encoded(row) + b"\n" for row in rows)
            (root / "validation.jsonl").write_bytes(raw)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["outputs"]["validation"] = hashlib.sha256(raw).hexdigest()
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recomputed policy output"):
                self._load(root, plan, policy)

    def test_near_duplicate_work_budget_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screened"
            plan, policy = self._corpus(root)
            config = self._config(root, plan, policy, max_near_duplicate_work=1)
            with self.assertRaisesRegex(ValueError, "work budget"):
                comparison.load_screened_corpus(root, plan, config["max_rows"], config["max_text_bytes"],
                    config["max_source_bytes"], config["max_total_source_bytes"], config["max_manifest_bytes"],
                    config["max_screened_file_bytes"], config["near_duplicate_similarity"],
                    config["max_near_duplicate_pairs"], config["max_near_duplicate_work"], policy,
                    config["max_policy_bytes"])

    def test_end_to_end_writes_metrics_without_raw_text_or_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screened"
            plan, policy = self._corpus(root)
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(self._config(root, plan, policy)))
            output = Path(tmp) / "report"
            report = comparison.run(config_path, output)
            self.assertEqual(set(report["algorithms"]), {"logistic_regression", "knn", "decision_tree"})
            for item in report["algorithms"].values():
                self.assertEqual(item["status"], "completed", msg=json.dumps(item))
                self.assertEqual(item["fit_rows"], 2)
                self.assertEqual(item["prediction_rows"], 1)
                self.assertEqual(item["metrics"]["examples"], 1)
            self.assertGreater(report["algorithms"]["knn"]["retained_training_features"]["sparse_storage_bytes_across_indexes"], 0)
            self.assertEqual(report["algorithms"]["knn"]["persisted_model_artifact_bytes"], 0)
            encoded = (output / "comparison-report.json").read_text()
            self.assertNotIn("screened train text", encoded)
            self.assertNotIn("SCREEN-SECRET-123", encoded)
            self.assertNotIn('"id": "a"', encoded)
            self.assertTrue(report["data"]["screening_recomputed_with_local_code"])
            self.assertFalse(report["promotion_eligible"])
            comparison.require_private_report_path(output / "comparison-report.json", require_existing=True)

    def test_end_to_end_excludes_fully_unknown_training_rows_from_features_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screened"
            plan, policy = self._corpus(root, fully_unknown_train_row=True)
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(self._config(root, plan, policy)))
            report = comparison.run(config_path, Path(tmp) / "report")
            self.assertEqual(report["data"]["train_rows"], 3)
            for result in report["algorithms"].values():
                self.assertNotEqual(result["status"], "failed", msg=json.dumps(result))
                self.assertEqual(result["fit_rows"], 2)

    def _load(self, root, plan, policy):
        config = self._config(root, plan, policy)
        return comparison.load_screened_corpus(root, plan, config["max_rows"], config["max_text_bytes"],
            config["max_source_bytes"], config["max_total_source_bytes"], config["max_manifest_bytes"],
            config["max_screened_file_bytes"], config["near_duplicate_similarity"],
            config["max_near_duplicate_pairs"], config["max_near_duplicate_work"], policy,
            config["max_policy_bytes"])


if __name__ == "__main__":
    unittest.main()
