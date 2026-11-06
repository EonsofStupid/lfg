from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import classifier_comparison as comparison


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class ComparisonTests(unittest.TestCase):
    def _corpus(self, root: Path, include_test: bool = False):
        root.mkdir(parents=True)
        tax = {"schema_version": "lfg.encoder-evaluation-taxonomy.v1",
               "intents": ["build", "verify"], "capabilities": ["write"], "risks": ["sensitive"]}
        (root / "taxonomy.json").write_bytes(_encoded(tax))
        rows = {
            "train": [
                {"id": "a", "family": "fa", "text": "screened train text alpha", "intents": ["build", "verify"],
                 "capabilities": ["write"], "risks": [], "verified": False, "source": {"project": "train-project"}},
                {"id": "b", "family": "fb", "text": "screened train text beta", "intents": ["build"],
                 "capabilities": ["write"], "risks": [], "verified": False, "source": {"project": "train-project"}},
            ],
            "validation": [
                {"id": "c", "family": "fc", "text": "screened validation unique", "intents": ["verify"],
                 "capabilities": [], "risks": [], "verified": False, "source": {"project": "validation-project"}},
            ],
        }
        sources, output_digests, output_counts = [], {}, {}
        for split in ("train", "validation"):
            source_bytes, filtered = bytearray(), []
            group = f"{split}-group"
            for item in rows[split]:
                line = _encoded(item)
                source_bytes.extend(line + b"\n")
                filtered.append({"schema_version": "lfg.screened-text-candidate.v1", "id": item["id"],
                                 "family": item["family"], "project": item["source"]["project"],
                                 "project_group": group, "split": split, "text": item["text"],
                                 "candidate_file_sha256": "a" * 64,
                                 "source_record_sha256": hashlib.sha256(line).hexdigest(),
                                 "screened_text_sha256": hashlib.sha256(item["text"].encode()).hexdigest(),
                                 "screen_policy_sha256": "b" * 64, "redaction_counts": {},
                                 "review_status": "unreviewed", "redaction_complete": False})
            source_file = root / f"source-{split}.jsonl"
            source_file.write_bytes(source_bytes)
            source_sha = hashlib.sha256(source_bytes).hexdigest()
            for record in filtered:
                record["candidate_file_sha256"] = source_sha
            sources.append({"project": f"{split}-project", "project_group": group, "split": split,
                            "path": source_file.name})
            screened_bytes = b"".join(_encoded(x) + b"\n" for x in filtered)
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
        manifest = {"schema_version": "lfg.private-corpus-screen-manifest.v1",
                    "plan_sha256": hashlib.sha256(plan_raw).hexdigest(), "policy_sha256": "b" * 64,
                    "screen_code_sha256": "c" * 64,
                    "source_snapshots": [{"sha256": hashlib.sha256((root / s["path"]).read_bytes()).hexdigest(),
                                          "records": output_counts[s["split"]]} for s in sources if s["split"] != "test"],
                    "counts": output_counts, "matched_records": 0,
                    "claims": {"redaction_complete": False, "human_reviewed": False,
                               "original_source_verified": False, "independent_split_verified": False,
                               "promotion_eligible": False}, "outputs": output_digests}
        (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return plan_path

    def _config(self, corpus: Path, plan: Path, **extra):
        return {"schema_version": comparison.SCHEMA, "revision": "test-1", "screened_corpus": str(corpus),
                "label_plan": str(plan), "target_field": "intents", "selected_labels": ["build", "verify"],
                "seed": 17, "max_rows": 100, "max_text_bytes": 1024, "max_source_bytes": 4096,
                "max_total_source_bytes": 8192,
                "max_manifest_bytes": 4096, "max_screened_file_bytes": 8192, "max_features": 100,
                "knn_neighbors": 1, "tree_max_depth": None, "tfidf_ngram_range": [1, 1],
                "logistic_c": 1.0, "logistic_max_iter": 200, "knn_weights": "uniform", "knn_p": 2,
                "tree_criterion": "gini", "near_duplicate_similarity": 0.9, **extra}

    def test_load_config_accepts_inert_extension_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(self._config(Path(tmp), Path(tmp), extension_note={"future": True})))
            config, digest = comparison.load_config(path, 65536)
            self.assertEqual(config["extension_note"], {"future": True})
            self.assertEqual(len(digest), 64)

    def test_test_source_rejected_before_source_file_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screened"
            plan = self._corpus(root, include_test=True)
            with self.assertRaisesRegex(ValueError, "non-development source"):
                comparison.load_screened_corpus(root, plan, 100, 1024, 4096, 8192, 4096, 8192, 0.9)

    def test_complete_multilabel_arrays_preserved_and_missing_rejected(self):
        taxonomy = {"intents": {"build", "verify"}}
        rows = [{"intents": ["build", "verify"], "labels_pending": False, "verified": False},
                {"intents": [], "labels_pending": False, "verified": False}]
        encoder, matrix = comparison._matrix(rows, "intents", ["build", "verify"], taxonomy)
        self.assertEqual(list(encoder.classes_), ["build", "verify"])
        self.assertEqual(matrix.tolist(), [[1, 1], [0, 0]])
        with self.assertRaisesRegex(ValueError, "target arrays"):
            comparison._matrix([{"verified": False}], "intents", ["build"], taxonomy)

    def test_end_to_end_writes_metrics_without_raw_text_or_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screened"
            plan = self._corpus(root)
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(self._config(root, plan)))
            output = Path(tmp) / "report"
            report = comparison.run(config_path, output)
            self.assertEqual(set(report["algorithms"]), {"logistic_regression", "knn", "decision_tree"})
            for item in report["algorithms"].values():
                self.assertEqual(item["status"], "completed", msg=json.dumps(item))
                self.assertEqual(item["fit_rows"], 2)
                self.assertEqual(item["prediction_rows"], 1)
                self.assertEqual(item["metrics"]["examples"], 1)
            self.assertGreater(report["algorithms"]["knn"]["retained_training_features"]["sparse_storage_bytes"], 0)
            self.assertEqual(report["algorithms"]["knn"]["persisted_model_artifact_bytes"], 0)
            encoded = (output / "comparison-report.json").read_text()
            self.assertNotIn("screened train text", encoded)
            self.assertNotIn('"id": "a"', encoded)
            self.assertFalse(report["promotion_eligible"])


if __name__ == "__main__":
    unittest.main()
