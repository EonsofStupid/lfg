import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset.jsonl"
        self.predictions = self.root / "predictions.jsonl"
        self.taxonomy = self.root / "taxonomy.json"
        self.taxonomy.write_text(json.dumps({"schema_version": "lfg.encoder-evaluation-taxonomy.v1",
                                             "intents": ["build", "other", "unused"],
                                             "capabilities": ["write"], "risks": ["sensitive"]}), encoding="utf-8")
        self.rows = [
            {"id": "one", "text": "private prompt A", "family": "a", "verified": True,
             "intents": ["build"], "capabilities": ["write"], "risks": []},
            {"id": "two", "text": "private prompt B", "family": "b", "verified": True,
             "intents": ["build"], "capabilities": [], "risks": ["sensitive"]},
        ]
        self.model = {"id": "example", "version": "1", "sha256": "a" * 64}
        self.preds = [
            {"schema_version": evaluate.PREDICTION_SCHEMA, "id": "one", "model": self.model,
             "producer_sha256": "c" * 64,
             "input_sha256": hashlib.sha256(b"private prompt A").hexdigest(),
             "intents": ["build"], "capabilities": [], "risks": ["sensitive"]},
            {"schema_version": evaluate.PREDICTION_SCHEMA, "id": "two", "model": self.model,
             "producer_sha256": "c" * 64,
             "input_sha256": hashlib.sha256(b"private prompt B").hexdigest(),
             "intents": ["other"], "capabilities": [], "risks": []},
        ]
        self.write()

    def write(self):
        self.dataset.write_text("\n".join(map(json.dumps, self.rows)), encoding="utf-8")
        self.predictions.write_text("\n".join(map(json.dumps, self.preds)), encoding="utf-8")

    def test_counts_and_undefined_metrics_are_truthful(self):
        report = evaluate.evaluate(self.dataset, self.predictions, self.taxonomy, None, False)
        intent = report["groups"]["intents"]["labels"]
        self.assertEqual(intent["build"]["tp"], 1)
        self.assertEqual(intent["build"]["fn"], 1)
        self.assertEqual(intent["other"]["fp"], 1)
        self.assertIsNone(intent["other"]["recall"])
        self.assertAlmostEqual(report["groups"]["intents"]["macro_f1"], 2 / 3)
        self.assertAlmostEqual(report["groups"]["intents"]["micro_f1"], 0.5)
        self.assertIsNone(intent["unused"]["f1"])
        risk = report["groups"]["risks"]["labels"]["sensitive"]
        self.assertEqual((risk["tp"], risk["fp"], risk["fn"]), (0, 1, 1))
        self.assertFalse(report["promotion_eligible"])
        self.assertNotIn("private prompt", json.dumps(report))

    def test_rejects_unverified_and_missing_or_mixed_predictions(self):
        self.rows[0]["verified"] = False
        self.write()
        with self.assertRaisesRegex(ValueError, "unverified"):
            evaluate.evaluate(self.dataset, self.predictions, self.taxonomy, None, False)
        self.assertEqual(evaluate.evaluate(self.dataset, self.predictions, self.taxonomy, None, True)["unverified_examples"], 1)
        self.preds[1]["model"] = {**self.model, "version": "different"}
        self.write()
        with self.assertRaisesRegex(ValueError, "multiple model identities"):
            evaluate.evaluate(self.dataset, self.predictions, self.taxonomy, None, True)
        self.preds.pop()
        self.write()
        with self.assertRaisesRegex(ValueError, "exactly match"):
            evaluate.evaluate(self.dataset, self.predictions, self.taxonomy, None, True)

    def test_rejects_duplicate_labels_and_overlapping_families(self):
        self.preds[0]["intents"] = ["build", "build"]
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate labels"):
            evaluate.evaluate(self.dataset, self.predictions, self.taxonomy, None, False)
        self.preds[0]["intents"] = ["build"]
        self.write()
        split = self.root / "split.json"
        split.write_text(json.dumps({"schema_version": "lfg.encoder-development-split.v1",
                                    "train_ids": ["elsewhere"], "validation_ids": [],
                                    "train_families": ["a"], "validation_families": []}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate.evaluate(self.dataset, self.predictions, self.taxonomy, split, False)

    def test_rejects_predictions_for_changed_text(self):
        self.rows[0]["text"] = "a different prompt"
        self.write()
        with self.assertRaisesRegex(ValueError, "input text does not match"):
            evaluate.evaluate(self.dataset, self.predictions, self.taxonomy, None, False)

    def test_rejects_wrong_taxonomy_and_unknown_labels(self):
        for prediction in self.preds:
            prediction["model"] = {**self.model, "taxonomy": "b" * 64}
        self.write()
        with self.assertRaisesRegex(ValueError, "taxonomy differs"):
            evaluate.evaluate(self.dataset, self.predictions, self.taxonomy, None, False)
        for prediction in self.preds:
            prediction["model"] = self.model
        self.preds[0]["intents"] = ["unrecognized"]
        self.write()
        with self.assertRaisesRegex(ValueError, "unknown intents"):
            evaluate.evaluate(self.dataset, self.predictions, self.taxonomy, None, False)


if __name__ == "__main__":
    unittest.main()
