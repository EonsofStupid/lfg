import json
import tempfile
import unittest
from pathlib import Path

import corpus_audit


class CorpusAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.taxonomy = self.root / "taxonomy.json"
        self.taxonomy.write_text(json.dumps({"schema_version": "lfg.encoder-evaluation-taxonomy.v1",
                                             "intents": ["build"], "capabilities": ["write"],
                                             "risks": ["sensitive"]}), encoding="utf-8")
        self.rows = [
            {"id": "a", "text": "private input A", "family": "family-a", "verified": False,
             "intents": ["build"], "capabilities": ["write"], "risks": [], "source": {"project": "alpha"}},
            {"id": "b", "text": "private input B", "family": "family-b", "verified": False,
             "intents": ["build"], "capabilities": [], "risks": [], "source": {"project": "beta"}},
        ]
        self.plan = {"schema_version": corpus_audit.PLAN_SCHEMA, "taxonomy": "taxonomy.json", "review_ledger": None,
                     "sources": [{"path": "a.jsonl", "project": "alpha", "split": "train"},
                                 {"path": "b.jsonl", "project": "beta", "split": "test"}]}
        self.write()

    def write(self):
        for name, row in zip(("a.jsonl", "b.jsonl"), self.rows, strict=True):
            (self.root / name).write_text(json.dumps(row) + "\n", encoding="utf-8")
        (self.root / "plan.json").write_text(json.dumps(self.plan), encoding="utf-8")

    def test_draft_rows_stay_unapproved_and_private(self):
        report, queue = corpus_audit.audit(self.root / "plan.json")
        self.assertEqual(report["counts"]["train"]["examples"], 1)
        self.assertEqual(report["counts"]["train"]["ready_for_reviewed_subset"], 0)
        self.assertEqual({row["decision"] for row in queue}, {"pending"})
        self.assertFalse(report["promotion_eligible"])
        self.assertNotIn("private input", json.dumps(report) + json.dumps(queue))

    def test_attestation_is_bound_to_text_labels_taxonomy_and_split(self):
        _, queue = corpus_audit.audit(self.root / "plan.json")
        approval = {**queue[0], "decision": "approved", "reviewer": "reviewer-1", "reviewed_at": "2026-10-30T12:00:00+00:00"}
        (self.root / "reviews.jsonl").write_text(json.dumps(approval), encoding="utf-8")
        self.plan["review_ledger"] = "reviews.jsonl"
        self.rows[0]["verified"] = True
        self.write()
        report, _ = corpus_audit.audit(self.root / "plan.json")
        self.assertEqual(report["counts"]["train"]["ready_for_reviewed_subset"], 1)
        self.rows[0]["text"] = "changed text"
        self.write()
        report, _ = corpus_audit.audit(self.root / "plan.json")
        self.assertEqual(report["counts"]["train"]["ready_for_reviewed_subset"], 0)

    def test_rejects_declared_project_family_and_text_leakage(self):
        self.plan["sources"][1]["project"] = "alpha"
        self.write()
        with self.assertRaisesRegex(ValueError, "project appears"):
            corpus_audit.audit(self.root / "plan.json")
        self.plan["sources"][1]["project"] = "beta"
        self.rows[1]["family"] = "family-a"
        self.write()
        with self.assertRaisesRegex(ValueError, "family appears"):
            corpus_audit.audit(self.root / "plan.json")
        self.rows[1]["family"] = "family-b"
        self.rows[1]["text"] = "  PRIVATE  input a  "
        self.write()
        with self.assertRaisesRegex(ValueError, "normalized text"):
            corpus_audit.audit(self.root / "plan.json")

    def test_rejects_provenance_and_taxonomy_mismatch(self):
        self.rows[1]["source"]["project"] = "wrong"
        self.write()
        with self.assertRaisesRegex(ValueError, "source project"):
            corpus_audit.audit(self.root / "plan.json")
        self.rows[1]["source"]["project"] = "beta"
        self.rows[1]["intents"] = ["unknown"]
        self.write()
        with self.assertRaisesRegex(ValueError, "outside the configured taxonomy"):
            corpus_audit.audit(self.root / "plan.json")


if __name__ == "__main__":
    unittest.main()
