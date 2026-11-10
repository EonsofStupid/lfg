import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_label_guide_provenance_distinguishes_missing_explicit_none_and_verified_bytes(self):
        report, _ = corpus_audit.audit(self.root / "plan.json")
        self.assertEqual(report["label_guide_status"], "not_declared")
        self.assertIsNone(report["label_guide_sha256"])

        self.plan["label_guide"] = None
        self.write()
        report, _ = corpus_audit.audit(self.root / "plan.json")
        self.assertEqual(report["label_guide_status"], "explicitly_none")
        self.assertIsNone(report["label_guide_sha256"])

        labels = {"intents": {"build": {"apply": "build", "boundary": "no build"}},
                  "capabilities": {"write": {"apply": "write", "boundary": "no write"}},
                  "risks": {"sensitive": {"apply": "sensitive", "boundary": "not sensitive"}}}
        guide = {"schema_version": "lfg.review-label-guide.v1", "revision": "test-1",
                 "notice": "Review uncertain labels.", "labels": labels}
        raw = json.dumps(guide, sort_keys=True).encode()
        (self.root / "guide.json").write_bytes(raw)
        self.plan["label_guide"] = "guide.json"
        self.write()
        report, _ = corpus_audit.audit(self.root / "plan.json")
        self.assertEqual(report["label_guide_status"], "verified_sha256")
        self.assertEqual(report["label_guide_sha256"], __import__("hashlib").sha256(raw).hexdigest())

    def test_label_guide_reread_is_bounded_when_file_grows(self):
        labels = {"intents": {"build": {"apply": "build", "boundary": "no build"}},
                  "capabilities": {"write": {"apply": "write", "boundary": "no write"}},
                  "risks": {"sensitive": {"apply": "sensitive", "boundary": "not sensitive"}}}
        guide = {"schema_version": "lfg.review-label-guide.v1", "revision": "test-1",
                 "notice": "Review uncertain labels.", "labels": labels}
        raw = json.dumps(guide, sort_keys=True).encode()
        taxonomy = {"intents": {"build"}, "capabilities": {"write"}, "risks": {"sensitive"}}
        requested = []

        class BoundedStream(io.BytesIO):
            def read(self, size=-1):
                requested.append(size)
                return super().read(size)

        oversized = raw + b" " * ((4 * 1024 * 1024) + 1)
        with patch.object(Path, "open", side_effect=[BoundedStream(raw), BoundedStream(oversized)]):
            with self.assertRaisesRegex(ValueError, "audit bound"):
                corpus_audit._label_guide_identity(
                    self.root / "plan.json", {"label_guide": "guide.json"}, taxonomy)
        self.assertEqual(requested, [(4 * 1024 * 1024) + 1, (4 * 1024 * 1024) + 1])

    def test_invalid_or_taxonomy_mismatched_label_guide_is_rejected(self):
        self.plan["label_guide"] = "guide.json"
        guide = {"schema_version": "lfg.review-label-guide.v1", "revision": "test-1",
                 "notice": "Review uncertain labels.",
                 "labels": {"intents": {"other": {"apply": "x", "boundary": "y"}},
                            "capabilities": {"write": {"apply": "x", "boundary": "y"}},
                            "risks": {"sensitive": {"apply": "x", "boundary": "y"}}}}
        (self.root / "guide.json").write_text(json.dumps(guide), encoding="utf-8")
        self.write()
        with self.assertRaisesRegex(ValueError, "does not match"):
            corpus_audit.audit(self.root / "plan.json")

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

    def test_related_projects_must_stay_in_the_same_split(self):
        for source in self.plan["sources"]:
            source["project_group"] = "shared-product"
        self.write()
        with self.assertRaisesRegex(ValueError, "project group appears in more than one split"):
            corpus_audit.audit(self.root / "plan.json")
        self.plan["sources"][1]["split"] = "train"
        self.write()
        report, _ = corpus_audit.audit(self.root / "plan.json")
        self.assertEqual([source["project_group"] for source in report["sources"]], ["shared-product", "shared-product"])
        self.plan["sources"][1]["project"] = "alpha"
        self.plan["sources"][1]["project_group"] = "different-group"
        self.write()
        with self.assertRaisesRegex(ValueError, "inconsistent project groups"):
            corpus_audit.audit(self.root / "plan.json")

    def test_project_group_rejects_empty_value(self):
        self.plan["sources"][0]["project_group"] = " "
        self.write()
        with self.assertRaisesRegex(ValueError, "project_group must be"):
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

    def test_training_gate_requires_all_three_reviewed_partitions_and_exact_train_data(self):
        extra = {"id": "c", "text": "independent review", "family": "family-c", "verified": False,
                 "intents": ["build"], "capabilities": [], "risks": [], "source": {"project": "gamma"}}
        (self.root / "c.jsonl").write_text(json.dumps(extra), encoding="utf-8")
        self.plan["sources"].append({"path": "c.jsonl", "project": "gamma", "split": "validation"})
        self.write()
        _, queue = corpus_audit.audit(self.root / "plan.json")
        approvals = [{**item, "decision": "approved", "reviewer": "reviewer-1",
                      "reviewed_at": "2026-10-30T12:00:00Z"} for item in queue]
        (self.root / "reviews.jsonl").write_text("\n".join(map(json.dumps, approvals)), encoding="utf-8")
        self.plan["review_ledger"] = "reviews.jsonl"
        for row in self.rows:
            row["verified"] = True
        extra["verified"] = True
        (self.root / "c.jsonl").write_text(json.dumps(extra), encoding="utf-8")
        self.write()
        corpus_audit.validate_training_corpus([self.rows[0]], self.root / "plan.json")
        _, validation = corpus_audit.load_reviewed_validation([self.rows[0]], self.root / "plan.json")
        self.assertEqual([row["id"] for row in validation], ["c"])
        snapshot, _ = corpus_audit.assemble_training(self.root / "plan.json")
        self.assertEqual([json.loads(line)["id"] for line in snapshot.splitlines()], ["a"])
        changed = {**self.rows[0], "text": "not the reviewed text"}
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            corpus_audit.validate_training_corpus([changed], self.root / "plan.json")
        extra["text"] = "changed after review"
        (self.root / "c.jsonl").write_text(json.dumps(extra), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "validation example requires"):
            corpus_audit.load_reviewed_validation([self.rows[0]], self.root / "plan.json")

    def test_development_plan_rejects_test_before_opening_sources(self):
        self.plan['schema_version'] = corpus_audit.DEVELOPMENT_PLAN_SCHEMA
        self.write()
        with patch.object(corpus_audit.train, 'load_dataset', side_effect=AssertionError('source opened')):
            with self.assertRaisesRegex(ValueError, 'only train and validation'):
                corpus_audit.audit(self.root / 'plan.json')

    def test_development_only_training_rejects_complete_plan_before_reading_sources(self):
        with patch.object(corpus_audit.train, 'load_dataset', side_effect=AssertionError('source opened')):
            with patch.object(corpus_audit.evaluate, 'load_taxonomy', side_effect=AssertionError('taxonomy opened')):
                with self.assertRaisesRegex(ValueError, 'development-only training'):
                    corpus_audit.load_reviewed_validation([], self.root / 'plan.json', development_only=True)

    def test_development_training_uses_reviewed_train_and_validation_only(self):
        self.plan['schema_version'] = corpus_audit.DEVELOPMENT_PLAN_SCHEMA
        self.plan['sources'][1]['split'] = 'validation'
        self.write()
        _, queue = corpus_audit.audit(self.root / 'plan.json')
        approvals = [{**item, 'decision': 'approved', 'reviewer': 'test-reviewer',
                      'reviewed_at': '2026-10-30T12:00:00Z'} for item in queue]
        (self.root / 'reviews.jsonl').write_text('\n'.join(map(json.dumps, approvals)), encoding='utf-8')
        self.plan['review_ledger'] = 'reviews.jsonl'
        for row in self.rows:
            row['verified'] = True
        self.write()
        payload, report = corpus_audit.assemble_training(self.root / 'plan.json')
        self.assertEqual([json.loads(line)['id'] for line in payload.splitlines()], ['a'])
        self.assertEqual(report['scope'], 'development')
        self.assertEqual(report['holdout_assessment'], 'not_performed')
        self.assertFalse(report['promotion_eligible'])
        self.assertEqual(corpus_audit.load_reviewed_validation([self.rows[0]], self.root / 'plan.json',
                                                              development_only=True)[1][0]['id'], 'b')
        # A plan replaced between validation loading and the second audit must
        # fail before the newly declared test source can be opened.
        original_load = corpus_audit.train.load_dataset
        calls = []
        def replace_plan_after_validation(path, *args, **kwargs):
            calls.append(Path(path).name)
            result = original_load(path, *args, **kwargs)
            if len(calls) == 3:
                self.plan['schema_version'] = corpus_audit.PLAN_SCHEMA
                self.plan['sources'].append({'path': 'heldout.jsonl', 'project': 'holdout', 'split': 'test'})
                (self.root / 'plan.json').write_text(json.dumps(self.plan), encoding='utf-8')
            return result
        with patch.object(corpus_audit.train, 'load_dataset', side_effect=replace_plan_after_validation):
            with self.assertRaisesRegex(ValueError, 'development-only training'):
                corpus_audit.load_reviewed_validation([self.rows[0]], self.root / 'plan.json', development_only=True)
        self.assertNotIn('heldout.jsonl', calls)
        self.plan['schema_version'] = corpus_audit.DEVELOPMENT_PLAN_SCHEMA
        self.plan['sources'].pop()
        self.rows[1]['text'] = 'changed after review'
        self.write()
        with self.assertRaisesRegex(ValueError, 'validation example requires'):
            corpus_audit.assemble_training(self.root / 'plan.json')


if __name__ == "__main__":
    unittest.main()
