import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import train


def record(index, verified):
    return {"id": str(index), "text": "Explain the task", "intents": ["respond.general"],
            "capabilities": ["text.generate"], "risks": [], "verified": verified,
            "family": "general"}


class DatasetGateTests(unittest.TestCase):
    def test_unverified_records_do_not_satisfy_sample_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "examples.jsonl"
            dataset.write_text("\n".join(json.dumps(record(i, False)) for i in range(500)), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                train.validate(argparse.Namespace(dataset=dataset))
            report = json.loads(output.getvalue())
            self.assertEqual(report["verified"], 0)
            self.assertFalse(report["sample_gate_passed"])
            with self.assertRaisesRegex(SystemExit, "0 verified examples"):
                train.train(argparse.Namespace(dataset=dataset, allow_below_gate=False))

    def test_verified_must_be_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "examples.jsonl"
            dataset.write_text(json.dumps(record(1, "false")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verified must be a boolean"):
                train.load_records(dataset)

    def test_only_verified_rows_are_selected_without_development_override(self):
        rows = [record(1, True), record(2, False)]
        self.assertEqual(train.select_training_records(rows, 1, False), [rows[0]])
        self.assertEqual(train.select_training_records(rows, 1, True), rows)

    def test_split_keeps_families_together_and_is_repeatable(self):
        rows = [dict(record(i, True), family=f"family-{i // 3}") for i in range(15)]
        training, validation = train.grouped_split(rows, 731)
        self.assertEqual(train.grouped_split(rows, 731), (training, validation))
        self.assertTrue(training)
        self.assertTrue(validation)
        self.assertTrue({r["family"] for r in training}.isdisjoint({r["family"] for r in validation}))
        self.assertEqual({r["id"] for r in training + validation}, {r["id"] for r in rows})

    def test_split_rejects_one_family(self):
        with self.assertRaisesRegex(ValueError, "at least two distinct families"):
            train.grouped_split([record(1, True), record(2, True)], 731)


if __name__ == "__main__":
    unittest.main()
