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


if __name__ == "__main__":
    unittest.main()
