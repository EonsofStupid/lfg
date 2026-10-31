import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import train


SCRIPT = Path(__file__).with_name("distill.py")


class TraceAdmissionTests(unittest.TestCase):
    def test_checked_in_synthetic_classifier_examples_do_not_claim_review(self):
        rows, _ = train.load_dataset(train.DATASET)
        self.assertTrue(rows)
        self.assertTrue(all(row["verified"] is False for row in rows))

    def test_template_makes_no_check_or_teacher_claim_and_builder_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            subprocess.run([sys.executable, str(SCRIPT), "init-template", "--output", str(trace)],
                           check=True, capture_output=True, text=True)
            row = json.loads(trace.read_text(encoding="utf-8"))
            self.assertIs(row["checksPassed"], False)
            self.assertIn("placeholder", row["sourceModel"])
            # Even a claimed passed check and --include-failed do not bypass admission.
            row["checksPassed"] = True
            trace.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = subprocess.run([sys.executable, str(SCRIPT), "validate-traces", "--traces", str(trace)],
                                    check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(report.stdout)["declaredChecksPassed"], 1)
            train_output = root / "train.jsonl"
            eval_output = root / "eval.jsonl"
            train_output.write_text("preserve\n", encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT), "build-sft", "--traces", str(trace),
                                     "--train-output", str(train_output), "--eval-output", str(eval_output),
                                     "--include-failed"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("admission are not implemented", result.stderr)
            self.assertEqual(train_output.read_text(encoding="utf-8"), "preserve\n")
            self.assertFalse(eval_output.exists())


if __name__ == "__main__":
    unittest.main()
