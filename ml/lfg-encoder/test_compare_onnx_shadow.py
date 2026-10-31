import unittest

import compare_onnx_shadow


def row(identifier, intent):
    return {"schema_version": "lfg.evaluation-prediction.v1", "id": identifier,
            "input_sha256": "a" * 64, "intents": [intent], "capabilities": [], "risks": []}


class ShadowComparisonTests(unittest.TestCase):
    def test_counts_decision_disagreements_without_exposing_examples(self):
        result = compare_onnx_shadow.compare([row("one", "build"), row("two", "other")],
                                             [row("two", "build"), row("one", "build")])
        self.assertEqual(result, {"examples": 2, "decision_mismatches": 1})

    def test_rejects_wrong_or_duplicate_inputs(self):
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            compare_onnx_shadow.compare([row("one", "build")], [row("two", "build")])
        with self.assertRaisesRegex(ValueError, "unique"):
            compare_onnx_shadow.compare([row("one", "build"), row("one", "build")], [row("one", "build")])
        altered = {**row("one", "build"), "input_sha256": "b" * 64}
        with self.assertRaisesRegex(ValueError, "digests differ"):
            compare_onnx_shadow.compare([row("one", "build")], [altered])


if __name__ == "__main__":
    unittest.main()
