"""Standard-library tests for the source-controlled LFG training gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("train.py")
SPEC = importlib.util.spec_from_file_location("lfg_encoder_train", MODULE_PATH)
assert SPEC and SPEC.loader
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


def _record(record_id: str, *, verified: bool = True) -> dict[str, object]:
    return {
        "id": record_id,
        "text": f"Example {record_id}",
        "intents": ["respond.general"],
        "capabilities": ["text.generate"],
        "risks": [],
        "verified": verified,
        "family": "test",
    }


class TrainingGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset.jsonl"
        self.policy = self.root / "policy.json"
        self.policy.write_text(
            json.dumps({"minimum_human_verified_examples": 2}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_records(self, *records: dict[str, object]) -> None:
        self.dataset.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    def test_only_human_verified_examples_count_toward_gate(self) -> None:
        self.write_records(_record("one"), _record("two", verified=False))

        report = TRAIN.dataset_report(self.dataset, self.policy)

        self.assertEqual(report["records"], 2)
        self.assertEqual(report["verified"], 1)
        self.assertFalse(report["promotion_ready"])

    def test_duplicate_ids_are_rejected(self) -> None:
        self.write_records(_record("same"), _record("same"))

        with self.assertRaisesRegex(ValueError, "duplicate id"):
            TRAIN.load_records(self.dataset)

    def test_plan_is_gpu_first_pinned_and_blocked_below_gate(self) -> None:
        self.write_records(_record("one"))
        original_policy = TRAIN.POLICY
        TRAIN.POLICY = self.policy
        try:
            args = argparse.Namespace(
                dataset=self.dataset,
                development_run=False,
                engine="unsloth",
                base_model=TRAIN.BASE_MODEL,
                revision=TRAIN.BASE_REVISION,
                output=self.root / "output",
                max_length=512,
                epochs=3.0,
                batch_size=8,
                gradient_accumulation_steps=1,
                learning_rate=5e-5,
                threshold=0.5,
            )
            plan = TRAIN.training_plan(args)
        finally:
            TRAIN.POLICY = original_policy

        self.assertEqual(plan["engine"], "unsloth")
        self.assertEqual(plan["training_device"], "cuda-required")
        self.assertEqual(plan["base_revision"], TRAIN.BASE_REVISION)
        self.assertFalse(plan["training_allowed"])
        self.assertFalse(plan["promotion_eligible_after_training"])


if __name__ == "__main__":
    unittest.main()
