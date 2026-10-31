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

    def test_verified_count_alone_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "examples.jsonl"
            dataset.write_text("\n".join(json.dumps(record(i, True)) for i in range(500)), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "audited-plan is required"):
                train.train(argparse.Namespace(dataset=dataset, allow_below_gate=False, audited_plan=None))

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

    def test_validation_cannot_silently_drop_a_label_absent_from_training(self):
        training = [record(1, True)]
        validation = [dict(record(2, True), risks=["sensitive-data"])]
        with self.assertRaisesRegex(ValueError, "labels absent from the training partition"):
            train.validate_validation_labels(training, validation)
        training[0]["risks"] = ["sensitive-data"]
        train.validate_validation_labels(training, validation)

    def test_reviewed_support_checks_positive_and_negative_per_label_and_split(self):
        taxonomy = {"intents": {"respond.general"}, "capabilities": {"text.generate"}, "risks": {"sensitive-data"}}
        policy = train.SUPPORT_POLICY.read_bytes()
        positive = dict(record(1, True), risks=["sensitive-data"])
        negative = record(2, True)
        negative["intents"] = []
        negative["capabilities"] = []
        rows = [positive, negative]
        result = train.training_support(rows, rows, taxonomy, policy)
        self.assertEqual(result["counts"]["train"]["sensitive-data"]["positive"], 1)
        with self.assertRaisesRegex(ValueError, "train:sensitive-data has 0 positive/2 negative"):
            train.training_support([negative, negative], rows, taxonomy, policy)
        with self.assertRaisesRegex(ValueError, "validation:sensitive-data has 2 positive/0 negative"):
            train.training_support(rows, [positive, positive], taxonomy, policy)

    def test_support_policy_is_configurable_and_rejects_unknown_or_invalid_rules(self):
        taxonomy = {"intents": {"respond.general"}, "capabilities": {"text.generate"}, "risks": {"sensitive-data"}}
        policy = json.loads(train.SUPPORT_POLICY.read_bytes())
        policy["overrides"] = {"sensitive-data": {"train": {"minimum_positive": 2}}}
        positive = dict(record(1, True), risks=["sensitive-data"])
        negative = dict(record(2, True), intents=[], capabilities=[])
        result = train.training_support([positive, positive, negative], [positive, negative], taxonomy, json.dumps(policy).encode())
        self.assertEqual(result["counts"]["train"]["sensitive-data"]["minimum_positive"], 2)
        for overrides in ({"missing": {"train": {"minimum_positive": 0}}},
                          {"sensitive-data": {"test": {"minimum_positive": 0}}},
                          {"sensitive-data": {"train": {"minimum_positive": True}}},
                          {"sensitive-data": {"train": {"minimum_positive": 0}}},
                          {"sensitive-data": {"train": {"minimum_negative": -1}}}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                train.training_support([positive, negative], [positive, negative], taxonomy,
                                       json.dumps({**policy, "overrides": overrides}).encode())

    def test_run_manifest_prevents_overwrite_and_mismatched_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            manifest = {"dataset_sha256": "original", "seed": 7}
            train.prepare_run(output, manifest, False)
            train.prepare_run(output, manifest, True)
            with self.assertRaisesRegex(ValueError, "not empty"):
                train.prepare_run(output, manifest, False)
            with self.assertRaisesRegex(ValueError, "resume refused"):
                train.prepare_run(output, {**manifest, "dataset_sha256": "changed"}, True)
            with self.assertRaisesRegex(ValueError, "existing run manifest"):
                train.prepare_run(Path(directory) / "missing", manifest, True)

    def test_resume_requires_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            with self.assertRaisesRegex(ValueError, "incomplete checkpoint"):
                train.validate_checkpoint(checkpoint)
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
                (checkpoint / name).write_bytes(b"test")
            (checkpoint / "trainer_state.json").write_text('{"global_step": 1}', encoding="utf-8")
            train.validate_checkpoint(checkpoint)
            (checkpoint / "trainer_state.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid trainer state"):
                train.validate_checkpoint(checkpoint)

    def test_run_lock_rejects_concurrent_writers_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with train.run_lock(output):
                with self.assertRaisesRegex(ValueError, "run lock exists"):
                    with train.run_lock(output):
                        self.fail("second writer acquired the lock")
            with train.run_lock(output):
                pass

    def test_training_settings_and_overrides_are_validated(self):
        config = train.load_training_config(train.TRAINING_CONFIG)
        train.validate_training_config(config)
        for field, invalid in (("batch_size", 0), ("checkpoint_steps", -1),
                               ("epochs", float("nan")), ("base_model", ""),
                               ("allow_mismatched_sizes", "true")):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    train.validate_training_config({**config, field: invalid})

    def test_promotion_policy_rejects_invalid_sample_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            policy = json.loads(train.POLICY.read_text(encoding="utf-8"))
            policy["minimum_human_verified_examples"] = 0
            path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "minimum_human_verified_examples"):
                train.load_promotion_policy(path)


if __name__ == "__main__":
    unittest.main()
