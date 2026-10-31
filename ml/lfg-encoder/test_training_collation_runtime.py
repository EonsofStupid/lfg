"""Optional integration test against an explicitly selected local tokenizer.

This exercises batch assembly only; its labels are test fixtures, not training data.
"""
import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import train


@unittest.skipUnless(os.environ.get("LFG_TEST_LOCAL_TOKENIZER"), "no local tokenizer selected")
class TrainingCollationRuntimeTests(unittest.TestCase):
    def test_trainer_rejects_overflow_before_creating_output(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(os.environ["LFG_TEST_LOCAL_TOKENIZER"], local_files_only=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"id": str(i), "text": "Review the requested change", "family": f"family-{i}",
                     "intents": ["respond.general"], "capabilities": ["text.generate"],
                     "risks": [], "verified": False} for i in range(2)]
            dataset = root / "fixture.jsonl"
            dataset.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
            args = SimpleNamespace(dataset=dataset, allow_below_gate=True, seed=1,
                                   model_revision="a" * 40, base_model="local-fixture",
                                   device="cuda", max_length=2, output=root / "run")
            fake_args = SimpleNamespace(device=SimpleNamespace(type="cuda"))
            with patch.object(train, "make_training_arguments", return_value=(fake_args, True)), \
                    patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer), \
                    self.assertRaisesRegex(SystemExit, "token coverage gate blocked"):
                train.train(args)
            self.assertFalse(args.output.exists())

    def test_real_tokenizer_preflight_and_batched_training_encoding(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(os.environ["LFG_TEST_LOCAL_TOKENIZER"], local_files_only=True)
        short = "Review the requested change"
        long = "review " * 600
        rows = [{"text": short}, {"text": long}]
        encoded = train.tokenize_complete_batch({"text": [short, long]}, tokenizer, 8192)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        self.assertGreater(lengths[1], lengths[0])
        self.assertEqual(train.token_coverage(rows, tokenizer, lengths[1], "train")["maximum_tokens"], lengths[1])
        with self.assertRaisesRegex(ValueError, "token coverage gate blocked"):
            train.token_coverage(rows, tokenizer, lengths[1] - 1, "train")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            train.tokenize_complete_batch({"text": [short, long]}, tokenizer, lengths[1] - 1)
        tokenizer.model_max_length = lengths[1] - 1
        with self.assertRaisesRegex(ValueError, "effective maximum"):
            train.token_coverage(rows, tokenizer, lengths[1], "train")
        with self.assertRaisesRegex(ValueError, "effective maximum"):
            train.tokenize_complete_batch({"text": [short, long]}, tokenizer, lengths[1])

    def test_variable_lengths_work_in_training_and_validation_loaders(self):
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments

        tokenizer = AutoTokenizer.from_pretrained(os.environ["LFG_TEST_LOCAL_TOKENIZER"],
                                                  local_files_only=True)
        # Texts are the LFG kernel golden request and an authored boundary question.
        encoded = tokenizer(["logout doesn't clear the session",
                             "Is coupling to other modules limited to their explicit interfaces (no reaching into internals)?"],
                            truncation=True, max_length=192)
        lengths = [len(tokens) for tokens in encoded["input_ids"]]
        self.assertNotEqual(*lengths)
        data = Dataset.from_dict({**encoded, "labels": [[1., 0.], [0., 1.]]})

        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, input_ids=None, attention_mask=None, labels=None):
                return {"loss": self.weight.sum() * 0}

        with tempfile.TemporaryDirectory() as directory:
            # This is a CPU-only data-loader test, not a training run.
            args = TrainingArguments(output_dir=directory, use_cpu=True,
                                     per_device_train_batch_size=2, per_device_eval_batch_size=2,
                                     report_to="none")
            unpadded = Trainer(model=FixtureModel(), args=args, train_dataset=data)
            with self.assertRaises((ValueError, RuntimeError)):
                next(iter(unpadded.get_train_dataloader()))
            padded = train.make_trainer(FixtureModel, args, data, data, tokenizer,
                                        Trainer, DataCollatorWithPadding, TrainerCallback)
            for loader in (padded.get_train_dataloader(), padded.get_eval_dataloader()):
                batch = next(iter(loader))
                self.assertEqual(list(batch["input_ids"].shape), [2, max(lengths)])
                self.assertEqual(list(batch["labels"].shape), [2, 2])
                self.assertEqual(sorted(batch["attention_mask"].sum(dim=1).tolist()), sorted(lengths))


if __name__ == "__main__":
    unittest.main()
