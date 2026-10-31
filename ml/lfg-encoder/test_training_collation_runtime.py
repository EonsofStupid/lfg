"""Optional integration test against an explicitly selected local tokenizer.

This exercises batch assembly only; its labels are test fixtures, not training data.
"""
import os
import tempfile
import unittest

import train


@unittest.skipUnless(os.environ.get("LFG_TEST_LOCAL_TOKENIZER"), "no local tokenizer selected")
class TrainingCollationRuntimeTests(unittest.TestCase):
    def test_variable_lengths_work_in_training_and_validation_loaders(self):
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

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
                                        Trainer, DataCollatorWithPadding)
            for loader in (padded.get_train_dataloader(), padded.get_eval_dataloader()):
                batch = next(iter(loader))
                self.assertEqual(list(batch["input_ids"].shape), [2, max(lengths)])
                self.assertEqual(list(batch["labels"].shape), [2, 2])
                self.assertEqual(sorted(batch["attention_mask"].sum(dim=1).tolist()), sorted(lengths))


if __name__ == "__main__":
    unittest.main()
