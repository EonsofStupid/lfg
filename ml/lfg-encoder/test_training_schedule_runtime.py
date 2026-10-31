"""Optional Transformers integration test; fixture models and labels are not training data."""
import os
import tempfile
import unittest
from pathlib import Path

import train


@unittest.skipUnless(os.environ.get("LFG_TEST_TRANSFORMERS_RUNTIME") == "1",
                     "no Transformers runtime selected")
class TrainingScheduleRuntimeTests(unittest.TestCase):
    def test_final_checkpoint_is_evaluated_even_between_intervals(self):
        import torch
        from datasets import Dataset
        from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator

        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, input_ids=None, labels=None):
                return {"loss": self.weight.square().mean(),
                        "logits": self.weight.expand(input_ids.shape[0], 1)}

        data = Dataset.from_dict({"input_ids": [[1, 2], [3, 4]], "labels": [[1.], [0.]]})
        for steps, interval, expected in ((1, 100, [1]), (3, 2, [2, 3]), (2, 2, [2])):
            with self.subTest(steps=steps, interval=interval), tempfile.TemporaryDirectory() as directory:
                args = TrainingArguments(output_dir=directory, use_cpu=True, max_steps=steps,
                                         per_device_train_batch_size=2, per_device_eval_batch_size=2,
                                         eval_strategy="steps", eval_steps=interval,
                                         save_strategy="steps", save_steps=interval,
                                         load_best_model_at_end=True, save_only_model=False,
                                         report_to="none", disable_tqdm=True)
                trainer = train.make_trainer(FixtureModel, args, data, data, object(), Trainer,
                                             lambda tokenizer: default_data_collator, TrainerCallback)
                trainer.train()
                self.assertEqual([row["step"] for row in trainer.state.log_history if "eval_loss" in row], expected)
                train.validate_selected_checkpoint(trainer.state)
                self.assertEqual(Path(trainer.state.best_model_checkpoint).name, f"checkpoint-{steps}")
                self.assertTrue((Path(directory) / f"checkpoint-{steps}").is_dir())
                trainer.save_state()
                self.assertTrue((Path(directory) / "trainer_state.json").is_file())


if __name__ == "__main__":
    unittest.main()
