"""Fixture-only tests for byte evaluation; no assessment quality claim."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'safetensors', 'transformers'))


@unittest.skipUnless(AVAILABLE, 'torch, transformers and safetensors required')
class ByteEvaluationTests(unittest.TestCase):
    def setUp(self):
        import train_owned_student as base
        import distill_owned_student as distill
        import evaluate_owned_bytes as subject
        import training_inputs
        from safetensors.torch import save_file
        from transformers import GPT2Config, GPT2LMHeadModel
        self.subject = subject
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        training_config = json.loads(Path(base.__file__).with_name('owned-student.example.json').read_text())
        training_config.update(steps=2, batch_size=1, cpu_threads=1)
        training_config['architecture'].update(width=8, layers=1, heads=2, context=8, attention_every=1)
        self.training_config = self.root / 'training.json'
        self.training_config.write_text(json.dumps(training_config))
        self.source = self.root / 'training.txt'
        self.source.write_text('Fixture training source text for testing only.')
        self.validation = self.root / 'validation.txt'
        self.validation.write_text('Another fixture for validation only.')
        self.control = self.root / 'control'
        base.experiment(self.training_config, self.source, self.control)
        self.snapshot = self.root / 'snapshot'
        self.snapshot.mkdir()
        config = GPT2Config(vocab_size=256, n_positions=8, n_embd=8, n_layer=1, n_head=2,
                            bos_token_id=None, eos_token_id=None, tie_word_embeddings=False)
        (self.snapshot / 'config.json').write_text(config.to_json_string())
        model = GPT2LMHeadModel(config)
        save_file({name: tensor.contiguous() for name, tensor in model.state_dict().items()},
                  str(self.snapshot / 'model.safetensors'))
        self.teacher_manifest = self.root / 'teacher-manifest.json'
        self.revision = 'a' * 40
        self.teacher_manifest.write_bytes(training_inputs.model_manifest(self.snapshot, 'fixture/model', self.revision))
        self.distill_config = self.root / 'distill.json'
        self.distill_config.write_text(Path(distill.__file__).with_name('owned-distillation.example.json').read_text())
        self.candidate = self.root / 'candidate'
        distill.experiment(self.training_config, self.distill_config, self.source, self.teacher_manifest,
                           self.snapshot, self.candidate, 'gpt2-byte', self.revision)
        evaluation_config = json.loads(Path(subject.__file__).with_name('owned-byte-evaluation.example.json').read_text())
        evaluation_config.update(context=8, cpu_threads=1, warmup_windows=0)
        self.config = self.root / 'evaluation.json'
        self.config.write_text(json.dumps(evaluation_config))

    def run_evaluation(self, source, output):
        return self.subject.evaluate(self.config, self.training_config, source, output,
            self.candidate / 'manifest.json', self.candidate / 'weights.safetensors',
            self.control / 'manifest.json', self.control / 'weights.safetensors',
            self.teacher_manifest, self.snapshot, self.revision)

    def test_evaluates_every_next_byte_and_keeps_text_private(self):
        output = self.root / 'report.json'
        report = self.run_evaluation(self.validation, output)
        length = len(self.validation.read_bytes())
        self.assertEqual(report['predicted_bytes'], length - 1)
        self.assertEqual(report['windows'], (length - 1 + 7) // 8)
        self.assertEqual(set(report['results']), {'initial', 'cross_entropy', 'distilled', 'teacher'})
        self.assertTrue(all(item['bits_per_byte'] > 0 for item in report['results'].values()))
        self.assertFalse(report['assessment_evaluated'])
        self.assertFalse(report['promotion_eligible'])
        self.assertNotIn(self.validation.read_text(), output.read_text())

    def test_training_source_and_tampered_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'different source'):
            self.run_evaluation(self.source, self.root / 'same.json')
        weights = self.candidate / 'weights.safetensors'
        weights.write_bytes(weights.read_bytes() + b'tamper')
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.run_evaluation(self.validation, self.root / 'changed.json')
        self.assertFalse((self.root / 'changed.json').exists())


if __name__ == '__main__':
    unittest.main()
