"""Fixture-only integration tests; no task-quality or external teacher claim."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

AVAILABLE = bool(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'))


@unittest.skipUnless(AVAILABLE, 'torch and safetensors required')
class OwnedDistillationTests(unittest.TestCase):
    def setUp(self):
        import train_owned_student as base
        import distill_owned_student as subject
        self.base, self.subject = base, subject
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.student_config = json.loads(Path(base.__file__).with_name('owned-student.example.json').read_text())
        self.student_config.update(steps=2, batch_size=1, cpu_threads=1)
        self.student_config['architecture'].update(width=8, layers=1, heads=2, context=8, attention_every=1)
        self.student_path = self.root / 'student-config.json'
        self.student_path.write_text(json.dumps(self.student_config))
        self.distill_config = json.loads(Path(subject.__file__).with_name('owned-distillation.example.json').read_text())
        self.distill_path = self.root / 'distill-config.json'
        self.distill_path.write_text(json.dumps(self.distill_config))
        self.source = self.root / 'source.txt'
        self.source.write_text('real application training input fixture for code tests')
        self.teacher = self.root / 'teacher'
        base.experiment(self.student_path, self.source, self.teacher)
        self.student_config['seed'] += 1
        self.student_path.write_text(json.dumps(self.student_config))

    def run_student(self, name):
        return self.subject.experiment(self.student_path, self.distill_path, self.source,
             self.teacher / 'manifest.json', self.teacher / 'weights.safetensors', self.root / name)

    def test_online_distillation_is_repeatable_and_teacher_unchanged(self):
        before = (self.teacher / 'weights.safetensors').read_bytes()
        first, second = self.run_student('first'), self.run_student('second')
        self.assertEqual(first['weights_sha256'], second['weights_sha256'])
        self.assertEqual(first['losses'], second['losses'])
        self.assertGreater(first['losses'][0]['distillation'], 0)
        self.assertGreater(first['losses'][0]['total'], first['losses'][0]['cross_entropy'])
        self.assertEqual(before, (self.teacher / 'weights.safetensors').read_bytes())
        self.assertTrue(first['teacher_distillation_performed'])
        self.assertFalse(first['teacher']['quality_verified'])
        self.assertFalse(first['validation_performed'])
        self.assertFalse(first['promotion_eligible'])

    def test_topk_uses_full_tail_and_stable_ties(self):
        import torch
        logits = torch.zeros((1, 2, 256))
        indices, top, tail = self.subject.teacher_distribution(logits, 4)
        self.assertEqual(indices[0, 0].tolist(), [0, 1, 2, 3])
        torch.testing.assert_close(top.sum(-1) + tail, torch.ones_like(tail))
        torch.testing.assert_close(tail, torch.full_like(tail, 252 / 256))

    def test_tampered_teacher_is_rejected_before_output(self):
        path = self.teacher / 'weights.safetensors'
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'teacher identity'):
            self.run_student('bad')
        self.assertFalse((self.root / 'bad').exists())

    def test_teacher_parameter_limit_is_checked(self):
        self.distill_config['max_teacher_parameters'] = 1
        self.distill_path.write_text(json.dumps(self.distill_config))
        with self.assertRaisesRegex(ValueError, 'teacher exceeds'):
            self.run_student('bad')
        self.assertFalse((self.root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
