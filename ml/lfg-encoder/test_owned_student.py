"""Small software fixtures for model mechanics; not LFG training evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

TORCH = importlib.util.find_spec('torch') is not None
SAFETENSORS = importlib.util.find_spec('safetensors') is not None


@unittest.skipUnless(TORCH, 'torch required')
class StudentTests(unittest.TestCase):
    def test_causality_of_hybrid_backbone(self):
        import torch
        import owned_student as subject
        config = {'width': 12, 'layers': 2, 'heads': 3, 'conv_kernel': 3,
                  'context': 12, 'ff_multiplier': 2, 'attention_every': 2}
        torch.manual_seed(7)
        model = subject.OwnedByteStudent(config).eval()
        first = torch.tensor([[1, 2, 3, 4, 5, 6]])
        changed = torch.tensor([[1, 2, 3, 9, 8, 7]])
        with torch.no_grad():
            a, b = model(first), model(changed)
        torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=0)
        self.assertFalse(torch.equal(a[:, 3:], b[:, 3:]))
        with self.assertRaisesRegex(ValueError, 'context'):
            model(torch.zeros((1, 13), dtype=torch.long))

    def test_topk_matches_collapsed_full_kl_and_has_gradients(self):
        import torch
        import owned_student as subject
        teacher = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float64)
        student = torch.tensor([[1.0, -0.1, 0.4, -0.4]], dtype=torch.float64, requires_grad=True)
        indices = torch.tensor([[0, 1]])
        actual = subject.decoupled_topk_loss(student, indices, teacher[:, :2], teacher[:, 2:].sum(-1), 1.0)
        probs = student.softmax(-1)
        teacher_collapsed = torch.cat((teacher[:, :2], teacher[:, 2:].sum(-1, keepdim=True)), dim=-1)
        student_collapsed = torch.cat((probs[:, :2], probs[:, 2:].sum(-1, keepdim=True)), dim=-1)
        expected = (teacher_collapsed * (teacher_collapsed.log() - student_collapsed.log())).sum(-1).mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(float(student.grad.abs().sum()), 0)

    def test_topk_rejects_missing_tail_and_duplicate_indices(self):
        import torch
        import owned_student as subject
        logits = torch.zeros(1, 4)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            subject.decoupled_topk_loss(logits, torch.tensor([[0, 0]]), torch.tensor([[.4, .3]]), torch.tensor([.3]), 2)
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            subject.decoupled_topk_loss(logits, torch.tensor([[0, 1]]), torch.tensor([[.4, .3]]), torch.tensor([.2]), 2)

    def test_temperature_changes_conditional_component(self):
        import torch
        import owned_student as subject
        logits = torch.tensor([[0.2, 0.7, -0.1, 0.4]])
        index, top, tail = torch.tensor([[0, 1]]), torch.tensor([[.5, .3]]), torch.tensor([.2])
        a = subject.decoupled_topk_loss(logits, index, top, tail, 1.0)
        b = subject.decoupled_topk_loss(logits, index, top, tail, 2.0)
        self.assertNotEqual(float(a), float(b))

    def test_topk_extreme_logits_are_finite(self):
        import torch
        import owned_student as subject
        logits = torch.tensor([[1000., -1000., 0., -500.]], requires_grad=True)
        loss = subject.decoupled_topk_loss(logits, torch.tensor([[0, 1]]),
                                           torch.tensor([[0.5, 0.3]]), torch.tensor([0.2]), 2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    @unittest.skipUnless(SAFETENSORS, 'safetensors required')
    def test_all_private_outputs_must_be_ignored_and_untracked(self):
        import train_owned_student as trainer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / '.gitignore').write_text('/out/weights.safetensors\n')
            with self.assertRaisesRegex(ValueError, 'ignored and untracked'):
                trainer.require_private_outputs(root / 'out')
            (root / '.gitignore').write_text('/out/\n')
            trainer.require_private_outputs(root / 'out')
            (root / 'out').mkdir()
            tracked = root / 'out' / 'manifest.json'
            tracked.write_text('fixture')
            subprocess.run(['git', '-C', str(root), 'add', '-f', 'out/manifest.json'], check=True)
            tracked.unlink()
            with self.assertRaisesRegex(ValueError, 'ignored and untracked'):
                trainer.require_private_outputs(root / 'out')

    @unittest.skipUnless(SAFETENSORS, 'safetensors required')
    def test_seeded_training_repeats_with_fixture_text(self):
        import train_owned_student as trainer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(Path(trainer.__file__).with_name('owned-student.example.json').read_text())
            config.update(steps=2, batch_size=1, cpu_threads=1)
            config['architecture'].update(width=8, layers=1, heads=2, context=8, attention_every=1)
            config_path, data = root / 'config.json', root / 'data.txt'
            config_path.write_text(json.dumps(config))
            data.write_text('software testing fixture text for reproducible code')
            first = trainer.experiment(config_path, data, root / 'first')
            second = trainer.experiment(config_path, data, root / 'second')
            self.assertEqual(first['losses'], second['losses'])
            self.assertEqual(first['weights_sha256'], second['weights_sha256'])
            self.assertFalse(first['validation_performed'])
            self.assertFalse(first['teacher_distillation_performed'])
            self.assertFalse(first['promotion_eligible'])


if __name__ == '__main__':
    unittest.main()
