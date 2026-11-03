"""Fixture tests for private two-checkpoint byte evaluation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'safetensors'))


@unittest.skipUnless(AVAILABLE, 'torch and safetensors required')
class CheckpointPairTests(unittest.TestCase):
    def setUp(self):
        import evaluate_owned_checkpoint_pair as subject
        import owned_student
        import train_owned_student as base
        from safetensors.torch import save
        import torch

        self.subject, self.model_code, self.base = subject, owned_student, base
        self.torch, self.save = torch, save
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / 'evaluation.json'
        self.config.write_text(json.dumps({
            'schema_version': 'lfg.owned-checkpoint-pair-evaluation-config.v1',
            'context': 8, 'cpu_threads': 1, 'warmup_windows': 0, 'max_windows': 30,
            'max_data_bytes': 10000, 'max_model_bytes': 1000000,
            'max_model_parameters': 100000, 'max_manifest_bytes': 100000,
            'max_policy_bytes': 65536}))
        self.policy = Path(subject.__file__).with_name('private-corpus-screen.example.json')
        self.source = self.root / 'selected-source.txt'
        self.source.write_text('This is a meaningful fixture source, with Bearer fixtureSecret0123456789 removed.')
        self.architecture = {'width': 8, 'layers': 1, 'heads': 2, 'conv_kernel': 3,
                             'context': 8, 'ff_multiplier': 2, 'attention_every': 1}
        self.left = self._checkpoint('left', 101)
        self.right = self._checkpoint('right', 202)

    def _student_config(self, name, architecture):
        example = Path(self.base.__file__).with_name('owned-student.example.json')
        value = json.loads(example.read_text())
        value['architecture'] = architecture
        path = self.root / f'{name}-student-config.json'
        path.write_text(json.dumps(value))
        return path, self.base.load_config(path)[1]

    def _checkpoint(self, name, seed, architecture=None):
        torch = self.torch
        architecture = architecture or self.architecture
        student_config_path, student_config_sha = self._student_config(name, architecture)
        torch.manual_seed(seed)
        model = self.model_code.OwnedByteStudent(architecture).eval()
        weights = self.save({key: value.detach().contiguous() for key, value in model.state_dict().items()})
        weights_path = self.root / f'{name}.safetensors'
        weights_path.write_bytes(weights)
        manifest = {'schema_version': 'lfg.owned-distillation-result.v1', 'completed': True,
                    'architecture': architecture, 'weights_sha256': self.base.sha(weights),
                    'code_sha256': {'owned_student.py': self.base.sha(Path(self.model_code.__file__).read_bytes())},
                    'source_sha256': (f'{seed:064x}'), 'steps': 10, 'tokens_processed': 80,
                    'student_config_sha256': student_config_sha,
                    'distillation_config_sha256': f'{seed + 2:064x}',
                    'teacher_distillation_performed': True,
                    'teacher': {'kind': 'fixture', 'weights_sha256': f'{seed + 3:064x}'}}
        manifest_path = self.root / f'{name}.json'
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path, weights_path, student_config_path

    def _evaluate(self, source, output):
        return self.subject.evaluate(self.config, source, self.policy, output,
                                     *self.left, *self.right)

    def test_scores_same_screened_stream_and_keeps_content_out_of_report(self):
        report = self._evaluate(self.source, self.root / 'report.json')
        self.assertEqual(report['predicted_bytes_per_checkpoint'], report['evaluation_source']['screened_bytes'] - 1)
        self.assertEqual(set(report['metrics']), {'left', 'right'})
        self.assertEqual(report['metric_deltas_right_minus_left']['bits_per_byte_right_minus_left'],
                         report['metrics']['right']['bits_per_byte'] - report['metrics']['left']['bits_per_byte'])
        self.assertNotIn('fixtureSecret0123456789', json.dumps(report))
        self.assertNotIn(self.source.read_text(), json.dumps(report))
        self.assertEqual(report['semantic_independence'], 'unknown')
        self.assertEqual(report['project_independence'], 'unknown')
        self.assertEqual(report['training_histories_comparison'], 'not_performed')
        self.assertFalse(report['promotion_eligible'])
        self.assertGreater(report['metrics']['left']['bits_per_byte'], 0)

    def test_rejects_data_declared_by_either_checkpoint(self):
        manifest_path, weights_path, _config_path = self.right
        manifest = json.loads(manifest_path.read_text())
        manifest['source_sha256'] = self.base.sha(self.source.read_bytes())
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'overlaps declared'):
            self._evaluate(self.source, self.root / 'overlap.json')
        self.assertFalse((self.root / 'overlap.json').exists())

    def test_rejects_incompatible_architecture(self):
        architecture = dict(self.architecture, width=16)
        self.right = self._checkpoint('right-wide', 202, architecture)
        with self.assertRaisesRegex(ValueError, 'architectures are incompatible'):
            self._evaluate(self.source, self.root / 'architecture.json')

    def test_rejects_student_config_drift(self):
        config_path = self.left[2]
        value = json.loads(config_path.read_text())
        value['weight_decay'] = value['weight_decay'] + 0.0001
        config_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'checkpoint does not match'):
            self._evaluate(self.source, self.root / 'config-drift.json')


if __name__ == '__main__':
    unittest.main()
