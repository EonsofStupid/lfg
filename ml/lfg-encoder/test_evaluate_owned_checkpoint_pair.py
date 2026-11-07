"""Fixture tests for private two-checkpoint byte evaluation."""
import importlib.util
import json
import os
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
        from unittest.mock import patch

        self.subject, self.model_code, self.base = subject, owned_student, base
        # Successful fixture runs exercise scoring while explicitly bypassing
        # the platform guard; OutputSafetyTests verifies Windows fails closed.
        self.output_guard = patch.object(subject.output_safety, 'require_private_output_directory', lambda _path: None)
        self.output_guard.start()
        self.addCleanup(self.output_guard.stop)
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

    @unittest.skipIf(os.name == 'nt', 'POSIX owner/mode checks are platform-specific')
    def test_scoring_writes_through_real_private_output_guard(self):
        import stat

        self.output_guard.stop()
        output = self.root / 'guarded-report.json'
        self._evaluate(self.source, output)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        with self.assertRaisesRegex(ValueError, 'output already exists'):
            self._evaluate(self.source, output)

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

    def test_rejects_non_boolean_completion_claims(self):
        manifest_path, _weights_path, _config_path = self.left
        original = json.loads(manifest_path.read_text())
        for index, invalid in enumerate((0, 'no', None)):
            manifest = dict(original, completed=invalid)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'not complete'):
                self._evaluate(self.source, self.root / f'incomplete-{index}.json')
            self.assertFalse((self.root / f'incomplete-{index}.json').exists())
        manifest_path.write_text(json.dumps(original))

    def test_accepts_absent_completion_field_from_legacy_writer(self):
        manifest_path, _weights_path, _config_path = self.left
        manifest = json.loads(manifest_path.read_text())
        manifest.pop('completed')
        manifest_path.write_text(json.dumps(manifest))
        result = self._evaluate(self.source, self.root / 'legacy-complete.json')
        self.assertIsNone(result['left_checkpoint']['completed'])


@unittest.skipUnless(AVAILABLE, 'torch and safetensors required')
class OutputSafetyIntegrationTests(unittest.TestCase):
    def test_windows_guard_precedes_all_input_reads(self):
        import evaluate_owned_checkpoint_pair as subject
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'must-not-exist.json'
            with patch.object(subject.output_safety, '_is_windows', return_value=True), \
                 patch.object(subject, 'load_config', side_effect=AssertionError('input read')):
                with self.assertRaisesRegex(ValueError, 'unavailable on Windows'):
                    subject.evaluate(root / 'missing-config', root / 'missing-source',
                        root / 'missing-policy', output, root / 'left-manifest',
                        root / 'left-weights', root / 'left-config',
                        root / 'right-manifest', root / 'right-weights', root / 'right-config')
            self.assertFalse(output.exists())


class OutputSafetyTests(unittest.TestCase):
    @unittest.skipIf(os.name == 'nt', 'POSIX owner/mode checks are platform-specific')
    def test_accepts_private_parent_and_creates_owner_only_file(self):
        import checkpoint_pair_output_safety as subject
        import stat

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'report.json'
            descriptor = subject.create_private_output(output)
            os.close(descriptor)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_windows_output_creation_fails_closed(self):
        import checkpoint_pair_output_safety as subject
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'must-not-exist.json'
            with patch.object(subject, '_is_windows', return_value=True):
                with self.assertRaisesRegex(ValueError, 'unavailable on Windows'):
                    subject.create_private_output(output)
            self.assertFalse(output.exists())

    @unittest.skipIf(os.name == 'nt', 'POSIX owner/mode checks are platform-specific')
    def test_rejects_group_accessible_output_parent(self):
        import checkpoint_pair_output_safety as subject

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / 'public-parent'
            parent.mkdir()
            parent.chmod(0o755)
            with self.assertRaisesRegex(ValueError, 'must be owned and private'):
                subject.require_private_output_directory(parent / 'report.json')
            self.assertFalse((parent / 'report.json').exists())


if __name__ == '__main__':
    unittest.main()
