"""Service recovery and corruption checks; fixtures are not training data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import owned_training_service as service


class CheckpointRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def checkpoint(self, segment, step, weights=b'weights', optimizer=b'optimizer'):
        path = self.root / segment / 'checkpoints' / f'step-{step:016d}'
        path.mkdir(parents=True)
        (path / 'weights.safetensors').write_bytes(weights)
        (path / 'optimizer.safetensors').write_bytes(optimizer)
        state = {'schema_version': 'lfg.owned-distillation-checkpoint.v1', 'progress': {'step': step},
                 'model_sha256': hashlib.sha256(weights).hexdigest(),
                 'optimizer_sha256': hashlib.sha256(optimizer).hexdigest()}
        (path / 'state.json').write_text(json.dumps(state))
        return path

    def test_recovers_latest_published_checkpoint_without_pointer(self):
        self.checkpoint('segment-a', 10)
        last = self.checkpoint('segment-b', 20)
        incomplete = self.root / 'segment-c' / 'checkpoints' / '.writing-temp'
        incomplete.mkdir(parents=True)
        self.assertEqual(service.latest_checkpoint(self.root)[1], last)

    def test_rejects_corrupted_and_conflicting_checkpoints(self):
        path = self.checkpoint('segment-a', 10)
        (path / 'weights.safetensors').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'validation'):
            service.latest_checkpoint(self.root)
        (path / 'weights.safetensors').write_bytes(b'weights')
        self.checkpoint('segment-b', 10, weights=b'different')
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            service.latest_checkpoint(self.root)

    def test_only_hashes_latest_checkpoint_tensors(self):
        old = self.checkpoint('segment-a', 10)
        latest = self.checkpoint('segment-b', 20)
        with patch.object(service, 'digest', wraps=service.digest) as hashing:
            self.assertEqual(service.latest_checkpoint(self.root)[1], latest)
        files = {call.args[0] for call in hashing.call_args_list}
        self.assertNotIn(old / 'weights.safetensors', files)
        self.assertIn(latest / 'weights.safetensors', files)

    def test_extra_configuration_fields_are_allowed(self):
        value = {'schema_version': 'lfg.owned-training-service.v1', 'teacher_format': 'owned-byte',
                 'segment_steps': 2, 'status_poll_seconds': 10, 'extension': {'future': True}}
        for key in ('python', 'trainer', 'student_config', 'distillation_config', 'runtime_config',
                    'corpus_plan', 'screen_policy', 'teacher_manifest', 'teacher_weights', 'state_dir'):
            value[key] = str(self.root / 'example')
        path = self.root / 'config.json'
        path.write_text(json.dumps(value))
        self.assertTrue(service.config(path)['extension']['future'])

    def test_assessment_service_requires_its_own_configured_paths(self):
        value = {'schema_version': 'lfg.owned-training-service.v1', 'trainer_kind': 'assessment_sft',
                 'segment_steps': 4, 'status_poll_seconds': 10}
        for key in ('python', 'trainer', 'state_dir', 'dataset', 'checkpoint', 'sft_config'):
            value[key] = str(self.root / 'example')
        path = self.root / 'config.json'
        path.write_text(json.dumps(value))
        self.assertEqual(service.config(path)['trainer_kind'], 'assessment_sft')
        del value['dataset']
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'dataset'):
            service.config(path)


if __name__ == '__main__':
    unittest.main()
