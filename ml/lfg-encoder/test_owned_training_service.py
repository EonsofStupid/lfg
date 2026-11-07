"""Service recovery and corruption checks; fixtures are not training data."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
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
        self.assertNotIn(old / 'state.json', files)
        self.assertIn(latest / 'weights.safetensors', files)
        (old / 'state.json').write_text('invalid old metadata')
        self.assertEqual(service.latest_checkpoint(self.root)[1], latest)

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


class MultilabelSupervisorLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        (self.dataset / 'manifest.json').write_text('{"schema_version":"lfg.owned-multilabel-training.v1"}\n')
        (self.dataset / 'train.jsonl').write_text('{"row":"synthetic"}\n')
        self.source = self.root / 'source'
        self.source.mkdir()
        source_weights = b'synthetic source weights'
        (self.source / 'weights.safetensors').write_bytes(source_weights)
        (self.source / 'manifest.json').write_text(json.dumps({
            'schema_version': 'lfg.owned-distillation-result.v1',
            'weights_sha256': hashlib.sha256(source_weights).hexdigest(),
            'architecture': {'context': 4, 'layers': 1},
            'code_sha256': {'owned_student.py': service.digest(Path(service.__file__).with_name('owned_student.py'))},
        }))
        self.sft_path = self.root / 'sft.json'
        self.sft_path.write_text(json.dumps({
            'schema_version': 'lfg.owned-multilabel-sft-config.v1', 'steps': 2, 'context': 4,
            'max_dataset_bytes': 4096, 'max_manifest_bytes': 4096,
            'checkpoint_every': 1, 'learning_rate': 0.001, 'extension': {'inert': True},
            'expected_manifest_sha256': hashlib.sha256((self.dataset / 'manifest.json').read_bytes()).hexdigest(),
        }))
        self.trainer = Path(service.__file__).with_name('train_owned_assessment_sft.py')
        self.config_path = self.root / 'service.json'
        self.service_config = {
            'schema_version': 'lfg.owned-training-service.v1', 'trainer_kind': 'assessment_sft',
            'python': sys.executable, 'trainer': str(self.trainer), 'state_dir': str(self.root / 'state'),
            'dataset': str(self.dataset), 'checkpoint': str(self.source), 'sft_config': str(self.sft_path),
            'segment_steps': 1, 'status_poll_seconds': 1,
        }
        self.config_path.write_text(json.dumps(self.service_config))

    def _checkpoint(self, output, step):
        contract = service.sft_context(self.service_config)
        identity = dict(contract['identity'])
        identity.update({'multilabel_coverage_sha256': 'a' * 64, 'python': '3.11', 'torch': '2.0',
                         'safetensors': '0.4', 'cuda_version': None, 'cublas_workspace_config': None,
                         'cuda_devices': []})
        checkpoint = output / 'checkpoints' / f'step-{step:016d}'
        checkpoint.mkdir(parents=True)
        weights, optimizer = b'synthetic model', b'synthetic optimizer'
        (checkpoint / 'weights.safetensors').write_bytes(weights)
        (checkpoint / 'optimizer.safetensors').write_bytes(optimizer)
        state = {'schema_version': contract['checkpoint_schema'], 'identity': identity,
                 'progress': {'step': step, 'losses': [1.0] * step, 'supervised_tokens': 3},
                 'model_sha256': hashlib.sha256(weights).hexdigest(),
                 'optimizer_sha256': hashlib.sha256(optimizer).hexdigest()}
        state_raw = json.dumps(state, sort_keys=True, separators=(',', ':')).encode()
        (checkpoint / 'state.json').write_bytes(state_raw)
        return contract

    def _complete_result(self, output, contract):
        weights = b'synthetic completed model'
        (output / 'weights.safetensors').write_bytes(weights)
        identity = contract['identity']
        result = {'schema_version': contract['result_schema'], 'completed': True,
                  'steps': contract['steps'], 'config_sha256': contract['config_sha256'],
                  'dataset_manifest_sha256': identity['dataset_manifest_sha256'],
                  'training_file_sha256': identity['training_file_sha256'],
                  'source_checkpoint_manifest_sha256': identity['source_checkpoint_manifest_sha256'],
                  'source_checkpoint_weights_sha256': identity['source_checkpoint_weights_sha256'],
                  'source_checkpoint_schema': identity['source_checkpoint_schema'],
                  'code_sha256': identity['code_sha256'],
                  'weights_sha256': hashlib.sha256(weights).hexdigest()}
        (output / 'manifest.json').write_text(json.dumps(result))

    def _fake_fcntl(self):
        return SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *_args: None)

    def test_supervisor_resumes_multilabel_segments_and_accepts_matching_completion(self):
        commands = []

        class Child:
            def __init__(child, command):
                commands.append(command)
                output = Path(command[command.index('--output') + 1])
                contract = self._checkpoint(output, len(commands))
                if len(commands) == 2:
                    self._complete_result(output, contract)

            def wait(child):
                return 0

            def poll(child):
                return 0

            def send_signal(child, _signal):
                pass

        with patch.dict(sys.modules, {'fcntl': self._fake_fcntl()}), patch.object(service.subprocess, 'Popen', Child):
            service.run(self.config_path, exit_on_complete=True)
        self.assertEqual(len(commands), 2)
        self.assertIn('--resume', commands[1])
        status = json.loads((Path(self.service_config['state_dir']) / 'status.json').read_text())
        self.assertEqual(status['state'], 'complete')
        complete = json.loads((Path(self.service_config['state_dir']) / 'complete.json').read_text())
        self.assertEqual(complete['completed_steps'], 2)

    def test_supervisor_rejects_stale_checkpoint_without_retry(self):
        commands = []

        class Child:
            def __init__(child, command):
                commands.append(command)
                self._checkpoint(Path(command[command.index('--output') + 1]), 1)

            def wait(child):
                return 0

            def poll(child):
                return 0

            def send_signal(child, _signal):
                pass

        with patch.dict(sys.modules, {'fcntl': self._fake_fcntl()}), patch.object(service.subprocess, 'Popen', Child):
            with self.assertRaisesRegex(ValueError, 'without advancing'):
                service.run(self.config_path, exit_on_complete=True)
        self.assertEqual(len(commands), 2)
        self.assertIn('--resume', commands[1])

    def test_multilabel_checkpoint_identity_is_bound_to_configured_source(self):
        contract = service.sft_context(self.service_config)
        path = self.root / 'segment' / 'checkpoints' / 'step-0000000000000001'
        path.mkdir(parents=True)
        (path / 'weights.safetensors').write_bytes(b'w')
        (path / 'optimizer.safetensors').write_bytes(b'o')
        identity = dict(contract['identity'])
        identity.update({'multilabel_coverage_sha256': 'a' * 64, 'python': '3.11', 'torch': '2.0',
                         'safetensors': '0.4', 'cuda_version': None, 'cublas_workspace_config': None,
                         'cuda_devices': []})
        identity['source_checkpoint_manifest_sha256'] = '0' * 64
        state = {'schema_version': contract['checkpoint_schema'], 'identity': identity,
                 'progress': {'step': 1}, 'model_sha256': hashlib.sha256(b'w').hexdigest(),
                 'optimizer_sha256': hashlib.sha256(b'o').hexdigest()}
        (path / 'state.json').write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, 'source_checkpoint_manifest_sha256'):
            service.latest_checkpoint(path.parents[2], contract['checkpoint_schema'], sft=contract)

    def test_unknown_sft_schema_fails_closed_and_assessment_schema_is_explicit(self):
        config = json.loads(self.sft_path.read_text())
        config['schema_version'] = 'lfg.future-sft-config.v9'
        self.sft_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'unsupported SFT configuration schema'):
            service.sft_context(self.service_config)
        config['schema_version'] = 'lfg.owned-assessment-sft-config.v1'
        self.sft_path.write_text(json.dumps(config))
        contract = service.sft_context(self.service_config)
        self.assertEqual(contract['checkpoint_schema'], 'lfg.owned-assessment-sft-checkpoint.v1')

    def test_existing_assessment_checkpoint_schema_remains_resumable(self):
        config = json.loads(self.sft_path.read_text())
        config['schema_version'] = 'lfg.owned-assessment-sft-config.v1'
        self.sft_path.write_text(json.dumps(config))
        contract = service.sft_context(self.service_config)
        self.assertEqual(contract['checkpoint_schema'], 'lfg.owned-assessment-sft-checkpoint.v1')
        identity = dict(contract['identity'])
        identity.update({'python': '3.11', 'torch': '2.0', 'safetensors': '0.4', 'cuda_version': None,
                         'cublas_workspace_config': None, 'cuda_devices': []})
        segments = self.root / 'assessment-segments'
        path = segments / 'segment-a' / 'checkpoints' / 'step-0000000000000001'
        path.mkdir(parents=True)
        weights, optimizer = b'assessment weights', b'assessment optimizer'
        (path / 'weights.safetensors').write_bytes(weights)
        (path / 'optimizer.safetensors').write_bytes(optimizer)
        (path / 'state.json').write_text(json.dumps({
            'schema_version': contract['checkpoint_schema'], 'identity': identity,
            'progress': {'step': 1}, 'model_sha256': hashlib.sha256(weights).hexdigest(),
            'optimizer_sha256': hashlib.sha256(optimizer).hexdigest(),
        }))
        self.assertEqual(service.latest_checkpoint(segments, contract['checkpoint_schema'], sft=contract)[0], 1)

    def test_user_request_schema_uses_its_versioned_checkpoint(self):
        dataset = self.root / 'user-request.json'
        dataset.write_text(json.dumps({'train_records': [{'id': 'synthetic'}]}))
        config = json.loads(self.sft_path.read_text())
        config['schema_version'] = 'lfg.owned-user-request-sft-config.v1'
        self.sft_path.write_text(json.dumps(config))
        self.service_config['dataset'] = str(dataset)
        contract = service.sft_context(self.service_config)
        self.assertEqual(contract['checkpoint_schema'], 'lfg.owned-user-request-sft-checkpoint.v1')

    def test_completed_service_stays_inspectable_without_reopening_training_inputs(self):
        state = Path(self.service_config['state_dir'])
        state.mkdir()
        (state / 'complete.json').write_text(json.dumps({'completed_steps': 2, 'weights_sha256': 'a' * 64}))
        (self.dataset / 'manifest.json').unlink()
        with patch.dict(sys.modules, {'fcntl': self._fake_fcntl()}), \
                patch.object(service.subprocess, 'Popen', side_effect=AssertionError('completed job relaunched')):
            service.run(self.config_path, exit_on_complete=True)
        status = json.loads((state / 'status.json').read_text())
        self.assertEqual(status['state'], 'complete')
        self.assertEqual(status['step'], 2)


if __name__ == '__main__':
    unittest.main()
