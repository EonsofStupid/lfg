"""Service recovery and corruption checks; fixtures are not training data."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
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
        coverage_patch = patch.object(service, 'trainer_multilabel_coverage', return_value=None)
        coverage_patch.start()
        self.addCleanup(coverage_patch.stop)
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
                  'source_checkpoint_system_linkage': identity['source_checkpoint_system_linkage'],
                  'source_checkpoint_framing_linkage': identity['source_checkpoint_framing_linkage'],
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
        self.assertEqual(commands[0][commands[0].index('--max-config-bytes') + 1], '65536')
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
        self.assertNotIn('training_config_sha256', contract['identity'])

    def test_user_request_checkpoint_state_omits_training_config_digest(self):
        dataset = self.root / 'user-request.json'
        dataset.write_text(json.dumps({'train_records': [{'id': 'synthetic'}]}))
        config = json.loads(self.sft_path.read_text())
        config['schema_version'] = 'lfg.owned-user-request-sft-config.v1'
        self.sft_path.write_text(json.dumps(config))
        self.service_config['dataset'] = str(dataset)
        contract = service.sft_context(self.service_config)
        identity = dict(contract['identity'])
        identity.update({'python': '3.11', 'torch': '2.14.1', 'safetensors': '0.8.0',
                         'cuda_version': None, 'cublas_workspace_config': None, 'cuda_devices': []})
        path = self.root / 'user-segments' / 'segment-a' / 'checkpoints' / 'step-0000000000000001'
        path.mkdir(parents=True)
        weights, optimizer = b'user weights', b'user optimizer'
        (path / 'weights.safetensors').write_bytes(weights)
        (path / 'optimizer.safetensors').write_bytes(optimizer)
        (path / 'state.json').write_text(json.dumps({'schema_version': contract['checkpoint_schema'],
            'identity': identity, 'progress': {'step': 1},
            'model_sha256': hashlib.sha256(weights).hexdigest(),
            'optimizer_sha256': hashlib.sha256(optimizer).hexdigest()}))
        found = service.latest_checkpoint(path.parents[2], contract['checkpoint_schema'], sft=contract)
        self.assertEqual(found[0], 1)

    def test_linked_sft_source_checkpoint_identity_records_verified_status(self):
        system_sha, framing_sha = '1' * 64, '2' * 64
        source_weights_sha = service.digest(self.source / 'weights.safetensors')
        (self.source / 'manifest.json').write_text(json.dumps({
            'schema_version': 'lfg.owned-assessment-sft-result.v1',
            'weights_sha256': source_weights_sha,
            'architecture': {'context': 4, 'layers': 1},
            'code_sha256': {'owned_student.py': service.digest(Path(service.__file__).with_name('owned_student.py'))},
            'system_sha256': system_sha, 'framing_sha256': framing_sha,
            'coverage': {'system_sha256': system_sha, 'framing_sha256': framing_sha},
        }))
        contract = service.sft_context(self.service_config)
        self.assertEqual(contract['identity']['source_checkpoint_system_linkage'], 'verified_linked')
        self.assertEqual(contract['identity']['source_checkpoint_framing_linkage'], 'verified_linked')

    def test_legacy_unlinked_sft_source_requires_explicit_user_request_policy(self):
        source_weights_sha = service.digest(self.source / 'weights.safetensors')
        (self.source / 'manifest.json').write_text(json.dumps({
            'schema_version': 'lfg.owned-assessment-sft-result.v1',
            'weights_sha256': source_weights_sha, 'architecture': {'context': 4, 'layers': 1},
            'code_sha256': {'owned_student.py': service.digest(Path(service.__file__).with_name('owned_student.py'))},
            'coverage': {},
        }))
        config = json.loads(self.sft_path.read_text())
        config['schema_version'] = 'lfg.owned-user-request-sft-config.v1'
        config['allow_unlinked_legacy_sft_checkpoint'] = True
        self.sft_path.write_text(json.dumps(config))
        dataset = self.root / 'user-request.json'
        dataset.write_text(json.dumps({'train_records': [{'id': 'synthetic'}]}))
        self.service_config['dataset'] = str(dataset)
        contract = service.sft_context(self.service_config)
        self.assertEqual(contract['identity']['source_checkpoint_system_linkage'], 'unknown_unlinked')
        config['schema_version'] = 'lfg.owned-assessment-sft-config.v1'
        self.sft_path.write_text(json.dumps(config))
        self.service_config['dataset'] = str(self.dataset)
        with self.assertRaisesRegex(ValueError, 'unlinked provenance'):
            service.sft_context(self.service_config)

    def test_sft_config_size_limits_are_enforced_and_forwarded(self):
        for limit, extension_size in ((32768, 0), (131072, 70000)):
            with self.subTest(limit=limit):
                self.service_config['state_dir'] = str(self.root / f'state-{limit}')
                self.service_config['max_sft_config_bytes'] = limit
                config = json.loads(self.sft_path.read_text())
                if extension_size:
                    config['large_inert_extension'] = 'x' * extension_size
                else:
                    config.pop('large_inert_extension', None)
                self.sft_path.write_text(json.dumps(config))
                self.config_path.write_text(json.dumps(self.service_config))
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

                with patch.dict(sys.modules, {'fcntl': self._fake_fcntl()}), \
                        patch.object(service.subprocess, 'Popen', Child):
                    service.run(self.config_path, exit_on_complete=True)
                self.assertEqual(commands[0][commands[0].index('--max-config-bytes') + 1], str(limit))
                self.assertEqual(len(commands), 2)

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


@unittest.skipUnless(os.name == 'posix' and all(importlib.util.find_spec(name)
    for name in ('torch', 'safetensors')), 'POSIX torch and safetensors required for real trainer integration')
class TrainerGeneratedCheckpointIntegrationTests(unittest.TestCase):
    def _capture_child(self, command, real_popen, commands, **kwargs):
        if kwargs:
            return real_popen(command, **kwargs)
        commands.append(command)
        process = real_popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        class CapturedChild:
            def wait(child):
                stdout, stderr = process.communicate()
                if process.returncode:
                    raise RuntimeError(f'trainer child failed: {stderr or stdout}')
                return process.returncode
            def poll(child):
                return process.poll()
            def send_signal(child, signum):
                process.send_signal(signum)

        return CapturedChild()

    def test_real_assessment_checkpoint_is_accepted_and_resumed_by_supervisor(self):
        import subprocess
        from test_train_owned_assessment_sft import OwnedAssessmentSFTTests

        fixture = OwnedAssessmentSFTTests('test_multilabel_experiment_resumes_and_binds_source_and_configuration')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config.update(steps=2, checkpoint_every=1, batch_size=1, max_attention_elements=100_000)
        fixture.write_config()
        state_dir = fixture.root / 'assessment-service-state'
        seed_output = state_dir / 'segments' / 'seeded-segment'
        seed_output.parent.mkdir(parents=True)
        partial = fixture.subject.experiment(fixture.dataset, fixture.checkpoint, fixture.config_path,
                                             seed_output, max_steps_this_run=1)
        self.assertFalse(partial['completed'])
        service_config = {'schema_version': 'lfg.owned-training-service.v1', 'trainer_kind': 'assessment_sft',
            'python': sys.executable, 'trainer': str(Path(fixture.subject.__file__).resolve()),
            'state_dir': str(state_dir), 'dataset': str(fixture.dataset),
            'checkpoint': str(fixture.checkpoint), 'sft_config': str(fixture.config_path),
            'segment_steps': 1, 'status_poll_seconds': 1}
        service_config_path = fixture.root / 'assessment-service.json'
        service_config_path.write_text(json.dumps(service_config))
        contract = service.sft_context(service.config(service_config_path))
        self.assertEqual(contract['checkpoint_schema'], 'lfg.owned-assessment-sft-checkpoint.v1')
        self.assertEqual(service.latest_checkpoint(state_dir / 'segments', contract['checkpoint_schema'], sft=contract)[0], 1)
        commands = []
        real_popen = subprocess.Popen
        with patch.object(service.subprocess, 'Popen',
                          lambda command, **kwargs: self._capture_child(command, real_popen, commands, **kwargs)):
            service.run(service_config_path, exit_on_complete=True)
        self.assertEqual(len(commands), 1)
        self.assertIn('--resume', commands[0])
        result = json.loads((Path(json.loads((state_dir / 'complete.json').read_text())['result_path'])
                             / 'manifest.json').read_text())
        self.assertEqual(result['schema_version'], 'lfg.owned-assessment-sft-result.v1')
        self.assertEqual(result['steps'], 2)

    def test_real_user_request_identity_and_linked_or_allowed_unlinked_sources(self):
        import shutil
        import subprocess
        from test_train_owned_assessment_sft import OwnedAssessmentSFTTests
        from test_user_request_sft import UserRequestSFTTests
        import train_owned_assessment_sft as trainer_module

        assessment = OwnedAssessmentSFTTests('test_multilabel_experiment_resumes_and_binds_source_and_configuration')
        assessment.setUp()
        self.addCleanup(assessment.doCleanups)
        assessment.config.update(steps=1, checkpoint_every=1, batch_size=1, max_attention_elements=100_000)
        assessment.write_config()
        linked_source = assessment.root / 'linked-assessment-source'
        linked_result = assessment.subject.experiment(assessment.dataset, assessment.checkpoint,
            assessment.config_path, linked_source)
        self.assertEqual(linked_result['schema_version'], 'lfg.owned-assessment-sft-result.v1')

        request = UserRequestSFTTests('test_user_request_task_runs_one_private_fixture_step')
        request.setUp()
        self.addCleanup(request.doCleanups)
        request._build()
        user_request_config = json.loads(Path(trainer_module.__file__).with_name(
            'owned-user-request-sft.example.json').read_text())
        user_request_config.update(context=2048, batch_size=1, steps=2, checkpoint_every=1,
            max_parameters=100_000, max_attention_elements=10_000_000, max_dataset_bytes=1_000_000,
            max_record_bytes=200_000, max_manifest_bytes=1_000_000, max_checkpoint_bytes=10_000_000,
            max_optimizer_bytes=10_000_000, max_examples=10)
        user_request_config_path = request.root / 'owned-user-request-sft-train.json'
        user_request_config_path.write_text(json.dumps(user_request_config))

        def run_user_request(source_path, sft_config_path, state_name):
            state_dir = request.root / state_name
            segment = state_dir / 'segments' / 'preexisting-segment'
            segment.parent.mkdir(parents=True)
            result = trainer_module.experiment(request.root / 'dataset.json', source_path, sft_config_path,
                                               segment, max_steps_this_run=1)
            self.assertFalse(result['completed'])
            service_config = {'schema_version': 'lfg.owned-training-service.v1', 'trainer_kind': 'assessment_sft',
                'python': sys.executable, 'trainer': str(Path(trainer_module.__file__).resolve()),
                'state_dir': str(state_dir), 'dataset': str(request.root / 'dataset.json'),
                'checkpoint': str(source_path), 'sft_config': str(sft_config_path),
                'segment_steps': 1, 'status_poll_seconds': 1}
            service_path = request.root / f'{state_name}.json'
            service_path.write_text(json.dumps(service_config))
            contract = service.sft_context(service.config(service_path))
            identity = json.loads((Path(result['latest_checkpoint']) / 'state.json').read_text())['identity']
            self.assertNotIn('training_config_sha256', identity)
            self.assertEqual(identity['source_checkpoint_system_linkage'], contract['identity']['source_checkpoint_system_linkage'])
            found = service.latest_checkpoint(state_dir / 'segments', contract['checkpoint_schema'], sft=contract)
            self.assertEqual(found[0], 1)
            return service_path, state_dir, contract

        linked_path, linked_state, linked_contract = run_user_request(linked_source,
            user_request_config_path, 'linked-user-request-service')
        self.assertEqual(linked_contract['identity']['source_checkpoint_system_linkage'], 'verified_linked')
        original_dataset = (request.root / 'dataset.json').read_bytes()
        (request.root / 'dataset.json').write_bytes(original_dataset + b' ')
        with self.assertRaisesRegex(ValueError, 'dataset_manifest_sha256'):
            service.latest_checkpoint(linked_state / 'segments', linked_contract['checkpoint_schema'],
                                      sft=service.sft_context(service.config(linked_path)))
        (request.root / 'dataset.json').write_bytes(original_dataset)
        changed_config = json.loads(user_request_config_path.read_text())
        changed_config['learning_rate'] *= 2
        changed_config_path = request.root / 'changed-user-request-sft.json'
        changed_config_path.write_text(json.dumps(changed_config))
        changed_service = service.config(linked_path)
        changed_service['sft_config'] = str(changed_config_path)
        changed_service_path = request.root / 'changed-linked-user-request-service.json'
        changed_service_path.write_text(json.dumps(changed_service))
        changed_contract = service.sft_context(service.config(changed_service_path))
        with self.assertRaisesRegex(ValueError, 'continuation_config_sha256'):
            service.latest_checkpoint(linked_state / 'segments', linked_contract['checkpoint_schema'],
                                      sft=changed_contract)
        copied_trainer = request.root / 'changed-trainer'
        copied_trainer.mkdir()
        trainer_dir = Path(trainer_module.__file__).resolve().parent
        for filename in ('train_owned_assessment_sft.py', 'train_owned_student.py',
                         'owned_student.py', 'owned_sft_checkpoint.py'):
            shutil.copyfile(trainer_dir / filename, copied_trainer / filename)
        with (copied_trainer / 'train_owned_assessment_sft.py').open('ab') as altered:
            altered.write(b'\n# synthetic code identity drift\n')
        changed_code_service = service.config(linked_path)
        changed_code_service['trainer'] = str(copied_trainer / 'train_owned_assessment_sft.py')
        changed_code_service_path = request.root / 'changed-code-user-request-service.json'
        changed_code_service_path.write_text(json.dumps(changed_code_service))
        changed_code_contract = service.sft_context(service.config(changed_code_service_path))
        with self.assertRaisesRegex(ValueError, 'code_sha256'):
            service.latest_checkpoint(linked_state / 'segments', linked_contract['checkpoint_schema'],
                                      sft=changed_code_contract)
        commands = []
        real_popen = subprocess.Popen
        with patch.object(service.subprocess, 'Popen',
                          lambda command, **kwargs: self._capture_child(command, real_popen, commands, **kwargs)):
            service.run(linked_path, exit_on_complete=True)
        self.assertEqual(len(commands), 1)
        self.assertIn('--resume', commands[0])
        linked_result = json.loads((Path(json.loads((linked_state / 'complete.json').read_text())['result_path'])
                                    / 'manifest.json').read_text())
        self.assertEqual(linked_result['schema_version'], 'lfg.owned-user-request-sft-result.v1')
        self.assertEqual(linked_result['steps'], 2)

        legacy_source = request.root / 'legacy-unlinked-assessment-source'
        legacy_source.mkdir()
        (legacy_source / 'weights.safetensors').write_bytes((linked_source / 'weights.safetensors').read_bytes())
        legacy_manifest = json.loads((linked_source / 'manifest.json').read_text())
        legacy_manifest.pop('system_sha256')
        legacy_manifest.pop('framing_sha256')
        legacy_manifest['coverage'].pop('system_sha256')
        legacy_manifest['coverage'].pop('framing_sha256')
        (legacy_source / 'manifest.json').write_text(json.dumps(legacy_manifest))
        user_request_config['allow_unlinked_legacy_sft_checkpoint'] = True
        user_request_config_path.write_text(json.dumps(user_request_config))
        _, _, legacy_contract = run_user_request(legacy_source, user_request_config_path,
                                                 'legacy-user-request-service')
        self.assertEqual(legacy_contract['identity']['source_checkpoint_system_linkage'], 'unknown_unlinked')
    def test_real_multilabel_trainer_checkpoint_resumes_after_supervisor_restart(self):
        import subprocess
        from test_train_owned_assessment_sft import OwnedAssessmentSFTTests

        fixture = OwnedAssessmentSFTTests('test_multilabel_experiment_resumes_and_binds_source_and_configuration')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        dataset, _ = fixture.multilabel_fixture()
        fixture.config.update(context=512, batch_size=2, steps=2, checkpoint_every=1,
                              max_attention_elements=5_000_000, max_parameters=100_000, cpu_threads=1)
        fixture.write_config()
        trainer_path = Path(fixture.subject.__file__).resolve()
        state_dir = fixture.root / 'supervised-state'
        service_config = {'schema_version': 'lfg.owned-training-service.v1', 'trainer_kind': 'assessment_sft',
            'python': sys.executable, 'trainer': str(trainer_path), 'state_dir': str(state_dir),
            'dataset': str(dataset), 'checkpoint': str(fixture.checkpoint),
            'sft_config': str(fixture.config_path), 'segment_steps': 1, 'status_poll_seconds': 1}
        service_config_path = fixture.root / 'service.json'
        service_config_path.write_text(json.dumps(service_config))
        actual_popen = subprocess.Popen
        commands = []

        class InterruptedChild:
            def __init__(child, command):
                commands.append(command)
                child.process = actual_popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            def wait(child):
                stdout, stderr = child.process.communicate()
                if child.process.returncode:
                    raise RuntimeError(f'trainer child failed: {stderr or stdout}')
                raise RuntimeError('simulated supervisor interruption after child checkpoint')
            def poll(child):
                return child.process.poll()
            def send_signal(child, signum):
                child.process.send_signal(signum)

        with patch.object(service.subprocess, 'Popen',
                          lambda command, **kwargs: actual_popen(command, **kwargs)
                          if kwargs else InterruptedChild(command)):
            with self.assertRaisesRegex(RuntimeError, 'simulated supervisor interruption'):
                service.run(service_config_path)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][commands[0].index('--max-config-bytes') + 1], '65536')
        contract = service.sft_context(service.config(service_config_path))
        first_checkpoint = service.latest_checkpoint(state_dir / 'segments', contract['checkpoint_schema'], sft=contract)
        self.assertEqual(first_checkpoint[0], 1)
        first_state = json.loads((first_checkpoint[1] / 'state.json').read_text())
        self.assertIn('multilabel_coverage_sha256', first_state['identity'])
        with patch.object(service.subprocess, 'Popen',
                          lambda command, **kwargs: self._capture_child(command, actual_popen, commands, **kwargs)):
            service.run(service_config_path, exit_on_complete=True)
        self.assertEqual(len(commands), 2)
        self.assertIn('--resume', commands[1])
        self.assertEqual(service.latest_checkpoint(state_dir / 'segments', contract['checkpoint_schema'], sft=contract)[0], 2)
        complete = json.loads((state_dir / 'complete.json').read_text())
        result_manifest = json.loads((Path(complete['result_path']) / 'manifest.json').read_text())
        self.assertEqual(result_manifest['schema_version'], 'lfg.owned-multilabel-sft-result.v1')
        self.assertTrue(result_manifest['completed'])
        self.assertEqual(result_manifest['steps'], 2)

    def test_dependencyless_supervisor_validates_checkpoint_in_configured_trainer_environment(self):
        """A lightweight supervisor can verify a real ML checkpoint using the configured trainer Python."""
        import subprocess
        from test_train_owned_assessment_sft import OwnedAssessmentSFTTests

        lightweight_python = Path('/usr/bin/python3')
        if not lightweight_python.is_file():
            self.skipTest('dependency-less system Python is unavailable')
        dependency_probe = subprocess.run([str(lightweight_python), '-c',
            "import importlib.util,sys; sys.exit(0 if not importlib.util.find_spec('torch') and not importlib.util.find_spec('safetensors') else 1)"],
            capture_output=True, text=True, check=False)
        if dependency_probe.returncode != 0:
            self.skipTest('system Python is not dependency-less')
        fixture = OwnedAssessmentSFTTests('test_multilabel_experiment_resumes_and_binds_source_and_configuration')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        dataset, _ = fixture.multilabel_fixture()
        fixture.config.update(context=512, batch_size=2, steps=2, checkpoint_every=1,
                              max_attention_elements=5_000_000, max_parameters=100_000, cpu_threads=1)
        fixture.write_config()
        trainer_path = Path(fixture.subject.__file__).resolve()
        state_dir = fixture.root / 'dependencyless-supervisor-state'
        segment = state_dir / 'segments' / 'seeded-segment'
        segment.parent.mkdir(parents=True)
        trained = fixture.subject.experiment(dataset, fixture.checkpoint, fixture.config_path,
                                             segment, max_steps_this_run=1)
        self.assertFalse(trained['completed'])
        service_config = {'schema_version': 'lfg.owned-training-service.v1', 'trainer_kind': 'assessment_sft',
            'python': sys.executable, 'trainer': str(trainer_path), 'state_dir': str(state_dir),
            'dataset': str(dataset), 'checkpoint': str(fixture.checkpoint),
            'sft_config': str(fixture.config_path), 'segment_steps': 1, 'status_poll_seconds': 1}
        service_path = fixture.root / 'dependencyless-supervisor.json'
        service_path.write_text(json.dumps(service_config))
        driver = r'''import json, sys
sys.path.insert(0, sys.argv[1])
import owned_training_service as service
value = service.config(service.Path(sys.argv[2]))
contract = service.sft_context(value)
latest = service.latest_checkpoint(service.Path(value['state_dir']) / 'segments', contract['checkpoint_schema'], sft=contract)
print(json.dumps({'schema': contract['checkpoint_schema'], 'step': latest[0]}))
'''
        result = subprocess.run([str(lightweight_python), '-c', driver,
                                 str(Path(service.__file__).resolve().parent), str(service_path)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(json.loads(result.stdout),
                         {'schema': 'lfg.owned-multilabel-sft-checkpoint.v1', 'step': 1})

        # A configured interpreter without ML dependencies must fail closed, without a raw path/error leak.
        bad_config = dict(service_config, python=str(lightweight_python))
        bad_path = fixture.root / 'missing-trainer-dependencies.json'
        bad_path.write_text(json.dumps(bad_config))
        bad_driver = r'''import sys
sys.path.insert(0, sys.argv[1])
import owned_training_service as service
try:
    service.sft_context(service.config(service.Path(sys.argv[2])))
except ValueError as error:
    print(str(error))
else:
    raise SystemExit(3)
'''
        failed = subprocess.run([str(lightweight_python), '-c', bad_driver,
                                 str(Path(service.__file__).resolve().parent), str(bad_path)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(failed.returncode, 0, msg=failed.stderr)
        self.assertEqual(failed.stdout.strip(),
                         'cannot validate multilabel checkpoint identity in configured trainer environment')
        self.assertNotIn(str(fixture.root), failed.stdout + failed.stderr)


if __name__ == '__main__':
    unittest.main()
