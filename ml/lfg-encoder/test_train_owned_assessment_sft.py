"""Software fixtures for supervised training mechanics; not real assessment data."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(x) for x in ('torch', 'safetensors')), 'torch and safetensors required')
class OwnedAssessmentSFTTests(unittest.TestCase):
    def setUp(self):
        import assessment_distill
        import owned_student
        from safetensors.torch import save_file
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sha = lambda b: hashlib.sha256(b).hexdigest()
        self.checkpoint = self.root / 'checkpoint'
        self.checkpoint.mkdir()
        self.architecture = {'width': 8, 'layers': 2, 'heads': 2, 'conv_kernel': 3,
                             'context': 8, 'ff_multiplier': 2, 'attention_every': 2}
        weights = self.checkpoint / 'weights.safetensors'
        save_file(owned_student.OwnedByteStudent(self.architecture).state_dict(), str(weights))
        (self.checkpoint / 'manifest.json').write_text(json.dumps({
            'schema_version': 'lfg.owned-student-experiment-result.v1', 'architecture': self.architecture,
            'weights_sha256': sha(weights.read_bytes()),
            'code_sha256': {'owned_student.py': sha(Path(owned_student.__file__).read_bytes())}}))
        context_text = 'P \u03b1'
        trace = {'id': 'fixture', 'request': context_text + '\n\nQ',
                 'project_context_sha256': hashlib.sha256(context_text.encode()).hexdigest(),
                 'project_context_byte_length': len(context_text.encode()),
                 'teacher_system_sha256': hashlib.sha256(b'Assess the supplied request.').hexdigest(),
                 'project_context_extensions': {'custom_extension': 'preserve-me'},
                 'response': 'Fixture teacher response',
                 'teacher': 'fixture', 'review_status': 'unreviewed', 'outcome': 'unknown',
                 'source_ref': 'fixture', 'source_sha256': 'a' * 64}
        source = self.root / 'trace.jsonl'
        source.write_text(json.dumps(trace) + '\n')
        plan = {'schema_version': assessment_distill.PLAN_SCHEMA, 'revision': 'fixture', 'task': 'fixture',
                'system_prompt': 'Assess the supplied request.', 'authorization_attestation': 'fixture',
                'redaction_attestation': 'fixture', 'forbidden_literals': [], 'include_outcomes': ['unknown'],
                'max_source_bytes': 8192, 'max_total_bytes': 8192, 'max_examples': 1,
                'sources': [{'path': source.name, 'sha256': sha(source.read_bytes()), 'project': 'fixture',
                             'project_group': 'fixture', 'split': 'train', 'origin': 'owner_chat'}]}
        plan_path = self.root / 'plan.json'
        plan_path.write_text(json.dumps(plan))
        self.dataset = self.root / 'prepared'
        assessment_distill.build(plan_path, self.dataset)
        import train_owned_assessment_sft as subject
        self.subject = subject
        self.config = json.loads(Path(subject.__file__).with_name('owned-assessment-sft.example.json').read_text())
        self.config.update(context=128, batch_size=2, steps=3, max_attention_elements=100000,
                           max_parameters=100000, cpu_threads=1)
        self.config_path = self.root / 'config.json'
        self.write_config()

    def test_training_prompt_preserves_rendered_context_bytes(self):
        records, _, _, coverage = self.subject.read_training(self.dataset, self.config)
        expected = (self.config['framing']['system_prefix'] + 'Assess the supplied request.'
                    + self.config['framing']['system_suffix'] + self.config['framing']['user_prefix']
                    + 'P \u03b1\n\nQ'
                    + self.config['framing']['user_suffix']
                    + self.config['framing']['assistant_prefix']).encode('utf-8')
        self.assertTrue(records[0]['bytes'].startswith(expected))
        self.assertEqual(records[0]['response_start'], len(expected))
        self.assertEqual(coverage['selected'], 1)
        self.assertEqual(coverage['system_sha256'], hashlib.sha256(b'Assess the supplied request.').hexdigest())
        self.assertEqual(coverage['framing_sha256'], self.subject.framing_sha256(self.config['framing']))

    def test_training_rejects_mixed_system_prompts(self):
        manifest_path = self.dataset / 'manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        train_path = self.dataset / 'train.jsonl'
        rows = [json.loads(line) for line in train_path.read_bytes().splitlines()]
        second = json.loads(json.dumps(rows[0]))
        second['id'] = 'fixture-2'
        second['messages'][0]['content'] += ' changed'
        raw = b''.join((json.dumps(row, sort_keys=True) + '\n').encode() for row in (rows[0], second))
        train_path.write_bytes(raw)
        manifest['files']['train.jsonl'] = hashlib.sha256(raw).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'different system prompts'):
            self.subject.read_training(self.dataset, self.config)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def test_sample_masks_prompt_and_padding(self):
        records = [{'bytes': b'promptANSWER', 'response_start': 6, 'weight': 1.0}]
        source, target, corrections = self.subject.sample_batch(records, random.Random(3), 5, 8)
        for inputs, labels, weights in zip(source, target, corrections, strict=True):
            self.assertEqual(len(inputs), len(labels))
            known = [value for value in labels if value != -100]
            self.assertTrue(known)
            self.assertTrue(all(value in b'ANSWER' for value in known))
            self.assertTrue(all(value not in b'prompt' for value in known))
            self.assertTrue(all(weight == 0 for value, weight in zip(labels, weights, strict=True) if value == -100))

    def test_window_weights_give_each_response_byte_equal_expected_contribution(self):
        record = {'bytes': b'ppABCDEF', 'response_start': 2, 'weight': 1.0}
        totals = {byte: 0.0 for byte in b'ABCDEF'}
        class FixedEnd:
            def __init__(self, end): self.end = end
            def choices(self, records, weights, k): return [records[0]] * k
            def randrange(self, start, stop): return self.end
        for end in range(record['response_start'], len(record['bytes'])):
            _, labels, corrections = self.subject.sample_batch([record], FixedEnd(end), 1, 3)
            for value, weight in zip(labels[0], corrections[0], strict=True):
                if value != -100:
                    totals[value] += weight
        for total in totals.values():
            self.assertAlmostEqual(total, 1.0)

    def test_reproducible_training_preserves_claims_and_ignores_validation(self):
        manifest_path = self.dataset / 'manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        manifest['files']['validation.jsonl'] = '0' * 64  # No such file exists.
        manifest_path.write_text(json.dumps(manifest))
        a = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'first')
        b = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'second')
        self.assertEqual(a['weights_sha256'], b['weights_sha256'])
        self.assertEqual(a['losses'], b['losses'])
        self.assertEqual(a['system_sha256'], hashlib.sha256(b'Assess the supplied request.').hexdigest())
        self.assertEqual(a['framing_sha256'], self.subject.framing_sha256(self.config['framing']))
        self.assertEqual(a['position_rows_reused'], 8)
        self.assertEqual(a['position_rows_initialized'], 120)
        self.assertGreater(a['supervised_tokens'], 0)
        self.assertEqual(a['coverage']['claims']['review_status:unreviewed'], 1)
        self.assertFalse(a['teacher_claims_verified'])
        self.assertFalse(a['validation_opened'])
        self.assertFalse(a['test_opened'])
        self.assertFalse(a['assessment_quality_evaluated'])
        self.assertFalse(a['promotion_eligible'])

    def test_interrupted_resume_matches_uninterrupted_weights_and_optimizer(self):
        full = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'full')
        partial = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'partial',
                                          max_steps_this_run=1)
        self.assertFalse(partial['completed'])
        self.assertFalse((self.root / 'partial' / 'manifest.json').exists())
        checkpoint = Path(partial['latest_checkpoint'])
        state_sha = partial['latest_checkpoint_state_sha256']
        self.assertTrue((checkpoint / 'optimizer.safetensors').is_file())
        resumed = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'resumed',
                                          resume=checkpoint, resume_state_sha256=state_sha)
        self.assertTrue(resumed['completed'])
        self.assertEqual(resumed['resumed_from_step'], 1)
        self.assertEqual(full['weights_sha256'], resumed['weights_sha256'])
        self.assertEqual(full['losses'], resumed['losses'])
        self.assertEqual(full['supervised_tokens'], resumed['supervised_tokens'])
        optimizer = checkpoint / 'optimizer.safetensors'
        optimizer.write_bytes(optimizer.read_bytes() + b'tampered')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'bad',
                                    resume=checkpoint, resume_state_sha256=state_sha)
        self.assertFalse((self.root / 'bad').exists())

    def test_resume_rejects_changed_training_policy(self):
        partial = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'partial',
                                          max_steps_this_run=1)
        self.config['framing']['user_prefix'] = 'changed fixture frame'
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'checkpoint identity'):
            self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'bad',
                                    resume=Path(partial['latest_checkpoint']),
                                    resume_state_sha256=partial['latest_checkpoint_state_sha256'])
        self.assertFalse((self.root / 'bad').exists())

    @unittest.skipUnless(importlib.util.find_spec('fcntl'), 'Linux supervisor lock required')
    def test_supervisor_runs_segmented_training_and_exits_idempotently(self):
        script = Path(self.subject.__file__).with_name('owned_training_service.py')
        state = self.root / 'service-state'
        service = {'schema_version': 'lfg.owned-training-service.v1', 'trainer_kind': 'assessment_sft',
            'python': sys.executable, 'trainer': str(Path(self.subject.__file__).resolve()), 'state_dir': str(state),
            'dataset': str(self.dataset), 'checkpoint': str(self.checkpoint), 'sft_config': str(self.config_path),
            'segment_steps': 1, 'status_poll_seconds': 1}
        path = self.root / 'service.json'; path.write_text(json.dumps(service))
        command = [sys.executable, str(script), '--config', str(path), '--exit-on-complete']
        subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
        result = json.loads((state / 'complete.json').read_text())
        self.assertEqual(result['completed_steps'], 3)
        self.assertEqual(len(list((state / 'segments').iterdir())), 3)
        subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
        self.assertEqual(len(list((state / 'segments').iterdir())), 3)

    def test_final_checkpoint_can_publish_after_interruption(self):
        output = self.root / 'full'
        full = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, output)
        pointer = json.loads((output / 'latest.json').read_text())
        recovered = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'finalized',
            resume=output / 'checkpoints' / pointer['checkpoint'], resume_state_sha256=pointer['state_sha256'])
        self.assertEqual(recovered['weights_sha256'], full['weights_sha256'])
        self.assertEqual(recovered['losses'], full['losses'])
        self.assertEqual(recovered['resumed_from_step'], self.config['steps'])

    def test_failed_checkpoint_leaves_previous_snapshot_resumable(self):
        self.config['checkpoint_every'] = 1
        self.write_config()
        full = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'full')
        writer = self.subject.checkpoints._write
        optimizer_writes = 0
        def fail_second_optimizer(path, raw):
            nonlocal optimizer_writes
            if path.name == 'optimizer.safetensors':
                optimizer_writes += 1
                if optimizer_writes == 2:
                    raise OSError('fixture interruption')
            return writer(path, raw)
        interrupted = self.root / 'interrupted'
        with patch.object(self.subject.checkpoints, '_write', side_effect=fail_second_optimizer):
            with self.assertRaisesRegex(OSError, 'fixture interruption'):
                self.subject.experiment(self.dataset, self.checkpoint, self.config_path, interrupted)
        latest = json.loads((interrupted / 'latest.json').read_bytes())
        self.assertEqual(latest['step'], 1)
        checkpoint = interrupted / 'checkpoints' / latest['checkpoint']
        self.assertFalse((interrupted / 'checkpoints' / 'step-0000000000000002').exists())
        resumed = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'resumed',
                                          resume=checkpoint, resume_state_sha256=latest['state_sha256'])
        self.assertEqual(full['weights_sha256'], resumed['weights_sha256'])
        self.assertEqual(full['losses'], resumed['losses'])

    def test_resume_rejects_changed_state_digest(self):
        partial = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'partial',
                                          max_steps_this_run=1)
        checkpoint = Path(partial['latest_checkpoint'])
        state = checkpoint / 'state.json'
        state.write_bytes(state.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'state digest mismatch'):
            self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'bad',
                                    resume=checkpoint, resume_state_sha256=partial['latest_checkpoint_state_sha256'])
        self.assertFalse((self.root / 'bad').exists())

    def test_resume_can_extend_steps_and_change_checkpoint_frequency(self):
        partial = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'partial',
                                          max_steps_this_run=1)
        self.config.update(steps=4, checkpoint_every=1)
        self.write_config()
        full = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'full')
        resumed = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'resumed',
                    resume=Path(partial['latest_checkpoint']), resume_state_sha256=partial['latest_checkpoint_state_sha256'])
        self.assertEqual(full['weights_sha256'], resumed['weights_sha256'])
        self.assertEqual(full['losses'], resumed['losses'])

    def test_resume_probe_retains_identity_and_reports_match(self):
        import probe_owned_sft_resume
        output = self.root / 'probe'
        report = probe_owned_sft_resume.run(self.dataset, self.checkpoint, self.config_path, output)
        self.assertTrue(report['same_weights'])
        self.assertTrue(report['same_losses'])
        self.assertTrue(report['same_supervised_tokens'])
        self.assertFalse(report['assessment_quality_evaluated'])
        self.assertEqual(report, json.loads((output / 'report.json').read_bytes()))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            probe_owned_sft_resume.run(self.dataset, self.checkpoint, self.config_path, output)

    def test_private_output_tree_requires_ignored_and_untracked_directory(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True, capture_output=True)
        target = self.root / 'private-output'
        with self.assertRaisesRegex(ValueError, 'ignored and untracked'):
            self.subject.checkpoints.private_directory(target)
        (self.root / '.gitignore').write_text('private-output/\n')
        self.subject.checkpoints.private_directory(target)
        target.mkdir()
        (target / 'tracked').write_text('fixture')
        subprocess.run(['git', '-C', str(self.root), 'add', '-f', 'private-output/tracked'], check=True, capture_output=True)
        with self.assertRaisesRegex(ValueError, 'ignored and untracked'):
            self.subject.checkpoints.private_directory(target)

    def test_user_request_cli_redacts_os_paths_without_changing_legacy_errors(self):
        user_config = json.loads(Path(self.subject.__file__).with_name('owned-user-request-sft.example.json').read_bytes())
        user_config_path = self.root / 'user-request-config.json'
        user_config_path.write_text(json.dumps(user_config), encoding='utf-8')
        private_marker = 'PRIVATE_PATH_MARKER_DO_NOT_LOG'
        user_dataset = self.root / private_marker / 'missing-user-dataset.json'
        common = [sys.executable, str(Path(self.subject.__file__)), '--checkpoint', str(self.checkpoint)]
        user_run = subprocess.run(common + ['--dataset', str(user_dataset), '--config', str(user_config_path),
            '--output', str(self.root / 'user-output')], capture_output=True, text=True)
        self.assertNotEqual(user_run.returncode, 0)
        self.assertIn('FileNotFoundError', user_run.stderr)
        self.assertIn('filesystem operation failed', user_run.stderr)
        self.assertNotIn(private_marker, user_run.stderr)
        legacy_dataset = self.root / private_marker / 'missing-legacy-dataset'
        legacy_run = subprocess.run(common + ['--dataset', str(legacy_dataset), '--config', str(self.config_path),
            '--output', str(self.root / 'legacy-output')], capture_output=True, text=True)
        self.assertNotEqual(legacy_run.returncode, 0)
        self.assertIn(str(legacy_dataset), legacy_run.stderr)

    def test_additional_input_and_config_fields_are_accepted(self):
        train_path = self.dataset / 'train.jsonl'
        row = json.loads(train_path.read_bytes())
        row['extra'] = {'future': True}
        row['metadata']['extra'] = 'fixture'
        raw = (json.dumps(row) + '\n').encode()
        train_path.write_bytes(raw)
        manifest_path = self.dataset / 'manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        manifest['files']['train.jsonl'] = hashlib.sha256(raw).hexdigest()
        manifest['files']['future-metadata.json'] = 'fixture'
        manifest['sources'][0]['extra'] = 42
        manifest['extra'] = {'future': True}
        manifest_path.write_text(json.dumps(manifest))
        self.config['extra'] = {'future': True}
        self.write_config()
        result = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'output')
        self.assertTrue(result['completed'])

    @unittest.skipUnless(all(importlib.util.find_spec(x) for x in ('onnx', 'onnxruntime')), 'onnx and onnxruntime required')
    def test_completed_sft_checkpoint_exports_with_onnx_parity(self):
        import export_owned_byte_onnx
        output = self.root / 'trained'
        trained = self.subject.experiment(self.dataset, self.checkpoint, self.config_path, output)
        export_config = json.loads(Path(export_owned_byte_onnx.__file__).with_name('owned-byte-onnx-export.example.json').read_text())
        export_config['parity_lengths'] = [1, 4, 8]
        config_path, source = self.root / 'export.json', self.root / 'parity.txt'
        config_path.write_text(json.dumps(export_config))
        source.write_bytes(b'fixture!')
        result = export_owned_byte_onnx.export(output, config_path, source, self.root / 'onnx')
        manifest = json.loads((self.root / 'onnx' / 'manifest.json').read_bytes())
        self.assertEqual(manifest['source_weights_sha256'], trained['weights_sha256'])
        self.assertEqual(result['model_sha256'], manifest['model_sha256'])
        self.assertFalse(manifest['assessment_quality_checked'])
        if os.environ.get('LFG_OWNED_BYTE_TEST_BINARY'):
            import compare_owned_byte_native
            import compare_owned_generation
            native = compare_owned_byte_native.compare(Path(os.environ['LFG_OWNED_BYTE_TEST_BINARY']), output, source,
                [1, 4, 8], absolute_tolerance=1e-5, relative_tolerance=1e-5, timeout=120,
                artifact=self.root / 'onnx', manifest_sha256=result['manifest_sha256'],
                runtime_config=Path(__file__).resolve().parents[2] / 'crates/lfg-server/config/owned-byte-runtime.example.json')
            self.assertFalse(native['assessment_quality_checked'])
            generated = compare_owned_generation.compare(Path(os.environ['LFG_OWNED_BYTE_TEST_BINARY']), output, source,
                prefix_bytes=8, max_new_bytes=8, cpu_threads=1, timeout=120,
                artifact=self.root / 'onnx', manifest_sha256=result['manifest_sha256'],
                runtime_config=Path(__file__).resolve().parents[2] / 'crates/lfg-server/config/owned-byte-runtime.example.json')
            self.assertEqual(generated['generated_bytes'], 8)
            self.assertFalse(generated['assessment_quality_checked'])

    def test_heldout_and_prompt_truncation_require_explicit_handling(self):
        manifest_path = self.dataset / 'manifest.json'
        original = json.loads(manifest_path.read_bytes())
        manifest_path.write_text(json.dumps({**original, 'files': {**original['files'], 'test.jsonl': '0' * 64}}))
        with self.assertRaisesRegex(ValueError, 'held-out'):
            self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'blocked')
        manifest_path.write_text(json.dumps(original))
        self.config['context'] = 8
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'full prompt and response exceed context'):
            self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'blocked')
        self.config['allow_truncated_prompt'] = True
        records, _, _, coverage = self.subject.read_training(self.dataset, self.config)
        self.assertEqual(len(records), 1)
        self.assertEqual(coverage['initial_prompt_exceeds_context'], 1)
        self.assertEqual(coverage['complete_example_exceeds_context'], 1)
        self.assertFalse((self.root / 'blocked').exists())

    def test_exclusion_and_attention_limits_fail_before_output(self):
        self.config['review_status_weights']['unreviewed'] = 0
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'no positively weighted'):
            self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'blocked')
        self.config['review_status_weights']['unreviewed'] = 1
        self.config['max_attention_elements'] = 1
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'attention estimate'):
            self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'blocked')
        self.assertFalse((self.root / 'blocked').exists())

    def test_config_size_limit_is_configurable(self):
        with self.assertRaisesRegex(ValueError, 'configured limit'):
            self.subject.configuration(self.config_path, 8)
        self.assertEqual(self.subject.configuration(self.config_path, 16384)[0]['context'], 128)

    def test_cuda_requires_a_valid_fraction_and_never_falls_back(self):
        self.config['cuda_memory_fraction'] = 0
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'invalid supervised training configuration'):
            self.subject.configuration(self.config_path)
        self.config.update(device='cuda', cuda_memory_fraction=0.01)
        self.write_config()
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'CUDA device is unavailable'):
                self.subject.experiment(self.dataset, self.checkpoint, self.config_path, self.root / 'blocked')
        self.assertFalse((self.root / 'blocked').exists())

    def test_heldout_source_rejected_before_training_file_open_and_provenance_checked(self):
        manifest_path = self.dataset / 'manifest.json'
        original = json.loads(manifest_path.read_bytes())
        heldout = json.loads(json.dumps(original))
        heldout['sources'][0]['split'] = 'test'
        manifest_path.write_text(json.dumps(heldout))
        training = self.dataset / 'train.jsonl'
        training_bytes = training.read_bytes()
        training.unlink()
        with self.assertRaisesRegex(ValueError, 'held-out'):
            self.subject.read_training(self.dataset, self.config)
        training.write_bytes(training_bytes)
        row = json.loads(training_bytes)
        row['metadata']['project_group'] = 'other'
        changed = (json.dumps(row) + '\n').encode()
        training.write_bytes(changed)
        original['files']['train.jsonl'] = hashlib.sha256(changed).hexdigest()
        manifest_path.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, 'invalid training trace'):
            self.subject.read_training(self.dataset, self.config)


if __name__ == '__main__':
    unittest.main()
