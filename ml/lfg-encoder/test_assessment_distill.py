"""Fixtures test preparation only; they are not real teacher data or training."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import assessment_distill as subject


class DistillationPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan_path = self.root / 'plan.json'
        self.output = self.root / 'output'
        self.row = {'id': 'fixture-1', 'request': 'Fixture request', 'response': '{"intent":"fixture"}',
                    'teacher': 'fixture-teacher', 'review_status': 'unreviewed', 'outcome': 'unknown',
                    'source_ref': 'fixture-session', 'source_sha256': 'a' * 64}
        self.source = self.root / 'source.jsonl'
        self.write_source([self.row])
        self.plan = {'schema_version': subject.PLAN_SCHEMA, 'revision': 'fixture', 'task': 'assessment-fixture',
                     'system_prompt': 'Use the fixture assessment format.', 'authorization_attestation': 'fixture only',
                     'redaction_attestation': 'fixture only', 'forbidden_literals': ['fixture-secret-marker'],
                     'include_outcomes': ['success', 'unknown'], 'max_source_bytes': 8192,
                     'max_total_bytes': 16384, 'max_examples': 20,
                     'sources': [self.source_entry(self.source, 'train', 'project-one')]}
        self.write_plan()

    def write_source(self, rows, path=None):
        (path or self.source).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    def source_entry(self, path, split, project):
        return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'project': project,
                'project_group': project, 'split': split, 'origin': 'owner_chat'}

    def write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan), encoding='utf-8')

    def test_unreviewed_teacher_trace_is_preserved_as_claim_and_not_promoted(self):
        project_context = 'project status: active'
        context_request = project_context + '\n\nfixture request'
        self.row['request'] = context_request
        self.row['teacher_system_sha256'] = hashlib.sha256(self.plan['system_prompt'].encode()).hexdigest()
        self.row['project_context_sha256'] = hashlib.sha256(project_context.encode()).hexdigest()
        self.row['project_context_byte_length'] = len(project_context.encode())
        self.row['custom_extension'] = {'kept': True}
        self.write_source([self.row])
        self.plan['sources'][0] = self.source_entry(self.source, 'train', 'project-one')
        self.write_plan()
        result = subject.build(self.plan_path, self.output)
        record = json.loads((self.output / 'train.jsonl').read_bytes())
        manifest = json.loads((self.output / 'manifest.json').read_bytes())
        self.assertEqual(record['messages'][0]['content'], self.plan['system_prompt'])
        self.assertEqual(record['messages'][1]['content'], context_request)
        self.assertEqual(record['metadata']['trace_extensions'], {
            'project_context_sha256': self.row['project_context_sha256'],
            'project_context_byte_length': self.row['project_context_byte_length'],
            'teacher_system_sha256': self.row['teacher_system_sha256'],
            'custom_extension': {'kept': True}})
        self.assertEqual(record['messages'][2]['content'], self.row['response'])
        self.assertEqual(record['metadata']['review_status_claim'], 'unreviewed')
        self.assertEqual(record['metadata']['outcome_claim'], 'unknown')
        self.assertEqual(record['metadata']['original_source_ref_claim'], 'fixture-session')
        self.assertEqual(record['metadata']['original_source_sha256_claim'], 'a' * 64)
        self.assertFalse(result['promotion_eligible'])
        self.assertFalse(manifest['claims_verified'])
        self.assertFalse(manifest['validation_is_independent'])
        self.assertFalse((self.output / 'validation.jsonl').exists())
        self.assertNotIn('fixture-secret-marker', json.dumps(manifest))

    def test_context_extension_digest_is_verified_before_output(self):
        project_context = 'project facts'
        self.row.update(request=project_context + '\n\nfixture request',
                        project_context_sha256='0' * 64,
                        project_context_byte_length=len(project_context.encode()))
        self.write_source([self.row])
        self.plan['sources'][0] = self.source_entry(self.source, 'train', 'project-one')
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'context digest'):
            subject.build(self.plan_path, self.output)
        self.assertFalse(self.output.exists())

    def test_collected_teacher_system_prompt_must_match_distillation_plan(self):
        self.row['teacher_system_sha256'] = hashlib.sha256(b'different system').hexdigest()
        self.write_source([self.row])
        self.plan['sources'][0] = self.source_entry(self.source, 'train', 'project-one')
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'system prompt digest'):
            subject.build(self.plan_path, self.output)
        self.assertFalse(self.output.exists())

    def test_context_requires_present_non_null_matching_system_digest(self):
        context = 'context facts'
        self.row.update(request=context + '\n\nfixture request',
                        project_context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                        project_context_byte_length=len(context.encode()))
        for mode, digest_value in (('omitted', None), ('null', None),
                                   ('mismatch', hashlib.sha256(b'wrong system').hexdigest())):
            with self.subTest(mode=mode):
                if mode == 'omitted':
                    self.row.pop('teacher_system_sha256', None)
                else:
                    self.row['teacher_system_sha256'] = digest_value
                self.write_source([self.row])
                self.plan['sources'][0] = self.source_entry(self.source, 'train', 'project-one')
                self.write_plan()
                with self.assertRaisesRegex(ValueError, 'system prompt digest'):
                    subject.build(self.plan_path, self.output)
                self.assertFalse(self.output.exists())

    def test_contextual_render_collision_with_legacy_request_is_rejected(self):
        context = 'context A'
        rendered = context + '\n\nrequest B'
        first = {**self.row, 'request': rendered,
                 'teacher_system_sha256': hashlib.sha256(self.plan['system_prompt'].encode()).hexdigest(),
                 'project_context_sha256': hashlib.sha256(context.encode()).hexdigest(),
                 'project_context_byte_length': len(context.encode())}
        second = {**self.row, 'id': 'fixture-2', 'request': rendered,
                  'source_ref': 'fixture-session-2', 'source_sha256': 'b' * 64}
        second_path = self.root / 'legacy-collision.jsonl'
        self.write_source([first])
        self.write_source([second], second_path)
        self.plan['sources'] = [self.source_entry(self.source, 'train', 'project-one'),
                                self.source_entry(second_path, 'validation', 'project-two')]
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'duplicate normalized request'):
            subject.build(self.plan_path, self.output)
        self.assertFalse(self.output.exists())

    def test_underlying_requests_deduplicate_across_context_and_splits(self):
        source_request = 'ＦＩＸＴＵＲＥ\trequest'
        for second_is_contextual in (True, False):
            with self.subTest(second_is_contextual=second_is_contextual):
                first = {**self.row, 'request': 'first facts\n\n' + source_request,
                         'teacher_system_sha256': hashlib.sha256(self.plan['system_prompt'].encode()).hexdigest(),
                         'project_context_sha256': hashlib.sha256(b'first facts').hexdigest(),
                         'project_context_byte_length': len('first facts'.encode())}
                second_request = ('second facts\n\nfixture   REQUEST' if second_is_contextual
                                  else ' fixture   REQUEST ')
                second = {**self.row, 'id': 'fixture-2', 'request': second_request,
                          'source_ref': 'fixture-session-2', 'source_sha256': 'b' * 64}
                if second_is_contextual:
                    second.update(project_context_sha256=hashlib.sha256(b'second facts').hexdigest(),
                                  project_context_byte_length=len('second facts'.encode()),
                                  teacher_system_sha256=hashlib.sha256(self.plan['system_prompt'].encode()).hexdigest())
                second_path = self.root / f'second-{second_is_contextual}.jsonl'
                self.write_source([first])
                self.write_source([second], second_path)
                self.plan['sources'] = [self.source_entry(self.source, 'train', 'project-one'),
                                        self.source_entry(second_path, 'validation', 'project-two')]
                self.write_plan()
                output = self.root / f'output-{second_is_contextual}'
                with self.assertRaisesRegex(ValueError, 'duplicate normalized request'):
                    subject.build(self.plan_path, output)
                self.assertFalse(output.exists())

    def test_cli_uses_real_plan_and_refuses_existing_output(self):
        cli = Path(subject.__file__).with_name('distill.py')
        args = [sys.executable, str(cli), 'build-sft', '--plan', str(self.plan_path), '--output-dir', str(self.output)]
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        self.assertFalse(json.loads(result.stdout)['promotion_eligible'])
        original = (self.output / 'train.jsonl').read_bytes()
        self.assertNotEqual(subprocess.run(args, capture_output=True).returncode, 0)
        self.assertEqual((self.output / 'train.jsonl').read_bytes(), original)

    @unittest.skipUnless(shutil.which('pwsh') or shutil.which('powershell'), 'PowerShell is not installed')
    def test_powershell_entry_point_passes_selected_plan_and_output(self):
        shell = shutil.which('pwsh') or shutil.which('powershell')
        script = subject.REPOSITORY / 'scripts' / 'deploy.ps1'
        subprocess.run([shell, '-NoProfile', '-File', str(script), '-Action', 'DistillBuild',
                        '-PythonCommand', sys.executable, '-DistillPlan', str(self.plan_path),
                        '-DistillOutputDirectory', str(self.output)],
                       cwd=subject.REPOSITORY, check=True, capture_output=True, text=True)
        self.assertTrue((self.output / 'manifest.json').is_file())

    @unittest.skipUnless(shutil.which('make'), 'make is not installed')
    def test_make_entry_point_passes_selected_plan_and_output(self):
        subprocess.run([shutil.which('make'), '-C', str(subject.REPOSITORY), 'ml-distill-build',
                        f'PYTHON={sys.executable}', f'DISTILL_PLAN={self.plan_path}',
                        f'DISTILL_OUTPUT_DIR={self.output}'],
                       check=True, capture_output=True, text=True)
        self.assertTrue((self.output / 'manifest.json').is_file())

    def test_test_split_rejected_before_opening_declared_source(self):
        self.plan['sources'][0].update(split='test', path='missing-must-not-open')
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'only selected'):
            subject.build(self.plan_path, self.output)
        self.assertFalse(self.output.exists())

    def test_overlap_secret_and_digest_fail_before_output(self):
        second = self.root / 'second.jsonl'
        duplicate = {**self.row, 'id': 'fixture-2', 'request': ' FIXTURE  request ', 'source_sha256': 'b' * 64}
        self.write_source([duplicate], second)
        self.plan['sources'].append(self.source_entry(second, 'validation', 'project-two'))
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'duplicate normalized'):
            subject.build(self.plan_path, self.output)
        self.plan['sources'].pop()
        self.row['response'] = 'fixture-secret-marker'
        self.write_source([self.row])
        self.plan['sources'][0]['sha256'] = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'forbidden literal'):
            subject.build(self.plan_path, self.output)
        self.plan['sources'][0]['sha256'] = '0' * 64
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'source digest'):
            subject.build(self.plan_path, self.output)
        self.assertFalse(self.output.exists())

    def test_group_separation_and_record_limit(self):
        second = self.root / 'second.jsonl'
        self.write_source([{**self.row, 'id': 'other', 'request': 'Different fixture', 'source_sha256': 'b' * 64}], second)
        entry = self.source_entry(second, 'validation', 'project-two')
        entry['project_group'] = 'project-one'
        self.plan['sources'].append(entry)
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'group spans'):
            subject.build(self.plan_path, self.output)
        self.plan['sources'][1]['project_group'] = 'project-two'
        self.plan['max_examples'] = 1
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'trace count'):
            subject.build(self.plan_path, self.output)

    def test_separate_validation_is_written_separately_and_not_called_independent(self):
        second = self.root / 'second.jsonl'
        self.write_source([{**self.row, 'id': 'other', 'request': 'Different fixture', 'source_sha256': 'b' * 64}], second)
        self.plan['sources'].append(self.source_entry(second, 'validation', 'project-two'))
        self.write_plan()
        subject.build(self.plan_path, self.output)
        training = (self.output / 'train.jsonl').read_text()
        validation = (self.output / 'validation.jsonl').read_text()
        self.assertNotIn('Different fixture', training)
        self.assertIn('Different fixture', validation)
        self.assertFalse(json.loads((self.output / 'manifest.json').read_bytes())['validation_is_independent'])

    def test_same_original_source_cannot_span_splits(self):
        second = self.root / 'second.jsonl'
        self.write_source([{**self.row, 'id': 'other', 'request': 'Different fixture'}], second)
        self.plan['sources'].append(self.source_entry(second, 'validation', 'project-two'))
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'original source spans'):
            subject.build(self.plan_path, self.output)
        self.assertFalse(self.output.exists())

    def test_repository_output_requires_git_ignore(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True, capture_output=True)
        with self.assertRaisesRegex(ValueError, 'Git-ignored'):
            subject.build(self.plan_path, self.output)
        (self.root / '.gitignore').write_text('output/\n')
        subject.build(self.plan_path, self.output)
        self.assertTrue((self.output / 'manifest.json').exists())

    def test_plan_and_source_identity_are_rechecked(self):
        initial = self.source.read_bytes()
        calls = 0
        original = subject.sha
        def mutate(raw):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.source.write_bytes(initial + b' ')
            return original(raw)
        with patch.object(subject, 'sha', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed during preparation'):
                subject.build(self.plan_path, self.output)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
