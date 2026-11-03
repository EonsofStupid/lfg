import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

import assessment_distill

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'safetensors'))
if AVAILABLE:
    import collect_local_teacher as collector
    import select_teacher_requests as selection


@unittest.skipUnless(AVAILABLE, 'torch and safetensors required')
class LocalTeacherCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        self.policy = self.root / 'policy.json'
        self.policy.write_text(json.dumps({'schema_version': 'lfg.private-corpus-screen.v1', 'revision': 'fixture',
            'limits': {'max_policy_bytes': 65536, 'max_plan_bytes': 65536, 'max_source_bytes': 100000,
                       'max_total_source_bytes': 100000, 'max_total_output_bytes': 100000, 'max_records': 100,
                       'max_text_bytes': 10000, 'max_matches_per_record': 100},
            'rules': [{'id': 'test-secret', 'pattern': r'fixture-secret-[0-9]+', 'replacement': '[REDACTED]'}]}))
        rows = []
        for i, project in enumerate(('a', 'a', 'b', 'b')):
            text = f'Fixture request {i} fixture-secret-{i} with other ordinary words.'
            rows.append({'schema_version': 'lfg.screened-text-candidate.v1', 'split': 'train',
                         'review_status': 'unreviewed', 'id': f'id-{i}', 'project': project,
                         'project_group': 'related', 'family': f'family-{i}', 'text': text,
                         'screened_text_sha256': selection.sha(text.encode()), 'source_text_sha256': 'a' * 64,
                         'candidate_file_sha256': 'b' * 64, 'extra': 'allowed'})
        self.train_raw = b''.join(selection.encoded(row) for row in rows)
        (self.dataset / 'train.jsonl').write_bytes(self.train_raw)
        (self.dataset / 'manifest.json').write_bytes(selection.encoded({
            'schema_version': 'lfg.private-corpus-screen-manifest.v1',
            'outputs': {'train': selection.sha(self.train_raw), 'validation': 'c' * 64}}))
        self.selection_config = self.root / 'selection.json'
        self.selection_config.write_text(json.dumps({'schema_version': 'lfg.teacher-request-selection.v1',
            'seed': 'fixture', 'max_manifest_bytes': 65536, 'max_train_bytes': 100000, 'max_record_bytes': 10000,
            'max_candidates': 10, 'max_per_project': 1, 'max_selected': 2, 'min_text_bytes': 1, 'max_text_bytes': 1000,
            'extra': True}))
        self.selected = self.root / 'selected'
        self.collection_config = self.root / 'teacher.json'
        self.config = {'schema_version': 'lfg.local-teacher-collection.v1', 'endpoint': 'http://127.0.0.1:1234/v1/chat/completions',
            'model': 'fixture-model', 'teacher_identity': 'fixture-model-unpinned', 'source_origin': 'owner_chat', 'api_key_env': 'LFG_TEST_TEACHER_KEY',
            'system_prompt': 'Assess the supplied request.', 'required_fields': ['intent', 'complexity'],
            'json_mode': True, 'temperature': 0.0, 'max_tokens': 32, 'timeout_seconds': 2,
            'max_requests': 2, 'max_request_bytes': 2000, 'max_http_response_bytes': 4000,
            'max_response_bytes': 2000, 'max_manifest_bytes': 65536, 'extra': 'allowed'}
        self.collection_config.write_text(json.dumps(self.config))
        self.output = self.root / 'traces'

    def test_selection_rescreens_balances_projects_and_never_opens_validation(self):
        result = selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        self.assertEqual(result['projects'], {'a': 1, 'b': 1})
        self.assertEqual(result['requests'], 2)
        self.assertFalse(result['validation_opened'])
        self.assertFalse(result['test_opened'])
        self.assertNotIn('fixture-secret-', (self.selected / 'requests.jsonl').read_text())
        self.assertFalse((self.dataset / 'validation.jsonl').exists())
        (self.dataset / 'train.jsonl').write_bytes(self.train_raw + b' ')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            selection.select(self.dataset, self.policy, self.selection_config, self.root / 'bad')

    def test_interruption_resumes_and_outputs_preparable_unreviewed_traces(self):
        selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        response = json.dumps({'intent': 'fixture-secret-7', 'complexity': 'unknown', 'extra': 'allowed'})
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}):
            with patch.object(collector, 'ask', side_effect=[response, OSError('fixture interruption')]) as ask:
                with self.assertRaises(OSError):
                    collector.collect(self.selected, self.collection_config, self.policy, self.output)
                self.assertEqual(ask.call_count, 2)
            self.assertFalse((self.output / 'manifest.json').exists())
            with patch.object(collector, 'ask', return_value=response) as ask:
                result = collector.collect(self.selected, self.collection_config, self.policy, self.output, resume=True)
                self.assertEqual(ask.call_count, 1)
            self.assertEqual(result['requests'], 2)
            self.assertFalse(result['test_opened'])
            self.assertFalse(result['promotion_eligible'])
            self.assertNotIn('fixture-secret-', (self.output / 'trace-000000.jsonl').read_text())
            with patch.object(collector, 'ask', side_effect=AssertionError('should not call teacher')):
                self.assertEqual(collector.collect(self.selected, self.collection_config, self.policy, self.output, resume=True), result)
        plan = {'schema_version': 'lfg.assessment-distillation-plan.v1', 'revision': 'fixture', 'task': 'assessment',
                'system_prompt': self.config['system_prompt'], 'authorization_attestation': 'fixture',
                'redaction_attestation': 'fixture', 'forbidden_literals': [], 'include_outcomes': ['unknown'],
                'max_source_bytes': 100000, 'max_total_bytes': 100000, 'max_examples': 10,
                'sources': [{k: v for k, v in item.items() if k != 'records'} for item in result['files']]}
        plan_path = self.output / 'plan.json'
        plan_path.write_bytes(selection.encoded(plan))
        built = assessment_distill.build(plan_path, self.root / 'prepared')
        self.assertEqual(built['counts']['train']['selected'], 2)
        rows = [json.loads(line) for line in (self.root / 'prepared' / 'train.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['metadata']['review_status_claim'] == 'unreviewed' for row in rows))
        self.assertTrue(all(row['metadata']['outcome_claim'] == 'unknown' for row in rows))

    def test_resume_rejects_changed_config_and_corrupt_record(self):
        selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}), patch.object(collector, 'ask', return_value='{"intent":"x","complexity":"unknown"}'):
            collector.collect(self.selected, self.collection_config, self.policy, self.output)
            self.config['temperature'] = 0.5
            self.collection_config.write_text(json.dumps(self.config))
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                collector.collect(self.selected, self.collection_config, self.policy, self.output, resume=True)
            self.config['temperature'] = 0.0
            self.collection_config.write_text(json.dumps(self.config))
            path = self.output / 'records' / '000000.json'
            record = json.loads(path.read_bytes()); record['trace']['response'] = '{}'
            path.write_bytes(selection.encoded(record))
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                collector.collect(self.selected, self.collection_config, self.policy, self.output, resume=True)

    def test_rejects_non_loopback_teacher_and_truncated_responses(self):
        self.config['endpoint'] = 'http://example.com:80/v1/chat/completions'
        self.collection_config.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, 'loopback'):
            collector.config_file(self.collection_config)
        self.config['endpoint'] = 'http://127.0.0.1:1234/v1/chat/completions'
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit):
                return selection.encoded({'model': 'fixture-model', 'choices': [{'finish_reason': 'length',
                        'message': {'content': '{"intent":"x"}'}}]})
        with patch('urllib.request.build_opener') as opener:
            opener.return_value.open.return_value = Response()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                collector.ask(self.config, 'request', 'key')
            opener.return_value.open.side_effect = urllib.error.HTTPError('http://127.0.0.1', 500,
                                                                          'private details', {}, None)
            with self.assertRaises(collector.TeacherHTTPError) as caught:
                collector.ask(self.config, 'request', 'key')
            self.assertEqual(caught.exception.status, 500)
            self.assertNotIn('private details', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
