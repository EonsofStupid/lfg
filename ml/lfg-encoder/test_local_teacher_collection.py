import importlib.util
import json
import os
import hashlib
from pathlib import Path
import tempfile
import unittest
import urllib.error
import unicodedata
from unittest.mock import patch

import assessment_distill
import inventory_codex_sessions as codex_inventory
import source_bound_snapshot

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

    def test_collector_rejects_heldout_manifest_before_opening_requests(self):
        selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        context_text = 'should not be opened'
        context_file = self.root / 'project-context.json'
        context_file.write_text(json.dumps({'schema_version': 'lfg.project-training-context.v1',
            'projects': {'a': {'text': context_text, 'sha256': hashlib.sha256(context_text.encode()).hexdigest()}}}))
        manifest_path = self.selected / 'manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        manifest['validation_opened'] = True
        manifest_path.write_bytes(selection.encoded(manifest))
        opened = []
        original_read = collector.screen._read
        def record_read(path, *args, **kwargs):
            opened.append(Path(path).name)
            return original_read(path, *args, **kwargs)
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}), \
                patch.object(collector.screen, '_read', side_effect=record_read), patch.object(
                collector.project_context, 'load', side_effect=AssertionError('must not open context')) as load_context:
            with self.assertRaisesRegex(ValueError, 'selected request identity'):
                collector.collect(self.selected, self.collection_config, self.policy, self.output,
                                  project_context_path=context_file)
        load_context.assert_not_called()
        self.assertNotIn('requests.jsonl', opened)
        self.assertFalse(self.output.exists())

    def test_collector_rejects_non_train_rows_before_opening_context(self):
        selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        request_path = self.selected / 'requests.jsonl'
        rows = [json.loads(line) for line in request_path.read_bytes().splitlines()]
        rows[0]['split'] = 'validation'
        request_raw = b''.join(selection.encoded(row) for row in rows)
        request_path.write_bytes(request_raw)
        manifest_path = self.selected / 'manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        manifest['requests_sha256'] = selection.sha(request_raw)
        manifest_path.write_bytes(selection.encoded(manifest))
        context_file = self.root / 'project-context.json'
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}), patch.object(
                collector.project_context, 'load', side_effect=AssertionError('must not open context')) as load_context:
            with self.assertRaisesRegex(ValueError, 'invalid selected train request'):
                collector.collect(self.selected, self.collection_config, self.policy, self.output,
                                  project_context_path=context_file)
        load_context.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_collector_rechecks_selection_manifest_after_request_read(self):
        selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        manifest_path = self.selected / 'manifest.json'
        original_read = collector.screen._read
        changed = False
        def race_manifest(path, *args, **kwargs):
            nonlocal changed
            value = original_read(path, *args, **kwargs)
            if Path(path).name == 'requests.jsonl' and not changed:
                manifest_path.write_bytes(selection.encoded({'schema_version': 'lfg.teacher-request-selection-result.v1',
                    'requests_sha256': '0' * 64, 'validation_opened': False, 'test_opened': False}))
                changed = True
            return value
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}), patch.object(
                collector.screen, '_read', side_effect=race_manifest):
            with self.assertRaisesRegex(ValueError, 'selected request identity'):
                collector.collect(self.selected, self.collection_config, self.policy, self.output)
        self.assertTrue(changed)
        self.assertFalse(self.output.exists())

    def test_selection_can_exclude_requests_collected_in_an_earlier_batch(self):
        config = json.loads(self.selection_config.read_text())
        config['exclude_ids'] = ['id-0', 'id-1']
        self.selection_config.write_text(json.dumps(config))
        result = selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        self.assertEqual(result['requests'], 1)
        selected = [json.loads(line) for line in (self.selected / 'requests.jsonl').read_text().splitlines()]
        self.assertNotIn(selected[0]['id'], config['exclude_ids'])

    def test_split_preserves_and_revalidates_completed_records(self):
        import split_teacher_collection
        selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        context_text = 'bounded project context'
        context_file = self.root / 'project-context.json'
        context_file.write_text(json.dumps({'schema_version': 'lfg.project-training-context.v1',
            'projects': {name: {'text': context_text, 'sha256': hashlib.sha256(context_text.encode()).hexdigest()}
                         for name in ('a', 'b')}}))
        response = '{"intent":"x","complexity":"unknown"}'
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}):
            with patch.object(collector, 'ask', side_effect=[response, ValueError('incomplete')]):
                with self.assertRaises(ValueError):
                    collector.collect(self.selected, self.collection_config, self.policy, self.output,
                                      project_context_path=context_file)
            dest = self.root / 'split'
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                split_teacher_collection.split(self.selected, self.collection_config, self.policy, self.output,
                                                self.root / 'split-without-context')
            report = split_teacher_collection.split(self.selected, self.collection_config, self.policy, self.output, dest,
                                                    project_context_path=context_file)
            self.assertEqual(report['completed_prefix'], 1)
            self.assertEqual(report['remaining'], 0)
            with patch.object(collector, 'ask', side_effect=AssertionError('must reuse saved record')):
                result = collector.collect(dest / 'prefix-selection', self.collection_config, self.policy,
                                           dest / 'prefix-output', resume=True, project_context_path=context_file)
            self.assertEqual(result['requests'], 1)
            self.assertFalse(result['promotion_eligible'])

    def test_selection_excludes_normalized_text_from_prior_batches(self):
        config = json.loads(self.selection_config.read_text())
        row = json.loads(self.train_raw.splitlines()[0])
        policy, rules, _ = selection.screen.load_policy(self.policy, 65536)
        screened, _ = selection.screen._screen(row['text'], rules, policy['limits'])
        normalized = ' '.join(unicodedata.normalize('NFKC', screened).casefold().split())
        config['exclude_normalized_sha256'] = [hashlib.sha256(normalized.encode()).hexdigest()]
        self.selection_config.write_text(json.dumps(config))
        selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        selected = [json.loads(line) for line in (self.selected / 'requests.jsonl').read_text().splitlines()]
        self.assertNotIn(row['id'], [item['id'] for item in selected])

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

    def test_request_extensions_reach_teacher_and_cannot_override_inputs(self):
        self.config['request_extensions'] = {'chat_template_kwargs': {'enable_thinking': False}}
        self.collection_config.write_text(json.dumps(self.config))
        config, _ = collector.config_file(self.collection_config)
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit):
                return selection.encoded({'model': 'fixture-model', 'choices': [{'finish_reason': 'stop',
                    'message': {'content': '{"intent":"x","complexity":"unknown"}'}}]})
        with patch('urllib.request.build_opener') as opener:
            opener.return_value.open.return_value = Response()
            collector.ask(config, 'request', 'key')
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(json.loads(request.data)['chat_template_kwargs'], {'enable_thinking': False})
        self.config['request_extensions'] = {'messages': []}
        self.collection_config.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, 'reserved'):
            collector.config_file(self.collection_config)

    def test_project_context_bytes_are_sent_and_saved_for_sft(self):
        selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        project = 'project facts'
        context_file = self.root / 'project-context.json'
        context_file.write_text(json.dumps({'schema_version': 'lfg.project-training-context.v1',
            'top_extension': 'file-retained',
            'projects': {'a': {'text': project, 'sha256': hashlib.sha256(project.encode()).hexdigest(),
                               'custom_extension': 'preserve-me'},
                         'b': {'text': project, 'sha256': hashlib.sha256(project.encode()).hexdigest(),
                               'custom_extension': 'preserve-me'}}}))
        sent = []
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}), patch.object(
                collector, 'ask', side_effect=lambda config, text, key: sent.append(
                    (config['system_prompt'], text)) or
                '{"intent":"x","complexity":"unknown"}'):
            report = collector.collect(self.selected, self.collection_config, self.policy, self.output,
                                       project_context_path=context_file)
        self.assertEqual(report['requests'], 2)
        traces = [json.loads(path.read_text()) for path in sorted((self.output / 'records').glob('*.json'))]
        rendered = {trace['trace']['request'] for trace in traces}
        self.assertEqual(rendered, {text for _, text in sent})
        self.assertEqual(len(rendered), 2)
        self.assertTrue(all(system == self.config['system_prompt'] for system, _ in sent))
        self.assertTrue(all(text.startswith(project + '\n\nFixture request ') for text in rendered))
        self.assertTrue(all(trace['trace']['project_context_sha256'] == hashlib.sha256(project.encode()).hexdigest()
                            for trace in traces))
        self.assertTrue(all(trace['trace']['project_context_extensions'] == {'custom_extension': 'preserve-me'}
                            for trace in traces))
        self.assertTrue(all(trace['trace']['project_context_file_extensions'] == {'top_extension': 'file-retained'}
                            for trace in traces))
        self.assertTrue(all(trace['trace']['teacher_system_sha256'] ==
                            hashlib.sha256(self.config['system_prompt'].encode()).hexdigest() for trace in traces))
        plan_path = self.output / 'distill-plan.json'
        plan_path.write_text(json.dumps({'schema_version': assessment_distill.PLAN_SCHEMA, 'revision': 'fixture',
            'task': 'context-contract', 'system_prompt': self.config['system_prompt'],
            'authorization_attestation': 'fixture', 'redaction_attestation': 'fixture',
            'forbidden_literals': [], 'include_outcomes': ['unknown'], 'max_source_bytes': 65536,
            'max_total_bytes': 131072, 'max_examples': 10,
            'sources': [{key: item[key] for key in ('path', 'sha256', 'project', 'project_group', 'split', 'origin')}
                        for item in report['files']]}))
        prepared = self.root / 'prepared'
        assessment_distill.build(plan_path, prepared)
        prepared_rows = [json.loads(line) for line in (prepared / 'train.jsonl').read_text().splitlines()]
        self.assertEqual({row['messages'][1]['content'] for row in prepared_rows}, rendered)
        self.assertTrue(all(row['metadata']['trace_extensions']['project_context_sha256']
                            == hashlib.sha256(project.encode()).hexdigest() for row in prepared_rows))
        self.assertTrue(all(row['metadata']['trace_extensions']['project_context_extensions']
                            == {'custom_extension': 'preserve-me'} for row in prepared_rows))
        self.assertTrue(all(row['metadata']['trace_extensions']['project_context_file_extensions']
                            == {'top_extension': 'file-retained'} for row in prepared_rows))
        self.assertTrue(all(row['metadata']['trace_extensions']['teacher_system_sha256'] ==
                            hashlib.sha256(self.config['system_prompt'].encode()).hexdigest() for row in prepared_rows))
        import train_owned_assessment_sft
        training_config = json.loads(Path(train_owned_assessment_sft.__file__).with_name(
            'owned-assessment-sft.example.json').read_text())
        training_rows, _, _, _ = train_owned_assessment_sft.read_training(prepared, training_config)
        expected_prefix = (training_config['framing']['system_prefix'] + self.config['system_prompt']
                           + training_config['framing']['system_suffix'] + training_config['framing']['user_prefix']).encode()
        self.assertTrue(all(row['bytes'].startswith(expected_prefix + text.encode() +
                            training_config['framing']['user_suffix'].encode() +
                            training_config['framing']['assistant_prefix'].encode()) for row, text in zip(
                                training_rows, sorted(rendered), strict=True)))

    def _source_bound_inputs(self):
        source_root = self.root / 'bound-project'
        source_root.mkdir()
        session = self.root / 'bound-session.jsonl'
        message = {'type': 'response_item', 'payload': {'type': 'message', 'role': 'user',
                   'content': [{'type': 'input_text', 'text': 'Please assess fixture-model.'}],
                   'internal_chat_message_metadata_passthrough': {'content_item_kinds': ['user.text']}}}
        session.write_text(json.dumps({'type': 'session_meta', 'payload': {'cwd': str(source_root)}}) + '\n'
                           + json.dumps(message) + '\n', encoding='utf-8')
        state, rows = codex_inventory.scan_file(session, {'a': codex_inventory.normalize(str(source_root))},
                                                65536, 65536)
        self.assertEqual(state, 'scanned')
        record = rows[0]
        provider = self.root / 'provider-cache.json'
        provider.write_text(json.dumps({'fetched_at': '2026-11-03T10:00:00Z', 'models': [
            {'slug': 'fixture-model', 'display_name': 'Fixture Model', 'description': 'Fixture teacher model',
             'visibility': 'list', 'supported_reasoning_levels': [{'effort': 'low'}]}]}), encoding='utf-8')
        value = source_bound_snapshot.build_snapshot(record, provider, split_policy='exploratory_train',
                                                     screen_policy_path=self.policy)
        manifest, row, context, request_raw, manifest_raw, context_raw, projection_raw, bundle_raw = (
            source_bound_snapshot.prepare_single_request(value, policy_path=self.policy,
                                                        max_request_bytes=2000, max_context_bytes=2000))
        self.selected.mkdir()
        (self.selected / 'requests.jsonl').write_bytes(request_raw)
        (self.selected / 'manifest.json').write_bytes(manifest_raw)
        context_path = self.root / 'bound-context.json'
        context_path.write_bytes(context_raw)
        snapshot_path = self.root / 'snapshot-projection.json'
        snapshot_path.write_bytes(projection_raw)
        bundle_path = self.root / 'bundle.json'
        bundle_path.write_bytes(bundle_raw)
        return context_path, snapshot_path, bundle_path, row, context

    def test_source_bound_collector_revalidates_copy_and_preserves_link(self):
        context_path, snapshot_path, bundle_path, row, context = self._source_bound_inputs()
        sent = []
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}), patch.object(
                collector, 'ask', side_effect=lambda config, text, key: sent.append(text) or
                '{"intent":"x","complexity":"unknown"}'):
            result = collector.collect(self.selected, self.collection_config, self.policy, self.output,
                                       project_context_path=context_path,
                                       source_bound_snapshot_path=snapshot_path,
                                       source_bound_bundle_path=bundle_path)
        self.assertEqual(result['requests'], 1)
        trace = json.loads(next((self.output / 'records').glob('*.json')).read_text())['trace']
        self.assertEqual(len(sent), 1)
        self.assertTrue(trace['source_bound_snapshot_sha256'])
        self.assertTrue(trace['source_bound_bundle_sha256'])
        self.assertEqual(trace['project_context_extensions']['source_bound_snapshot_sha256'],
                         row['source_bound_snapshot_sha256'])
        self.assertEqual(trace['project_context_extensions']['source_text_sha256_claim'],
                         row['source_text_sha256_claim'])
        self.assertEqual(context['projects']['a']['heldout_separation'], 'unknown')
        self.assertEqual(trace['outcome'], 'unknown')
        self.assertEqual(trace['review_status'], 'unreviewed')

    def test_tampered_source_bound_bundle_fails_before_teacher_call(self):
        context_path, snapshot_path, bundle_path, _row, _context = self._source_bound_inputs()
        bundle = json.loads(bundle_path.read_bytes())
        bundle['requests_sha256'] = '0' * 64
        bundle_path.write_text(json.dumps(bundle), encoding='utf-8')
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}), patch.object(
                collector, 'ask', side_effect=AssertionError('must not call teacher')):
            with self.assertRaisesRegex(ValueError, 'bundle digest'):
                collector.collect(self.selected, self.collection_config, self.policy, self.output,
                                  project_context_path=context_path,
                                  source_bound_snapshot_path=snapshot_path,
                                  source_bound_bundle_path=bundle_path)
        self.assertFalse(self.output.exists())

    def test_source_bound_non_train_selection_never_opens_snapshot_or_context(self):
        context_path, snapshot_path, bundle_path, _row, _context = self._source_bound_inputs()
        request_path = self.selected / 'requests.jsonl'
        request = json.loads(request_path.read_bytes())
        request['split'] = 'test'
        request_raw = selection.encoded(request)
        request_path.write_bytes(request_raw)
        manifest_path = self.selected / 'manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        manifest['requests_sha256'] = selection.sha(request_raw)
        manifest_path.write_bytes(selection.encoded(manifest))
        opened = []
        original = collector.screen._read
        def tracked(path, *args, **kwargs):
            opened.append(Path(path).resolve())
            return original(path, *args, **kwargs)
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}), \
                patch.object(collector.screen, '_read', side_effect=tracked), \
                patch.object(collector, 'ask', side_effect=AssertionError('must not call teacher')):
            with self.assertRaisesRegex(ValueError, 'invalid selected train request'):
                collector.collect(self.selected, self.collection_config, self.policy, self.output,
                                  project_context_path=context_path,
                                  source_bound_snapshot_path=snapshot_path,
                                  source_bound_bundle_path=bundle_path)
        self.assertNotIn(snapshot_path.resolve(), opened)
        self.assertNotIn(context_path.resolve(), opened)
        self.assertFalse(self.output.exists())

    def test_source_bound_copy_is_rechecked_immediately_before_teacher_call(self):
        context_path, snapshot_path, bundle_path, _row, _context = self._source_bound_inputs()
        real_validate = collector.source_snapshot.validate_staged_artifacts
        calls = 0
        attempts = 0
        sent = []
        def mutate_after_preflight(*args, **kwargs):
            nonlocal calls, attempts
            attempts += 1
            result = real_validate(*args, **kwargs)
            calls += 1
            if calls == 1:
                bundle = json.loads(bundle_path.read_bytes())
                bundle['requests_sha256'] = '0' * 64
                bundle_path.write_text(json.dumps(bundle), encoding='utf-8')
            return result
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}), \
                patch.object(collector.source_snapshot, 'validate_staged_artifacts',
                             side_effect=mutate_after_preflight), \
                patch.object(collector, 'ask', side_effect=lambda *args: sent.append(True)):
            with self.assertRaisesRegex(ValueError, 'bundle digest'):
                collector.collect(self.selected, self.collection_config, self.policy, self.output,
                                  project_context_path=context_path,
                                  source_bound_snapshot_path=snapshot_path,
                                  source_bound_bundle_path=bundle_path)
        self.assertEqual(attempts, 2)
        self.assertEqual(calls, 1)
        self.assertEqual(sent, [])
        self.assertFalse((self.output / 'records' / '000000.json').exists())

    def test_project_context_requires_redaction_before_teacher_access(self):
        selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        project = 'fixture-secret-123'
        context_file = self.root / 'project-context.json'
        context_file.write_text(json.dumps({'schema_version': 'lfg.project-training-context.v1',
            'projects': {'a': {'text': project, 'sha256': hashlib.sha256(project.encode()).hexdigest(),
                               'private_marker': 'fixture-secret-123'},
                         'b': {'text': project, 'sha256': hashlib.sha256(project.encode()).hexdigest(),
                               'private_marker': 'fixture-secret-123'}}}))
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}), patch.object(
                collector, 'ask', side_effect=AssertionError('must reject before teacher call')):
            with self.assertRaisesRegex(ValueError, 'additional screening'):
                collector.collect(self.selected, self.collection_config, self.policy, self.output,
                                  project_context_path=context_file)

    def test_missing_project_context_fails_before_any_teacher_call(self):
        selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        project = 'bounded project context'
        context_file = self.root / 'project-context.json'
        context_file.write_text(json.dumps({'schema_version': 'lfg.project-training-context.v1',
            'projects': {'a': {'text': project, 'sha256': hashlib.sha256(project.encode()).hexdigest()}}}))
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}), patch.object(
                collector, 'ask', side_effect=AssertionError('must reject before teacher call')):
            with self.assertRaisesRegex(ValueError, 'no supplied context'):
                collector.collect(self.selected, self.collection_config, self.policy, self.output,
                                  project_context_path=context_file)
        self.assertFalse(self.output.exists())

    def test_credential_shaped_context_extension_key_is_screened_before_teacher_access(self):
        selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        policy = json.loads(self.policy.read_text())
        policy['rules'].append({'id': 'credential-shaped-key', 'pattern': r'(?i)api[_-]?key',
                                'replacement': '[REDACTED]'})
        self.policy.write_text(json.dumps(policy))
        project = 'bounded project context'
        context_file = self.root / 'project-context.json'
        context_file.write_text(json.dumps({'schema_version': 'lfg.project-training-context.v1',
            'api_key': 'dummy marker',
            'projects': {'a': {'text': project, 'sha256': hashlib.sha256(project.encode()).hexdigest()},
                         'b': {'text': project, 'sha256': hashlib.sha256(project.encode()).hexdigest()}}}))
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}), patch.object(
                collector, 'ask', side_effect=AssertionError('must reject before teacher call')):
            with self.assertRaisesRegex(ValueError, 'additional screening'):
                collector.collect(self.selected, self.collection_config, self.policy, self.output,
                                  project_context_path=context_file)
        self.assertFalse(self.output.exists())

    def test_skip_policy_records_bad_response_and_resumes_valid_ones(self):
        selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        self.config['on_incomplete'] = 'skip'
        self.collection_config.write_text(json.dumps(self.config))
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}):
            with patch.object(collector, 'ask', side_effect=[ValueError('incomplete'), '{"intent":"x","complexity":"unknown"}']):
                result = collector.collect(self.selected, self.collection_config, self.policy, self.output)
            self.assertEqual(result['requested_requests'], 2)
            self.assertEqual(result['failed_requests'], 1)
            self.assertEqual(result['requests'], 1)
            self.assertTrue((self.output / 'failures' / '000000.json').is_file())
            with patch.object(collector, 'ask', side_effect=AssertionError('must reuse outcomes')):
                self.assertEqual(collector.collect(self.selected, self.collection_config, self.policy, self.output, resume=True), result)

    def test_skip_policy_does_not_hide_server_errors(self):
        selection.select(self.dataset, self.policy, self.selection_config, self.selected)
        self.config['on_incomplete'] = 'skip'
        self.collection_config.write_text(json.dumps(self.config))
        with patch.dict(os.environ, {'LFG_TEST_TEACHER_KEY': 'test-only'}), patch.object(
                collector, 'ask', side_effect=collector.TeacherHTTPError(503)):
            with self.assertRaises(collector.TeacherHTTPError):
                collector.collect(self.selected, self.collection_config, self.policy, self.output)


if __name__ == '__main__':
    unittest.main()
