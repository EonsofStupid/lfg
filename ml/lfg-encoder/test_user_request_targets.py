"""Synthetic-only source-bound model/effort request-target tests."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import inventory_codex_sessions as inventory
import source_bound_snapshot
import user_requested_observation
import user_request_targets as targets


@unittest.skipUnless(os.name == 'posix', 'source-bound observations are fail-closed on Windows')
class UserRequestTargetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'fixture-project'
        self.project.mkdir()
        self.source = self.root / 'fixture-session.jsonl'
        self.text = 'For planning 😀 use Model Z at high effort; avoid Model Q and request Unlisted Model R.'
        self._write_source(self.text)
        status, records = inventory.scan_file(
            self.source, {'fixture-project': inventory.normalize(str(self.project))}, 16_384, 8192)
        self.assertEqual(status, 'scanned')
        self.record = records[0]
        self.policy = Path(__file__).with_name('private-corpus-screen.example.json')
        self.config = {
            'schema_version': user_requested_observation.CONFIG_SCHEMA,
            'phase_labels': ['planning'], 'claim_polarities': ['affirmed', 'negated', 'quoted', 'uncertain'],
            'max_source_prefix_bytes': 16_384, 'max_line_bytes': 8192, 'max_user_message_bytes': 2048,
            'max_span_count': 8, 'max_span_bytes': 256, 'max_extension_bytes': 4096,
            'max_extension_nodes': 128, 'max_extension_depth': 8, 'max_extension_string_bytes': 1024,
            'max_observation_bytes': 32_768, 'max_reserved_manifest_bytes': 4096,
            'max_reserved_project_labels': 32, 'max_claims_per_field': 8,
        }
        self.spans = self._spans()
        self.interpretation = {
            'phase_label': 'planning', 'phase_span_id': 'phase',
            'model_claims': [
                {'text': 'Model Z', 'span_id': 'model-z', 'polarity': 'affirmed'},
                {'text': 'Model Q', 'span_id': 'model-q', 'polarity': 'negated'},
                {'text': 'Unlisted Model R', 'span_id': 'model-unlisted', 'polarity': 'affirmed'},
            ],
            'effort_claims': [{'text': 'high', 'span_id': 'effort-high', 'polarity': 'affirmed'}],
        }
        self.cache = self.root / 'provider-cache.json'
        self._write_cache()

    def _write_source(self, text):
        header = {'type': 'session_meta', 'payload': {'cwd': str(self.project)}}
        message = {'type': 'response_item', 'payload': {
            'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': text}],
            'internal_chat_message_metadata_passthrough': {'content_item_kinds': ['user.text']},
        }}
        self.source.write_bytes((json.dumps(header, ensure_ascii=False) + '\n'
                                 + json.dumps(message, ensure_ascii=False) + '\n').encode('utf-8'))

    def _spans(self):
        spans = []
        raw = self.text.encode('utf-8')
        for identifier, field, value in (
                ('phase', 'phase', 'planning'), ('model-z', 'model', 'Model Z'),
                ('effort-high', 'effort', 'high'), ('model-q', 'model', 'Model Q'),
                ('model-unlisted', 'model', 'Unlisted Model R')):
            start = raw.index(value.encode('utf-8'))
            spans.append({'id': identifier, 'field': field, 'start_byte': start,
                          'end_byte': start + len(value.encode('utf-8'))})
        return spans

    def _write_cache(self, duplicate_display_name=False):
        models = [
            {'slug': 'provider/model-z', 'display_name': 'Model Z', 'visibility': 'list',
             'supported_reasoning_levels': [{'effort': 'high'}, {'effort': 'low'}]},
            {'slug': 'provider/model-q', 'display_name': 'Model Q', 'visibility': 'list',
             'supported_reasoning_levels': [{'effort': 'medium'}]},
        ]
        if duplicate_display_name:
            models.append({'slug': 'provider/model-z2', 'display_name': 'Model Z', 'visibility': 'list',
                           'supported_reasoning_levels': [{'effort': 'high'}]})
        self.cache.write_text(json.dumps({'fetched_at': '2026-11-03T12:00:00Z', 'models': models}),
                              encoding='utf-8')

    def _inputs(self):
        snapshot = source_bound_snapshot.build_snapshot(
            self.record, self.cache, split_policy='exploratory_train', screen_policy_path=self.policy)
        observation = user_requested_observation.build_observation(
            self.record, self.spans, self.interpretation, self.config, self.policy)
        relations = {
            'schema_version': targets.RELATIONS_SCHEMA,
            'origin': 'caller_supplied_interpretation',
            'review_status': 'unreviewed',
            'items': [
                {'model_span_id': 'span-2', 'effort_span_id': 'span-3'},
                {'model_span_id': 'span-4'},
                {'model_span_id': 'span-5'},
            ],
        }
        return snapshot, observation, relations

    def _import(self):
        snapshot, observation, relations = self._inputs()
        return targets.import_request_targets(observation, snapshot, relations, self.config, self.policy)

    def test_builds_partial_unreviewed_request_targets_without_synthetic_negatives(self):
        value = self._import()
        self.assertEqual(value['schema_version'], targets.SCHEMA)
        self.assertEqual(value['target_semantics'], 'explicit_request_for_this_prompt')
        self.assertEqual(value['origin'], 'caller_supplied_interpretation')
        self.assertEqual(value['review_status'], 'unreviewed')
        self.assertEqual(value['source_split'], {'assignment': 'unknown', 'heldout_separation': 'unknown'})
        self.assertEqual(value['state'], {
            'provider_entitlement': 'unknown', 'execution': 'unknown', 'outcome': 'unknown',
            'total_agent_count': None, 'promotion_eligible': False,
        })
        rows = value['targets']
        self.assertEqual(rows[0]['model_id'], 'provider/model-z')
        self.assertEqual(rows[0]['requested_model_target'], 1)
        self.assertEqual(rows[0]['effort'], 'high')
        self.assertEqual(rows[0]['requested_effort_target'], 1)
        self.assertEqual(rows[1]['model_id'], 'provider/model-q')
        self.assertEqual(rows[1]['requested_model_target'], 0)
        self.assertEqual(rows[1]['effort_match_status'], 'not_supplied')
        self.assertIsNone(rows[1]['requested_effort_target'])
        self.assertEqual(rows[2]['model_match_status'], 'unmatched')
        self.assertIsNone(rows[2]['model_id'])
        self.assertIsNone(rows[2]['requested_model_target'])
        encoded = json.dumps(value, ensure_ascii=False)
        self.assertNotIn(self.text, encoded)
        self.assertNotIn(str(self.source), encoded)
        self.assertNotIn('Unlisted Model R', encoded)
        self.assertFalse(value['provider_observation']['listing_is_entitlement'])
        targets.validate_target_record(value)
        altered = json.loads(json.dumps(value))
        altered['state']['provider_entitlement'] = 'available'
        with self.assertRaisesRegex(ValueError, 'state must remain unknown'):
            targets.validate_target_record(altered)

    def test_duplicate_display_names_are_ambiguous_and_unlabeled(self):
        self._write_cache(duplicate_display_name=True)
        value = self._import()
        row = value['targets'][0]
        self.assertEqual(row['model_match_status'], 'ambiguous')
        self.assertIsNone(row['model_id'])
        self.assertIsNone(row['requested_model_target'])
        self.assertEqual(row['effort_match_status'], 'model_unresolved')
        self.assertIsNone(row['requested_effort_target'])

    def test_stale_provider_cache_is_rejected(self):
        snapshot, observation, relations = self._inputs()
        self._write_cache(duplicate_display_name=True)
        with self.assertRaisesRegex(ValueError, 'provider cache changed'):
            targets.import_request_targets(observation, snapshot, relations, self.config, self.policy)

    def test_test_assignment_is_rejected_before_source_revalidation(self):
        snapshot, observation, relations = self._inputs()
        snapshot['split']['assignment'] = 'test'
        from unittest import mock
        with mock.patch.object(source_bound_snapshot, 'revalidate_snapshot',
                               side_effect=AssertionError('source should not open')) as revalidate:
            with self.assertRaisesRegex(ValueError, 'only an explicitly selected exploratory train source'):
                targets.import_request_targets(observation, snapshot, relations, self.config, self.policy)
        revalidate.assert_not_called()

    def test_bad_span_relation_and_duplicate_assignment_fail_closed(self):
        snapshot, observation, relations = self._inputs()
        relations['items'][0]['effort_span_id'] = 'span-5'
        with self.assertRaisesRegex(ValueError, 'existing caller-interpreted'):
            targets.import_request_targets(observation, snapshot, relations, self.config, self.policy)
        relations['items'][0]['effort_span_id'] = 'span-3'
        relations['items'].append({'model_span_id': 'span-2'})
        with self.assertRaisesRegex(ValueError, 'multiple model-effort relations'):
            targets.import_request_targets(observation, snapshot, relations, self.config, self.policy)


if __name__ == '__main__':
    unittest.main()
