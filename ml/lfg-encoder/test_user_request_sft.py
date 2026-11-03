"""Synthetic integration tests for the teacherless user-request SFT adapter."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import inventory_codex_sessions as inventory
import source_bound_snapshot
import user_requested_observation
import user_request_sft
import user_request_targets


@unittest.skipUnless(os.name == 'posix', 'private source-bound SFT output is POSIX-only')
class UserRequestSFTTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / 'fixture-project'
        self.project.mkdir()
        self.source = self.root / 'fixture-session.jsonl'
        self.text = 'For planning use Model Z at high effort; avoid Model Q and request Unlisted Model R.'
        self._write_source(self.text)
        status, records = inventory.scan_file(
            self.source, {'fixture-project': inventory.normalize(str(self.project))}, 16_384, 8192)
        self.assertEqual(status, 'scanned')
        self.record = records[0]
        self.policy = Path(__file__).with_name('private-corpus-screen.example.json')
        self.obs_config = {
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
        self.obs_config_path = self.root / 'observation-config.json'
        self.obs_config_path.write_text(json.dumps(self.obs_config), encoding='utf-8')
        self.observation, self.snapshot, self.relations = self._inputs()
        self.observation_path = self.root / 'observation.json'
        self.snapshot_path = self.root / 'snapshot.json'
        self.relations_path = self.root / 'relations.json'
        self.observation_path.write_text(json.dumps(self.observation), encoding='utf-8')
        self.snapshot_path.write_text(json.dumps(self.snapshot), encoding='utf-8')
        self.relations_path.write_text(json.dumps(self.relations), encoding='utf-8')
        self.caller_input_path = self.root / 'caller-input.json'
        self.caller_input_path.write_text(json.dumps({
            'schema_version': user_request_sft.CALLER_INPUT_SCHEMA,
            'spans': self.spans, 'interpretation': self.interpretation,
            'relations': self.relations,
        }), encoding='utf-8')
        config_path = Path(user_request_sft.__file__).with_name('user-request-sft.example.json')
        self.sft_config_path = self.root / 'user-request-sft.json'
        config = json.loads(config_path.read_bytes())
        config.update(max_dataset_bytes=1_000_000, max_record_bytes=200_000, max_request_bytes=4096)
        self.sft_config_path.write_text(json.dumps(config), encoding='utf-8')

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
            self.record, self.spans, self.interpretation, self.obs_config, self.policy)
        relations = {'schema_version': user_request_targets.RELATIONS_SCHEMA,
                     'origin': 'caller_supplied_interpretation', 'review_status': 'unreviewed',
                     'items': [{'model_span_id': 'span-2', 'effort_span_id': 'span-3'},
                               {'model_span_id': 'span-4'}, {'model_span_id': 'span-5'}]}
        return observation, snapshot, relations

    def _build(self, output=None):
        return user_request_sft.build_dataset(
            None, self.snapshot_path, None,
            self.obs_config_path, self.sft_config_path, self.policy,
            output or (self.root / 'dataset.json'), caller_input_path=self.caller_input_path)

    def test_builds_sparse_explicit_targets_and_trainer_reads_separate_task(self):
        result = self._build()
        envelope = json.loads((self.root / 'dataset.json').read_bytes())
        self.assertEqual(result['records'], 1)
        self.assertEqual(envelope['schema_version'], user_request_sft.DATASET_SCHEMA)
        row = envelope['train_records'][0]
        user = json.loads(row['messages'][1]['content'])
        target = json.loads(row['messages'][2]['content'])
        self.assertEqual(target['task'], user_request_sft.TASK)
        self.assertEqual(target['requested'], [
            {'model_id': 'provider/model-z', 'polarity': 'affirmed',
             'effort': 'high', 'effort_polarity': 'affirmed'},
            {'model_id': 'provider/model-q', 'polarity': 'negated'},
        ])
        self.assertEqual([item['model_id'] for item in user['provider_candidates']],
                         ['provider/model-z', 'provider/model-q'])
        self.assertNotIn('provider/model-unlisted', json.dumps(target))
        manifest = envelope['manifest']
        self.assertEqual(manifest['review_status'], 'unreviewed')
        self.assertEqual(manifest['outcome'], 'unknown')
        self.assertEqual(manifest['provider_entitlement'], 'unknown')
        self.assertFalse(manifest['provider_listing_is_entitlement'])
        self.assertEqual(manifest['heldout_separation'], 'unknown')
        self.assertFalse(manifest['promotion_eligible'])
        self.assertNotIn(str(self.source), json.dumps(envelope))
        self.assertNotIn('teacher', row['metadata'])

        import train_owned_assessment_sft as trainer
        train_config = json.loads(Path(trainer.__file__).with_name('owned-user-request-sft.example.json').read_bytes())
        train_config.update(context=2048, batch_size=1, steps=2, checkpoint_every=1,
                            max_parameters=100_000, max_attention_elements=100_000)
        records, _, _, coverage = trainer.read_training(self.root / 'dataset.json', train_config)
        self.assertEqual(coverage['training_task'], user_request_sft.TASK)
        self.assertEqual(coverage['selected'], 1)
        self.assertEqual(coverage['claims'], {'explicit_user_request_records': 1,
                                               'outcome_unknown': 1, 'unreviewed': 1})
        expected_prefix = (train_config['framing']['system_prefix'] + row['messages'][0]['content']
            + train_config['framing']['system_suffix'] + train_config['framing']['user_prefix']
            + row['messages'][1]['content'] + train_config['framing']['user_suffix']
            + train_config['framing']['assistant_prefix']).encode('utf-8')
        self.assertTrue(records[0]['bytes'].startswith(expected_prefix))
        self.assertEqual(records[0]['response_start'], len(expected_prefix))

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'safetensors')),
                         'torch and safetensors required for one-step fixture training')
    def test_user_request_task_runs_one_private_fixture_step(self):
        self._build()
        import owned_student
        import train_owned_assessment_sft as trainer
        from safetensors.torch import save_file
        sha = lambda raw: hashlib.sha256(raw).hexdigest()
        checkpoint = self.root / 'checkpoint'
        checkpoint.mkdir()
        architecture = {'width': 8, 'layers': 2, 'heads': 2, 'conv_kernel': 3,
                        'context': 8, 'ff_multiplier': 2, 'attention_every': 2}
        weights = checkpoint / 'weights.safetensors'
        save_file(owned_student.OwnedByteStudent(architecture).state_dict(), str(weights))
        (checkpoint / 'manifest.json').write_text(json.dumps({
            'schema_version': 'lfg.owned-student-experiment-result.v1', 'architecture': architecture,
            'weights_sha256': sha(weights.read_bytes()),
            'code_sha256': {'owned_student.py': sha(Path(owned_student.__file__).read_bytes())}}))
        train_config = json.loads(Path(trainer.__file__).with_name('owned-user-request-sft.example.json').read_bytes())
        train_config.update(context=2048, batch_size=1, steps=1, checkpoint_every=1,
                            max_parameters=100_000, max_attention_elements=10_000_000,
                            max_dataset_bytes=1_000_000, max_record_bytes=200_000,
                            max_manifest_bytes=1_000_000, max_checkpoint_bytes=10_000_000,
                            max_optimizer_bytes=10_000_000, max_examples=10)
        config_path = self.root / 'owned-user-request-sft.json'
        config_path.write_text(json.dumps(train_config), encoding='utf-8')
        output = self.root / 'trained-user-request-task'
        result = trainer.experiment(self.root / 'dataset.json', checkpoint, config_path, output)
        self.assertEqual(result['schema_version'], 'lfg.owned-user-request-sft-result.v1')
        self.assertEqual(result['training_task'], user_request_sft.TASK)
        self.assertFalse(result['assistant_teacher_claimed'])
        self.assertEqual(result['outcome_claim'], 'unknown')
        self.assertEqual(result['provider_entitlement'], 'unknown')
        self.assertEqual(result['heldout_separation'], 'unknown')
        self.assertFalse(result['runtime_compatible'])
        self.assertFalse(result['deployment_ready'])
        self.assertFalse(result['promotion_eligible'])

    def test_ambiguous_provider_match_is_not_emitted_as_a_model_target(self):
        self._write_cache(duplicate_display_name=True)
        self.snapshot = source_bound_snapshot.build_snapshot(
            self.record, self.cache, split_policy='exploratory_train', screen_policy_path=self.policy)
        self.snapshot_path.write_text(json.dumps(self.snapshot), encoding='utf-8')
        self.assertFalse((self.root / 'dataset.json').exists())
        result = self._build()
        envelope = json.loads((self.root / 'dataset.json').read_bytes())
        target = json.loads(envelope['train_records'][0]['messages'][2]['content'])
        self.assertEqual(target['requested'], [{'model_id': 'provider/model-q', 'polarity': 'negated'}])
        self.assertNotIn('provider/model-z', json.dumps(target))
        self.assertEqual(result['heldout_separation'], 'unknown')

    def test_changed_provider_cache_and_changed_source_are_rejected(self):
        self._write_cache(duplicate_display_name=True)
        with self.assertRaisesRegex(ValueError, 'provider cache changed'):
            self._build()
        self._write_cache()
        self.snapshot = source_bound_snapshot.build_snapshot(
            self.record, self.cache, split_policy='exploratory_train', screen_policy_path=self.policy)
        self.snapshot_path.write_text(json.dumps(self.snapshot), encoding='utf-8')
        self._write_source(self.text + ' changed')
        with self.assertRaisesRegex(ValueError, 'source identity|source request changed'):
            self._build()
        self.assertFalse((self.root / 'dataset.json').exists())

    def test_existing_assessment_sft_reader_keeps_legacy_schema(self):
        import assessment_distill
        import train_owned_assessment_sft as trainer
        request_file = self.root / 'legacy-teacher-trace.jsonl'
        trace = {'id': 'fixture-assessment', 'request': 'Fixture request',
                 'response': '{"intent":"fixture"}', 'teacher': 'fixture-teacher',
                 'review_status': 'unreviewed', 'outcome': 'unknown',
                 'source_ref': 'fixture-source', 'source_sha256': 'a' * 64}
        request_file.write_text(json.dumps(trace) + '\n', encoding='utf-8')
        plan = {'schema_version': assessment_distill.PLAN_SCHEMA, 'revision': 'legacy-fixture',
                'task': 'assessment-recommendation', 'system_prompt': 'Assess this fixture.',
                'authorization_attestation': 'fixture-only', 'redaction_attestation': 'fixture-only',
                'forbidden_literals': [], 'include_outcomes': ['unknown'],
                'max_source_bytes': 8192, 'max_total_bytes': 8192, 'max_examples': 1,
                'sources': [{'path': request_file.name,
                    'sha256': hashlib.sha256(request_file.read_bytes()).hexdigest(),
                    'project': 'fixture-project', 'project_group': 'fixture-group',
                    'split': 'train', 'origin': 'owner_chat'}]}
        plan_path = self.root / 'legacy-plan.json'
        plan_path.write_text(json.dumps(plan), encoding='utf-8')
        legacy_dataset = self.root / 'legacy-assessment-dataset'
        assessment_distill.build(plan_path, legacy_dataset)
        config = json.loads(Path(trainer.__file__).with_name('owned-assessment-sft.example.json').read_bytes())
        config.update(context=256, batch_size=1, max_examples=2)
        records, _, _, coverage = trainer.read_training(legacy_dataset, config)
        self.assertEqual(len(records), 1)
        self.assertNotIn('training_task', coverage)
        self.assertEqual(coverage['claims'], {'outcome:unknown': 1, 'review_status:unreviewed': 1})


if __name__ == '__main__':
    unittest.main()
