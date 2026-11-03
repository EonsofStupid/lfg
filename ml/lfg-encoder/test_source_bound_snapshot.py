import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import inventory_codex_sessions as inventory
import source_bound_snapshot as snapshot


class SourceBoundSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        project = self.root / 'project'
        project.mkdir()
        self.session = self.root / 'session.jsonl'
        message = {'type': 'response_item', 'payload': {'type': 'message', 'role': 'user',
                   'content': [{'type': 'input_text', 'text': 'Review the Terra project plan.'}],
                   'internal_chat_message_metadata_passthrough': {'content_item_kinds': ['user.text']}}}
        raw = (json.dumps({'type': 'session_meta', 'payload': {'cwd': str(project)}}) + '\n'
               + json.dumps(message) + '\n').encode()
        self.session.write_bytes(raw)
        status, records = inventory.scan_file(self.session, {'lfg': inventory.normalize(str(project))}, 4096, 4096)
        self.assertEqual(status, 'scanned')
        self.record = records[0]
        self.cache = self.root / 'models_cache.json'
        self.write_cache()

    def write_cache(self, extra=None):
        value = {'fetched_at': '2026-11-03T08:22:51Z',
                 'models': [
                     {'slug': 'gpt-6.1-terra', 'display_name': 'Terra', 'description': 'listed',
                      'visibility': 'list', 'supported_reasoning_levels': [{'effort': 'xhigh'}],
                      'vendor_extension': {'generation': 7}},
                     {'slug': 'hidden', 'visibility': 'hide'},
                 ], 'cache_extension': {'revision': 'r1'}}
        if extra:
            value.update(extra)
        self.cache.write_text(json.dumps(value), encoding='utf-8')

    def test_snapshot_verifies_exact_line_and_keeps_unknown_claims_unknown(self):
        self.write_cache({'models': [{'slug': 'gpt-6.1-terra', 'display_name': 'Terra',
                                      'description': 'listed', 'visibility': 'list',
                                      'supported_reasoning_levels': [{'effort': 'xhigh'}],
                                      'vendor_extension': {'generation': 7,
                                                           'note': 'password=extensionSecret987654'}}]})
        value = snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train',
                                        observed_at='2026-11-03T09:00:00Z')
        self.assertEqual(value['schema_version'], snapshot.SCHEMA)
        self.assertTrue(value['source']['verified'])
        self.assertEqual(value['source']['project_association'], 'inventory_label_unverified')
        self.assertEqual(value['source']['user_text_sha256'], self.record['text_sha256'])
        self.assertEqual(value['source']['current_whole_file_sha256'], 'unknown')
        self.assertEqual(value['split']['assignment'], 'train')
        self.assertEqual(value['split']['heldout_separation'], 'unknown')
        self.assertEqual(value['access_claims']['subscription_entitlement'], 'unknown')
        self.assertEqual(value['access_claims']['historical_session_effort'], 'unknown')
        self.assertEqual(value['provider_list']['models'][0]['provider_supported_efforts'], ['xhigh'])
        extension = value['provider_list']['models'][0]['extensions']['vendor_extension']
        self.assertEqual(extension['generation'], 7)
        self.assertIn('REDACTED', extension['note'])
        self.assertNotIn('extensionSecret987654', json.dumps(value))
        self.assertEqual(value['provider_list']['extensions']['cache_extension'], {'revision': 'r1'})
        self.assertEqual(value['training_claims']['task_outcome'], 'unknown')

    def test_snapshot_revalidates_sources_and_context_requires_exact_prompt(self):
        value = snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train',
                                        observed_at='2026-11-03T09:00:00Z')
        snapshot.revalidate_snapshot(value)
        policy = Path(__file__).with_name('private-corpus-screen.example.json')
        rendered, digest = snapshot.render_bound_context(value, 'Review the Terra project plan.',
                                                        max_context_bytes=100_000, policy_path=policy)
        self.assertEqual(snapshot.sha(rendered.encode('utf-8')), digest)
        self.assertIn('entitlement=unknown', rendered)
        self.assertIn('historical_effort=unknown', rendered)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            snapshot.render_bound_context(value, 'A different prompt.', max_context_bytes=100_000,
                                          policy_path=policy)
        self.cache.write_text(self.cache.read_text(encoding='utf-8').replace('listed', 'changed'), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'provider cache changed'):
            snapshot.revalidate_snapshot(value)

    def test_revalidation_rejects_mutated_provider_projection_and_access_claims(self):
        import copy
        value = snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train')
        # Safe provider extensions are retained, but all security-relevant rows
        # and claims must remain exactly bound to the freshly parsed cache.
        snapshot.revalidate_snapshot(value)
        mutations = (
            ('model_id', lambda row: row.__setitem__('model_id', 'invented-model')),
            ('description', lambda row: row.__setitem__('description', 'invented description')),
            ('priority', lambda row: row.__setitem__('priority', 0)),
            ('efforts', lambda row: row.__setitem__('provider_supported_efforts', ['unlisted-effort'])),
            ('access', lambda row: row.__setitem__('subscription_entitlement', 'available')),
        )
        for name, mutate in mutations:
            changed = copy.deepcopy(value)
            target = (changed['provider_list']['models'][0] if name != 'access'
                      else changed['access_claims'])
            mutate(target)
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, 'provider cache changed'):
                snapshot.revalidate_snapshot(changed)

    def test_single_request_selection_and_context_are_snapshot_bound(self):
        value = snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train',
                                        known_reserved_project_labels=['Makers-Impulse'])
        policy = Path(__file__).with_name('private-corpus-screen.example.json')
        manifest, row, context, request_raw, manifest_raw, context_raw, projection_raw, bundle_raw = snapshot.prepare_single_request(
            value, policy_path=policy, max_request_bytes=4096, max_context_bytes=100_000)
        self.assertEqual(manifest['requests'], 1)
        self.assertFalse(manifest['test_opened'])
        self.assertEqual(manifest['heldout_separation'], 'unknown')
        self.assertEqual(row['split'], 'train')
        self.assertEqual(row['review_status'], 'unreviewed')
        self.assertEqual(row['outcome'], 'unknown')
        self.assertEqual(row['source_bound_snapshot_sha256'], manifest['source_bound_snapshot_sha256'])
        self.assertEqual(row['source_bound_snapshot_sha256'], context['projects']['lfg']['source_bound_snapshot_sha256'])
        self.assertEqual(row['source_text_sha256_claim'], self.record['text_sha256'])
        self.assertEqual(snapshot.sha(request_raw), manifest['requests_sha256'])
        self.assertEqual(json.loads(context_raw)['projects']['lfg']['sha256'],
                         context['projects']['lfg']['sha256'])
        self.assertEqual(json.loads(manifest_raw)['selection_kind'], 'direct_source_bound_single_request')
        copied = snapshot.validate_staged_artifacts(projection_raw, request_raw, manifest_raw, context_raw, bundle_raw)
        self.assertEqual(copied['snapshot_sha256'], manifest['source_bound_snapshot_sha256'])
        self.assertEqual(copied['heldout_separation'], 'unknown')
        with self.assertRaises(ValueError):
            snapshot.validate_staged_artifacts(projection_raw, request_raw, manifest_raw,
                                               context_raw.replace(b'provider_cache_sha256', b'provider_cache_sha25', 1),
                                               bundle_raw)

    def test_changed_selected_line_fails_closed(self):
        self.session.write_bytes(self.session.read_bytes().replace(b'Review the Terra project plan.', b'Changed the Terra project plan.'))
        with self.assertRaisesRegex(ValueError, 'source identity'):
            snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train')

    def test_append_only_suffix_keeps_selected_prefix_verifiable(self):
        with self.session.open('ab') as stream:
            stream.write(b'{"type":"later_event"}\n')
        value = snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train')
        self.assertTrue(value['source']['verified'])
        self.assertEqual(value['source']['source_prefix_sha256'], self.record['source_prefix_sha256'])

    def test_known_reserved_project_is_rejected_before_opening_session(self):
        with mock.patch.object(snapshot, 'verify_selected_source', side_effect=AssertionError('opened source')):
            with self.assertRaisesRegex(ValueError, 'reserved group'):
                snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train',
                                        known_reserved_project_labels=['lfg'])

    def test_reserved_group_is_rejected_before_opening_session(self):
        group = snapshot.source_group_id(self.record)
        with mock.patch.object(snapshot.inspect_codex_inventory, 'verify', side_effect=AssertionError('opened source')):
            with self.assertRaisesRegex(ValueError, 'reserved group'):
                snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train',
                                        known_reserved_group_sha256=[group])

    def test_session_group_identity_ignores_mutable_project_labels_and_rejects_before_open(self):
        import copy
        changed = copy.deepcopy(self.record)
        changed['project'] = 'renamed-alias'
        self.assertEqual(snapshot.source_group_id(self.record), snapshot.source_group_id(changed))
        with mock.patch.object(snapshot, 'verify_selected_source', side_effect=AssertionError('opened source')):
            with self.assertRaisesRegex(ValueError, 'reserved group'):
                snapshot.build_snapshot(changed, self.cache, split_policy='exploratory_train',
                                        known_reserved_group_sha256=[snapshot.source_group_id(self.record)])

    def test_session_group_identity_normalizes_path_aliases(self):
        alias = str(self.root / 'project' / '..' / 'session.jsonl')
        changed = {**self.record, 'source_file': alias}
        self.assertEqual(snapshot.source_group_id(self.record), snapshot.source_group_id(changed))

    def test_reserved_project_aliases_are_case_and_punctuation_insensitive(self):
        with mock.patch.object(snapshot, 'verify_selected_source', side_effect=AssertionError('opened source')):
            with self.assertRaisesRegex(ValueError, 'reserved group'):
                snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train',
                                        known_reserved_project_labels=['L.F.G'])

    def test_reserved_train_requires_split_evidence(self):
        with self.assertRaisesRegex(ValueError, 'split evidence'):
            snapshot.build_snapshot(self.record, self.cache, split_policy='reserved_train')

    def test_provider_cache_rejects_duplicate_model_and_sensitive_extension(self):
        rows = [{'slug': 'same', 'visibility': 'list'}, {'slug': 'same', 'visibility': 'list'}]
        self.write_cache({'models': rows})
        with self.assertRaisesRegex(ValueError, 'duplicate provider model'):
            snapshot.read_provider_cache(self.cache)
        self.write_cache({'models': [{'slug': 'visible', 'visibility': 'list', 'auth_token': 'private'}]})
        with self.assertRaisesRegex(ValueError, 'sensitive provider extension'):
            snapshot.read_provider_cache(self.cache)

    def test_context_screens_provider_metadata_before_returning_it(self):
        self.write_cache({'models': [{'slug': 'terra-visible', 'display_name': 'Visible Terra',
                                      'description': 'example password=supersecretvalue',
                                      'visibility': 'list'}]})
        value = snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train')
        policy = Path(__file__).with_name('private-corpus-screen.example.json')
        rendered, _digest = snapshot.render_bound_context(value, 'Review the Terra project plan.',
                                                          max_context_bytes=100_000, policy_path=policy)
        self.assertNotIn('supersecretvalue', rendered)
        self.assertIn('REDACTED', rendered)

    def test_context_keeps_ambiguous_family_variants_as_candidates(self):
        self.write_cache({'models': [
            {'slug': 'gpt-6.1-terra', 'display_name': 'GPT-6.1 Terra', 'visibility': 'list'},
            {'slug': 'gpt-6.2-terra', 'display_name': 'GPT-6.2 Terra', 'visibility': 'list'},
            {'slug': 'gpt-6-codex', 'display_name': 'GPT-6 Codex', 'visibility': 'list'},
        ]})
        value = snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train')
        policy = Path(__file__).with_name('private-corpus-screen.example.json')
        rendered, _digest = snapshot.render_bound_context(value, 'Review the Terra project plan.',
                                                          max_context_bytes=100_000, policy_path=policy)
        parsed = [json.loads(line) for line in rendered.splitlines()[1:]]
        self.assertEqual({model[0] for model in parsed}, {'gpt-6.1-terra', 'gpt-6.2-terra'})
        self.assertIn('not recommendations', rendered)

    def test_nonfinite_extension_is_rejected(self):
        raw = b'{"models":[{"slug":"visible","visibility":"list","extension":NaN}]}'
        self.cache.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'invalid provider cache JSON'):
            snapshot.read_provider_cache(self.cache)

    def test_windows_private_write_fails_closed(self):
        with mock.patch.object(snapshot.os, 'name', 'nt'), \
                mock.patch.object(snapshot, '_secure_windows_file', side_effect=ValueError('ACL unavailable')):
            with self.assertRaisesRegex(ValueError, 'ACL unavailable'):
                snapshot.write_private_snapshot(self.root / 'snapshot.json', {'ok': True})
        self.assertFalse((self.root / 'snapshot.json').exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows ACL verification is Windows-specific')
    def test_windows_private_write_fails_closed_without_verified_parent_chain(self):
        destination = self.root / 'snapshot.json'
        with mock.patch.object(snapshot, '_secure_windows_file', side_effect=ValueError('parent chain unverified')):
            with self.assertRaisesRegex(ValueError, 'parent chain unverified'):
                snapshot.write_private_snapshot(destination, {'schema_version': snapshot.SCHEMA})
        self.assertFalse(destination.exists())

    def test_windows_writer_rejects_replaced_path_identity(self):
        destination = self.root / 'replacement.json'
        destination.write_bytes(b'')
        descriptor = os.open(destination, os.O_WRONLY)
        self.addCleanup(lambda: os.close(descriptor) if descriptor >= 0 else None)
        original_stat = destination.stat()
        replaced = mock.Mock(st_dev=original_stat.st_dev, st_ino=original_stat.st_ino + 1)
        with mock.patch.object(Path, 'stat', return_value=replaced):
            with self.assertRaisesRegex(ValueError, 'file identity changed'):
                snapshot._write_verified_windows_handle(descriptor, destination, b'private')


if __name__ == '__main__':
    unittest.main()
