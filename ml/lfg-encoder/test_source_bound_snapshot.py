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

    def test_cli_reserved_session_manifest_rejects_renamed_label_before_source_open(self):
        inventory_path = self.root / 'inventory.jsonl'
        inventory_path.write_bytes(snapshot.json_bytes(self.record))
        manifest_path = self.root / 'reserved-groups.json'
        manifest_value = {'schema_version': 'lfg.reserved-session-groups.v1',
                          'coverage': 'partial',
                          'groups': [snapshot.source_group_id(self.record)],
                          'extension': {'kept_in_digest': True}}
        manifest_path.write_bytes(snapshot.json_bytes(manifest_value))
        groups, digest, coverage = snapshot.read_reserved_session_manifest(manifest_path)
        self.assertEqual(groups, manifest_value['groups'])
        self.assertEqual(digest, snapshot.sha(manifest_path.read_bytes()))
        self.assertEqual(coverage, 'partial')
        changed = dict(self.record, project='renamed-project-label')
        inventory_path.write_bytes(snapshot.json_bytes(changed))
        output_path = self.root / 'snapshot.json'
        argv = ['source_bound_snapshot.py', '--inventory', str(inventory_path), '--record', '1',
                '--provider-cache', str(self.cache), '--split-policy', 'exploratory_train',
                '--reserved-session-ids', str(manifest_path), '--output', str(output_path)]
        with mock.patch('sys.argv', argv), \
                mock.patch.object(snapshot, 'verify_selected_source', side_effect=AssertionError('opened source')) as opened:
            with self.assertRaises(SystemExit):
                snapshot.main()
        opened.assert_not_called()
        self.assertFalse(output_path.exists())

    def test_reserved_session_manifest_cannot_claim_complete_coverage(self):
        path = self.root / 'complete-reserved.json'
        path.write_bytes(snapshot.json_bytes({'schema_version': 'lfg.reserved-session-groups.v1',
                                              'coverage': 'complete', 'groups': []}))
        with self.assertRaisesRegex(ValueError, 'invalid reserved-session manifest'):
            snapshot.read_reserved_session_manifest(path)

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

    def test_provider_metadata_is_screened_before_persistence_and_identity_changes_rejected(self):
        policy = self.root / 'screen-policy.json'
        policy.write_text(json.dumps({
            'schema_version': 'lfg.private-corpus-screen.v1', 'revision': 'metadata-test',
            'limits': {'max_policy_bytes': 65536, 'max_plan_bytes': 65536,
                       'max_source_bytes': 100000, 'max_total_source_bytes': 100000,
                       'max_total_output_bytes': 100000, 'max_records': 100,
                       'max_text_bytes': 10000, 'max_matches_per_record': 100},
            'rules': [{'id': 'private-value', 'pattern': 'owner-private-[A-Za-z0-9]+',
                       'replacement': '[REDACTED]'}]}))
        self.write_cache({'models': [{'slug': 'safe-model', 'display_name': 'owner-private-label9',
                                      'description': 'owner-private-description8', 'visibility': 'list',
                                      'supported_reasoning_levels': [{'effort': 'low'}]}]})
        parsed, _raw, _stat = snapshot.read_provider_cache(self.cache, screen_policy_path=policy)
        self.assertEqual(parsed['models'][0]['display_name'], '[REDACTED]')
        self.assertEqual(parsed['models'][0]['description'], '[REDACTED]')
        self.assertNotIn('owner-private', json.dumps(parsed))
        self.write_cache({'models': [{'slug': 'owner-private-id6', 'visibility': 'list'}]})
        with self.assertRaisesRegex(ValueError, 'model ID requires redaction'):
            snapshot.read_provider_cache(self.cache, screen_policy_path=policy)

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
                mock.patch.object(snapshot, '_secure_windows_directory', side_effect=ValueError('ACL unavailable')), \
                mock.patch.object(snapshot, '_secure_windows_file', side_effect=ValueError('ACL unavailable')):
            with self.assertRaisesRegex(ValueError, 'ACL unavailable'):
                snapshot.write_private_snapshot(self.root / 'snapshot.json', {'ok': True})
        self.assertFalse((self.root / 'snapshot.json').exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows ACL verification is Windows-specific')
    def test_windows_private_write_succeeds_under_normal_readable_ancestor_chain(self):
        with tempfile.TemporaryDirectory(dir=os.environ['PROGRAMDATA']) as private_directory:
            destination = Path(private_directory) / 'snapshot.json'
            snapshot.write_private_snapshot(destination, {'schema_version': snapshot.SCHEMA})
            self.assertEqual(json.loads(destination.read_text(encoding='utf-8'))['schema_version'], snapshot.SCHEMA)

    def test_windows_writer_rejects_replaced_path_identity(self):
        destination = self.root / 'replacement.json'
        destination.write_bytes(b'')
        descriptor = os.open(destination, os.O_WRONLY)
        original_stat = destination.stat()
        replaced = mock.Mock(st_dev=original_stat.st_dev, st_ino=original_stat.st_ino + 1)
        real_stat = Path.stat
        with mock.patch.object(Path, 'stat', autospec=True,
                               side_effect=lambda path, *args, **kwargs:
                                   replaced if path == destination else real_stat(path, *args, **kwargs)):
            with self.assertRaisesRegex(ValueError, 'file identity changed'):
                snapshot._write_verified_windows_handle(descriptor, destination, b'private')
        with self.assertRaises(OSError):
            os.fstat(descriptor)

    def test_windows_writers_remove_empty_file_and_close_handle_on_identity_mismatch(self):
        for name, writer, value in (
            ('snapshot', snapshot.write_private_snapshot, {'schema_version': snapshot.SCHEMA}),
            ('bytes', snapshot.write_private_bytes, b'private'),
        ):
            with self.subTest(writer=name):
                destination = self.root / f'{name}.json'
                mismatch = mock.Mock(st_dev=os.stat(destination.parent).st_dev, st_ino=-1)
                real_stat = Path.stat
                real_open = os.open
                opened_descriptors = []

                def tracked_open(*args, **kwargs):
                    descriptor = real_open(*args, **kwargs)
                    opened_descriptors.append(descriptor)
                    return descriptor

                with mock.patch.object(snapshot.os, 'name', 'nt'), \
                        mock.patch.object(snapshot.os, 'open', side_effect=tracked_open), \
                        mock.patch.object(snapshot, '_require_ignored_untracked'), \
                        mock.patch.object(snapshot, '_secure_windows_directory'), \
                        mock.patch.object(snapshot, '_secure_windows_file'), \
                        mock.patch.object(Path, 'stat', autospec=True,
                                          side_effect=lambda path, *args, **kwargs:
                                              mismatch if path == destination else real_stat(path, *args, **kwargs)):
                    with self.assertRaisesRegex(ValueError, 'file identity changed'):
                        writer(destination, value)
                self.assertFalse(destination.exists())
                self.assertEqual(len(opened_descriptors), 1)
                with self.assertRaises(OSError):
                    os.fstat(opened_descriptors[0])


if __name__ == '__main__':
    unittest.main()
