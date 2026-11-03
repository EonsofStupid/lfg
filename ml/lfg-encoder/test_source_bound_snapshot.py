import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import inventory_codex_sessions as inventory
import source_bound_snapshot as snapshot


@unittest.skipUnless(os.name == 'posix', 'source-bound traversal is fail-closed on Windows')
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
            ('outcome', lambda row: row.__setitem__('task_outcome', 'confirmed')),
            ('teacher_proposal', lambda row: row.__setitem__('teacher_proposal', 'accepted')),
            ('review', lambda row: row.__setitem__('review_status', 'reviewed')),
            ('promotion', lambda row: row.__setitem__('promotion_eligible', True)),
        )
        for name, mutate in mutations:
            changed = copy.deepcopy(value)
            if name == 'access':
                target = changed['access_claims']
            elif name in {'outcome', 'teacher_proposal', 'review', 'promotion'}:
                target = changed['training_claims']
            else:
                target = changed['provider_list']['models'][0]
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
        for field, value in (('task_outcome', 'confirmed'), ('teacher_proposal', 'accepted'),
                             ('review_status', 'reviewed'), ('promotion_eligible', True)):
            changed_projection = json.loads(projection_raw)
            changed_projection['training_claims'][field] = value
            with self.subTest(staged_claim=field), self.assertRaisesRegex(
                    ValueError, 'training claims must remain unknown and unreviewed'):
                snapshot.validate_staged_artifacts(snapshot.json_bytes(changed_projection),
                                                   request_raw, manifest_raw, context_raw)
        with self.assertRaises(ValueError):
            snapshot.validate_staged_artifacts(projection_raw, request_raw, manifest_raw,
                                               context_raw.replace(b'provider_cache_sha256', b'provider_cache_sha25', 1),
                                               bundle_raw)

        for field in ('human_reviewed', 'promotion_eligible'):
            changed_manifest = json.loads(manifest_raw)
            changed_manifest[field] = True
            changed_manifest_raw = snapshot.json_bytes(changed_manifest)
            changed_bundle = json.loads(bundle_raw)
            changed_bundle['selection_manifest_sha256'] = snapshot.sha(changed_manifest_raw)
            changed_bundle_raw = snapshot.json_bytes(changed_bundle)
            with self.subTest(selection_claim=field), self.assertRaisesRegex(
                    ValueError, 'staged selection claims or digests'):
                snapshot.validate_staged_artifacts(projection_raw, request_raw,
                                                   changed_manifest_raw, context_raw,
                                                   changed_bundle_raw)

    def test_staged_access_and_provenance_claims_reject_rehashed_tampering(self):
        value = snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train')
        policy = Path(__file__).with_name('private-corpus-screen.example.json')
        prepared = snapshot.prepare_single_request(
            value, policy_path=policy, max_request_bytes=4096, max_context_bytes=100_000)
        _, _, _, request_raw, manifest_raw, context_raw, projection_raw, bundle_raw = prepared

        def reseal_projection(mutate):
            projection = json.loads(projection_raw)
            mutate(projection)
            changed_raw = snapshot.json_bytes(projection)
            bundle = json.loads(bundle_raw)
            bundle['staged_snapshot_sha256'] = snapshot.sha(changed_raw)
            return changed_raw, snapshot.json_bytes(bundle)

        access_mutations = {
            'provider_listing_is_entitlement': True,
            'subscription_entitlement': 'available',
            'historical_session_effort': 'xhigh',
            'historical_model_access': 'available',
        }
        for key, changed_value in access_mutations.items():
            changed_raw, changed_bundle = reseal_projection(
                lambda projection, k=key, v=changed_value:
                    projection['access_claims'].__setitem__(k, v))
            with self.subTest(access_claim=key), self.assertRaisesRegex(
                    ValueError, 'access claims must remain unknown'):
                snapshot.validate_staged_artifacts(changed_raw, request_raw, manifest_raw,
                                                   context_raw, changed_bundle)

        # Unknown extension fields survive as inert claims; their presence does
        # not authorize or alter entitlement/access behavior.
        changed_raw, changed_bundle = reseal_projection(
            lambda projection: projection['access_claims'].__setitem__('owner_note_extension', 'kept'))
        self.assertEqual(
            snapshot.validate_staged_artifacts(changed_raw, request_raw, manifest_raw,
                                               context_raw, changed_bundle)['requests'], 1)

        provenance_mutations = {
            'kind': 'unverified-projection',
            'source_and_cache_verified_on_origin_host': False,
            'full_source_paths_transferred': True,
            'provider_extension_payloads_transferred': True,
        }
        for key, changed_value in provenance_mutations.items():
            changed_raw, changed_bundle = reseal_projection(
                lambda projection, k=key, v=changed_value:
                    projection['materialization'].__setitem__(k, v))
            with self.subTest(materialization_claim=key), self.assertRaisesRegex(
                    ValueError, 'projection provenance is invalid'):
                snapshot.validate_staged_artifacts(changed_raw, request_raw, manifest_raw,
                                                   context_raw, changed_bundle)

        bundle_mutations = {
            'projection_kind': 'unverified-projection',
            'full_source_paths_transferred': True,
            'provider_extension_payloads_transferred': True,
        }
        for key, changed_value in bundle_mutations.items():
            changed_bundle = json.loads(bundle_raw)
            changed_bundle[key] = changed_value
            with self.subTest(bundle_claim=key), self.assertRaisesRegex(
                    ValueError, 'staged bundle digest mismatch'):
                snapshot.validate_staged_artifacts(projection_raw, request_raw, manifest_raw,
                                                   context_raw, snapshot.json_bytes(changed_bundle))

        selected_group = json.loads(projection_raw)['source']['source_group_sha256']
        reserved_mutations = {
            'known_reserved_group_sha256': [selected_group],
            'known_reserved_source_identities': [snapshot._source_identity_claim(self.record)],
            'known_reserved_groups_checked': 1,
            'known_reserved_groups_coverage': 'complete',
            'known_reserved_manifest_sha256': 'bad-digest',
            'known_reserved_project_labels': ['L.F.G'],
        }
        for key, changed_value in reserved_mutations.items():
            changed_raw, changed_bundle = reseal_projection(
                lambda projection, k=key, v=changed_value:
                    projection['split'].__setitem__(k, v))
            with self.subTest(reserved_claim=key), self.assertRaisesRegex(
                    ValueError, 'reserved-group claims'):
                snapshot.validate_staged_artifacts(changed_raw, request_raw, manifest_raw,
                                                   context_raw, changed_bundle)

    def test_changed_selected_line_fails_closed(self):
        self.session.write_bytes(self.session.read_bytes().replace(b'Review the Terra project plan.', b'Changed the Terra project plan.'))
        with self.assertRaisesRegex(ValueError, 'source identity'):
            snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train')

    def test_selected_source_rejects_hardlinks_before_reading_content(self):
        alias = self.root / 'session-hardlink.jsonl'
        try:
            os.link(self.session, alias)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f'hardlinks unavailable: {error}')
        changed_record = dict(self.record, source_file=str(alias))
        with mock.patch('source_bound_snapshot.os.fdopen', side_effect=AssertionError('read source bytes')) as opened:
            with self.assertRaisesRegex(ValueError, 'single-link regular file'):
                snapshot.verify_selected_source(
                    changed_record, max_session_bytes=4096, max_line_bytes=4096)
        opened.assert_not_called()

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

    @unittest.skipUnless(os.name == 'posix', 'source identity checks require POSIX no-follow reads')
    def test_reserved_identity_rejects_copy_and_relocated_hardlink_before_read(self):
        identity = snapshot._source_identity_claim(self.record)
        manifest = self.root / 'reserved-identities.json'
        manifest.write_bytes(snapshot.json_bytes({'schema_version': 'lfg.reserved-session-groups.v1',
                                                  'coverage': 'partial', 'groups': [],
                                                  'identities': [identity]}))
        manifest.chmod(0o600)
        groups, digest, coverage, identities = snapshot.read_reserved_session_manifest(manifest)
        copy_path = self.root / 'renamed-copy.jsonl'
        copy_path.write_bytes(self.session.read_bytes())
        copied = dict(self.record, source_file=str(copy_path))
        appended_path = self.root / 'appended-copy.jsonl'
        appended_path.write_bytes(self.session.read_bytes() + b'{"type":"later_event"}\n')
        status, appended_records = inventory.scan_file(
            appended_path, {'lfg': inventory.normalize(str(self.root / 'project'))}, 4096, 4096)
        self.assertEqual(status, 'scanned')
        appended = appended_records[0]
        self.assertNotEqual(appended['source_file_sha256'], identity['source_file_sha256'])
        self.assertEqual(appended['text_sha256'], identity['text_sha256'])
        hardlink = self.root / 'moved-hardlink.jsonl'
        os.link(self.session, hardlink)
        self.session.unlink()
        moved = dict(self.record, source_file=str(hardlink), source_file_sha256='b' * 64)
        self.assertEqual(hardlink.stat().st_nlink, 1)
        with mock.patch.object(snapshot, 'verify_selected_source', side_effect=AssertionError('opened source')) as opened:
            for candidate in (copied, appended, moved):
                with self.subTest(path=candidate['source_file']), self.assertRaisesRegex(ValueError, 'reserved group'):
                    snapshot.build_snapshot(candidate, self.cache, split_policy='exploratory_train',
                                            known_reserved_group_sha256=groups,
                                            known_reserved_source_identities=identities,
                                            known_reserved_manifest_sha256=digest,
                                            known_reserved_manifest_locator=str(manifest.resolve()),
                                            known_reserved_groups_coverage=coverage)
        opened.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'source identity checks require POSIX no-follow reads')
    def test_reserved_source_identity_claim_requires_pinned_manifest(self):
        with self.assertRaisesRegex(ValueError, 'require a pinned manifest'):
            snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train',
                                    known_reserved_source_identities=[
                                        snapshot._source_identity_claim(self.record)])

    def test_reserved_project_aliases_are_case_and_punctuation_insensitive(self):
        with mock.patch.object(snapshot, 'verify_selected_source', side_effect=AssertionError('opened source')):
            with self.assertRaisesRegex(ValueError, 'reserved group'):
                snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train',
                                        known_reserved_project_labels=['L.F.G'])

    @unittest.skipUnless(os.name == 'posix', 'reserved source identity checks are POSIX-specific')
    def test_manifest_identity_rejects_candidate_before_source_open(self):
        manifest = self.root / 'reserved-identity.json'
        identity = snapshot._source_identity_claim(self.record)
        manifest.write_bytes(snapshot.json_bytes({'schema_version': 'lfg.reserved-session-groups.v1',
                                                  'coverage': 'partial', 'groups': [],
                                                  'identities': [identity]}))
        manifest.chmod(0o600)
        groups, digest, coverage, identities = snapshot.read_reserved_session_manifest(manifest)
        with mock.patch.object(snapshot, 'verify_selected_source', side_effect=AssertionError('opened source')) as opened:
            with self.assertRaisesRegex(ValueError, 'reserved group'):
                snapshot.build_snapshot(self.record, self.cache, split_policy='exploratory_train',
                                        known_reserved_group_sha256=groups,
                                        known_reserved_source_identities=identities,
                                        known_reserved_manifest_sha256=digest,
                                        known_reserved_manifest_locator=str(manifest.resolve()),
                                        known_reserved_groups_coverage=coverage)
        opened.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'reserved identity metadata reads require POSIX')
    def test_reserved_identity_extensions_are_screened_preserved_and_inert(self):
        known = snapshot._source_identity_claim(self.record)
        known.update({key: 'a' * 64 for key in (
            'source_file_sha256', 'source_prefix_sha256', 'source_line_sha256', 'text_sha256')})
        known['source_line'] += 10
        known['note'] = 'password=SecretValue12345'
        known['operator_tag'] = 'review-queue-4'
        path = self.root / 'reserved-identity-extension.json'
        path.write_bytes(snapshot.json_bytes({'schema_version': 'lfg.reserved-session-groups.v1',
                                              'coverage': 'partial', 'groups': [],
                                              'identities': [known]}))
        path.chmod(0o600)
        groups, digest, coverage, identities = snapshot.read_reserved_session_manifest(path)
        self.assertIn('REDACTED', identities[0]['note'])
        self.assertEqual(identities[0]['operator_tag'], 'review-queue-4')
        self.assertNotIn('SecretValue12345', json.dumps(identities))
        value = snapshot.build_snapshot(
            self.record, self.cache, split_policy='exploratory_train',
            known_reserved_group_sha256=groups, known_reserved_source_identities=identities,
            known_reserved_manifest_sha256=digest, known_reserved_manifest_locator=str(path.resolve()),
            known_reserved_groups_coverage=coverage)
        self.assertEqual(value['split']['known_reserved_source_identities'][0]['note'], identities[0]['note'])
        self.assertNotIn('SecretValue12345', json.dumps(value['split']))

    @unittest.skipUnless(os.name == 'posix', 'reserved identity metadata reads require POSIX')
    def test_duplicate_known_identity_with_different_extensions_is_rejected(self):
        identity = snapshot._source_identity_claim(self.record)
        path = self.root / 'reserved-duplicate-identities.json'
        path.write_bytes(snapshot.json_bytes({'schema_version': 'lfg.reserved-session-groups.v1',
                                              'coverage': 'partial', 'groups': [],
                                              'identities': [{**identity, 'note': 'one'},
                                                             {**identity, 'note': 'two'}]}))
        path.chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'duplicate reserved source identity'):
            snapshot.read_reserved_session_manifest(path)

    @unittest.skipUnless(os.name == 'posix', 'reserved identity screening requires POSIX manifest access')
    def test_reserved_identity_extensions_use_declared_policy_through_staging(self):
        policy_path = self.root / 'reserved-screen-policy.json'
        policy = json.loads(Path(__file__).with_name('private-corpus-screen.example.json').read_text())
        policy['revision'] = 'reserved-identity-custom-test'
        policy['rules'].append({'id': 'custom-private-marker',
                                'pattern': r'custom-secret-[A-Za-z0-9]+',
                                'replacement': '[CUSTOM REDACTED]'})
        policy_raw = snapshot.json_bytes(policy)
        policy_path.write_bytes(policy_raw)
        identity = snapshot._source_identity_claim(self.record)
        for key in ('source_file_sha256', 'source_prefix_sha256', 'source_line_sha256', 'text_sha256'):
            identity[key] = 'a' * 64
        identity['source_line'] += 10
        identity['note'] = 'custom-secret-Abc123456'
        manifest = self.root / 'reserved-policy-identities.json'
        manifest.write_bytes(snapshot.json_bytes({'schema_version': 'lfg.reserved-session-groups.v1',
                                                   'coverage': 'partial', 'groups': [],
                                                   'identities': [identity]}))
        manifest.chmod(0o600)
        groups, digest, coverage, identities = snapshot.read_reserved_session_manifest(
            manifest, screen_policy_path=policy_path)
        self.assertEqual(identities[0]['note'], '[CUSTOM REDACTED]')
        self.assertNotIn('custom-secret-Abc123456', json.dumps(identities))
        value = snapshot.build_snapshot(
            self.record, self.cache, split_policy='exploratory_train',
            known_reserved_group_sha256=groups, known_reserved_source_identities=identities,
            known_reserved_manifest_sha256=digest, known_reserved_manifest_locator=str(manifest.resolve()),
            known_reserved_groups_coverage=coverage, screen_policy_path=policy_path)
        prepared = snapshot.prepare_single_request(value, policy_path=policy_path,
                                                  max_request_bytes=4096, max_context_bytes=100_000)
        projection_raw, bundle_raw = prepared[6], prepared[7]
        copied = snapshot.validate_staged_artifacts(projection_raw, prepared[3], prepared[4], prepared[5],
                                                    bundle_raw, policy_raw)
        self.assertEqual(copied['requests'], 1)
        changed = json.loads(projection_raw)
        changed['split']['known_reserved_source_identities'][0]['note'] = 'custom-secret-Abc123456'
        changed_raw = snapshot.json_bytes(changed)
        changed_bundle = json.loads(bundle_raw)
        changed_bundle['staged_snapshot_sha256'] = snapshot.sha(changed_raw)
        with self.assertRaisesRegex(ValueError, 'require screening'):
            snapshot.validate_staged_artifacts(changed_raw, prepared[3], prepared[4], prepared[5],
                                               snapshot.json_bytes(changed_bundle), policy_raw)
        with self.assertRaisesRegex(ValueError, 'declared policy bytes'):
            snapshot.validate_staged_artifacts(projection_raw, prepared[3], prepared[4], prepared[5],
                                               bundle_raw)

    @unittest.skipUnless(os.name == 'posix', 'reserved manifest reads require POSIX no-follow handles')
    def test_cli_reserved_session_manifest_rejects_renamed_label_before_source_open(self):
        inventory_path = self.root / 'inventory.jsonl'
        inventory_path.write_bytes(snapshot.json_bytes(self.record))
        manifest_path = self.root / 'reserved-groups.json'
        manifest_value = {'schema_version': 'lfg.reserved-session-groups.v1',
                          'coverage': 'partial',
                          'groups': [snapshot.source_group_id(self.record)],
                          'identities': [],
                          'extension': {'kept_in_digest': True}}
        manifest_path.write_bytes(snapshot.json_bytes(manifest_value))
        if os.name == 'posix':
            manifest_path.chmod(0o600)
        groups, digest, coverage, identities = snapshot.read_reserved_session_manifest(manifest_path)
        self.assertEqual(groups, manifest_value['groups'])
        self.assertEqual(digest, snapshot.sha(manifest_path.read_bytes()))
        self.assertEqual(coverage, 'partial')
        self.assertEqual(identities, [])
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
        if os.name != 'posix':
            self.skipTest('reserved manifest reads require POSIX no-follow handles')
        path = self.root / 'complete-reserved.json'
        path.write_bytes(snapshot.json_bytes({'schema_version': 'lfg.reserved-session-groups.v1',
                                              'coverage': 'complete', 'groups': [], 'identities': []}))
        path.chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'invalid reserved-session manifest'):
            snapshot.read_reserved_session_manifest(path)

    @unittest.skipUnless(os.name == 'posix', 'POSIX private-file checks are platform-specific')
    def test_reserved_manifest_rejects_symlink_hardlink_and_permissive_file(self):
        raw = snapshot.json_bytes({'schema_version': 'lfg.reserved-session-groups.v1',
                                   'coverage': 'partial', 'groups': [], 'identities': []})
        original = self.root / 'reserved-private.json'
        original.write_bytes(raw)
        original.chmod(0o600)
        self.assertEqual(snapshot.read_reserved_session_manifest(original)[0], [])

        permissive = self.root / 'reserved-permissive.json'
        permissive.write_bytes(raw)
        permissive.chmod(0o644)
        with self.assertRaisesRegex(ValueError, 'owner-only'):
            snapshot.read_reserved_session_manifest(permissive)

        hardlink = self.root / 'reserved-hardlink.json'
        os.link(original, hardlink)
        with self.assertRaisesRegex(ValueError, 'owner-only'):
            snapshot.read_reserved_session_manifest(original)

        symlink = self.root / 'reserved-symlink.json'
        symlink.symlink_to(original)
        with self.assertRaises(ValueError):
            snapshot.read_reserved_session_manifest(symlink)

    @unittest.skipUnless(os.name == 'nt', 'Windows library boundary is platform-specific')
    def test_reserved_manifest_library_read_fails_before_path_access_on_windows(self):
        with mock.patch.object(Path, 'stat', side_effect=AssertionError('path stat')), \
                mock.patch.object(Path, 'open', side_effect=AssertionError('path open')), \
                mock.patch.object(Path, 'resolve', side_effect=AssertionError('path resolve')):
            with self.assertRaisesRegex(ValueError, 'require POSIX no-follow'):
                snapshot.read_reserved_session_manifest(self.root / 'reserved.json')

    @unittest.skipUnless(os.name == 'nt', 'Windows direct-library boundary is platform-specific')
    def test_direct_builder_and_verifier_fail_before_retargeted_junction_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / 'first', root / 'second'
            first.mkdir(); second.mkdir()
            source_a, source_b = first / 'session.jsonl', second / 'session.jsonl'
            source_a.write_bytes(b'fixture source A')
            source_b.write_bytes(b'fixture source B')
            junction = root / 'sessions'
            made = subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(junction), str(first)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if made.returncode != 0:
                self.skipTest('junction creation unavailable')
            junction.rmdir()
            subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(junction), str(second)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            record = {'source_file': str(junction / 'session.jsonl')}
            with mock.patch.object(Path, 'stat', side_effect=AssertionError('junction traversed')), \
                 mock.patch.object(Path, 'lstat', side_effect=AssertionError('junction traversed')), \
                 mock.patch.object(Path, 'open', side_effect=AssertionError('junction traversed')):
                with self.assertRaisesRegex(ValueError, 'requires POSIX no-follow'):
                    snapshot.verify_selected_source(record, max_session_bytes=4096, max_line_bytes=4096)
                with self.assertRaisesRegex(ValueError, 'require POSIX no-follow'):
                    snapshot.build_snapshot(record, root / 'cache.json', split_policy='exploratory_train')
            self.assertEqual(source_a.read_bytes(), b'fixture source A')
            self.assertEqual(source_b.read_bytes(), b'fixture source B')
            junction.rmdir()

    @unittest.skipUnless(os.name == 'posix', 'reserved-manifest no-follow checks are POSIX-specific')
    def test_materialization_rechecks_reserved_manifest_after_staging(self):
        manifest = self.root / 'reserved-final-check.json'
        initial = {'schema_version': 'lfg.reserved-session-groups.v1',
                   'coverage': 'partial', 'groups': [], 'identities': []}
        manifest.write_bytes(snapshot.json_bytes(initial))
        manifest.chmod(0o600)
        groups, digest, coverage, identities = snapshot.read_reserved_session_manifest(manifest)
        value = snapshot.build_snapshot(
            self.record, self.cache, split_policy='exploratory_train',
            known_reserved_group_sha256=groups, known_reserved_manifest_sha256=digest,
            known_reserved_source_identities=identities,
            known_reserved_manifest_locator=str(manifest.resolve()),
            known_reserved_groups_coverage=coverage)
        original_stage = snapshot.stage_snapshot_projection

        def mutate_after_staging(snapshot_value, context_text):
            result = original_stage(snapshot_value, context_text)
            manifest.write_bytes(snapshot.json_bytes({
                'schema_version': 'lfg.reserved-session-groups.v1',
                'coverage': 'partial', 'groups': [snapshot.source_group_id(self.record)], 'identities': []}))
            manifest.chmod(0o600)
            return result

        with mock.patch.object(snapshot, 'stage_snapshot_projection', side_effect=mutate_after_staging):
            with self.assertRaisesRegex(ValueError, 'reserved-session manifest changed'):
                snapshot.prepare_single_request(
                    value, policy_path=Path(__file__).with_name('private-corpus-screen.example.json'),
                    max_request_bytes=4096, max_context_bytes=100_000)

    def test_snapshot_revalidation_detects_newly_reserved_session_before_source_open(self):
        if os.name != 'posix':
            self.skipTest('reserved manifest reads require POSIX no-follow handles')
        manifest = self.root / 'mutable-reserved.json'
        manifest.write_bytes(snapshot.json_bytes({'schema_version': 'lfg.reserved-session-groups.v1',
                                                  'coverage': 'partial', 'groups': [], 'identities': []}))
        manifest.chmod(0o600)
        groups, digest, coverage, identities = snapshot.read_reserved_session_manifest(manifest)
        value = snapshot.build_snapshot(
            self.record, self.cache, split_policy='exploratory_train',
            known_reserved_group_sha256=groups, known_reserved_manifest_sha256=digest,
            known_reserved_source_identities=identities,
            known_reserved_manifest_locator=str(manifest.resolve()),
            known_reserved_groups_coverage=coverage)
        prepared = snapshot.prepare_single_request(
            value, policy_path=Path(__file__).with_name('private-corpus-screen.example.json'),
            max_request_bytes=4096, max_context_bytes=100_000)
        projection = json.loads(prepared[6])
        self.assertNotIn('reserved_session_manifest_locator', projection)
        self.assertEqual(projection['split']['known_reserved_manifest_sha256'], digest)
        manifest.write_bytes(snapshot.json_bytes({'schema_version': 'lfg.reserved-session-groups.v1',
                                                  'coverage': 'partial',
                                                  'groups': [snapshot.source_group_id(self.record)], 'identities': []}))
        with mock.patch.object(snapshot, 'verify_selected_source', side_effect=AssertionError('opened source')) as opened:
            with self.assertRaisesRegex(ValueError, 'reserved-session manifest changed'):
                snapshot.revalidate_snapshot(value)
        opened.assert_not_called()

    def test_reserved_train_requires_split_evidence(self):
        with self.assertRaisesRegex(ValueError, 'split evidence'):
            snapshot.build_snapshot(self.record, self.cache, split_policy='reserved_train')

    def test_provider_cache_rejects_duplicate_model_and_sensitive_extension(self):
        rows = [{'slug': 'same', 'visibility': 'list'}, {'slug': 'same', 'visibility': 'list'}]
        self.write_cache({'models': rows})
        with self.assertRaisesRegex(ValueError, 'duplicate provider model'):
            snapshot.read_provider_cache(self.cache)
        for key in ('auth_token', 'session_token', 'id_token', 'jwt'):
            with self.subTest(sensitive_key=key):
                self.write_cache({'models': [{'slug': 'visible', 'visibility': 'list', key: 'private'}]})
                with self.assertRaisesRegex(ValueError, 'sensitive provider extension'):
                    snapshot.read_provider_cache(self.cache)

    def test_provider_extension_values_use_baseline_and_configured_screening(self):
        policy = json.loads(Path(snapshot.__file__).with_name('private-corpus-screen.example.json').read_bytes())
        policy['revision'] = 'fixture-custom'
        policy['rules'] = [{'id': 'fixture', 'pattern': 'CUSTOM_SECRET_[A-Z]+', 'replacement': '[REDACTED]'}]
        policy_path = self.root / 'custom-policy.json'
        policy_path.write_bytes(snapshot.json_bytes(policy))
        self.write_cache({'models': [{'slug': 'visible', 'visibility': 'list',
                                     'note': 'Basic YWxhZGRpbjpvcGVuc2VzYW1l CUSTOM_SECRET_VALUE'}]})
        metadata, _, _ = snapshot.read_provider_cache(self.cache, screen_policy_path=policy_path)
        note = metadata['models'][0]['extensions']['note']
        self.assertNotIn('YWxhZGRpbjpvcGVuc2VzYW1l', note)
        self.assertNotIn('CUSTOM_SECRET_VALUE', note)
        self.assertIn('[REDACTED]', note)

    def test_provider_known_fields_use_both_policies_before_persistence(self):
        policy = json.loads(Path(snapshot.__file__).with_name('private-corpus-screen.example.json').read_bytes())
        policy['revision'] = 'fixture-custom'
        policy['rules'] = [{'id': 'fixture', 'pattern': 'CUSTOM_SECRET_[A-Z]+', 'replacement': '[REDACTED]'}]
        policy_path = self.root / 'custom-policy.json'
        policy_path.write_bytes(snapshot.json_bytes(policy))
        self.write_cache({'fetched_at': 'Basic YWxhZGRpbjpvcGVuc2VzYW1l', 'models': [
            {'slug': 'visible', 'visibility': 'list', 'display_name': 'Cookie: session=fixture-cookie-value',
             'description': 'Basic YWxhZGRpbjpvcGVuc2VzYW1l CUSTOM_SECRET_VALUE'}]})
        metadata, _, _ = snapshot.read_provider_cache(self.cache, screen_policy_path=policy_path)
        encoded = json.dumps(metadata)
        for secret in ('YWxhZGRpbjpvcGVuc2VzYW1l', 'fixture-cookie-value', 'CUSTOM_SECRET_VALUE'):
            self.assertNotIn(secret, encoded)
        for key, item in [('slug', 'Basic YWxhZGRpbjpvcGVuc2VzYW1l'),
                          ('supported_reasoning_levels', [{'effort': 'CUSTOM_SECRET_VALUE'}])]:
            with self.subTest(identity=key):
                model = {'slug': 'visible', 'visibility': 'list', key: item}
                self.write_cache({'models': [model]})
                with self.assertRaisesRegex(ValueError, 'requires redaction'):
                    snapshot.read_provider_cache(self.cache, screen_policy_path=policy_path)

    def test_selected_request_is_screened_by_both_policies_before_staging(self):
        raw = self.session.read_bytes().replace(
            b'Review the Terra project plan.',
            b'Review the Terra project plan. Basic YWxhZGRpbjpvcGVuc2VzYW1l CUSTOM_SECRET_VALUE')
        self.session.write_bytes(raw)
        status, records = inventory.scan_file(
            self.session, {'lfg': inventory.normalize(str(self.root / 'project'))}, 4096, 4096)
        self.assertEqual(status, 'scanned')
        policy = json.loads(Path(snapshot.__file__).with_name('private-corpus-screen.example.json').read_bytes())
        policy['revision'] = 'fixture-custom'
        policy['rules'] = [{'id': 'fixture', 'pattern': 'CUSTOM_SECRET_[A-Z]+', 'replacement': '[REDACTED]'}]
        policy_path = self.root / 'custom-policy.json'
        policy_path.write_bytes(snapshot.json_bytes(policy))
        value = snapshot.build_snapshot(records[0], self.cache, split_policy='exploratory_train',
                                        screen_policy_path=policy_path)
        prepared = snapshot.prepare_single_request(value, policy_path=policy_path,
                                                   max_request_bytes=4096, max_context_bytes=100_000)
        request_raw = prepared[3]
        self.assertNotIn(b'YWxhZGRpbjpvcGVuc2VzYW1l', request_raw)
        self.assertNotIn(b'CUSTOM_SECRET_VALUE', request_raw)
        snapshot.validate_staged_artifacts(prepared[6], request_raw, prepared[4], prepared[5],
                                           prepared[7], policy_path.read_bytes())

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
        destination = self.root / 'snapshot.json'
        with mock.patch.object(snapshot.os, 'name', 'nt'), \
                mock.patch.object(Path, 'resolve', side_effect=AssertionError('resolved Windows path')):
            with self.assertRaisesRegex(ValueError, 'Windows private snapshot output is disabled'):
                snapshot.write_private_snapshot(self.root / 'snapshot.json', {'ok': True})
            with self.assertRaisesRegex(ValueError, 'Windows private snapshot output is disabled'):
                snapshot.write_private_bytes(self.root / 'bytes.bin', b'private')
        self.assertFalse(destination.exists())
        self.assertFalse((self.root / 'bytes.bin').exists())

    @unittest.skipUnless(os.name == 'posix', 'POSIX directory-handle verification is POSIX-specific')
    def test_posix_private_write_uses_existing_protected_parent(self):
        destination = self.root / 'private-output.json'
        snapshot.write_private_snapshot(destination, {'ok': True})
        self.assertEqual(json.loads(destination.read_bytes()), {'ok': True})
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)

    @unittest.skipUnless(os.name == 'posix', 'POSIX directory-handle verification is POSIX-specific')
    def test_posix_parent_swap_after_git_preflight_is_rejected(self):
        parent = self.root / 'swap-parent'
        original = self.root / 'original-parent'
        target = self.root / 'attacker-target'
        parent.mkdir(mode=0o700)
        target.mkdir(mode=0o700)
        destination = parent / 'snapshot.json'

        def swap_after_preflight(_path):
            parent.rename(original)
            parent.symlink_to(target, target_is_directory=True)

        with mock.patch.object(snapshot, '_require_ignored_untracked', side_effect=swap_after_preflight):
            with self.assertRaisesRegex(ValueError, 'private output path changed during write'):
                snapshot.write_private_snapshot(destination, {'ok': True})
        self.assertFalse((target / 'snapshot.json').exists())
        self.assertFalse((original / 'snapshot.json').exists())
        parent.unlink()

    @unittest.skipUnless(os.name == 'nt', 'Windows CLI path handling is Windows-specific')
    def test_windows_cli_fails_before_opening_inventory_or_source(self):
        argv = ['source_bound_snapshot.py', '--inventory', str(self.root / 'inventory.jsonl'),
                '--record', '1', '--provider-cache', str(self.cache),
                '--split-policy', 'exploratory_train', '--output', str(self.root / 'snapshot.json')]
        with mock.patch('sys.argv', argv), \
                mock.patch.object(snapshot.os, 'name', 'nt'), \
                mock.patch.object(Path, 'open', side_effect=AssertionError('opened private input')) as opened:
            with self.assertRaisesRegex(SystemExit, 'Windows private snapshot output is disabled'):
                snapshot.main()
        opened.assert_not_called()

    @unittest.skipUnless(os.name == 'nt', 'Windows ACL verification is Windows-specific')
    def test_windows_existing_directory_acl_is_unchanged_when_write_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=os.environ['PROGRAMDATA']) as sandbox:
            existing = Path(sandbox)
            before = subprocess.run(['icacls.exe', str(existing)], capture_output=True, check=True).stdout
            with self.assertRaisesRegex(ValueError, 'Windows private snapshot output is disabled'):
                snapshot.write_private_snapshot(existing / 'snapshot.json', {'schema_version': snapshot.SCHEMA})
            after = subprocess.run(['icacls.exe', str(existing)], capture_output=True, check=True).stdout
            self.assertEqual(before, after)

    @unittest.skipUnless(os.name == 'nt', 'Windows ACL verification is Windows-specific')
    def test_windows_junction_path_is_refused_without_writing_target(self):
        with tempfile.TemporaryDirectory(dir=os.environ['PROGRAMDATA']) as sandbox:
            target = Path(sandbox) / 'target'
            junction = Path(sandbox) / 'junction'
            target.mkdir()
            link = subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(junction), str(target)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            self.assertEqual(link.returncode, 0, 'could not create disposable junction for regression')
            with mock.patch.object(Path, 'resolve', side_effect=AssertionError('resolved junction path')):
                with self.assertRaisesRegex(ValueError, 'Windows private snapshot output is disabled'):
                    snapshot.write_private_snapshot(junction / 'snapshot.json', {'ok': True})
            self.assertFalse((target / 'snapshot.json').exists())
            junction.rmdir()


@unittest.skipUnless(os.name == 'nt', 'Windows direct-library guard regression')
class WindowsSourceBoundFailClosedTests(unittest.TestCase):
    def test_builder_verifier_and_manifest_reader_refuse_retargeted_junction_before_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / 'first', root / 'second'
            first.mkdir(); second.mkdir()
            (first / 'session.jsonl').write_bytes(b'fixture source A')
            (second / 'session.jsonl').write_bytes(b'fixture source B')
            junction = root / 'sessions'
            made = subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(junction), str(first)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if made.returncode != 0:
                self.skipTest('junction creation unavailable')
            # Change the target after a hypothetical inventory was created.
            junction.rmdir()
            subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(junction), str(second)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            record = {'source_file': str(junction / 'session.jsonl')}
            with mock.patch.object(Path, 'stat', side_effect=AssertionError('junction traversed')), \
                 mock.patch.object(Path, 'lstat', side_effect=AssertionError('junction traversed')), \
                 mock.patch.object(Path, 'open', side_effect=AssertionError('junction traversed')):
                with self.assertRaisesRegex(ValueError, 'requires POSIX no-follow'):
                    snapshot.verify_selected_source(record, max_session_bytes=4096, max_line_bytes=4096)
                with self.assertRaisesRegex(ValueError, 'require POSIX no-follow'):
                    snapshot.build_snapshot(record, root / 'cache.json', split_policy='exploratory_train')
                with self.assertRaisesRegex(ValueError, 'require POSIX no-follow'):
                    snapshot.read_reserved_session_manifest(junction / 'reserved.json')
            self.assertEqual((first / 'session.jsonl').read_bytes(), b'fixture source A')
            self.assertEqual((second / 'session.jsonl').read_bytes(), b'fixture source B')
            junction.rmdir()


class StagedBindingTests(unittest.TestCase):
    """In-memory staged artifacts exercise the consumer on every platform."""

    def setUp(self):
        policy_path = Path(__file__).with_name('private-corpus-screen.example.json')
        policy = json.loads(policy_path.read_bytes())
        policy['revision'] = 'fixture-custom'
        policy['rules'] = [{'id': 'custom-secret', 'pattern': 'CUSTOM_SECRET_[A-Z]+', 'replacement': '[REDACTED]'}]
        self.policy_raw = snapshot.json_bytes(policy)
        self.snapshot_sha = 'f' * 64
        self.group = 'a' * 64
        self.source_text = 'b' * 64
        self.projection = {
            'schema_version': snapshot.SCHEMA,
            'materialization': {'kind': 'origin-verified-private-projection',
                'origin_snapshot_sha256': self.snapshot_sha, 'source_and_cache_verified_on_origin_host': True,
                'full_source_paths_transferred': False, 'provider_extension_payloads_transferred': False},
            'source': {'project': 'fixture', 'project_association': 'inventory_label_unverified', 'verified': True,
                'source_group_sha256': self.group, 'source_file_sha256_at_inventory': 'c' * 64,
                'source_prefix_sha256': 'd' * 64, 'source_line_sha256': 'e' * 64,
                'user_text_sha256': self.source_text, 'source_prefix_bytes': 50, 'selected_line': 1},
            'split': {'assignment': 'train', 'policy': 'exploratory_train', 'heldout_separation': 'unknown',
                'known_reserved_group_exclusion': 'checked', 'known_reserved_groups_checked': 0,
                'known_reserved_group_sha256': [], 'known_reserved_source_identities': [],
                'known_reserved_manifest_sha256': None, 'known_reserved_groups_coverage': 'unknown',
                'known_reserved_project_labels': ['reserved-project']},
            'provider_list': {'cache_sha256': '1' * 64, 'screen_policy_sha256': snapshot.sha(self.policy_raw),
                              'models': [{'model_id': 'fixture-model', 'provider_supported_efforts': []}]},
            'access_claims': {'provider_listing_is_entitlement': False, 'subscription_entitlement': 'unknown',
                              'historical_session_effort': 'unknown', 'historical_model_access': 'unknown'},
            'training_claims': {'task_outcome': 'unknown', 'teacher_proposal': 'none',
                                'review_status': 'unreviewed', 'promotion_eligible': False},
            'extensions': {},
        }
        self.request = {'schema_version': 'lfg.source-bound-teacher-request.v1',
            'id': snapshot.sha((self.group + '\0' + self.source_text + '\0' + self.snapshot_sha).encode()),
            'project': 'fixture', 'project_group': self.group, 'family': self.group, 'text': 'fixture request',
            'screened_text_sha256': snapshot.sha(b'fixture request'), 'source_text_sha256_claim': self.source_text,
            'candidate_file_sha256_claim': self.snapshot_sha, 'source_bound_snapshot_sha256': self.snapshot_sha,
            'source_prefix_sha256_claim': 'd' * 64, 'source_line_sha256_claim': 'e' * 64,
            'split': 'train', 'split_policy': 'exploratory_train', 'heldout_separation': 'unknown',
            'review_status': 'unreviewed', 'outcome': 'unknown'}
        self.manifest = {'schema_version': 'lfg.teacher-request-selection-result.v1', 'requests': 1,
            'test_opened': False, 'validation_opened': False, 'heldout_separation': 'unknown',
            'human_reviewed': False, 'promotion_eligible': False, 'projects': {'fixture': 1},
            'source_bound_snapshot_sha256': self.snapshot_sha}
        self.context = {'schema_version': 'lfg.project-training-context.v1',
            'source_bound_snapshot_sha256': self.snapshot_sha, 'materialized_from_verified_snapshot': True,
            'projects': {'fixture': {'text': 'fixture context', 'sha256': snapshot.sha(b'fixture context'),
                'source_bound_snapshot_sha256': self.snapshot_sha, 'source_text_sha256_claim': self.source_text,
                'provider_cache_sha256': '1' * 64, 'heldout_separation': 'unknown'}}}
        self.bundle = {'schema_version': 'lfg.source-bound-staging-bundle.v1',
            'origin_snapshot_sha256': self.snapshot_sha, 'projection_kind': 'origin-verified-private-projection',
            'full_source_paths_transferred': False, 'provider_extension_payloads_transferred': False,
            'screen_policy_sha256': snapshot.sha(self.policy_raw)}

    def check(self):
        projection = snapshot.json_bytes(self.projection)
        request = snapshot.json_bytes(self.request)
        self.manifest['requests_sha256'] = snapshot.sha(request)
        manifest = snapshot.json_bytes(self.manifest)
        context = snapshot.json_bytes(self.context)
        self.bundle.update({'staged_snapshot_sha256': snapshot.sha(projection), 'requests_sha256': snapshot.sha(request),
                            'selection_manifest_sha256': snapshot.sha(manifest), 'project_context_sha256': snapshot.sha(context)})
        return snapshot.validate_staged_artifacts(projection, request, manifest, context,
                                                   snapshot.json_bytes(self.bundle), self.policy_raw)

    def test_rehashed_project_group_family_and_request_id_must_match_source(self):
        self.assertEqual(self.check()['project'], 'fixture')
        for key, changed in [('project', 'reserved-project'), ('project_group', '2' * 64),
                             ('family', '2' * 64), ('id', '3' * 64)]:
            with self.subTest(field=key):
                old = self.request[key]
                self.request[key] = changed
                if key == 'project':
                    self.context['projects']['reserved-project'] = self.context['projects'].pop('fixture')
                    self.manifest['projects'] = {'reserved-project': 1}
                with self.assertRaises(ValueError):
                    self.check()
                self.request[key] = old
                if key == 'project':
                    self.context['projects']['fixture'] = self.context['projects'].pop('reserved-project')
                    self.manifest['projects'] = {'fixture': 1}

    def test_manifest_and_context_project_sets_must_match_source_exactly(self):
        for projects in ({'other': 1}, {'fixture': 1, 'other': 1}, {'fixture': True}):
            with self.subTest(projects=projects):
                self.manifest['projects'] = projects
                with self.assertRaises(ValueError):
                    self.check()
        self.manifest['projects'] = {'fixture': 1}
        self.context['projects']['other'] = dict(self.context['projects']['fixture'])
        with self.assertRaises(ValueError):
            self.check()

    def test_safe_unknown_fields_remain_inert_but_both_policies_screen_extensions(self):
        containers = [self.projection, self.projection['source'], self.projection['split'],
                      self.projection['provider_list'], self.projection['provider_list']['models'][0],
                      self.projection['materialization'], self.projection['access_claims'],
                      self.projection['training_claims'], self.request, self.manifest,
                      self.context, self.context['projects']['fixture'], self.bundle]
        for container in containers:
            container['safe_extra'] = {'nested': ['retained']}
        self.assertEqual(self.check()['project'], 'fixture')
        for container in containers:
            for key, value in [('safe_extra', {'nested': ['CUSTOM_SECRET_VALUE']}),
                               ('safe_extra', {'nested': ['Basic YWxhZGRpbjpvcGVuc2VzYW1l']}),
                               ('safe_extra', {'nested': ['Cookie: session=fixture-cookie-value']}),
                               ('authorization', 'fixture'), ('cookie', 'fixture')]:
                with self.subTest(container=id(container), key=key, kind=value if isinstance(value, str) else 'nested'):
                    previous = container.get(key)
                    container[key] = value
                    with self.assertRaises(ValueError):
                        self.check()
                    if previous is None:
                        del container[key]
                    else:
                        container[key] = previous
        self.assertEqual(self.check()['project'], 'fixture')

    def test_sensitive_provider_keys_and_basic_values_are_screened(self):
        policy, rules, _ = snapshot.screen_private_corpus.load_policy(snapshot._default_screen_policy_path(), 1024 * 1024)
        for key in ('authorization', 'proxyAuthorization', 'cookie', 'setCookie'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'sensitive provider extension key'):
                snapshot._safe_extensions({key: 'fixture'}, rules, policy['limits'])
        result = snapshot._safe_extensions({'note': 'Basic YWxhZGRpbjpvcGVuc2VzYW1l'}, rules, policy['limits'])
        self.assertNotIn('YWxhZGRpbjpvcGVuc2VzYW1l', result['note'])
        result = snapshot._safe_extensions({'note': 'Cookie: session=fixture-cookie-value'}, rules, policy['limits'])
        self.assertNotIn('fixture-cookie-value', result['note'])

    def test_structured_unknown_data_cannot_hide_inside_provider_scalar_fields(self):
        model = self.projection['provider_list']['models'][0]
        model['provider_supported_efforts'] = [{'nested': 'CUSTOM_SECRET_VALUE'}]
        with self.assertRaisesRegex(ValueError, 'provider model metadata'):
            self.check()
        model['provider_supported_efforts'] = []
        model['description'] = {'nested': 'CUSTOM_SECRET_VALUE'}
        with self.assertRaisesRegex(ValueError, 'provider model metadata'):
            self.check()

    def test_resealed_known_strings_must_pass_baseline_and_configured_policy(self):
        model = self.projection['provider_list']['models'][0]
        entry = self.context['projects']['fixture']
        for secret in ('Basic YWxhZGRpbjpvcGVuc2VzYW1l', 'Cookie: session=fixture-cookie-value',
                       'CUSTOM_SECRET_VALUE'):
            for field in ('model_id', 'display_name', 'description', 'provider_supported_efforts', 'fetched_at',
                          'request', 'context'):
                with self.subTest(field=field, policy='baseline-or-configured'):
                    provider = self.projection['provider_list']
                    if field == 'fetched_at':
                        provider[field] = secret
                    elif field == 'request':
                        self.request['text'] = secret
                        self.request['screened_text_sha256'] = snapshot.sha(secret.encode())
                    elif field == 'context':
                        entry['text'] = secret
                        entry['sha256'] = snapshot.sha(secret.encode())
                    else:
                        model[field] = [secret] if field == 'provider_supported_efforts' else secret
                    with self.assertRaises(ValueError):
                        self.check()
                    if field == 'fetched_at':
                        del provider[field]
                    elif field == 'request':
                        self.request['text'] = 'fixture request'
                        self.request['screened_text_sha256'] = snapshot.sha(b'fixture request')
                    elif field == 'context':
                        entry['text'] = 'fixture context'
                        entry['sha256'] = snapshot.sha(b'fixture context')
                    elif field == 'model_id':
                        model[field] = 'fixture-model'
                    elif field == 'provider_supported_efforts':
                        model[field] = []
                    else:
                        del model[field]
        self.assertEqual(self.check()['project'], 'fixture')

    def test_configured_replacement_cannot_reintroduce_a_baseline_secret(self):
        policy = json.loads(self.policy_raw)
        policy['rules'][0]['replacement'] = 'Basic YWxhZGRpbjpvcGVuc2VzYW1l'
        custom, custom_rules, _ = snapshot.screen_private_corpus.load_policy_bytes(snapshot.json_bytes(policy), 1024 * 1024)
        baseline, baseline_rules, _ = snapshot.screen_private_corpus.load_policy(snapshot._default_screen_policy_path(), 1024 * 1024)
        policies = [(baseline, baseline_rules), (custom, custom_rules)]
        with self.assertRaisesRegex(ValueError, 'requiring further screening'):
            snapshot._screen_text_policies('CUSTOM_SECRET_VALUE', policies)
        with self.assertRaisesRegex(ValueError, 'requiring further screening'):
            snapshot._screen_extension_policies({'note': 'CUSTOM_SECRET_VALUE'}, policies)


if __name__ == '__main__':
    unittest.main()
