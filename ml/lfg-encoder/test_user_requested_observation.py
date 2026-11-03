"""Synthetic-only contract tests; no owner session or routing data is read."""
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import inventory_codex_sessions as inventory
import user_requested_observation as subject
import source_bound_snapshot


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@unittest.skipUnless(os.name == 'posix', 'source-bound observations are fail-closed on Windows')
class UserRequestedObservationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'fixture-project'
        self.project.mkdir()
        self.source = self.root / 'fixture-session.jsonl'
        self.text = 'Use Model Z for planning at high effort; emoji 😀 is fine.'
        self.write_session(self.text)
        status, records = inventory.scan_file(
            self.source, {'fixture-project': inventory.normalize(str(self.project))}, 16_384, 8192)
        self.assertEqual(status, 'scanned')
        self.record = records[0]
        self.policy = Path(__file__).with_name('private-corpus-screen.example.json')
        self.config = {
            'schema_version': subject.CONFIG_SCHEMA,
            'phase_labels': ['plan-phase', 'planning', 'execution-phase'],
            'claim_polarities': ['affirmed', 'negated', 'quoted', 'uncertain'],
            'max_source_prefix_bytes': 16_384,
            'max_line_bytes': 8192,
            'max_user_message_bytes': 2048,
            'max_span_count': 8,
            'max_span_bytes': 256,
            'max_extension_bytes': 4096,
            'max_extension_nodes': 128,
            'max_extension_depth': 8,
            'max_extension_string_bytes': 1024,
            'max_observation_bytes': 32_768,
            'max_reserved_manifest_bytes': 4096,
            'max_reserved_project_labels': 32,
            'max_claims_per_field': 8,
        }
        self.spans = self.make_spans()
        self.interpretation = {
            'phase_label': 'plan-phase', 'phase_span_id': 'phase',
            'model_claims': [{'text': 'Model Z', 'span_id': 'model', 'polarity': 'affirmed'}],
            'effort_claims': [{'text': 'high', 'span_id': 'effort', 'polarity': 'affirmed'}],
        }

    def write_session(self, text):
        header = {'type': 'session_meta', 'payload': {'cwd': str(self.project)}}
        message = {'type': 'response_item', 'payload': {
            'type': 'message', 'role': 'user',
            'content': [{'type': 'input_text', 'text': text}],
            'internal_chat_message_metadata_passthrough': {'content_item_kinds': ['user.text']},
        }}
        raw = (json.dumps(header, ensure_ascii=False) + '\n'
               + json.dumps(message, ensure_ascii=False) + '\n').encode('utf-8')
        self.source.write_bytes(raw)

    def make_spans(self):
        raw = self.text.encode('utf-8')
        spans = []
        for identifier, field, value in (('phase', 'phase', 'planning'),
                                         ('model', 'model', 'Model Z'),
                                         ('effort', 'effort', 'high')):
            start = raw.index(value.encode('utf-8'))
            spans.append({'id': identifier, 'field': field, 'start_byte': start,
                          'end_byte': start + len(value.encode('utf-8'))})
        return spans

    def build(self, *, record=None, **kwargs):
        return subject.build_observation(record or self.record, self.spans, self.interpretation,
                                         self.config, self.policy, **kwargs)

    def test_user_request_claims_are_text_free_unreviewed_and_not_teacher_labels(self):
        value = self.build(extensions={'local_note': {'keep': 'fixture extension'}})
        subject.validate_observation(value, self.config, self.policy)
        encoded = json.dumps(value, ensure_ascii=False)
        self.assertEqual(value['kind'], 'user_requested')
        self.assertEqual(value['source']['role'], 'user')
        self.assertNotIn(str(self.source), encoded)
        self.assertNotIn(self.text, encoded)
        self.assertNotIn('Model Z', encoded)
        self.assertNotIn('high', encoded)
        self.assertEqual(value['interpretation']['origin'], 'caller_supplied_interpretation')
        self.assertEqual(value['interpretation']['model_claims']['status'], 'unreviewed_source_claims')
        self.assertEqual(value['interpretation']['effort_claims']['status'], 'unreviewed_source_claims')
        self.assertEqual(value['interpretation']['model_claims']['items'][0]['polarity'], 'affirmed')
        self.assertEqual(value['interpretation']['claim_review_status'], 'unreviewed')
        self.assertEqual(value['state'], {
            'provider_entitlement': 'unknown', 'executor_effort': 'unknown',
            'execution': 'unknown', 'outcome': 'unknown', 'total_agent_count': None,
            'review_status': 'unreviewed', 'promotion_eligible': False,
        })
        summary = subject.consume_observation(value, self.config, self.policy)
        self.assertEqual(summary['kind'], 'user_requested')
        self.assertNotIn('request', summary)
        self.assertNotIn('response', summary)
        self.assertNotIn('teacher', summary)
        self.assertNotIn('preference', summary)
        self.assertEqual(summary['state']['outcome'], 'unknown')

    def test_negated_quoted_and_conflicting_claims_remain_neutral_and_unresolved(self):
        self.text = 'Do not use Model Z; "Model Q" was only quoted while planning.'
        self.write_session(self.text)
        status, records = inventory.scan_file(
            self.source, {'fixture-project': inventory.normalize(str(self.project))}, 16_384, 8192)
        self.assertEqual(status, 'scanned')
        self.record = records[0]
        raw = self.text.encode('utf-8')

        def span(identifier, field, literal):
            start = raw.index(literal.encode('utf-8'))
            return {'id': identifier, 'field': field, 'start_byte': start,
                    'end_byte': start + len(literal.encode('utf-8'))}

        spans = [span('phase', 'phase', 'planning'),
                 span('negative-model', 'model', 'Model Z'),
                 span('quoted-model', 'model', 'Model Q')]
        interpretation = {
            'phase_label': 'planning', 'phase_span_id': 'phase',
            'model_claims': [
                {'text': 'Model Z', 'span_id': 'negative-model', 'polarity': 'negated'},
                {'text': 'Model Q', 'span_id': 'quoted-model', 'polarity': 'quoted'},
            ],
            'effort_claims': [],
        }
        value = subject.build_observation(self.record, spans, interpretation, self.config, self.policy)
        summary = subject.consume_observation(value, self.config, self.policy)
        self.assertEqual(value['interpretation']['phase_label'], 'planning')
        claims = summary['interpretation']['model_claims']
        self.assertEqual(claims['status'], 'unreviewed_source_claims')
        self.assertEqual([item['polarity'] for item in claims['items']], ['negated', 'quoted'])
        self.assertTrue(all(item['status'] == 'unreviewed_source_claim' for item in claims['items']))
        self.assertEqual(summary['interpretation']['claim_review_status'], 'unreviewed')
        self.assertNotIn('preference', summary)
        encoded = json.dumps(value, ensure_ascii=False)
        self.assertNotIn('Model Z', encoded)
        self.assertNotIn('Model Q', encoded)
        self.assertNotIn(self.text, encoded)

        missing_polarity = dict(interpretation, model_claims=[{'text': 'Model Z', 'span_id': 'negative-model'}])
        with self.assertRaisesRegex(ValueError, 'caller-interpreted model claim'):
            subject.build_observation(self.record, spans, missing_polarity, self.config, self.policy)

    def test_missing_model_and_effort_remain_unknown_without_defaults(self):
        spans = [self.spans[0]]
        interpretation = {'phase_label': 'plan-phase', 'phase_span_id': 'phase',
                          'model_claims': [], 'effort_claims': []}
        value = subject.build_observation(self.record, spans, interpretation, self.config, self.policy)
        self.assertEqual(value['interpretation']['model_claims'], {'status': 'unknown'})
        self.assertEqual(value['interpretation']['effort_claims'], {'status': 'unknown'})
        self.assertEqual(value['state']['executor_effort'], 'unknown')
        self.assertIsNone(value['state']['total_agent_count'])

    def test_appended_suffix_is_allowed_only_while_selected_prefix_verifies(self):
        value = self.build()
        with self.source.open('ab') as stream:
            stream.write(b'{"type":"later_fixture_event"}\n')
        self.assertEqual(self.build()['source']['source_prefix_sha256'], value['source']['source_prefix_sha256'])
        self.write_session('Use a different fixture request.')
        with self.assertRaisesRegex(ValueError, 'identity|prefix|line|source'):
            self.build()

    def test_selected_source_is_rechecked_before_observation_returns(self):
        original = source_bound_snapshot.verify_selected_source
        calls = 0

        def change_after_first_verification(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = original(*args, **kwargs)
            if calls == 1:
                self.write_session('Changed after initial verification.')
            return result

        with mock.patch.object(source_bound_snapshot, 'verify_selected_source',
                               side_effect=change_after_first_verification):
            with self.assertRaisesRegex(ValueError, 'identity|changed|source'):
                self.build()
        self.assertEqual(calls, 2)

    def test_altered_prefix_line_and_text_digests_are_rejected(self):
        for field in ('source_prefix_sha256', 'source_line_sha256', 'text_sha256'):
            changed = dict(self.record, **{field: '0' * 64})
            with self.subTest(identity_digest=field), self.assertRaisesRegex(
                    ValueError, 'identity|prefix|line|source'):
                subject.build_observation(changed, self.spans, self.interpretation,
                                          self.config, self.policy)

    def test_invalid_utf8_span_boundary_and_mismatched_text_are_rejected(self):
        bad = copy.deepcopy(self.spans)
        emoji = self.text.encode('utf-8').index('😀'.encode('utf-8'))
        bad[0]['start_byte'] = emoji + 1
        bad[0]['end_byte'] = emoji + 2
        with self.assertRaisesRegex(ValueError, 'UTF-8'):
            subject.build_observation(self.record, bad, self.interpretation, self.config, self.policy)
        changed = dict(self.interpretation, model_claims=[
            {'text': 'Different model', 'span_id': 'model', 'polarity': 'affirmed'}])
        with self.assertRaisesRegex(ValueError, 'does not match a unique source span'):
            subject.build_observation(self.record, self.spans, changed, self.config, self.policy)
        changed_phase = dict(self.interpretation, phase_label='unconfigured-phase')
        with self.assertRaisesRegex(ValueError, 'phase interpretation'):
            subject.build_observation(self.record, self.spans, changed_phase, self.config, self.policy)

    def test_assistant_message_cannot_be_used_as_user_request_evidence(self):
        assistant = {'type': 'response_item', 'payload': {
            'type': 'message', 'role': 'assistant',
            'content': [{'type': 'input_text', 'text': self.text}],
            'internal_chat_message_metadata_passthrough': {'content_item_kinds': ['user.text']},
        }}
        header = {'type': 'session_meta', 'payload': {'cwd': str(self.project)}}
        selected = json.dumps(assistant, ensure_ascii=False).encode('utf-8')
        prefix = (json.dumps(header, ensure_ascii=False).encode('utf-8') + b'\n' + selected + b'\n')
        changed = dict(self.record, source_file_sha256=digest(prefix),
                       source_prefix_sha256=digest(prefix), source_prefix_bytes=len(prefix),
                       source_line_sha256=digest(selected), text_sha256=digest(self.text.encode('utf-8')))
        self.source.write_bytes(prefix)
        with self.assertRaisesRegex(ValueError, 'user-text message'):
            subject.build_observation(changed, self.spans, self.interpretation, self.config, self.policy)

    def test_extension_values_are_screened_and_reserved_or_secret_keys_rejected(self):
        value = self.build(extensions={'note': 'password=fixture-secret-value-98765'})
        self.assertNotIn('fixture-secret-value-98765', json.dumps(value))
        self.assertIn('REDACTED', value['extensions']['note'])
        for key in ('kind', 'outcome', 'teacher', 'source_path', 'session_token'):
            with self.subTest(extension_key=key), self.assertRaisesRegex(ValueError, 'extension key'):
                self.build(extensions={key: 'value'})
        for copied in ('Model Z', self.text, str(self.source)):
            with self.subTest(copied_source=copied), self.assertRaisesRegex(
                    ValueError, 'copies selected source text'):
                self.build(extensions={'note': copied})

    def test_observation_extensions_apply_baseline_and_configured_policy(self):
        policy = json.loads(self.policy.read_bytes())
        policy['revision'] = 'fixture-custom'
        policy['rules'] = [{'id': 'fixture', 'pattern': 'CUSTOM_SECRET_[A-Z]+', 'replacement': '[REDACTED]'}]
        custom = self.root / 'custom-policy.json'
        custom.write_bytes(subject.canonical(policy))
        value = subject.build_observation(self.record, self.spans, self.interpretation, self.config, custom,
                    extensions={'note': 'Basic YWxhZGRpbjpvcGVuc2VzYW1l CUSTOM_SECRET_VALUE', 'safe_extra': 'retained'})
        self.assertEqual(value['extensions']['safe_extra'], 'retained')
        self.assertNotIn('YWxhZGRpbjpvcGVuc2VzYW1l', json.dumps(value))
        self.assertNotIn('CUSTOM_SECRET_VALUE', json.dumps(value))
        for key in ('privateKey', 'PrivateKey', 'oauthToken', 'OAuthToken', 'authorization', 'cookie'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'extension key'):
                subject.build_observation(self.record, self.spans, self.interpretation, self.config, custom,
                                          extensions={key: 'fixture'})
        tampered = copy.deepcopy(value)
        tampered['extensions']['note'] = 'Basic YWxhZGRpbjpvcGVuc2VzYW1l'
        with self.assertRaisesRegex(ValueError, 'requires screening'):
            subject.validate_observation(tampered, self.config, custom)

    def test_reserved_identity_extensions_use_pinned_configured_policy(self):
        policy = json.loads(self.policy.read_bytes())
        policy['revision'] = 'fixture-custom'
        policy['rules'] = [{'id': 'fixture', 'pattern': 'CUSTOM_SECRET_[A-Z]+', 'replacement': '[REDACTED]'}]
        custom = self.root / 'custom-policy.json'
        custom.write_bytes(subject.canonical(policy))
        identity = source_bound_snapshot._source_identity_claim(self.record)
        identity.update({'source_file_sha256': '1' * 64, 'source_prefix_sha256': '2' * 64,
                         'source_line_sha256': '3' * 64, 'text_sha256': '4' * 64,
                         'note': 'CUSTOM_SECRET_VALUE'})
        manifest = self.root / 'reserved-custom.json'
        manifest.write_bytes(subject.canonical({'schema_version': subject.MANIFEST_SCHEMA,
                                                'coverage': 'partial', 'groups': [], 'identities': [identity]}))
        manifest.chmod(0o600)
        pinned = subject.pin_reserved_manifest(manifest, self.config, screen_policy_path=custom)
        self.assertNotIn('CUSTOM_SECRET_VALUE', json.dumps(pinned))
        value = subject.build_observation(self.record, self.spans, self.interpretation, self.config, custom,
                                          reserved=pinned)
        self.assertEqual(value['reservation']['source_identities_checked'], 1)
        self.assertEqual(value['state']['outcome'], 'unknown')

    def test_observation_consumer_rejects_success_teacher_and_agent_injection(self):
        valid = self.build()
        for section, field, replacement in (
                ('state', 'outcome', 'success'), ('state', 'promotion_eligible', True),
                ('state', 'total_agent_count', 3), ('interpretation', 'teacher_proposal', 'accepted'),
                ('interpretation', 'claim_review_status', 'reviewed')):
            changed = copy.deepcopy(valid)
            changed[section][field] = replacement
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                subject.consume_observation(changed, self.config, self.policy)

    @unittest.skipUnless(os.name == 'posix', 'reserved-manifest handle checks are POSIX-specific')
    def test_reserved_group_exclusion_happens_before_selected_source_open(self):
        manifest = self.root / 'reserved.json'
        manifest.write_bytes(subject.canonical({
            'schema_version': subject.MANIFEST_SCHEMA, 'coverage': 'partial',
            'groups': [source_bound_snapshot.source_group_id(self.record)], 'identities': []}))
        manifest.chmod(0o600)
        reserved = subject.pin_reserved_manifest(manifest, self.config)
        with mock.patch.object(source_bound_snapshot, 'verify_selected_source',
                               side_effect=AssertionError('selected source opened')) as opened:
            with self.assertRaisesRegex(ValueError, 'reserved group'):
                self.build(reserved=reserved)
        opened.assert_not_called()
        with mock.patch.object(source_bound_snapshot, 'verify_selected_source',
                               side_effect=AssertionError('selected source opened')) as opened:
            with self.assertRaisesRegex(ValueError, 'reserved group'):
                self.build(reserved_project_labels=['FIXTURE.PROJECT'])
        opened.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'reserved and source no-follow checks are POSIX-specific')
    def test_hardlink_alias_cannot_bypass_reserved_path_group_exclusion(self):
        reserved_path = self.root / 'reserved-hardlink.json'
        original_group = source_bound_snapshot.source_group_id(self.record)
        reserved_path.write_bytes(subject.canonical({
            'schema_version': subject.MANIFEST_SCHEMA, 'coverage': 'partial',
            'groups': [original_group], 'identities': []}))
        reserved_path.chmod(0o600)
        reserved = subject.pin_reserved_manifest(reserved_path, self.config)
        alias = self.root / 'session-alias.jsonl'
        os.link(self.source, alias)
        changed_record = dict(self.record, source_file=str(alias))
        self.assertNotEqual(source_bound_snapshot.source_group_id(changed_record), original_group)
        with mock.patch('source_bound_snapshot.os.fdopen', side_effect=AssertionError('read source bytes')) as opened:
            with self.assertRaisesRegex(ValueError, 'single-link regular file'):
                subject.build_observation(changed_record, self.spans, self.interpretation,
                                          self.config, self.policy, reserved=reserved)
        opened.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'reserved identity source checks are POSIX-specific')
    def test_renamed_copy_and_relocated_hardlink_match_predeclared_text_identity(self):
        identity = source_bound_snapshot._source_identity_claim(self.record)
        reserved_path = self.root / 'reserved-identities.json'
        reserved_path.write_bytes(subject.canonical({
            'schema_version': subject.MANIFEST_SCHEMA, 'coverage': 'partial', 'groups': [],
            'identities': [identity]}))
        reserved_path.chmod(0o600)
        reserved = subject.pin_reserved_manifest(reserved_path, self.config)

        copied = self.root / 'copied-session.jsonl'
        copied.write_bytes(self.source.read_bytes())
        changed_copy_record = dict(self.record, source_file=str(copied), source_file_sha256='a' * 64)

        alias = self.root / 'moved-hardlink.jsonl'
        os.link(self.source, alias)
        self.source.unlink()
        moved_record = dict(self.record, source_file=str(alias), source_file_sha256='b' * 64)
        self.assertEqual(alias.stat().st_nlink, 1)

        with mock.patch.object(source_bound_snapshot, 'verify_selected_source',
                               side_effect=AssertionError('selected source opened')) as opened:
            for candidate in (changed_copy_record, moved_record):
                with self.subTest(path=Path(candidate['source_file']).name), self.assertRaisesRegex(
                        ValueError, 'reserved group'):
                    self.build(record=candidate, reserved=reserved)
        opened.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'reserved-manifest handle checks are POSIX-specific')
    def test_stale_reserved_manifest_is_rejected_before_source_open_and_rechecked_after(self):
        manifest = self.root / 'reserved-mutable.json'
        manifest.write_bytes(subject.canonical({'schema_version': subject.MANIFEST_SCHEMA,
                                                'coverage': 'partial', 'groups': [], 'identities': []}))
        manifest.chmod(0o600)
        reserved = subject.pin_reserved_manifest(manifest, self.config)
        manifest.write_bytes(subject.canonical({'schema_version': subject.MANIFEST_SCHEMA,
                                                'coverage': 'partial',
                                                'groups': ['a' * 64], 'identities': []}))
        with mock.patch.object(source_bound_snapshot, 'verify_selected_source',
                               side_effect=AssertionError('selected source opened')) as opened:
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                self.build(reserved=reserved)
        opened.assert_not_called()

        reserved = subject.pin_reserved_manifest(manifest, self.config)
        original = source_bound_snapshot.verify_selected_source

        def mutate_after_source(*args, **kwargs):
            result = original(*args, **kwargs)
            manifest.write_bytes(subject.canonical({'schema_version': subject.MANIFEST_SCHEMA,
                                                    'coverage': 'partial', 'groups': [], 'identities': []}))
            return result

        with mock.patch.object(source_bound_snapshot, 'verify_selected_source', side_effect=mutate_after_source):
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                self.build(reserved=reserved)

    @unittest.skipUnless(os.name == 'posix', 'reserved-manifest handle checks are POSIX-specific')
    def test_reserved_manifest_rejects_symlink_and_non_private_file(self):
        raw = subject.canonical({'schema_version': subject.MANIFEST_SCHEMA,
                                 'coverage': 'partial', 'groups': [], 'identities': []})
        source = self.root / 'private-reserved.json'
        source.write_bytes(raw)
        source.chmod(0o600)
        self.assertEqual(subject.read_reserved_manifest(source, self.config)[0], [])
        symlink = self.root / 'reserved-link.json'
        symlink.symlink_to(source)
        with self.assertRaises(ValueError):
            subject.read_reserved_manifest(symlink, self.config)
        hardlink = self.root / 'reserved-hardlink.json'
        os.link(source, hardlink)
        source.chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'private regular file'):
            subject.read_reserved_manifest(source, self.config)
        hardlink.unlink()
        source.write_bytes(raw + b' ' * self.config['max_reserved_manifest_bytes'])
        source.chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'private regular file'):
            subject.read_reserved_manifest(source, self.config)

    def test_config_and_observation_limits_are_enforced(self):
        invalid = dict(self.config, max_span_count=0)
        with self.assertRaisesRegex(ValueError, 'invalid observation config'):
            subject.build_observation(self.record, self.spans, self.interpretation, invalid, self.policy)
        small = dict(self.config, max_span_bytes=2)
        with self.assertRaisesRegex(ValueError, 'source span bounds'):
            subject.build_observation(self.record, self.spans, self.interpretation, small, self.policy)


@unittest.skipUnless(os.name == 'nt', 'Windows direct-library guard regression')
class WindowsObservationGuardTests(unittest.TestCase):
    def test_builder_fails_before_retargeted_junction_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / 'first', root / 'second'
            first.mkdir(); second.mkdir()
            (first / 'session.jsonl').write_bytes(b'fixture first')
            (second / 'session.jsonl').write_bytes(b'fixture second')
            junction = root / 'sessions'
            made = subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(junction), str(first)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if made.returncode != 0:
                self.skipTest('junction creation unavailable')
            junction.rmdir()
            subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(junction), str(second)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            with mock.patch.object(Path, 'lstat', side_effect=AssertionError('junction traversed')), \
                 mock.patch.object(Path, 'stat', side_effect=AssertionError('junction traversed')):
                with self.assertRaisesRegex(ValueError, 'unavailable on Windows'):
                    subject.build_observation({}, [], {}, {}, junction / 'policy.json')
                with self.assertRaisesRegex(ValueError, 'unavailable on Windows'):
                    subject.read_reserved_manifest(junction / 'reserved.json', {})
            self.assertEqual((first / 'session.jsonl').read_bytes(), b'fixture first')
            self.assertEqual((second / 'session.jsonl').read_bytes(), b'fixture second')
            junction.rmdir()


if __name__ == '__main__':
    unittest.main()
