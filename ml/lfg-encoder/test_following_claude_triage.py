import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import corpus_audit
import following_claude_triage as claude
import following_review_triage as triage
import review_corpus


class ClaudeFollowingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.path = self.root / 'session.jsonl'
        self.config, _ = triage.load_config(Path(triage.__file__).with_name('following-review-triage.example.json'))

    def human(self, uid, parent, text, session='session'):
        return {'type': 'user', 'turnOrigin': 'human', 'isSidechain': False, 'cwd': str(self.root),
                'uuid': uid, 'parentUuid': parent, 'sessionId': session,
                'message': {'role': 'user', 'content': text}}

    def assistant(self, uid, parent, session='session'):
        return {'type': 'assistant', 'uuid': uid, 'parentUuid': parent, 'sessionId': session,
                'message': {'role': 'assistant', 'content': 'private fixture'}}

    def write(self, rows):
        lines = [(json.dumps(row) + '\n').encode() for row in rows]
        self.path.write_bytes(b''.join(lines))
        first = lines[0]
        source = {'source_file': str(self.path), 'project_root': str(self.root), 'source_line': 1,
                  'source_prefix_bytes': len(first), 'source_prefix_sha256': hashlib.sha256(first).hexdigest(),
                  'source_line_sha256': hashlib.sha256(first.rstrip(b'\r\n')).hexdigest(),
                  'source_file_sha256': hashlib.sha256(b''.join(lines)).hexdigest(),
                  'text_sha256': hashlib.sha256(b'original request').hexdigest()}
        return [{'source': source, 'text': 'original request', 'report': {'id': 'fixture'}}]

    def test_unique_descendant_chain_finds_marker_without_exposing_text(self):
        candidates = self.write([self.human('u1', None, 'original request'), self.assistant('a1', 'u1'),
                                 self.human('u2', 'a1', 'wrong chat')])
        result = claude.session_scan(self.path, candidates, self.config)
        self.assertEqual(result[0]['hits'][0]['marker_id'], 'wrong-chat')
        self.assertEqual(result[0]['hits'][0]['source_line'], 3)
        self.assertNotIn('wrong chat', json.dumps(result))
        self.assertEqual(result[0]['context_end'], 'no_observed_child')

    def test_branch_is_reported_as_ambiguous_instead_of_picking_a_child(self):
        candidates = self.write([self.human('u1', None, 'original request'), self.assistant('a1', 'u1'),
                                 self.human('u2', 'a1', 'wrong chat'), self.human('u3', 'a1', 'other')])
        result = claude.session_scan(self.path, candidates, self.config)
        self.assertEqual(result[0]['context_end'], 'ambiguous_branch')
        self.assertEqual(result[0]['hits'], [])

    def test_other_session_does_not_become_a_descendant(self):
        candidates = self.write([self.human('u1', None, 'original request'),
                                 self.human('u2', 'u1', 'wrong chat', session='another')])
        result = claude.session_scan(self.path, candidates, self.config)
        self.assertEqual(result[0]['context_end'], 'no_observed_child')
        self.assertEqual(result[0]['hits'], [])

    def test_conflicting_identity_and_record_limit_fail_closed(self):
        candidates = self.write([self.human('u1', None, 'original request'), self.assistant('a1', 'u1'),
                                 self.human('u2', 'a1', 'wrong chat'), self.human('u2', 'u1', 'different')])
        result = claude.session_scan(self.path, candidates, self.config)
        self.assertIn(result[0]['context_end'], ('ambiguous_branch', 'ambiguous_identity'))
        self.assertEqual(result[0]['hits'], [])
        self.config['max_ancestry_records'] = 2
        with self.assertRaisesRegex(ValueError, 'ancestry record limit'):
            claude.session_scan(self.path, candidates, self.config)

    def test_source_prefix_change_is_rejected(self):
        candidates = self.write([self.human('u1', None, 'original request'), self.human('u2', 'u1', 'wrong chat')])
        self.path.write_bytes(self.path.read_bytes().replace(b'original request', b'changed request!'))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            claude.session_scan(self.path, candidates, self.config)

    def test_append_marks_unfinished_chain_and_changed_scanned_bytes_fail(self):
        candidates = self.write([self.human('u1', None, 'original request'), self.assistant('a1', 'u1')])
        original_stamp = claude.stamp(self.path)
        calls = 0
        real_stamp = claude.stamp
        def stamp_with_append(path):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original_stamp
            with self.path.open('ab') as stream:
                stream.write((json.dumps(self.human('u2', 'a1', 'later message')) + '\n').encode())
            return real_stamp(path)
        with patch.object(claude, 'stamp', side_effect=stamp_with_append):
            result = claude.session_scan(self.path, candidates, self.config)
        self.assertEqual(result[0]['context_end'], 'source_changed_after_scan')

        candidates = self.write([self.human('u1', None, 'original request'), self.assistant('a1', 'u1')])
        original_stamp = claude.stamp(self.path)
        calls = 0
        def stamp_with_mutation(path):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original_stamp
            self.path.write_bytes(self.path.read_bytes().replace(b'original request', b'changed request!'))
            changed = real_stamp(path)
            return changed[0], changed[1] + 1, changed[2]
        with patch.object(claude, 'stamp', side_effect=stamp_with_mutation):
            with self.assertRaisesRegex(ValueError, 'source changed during context scan'):
                claude.session_scan(self.path, candidates, self.config)

    def test_training_workspace_integration_and_explicit_ancestry_limit(self):
        candidates = self.write([self.human('u1', None, 'original request'), self.assistant('a1', 'u1'),
                                 self.human('u2', 'a1', 'wrong chat')])
        source = {**candidates[0]['source'], 'kind': 'claude-session-human-text', 'project': 'fixture'}
        row = {'id': 'fixture', 'text': 'original request', 'family': 'family', 'verified': False,
               'intents': [], 'capabilities': [], 'risks': [], 'labels_pending': True, 'source': source}
        (self.root / 'taxonomy.json').write_text(json.dumps({'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
            'intents': ['build'], 'capabilities': ['write'], 'risks': ['sensitive']}))
        (self.root / 'source.jsonl').write_bytes(review_corpus.encoded_rows([row]))
        plan = {'schema_version': corpus_audit.PLAN_SCHEMA, 'taxonomy': 'taxonomy.json', 'review_ledger': None,
                'sources': [{'path': 'source.jsonl', 'project': 'fixture', 'split': 'train'}]}
        (self.root / 'plan.json').write_text(json.dumps(plan))
        workspace = self.root / 'workspace'
        review_corpus.init_workspace(self.root / 'plan.json', workspace)
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps(self.config))
        result = triage.triage(workspace, config_path)
        self.assertEqual(result['counts']['checked_claude-session-human-text'], 1)
        self.assertEqual(result['flagged'], 1)
        self.assertFalse(result['review_decisions_made'])
        self.config.pop('max_ancestry_records')
        config_path.write_text(json.dumps(self.config))
        with patch.object(claude, 'session_scan', side_effect=AssertionError('Claude source opened')):
            result = triage.triage(workspace, config_path)
        self.assertEqual(result['counts']['unsupported_source'], 1)


if __name__ == '__main__':
    unittest.main()
