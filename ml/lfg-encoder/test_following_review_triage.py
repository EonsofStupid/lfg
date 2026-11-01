import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import corpus_audit
import following_review_triage as triage
import review_corpus


def message(text):
    return (json.dumps({'type': 'response_item', 'payload': {'type': 'message', 'role': 'user',
        'content': [{'type': 'input_text', 'text': text}],
        'internal_chat_message_metadata_passthrough': {'content_item_kinds': ['user.text']}}}) + '\n').encode()


class FollowingReviewTriageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        lines = [message('first request'), message('wrong chat'), message('other request')]
        self.session = root / 'session.jsonl'
        self.session.write_bytes(b''.join(lines))
        def record(index, name):
            prefix = b''.join(lines[:index + 1])
            origin = {'kind': 'codex-session-user-text', 'project': 'fixture', 'source_file': str(self.session),
                      'source_line': index + 1, 'source_line_sha256': hashlib.sha256(lines[index].rstrip(b'\r\n')).hexdigest(),
                      'source_prefix_bytes': len(prefix), 'source_prefix_sha256': hashlib.sha256(prefix).hexdigest(),
                      'source_file_sha256': hashlib.sha256(b''.join(lines)).hexdigest(),
                      'text_sha256': hashlib.sha256(name.encode()).hexdigest()}
            return {'id': name, 'text': name, 'family': name, 'verified': False, 'intents': [],
                    'capabilities': [], 'risks': [], 'labels_pending': True, 'source': origin}
        rows = [record(0, 'first request'), record(2, 'other request')]
        (root / 'taxonomy.json').write_text(json.dumps({'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
                                                        'intents': ['build'], 'capabilities': ['write'], 'risks': ['sensitive']}))
        (root / 'source.jsonl').write_bytes(review_corpus.encoded_rows(rows))
        plan = {'schema_version': corpus_audit.PLAN_SCHEMA, 'taxonomy': 'taxonomy.json', 'review_ledger': None,
                'sources': [{'path': 'source.jsonl', 'project': 'fixture', 'split': 'train'}]}
        (root / 'plan.json').write_text(json.dumps(plan))
        self.workspace = root / 'workspace'
        review_corpus.init_workspace(root / 'plan.json', self.workspace)
        self.config = Path(triage.__file__).with_name('following-review-triage.example.json')

    def test_real_source_matching_reports_only_metadata_and_leaves_reviews_pending(self):
        before = (self.workspace / 'reviews.jsonl').read_bytes()
        result = triage.triage(self.workspace, self.config)
        self.assertEqual(result['counts']['checked'], 2)
        self.assertEqual(result['flagged'], 1)
        self.assertEqual(result['items'][0]['id'], 'first request')
        self.assertEqual(result['items'][0]['hits'][0]['marker_id'], 'wrong-chat')
        self.assertEqual(result['items'][0]['hits'][0]['source_line'], 2)
        self.assertNotIn('wrong chat', json.dumps(result))
        self.assertFalse(result['review_decisions_made'])
        self.assertEqual((self.workspace / 'reviews.jsonl').read_bytes(), before)

    def test_changed_original_prefix_is_not_treated_as_checked(self):
        self.session.write_bytes(self.session.read_bytes().replace(b'first request', b'changed entry'))
        result = triage.triage(self.workspace, self.config)
        self.assertEqual(result['counts']['source_unavailable'], 2)
        self.assertEqual(result['flagged'], 0)
        self.assertNotIn('checked', result['counts'])

    def test_rejects_test_split_before_opening_sources(self):
        path = self.workspace / 'plan.json'
        plan = json.loads(path.read_bytes())
        plan['sources'][0]['split'] = 'test'
        path.write_text(json.dumps(plan))
        with patch.object(review_corpus, 'load_workspace', side_effect=AssertionError('sources opened')):
            with self.assertRaisesRegex(ValueError, 'only training sources'):
                triage.triage(self.workspace, self.config)

    def test_scan_limit_does_not_claim_complete_context(self):
        config = json.loads(self.config.read_bytes())
        config['max_messages'] = 1
        config_path = self.workspace.parent / 'custom-config.json'
        config_path.write_text(json.dumps(config))
        result = triage.triage(self.workspace, config_path)
        self.assertEqual(result['items'][0]['context_end'], 'message_limit')
        config['max_messages'] = 3
        config['max_scan_lines'] = 1
        config_path.write_text(json.dumps(config))
        result = triage.triage(self.workspace, config_path)
        self.assertEqual(result['items'][0]['context_end'], 'scan_limit')

    def test_append_during_scan_rechecks_observed_bytes_and_marks_eof_incomplete(self):
        actual = triage.signature(self.session)
        with patch.object(triage, 'signature', side_effect=[actual, (actual[0] + 1, actual[1], actual[2])]):
            result = triage.triage(self.workspace, self.config)
        self.assertEqual(result['counts']['checked'], 2)
        self.assertEqual(result['counts']['context_source_changed_after_eof'], 2)
        self.assertEqual(result['flagged'], 1)

    def test_changed_scanned_prefix_during_scan_is_not_accepted(self):
        actual = triage.signature(self.session)
        calls = 0
        def changed_on_second_call(_path):
            nonlocal calls
            calls += 1
            if calls == 1:
                return actual
            self.session.write_bytes(self.session.read_bytes().replace(b'wrong chat', b'wrong home'))
            return (actual[0], actual[1] + 1, actual[2])
        with patch.object(triage, 'signature', side_effect=changed_on_second_call):
            result = triage.triage(self.workspace, self.config)
        self.assertEqual(result['counts']['source_unavailable'], 2)
        self.assertEqual(result['flagged'], 0)


if __name__ == '__main__':
    unittest.main()
