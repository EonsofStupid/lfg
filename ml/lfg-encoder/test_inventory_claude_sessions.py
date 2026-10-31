import json
import tempfile
import unittest
from pathlib import Path

import inspect_claude_inventory
import inventory_claude_sessions as inventory
import corpus_audit
import prepare_codex_review
import review_corpus
import review_web
import train


class ClaudeInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.sessions = self.root / 'sessions'
        self.sessions.mkdir()
        self.project = self.root / 'project'
        self.project.mkdir()
        self.path = self.sessions / 'session.jsonl'

    def row(self, text='test user request'):
        return {'type': 'user', 'turnOrigin': 'human', 'isSidechain': False,
                'cwd': str(self.project), 'message': {'role': 'user', 'content': [{'type': 'text', 'text': text}]}}

    def write(self, rows):
        self.path.write_bytes(b''.join((json.dumps(row) + '\n').encode() for row in rows))

    def inspect(self):
        return inventory.inventory({'test': (self.sessions, str(self.project))}, 100000, 10000)

    def test_only_explicit_human_main_session_single_text_is_inventoried(self):
        accepted = self.row()
        rejected = [dict(self.row(), turnOrigin='task_notification'),
                    dict(self.row(), isSidechain=True), dict(self.row(), isMeta=True),
                    dict(self.row(), isCompactSummary=True), dict(self.row(), sourceToolAssistantUUID='tool'),
                    dict(self.row(), toolUseResult={}), dict(self.row(), promptSource='system'),
                    dict(self.row(), cwd=str(self.root)),
                    dict(self.row(), message={'role': 'user', 'content': [{'type': 'text', 'text': 'x'}, {'type': 'image'}]}),
                    dict(self.row(), message={'role': 'assistant', 'content': 'x'})]
        self.write([accepted, *rejected])
        summary, rows = self.inspect()
        self.assertEqual(summary['candidate_messages'], 1)
        self.assertEqual(inspect_claude_inventory.verify(rows[0], 100000, 10000), 'test user request')
        self.assertNotIn('test user request', json.dumps(rows))
        self.assertFalse(rows[0]['verified'])
        self.assertIsNone(rows[0]['split'])

    def test_original_prefix_is_bound_but_later_append_is_allowed(self):
        self.write([self.row('original')])
        _, rows = self.inspect()
        with self.path.open('ab') as stream:
            stream.write((json.dumps({'type': 'system'}) + '\n').encode())
        self.assertEqual(inspect_claude_inventory.verify(rows[0], 100000, 10000), 'original')
        self.write([self.row('modified')])
        with self.assertRaisesRegex(ValueError, 'prefix|unavailable'):
            inspect_claude_inventory.verify(rows[0], 100000, 10000)

    def test_duplicate_source_and_byte_limits_fail_closed(self):
        self.write([self.row()])
        with self.assertRaisesRegex(ValueError, 'more than once'):
            inventory.inventory({'one': (self.sessions, str(self.project)),
                                 'two': (self.sessions, str(self.project))}, 100000, 10000)
        with self.assertRaisesRegex(ValueError, 'byte limit'):
            inventory.inventory({'one': (self.sessions, str(self.project))}, 1, 10000)
        with self.assertRaisesRegex(ValueError, 'line exceeds'):
            inventory.inventory({'one': (self.sessions, str(self.project))}, 100000, 8)

    def test_inventory_prepares_pending_workspace_and_verifies_original_source(self):
        self.write([self.row('first request'), self.row('second request'), self.row('first request')])
        _, rows = self.inspect()
        inventory_path = self.root / 'inventory.jsonl'
        inventory_path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        assignments = self.root / 'assignments.json'
        assignments.write_text(json.dumps({'schema_version': 'lfg.claude-review-assignments.v1',
            'projects': {'test': {'project_group': 'related', 'split': 'train'}}}), encoding='utf-8')
        taxonomy = self.root / 'taxonomy.json'
        taxonomy.write_text(json.dumps({'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
            'intents': ['build'], 'capabilities': ['write'], 'risks': ['sensitive']}), encoding='utf-8')
        output = self.root / 'prepared'
        summary = prepare_codex_review.prepare(inventory_path, assignments, taxonomy, output, 100000, 10000)
        self.assertEqual(summary['pending_review'], 2)
        workspace = output / 'workspace'
        self.assertEqual(review_corpus.status(workspace)['decisions'], {'pending': 2, 'approved': 0, 'rejected': 0})
        with self.assertRaisesRegex(ValueError, 'pending labels'):
            train.load_dataset(workspace / 'source-0000.jsonl')
        plan, _, ledger = review_corpus.load_workspace(workspace)
        identifier = next(iter(ledger))
        _, source_rows, index, source = review_corpus.find_row(workspace, plan, identifier)
        digest = corpus_audit.annotation_digest(source_rows[index], source['project'], source['split'],
                                                review_corpus.evaluate.load_taxonomy(workspace / 'taxonomy.json')[1])
        limits = {'max_session_bytes': 100000, 'max_line_bytes': 10000}
        self.assertEqual(review_web.verify_source(workspace, identifier, digest, limits),
                         'source_prefix_and_original_text_match')
        self.write([self.row('changed')])
        with self.assertRaises(ValueError):
            review_web.verify_source(workspace, identifier, digest, limits)


if __name__ == '__main__':
    unittest.main()
