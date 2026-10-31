import json
import http.client
import secrets
import tempfile
import threading
import unittest
from pathlib import Path

import inspect_claude_inventory
import inspect_review_context
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

    def test_ancestry_limit_is_configurable_and_must_be_positive(self):
        config = review_web.load_config(review_web.DEFAULT_CONFIG)
        path = self.root / 'review-config.json'
        for value in (0, True, -1):
            config['source_context']['max_ancestry_records'] = value
            path.write_text(json.dumps(config), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source context'):
                review_web.load_config(path)
        config['source_context']['max_ancestry_records'] = 17
        path.write_text(json.dumps(config), encoding='utf-8')
        self.assertEqual(review_web.load_config(path)['source_context']['max_ancestry_records'], 17)

    def test_prior_context_is_bounded_filtered_and_prefix_verified(self):
        def linked(text, uid, parent, **extra):
            return {**self.row(text), 'uuid': uid, 'parentUuid': parent, 'sessionId': 'session', **extra}
        self.write([linked('first', 'u1', None), linked('first', 'u1', None),
                    linked('tool result', 'tool', 'u1', sourceToolAssistantUUID='tool'),
                    linked('other branch', 'branch', 'u1'), linked('second', 'u2', 'tool'),
                    linked('selected', 'u3', 'u2'), linked('later', 'u4', 'u3')])
        _, records = self.inspect()
        record = next(item for item in records if item['source_line'] == 6)
        text, context, truncated = inspect_claude_inventory.prior_user_context(record, 100000, 10000, 1, 1000, 100)
        self.assertEqual(text, 'selected')
        self.assertEqual([item['text'] for item in context], ['second'])
        self.assertTrue(truncated)
        _, context, truncated = inspect_claude_inventory.prior_user_context(record, 100000, 10000, 5, 1000, 100)
        self.assertEqual([item['text'] for item in context], ['first', 'second'])
        self.assertFalse(truncated)
        _, context, truncated = inspect_claude_inventory.prior_user_context(record, 100000, 10000, 5, 3, 100)
        self.assertEqual(context, [])
        self.assertTrue(truncated)
        self.write([self.row('tampered'), self.row('selected')])
        with self.assertRaises(ValueError):
            inspect_claude_inventory.prior_user_context(record, 100000, 10000, 5, 1000, 100)

    def test_missing_ambiguous_cross_session_and_cyclic_ancestry_fails_closed(self):
        def linked(uid, parent, session='session'):
            return {**self.row(uid), 'uuid': uid, 'parentUuid': parent, 'sessionId': session}
        cases = [
            ([linked('selected', 'missing')], 'missing or ambiguous'),
            ([linked('ancestor', None), linked('ancestor', 'other'), linked('selected', 'ancestor')], 'missing or ambiguous'),
            ([linked('ancestor', None, 'other'), linked('selected', 'ancestor')], 'crosses session'),
            ([linked('ancestor', 'ancestor'), linked('selected', 'ancestor')], 'cyclic'),
        ]
        for rows, error in cases:
            with self.subTest(error=error):
                self.write(rows)
                _, records = self.inspect()
                with self.assertRaisesRegex(ValueError, error):
                    inspect_claude_inventory.prior_user_context(records[-1], 100000, 10000, 5, 1000, 100)
        self.write([linked('first', None), linked('middle', 'first'), linked('selected', 'middle')])
        _, records = self.inspect()
        with self.assertRaisesRegex(ValueError, 'ancestry record limit'):
            inspect_claude_inventory.prior_user_context(records[-1], 100000, 10000, 5, 1000, 1)

    def test_missing_parent_field_is_not_treated_as_an_explicit_root(self):
        root = {**self.row('root'), 'uuid': 'root', 'sessionId': 'session'}
        selected = {**self.row('selected'), 'uuid': 'selected', 'parentUuid': 'root', 'sessionId': 'session'}
        self.write([root, selected])
        _, records = self.inspect()
        with self.assertRaisesRegex(ValueError, 'invalid or cyclic'):
            inspect_claude_inventory.prior_user_context(records[-1], 100000, 10000, 5, 1000, 100)
        selected.pop('parentUuid')
        self.write([selected])
        _, records = self.inspect()
        with self.assertRaisesRegex(ValueError, 'invalid or cyclic'):
            inspect_claude_inventory.prior_user_context(records[-1], 100000, 10000, 5, 1000, 100)

    def test_inventory_prepares_pending_workspace_and_verifies_original_source(self):
        self.write([{**self.row('first request'), 'uuid': 'u1', 'parentUuid': None, 'sessionId': 'session'},
                    {**self.row('second request'), 'uuid': 'u2', 'parentUuid': 'u1', 'sessionId': 'session'},
                    {**self.row('first request'), 'uuid': 'u3', 'parentUuid': 'u2', 'sessionId': 'session'}])
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
        second = source_rows[1]
        second_digest = corpus_audit.annotation_digest(second, source['project'], source['split'],
                                                       review_corpus.evaluate.load_taxonomy(workspace / 'taxonomy.json')[1])
        context = review_web.verify_source(workspace, second['id'], second_digest, limits,
                                           {'max_messages': 3, 'max_context_bytes': 1000, 'max_ancestry_records': 100})
        self.assertEqual([item['text'] for item in context['messages']], ['first request'])
        self.assertEqual(context['context_scope'], 'same_session_source_parent_chain')
        self.assertEqual(review_corpus.status(workspace)['decisions'], {'pending': 2, 'approved': 0, 'rejected': 0})
        token = secrets.token_urlsafe(32)
        config = review_web.load_config(review_web.DEFAULT_CONFIG)
        config['source_verification'] = limits
        config_path = self.root / 'review-config.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        with train.run_lock(workspace / 'review'):
            summary = inspect_review_context.inspect(workspace, second['id'], config_path)
            self.assertNotIn('messages', summary)
            self.assertEqual(summary['prior_messages'], 1)
            detailed = inspect_review_context.inspect(workspace, second['id'], config_path, show_context=True)
            self.assertEqual([item['text'] for item in detailed['messages']], ['first request'])
        self.assertEqual(review_corpus.status(workspace)['decisions'], {'pending': 2, 'approved': 0, 'rejected': 0})
        server = review_web.ReviewServer(workspace, 'fixture-reviewer', 0, config, token)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
            body = json.dumps({'id': second['id'], 'annotation_sha256': second_digest})
            connection.request('POST', '/api/source-context', body=body,
                               headers={'Authorization': 'Bearer ' + token, 'Origin': server.origin,
                                        'Content-Type': 'application/json'})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader('Cache-Control'), 'no-store')
            payload = json.loads(response.read())
            self.assertEqual([item['text'] for item in payload['messages']], ['first request'])
            self.assertEqual(payload['context_scope'], 'same_session_source_parent_chain')
            connection.close()
        finally:
            server.shutdown()
            worker.join()
            server.server_close()
        self.write([self.row('changed')])
        with self.assertRaises(ValueError):
            review_web.verify_source(workspace, identifier, digest, limits)


if __name__ == '__main__':
    unittest.main()
