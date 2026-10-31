import json
import http.client
import secrets
import tempfile
import threading
import unittest
from pathlib import Path

import corpus_audit
import inventory_codex_sessions as inventory
import prepare_codex_review as prepare
import review_corpus as review
import review_web
import train


class PrepareSessionReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        self.sessions = self.root / 'sessions'
        self.sessions.mkdir()
        self.session = self.sessions / 'source.jsonl'
        rows = [{'type': 'session_meta', 'payload': {'cwd': str(self.project)}}]
        for text in ('fixture A', 'fixture B', 'fixture A'):
            rows.append({'type': 'response_item', 'payload': {'type': 'message', 'role': 'user',
                'internal_chat_message_metadata_passthrough': {'content_item_kinds': ['user.text']},
                'content': [{'type': 'input_text', 'text': text}]}})
        self.session.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        _, records = inventory.inventory(self.sessions, {'project': inventory.normalize(str(self.project))}, 100000)
        self.inventory = self.root / 'inventory.jsonl'
        self.inventory.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
        self.assignments = self.root / 'assignments.json'
        self.assignments.write_text(json.dumps({'schema_version': 'lfg.codex-review-assignments.v1',
            'projects': {'project': {'project_group': 'related', 'split': 'train'}}}), encoding='utf-8')
        self.taxonomy = self.root / 'taxonomy.json'
        self.taxonomy.write_text(json.dumps({'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
            'intents': ['build'], 'capabilities': ['write'], 'risks': ['sensitive']}), encoding='utf-8')

    def run_prepare(self, name='prepared'):
        output = self.root / name
        result = prepare.prepare(self.inventory, self.assignments, self.taxonomy, output, 100000, 100000)
        return output, result

    def test_unlabeled_material_is_pending_and_cannot_be_trained_or_approved_without_labels(self):
        output, result = self.run_prepare()
        self.assertEqual(result['selected_source_messages'], 3)
        self.assertEqual(result['normalized_duplicates_excluded'], 1)
        self.assertEqual(result['pending_review'], 2)
        workspace = output / 'workspace'
        self.assertEqual(review.status(workspace)['decisions'], {'pending': 2, 'approved': 0, 'rejected': 0})
        with self.assertRaisesRegex(ValueError, 'pending labels'):
            train.load_dataset(workspace / 'source-0000.jsonl')
        plan, _, ledger = review.load_workspace(workspace)
        identifier = next(iter(ledger))
        _, rows, index, source = review.find_row(workspace, plan, identifier)
        row = rows[index]
        self.assertEqual(row['intents'], [])
        digest = corpus_audit.annotation_digest(row, source['project'], source['split'],
                                                review.evaluate.load_taxonomy(workspace / 'taxonomy.json')[1])
        with self.assertRaisesRegex(ValueError, 'explicitly supplied labels'):
            review.record_decision(workspace, identifier, 'reviewer', 'approved', digest)
        with self.assertRaisesRegex(ValueError, 'at least one intent'):
            review.record_decision(workspace, identifier, 'reviewer', 'approved', digest,
                                   {'intents': [], 'capabilities': [], 'risks': []})
        review.record_decision(workspace, identifier, 'reviewer', 'approved', digest,
                               {'intents': ['build'], 'capabilities': ['write'], 'risks': []})
        exported = self.root / 'exported'
        self.assertEqual(review.export_approved(workspace, exported)['exported'], 1)
        approved = train.load_dataset(exported / 'source-0000.jsonl')[0][0]
        self.assertNotIn('labels_pending', approved)
        self.assertTrue(approved['verified'])

    def test_changed_source_refuses_to_create_workspace(self):
        self.session.write_text(self.session.read_text(encoding='utf-8').replace('fixture A', 'tampered!', 1), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'source prefix'):
            self.run_prepare()
        self.assertFalse((self.root / 'prepared').exists())

    def test_browser_exposes_unlabeled_state_and_refuses_empty_intent(self):
        output, _ = self.run_prepare()
        workspace = output / 'workspace'
        token = secrets.token_urlsafe(32)
        server = review_web.ReviewServer(workspace, 'reviewer', 0,
                                         review_web.load_config(review_web.DEFAULT_CONFIG), token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, timeout=5)
        self.addCleanup(server.shutdown)
        conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        headers = {'Authorization': 'Bearer ' + token}
        conn.request('GET', '/api/item?offset=0', headers=headers)
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        item = json.loads(response.read())['item']
        self.assertTrue(item['labels_pending'])
        self.assertEqual(item['labels']['intents'], [])
        decision = {'id': item['id'], 'annotation_sha256': item['annotation_sha256'],
                    'decision': 'approved', 'confirmation': 'APPROVE', 'labels': item['labels']}
        conn.request('POST', '/api/decision', body=json.dumps(decision),
                     headers={**headers, 'Origin': server.origin, 'Content-Type': 'application/json'})
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        response.read()
        conn.close()
        self.assertEqual(review.status(workspace)['decisions']['approved'], 0)

    def test_related_project_group_cannot_be_split(self):
        assignments = json.loads(self.assignments.read_text(encoding='utf-8'))
        assignments['projects']['other'] = {'project_group': 'related', 'split': 'test'}
        self.assignments.write_text(json.dumps(assignments), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'project group crosses'):
            self.run_prepare()
        self.assertFalse((self.root / 'prepared').exists())

    def test_duplicate_text_cannot_cross_splits(self):
        other = self.root / 'other'
        other.mkdir()
        second = self.sessions / 'other.jsonl'
        rows = [json.loads(line) for line in self.session.read_text(encoding='utf-8').splitlines()]
        rows[0]['payload']['cwd'] = str(other)
        second.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        _, records = inventory.inventory(self.sessions, {'project': inventory.normalize(str(self.project)),
                                                           'other': inventory.normalize(str(other))}, 100000)
        self.inventory.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
        assignments = json.loads(self.assignments.read_text(encoding='utf-8'))
        assignments['projects']['other'] = {'project_group': 'other-group', 'split': 'test'}
        self.assignments.write_text(json.dumps(assignments), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'duplicate crosses'):
            self.run_prepare()
        self.assertFalse((self.root / 'prepared').exists())


if __name__ == '__main__':
    unittest.main()
