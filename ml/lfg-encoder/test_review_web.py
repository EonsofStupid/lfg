import http.client
import json
import secrets
import tempfile
import threading
import unittest
from pathlib import Path

import corpus_audit
import review_corpus
import review_web


class ReviewWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        taxonomy = {'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
                    'intents': ['build', 'explain'], 'capabilities': ['write'], 'risks': ['sensitive']}
        (root / 'taxonomy.json').write_text(json.dumps(taxonomy), encoding='utf-8')
        row = {'id': 'one', 'text': '<script>fixture</script>', 'intents': ['build'],
               'capabilities': ['write'], 'risks': [], 'verified': False, 'family': 'fixture',
               'source': {'project': 'fixture'}, 'review_note': 'Check source.'}
        (root / 'source.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
        plan = {'schema_version': corpus_audit.PLAN_SCHEMA, 'taxonomy': 'taxonomy.json', 'review_ledger': None,
                'sources': [{'path': 'source.jsonl', 'project': 'fixture', 'split': 'train'}]}
        (root / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
        self.workspace = root / 'workspace'
        review_corpus.init_workspace(root / 'plan.json', self.workspace)
        self.token = secrets.token_urlsafe(32)
        config = review_web.load_config(review_web.DEFAULT_CONFIG)
        self.server = review_web.ReviewServer(self.workspace, 'fixture-reviewer', 0, config, self.token)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

    def request(self, method, path, body=None, auth=True, origin=None, host=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        headers = {'Host': host or f'127.0.0.1:{self.server.server_port}'}
        if auth:
            headers['Authorization'] = 'Bearer ' + self.token
        if origin is not None:
            headers['Origin'] = origin
        if body is not None:
            headers['Content-Type'] = 'application/json'
            body = json.dumps(body).encode('utf-8')
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            content = response.read()
            return (response.status, dict(response.getheaders()), content)
        finally:
            connection.close()

    def test_private_api_requires_token_and_expected_host_origin(self):
        self.assertEqual(self.request('GET', '/api/status', auth=False)[0], 403)
        valid_token = self.token
        self.token = 'wrong-token'
        self.assertEqual(self.request('GET', '/api/status')[0], 403)
        self.token = valid_token
        self.assertEqual(self.request('GET', '/api/status', host='evil.example')[0], 403)
        self.assertEqual(self.request('GET', '/api/status', origin='https://evil.example')[0], 403)
        self.assertEqual(self.request('POST', '/api/decision', body={}, origin='https://evil.example')[0], 403)
        self.assertEqual(self.request('POST', '/api/decision', body={})[0], 403)
        status, headers, data = self.request('GET', '/api/status')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(json.loads(data)['decisions']['pending'], 1)
        self.assertNotIn(b'fixture</script>', data)

    def test_oversized_decision_is_rejected_without_approval(self):
        self.server.config['max_request_bytes'] = 10
        try:
            status, _, _ = self.request('POST', '/api/decision', body={'long': 'value'}, origin=self.server.origin)
        except (ConnectionAbortedError, ConnectionResetError):
            pass  # The OS may reset a connection with an unread oversized body.
        else:
            self.assertEqual(status, 400)
        self.assertEqual(review_corpus.status(self.workspace)['decisions']['approved'], 0)

    def test_review_requires_confirmation_and_updates_existing_audit(self):
        status, _, data = self.request('GET', '/api/item?offset=0')
        self.assertEqual(status, 200)
        item = json.loads(data)['item']
        decision = {'id': item['id'], 'annotation_sha256': item['annotation_sha256'],
                    'decision': 'approved', 'confirmation': '',
                    'labels': {'intents': ['explain'], 'capabilities': [], 'risks': []}}
        self.assertEqual(self.request('POST', '/api/decision', body=decision, origin=self.server.origin)[0], 400)
        self.assertEqual(review_corpus.status(self.workspace)['decisions']['approved'], 0)
        decision['confirmation'] = 'APPROVE'
        self.assertEqual(self.request('POST', '/api/decision', body=decision, origin=self.server.origin)[0], 200)
        self.assertEqual(review_corpus.status(self.workspace)['counts']['train']['ready_for_reviewed_subset'], 1)
        self.assertEqual(self.request('POST', '/api/decision', body=decision, origin=self.server.origin)[0], 400)

    def test_prompt_display_limit_does_not_return_text(self):
        self.server.config['max_prompt_bytes'] = 1
        status, _, data = self.request('GET', '/api/item?offset=0')
        self.assertEqual(status, 200)
        self.assertIsNone(json.loads(data)['item'])
        self.assertNotIn(b'fixture</script>', data)

    def test_changed_pending_record_is_flagged_and_cannot_be_approved(self):
        source = self.workspace / 'source-0000.jsonl'
        row = json.loads(source.read_text(encoding='utf-8'))
        row['text'] = 'changed fixture'
        source.write_text(json.dumps(row) + '\n', encoding='utf-8')
        status, _, raw = self.request('GET', '/api/item?offset=0')
        item = json.loads(raw)['item']
        self.assertEqual(status, 200)
        self.assertTrue(item['pending_changed'])
        decision = {'id': item['id'], 'annotation_sha256': item['annotation_sha256'],
                    'decision': 'approved', 'confirmation': 'APPROVE', 'labels': item['labels']}
        self.assertEqual(self.request('POST', '/api/decision', body=decision, origin=self.server.origin)[0], 400)
        self.assertEqual(review_corpus.status(self.workspace)['decisions']['approved'], 0)

    def test_static_assets_set_restrictive_headers(self):
        status, headers, data = self.request('GET', '/', auth=False)
        self.assertEqual(status, 200)
        self.assertIn("default-src 'none'", headers['Content-Security-Policy'])
        self.assertIn(b'Corpus review', data)
        self.assertEqual(self.request('GET', '/app.js', auth=False)[0], 200)


if __name__ == '__main__':
    unittest.main()
