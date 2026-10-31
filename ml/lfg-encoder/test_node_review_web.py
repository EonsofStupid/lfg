"""Local UI tests use simulated review decisions only."""
import http.client
import json
from pathlib import Path
import secrets
import tempfile
import threading
import unittest

import node_review
import node_review_web


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT.parent.parent / 'crates/lfg-core/config/kernel-core.v1.json'


class NodeReviewWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / 'queue.jsonl'
        self.ledger = self.root / 'ledger.json'
        self.source = self.root / 'source.jsonl'
        self.guide = self.root / 'guide.json'
        text = '<script>test-only request</script>'
        row = {'id': 'fixture', 'text': text, 'family': 'source-family',
               'source': {'project': 'source-project', 'kind': 'test-only'}, 'review_note': 'Verify context.'}
        self.source.write_bytes(node_review.encode(row) + b'\n')
        item = {'schemaVersion': 'lfg.node-review-queue-item.v1', 'status': 'pending_review',
            'requestId': 'fixture', 'text': text, 'inputSha256': node_review.digest(text.encode()),
            'sourceFileSha256': node_review.digest(self.source.read_bytes()),
            'snapshotSha256': node_review.digest(SNAPSHOT.read_bytes()), 'snapshotRevision': 'r1',
            'journalTailSha256': 'b' * 64, 'observationSequence': 1, 'node': 'operation',
            'options': ['chat', 'tool_call'], 'baselineState': 'escalate', 'baselineChoice': None,
            'reviewedChoice': None, 'reviewer': None, 'project': None, 'family': None, 'split': None}
        self.queue.write_bytes(node_review.encode(item) + b'\n')
        node_review.init(self.queue, SNAPSHOT, self.ledger)
        node_review_web.template(SNAPSHOT, self.guide, ['operation'])
        self.token = secrets.token_urlsafe(32)
        config = node_review_web.load_config(node_review_web.DEFAULT_CONFIG)
        self.server = node_review_web.NodeServer(self.queue, self.ledger, SNAPSHOT, self.guide, self.source,
                                                'test-reviewer', 0, config, self.token)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
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
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def ready_guide(self):
        guide = json.loads(self.guide.read_bytes())
        guide.update(status='ready', revision='test-only-ready', notice='Test-only definitions.')
        guide['nodes']['operation']['question'] = 'Which option?'
        for name, description in guide['nodes']['operation']['options'].items():
            description.update(meaning=f'{name} meaning', boundary=f'{name} boundary')
        self.guide.write_bytes(node_review.encode(guide) + b'\n')

    def item(self):
        code, headers, raw = self.request('GET', '/api/item?offset=0')
        self.assertEqual(code, 200)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        return json.loads(raw)

    def decision(self, item, guide_sha):
        return {'itemSha256': item['itemSha256'], 'guideSha256': guide_sha,
                'decision': 'approved', 'confirmation': 'APPROVE', 'attest': True,
                'choice': 'chat', 'project': 'source-project', 'projectGroup': 'related',
                'family': 'source-family', 'split': 'train'}

    def test_auth_and_source_preview_do_not_expose_baseline(self):
        self.assertEqual(self.request('GET', '/api/status', auth=False)[0], 403)
        self.assertEqual(self.request('GET', '/api/status', host='evil.example')[0], 403)
        self.assertEqual(self.request('GET', '/api/status', origin='https://evil.example')[0], 403)
        self.assertEqual(self.request('POST', '/api/decision', body={}, origin='https://evil.example')[0], 403)
        result = self.item()
        item = result['item']
        self.assertEqual(result['guideStatus'], 'draft')
        self.assertEqual(item['source']['status'], 'source_file_and_text_match')
        self.assertEqual(item['source']['source']['project'], 'source-project')
        self.assertEqual(item['source']['family'], 'source-family')
        self.assertNotIn('baselineChoice', item)
        self.assertNotIn('baselineState', item)
        self.assertEqual(self.request('POST', '/api/decision', body=self.decision(item, result['guideSha256']),
                                      origin=self.server.origin)[0], 400)
        self.assertEqual(node_review.load_ledger(self.queue, SNAPSHOT, self.ledger)[1]['records'][0]['decision'], 'pending')

    def test_approval_requires_ready_current_guide_and_records_its_digest(self):
        self.ready_guide()
        result = self.item()
        body = self.decision(result['item'], result['guideSha256'])
        body['confirmation'] = ''
        self.assertEqual(self.request('POST', '/api/decision', body=body, origin=self.server.origin)[0], 400)
        body['confirmation'] = 'APPROVE'
        self.assertEqual(self.request('POST', '/api/decision', body=body)[0], 403)
        self.assertEqual(self.request('POST', '/api/decision', body=body, origin=self.server.origin)[0], 200)
        ledger = node_review.load_ledger(self.queue, SNAPSHOT, self.ledger)[1]
        self.assertEqual(ledger['schemaVersion'], 'lfg.node-review-ledger.v2')
        record = ledger['records'][0]
        self.assertEqual(record['decision'], 'approved')
        self.assertEqual(record['guideSha256'], result['guideSha256'])
        self.assertEqual(self.request('POST', '/api/decision', body=body, origin=self.server.origin)[0], 400)

    def test_changed_guide_and_source_mismatch_never_prefill(self):
        self.ready_guide()
        result = self.item()
        body = self.decision(result['item'], result['guideSha256'])
        guide = json.loads(self.guide.read_bytes())
        guide['revision'] = 'changed'
        self.guide.write_bytes(node_review.encode(guide) + b'\n')
        self.assertEqual(self.request('POST', '/api/decision', body=body, origin=self.server.origin)[0], 400)
        self.source.write_bytes(b'{}\n')
        result = self.item()
        self.assertEqual(result['item']['source']['status'], 'source_digest_mismatch')
        self.assertNotIn('source', result['item']['source'])
        self.assertEqual(node_review.load_ledger(self.queue, SNAPSHOT, self.ledger)[1]['records'][0]['decision'], 'pending')

    def test_prompt_limit_and_invalid_ready_guide_fail_closed(self):
        self.server.config['maxPromptBytes'] = 1
        result = self.item()
        self.assertIsNone(result['item'])
        self.assertIn('display limit', result['blocked'])
        guide = json.loads(self.guide.read_bytes())
        guide['status'] = 'ready'
        self.guide.write_bytes(node_review.encode(guide) + b'\n')
        self.assertEqual(self.request('GET', '/api/status')[0], 400)

    def test_rejection_is_possible_with_a_draft_without_assigning_labels(self):
        result = self.item()
        body = {'itemSha256': result['item']['itemSha256'], 'guideSha256': result['guideSha256'],
                'decision': 'rejected', 'confirmation': 'REJECT', 'attest': True}
        self.assertEqual(self.request('POST', '/api/decision', body=body, origin=self.server.origin)[0], 200)
        record = node_review.load_ledger(self.queue, SNAPSHOT, self.ledger)[1]['records'][0]
        self.assertEqual(record['decision'], 'rejected')
        self.assertNotIn('choice', record)
        self.assertNotIn('split', record)


if __name__ == '__main__':
    unittest.main()
