import http.client
import hashlib
import json
import secrets
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import corpus_audit
import inventory_checkpoint_prompts
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

    def test_find_pending_record_by_id_and_reject_invalid_query(self):
        status, _, raw = self.request('GET', '/api/item?id=one')
        value = json.loads(raw)
        self.assertEqual((status, value['item']['id'], value['offset']), (200, 'one', 0))
        status, _, raw = self.request('GET', '/api/item?id=missing')
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(raw)['missing_id'])
        self.assertEqual(self.request('GET', '/api/item?id=one&offset=0')[0], 400)

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

    def test_redaction_is_applied_only_after_explicit_approval(self):
        _, _, data = self.request('GET', '/api/item?offset=0')
        item = json.loads(data)['item']
        decision = {'id': item['id'], 'annotation_sha256': item['annotation_sha256'],
                    'decision': 'approved', 'confirmation': 'APPROVE',
                    'labels': {'intents': ['build'], 'capabilities': [], 'risks': []},
                    'redactions': [{'start': 8, 'end': 15, 'replacement': '[removed]'}]}
        self.assertEqual(self.request('POST', '/api/decision', body=decision, origin=self.server.origin)[0], 200)
        row = json.loads((self.workspace / 'source-0000.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(row['text'], '<script>[removed]</script>')
        self.assertEqual(review_corpus.status(self.workspace)['decisions']['approved'], 1)

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

    def test_source_check_validates_original_and_never_approves(self):
        root = Path(self.temp.name)
        text = 'a private request'
        line = (json.dumps({'type': 'response_item', 'payload': {'type': 'message', 'role': 'user',
                 'content': [{'type': 'input_text', 'text': text}],
                 'internal_chat_message_metadata_passthrough': {'content_item_kinds': ['user.text']}}}) + '\n').encode()
        session = root / 'session.jsonl'
        session.write_bytes(line)
        source = {'project': 'fixture', 'kind': 'codex-session-user-text', 'source_file': str(session),
                  'source_line': 1, 'source_line_sha256': hashlib.sha256(line.rstrip(b'\r\n')).hexdigest(),
                  'source_prefix_sha256': hashlib.sha256(line).hexdigest(), 'source_prefix_bytes': len(line),
                  'source_file_sha256': hashlib.sha256(line).hexdigest(),
                  'text_sha256': hashlib.sha256(text.encode()).hexdigest()}
        path = root / 'source.jsonl'
        row = json.loads(path.read_text(encoding='utf-8'))
        row.update(text=text, source=source)
        path.write_text(json.dumps(row) + '\n', encoding='utf-8')
        workspace = root / 'codex-workspace'
        review_corpus.init_workspace(root / 'plan.json', workspace)
        self.server.workspace = workspace
        _, _, data = self.request('GET', '/api/item?offset=0')
        item = json.loads(data)['item']
        body = {'id': item['id'], 'annotation_sha256': item['annotation_sha256']}
        self.assertEqual(self.request('POST', '/api/verify-source', body=body, origin='https://evil.example')[0], 403)
        status, _, data = self.request('POST', '/api/verify-source', body=body, origin=self.server.origin)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)['status'], 'source_prefix_and_original_text_match')
        self.assertNotIn(text.encode(), data)
        self.assertEqual(review_corpus.status(workspace)['decisions']['approved'], 0)
        original_limit = self.server.config['source_verification']['max_line_bytes']
        self.server.config['source_verification']['max_line_bytes'] = 1
        self.assertEqual(self.request('POST', '/api/verify-source', body=body, origin=self.server.origin)[0], 400)
        self.server.config['source_verification']['max_line_bytes'] = original_limit
        with session.open('ab') as output:
            output.write(b'additional later content\n')
        self.assertEqual(self.request('POST', '/api/verify-source', body=body, origin=self.server.origin)[0], 200)
        row['text'] = 'different review copy'
        path.write_text(json.dumps(row) + '\n', encoding='utf-8')
        different = root / 'different-workspace'
        review_corpus.init_workspace(root / 'plan.json', different)
        _, _, ledger = review_corpus.load_workspace(different)
        with self.assertRaisesRegex(ValueError, 'differs from the original'):
            review_web.verify_source(different, 'one', ledger['one']['annotation_sha256'],
                                     self.server.config['source_verification'])
        session.write_bytes(line.replace(b'private', b'changed'))
        self.assertEqual(self.request('POST', '/api/verify-source', body=body, origin=self.server.origin)[0], 400)
        self.assertEqual(review_corpus.status(workspace)['decisions']['approved'], 0)

    def test_source_check_rejects_unsupported_and_stale_records(self):
        _, _, data = self.request('GET', '/api/item?offset=0')
        item = json.loads(data)['item']
        body = {'id': item['id'], 'annotation_sha256': item['annotation_sha256']}
        self.assertEqual(self.request('POST', '/api/verify-source', body=body, origin=self.server.origin)[0], 400)
        body['annotation_sha256'] = '0' * 64
        self.assertEqual(self.request('POST', '/api/verify-source', body=body, origin=self.server.origin)[0], 400)

    def test_git_checkpoint_check_uses_configured_repo_and_never_approves(self):
        root = Path(self.temp.name)
        repo = root / 'git-repo'
        repo.mkdir()
        def git(*args):
            return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True).stdout
        git('init')
        message = root / 'message.txt'
        text = 'Example request for a test fixture'
        message.write_text('checkpoint\n\nUser prompt:\n\n' + text +
                           '\n\nReplit-Commit-Author: Assistant\nReplit-Commit-Session-Id: test-session\n', encoding='utf-8')
        git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
            'commit', '--allow-empty', '-F', str(message))
        revision = git('rev-parse', 'HEAD').decode().strip()
        history = inventory_checkpoint_prompts.read_history(repo, revision, 100000)
        _, records = inventory_checkpoint_prompts.parse_history(history, 'fixture', revision)
        candidate = records[0]
        source = {**candidate['source'], 'text_sha256': candidate['text_sha256'],
                  'history_stream_sha256': hashlib.sha256(history).hexdigest()}
        row = {'id': 'checkpoint-' + candidate['id'], 'text': text,
               'family': 'checkpoint-session-' + candidate['session_sha256'],
               'intents': [], 'capabilities': [], 'risks': [], 'labels_pending': True,
               'verified': False, 'source': source}
        path = root / 'source.jsonl'
        path.write_text(json.dumps(row) + '\n', encoding='utf-8')
        workspace = root / 'checkpoint-workspace'
        review_corpus.init_workspace(root / 'plan.json', workspace)
        self.server.workspace = workspace
        _, _, data = self.request('GET', '/api/item?offset=0')
        item = json.loads(data)['item']
        body = {'id': item['id'], 'annotation_sha256': item['annotation_sha256']}
        self.assertEqual(self.request('POST', '/api/verify-source', body=body, origin=self.server.origin)[0], 400)
        self.server.config['checkpoint_verification'] = {'repo': str(repo), 'max_history_bytes': 100000}
        status, _, data = self.request('POST', '/api/verify-source', body=body, origin=self.server.origin)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)['status'], 'checkpoint_history_and_original_text_match')
        self.assertNotIn(text.encode(), data)
        self.assertEqual(self.request('POST', '/api/source-context', body=body, origin=self.server.origin)[0], 400)
        with patch.object(inventory_checkpoint_prompts, 'read_history', return_value=history + b'changed'):
            self.assertEqual(self.request('POST', '/api/verify-source', body=body, origin=self.server.origin)[0], 400)
        self.server.config['checkpoint_verification']['max_history_bytes'] = 1
        self.assertEqual(self.request('POST', '/api/verify-source', body=body, origin=self.server.origin)[0], 400)
        self.assertEqual(review_corpus.status(workspace)['decisions']['approved'], 0)

    def test_checkpoint_configuration_requires_absolute_repo_and_positive_limit(self):
        config = review_web.load_config(review_web.DEFAULT_CONFIG)
        config_path = Path(self.temp.name) / 'config.json'
        for limits in ({'repo': 'relative', 'max_history_bytes': 100},
                       {'repo': str(Path(self.temp.name).resolve()), 'max_history_bytes': 0},
                       {'repo': str(Path(self.temp.name).resolve()), 'max_history_bytes': True}):
            config_path.write_text(json.dumps({**config, 'checkpoint_verification': limits}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checkpoint verification'):
                review_web.load_config(config_path)
        limits = {'repo': str(Path(self.temp.name).resolve()), 'max_history_bytes': 100}
        config_path.write_text(json.dumps({**config, 'checkpoint_verification': limits}), encoding='utf-8')
        self.assertEqual(review_web.load_config(config_path)['checkpoint_verification'], limits)

    def test_source_context_is_local_bounded_and_does_not_record_review(self):
        root = Path(self.temp.name)
        def message(role, text, kind='user.text'):
            return {'type': 'response_item', 'payload': {'type': 'message', 'role': role,
                    'content': [{'type': 'input_text', 'text': text}],
                    'internal_chat_message_metadata_passthrough': {'content_item_kinds': [kind]}}}
        rows = [message('user', 'prior private text'), message('assistant', 'assistant text'),
                message('user', 'selected text'), message('user', 'later text')]
        lines = [(json.dumps(row) + '\n').encode() for row in rows]
        prefix = b''.join(lines[:3])
        session = root / 'context-session.jsonl'; session.write_bytes(b''.join(lines))
        source = {'project': 'fixture', 'kind': 'codex-session-user-text', 'source_file': str(session),
                  'source_line': 3, 'source_line_sha256': hashlib.sha256(lines[2].rstrip(b'\r\n')).hexdigest(),
                  'source_prefix_sha256': hashlib.sha256(prefix).hexdigest(), 'source_prefix_bytes': len(prefix),
                  'source_file_sha256': hashlib.sha256(session.read_bytes()).hexdigest(),
                  'text_sha256': hashlib.sha256(b'selected text').hexdigest()}
        source_path = root / 'source.jsonl'
        row = json.loads(source_path.read_text(encoding='utf-8'))
        row.update(text='selected text', source=source)
        source_path.write_text(json.dumps(row) + '\n', encoding='utf-8')
        workspace = root / 'context-workspace'
        review_corpus.init_workspace(root / 'plan.json', workspace)
        self.server.workspace = workspace
        _, _, data = self.request('GET', '/api/item?offset=0')
        item = json.loads(data)['item']
        body = {'id': item['id'], 'annotation_sha256': item['annotation_sha256']}
        self.assertEqual(self.request('POST', '/api/source-context', body=body, origin='https://evil.example')[0], 403)
        status, headers, data = self.request('POST', '/api/source-context', body=body, origin=self.server.origin)
        self.assertEqual(status, 200)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        value = json.loads(data)
        self.assertEqual([item['text'] for item in value['messages']], ['prior private text'])
        self.assertNotIn(b'assistant text', data)
        self.assertNotIn(b'later text', data)
        self.assertEqual(review_corpus.status(workspace)['decisions']['approved'], 0)
        session.write_bytes(b''.join(lines).replace(b'prior private text', b'changed private text'))
        self.assertEqual(self.request('POST', '/api/source-context', body=body, origin=self.server.origin)[0], 400)

    def test_guided_review_binds_approval_to_displayed_guide_and_rejects_changes(self):
        guide = {'schema_version': 'lfg.review-label-guide.v1', 'revision': 'fixture', 'notice': 'Review carefully.',
                 'labels': {'intents': {'build': {'apply': 'Build.', 'boundary': 'Not explanation.'},
                                        'explain': {'apply': 'Explain.', 'boundary': 'Not building.'}},
                            'capabilities': {'write': {'apply': 'Write.', 'boundary': 'Not reading.'}},
                            'risks': {'sensitive': {'apply': 'Sensitive.', 'boundary': 'Not generic.'}}}}
        path = Path(self.temp.name) / 'guide.json'
        path.write_text(json.dumps(guide), encoding='utf-8')
        self.stop_server()
        self.server = review_web.ReviewServer(self.workspace, 'fixture-reviewer', 0,
                                               review_web.load_config(review_web.DEFAULT_CONFIG), self.token, path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        status, _, data = self.request('GET', '/api/item?offset=0')
        self.assertEqual(status, 200)
        response = json.loads(data)
        self.assertEqual(response['guide']['labels']['intents']['build']['boundary'], 'Not explanation.')
        item = response['item']
        decision = {'id': item['id'], 'annotation_sha256': item['annotation_sha256'],
                    'decision': 'approved', 'confirmation': 'APPROVE',
                    'labels': {'intents': ['build'], 'capabilities': [], 'risks': []}}
        self.assertEqual(self.request('POST', '/api/decision', body=decision, origin=self.server.origin)[0], 400)
        decision['guide_sha256'] = response['guide_sha256']
        guide['notice'] = 'Changed after display.'
        path.write_text(json.dumps(guide), encoding='utf-8')
        self.assertEqual(self.request('POST', '/api/decision', body=decision, origin=self.server.origin)[0], 400)
        self.assertEqual(review_corpus.status(self.workspace)['decisions']['approved'], 0)
        guide['notice'] = 'Review carefully.'
        path.write_text(json.dumps(guide), encoding='utf-8')
        self.assertEqual(self.request('POST', '/api/decision', body=decision, origin=self.server.origin)[0], 200)
        row = json.loads((self.workspace / 'source-0000.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(row['review_guide_sha256'], response['guide_sha256'])
        self.assertEqual(review_corpus.status(self.workspace)['counts']['train']['ready_for_reviewed_subset'], 1)
        output = Path(self.temp.name) / 'guided-export'
        review_corpus.export_approved(self.workspace, output)
        exported = json.loads((output / 'source-0000.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(exported['review_guide_sha256'], response['guide_sha256'])
        self.assertEqual(corpus_audit.audit(output / 'plan.json')[0]['counts']['train']['ready_for_reviewed_subset'], 1)

    def test_guide_must_cover_the_workspace_taxonomy(self):
        taxonomy, _ = review_web.evaluate.load_taxonomy(self.workspace / 'taxonomy.json')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            review_web.load_guide(review_web.DEFAULT_GUIDE, taxonomy)


if __name__ == '__main__':
    unittest.main()
