import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name('inventory_codex_sessions.py')
spec = importlib.util.spec_from_file_location('inventory_codex_sessions', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
inspect_spec = importlib.util.spec_from_file_location('inspect_codex_inventory', SCRIPT.with_name('inspect_codex_inventory.py'))
inspect_module = importlib.util.module_from_spec(inspect_spec)
inspect_spec.loader.exec_module(inspect_module)


class SessionInventoryTests(unittest.TestCase):
    def test_only_explicit_user_text_in_selected_project_is_inventoried(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            sessions = base / 'sessions'
            sessions.mkdir()
            project = base / 'project'
            project.mkdir()
            session = sessions / 'session.jsonl'
            records = [
                {'type': 'session_meta', 'payload': {'cwd': str(project)}},
                self.message('a private request', ['user.text']),
                self.message('a private request', ['user.text']),
                self.message('<codex_internal_context/>', ['goal.internal_context']),
                self.message('missing source kind', None),
                self.message('mixed source kinds', ['user.text', 'agents_md.instructions']),
            ]
            session.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
            outside = sessions / 'outside.jsonl'
            outside.write_text(json.dumps({'type': 'session_meta', 'payload': {'cwd': str(base / 'other')}}) + '\nnot json\n', encoding='utf-8')
            summary, rows = module.inventory(sessions, {'project': module.normalize(str(project))}, 1024 * 1024)
            self.assertEqual(summary['candidate_messages'], 2)
            self.assertEqual(summary['exact_repetitions'], 1)
            self.assertEqual(summary['file_status']['outside_scope'], 1)
            self.assertEqual(rows[0]['source_line'], 2)
            self.assertEqual(rows[0]['source_file_sha256'], hashlib.sha256(session.read_bytes()).hexdigest())
            self.assertEqual(rows[0]['text_sha256'], hashlib.sha256(b'a private request').hexdigest())
            self.assertNotIn('a private request', json.dumps(rows))
            self.assertFalse(rows[0]['verified'])
            self.assertIsNone(rows[0]['labels'])
            self.assertIsNone(rows[0]['split'])
            self.assertEqual(inspect_module.verify(rows[0], 1024 * 1024, 1024 * 1024), 'a private request')
            inventory = base / 'inventory.jsonl'
            inventory.write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
            result = subprocess.run([sys.executable, str(SCRIPT.with_name('inspect_codex_inventory.py')),
                                     '--inventory', str(inventory), '--record', '1'],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)['status'], 'verified_source_prefix_match')
            self.assertNotIn('a private request', result.stdout)
            with session.open('a', encoding='utf-8') as output:
                output.write(json.dumps({'type': 'event_msg', 'payload': {}}) + '\n')
            self.assertEqual(inspect_module.verify(rows[0], 1024 * 1024, 1024 * 1024), 'a private request')
            session.write_text(session.read_text(encoding='utf-8').replace('a private request', 'a changed request', 1), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'no longer matches'):
                inspect_module.verify(rows[0], 1024 * 1024, 1024 * 1024)

    def test_cli_refuses_to_overwrite_and_rejects_overlapping_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            sessions = base / 'sessions'
            sessions.mkdir()
            project = base / 'project'
            project.mkdir()
            output = base / 'inventory.jsonl'
            output.write_text('keep', encoding='utf-8')
            common = [sys.executable, str(SCRIPT), '--sessions-root', str(sessions),
                      '--project-root', f'one={project}']
            result = subprocess.run(common + ['--output', str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(encoding='utf-8'), 'keep')
            result = subprocess.run(common + ['--project-root', f'two={project / "sub"}'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('overlap', result.stderr)

    def test_limits_and_corruption_discard_the_entire_session(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            project = base / 'project'
            project.mkdir()
            session = base / 'session.jsonl'
            session.write_text(json.dumps({'type': 'session_meta', 'payload': {'cwd': str(project)}}) + '\n'
                               + json.dumps(self.message('a request', ['user.text'])) + '\n'
                               + 'invalid json\n', encoding='utf-8')
            roots = {'project': module.normalize(str(project))}
            state, rows = module.scan_file(session, roots, 1024 * 1024, 1024 * 1024)
            self.assertEqual((state, rows), ('invalid_json', []))
            state, rows = module.scan_file(session, roots, 1024 * 1024, 8)
            self.assertEqual((state, rows), ('line_too_large', []))
            state, rows = module.scan_file(session, roots, 1, 1024 * 1024)
            self.assertEqual((state, rows), ('too_large', []))

    @staticmethod
    def message(text, kinds):
        meta = {} if kinds is None else {'content_item_kinds': kinds}
        return {'type': 'response_item', 'payload': {
            'type': 'message', 'role': 'user',
            'content': [{'type': 'input_text', 'text': text}],
            'internal_chat_message_metadata_passthrough': meta}}


if __name__ == '__main__':
    unittest.main()
