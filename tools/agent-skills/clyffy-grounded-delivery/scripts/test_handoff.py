import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
handoff = load('handoff')
hook = load('hook')


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / 'repo'
        self.repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        self.source = self.repo / 'source.txt'
        self.source.write_text('grounded', encoding='utf8')
        subprocess.run(['git', '-C', str(self.repo), 'add', 'source.txt'], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.email=test@example.invalid', '-c', 'user.name=Test', 'commit', '-qm', 'baseline'], check=True)
        commit = subprocess.check_output(['git', '-C', str(self.repo), 'rev-parse', 'HEAD'], text=True).strip()
        self.doc = {'schema_version': 'clyffy.handoff.v1', 'id': 'example', 'status': 'ready', 'objective': 'actual task',
            'baseline_commit': commit, 'evidence': [{'kind': 'source', 'location': 'source.txt', 'claim': 'exists'}],
            'constraints': ['privacy'], 'authorized_scope': ['source'],
            'anchors': [{'path': 'source.txt', 'sha256': hashlib.sha256(self.source.read_bytes()).hexdigest()}],
            'tasks': [{'id': 'one', 'title': 'one', 'status': 'pending', 'paths': ['source.txt'], 'depends_on': [],
                'steps': ['implement'], 'checks': [{'argv': ['false'], 'expected': 'recorded only'}], 'acceptance': ['works']}],
            'next_task': 'one', 'escalation_triggers': ['changed'], 'extra': {'allowed': True}}
        self.file = Path(self.tmp.name) / 'handoff.json'
        self.policy = Path(self.tmp.name) / 'routes.toml'
        self.policy.write_text('[executor]\nmodel="test-model"\neffort="high"\n', encoding='utf8')
        self.write()

    def write(self):
        self.file.write_text(json.dumps(self.doc), encoding='utf8')

    def test_valid_extension_fields_and_anchors(self):
        parsed, _, _ = handoff.read_handoff(self.file)
        self.assertTrue(parsed['extra']['allowed'])
        handoff.verify_repo(parsed, self.repo, check_anchors=True)
        self.source.write_text('changed', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'anchor changed'):
            handoff.verify_repo(parsed, self.repo, check_anchors=True)

    def test_rejects_cycles_and_unsafe_paths(self):
        bad = copy.deepcopy(self.doc)
        second = copy.deepcopy(bad['tasks'][0])
        second['id'], second['depends_on'] = 'two', ['one']
        bad['tasks'][0]['depends_on'] = ['two']
        bad['tasks'].append(second)
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            handoff.validate_document(bad)
        bad = copy.deepcopy(self.doc)
        bad['anchors'][0]['path'] = '../secret'
        with self.assertRaisesRegex(ValueError, 'repository-relative'):
            handoff.validate_document(bad)

    def test_text_anchor_accepts_git_checkout_line_endings(self):
        self.source.write_bytes(b'first\nsecond\n')
        self.doc['anchors'][0]['normalization'] = 'text-lf'
        self.doc['anchors'][0]['sha256'] = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.source.write_bytes(b'first\r\nsecond\r\n')
        handoff.verify_repo(self.doc, self.repo, check_anchors=True)
        self.source.write_bytes(b'first\r\nchanged\r\n')
        with self.assertRaisesRegex(ValueError, 'anchor changed'):
            handoff.verify_repo(self.doc, self.repo, check_anchors=True)
    def test_blocked_executor_and_completed_dependency(self):
        self.doc['status'] = 'blocked'
        self.write()
        with self.assertRaisesRegex(ValueError, 'does not permit'):
            handoff.main(['dispatch', str(self.file), '--repo', str(self.repo), '--policy', str(self.policy)])
        self.doc['status'] = 'ready'
        self.doc['tasks'][0]['status'] = 'complete'
        with self.assertRaises(ValueError):
            handoff.validate_document(self.doc)

    def test_dry_run_and_explicit_configured_execute(self):
        with mock.patch.object(handoff.shutil, 'which', return_value='codex'):
            self.assertEqual(handoff.main(['dispatch', str(self.file), '--repo', str(self.repo), '--policy', str(self.policy)]), 0)
            actual = handoff.subprocess.run
            calls = []
            def run(argv, **kwargs):
                if argv[0] == 'codex':
                    calls.append((argv, kwargs))
                    return subprocess.CompletedProcess(argv, 0)
                return actual(argv, **kwargs)
            with mock.patch.object(handoff.subprocess, 'run', side_effect=run):
                self.assertEqual(handoff.main(['dispatch', str(self.file), '--repo', str(self.repo), '--policy', str(self.policy), '--execute']), 0)
            self.assertEqual(len(calls), 1)
            argv, kwargs = calls[0]
            self.assertIn('test-model', argv)
            self.assertNotIn('--dangerously-bypass-approvals-and-sandbox', argv)
            self.assertIn('actual task', kwargs['input'])

    def test_event_log_cannot_write_inside_repo(self):
        with self.assertRaisesRegex(ValueError, 'outside the repository'):
            handoff.append_event(str(self.repo / 'events.jsonl'), {'event': 'started'}, self.repo.resolve())

    def test_reviewed_outcome_is_linked_to_finished_attempt(self):
        events = Path(self.tmp.name) / 'private-events.jsonl'
        self.policy.write_text('event_log = ' + json.dumps(str(events)) + '\n[executor]\nmodel="test-model"\neffort="high"\n', encoding='utf8')
        actual = handoff.subprocess.run
        def run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0) if argv[0] == 'codex' else actual(argv, **kwargs)
        with mock.patch.object(handoff.shutil, 'which', return_value='codex'), mock.patch.object(handoff.subprocess, 'run', side_effect=run):
            self.assertEqual(handoff.main(['dispatch', str(self.file), '--repo', str(self.repo), '--policy', str(self.policy), '--execute']), 0)
        entries = [json.loads(line) for line in events.read_text().splitlines()]
        attempt = entries[-1]['attempt_id']
        self.assertEqual(handoff.main(['record', str(self.file), '--repo', str(self.repo), '--policy', str(self.policy), '--attempt', attempt, '--result', 'accepted', '--checks-passed', '2', '--checks-failed', '0', '--rework-count', '1', '--escalated']), 0)
        last = json.loads(events.read_text().splitlines()[-1])
        self.assertEqual(last['result'], 'accepted')
        self.assertTrue(last['escalated'])
        self.assertNotIn('actual task', events.read_text())
        self.doc['objective'] = 'changed'
        self.write()
        with self.assertRaisesRegex(ValueError, 'different handoff revision'):
            handoff.main(['record', str(self.file), '--repo', str(self.repo), '--policy', str(self.policy), '--attempt', attempt, '--result', 'accepted', '--checks-passed', '2', '--checks-failed', '0', '--rework-count', '0'])

    def test_hook_does_not_echo_prompt(self):
        message = hook.respond({'hook_event_name': 'UserPromptSubmit', 'prompt': 'clyffy-grounded-delivery secret-value'})
        self.assertNotIn('secret-value', json.dumps(message))
        self.assertEqual(hook.respond({'hook_event_name': 'UserPromptSubmit', 'prompt': 'unrelated'}), {})
        self.assertTrue(hook.respond({'hook_event_name': 'SubagentStart', 'agent_type': 'clyffy_executor'}))

    def test_install_does_not_overwrite_user_configuration(self):
        home = Path(self.tmp.name) / 'codex'
        args = [sys.executable, str(HERE / 'install.py'), '--codex-home', str(home)]
        self.assertEqual(subprocess.run(args, capture_output=True, text=True).returncode, 0)
        with (home / 'agents' / 'clyffy_executor.toml').open('rb') as source:
            self.assertEqual(tomllib.load(source)['name'], 'clyffy_executor')
        hooks = home / 'hooks.json'
        original = hooks.read_bytes()
        self.assertIn('SubagentStart', json.loads(original)['hooks'])
        self.assertNotEqual(subprocess.run(args, capture_output=True, text=True).returncode, 0)
        self.assertEqual(hooks.read_bytes(), original)

if __name__ == '__main__':
    unittest.main()
