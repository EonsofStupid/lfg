"""Mechanical fixtures exercise admission behavior; these are not training data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'node_review.py'
SNAPSHOT = ROOT.parent.parent / 'crates/lfg-core/config/kernel-core.v1.json'
POLICY = ROOT / 'node-admission-policy.example.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


class NodeReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / 'queue.jsonl'
        self.ledger = self.root / 'ledger.json'
        snapshot_sha = sha(SNAPSHOT.read_bytes())
        records = []
        for i in range(7):
            text = f'isolated fixture {i}'
            records.append({'schemaVersion': 'lfg.node-review-queue-item.v1', 'status': 'pending_review',
                'requestId': f'fixture-{i}', 'text': text, 'inputSha256': sha(text.encode()),
                'sourceFileSha256': 'a' * 64, 'snapshotSha256': snapshot_sha, 'snapshotRevision': 'r1',
                'journalTailSha256': 'b' * 64, 'observationSequence': i + 1, 'node': 'operation',
                'options': ['chat', 'tool_call'], 'baselineState': 'escalate', 'baselineChoice': None,
                'reviewedChoice': None, 'reviewer': None, 'project': None, 'family': None, 'split': None})
        self.queue.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
        self.run_cli('init', *self.paths())
        self.items = self.run_cli('list', *self.paths())

    def paths(self):
        return ['--queue', str(self.queue), '--snapshot', str(SNAPSHOT), '--ledger', str(self.ledger)]

    def run_cli(self, *args, success=True):
        run = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
        self.assertEqual(run.returncode == 0, success, run.stderr)
        return json.loads(run.stdout) if success else run.stderr

    def decision(self, index, choice, split, *, group=None):
        return self.run_cli('decide', *self.paths(), '--item-sha256', self.items[index]['itemSha256'],
            '--decision', 'approved', '--reviewer', 'fixture-reviewer', '--choice', choice,
            '--project', split, '--project-group', group or split, '--family', f'family-{index}',
            '--split', split, '--attest')

    def audit(self):
        return self.run_cli('audit', '--input', str(self.queue), str(self.ledger), '--snapshot', str(SNAPSHOT),
                            '--policy', str(POLICY), '--node', 'operation')

    def test_pending_review_cannot_export_and_queue_changes_invalidate_ledger(self):
        report = self.audit()
        self.assertEqual(report['pending'], 7)
        self.assertFalse(report['admissionEligible'])
        output = self.root / 'output.json'
        self.run_cli('export', '--input', str(self.queue), str(self.ledger), '--snapshot', str(SNAPSHOT),
                     '--policy', str(POLICY), '--node', 'operation', '--output', str(output), success=False)
        self.assertFalse(output.exists())
        shown = self.run_cli('show', *self.paths(), '--item-sha256', self.items[0]['itemSha256'])
        self.assertIsNone(shown['text'])
        self.assertNotIn('baselineChoice', shown)
        shown = self.run_cli('show', *self.paths(), '--item-sha256', self.items[0]['itemSha256'], '--include-text')
        self.assertEqual(shown['text'], 'isolated fixture 0')
        self.assertNotIn('baselineChoice', shown)
        self.run_cli('decide', *self.paths(), '--item-sha256', self.items[0]['itemSha256'],
                     '--decision', 'approved', '--reviewer', 'fixture', '--choice', 'chat', '--project', 'p',
                     '--project-group', 'g', '--family', 'f', '--split', 'train', success=False)
        self.queue.write_text(self.queue.read_text(encoding='utf-8').replace('isolated fixture 0', 'changed fixture 0'), encoding='utf-8')
        self.run_cli('list', *self.paths(), success=False)

    def test_admission_requires_separation_and_export_excludes_test_text(self):
        for i, split in enumerate(('train', 'validation', 'test')):
            self.decision(i * 2, 'chat', split)
            self.decision(i * 2 + 1, 'tool_call', split)
        self.run_cli('decide', *self.paths(), '--item-sha256', self.items[6]['itemSha256'], '--decision',
                     'rejected', '--reviewer', 'fixture-reviewer', '--attest')
        report = self.audit()
        self.assertTrue(report['admissionEligible'])
        self.assertFalse(report['qualificationEstablished'])
        self.assertEqual(report['rejected'], 1)
        output = self.root / 'output.json'
        export_args = ['export', '--input', str(self.queue), str(self.ledger), '--snapshot', str(SNAPSHOT),
                       '--policy', str(POLICY), '--node', 'operation', '--output', str(output)]
        self.run_cli(*export_args)
        result = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual((len(result['train']), len(result['validation'])), (2, 2))
        self.assertNotIn('isolated fixture 4', output.read_text(encoding='utf-8'))
        self.assertNotIn('test', [row['project'] for row in result['train'] + result['validation']])
        self.run_cli(*export_args, success=False)
        second = self.root / 'output2.json'
        self.run_cli(*(export_args[:-1] + [str(second)]))
        self.assertEqual(output.read_bytes(), second.read_bytes())

    def test_cross_split_group_conflict_fails_closed(self):
        for i, split in enumerate(('train', 'validation', 'test')):
            self.decision(i * 2, 'chat', split, group='shared' if split != 'test' else 'test')
            self.decision(i * 2 + 1, 'tool_call', split, group='shared' if split != 'test' else 'test')
        self.run_cli('decide', *self.paths(), '--item-sha256', self.items[6]['itemSha256'], '--decision',
                     'rejected', '--reviewer', 'fixture-reviewer', '--attest')
        report = self.audit()
        self.assertIn('projectGroup', report['conflicts'])
        self.assertFalse(report['admissionEligible'])

    def test_cross_node_split_leakage_is_rejected(self):
        lines = self.queue.read_text(encoding='utf-8').splitlines()
        other = json.loads(lines[0])
        other['node'] = 'work-kind'
        other['options'] = ['code.fix', 'explain']
        self.queue.write_text('\n'.join(lines + [json.dumps(other)]) + '\n', encoding='utf-8')
        self.ledger.unlink()
        self.run_cli('init', *self.paths())
        self.items = self.run_cli('list', *self.paths())
        self.decision(0, 'chat', 'train')
        self.decision(7, 'code.fix', 'validation')
        report = self.audit()
        self.assertIn('text', report['conflicts'])
        self.assertIn('request', report['conflicts'])


if __name__ == '__main__':
    unittest.main()
