"""Exercise evaluation mechanics with test-only fixtures, never real training evidence."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest

import node_evaluate
import node_review

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SNAPSHOT = REPO / 'crates/lfg-core/config/kernel-core.v1.json'
POLICY = ROOT / 'node-admission-policy.example.json'
BINARY = Path(os.environ.get('LFG_TEST_KERNEL_BINARY', REPO / 'target/debug' / ('lfg-server.exe' if os.name == 'nt' else 'lfg-server')))


class BaselineScoringTests(unittest.TestCase):
    def test_unreached_is_not_an_escalation(self):
        records = [{'review': {'itemSha256': 'a', 'choice': 'chat'}}]
        output = {'a': {'state': 'complete', 'atNode': 'done', 'hops': []}}
        result = node_evaluate.score(records, output, 'operation', {'chat', 'tool_call'}, {'a': 'chat'})
        self.assertEqual(result['baseline']['notReached'], 1)
        self.assertEqual(result['baseline']['escalated'], 0)
        self.assertEqual(result['fallbackOnEscalation']['eligible'], 0)

    def test_provisional_hop_is_not_a_committed_choice(self):
        output = {'state': 'escalate', 'atNode': 'operation',
                  'hops': [{'node': 'operation', 'operation': 'choice', 'value': 'chat'}]}
        self.assertEqual(node_evaluate.baseline_choice(output, 'operation', {'chat', 'tool_call'}), ('escalated', None))
        output['atNode'] = 'work-kind'
        self.assertEqual(node_evaluate.baseline_choice(output, 'operation', {'chat', 'tool_call'}), ('selected', 'chat'))


@unittest.skipUnless(BINARY.is_file(), 'build the local lfg-server binary for kernel integration tests')
class KernelValidationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.queue = self.root / 'queue.jsonl'
        self.ledger = self.root / 'ledger.json'
        snapshot_sha = node_review.digest(SNAPSHOT.read_bytes())
        # The strings and attestations in this temporary corpus are test fixtures only.
        examples = [('train fixture one', 'chat', 'train'), ('train fixture two', 'tool_call', 'train'),
                    ('run command', 'tool_call', 'validation'), ('logout', 'tool_call', 'validation'),
                    ('ordinary unmatched words', 'chat', 'validation'), ('logout run', 'tool_call', 'validation'),
                    ('test fixture one', 'chat', 'test'), ('test fixture two', 'tool_call', 'test')]
        rows = []
        for i, (text, _, _) in enumerate(examples):
            rows.append({'schemaVersion': 'lfg.node-review-queue-item.v1', 'status': 'pending_review',
                'requestId': f'fixture-{i}', 'text': text, 'inputSha256': node_review.digest(text.encode()),
                'sourceFileSha256': 'a' * 64, 'snapshotSha256': snapshot_sha, 'snapshotRevision': 'r1',
                'journalTailSha256': 'b' * 64, 'observationSequence': i + 1, 'node': 'operation',
                'options': ['chat', 'tool_call'], 'baselineState': 'escalate', 'baselineChoice': None,
                'reviewedChoice': None, 'reviewer': None, 'project': None, 'family': None, 'split': None})
        self.queue.write_bytes(b''.join(node_review.encode(row) + b'\n' for row in rows))
        node_review.init(self.queue, SNAPSHOT, self.ledger)
        self.ids = [node_review.identity(row) for row in rows]
        for i, (_, choice, split) in enumerate(examples):
            node_review.decide(argparse.Namespace(attest=True, reviewer='test-only', decision='approved',
                choice=choice, project=split, project_group=split, family=f'family-{i}', split=split,
                queue=self.queue, snapshot=SNAPSHOT, ledger=self.ledger, item_sha256=self.ids[i]))
        self.rows = rows
        self.args = argparse.Namespace(input=[[self.queue, self.ledger]], snapshot=SNAPSHOT, policy=POLICY,
            node='operation', kernel_binary=BINARY, timeout_seconds=30.0, max_validation_items=100,
            max_input_bytes=100000, candidate=None)

    def test_real_kernel_escalation_and_candidate_metrics(self):
        report = node_evaluate.evaluate(self.args)
        baseline = report['metrics']['baseline']
        self.assertEqual((baseline['total'], baseline['correct'], baseline['wrong'], baseline['escalated'], baseline['notReached']), (4, 1, 1, 2, 0))
        self.assertEqual(baseline['selectionRate'], 0.5)
        self.assertEqual(baseline['accuracyAmongSelected'], 0.5)
        self.assertFalse(report['qualificationEstablished'])
        path = self.root / 'candidate.jsonl'
        values = ['tool_call', 'chat', 'chat', None]
        rows = [{'schemaVersion': 'lfg.node-prediction.v1', 'itemSha256': self.ids[i],
                 'inputSha256': self.rows[i]['inputSha256'], 'snapshotSha256': self.rows[i]['snapshotSha256'],
                 'choice': value} for i, value in zip(range(2, 6), values, strict=True)]
        path.write_bytes(b''.join(node_review.encode(row) + b'\n' for row in rows))
        self.args.candidate = path
        report = node_evaluate.evaluate(self.args)
        fallback = report['metrics']['fallbackOnEscalation']
        self.assertEqual((fallback['eligible'], fallback['correct'], fallback['wrong'], fallback['abstained']), (2, 1, 0, 1))
        self.assertEqual(fallback['combinedCorrectPerReviewedItem'], 0.5)
        self.assertFalse(report['candidateExecutionVerified'])
        self.assertNotIn('ordinary unmatched words', json.dumps(report))
        path.write_bytes(b''.join(node_review.encode(row) + b'\n' for row in rows[:-1]))
        with self.assertRaises(ValueError):
            node_evaluate.evaluate(self.args)

    def test_validation_limits_are_enforced(self):
        self.args.max_validation_items = 3
        with self.assertRaises(ValueError):
            node_evaluate.evaluate(self.args)
        self.args.max_validation_items = 4
        self.args.max_input_bytes = 1
        with self.assertRaises(ValueError):
            node_evaluate.evaluate(self.args)


if __name__ == '__main__':
    unittest.main()
