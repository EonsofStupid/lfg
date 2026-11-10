"""Test-only review attestations and process responses; not quality evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import evaluate_embedded_holdout as evaluator
import node_evaluate
import node_review

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT.parent.parent / 'crates/lfg-core/config/kernel-core.v1.json'
DEVELOPMENT_POLICY = ROOT / 'node-development-policy.example.json'
FINAL_POLICY = ROOT / 'node-admission-policy.example.json'


class FrozenHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dev = self.make_source('dev', ('train', 'train', 'validation', 'validation'))
        self.test = self.make_source('test', ('test', 'test'))
        self.commitment = self.root / 'commitment.json'
        self.commitment.write_bytes(node_review.encode(evaluator.holdout_commitment([self.test], SNAPSHOT, 'operation')) + b'\n')
        self.args = argparse.Namespace(input=[self.dev], holdout_input=[self.test], snapshot=SNAPSHOT,
            policy=DEVELOPMENT_POLICY, final_policy=FINAL_POLICY, node='operation', max_items=20,
            max_input_bytes=10000, max_output_bytes=20000, timeout_seconds=10,
            binary=self.root / 'binary', artifact=self.root, manifest_sha256='a' * 64,
            pack=self.root / 'pack', pack_sha256='b' * 64, runtime_config=self.root / 'runtime',
            exemplar_config=self.root / 'config', freeze_sha256='f' * 64, commitment=self.commitment,
            commitment_sha256=node_review.digest(self.commitment.read_bytes()))

    def make_source(self, name, splits, *, duplicate_text=None):
        queue = self.root / f'{name}.jsonl'
        ledger = self.root / f'{name}-ledger.json'
        snapshot_sha = node_review.digest(SNAPSHOT.read_bytes())
        rows = []
        for index, split in enumerate(splits):
            text = duplicate_text if duplicate_text is not None and index == 0 else f'{name} fixture {index}'
            rows.append({'schemaVersion': 'lfg.node-review-queue-item.v1', 'status': 'pending_review',
                'requestId': f'{name}-{index}', 'text': text, 'inputSha256': node_review.digest(text.encode()),
                'sourceFileSha256': 'a' * 64, 'snapshotSha256': snapshot_sha, 'snapshotRevision': 'r1',
                'journalTailSha256': node_review.digest(name.encode()), 'observationSequence': index + 1,
                'node': 'operation', 'options': ['chat', 'tool_call'], 'baselineState': 'escalate',
                'baselineChoice': None, 'reviewedChoice': None, 'reviewer': None, 'project': None,
                'family': None, 'split': None})
        queue.write_bytes(b''.join(node_review.encode(row) + b'\n' for row in rows))
        node_review.init(queue, SNAPSHOT, ledger)
        for index, (row, split) in enumerate(zip(rows, splits, strict=True)):
            node_review.decide(argparse.Namespace(attest=True, reviewer='test-only', decision='approved',
                choice='chat' if index % 2 == 0 else 'tool_call', project=split, project_group=split,
                family=f'{name}-{index}', split=split, queue=queue, snapshot=SNAPSHOT, ledger=ledger,
                item_sha256=node_review.identity(row)))
        return [queue, ledger]

    @staticmethod
    def baseline(binary, snapshot, snapshot_sha, records, *args):
        return {row['review']['itemSha256']: {'schemaVersion': 'lfg.kernel-result.v1',
                'snapshotSha256': snapshot_sha, 'state': 'escalate', 'atNode': 'operation', 'hops': []}
                for row in records}

    @staticmethod
    def candidate(args, records, *unused):
        return {row['review']['itemSha256']: row['review']['choice'] for row in records}

    def test_freeze_rejects_three_split_policy_before_native_validator(self):
        self.args.policy = FINAL_POLICY
        with mock.patch.object(evaluator.native, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'development-only policy'):
                evaluator.freeze_candidate(self.args)
            run.assert_not_called()

    def test_changed_freeze_rejects_before_loading_holdout(self):
        with mock.patch.object(evaluator, 'freeze_candidate', return_value={'changed': True}), \
             mock.patch.object(node_review, 'load_ledger') as load:
            with self.assertRaisesRegex(ValueError, 'differs from frozen'):
                evaluator.evaluate_holdout(self.args, {'expected': True})
            load.assert_not_called()

    def test_cli_rejects_wrong_freeze_digest_without_output(self):
        frozen = self.root / 'freeze.json'
        frozen.write_text('{}\n', encoding='utf-8')
        output = self.root / 'must-not-exist.json'
        script = ROOT / 'evaluate_embedded_holdout.py'
        command = [sys.executable, str(script), 'test', '--input', *map(str, self.dev),
            '--holdout-input', str(self.root / 'not-opened'), str(self.root / 'not-opened-ledger'),
            '--snapshot', str(SNAPSHOT), '--policy', str(DEVELOPMENT_POLICY), '--final-policy', str(FINAL_POLICY),
            '--binary', str(self.args.binary), '--artifact', str(self.args.artifact),
            '--runtime-config', str(self.args.runtime_config), '--pack', str(self.args.pack),
            '--exemplar-config', str(self.args.exemplar_config), '--node', 'operation',
            '--manifest-sha256', self.args.manifest_sha256, '--pack-sha256', self.args.pack_sha256,
            '--commitment', str(self.commitment), '--commitment-sha256', self.args.commitment_sha256,
            '--max-items', '20', '--max-input-bytes', '10000', '--max-output-bytes', '20000',
            '--timeout-seconds', '10', '--freeze', str(frozen), '--freeze-sha256', '0' * 64,
            '--output', str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('freeze digest mismatch', result.stderr)
        self.assertFalse(output.exists())

    def test_freeze_binds_development_metrics_and_requires_agreement(self):
        audit, records = node_review.audit(self.args.input, SNAPSHOT, DEVELOPMENT_POLICY, 'operation')
        validation = [record for record in records if record['review']['split'] == 'validation']
        baseline = self.baseline(None, None, audit['snapshotSha256'], validation)
        choices = self.candidate(self.args, validation)
        metrics = node_evaluate.score(validation, baseline, 'operation', {'chat', 'tool_call'}, choices)
        report = {'admission': audit, 'runtimeConfigSha256': 'c' * 64,
                  'exemplarConfigSha256': 'd' * 64, 'metrics': metrics}
        with mock.patch.object(evaluator.native, 'run', return_value=report), \
             mock.patch.object(node_evaluate, 'predictions', side_effect=self.baseline), \
             mock.patch.object(evaluator, 'external_predictions', side_effect=self.candidate), \
             mock.patch.object(evaluator, 'embedded_predictions', side_effect=self.candidate), \
             mock.patch.object(evaluator, 'holdout_commitment', side_effect=AssertionError('holdout opened')):
            frozen = evaluator.freeze_candidate(self.args)
        self.assertFalse(frozen['holdoutInputParametersAccepted'])
        self.assertFalse(frozen['qualificationEstablished'])
        self.assertEqual(frozen['embeddedValidationMetrics'], metrics)
        with mock.patch.object(evaluator.native, 'run', return_value=report), \
             mock.patch.object(node_evaluate, 'predictions', side_effect=self.baseline), \
             mock.patch.object(evaluator, 'external_predictions', side_effect=self.candidate), \
             mock.patch.object(evaluator, 'embedded_predictions', return_value={key: None for key in choices}):
            with self.assertRaisesRegex(ValueError, 'choices differ'):
                evaluator.freeze_candidate(self.args)

    def test_commitment_is_bound_and_pending_holdout_is_rejected(self):
        self.assertNotIn('test fixture', self.commitment.read_text(encoding='utf-8'))
        self.args.commitment_sha256 = '0' * 64
        with mock.patch.object(evaluator.native, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'commitment digest mismatch'):
                evaluator.freeze_candidate(self.args)
            run.assert_not_called()
        pending = self.root / 'pending.json'
        node_review.init(self.test[0], SNAPSHOT, pending)
        with self.assertRaisesRegex(ValueError, 'must be decided'):
            evaluator.holdout_commitment([[self.test[0], pending]], SNAPSHOT, 'operation')

    def test_holdout_comparison_requires_separation_and_keeps_output_aggregate(self):
        frozen = {'developmentValidation': {'runtimeConfigSha256': 'c' * 64,
                                            'exemplarConfigSha256': 'd' * 64}}
        with mock.patch.object(evaluator, 'freeze_candidate', return_value=frozen), \
             mock.patch.object(node_evaluate, 'predictions', side_effect=self.baseline), \
             mock.patch.object(evaluator, 'embedded_predictions', side_effect=self.candidate):
            report = evaluator.evaluate_holdout(self.args, frozen)
        self.assertEqual(report['metrics']['fallbackOnEscalation']['correct'], 2)
        self.assertFalse(report['qualificationEstablished'])
        self.assertNotIn('test fixture 0', json.dumps(report))
        def unreached(binary, snapshot, snapshot_sha, records, *args):
            return {row['review']['itemSha256']: {'schemaVersion': 'lfg.kernel-result.v1',
                    'snapshotSha256': snapshot_sha, 'state': 'complete', 'atNode': 'done', 'hops': []}
                    for row in records}
        with mock.patch.object(evaluator, 'freeze_candidate', return_value=frozen), \
             mock.patch.object(node_evaluate, 'predictions', side_effect=unreached), \
             mock.patch.object(evaluator, 'embedded_predictions', side_effect=self.candidate):
            with self.assertRaisesRegex(ValueError, 'no target escalation'):
                evaluator.evaluate_holdout(self.args, frozen)
        self.args.holdout_input = [self.make_source('overlap', ('test', 'test'), duplicate_text='dev fixture 0')]
        with mock.patch.object(evaluator, 'freeze_candidate', return_value=frozen), \
             mock.patch.object(node_evaluate, 'predictions') as predict:
            with self.assertRaisesRegex(ValueError, 'precommitted sources'):
                evaluator.evaluate_holdout(self.args, frozen)
            predict.assert_not_called()
        self.commitment.write_bytes(node_review.encode(evaluator.holdout_commitment(self.args.holdout_input, SNAPSHOT, 'operation')) + b'\n')
        self.args.commitment_sha256 = node_review.digest(self.commitment.read_bytes())
        with mock.patch.object(evaluator, 'freeze_candidate', return_value=frozen), \
             mock.patch.object(node_evaluate, 'predictions') as predict:
            with self.assertRaisesRegex(ValueError, 'separation or support'):
                evaluator.evaluate_holdout(self.args, frozen)
            predict.assert_not_called()

    def test_embedded_identity_and_escalation_are_checked(self):
        _, records = node_review.audit([self.dev], SNAPSHOT, DEVELOPMENT_POLICY, 'operation')
        validation = [record for record in records if record['review']['split'] == 'validation']
        audit = {'snapshotSha256': node_review.digest(SNAPSHOT.read_bytes())}
        baseline = self.baseline(None, None, audit['snapshotSha256'], validation)
        runtime_sha, config_sha = 'c' * 64, 'd' * 64
        rows = []
        for record in validation:
            key = record['review']['itemSha256']
            rows.append({'schemaVersion': 'lfg.node-suggestion-probe.v1', 'id': key,
                'inputSha256': record['item']['inputSha256'], 'snapshotSha256': audit['snapshotSha256'],
                'encoderManifestSha256': self.args.manifest_sha256, 'packSha256': self.args.pack_sha256,
                'runtimeConfigSha256': runtime_sha, 'exemplarConfigSha256': config_sha,
                'baseline': baseline[key], 'extensionStatus': 'suggested', 'promotionEligible': False,
                'suggestion': {'choice': record['review']['choice'], 'similarity': 0.9, 'margin': 0.1,
                               'source': 'exemplar'}})
        self.args.binary.write_bytes(b'fixture')
        response = argparse.Namespace(returncode=0, stdout=b''.join(node_review.encode(row) + b'\n' for row in rows))
        with mock.patch.object(evaluator.subprocess, 'run', return_value=response):
            predictions = evaluator.embedded_predictions(self.args, validation, baseline, audit,
                                                        {'chat', 'tool_call'}, runtime_sha, config_sha)
        self.assertEqual(len(predictions), 2)
        rows[0]['packSha256'] = '0' * 64
        response.stdout = b''.join(node_review.encode(row) + b'\n' for row in rows)
        with mock.patch.object(evaluator.subprocess, 'run', return_value=response):
            with self.assertRaisesRegex(ValueError, 'identity or baseline'):
                evaluator.embedded_predictions(self.args, validation, baseline, audit,
                                              {'chat', 'tool_call'}, runtime_sha, config_sha)


if __name__ == '__main__':
    unittest.main()
