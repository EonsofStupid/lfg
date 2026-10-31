"""Exercise exemplar mechanics with test-only vectors and attestations."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import node_exemplars
import node_review

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SNAPSHOT = REPO / 'crates/lfg-core/config/kernel-core.v1.json'
POLICY = ROOT / 'node-admission-policy.example.json'
CONFIG = ROOT / 'node-exemplar-config.example.json'


class SimilarityTests(unittest.TestCase):
    def test_similarity_margin_and_vector_validity(self):
        config, _ = node_exemplars.read_config(CONFIG)
        exemplars = [{'choice': 'a', 'embedding': [1.0, 0.0]}, {'choice': 'b', 'embedding': [0.0, 1.0]}]
        self.assertEqual(node_exemplars.choose([1.0, 0.0], exemplars, ['a', 'b'], config), 'a')
        self.assertIsNone(node_exemplars.choose([2 ** -0.5, 2 ** -0.5], exemplars, ['a', 'b'], config))
        with self.assertRaises(ValueError):
            node_exemplars.choose([2.0, 0.0], exemplars, ['a', 'b'], config)
        with self.assertRaises(ValueError):
            node_exemplars.choose([1.0, 0.0], exemplars[:1], ['a', 'b'], config)

    def test_threshold_tuning_does_not_change_encoder_contract(self):
        config, _ = node_exemplars.read_config(CONFIG)
        changed = dict(config, revision='new-decision-policy', minimumSimilarity=0.9, minimumMargin=0.4)
        self.assertEqual(node_exemplars.encoder_config_sha(config), node_exemplars.encoder_config_sha(changed))
        changed['maxTokens'] += 1
        self.assertNotEqual(node_exemplars.encoder_config_sha(config), node_exemplars.encoder_config_sha(changed))


class FakeEncoder:
    def __init__(self, model, config):
        self.versions = {'fixture': 'not-a-real-encoder'}

    def encode(self, texts):
        return [[1.0, 0.0] if 'alpha' in text else [0.0, 1.0] for text in texts], [2] * len(texts)


class ExemplarWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.queue = self.root / 'queue.jsonl'
        self.ledger = self.root / 'ledger.json'
        self.pack = self.root / 'pack.json'
        self.model = self.root / 'model'
        self.model.mkdir()
        snapshot_sha = node_review.digest(SNAPSHOT.read_bytes())
        rows = []
        for i, split in enumerate(('train', 'train', 'validation', 'validation', 'test', 'test')):
            choice = 'chat' if i % 2 == 0 else 'tool_call'
            text = f'{split} fixture ' + ('alpha' if choice == 'chat' else 'beta')
            rows.append({'schemaVersion': 'lfg.node-review-queue-item.v1', 'status': 'pending_review',
                'requestId': f'fixture-{i}', 'text': text, 'inputSha256': node_review.digest(text.encode()),
                'sourceFileSha256': 'a' * 64, 'snapshotSha256': snapshot_sha, 'snapshotRevision': 'r1',
                'journalTailSha256': 'b' * 64, 'observationSequence': i + 1, 'node': 'operation',
                'options': ['chat', 'tool_call'], 'baselineState': 'escalate', 'baselineChoice': None,
                'reviewedChoice': None, 'reviewer': None, 'project': None, 'family': None, 'split': None})
        self.queue.write_bytes(b''.join(node_review.encode(row) + b'\n' for row in rows))
        node_review.init(self.queue, SNAPSHOT, self.ledger)
        self.rows = rows
        self.args = argparse.Namespace(input=[[self.queue, self.ledger]], snapshot=SNAPSHOT, policy=POLICY,
            node='operation', config=CONFIG, model=self.model, model_manifest=self.root / 'unused', revision='fixture',
            environment_lock=self.root / 'unused-lock', wheelhouse=self.root, allowed_extra=[], output=self.pack)

    def approve_fixtures(self):
        for i, row in enumerate(self.rows):
            split = ('train', 'train', 'validation', 'validation', 'test', 'test')[i]
            node_review.decide(argparse.Namespace(attest=True, reviewer='test-only', decision='approved',
                choice='chat' if i % 2 == 0 else 'tool_call', project=split, project_group=split,
                family=f'fixture-{i}', split=split, queue=self.queue, snapshot=SNAPSHOT, ledger=self.ledger,
                item_sha256=node_review.identity(row)))

    def test_pending_data_cannot_initialize_encoder_or_create_pack(self):
        with mock.patch.object(node_exemplars, 'Encoder') as encoder:
            with self.assertRaisesRegex(ValueError, 'admission gates'):
                node_exemplars.build(self.args)
            encoder.assert_not_called()
        self.assertFalse(self.pack.exists())

    def test_probe_uses_pending_text_without_approving_or_saving_vectors(self):
        self.args.queue = self.queue
        self.args.ledger = self.ledger
        self.args.repetitions = 2
        with mock.patch.object(node_exemplars, 'Encoder', FakeEncoder), \
             mock.patch.object(node_exemplars, 'model_proof', return_value={'fixture': 'model'}):
            report = node_exemplars.probe(self.args)
        self.assertTrue(report['sameProcessExact'])
        self.assertFalse(report['labelsUsed'])
        self.assertFalse(report['vectorsSaved'])
        self.assertNotIn('fixture alpha', json.dumps(report))
        self.assertFalse(self.pack.exists())

    def test_reviewed_fixture_build_prediction_and_dynamic_threshold(self):
        self.approve_fixtures()
        with mock.patch.object(node_exemplars, 'Encoder', FakeEncoder), \
             mock.patch.object(node_exemplars, 'model_proof', return_value={'fixture': 'model'}), \
             mock.patch.object(node_exemplars, 'environment', return_value={'fixture': 'environment'}):
            result = node_exemplars.build(self.args)
            self.assertFalse(result['promotionEligible'])
            pack = json.loads(self.pack.read_text(encoding='utf-8'))
            self.assertEqual(len(pack['exemplars']), 2)
            self.assertNotIn('fixture alpha', self.pack.read_text(encoding='utf-8'))
            self.args.output = self.root / 'pack-again.json'
            self.assertEqual(node_exemplars.build(self.args)['packSha256'], result['packSha256'])
            self.assertEqual(self.pack.read_bytes(), self.args.output.read_bytes())
            self.args.pack = self.pack
            self.args.pack_sha256 = result['packSha256']
            self.args.output = self.root / 'predictions.jsonl'
            predicted = node_exemplars.predict(self.args)
            self.assertEqual((predicted['items'], predicted['abstained']), (2, 0))
            rows = [json.loads(line) for line in self.args.output.read_text(encoding='utf-8').splitlines()]
            self.assertEqual({row['choice'] for row in rows}, {'chat', 'tool_call'})
            config = json.loads(CONFIG.read_text(encoding='utf-8'))
            config['revision'] = 'stricter-fixture-policy'
            config['minimumMargin'] = 2.0
            changed = self.root / 'config.json'
            changed.write_text(json.dumps(config), encoding='utf-8')
            self.args.config = changed
            self.args.output = self.root / 'strict-predictions.jsonl'
            self.assertEqual(node_exemplars.predict(self.args)['abstained'], 2)
            self.args.pack_sha256 = '0' * 64
            self.args.output = self.root / 'bad.jsonl'
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                node_exemplars.predict(self.args)
            self.assertFalse(self.args.output.exists())

    def test_pack_limit_rejects_output(self):
        self.approve_fixtures()
        config = json.loads(CONFIG.read_text(encoding='utf-8'))
        config['maxPackBytes'] = 1
        config_path = self.root / 'small-config.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        self.args.config = config_path
        with mock.patch.object(node_exemplars, 'Encoder', FakeEncoder), \
             mock.patch.object(node_exemplars, 'model_proof', return_value={'fixture': 'model'}), \
             mock.patch.object(node_exemplars, 'environment', return_value={'fixture': 'environment'}):
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                node_exemplars.build(self.args)
        self.assertFalse(self.pack.exists())


if __name__ == '__main__':
    unittest.main()
