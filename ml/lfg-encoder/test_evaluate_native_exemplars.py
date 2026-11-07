"""Tests use simulated review records and never constitute training evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import evaluate_native_exemplars

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SNAPSHOT = REPO / 'crates/lfg-core/config/kernel-core.v1.json'
POLICY = ROOT / 'node-admission-policy.example.json'
CONFIG = ROOT / 'node-exemplar-config.example.json'
RUNTIME = REPO / 'crates/lfg-server/config/node-onnx-runtime.example.json'
ARTIFACT = os.environ.get('LFG_TEST_NODE_ARTIFACT')
PACK = os.environ.get('LFG_TEST_NODE_PACK')
BINARY = os.environ.get('LFG_TEST_KERNEL_BINARY')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class NativeEvaluationBoundaryTests(unittest.TestCase):
    def test_wrong_pack_is_rejected_before_review_or_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / 'artifact'
            artifact.mkdir()
            (artifact / 'model.onnx').write_bytes(b'fixture model')
            (artifact / 'tokenizer.json').write_bytes(b'fixture tokenizer')
            raw = json.dumps({'modelSha256': sha(b'fixture model'), 'tokenizerSha256': sha(b'fixture tokenizer')}).encode()
            (artifact / 'manifest.json').write_bytes(raw)
            pack = root / 'pack'
            pack.write_bytes(b'fixture pack')
            for name in ('binary', 'runtime', 'config'):
                (root / name).write_bytes(b'fixture')
            args = argparse.Namespace(max_items=5, max_input_bytes=1000, timeout_seconds=5,
                binary=root / 'binary', artifact=artifact, manifest_sha256=sha(raw), pack=pack,
                pack_sha256='0' * 64, runtime_config=root / 'runtime', exemplar_config=root / 'config')
            with mock.patch.object(evaluate_native_exemplars.node_review, 'audit') as audit, \
                 mock.patch.object(evaluate_native_exemplars.node_exemplars, 'read_config', return_value=({'maxPackBytes': 1000}, 'fixture')), \
                 mock.patch.object(evaluate_native_exemplars.subprocess, 'run') as process:
                with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                    evaluate_native_exemplars.run(args)
                audit.assert_not_called()
                process.assert_not_called()


@unittest.skipUnless(ARTIFACT and PACK and BINARY, 'set explicit test-only native artifact, pack and binary paths')
class NativeEvaluationFixtureTests(unittest.TestCase):
    def test_native_process_accepts_development_admission_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = Path(ARTIFACT)
            pack_value = json.loads(Path(PACK).read_bytes())
            audit = pack_value['admission']
            audit.update(schemaVersion='lfg.node-admission-report.v2', scope='development',
                         holdoutAssessment='not_performed')
            audit['counts'].pop('test', None)
            audit.get('projectGroups', {}).pop('test', None)
            pack = root / 'test-only-development-pack.json'
            pack.write_text(json.dumps(pack_value), encoding='utf-8')
            texts = ['run command', 'fixture unknown']
            choices = ['tool_call', 'chat']
            records = [{'review': {'itemSha256': str(i + 1) * 64, 'split': 'validation', 'choice': choices[i]},
                        'item': {'text': text, 'inputSha256': sha(text.encode())}}
                       for i, text in enumerate(texts)]
            for exemplar in pack_value['exemplars']:
                records.append({'review': {'itemSha256': exemplar['itemSha256'], 'split': 'train',
                                           'choice': exemplar['choice']}, 'item': {'text': 'test-only'}})
            args = argparse.Namespace(input=[], snapshot=SNAPSHOT, policy=POLICY, node='operation',
                binary=Path(BINARY), artifact=artifact, manifest_sha256=sha((artifact / 'manifest.json').read_bytes()),
                runtime_config=RUNTIME, pack=pack, pack_sha256=sha(pack.read_bytes()), exemplar_config=CONFIG,
                max_items=10, max_input_bytes=10000, timeout_seconds=180)
            with mock.patch.object(evaluate_native_exemplars.node_review, 'audit', return_value=(audit, records)):
                report = evaluate_native_exemplars.run(args)
            self.assertTrue(report['candidateProcessObserved'])
            self.assertEqual(report['admission']['holdoutAssessment'], 'not_performed')
            self.assertFalse(report['qualificationEstablished'])

    def test_actual_native_process_on_simulated_review_records(self):
        snapshot_sha = sha(SNAPSHOT.read_bytes())
        texts = ['run command', 'fixture unknown']
        choices = ['tool_call', 'chat']
        records = [{'review': {'itemSha256': str(i + 1) * 64, 'split': 'validation', 'choice': choices[i]},
                    'item': {'text': text, 'inputSha256': sha(text.encode())}} for i, text in enumerate(texts)]
        artifact = Path(ARTIFACT)
        pack = Path(PACK)
        pack_value = json.loads(pack.read_bytes())
        audit = pack_value['admission']
        for exemplar in pack_value['exemplars']:
            records.append({'review': {'itemSha256': exemplar['itemSha256'], 'split': 'train', 'choice': exemplar['choice']},
                            'item': {'text': 'test-only training input'}})
        args = argparse.Namespace(input=[], snapshot=SNAPSHOT, policy=POLICY, node='operation',
            binary=Path(BINARY), artifact=artifact, manifest_sha256=sha((artifact / 'manifest.json').read_bytes()),
            runtime_config=RUNTIME, pack=pack, pack_sha256=sha(pack.read_bytes()), exemplar_config=CONFIG,
            max_items=10, max_input_bytes=10000, timeout_seconds=180)
        with mock.patch.object(evaluate_native_exemplars.node_review, 'audit', return_value=(audit, records)):
            report = evaluate_native_exemplars.run(args)
        self.assertTrue(report['candidateProcessObserved'])
        self.assertFalse(report['candidateAuthenticated'])
        self.assertFalse(report['qualificationEstablished'])
        self.assertEqual(report['metrics']['baseline']['correct'], 1)
        self.assertEqual(report['metrics']['baseline']['escalated'], 1)
        self.assertEqual(report['metrics']['fallbackOnEscalation']['correct'], 1)
        self.assertNotIn('fixture unknown', json.dumps(report))
        changed = list(records)
        changed[-1] = {'review': dict(changed[-1]['review'], choice='chat'), 'item': changed[-1]['item']}
        with mock.patch.object(evaluate_native_exemplars.node_review, 'audit', return_value=(audit, changed)), \
             mock.patch.object(evaluate_native_exemplars.node_evaluate, 'predictions') as baseline:
            with self.assertRaisesRegex(ValueError, 'audited training inputs'):
                evaluate_native_exemplars.run(args)
            baseline.assert_not_called()


if __name__ == '__main__':
    unittest.main()
