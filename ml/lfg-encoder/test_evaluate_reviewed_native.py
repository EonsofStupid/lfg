import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_persistent
import corpus_audit
import evaluate_reviewed_native as native
import predict_model


class ReviewedNativeEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.taxonomy = self.root / 'taxonomy.json'
        self.taxonomy.write_text(json.dumps({'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
                                              'intents': ['build'], 'capabilities': ['write'], 'risks': ['sensitive']}))
        self.plan = self.root / 'plan.json'
        self.sources = []
        self.rows = []
        for split in ('train', 'validation', 'test'):
            row = {'id': split, 'family': split, 'text': f'private {split}', 'verified': False,
                   'intents': ['build'], 'capabilities': ['write'], 'risks': [], 'source': {'project': split}}
            self.rows.append(row)
            path = self.root / f'{split}.jsonl'
            path.write_text(json.dumps(row) + '\n')
            self.sources.append({'path': path.name, 'project': split, 'split': split})
        plan = {'schema_version': corpus_audit.PLAN_SCHEMA, 'taxonomy': 'taxonomy.json',
                'review_ledger': None, 'sources': self.sources}
        self.plan.write_text(json.dumps(plan))
        _, queue = corpus_audit.audit(self.plan)
        ledger = [{**row, 'decision': 'approved', 'reviewer': 'fixture-reviewer',
                   'reviewed_at': '2026-10-31T00:00:00Z'} for row in queue]
        (self.root / 'ledger.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in ledger))
        for row, source in zip(self.rows, self.sources, strict=True):
            row['verified'] = True
            (self.root / source['path']).write_text(json.dumps(row) + '\n')
        plan['review_ledger'] = 'ledger.jsonl'
        self.plan.write_text(json.dumps(plan))
        audit, _ = corpus_audit.audit(self.plan)
        digest = hashlib.sha256((self.root / 'train.jsonl').read_bytes()).hexdigest()
        fields = ('schema_version', 'plan_sha256', 'taxonomy_sha256', 'review_ledger_sha256',
                  'auditor_sha256', 'taxonomy_loader_sha256', 'sources', 'counts', 'duplicate_within_split')
        self.model = self.root / 'model'
        self.model.mkdir()
        manifest = {'schema_version': 'lfg.encoder-run.v1', 'development_override': False, 'dataset_sha256': digest,
                    'environment_evidence': {'lock_sha256': 'a' * 64, 'installed_versions_sha256': 'b' * 64,
                                             'locked_wheel_payload': {'schema_version': 'lfg.installed-wheel-payload.v1',
                                                'files': 3, 'bytes': 50, 'payload_sha256': 'f' * 64,
                                                'excluded_wheel_records': 1, 'excluded_wheel_scripts': 0}},
                    'model_snapshot_evidence': {'manifest_sha256': 'c' * 64},
                    'training_inputs_sha256': 'd' * 64, 'wheelhouse_verifier_sha256': 'e' * 64,
                    'split': {'development_override': False, 'validation_source': 'audited-validation-partition',
                              'dataset_sha256': digest, 'corpus_audit': {k: audit[k] for k in fields},
                              'train_ids': ['train'], 'validation_ids': ['validation'],
                              'train_families': ['train'], 'validation_families': ['validation']}}
        (self.model / 'run-manifest.json').write_text(json.dumps(manifest))
        (self.model / 'labels.json').write_text(json.dumps(['build', 'write', 'sensitive']))
        self.artifact = self.root / 'artifact'
        self.artifact.mkdir()
        (self.artifact / 'model.onnx').write_bytes(b'fixture-not-a-real-onnx')
        (self.artifact / 'labels.json').write_bytes((self.model / 'labels.json').read_bytes())
        (self.artifact / 'taxonomy.json').write_bytes(self.taxonomy.read_bytes())
        onnx = {'schema_version': 'lfg.encoder-onnx-artifact.v2',
                'source_artifact_sha256': predict_model.artifact_digest(self.model),
                'taxonomy_sha256': audit['taxonomy_sha256'],
                'files_sha256': {name: native.file_hash(self.artifact / name)
                                 for name in ('model.onnx', 'labels.json', 'taxonomy.json')}}
        (self.artifact / 'manifest.json').write_text(json.dumps(onnx))
        self.binary = self.root / 'binary'
        self.binary.write_bytes(b'fixture-not-a-real-binary')
        self.runtime = self.root / 'runtime.json'
        self.runtime.write_text('{}')
        self.config = Path(benchmark_persistent.__file__).with_name('benchmark-persistent.v1.json')
        self.audit = audit

    def compare(self):
        return native.compare(self.plan, self.root / 'train.jsonl', self.model, self.artifact,
                              self.binary, self.runtime, self.config, 'embedded')

    def fake_predictor(self, binary, arguments, config):
        binary_hash = native.file_hash(binary)
        if arguments[0] == 'baseline-predict':
            identity = {'id': 'fixture-rule', 'version': '1', 'sha256': '1' * 64}
            emitted = []
        else:
            components = {'artifact': native.file_hash(self.artifact / 'manifest.json'), 'runtime': binary_hash,
                          'taxonomy': self.audit['taxonomy_sha256'], 'config': native.file_hash(self.runtime)}
            encoded = json.dumps(components, sort_keys=True, separators=(',', ':')).encode()
            identity = {'id': 'fixture-native', 'version': '1', 'sha256': hashlib.sha256(encoded).hexdigest(), **components}
            emitted = ['build']

        class Predictor:
            producer = binary_hash
            model = identity
            process = SimpleNamespace(poll=lambda: None)
            def request_prediction(self, identifier, text):
                value = {'id': identifier, 'input_sha256': hashlib.sha256(text.encode()).hexdigest(),
                         'intents': emitted, 'capabilities': [], 'risks': []}
                if arguments[0] != 'baseline-predict':
                    value['tokenization'] = {'processed_tokens': 4, 'overflow_chunks': 0, 'truncated': False}
                if '--include-scores' in arguments:
                    value['scores'] = {'semantics': 'sigmoid-uncalibrated-v1',
                                       'values': {'build': 0.9, 'write': 0.1, 'sensitive': 0.1}}
                return 0, value
            def close(self):
                pass
        return Predictor()

    def test_compares_reviewed_test_without_reporting_text_or_claiming_promotion(self):
        with patch.object(benchmark_persistent, 'Predictor', side_effect=self.fake_predictor):
            report = self.compare()
        self.assertEqual(report['examples'], 1)
        self.assertFalse(report['promotion_eligible'])
        self.assertIsNone(report['calibration'])
        self.assertEqual(report['groups']['candidate']['intents']['labels']['build']['tp'], 1)
        self.assertEqual(report['groups']['baseline']['intents']['labels']['build']['fn'], 1)
        self.assertEqual(report['test_label_support']['risks']['sensitive'], {'positive': 0, 'negative': 1})
        self.assertNotIn('private test', json.dumps(report))

    def test_rejects_truncated_or_unknown_candidate_coverage(self):
        for coverage in (None, {'processed_tokens': 4, 'overflow_chunks': 1, 'truncated': True}):
            def predictor(binary, arguments, config):
                instance = self.fake_predictor(binary, arguments, config)
                original = instance.request_prediction
                def request(identifier, text):
                    elapsed, value = original(identifier, text)
                    if arguments[0] != 'baseline-predict':
                        if coverage is None:
                            value.pop('tokenization', None)
                        else:
                            value['tokenization'] = coverage
                    return elapsed, value
                instance.request_prediction = request
                return instance
            with self.subTest(coverage=coverage), patch.object(benchmark_persistent, 'Predictor', side_effect=predictor):
                with self.assertRaisesRegex(ValueError, 'truncated or coverage is unknown'):
                    self.compare()

    def test_rejects_unreviewed_changed_or_development_inputs(self):
        row = self.rows[2]
        row['text'] = 'changed after review'
        (self.root / 'test.jsonl').write_text(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ValueError, 'test example requires'):
            self.compare()
        (self.root / 'test.jsonl').write_text(json.dumps(self.rows[2] | {'text': 'private test'}) + '\n')
        manifest_path = self.model / 'run-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['development_override'] = True
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'not bound'):
            self.compare()

    def test_rejects_missing_wheel_payload_evidence(self):
        manifest_path = self.model / 'run-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['environment_evidence']['locked_wheel_payload'] = None
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'not bound'):
            self.compare()

    def test_rejects_changed_artifact_and_wrong_binary_identity(self):
        (self.artifact / 'model.onnx').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            self.compare()
        (self.artifact / 'model.onnx').write_bytes(b'fixture-not-a-real-onnx')
        def wrong(*args):
            predictor = self.fake_predictor(*args)
            predictor.producer = 'f' * 64
            return predictor
        with patch.object(benchmark_persistent, 'Predictor', side_effect=wrong):
            with self.assertRaisesRegex(ValueError, 'producer differs'):
                self.compare()

    def test_rejects_incomplete_label_head_and_invalid_native_digest(self):
        (self.model / 'labels.json').write_text(json.dumps(['build', 'write']))
        with self.assertRaisesRegex(ValueError, 'complete evaluation taxonomy'):
            self.compare()
        (self.model / 'labels.json').write_text(json.dumps(['build', 'write', 'sensitive']))
        def wrong(*args):
            predictor = self.fake_predictor(*args)
            if 'artifact' in predictor.model:
                predictor.model = {**predictor.model, 'sha256': '0' * 64}
            return predictor
        with patch.object(benchmark_persistent, 'Predictor', side_effect=wrong):
            with self.assertRaisesRegex(ValueError, 'identity digest'):
                self.compare()

    def test_optional_calibration_uses_native_scores_and_stays_ineligible(self):
        prediction = self.artifact / 'prediction-config.json'
        prediction.write_text(json.dumps({'schema_version': 'lfg.encoder-prediction-config.v1',
                                          'default_threshold': 0.5, 'label_thresholds': {},
                                          'batch_size': 1, 'cpu_threads': 1, 'max_input_bytes': 1024}))
        artifact_manifest = self.artifact / 'manifest.json'
        value = json.loads(artifact_manifest.read_text())
        value['prediction_config_sha256'] = native.file_hash(prediction)
        value['files_sha256']['prediction-config.json'] = native.file_hash(prediction)
        artifact_manifest.write_text(json.dumps(value))
        config = Path(native.__file__).with_name('calibration.v1.json')
        with patch.object(benchmark_persistent, 'Predictor', side_effect=self.fake_predictor):
            report = native.compare(self.plan, self.root / 'train.jsonl', self.model, self.artifact,
                                    self.binary, self.runtime, self.config, 'embedded', config)
        self.assertFalse(report['promotion_eligible'])
        self.assertEqual(report['calibration']['score_semantics'], 'sigmoid-uncalibrated-v1')
        self.assertAlmostEqual(report['calibration']['labels']['build']['ece'], 0.1)
        self.assertAlmostEqual(report['calibration']['labels']['write']['ece'], 0.9)
        self.assertIsNone(report['performance_qualification'])

    def test_rejects_reordered_artifact_labels_even_when_artifact_hash_is_updated(self):
        labels = self.artifact / 'labels.json'
        labels.write_text(json.dumps(['write', 'build', 'sensitive']))
        path = self.artifact / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['files_sha256']['labels.json'] = native.file_hash(labels)
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'label order'):
            self.compare()


if __name__ == '__main__':
    unittest.main()
