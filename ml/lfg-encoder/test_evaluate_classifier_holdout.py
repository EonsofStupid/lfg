import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_persistent
import corpus_audit
import evaluate_classifier_holdout as holdout
import predict_model
import test_evaluate_reviewed_native


class ClassifierHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_evaluate_reviewed_native.ReviewedNativeEvaluationTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        plan = json.loads(f.plan.read_bytes())
        ledger = [json.loads(line) for line in (f.root / 'ledger.jsonl').read_text().splitlines()]
        (f.root / 'dev-reviews.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in ledger if x['split'] != 'test'))
        (f.root / 'test-reviews.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in ledger if x['split'] == 'test'))
        dev = {**plan, 'schema_version': corpus_audit.DEVELOPMENT_PLAN_SCHEMA,
               'review_ledger': 'dev-reviews.jsonl', 'sources': f.sources[:2]}
        test = {**plan, 'review_ledger': 'test-reviews.jsonl', 'sources': f.sources[2:]}
        f.plan.write_text(json.dumps(dev))
        self.test_plan = f.root / 'test-plan.json'
        self.test_plan.write_text(json.dumps(test))
        audit, _ = corpus_audit.audit(f.plan)
        fields = ('schema_version', 'plan_sha256', 'taxonomy_sha256', 'review_ledger_sha256',
                  'auditor_sha256', 'taxonomy_loader_sha256', 'assessment_targets_validator_sha256',
                  'sources', 'counts', 'duplicate_within_split')
        manifest_path = f.model / 'run-manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        manifest['split']['corpus_audit'] = {**{key: audit[key] for key in fields},
                                               'reviewed_assessment_targets': {s: audit['reviewed_assessment_targets'][s] for s in ('train', 'validation')},
                                               'scope': 'development', 'holdout_assessment': 'not_performed'}
        manifest_path.write_text(json.dumps(manifest))
        artifact_path = f.artifact / 'manifest.json'
        artifact = json.loads(artifact_path.read_bytes())
        artifact['source_artifact_sha256'] = predict_model.artifact_digest(f.model)
        artifact_path.write_text(json.dumps(artifact))
        self.commit_path = f.root / 'commitment.json'
        self.write_commitment()
        self.args = SimpleNamespace(plan=f.plan, training_dataset=f.root / 'train.jsonl', model=f.model,
                                    artifact=f.artifact, binary=f.binary, runtime_config=f.runtime, config=f.config,
                                    mode='embedded', calibration_config=None, commitment=self.commit_path,
                                    commitment_sha256=holdout.sha(self.commit_path.read_bytes()),
                                    holdout_plan=self.test_plan, freeze_sha256='f' * 64)

    def write_commitment(self):
        self.commit_path.write_bytes(holdout.encode(holdout.commitment(self.test_plan)) + b'\n')

    def test_freeze_and_final_use_separate_reviewed_sources(self):
        f = self.fixture
        with patch.object(benchmark_persistent, 'Predictor', side_effect=f.fake_predictor):
            with patch.object(holdout, 'commitment', side_effect=AssertionError('holdout opened')):
                frozen = holdout.freeze(self.args)
            result = holdout.final(self.args, frozen)
        self.assertFalse(frozen['qualification_established'])
        self.assertFalse(result['qualification_established'])
        self.assertEqual(frozen['validation']['examples'], 1)
        self.assertEqual(result['test']['examples'], 1)
        self.assertEqual(result['test']['groups']['candidate']['intents']['labels']['build']['tp'], 1)
        self.assertNotIn('private test', json.dumps(result))

    def test_changed_holdout_and_candidate_fail_before_test_prediction(self):
        f = self.fixture
        with patch.object(benchmark_persistent, 'Predictor', side_effect=f.fake_predictor):
            frozen = holdout.freeze(self.args)
            test_path = f.root / 'test.jsonl'
            original = test_path.read_bytes()
            test_path.write_bytes(original + b' ')
            with self.assertRaisesRegex(ValueError, 'holdout differs'):
                holdout.final(self.args, frozen)
            test_path.write_bytes(original)
            f.binary.write_bytes(b'changed candidate')
            with self.assertRaisesRegex(ValueError, 'frozen development'):
                holdout.final(self.args, frozen)

    def test_rejects_cross_split_overlap_even_with_new_commitment(self):
        f = self.fixture
        row = json.loads((f.root / 'test.jsonl').read_text())
        row['text'] = self.fixture.rows[0]['text']
        (f.root / 'test.jsonl').write_text(json.dumps(row) + '\n')
        path = f.root / 'test-reviews.jsonl'
        approval = json.loads(path.read_text())
        approval['annotation_sha256'] = corpus_audit.annotation_digest(row, 'test', 'test', self.fixture.audit['taxonomy_sha256'])
        path.write_text(json.dumps(approval) + '\n')
        self.write_commitment()
        self.args.commitment_sha256 = holdout.sha(self.commit_path.read_bytes())
        with patch.object(benchmark_persistent, 'Predictor', side_effect=f.fake_predictor):
            frozen = holdout.freeze(self.args)
            with self.assertRaisesRegex(ValueError, 'normalized text appears'):
                holdout.final(self.args, frozen)

    def test_commitment_rejects_unreviewed_or_non_test_source(self):
        plan = json.loads(self.test_plan.read_bytes())
        plan['sources'][0]['split'] = 'train'
        self.test_plan.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'only test'):
            holdout.commitment(self.test_plan)
        plan['sources'][0]['split'] = 'test'
        self.test_plan.write_text(json.dumps(plan))
        path = self.fixture.root / 'test-reviews.jsonl'
        row = json.loads(path.read_text())
        row['decision'] = 'pending'
        path.write_text(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ValueError, 'currently reviewed'):
            holdout.commitment(self.test_plan)

    def test_commitment_cli_writes_once_and_reports_its_digest(self):
        output = self.fixture.root / 'cli-commitment.json'
        command = [sys.executable, str(Path(holdout.__file__)), 'commit-holdout',
                   '--holdout-plan', str(self.test_plan), '--output', str(output)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        message = json.loads(result.stdout)
        self.assertEqual(message['output_sha256'], holdout.sha(output.read_bytes()))
        self.assertNotIn('private test', output.read_text())
        again = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(message['output_sha256'], holdout.sha(output.read_bytes()))


if __name__ == '__main__':
    unittest.main()
