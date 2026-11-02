"""Fixture-only tests; these are not reviewed user data or a trained release."""
import copy
import importlib.util
import hashlib
import json
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import assessment_targets
import assessment_train as subject
import corpus_audit
import test_assessment_evaluate as fixtures


def profile():
    return {'project': 'fixture', 'complexities': [{'id': 'a', 'label': 'Alpha'}, {'id': 'b', 'label': 'Beta'}],
            'models': [{'id': mid, 'label': mid, 'provider': 'operator-provider', 'model': mid,
                        'subscription_id': 'private-seat-' + mid, 'availability': availability,
                        'effort_levels': ['light', 'deep']}
                       for mid, availability in [('one', 'available'), ('two', 'available'), ('three', 'unknown')]],
            'agent_range': {'min': 0, 'max': 5}}


class AssessmentTrainingTests(unittest.TestCase):
    def setUp(self):
        self.config_path = Path(subject.__file__).with_name('assessment-training-config.example.json')
        self.config, _ = subject.load_config(self.config_path)

    def row(self, values, p=None):
        p = profile() if p is None else p
        return {'text': 'fixture request', 'source': {'project': p['project']},
                'assessment_targets': assessment_targets.make_record(values, p, 'fixture', 'a' * 64)}

    def test_known_candidates_are_project_conditioned_and_unknowns_are_not_labels(self):
        rows = [self.row({'complexity': 'a', 'models': [{'id': 'one', 'effort': 'deep'},
                                                        {'id': 'two', 'effort': None}], 'agent_count': 0}),
                self.row({'complexity': None, 'models': None, 'agent_count': None})]
        examples, support = subject.project_examples(rows, self.config)
        self.assertEqual([pair[3] for pair in examples['complexity']], [1, 0])
        self.assertEqual([pair[3] for pair in examples['models']], [1, 1])
        self.assertEqual([pair[3] for pair in examples['effort']], [0, 1])
        self.assertEqual([pair[3] for pair in examples['agent_count']], [0])
        self.assertEqual(support['effort']['records'], 1)
        self.assertNotIn('three', ''.join(pair[2] for pair in examples['models']))
        self.assertNotIn('private-seat-', str(examples))

    def test_duplicate_descriptions_fail_unless_explicit_binding_ids_distinguish_them(self):
        p = profile()
        p['models'][1].update(label=p['models'][0]['label'], provider=p['models'][0]['provider'], model=p['models'][0]['model'])
        row = self.row({'complexity': 'a', 'models': [{'id': 'one', 'effort': 'light'}], 'agent_count': 1}, p)
        with self.assertRaisesRegex(ValueError, 'indistinguishable'):
            subject.project_examples([row], self.config)
        config = {**self.config, 'include_subscription_ids': True}
        examples, _ = subject.project_examples([row], config)
        self.assertIn('private-seat-one', examples['models'][0][2])

    def test_limits_and_explicit_device_configuration_are_validated(self):
        row = self.row({'complexity': 'a', 'models': [{'id': 'one', 'effort': 'light'}], 'agent_count': 1})
        with self.assertRaisesRegex(ValueError, 'pair limit'):
            subject.project_examples([row], {**self.config, 'max_pairs_per_split': 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            bad = copy.deepcopy(self.config)
            bad['device'] = 'auto'
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'configuration'):
                subject.load_config(path)

    def test_unreviewed_or_wrong_scope_plan_fails_before_model_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'schema_version': corpus_audit.PLAN_SCHEMA,
                                        'sources': [{'path': 'DO_NOT_OPEN', 'split': 'test'}]}))
            args = SimpleNamespace(plan=plan, config=self.config_path, model=root / 'missing-model',
                model_manifest=root / 'missing-manifest', revision='a' * 40, environment_lock=root / 'missing-lock',
                wheelhouse=root / 'missing-wheels', output=root / 'no-output', allowed_extra=[])
            with patch.object(subject.training_inputs, 'verify_model', side_effect=AssertionError('model accessed')):
                with self.assertRaisesRegex(ValueError, 'development plan'):
                    subject.build(args)
            self.assertFalse(args.output.exists())

    def test_exploratory_rejects_test_before_source_open_and_requires_targets(self):
        fixture = fixtures.AssessmentEvaluationTests('test_comparison_masks_unknowns_and_keeps_zero_distinct')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.plan_value['review_ledger'] = None
        fixture.plan_value['sources'] = [{'path': 'missing-heldout', 'project': 'heldout', 'split': 'test'}]
        fixture.plan.write_text(json.dumps(fixture.plan_value))
        with self.assertRaisesRegex(ValueError, 'development plan'):
            subject.exploratory_inputs(fixture.plan, self.config)
        fixture.plan_value['sources'] = [fixture.sources[0]]
        fixture.plan.write_text(json.dumps(fixture.plan_value))
        with self.assertRaisesRegex(ValueError, 'explicit assessment target'):
            subject.exploratory_inputs(fixture.plan, self.config)

    def test_exact_private_output_file_must_be_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / '.gitignore').write_text('other.json\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'must be ignored'):
                subject.require_private_file(root / 'pack.json')
            (root / '.gitignore').write_text('pack.json\n', encoding='utf-8')
            subject.require_private_file(root / 'pack.json')
            (root / 'pack.json').write_text('fixture', encoding='utf-8')
            subprocess.run(['git', '-C', str(root), 'add', '-f', 'pack.json'], check=True)
            (root / 'pack.json').unlink()
            with self.assertRaisesRegex(ValueError, 'already tracked'):
                subject.require_private_file(root / 'pack.json')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch is required for the numerical fit')
    def test_requested_cuda_does_not_fall_back_to_cpu(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'requested CUDA device is unavailable'):
                subject.fit({}, {}, {**self.config, 'device': 'cuda'})

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch is required for the numerical fit')
    def test_ridge_fit_uses_candidate_interactions_and_enforces_memory_limit(self):
        vectors = {'q1': [1.0, 0.0], 'q2': [0.0, 1.0], 'ctx': [0.5, 0.5],
                   'one': [1.0, 0.0], 'two': [0.0, 1.0]}
        pairs = [('q1', 'ctx', 'one', 1), ('q1', 'ctx', 'two', 0),
                 ('q2', 'ctx', 'one', 0), ('q2', 'ctx', 'two', 1)]
        examples = {'complexity': pairs, 'models': pairs, 'effort': pairs,
                    'agent_count': [('q1', 'ctx', None, 0), ('q2', 'ctx', None, 3)]}
        config = {**self.config, 'ridge': 0.001}
        learned = subject.fit(examples, vectors, config)
        def dot(query, candidate, weights):
            q, p, c = vectors[query], vectors['ctx'], vectors[candidate]
            feature = q + p + c + [a * b for a, b in zip(q, c)] + [1.0]
            return sum(a * b for a, b in zip(feature, weights))
        self.assertGreater(dot('q1', 'one', learned['weights']['models']), dot('q1', 'two', learned['weights']['models']))
        self.assertGreater(dot('q2', 'two', learned['weights']['models']), dot('q2', 'one', learned['weights']['models']))
        self.assertEqual(learned, subject.fit(examples, vectors, config))
        with self.assertRaisesRegex(ValueError, 'tensor estimate'):
            subject.fit(examples, vectors, {**config, 'max_fit_bytes': 1})

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch is required for the numerical fit')
    def test_unreviewed_partial_target_trains_without_validation_or_support_gate(self):
        fixture = fixtures.AssessmentEvaluationTests('test_comparison_masks_unknowns_and_keeps_zero_distinct')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        p = copy.deepcopy(fixture.validation[0]['assessment_targets']['project_profile'])
        p['project'] = 'train'
        fixture.train[0]['verified'] = False
        fixture.train[0]['assessment_targets'] = assessment_targets.make_record(
            {'complexity': None, 'models': None, 'agent_count': 2}, p, 'fixture', 'a' * 64)
        fixture.write_rows()
        fixture.plan_value['review_ledger'] = None
        fixture.plan_value['sources'] = [fixture.sources[0]]
        fixture.plan.write_text(json.dumps(fixture.plan_value))
        observed = []
        class FixtureEncoder:
            def __init__(self, _model, _config):
                self.versions = {'cublas_workspace_config': __import__('os').environ.get('CUBLAS_WORKSPACE_CONFIG'),
                                 'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES')}
            def encode(self, texts):
                observed.extend(texts)
                return {text: [(byte + 1) / 256 for byte in hashlib.sha256(text.encode()).digest()[:4]] for text in texts}
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(plan=fixture.plan, config=self.config_path, model=Path(directory) / 'model',
                model_manifest=Path(directory) / 'manifest', revision='a' * 40, environment_lock=Path(directory) / 'lock',
                wheelhouse=Path(directory) / 'wheels', output=Path(directory) / 'pack.json', allowed_extra=[],
                exploratory=True)
            with patch.object(subject, 'Encoder', FixtureEncoder), \
                 patch.object(subject.training_inputs, 'verify_model', return_value={'fixture': True}), \
                 patch.object(subject.training_inputs, 'verify_environment', return_value={'fixture': True}):
                result = subject.build(args)
            pack = json.loads(args.output.read_bytes())
            self.assertEqual(result['trained_targets'], ['agent_count'])
            self.assertEqual(set(pack['weights']), {'agent_count'})
            self.assertEqual(pack['schema_version'], 'lfg.assessment-heads-exploratory.v1')
            self.assertFalse(pack['promotion_eligible'])
            self.assertFalse(pack['validation_used_for_training'])
            self.assertTrue(pack['support_failures'])
            self.assertEqual(pack['audit']['counts']['train']['ready_for_reviewed_subset'], 0)
            self.assertIn('private train', observed)
            self.assertNotIn('private train', json.dumps(pack))
            observed.clear()
            fixture.plan_value['sources'] = fixture.sources
            fixture.plan.write_text(json.dumps(fixture.plan_value))
            args.output = Path(directory) / 'pack-with-validation.json'
            with patch.object(subject, 'Encoder', FixtureEncoder), \
                 patch.object(subject.training_inputs, 'verify_model', return_value={'fixture': True}), \
                 patch.object(subject.training_inputs, 'verify_environment', return_value={'fixture': True}):
                subject.build(args)
            second = json.loads(args.output.read_bytes())
            self.assertEqual(pack['weights'], second['weights'])
            self.assertFalse(any(text.startswith('private v') for text in observed))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch is required for the numerical fit')
    def test_reviewed_fixture_build_uses_training_text_only(self):
        fixture = fixtures.AssessmentEvaluationTests('test_comparison_masks_unknowns_and_keeps_zero_distinct')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        training_profile = copy.deepcopy(fixture.validation[0]['assessment_targets']['project_profile'])
        training_profile['project'] = 'train'
        train_values = {'complexity': 'c1', 'models': [{'id': 'm1', 'effort': 'e1'}, {'id': 'm2', 'effort': None}], 'agent_count': 2}
        fixture.train[0]['assessment_targets'] = assessment_targets.make_record(train_values, training_profile, 'fixture', 'a' * 64)
        fixture.write_rows()
        next(item for item in fixture.ledger if item['id'] == 'train')['annotation_sha256'] = corpus_audit.annotation_digest(
            fixture.train[0], 'train', 'train', fixture.audit['taxonomy_sha256'])
        (fixture.root / 'ledger.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in fixture.ledger))
        observed = []
        class FixtureEncoder:
            def __init__(self, _model, _config):
                self.versions = {'cublas_workspace_config': __import__('os').environ.get('CUBLAS_WORKSPACE_CONFIG'),
                                 'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES')}
            def encode(self, texts):
                observed.extend(texts)
                return {text: [(byte + 1) / 256 for byte in hashlib.sha256(text.encode()).digest()[:4]] for text in texts}
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(plan=fixture.plan, config=self.config_path, model=Path(directory) / 'model',
                model_manifest=Path(directory) / 'manifest', revision='a' * 40, environment_lock=Path(directory) / 'lock',
                wheelhouse=Path(directory) / 'wheels', output=Path(directory) / 'pack.json', allowed_extra=[])
            with patch.object(subject, 'Encoder', FixtureEncoder), \
                 patch.object(subject.training_inputs, 'verify_model', return_value={'fixture': True}), \
                 patch.object(subject.training_inputs, 'verify_environment', return_value={'fixture': True}):
                result = subject.build(args)
            pack = json.loads(args.output.read_bytes())
            self.assertFalse(result['promotion_eligible'])
            self.assertEqual(set(pack['weights']), set(subject.GROUPS))
            self.assertIn('private train', observed)
            self.assertFalse(any(text.startswith('private v') for text in observed))
            self.assertNotIn('private train', json.dumps(pack))

            # Alter reviewed validation judgments while leaving the training data intact.
            validation = fixture.validation[0]
            revised = {'complexity': 'c2', 'models': [{'id': 'm1', 'effort': 'e2'}], 'agent_count': 3}
            validation['assessment_targets'] = assessment_targets.make_record(
                revised, validation['assessment_targets']['project_profile'], 'fixture', 'a' * 64)
            fixture.write_rows()
            next(item for item in fixture.ledger if item['id'] == validation['id'])['annotation_sha256'] = (
                corpus_audit.annotation_digest(validation, 'validation', 'validation', fixture.audit['taxonomy_sha256']))
            (fixture.root / 'ledger.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in fixture.ledger))
            args.output = Path(directory) / 'second-pack.json'
            with patch.object(subject, 'Encoder', FixtureEncoder), \
                 patch.object(subject.training_inputs, 'verify_model', return_value={'fixture': True}), \
                 patch.object(subject.training_inputs, 'verify_environment', return_value={'fixture': True}):
                subject.build(args)
            second_pack = json.loads(args.output.read_bytes())
            self.assertEqual(pack['weights'], second_pack['weights'])


if __name__ == '__main__':
    unittest.main()
