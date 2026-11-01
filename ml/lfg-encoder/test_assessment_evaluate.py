"""Tests use fixture-only attestations and predictions; no model is trained."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import assessment_evaluate as subject
import assessment_targets
import corpus_audit


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class AssessmentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        taxonomy = {'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
                    'intents': ['build'], 'capabilities': ['write'], 'risks': ['sensitive']}
        (self.root / 'taxonomy.json').write_text(json.dumps(taxonomy))
        profile = {'project': 'validation', 'complexities': [{'id': 'c1', 'label': 'one'}, {'id': 'c2', 'label': 'two'}],
                   'models': [{'id': mid, 'label': mid, 'provider': 'fixture', 'model': mid,
                               'subscription_id': mid, 'availability': availability, 'effort_levels': ['e1', 'e2']}
                              for mid, availability in [('m1', 'available'), ('m2', 'available'),
                                                        ('m3', 'available'), ('m4', 'unknown')]],
                   'agent_range': {'min': 0, 'max': 4}}
        def row(identifier, project, values='absent'):
            result = {'id': identifier, 'family': identifier, 'text': 'private ' + identifier, 'verified': False,
                      'intents': ['build'], 'capabilities': ['write'], 'risks': [], 'source': {'project': project}}
            if values != 'absent':
                result['assessment_targets'] = assessment_targets.make_record(values, profile, 'fixture', 'a' * 64)
            return result
        self.train = [row('train', 'train')]
        self.validation = [row('v1', 'validation', {'complexity': 'c1', 'models': [{'id': 'm1', 'effort': 'e1'},
                                                                                   {'id': 'm2', 'effort': None}], 'agent_count': 2}),
                           row('v2', 'validation', {'complexity': None, 'models': None, 'agent_count': 0}),
                           row('v3', 'validation')]
        self.sources = [{'path': 'train.jsonl', 'project': 'train', 'split': 'train'},
                        {'path': 'validation.jsonl', 'project': 'validation', 'split': 'validation'}]
        self.write_rows()
        self.plan = self.root / 'plan.json'
        self.plan_value = {'schema_version': corpus_audit.DEVELOPMENT_PLAN_SCHEMA, 'taxonomy': 'taxonomy.json',
                           'review_ledger': None, 'sources': self.sources}
        self.plan.write_text(json.dumps(self.plan_value))
        _, queue = corpus_audit.audit(self.plan, development_only=True)
        self.ledger = [{**item, 'decision': 'approved', 'reviewer': 'fixture-reviewer',
                        'reviewed_at': '2026-11-01T00:00:00Z'} for item in queue]
        (self.root / 'ledger.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in self.ledger))
        for item in self.train + self.validation:
            item['verified'] = True
        self.write_rows()
        self.plan_value['review_ledger'] = 'ledger.jsonl'
        self.plan.write_text(json.dumps(self.plan_value))
        self.audit, self.records = subject.validation_records(self.plan)
        self.baseline = self.root / 'baseline.jsonl'
        self.candidate = self.root / 'candidate.jsonl'
        unknown = {'complexity': None, 'models': None, 'agent_count': None}
        self.baseline_rows = self.predictions([
            {'complexity': 'c2', 'models': [{'id': 'm1', 'effort': 'e2'}, {'id': 'm3', 'effort': None}], 'agent_count': 3},
            {**unknown, 'agent_count': 0}, unknown], 'b')
        self.candidate_rows = self.predictions([
            {'complexity': 'c1', 'models': [{'id': 'm1', 'effort': 'e1'}, {'id': 'm2', 'effort': None}], 'agent_count': 2},
            unknown, unknown], 'c')
        self.write_predictions(self.baseline, self.baseline_rows)
        self.write_predictions(self.candidate, self.candidate_rows)

    def write_rows(self):
        for name, rows in [('train.jsonl', self.train), ('validation.jsonl', self.validation)]:
            (self.root / name).write_text(''.join(json.dumps(item) + '\n' for item in rows))

    def predictions(self, values, character):
        return [{'schema_version': 'lfg.assessment-prediction.v1', 'id': record['row']['id'],
                 'input_sha256': digest(record['row']['text'].encode()), 'annotation_sha256': record['annotation_sha256'],
                 'profile_sha256': record['row'].get('assessment_targets', {}).get('profile_sha256'),
                 'producer_sha256': character * 64, 'artifact_sha256': character * 64, 'values': value}
                for record, value in zip(self.records, values, strict=True)]

    def write_predictions(self, path, values):
        path.write_text(''.join(json.dumps(value) + '\n' for value in values))

    def test_comparison_masks_unknowns_and_keeps_zero_distinct(self):
        result = subject.evaluate(self.plan, self.baseline, self.candidate, 100000)
        self.assertFalse(result['prediction_execution_verified'])
        self.assertFalse(result['qualification_established'])
        self.assertNotIn('private v1', json.dumps(result))
        baseline, candidate = result['metrics']['baseline'], result['metrics']['candidate']
        self.assertEqual((baseline['complexity']['known'], baseline['complexity']['unknown'], baseline['complexity']['correct']), (1, 2, 0))
        self.assertEqual((baseline['agent_count']['known'], baseline['agent_count']['predicted'], baseline['agent_count']['exact']), (2, 2, 1))
        self.assertEqual(baseline['agent_count']['mean_absolute_error'], .5)
        self.assertEqual((baseline['models']['tp'], baseline['models']['fp'], baseline['models']['fn']), (1, 1, 1))
        self.assertEqual((baseline['effort']['known'], baseline['effort']['unknown_for_expected_models'], baseline['effort']['correct']), (1, 1, 0))
        self.assertEqual(candidate['agent_count']['coverage'], .5)
        self.assertEqual(candidate['models']['recall_including_abstentions'], 1.0)
        self.assertEqual(candidate['effort']['correct'], 1)

    def test_rejects_wrong_or_incomplete_project_bound_predictions(self):
        for mutation in ('profile', 'annotation', 'input', 'unavailable', 'unprofiled', 'missing', 'extra', 'producer'):
            rows = copy.deepcopy(self.candidate_rows)
            if mutation == 'profile': rows[0]['profile_sha256'] = 'f' * 64
            if mutation == 'annotation': rows[0]['annotation_sha256'] = 'f' * 64
            if mutation == 'input': rows[0]['input_sha256'] = 'f' * 64
            if mutation == 'unavailable': rows[0]['values']['models'] = [{'id': 'm4', 'effort': None}]
            if mutation == 'unprofiled': rows[2]['values']['complexity'] = 'c1'
            if mutation == 'missing': rows.pop()
            if mutation == 'extra': rows.append(copy.deepcopy(rows[0]))
            if mutation == 'producer': rows[0]['producer_sha256'] = 'f' * 64
            self.write_predictions(self.candidate, rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                subject.evaluate(self.plan, self.baseline, self.candidate, 100000)
        self.write_predictions(self.candidate, self.candidate_rows)
        with self.assertRaisesRegex(ValueError, 'limit'):
            subject.evaluate(self.plan, self.baseline, self.candidate, 1)

    def test_rejects_unreviewed_records_and_test_scope_before_audit(self):
        self.ledger[1]['decision'] = 'pending'
        (self.root / 'ledger.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in self.ledger))
        with self.assertRaisesRegex(ValueError, 'review attestation'):
            subject.evaluate(self.plan, self.baseline, self.candidate, 100000)
        plan = copy.deepcopy(self.plan_value)
        plan['sources'].append({'path': 'DO_NOT_OPEN', 'project': 'test', 'split': 'test'})
        self.plan.write_text(json.dumps(plan))
        with patch.object(corpus_audit, 'audit', side_effect=AssertionError('source opened')):
            with self.assertRaisesRegex(ValueError, 'development plan'):
                subject.validation_records(self.plan)

    def test_pending_labels_cannot_be_hidden_by_an_approval(self):
        self.train[0].update(intents=[], capabilities=[], risks=[], labels_pending=True)
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'pending labels'):
            subject.evaluate(self.plan, self.baseline, self.candidate, 100000)

    def test_all_unknown_targets_cannot_be_scored(self):
        for row in self.validation:
            if row.get('assessment_targets'):
                row['assessment_targets']['values'] = {'complexity': None, 'models': None, 'agent_count': None}
            entry = next(item for item in self.ledger if item['id'] == row['id'])
            entry['annotation_sha256'] = corpus_audit.annotation_digest(row, 'validation', 'validation', self.audit['taxonomy_sha256'])
        self.write_rows()
        (self.root / 'ledger.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in self.ledger))
        with self.assertRaisesRegex(ValueError, 'no known assessment targets'):
            subject.evaluate(self.plan, self.baseline, self.candidate, 100000)


if __name__ == '__main__':
    unittest.main()
