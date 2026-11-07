import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import assessment_targets as targets


def profile():
    return {'project': 'fixture-project', 'complexities': [{'id': 'c1', 'label': 'First'}],
            'models': [{'id': 'm1', 'label': 'Model one', 'provider': 'provider', 'model': 'model',
                        'subscription_id': 'subscription', 'availability': 'available', 'effort_levels': ['e1']},
                       {'id': 'm2', 'label': 'Model two', 'provider': 'local', 'model': 'local-model',
                        'subscription_id': None, 'availability': 'unknown', 'effort_levels': []}],
            'agent_range': {'min': 0, 'max': 4}}


class AssessmentTargetTests(unittest.TestCase):
    def test_unknown_is_distinct_from_zero_and_profiles_are_bound(self):
        p = profile()
        values = {'complexity': None, 'models': None, 'agent_count': 0}
        record = targets.make_record(values, p, 'r1', 'a' * 64)
        self.assertIsNone(record['values']['complexity'])
        self.assertIsNone(record['values']['models'])
        self.assertEqual(record['values']['agent_count'], 0)
        self.assertEqual(targets.coverage(values), {'complexity': 'unknown', 'models': 'unknown',
                                                    'agent_count': 'known', 'effort': 'unknown'})
        self.assertEqual(targets.coverage({'complexity': None, 'agent_count': None,
             'models': [{'id': 'a', 'effort': 'one'}, {'id': 'b', 'effort': None}]})['effort'], 'partial')
        targets.validate_stored(record, 'fixture-project')
        with self.assertRaisesRegex(ValueError, 'project or digest'):
            targets.validate_stored(record, 'different-project')
        changed = copy.deepcopy(record)
        changed['project_profile']['complexities'][0]['label'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'project or digest'):
            targets.validate_stored(changed, 'fixture-project')

    def test_selections_must_match_declared_project_profile(self):
        p = profile()
        valid = {'complexity': 'c1', 'models': [{'id': 'm1', 'effort': 'e1'}], 'agent_count': 2}
        targets.validate_stored(targets.make_record(valid, p, 'r1', 'a' * 64), 'fixture-project')
        for change in ({'complexity': 'other'}, {'models': []}, {'models': [{'id': 'm2', 'effort': None}]},
                       {'models': [{'id': 'm1', 'effort': 'other'}]}, {'agent_count': True}, {'agent_count': 5}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                targets.make_record({**valid, **change}, p, 'r1', 'a' * 64)

    def test_config_rejects_duplicates_and_hashes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.json'
            value = {'schema_version': 'lfg.assessment-review-config.v1', 'revision': 'r1', 'projects': [profile()]}
            raw = json.dumps(value).encode()
            path.write_bytes(raw)
            loaded, sha = targets.load_config(path)
            self.assertEqual(sha, hashlib.sha256(raw).hexdigest())
            self.assertEqual(targets.load_config(path, len(raw))[1], sha)
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                targets.load_config(path, len(raw) - 1)
            self.assertEqual(targets.project_profile(loaded, 'fixture-project'), profile())
            value['projects'].append(copy.deepcopy(value['projects'][0]))
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                targets.load_config(path)


if __name__ == '__main__':
    unittest.main()
