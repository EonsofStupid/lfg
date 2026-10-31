import json
import tempfile
import unittest
from pathlib import Path

import calibration


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = {'intents': {'build'}, 'capabilities': {'write'}, 'risks': {'sensitive'}}
        self.decision = {'default_threshold': 0.5, 'label_thresholds': {'write': 0.3}}
        self.rows = [
            {'id': 'one', 'intents': ['build'], 'capabilities': ['write'], 'risks': []},
            {'id': 'two', 'intents': ['build'], 'capabilities': [], 'risks': ['sensitive']},
        ]
        self.predictions = {
            'one': {'intents': ['build'], 'capabilities': ['write'], 'risks': [],
                    'scores': {'semantics': 'sigmoid-uncalibrated-v1',
                               'values': {'build': 0.8, 'write': 0.4, 'sensitive': 0.1}}},
            'two': {'intents': ['build'], 'capabilities': [], 'risks': ['sensitive'],
                    'scores': {'semantics': 'sigmoid-uncalibrated-v1',
                               'values': {'build': 0.6, 'write': 0.2, 'sensitive': 0.7}}},
        }

    def test_binary_label_ece_and_brier_use_configured_bins_and_support(self):
        result = calibration.calculate(self.rows, self.predictions, self.taxonomy, self.decision, [0.0, 0.5, 1.0])
        self.assertAlmostEqual(result['labels']['build']['ece'], 0.3)
        self.assertAlmostEqual(result['labels']['build']['brier'], 0.1)
        self.assertAlmostEqual(result['labels']['write']['ece'], 0.2)
        self.assertAlmostEqual(result['labels']['sensitive']['ece'], 0.2)
        self.assertAlmostEqual(result['macro_label_ece'], 0.7 / 3)
        self.assertAlmostEqual(result['micro_label_observation']['ece'], 0.2)
        self.assertEqual(result['micro_label_observation']['observations'], 6)

    def test_missing_nonfinite_or_threshold_inconsistent_scores_are_rejected(self):
        scores = self.predictions['one']['scores']['values']
        for invalid in (float('nan'), float('inf'), -0.1, 1.1, True):
            scores['build'] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                calibration.calculate(self.rows, self.predictions, self.taxonomy, self.decision, [0, 0.5, 1])
        scores['build'] = 0.8
        scores.pop('write')
        with self.assertRaisesRegex(ValueError, 'complete taxonomy'):
            calibration.calculate(self.rows, self.predictions, self.taxonomy, self.decision, [0, 0.5, 1])
        scores['write'] = 0.4
        self.predictions['one']['capabilities'] = []
        with self.assertRaisesRegex(ValueError, 'thresholds disagree'):
            calibration.calculate(self.rows, self.predictions, self.taxonomy, self.decision, [0, 0.5, 1])

    def test_bins_are_configurable_and_include_zero_and_one(self):
        accumulator = calibration.Accumulator([0.0, 0.5, 1.0])
        for score in (0.0, 0.5, 1.0):
            accumulator.add(score, 0)
        self.assertEqual([item['count'] for item in accumulator.result()['bins']], [1, 2])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            for invalid in ([0, 0.5, 0.5, 1], [0, 0.2], [0, True, 1], [0, float('nan'), 1]):
                path.write_text(json.dumps({'schema_version': 'lfg.calibration-config.v1', 'bin_edges': invalid}))
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    calibration.load_config(path)


if __name__ == '__main__':
    unittest.main()
