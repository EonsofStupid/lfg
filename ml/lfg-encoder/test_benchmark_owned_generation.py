"""Benchmark software fixtures; not task-quality evidence."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import benchmark_owned_generation as subject


class GenerationBenchmarkTests(unittest.TestCase):
    def test_response_validation_rejects_wrong_identity_and_invalid_output(self):
        probe = subject.Generator.__new__(subject.Generator)
        probe.max_new_bytes = 2
        good = {'schema_version': 'lfg.owned-byte-generation.v1', 'id': 'fixture',
                'input_bytes': 1, 'generated_bytes': [65, 66], 'stop_reason': 'byte_limit'}
        probe.validate_response(good, 'fixture', 'x')
        for change in ({'id': 'other'}, {'generated_bytes': [256]}, {'generated_bytes': [65]},
                       {'input_bytes': 2}, {'stop_reason': 'other'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                probe.validate_response({**good, **change}, 'fixture', 'x')

    @unittest.skipUnless(os.environ.get('LFG_OWNED_BYTE_TEST_BINARY'), 'native test executable not supplied')
    def test_persistent_native_benchmark_counts_actual_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config_path = root / 'data.jsonl', root / 'config.json'
            dataset.write_text(json.dumps({'id': 'fixture', 'text': 'sample fixture'}) + '\n')
            config = {'schema_version': 'lfg.persistent-benchmark-config.v1', 'warmup_rounds': 1,
                      'measured_rounds': 2, 'max_records': 2, 'max_dataset_bytes': 1024,
                      'max_response_bytes': 4096, 'timeout_seconds': 30, 'future_field': True}
            config_path.write_text(json.dumps(config))
            result = subject.run(Path(os.environ['LFG_OWNED_BYTE_TEST_BINARY']), dataset, config_path, 2)
            self.assertEqual(result['latency']['count'], 2)
            self.assertEqual(result['stop_reasons'], {'byte_limit': 2})
            self.assertFalse(result['assessment_quality_evaluated'])
            self.assertFalse(result['deterministic_baseline_compared'])


if __name__ == '__main__':
    unittest.main()
