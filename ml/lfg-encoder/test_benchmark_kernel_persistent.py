import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark_kernel_persistent as benchmark


class KernelBenchmarkTests(unittest.TestCase):
    def test_response_is_bound_to_request_and_snapshot(self):
        predictor = object.__new__(benchmark.KernelPredictor)
        predictor.snapshot_sha = 'a' * 64
        response = {'schema_version': 'lfg.kernel-prediction.v1', 'id': 'one',
                    'input_sha256': hashlib.sha256(b'text').hexdigest(),
                    'result': {'schemaVersion': 'lfg.kernel-result.v1', 'snapshotSha256': 'a' * 64}}
        predictor.validate_response(response, 'one', 'text')
        for key, value in (('id', 'other'), ('input_sha256', 'b' * 64)):
            with self.assertRaisesRegex(ValueError, 'request'):
                predictor.validate_response({**response, key: value}, 'one', 'text')
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            predictor.validate_response({**response, 'result': {**response['result'], 'snapshotSha256': 'b' * 64}},
                                        'one', 'text')

    def test_repeated_input_result_drift_is_rejected(self):
        class UnstablePredictor:
            def __init__(self, *args):
                self.started = 0
                self.process = object()
                self.count = 0

            def request_prediction(self, identifier, text):
                self.count += 1
                return 1, {'result': {'state': 'escalate', 'atNode': 'operation', 'hops': [],
                                      'reason': str(self.count)}}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'input.jsonl'
            data.write_text(json.dumps({'id': 'one', 'text': 'test fixture'}) + '\n', encoding='utf-8')
            config = root / 'config.json'
            config.write_text(json.dumps({'schema_version': 'lfg.persistent-benchmark-config.v1',
                                          'warmup_rounds': 1, 'measured_rounds': 1, 'max_records': 1,
                                          'max_dataset_bytes': 10000, 'max_response_bytes': 65536,
                                          'timeout_seconds': 30}), encoding='utf-8')
            snapshot = root / 'snapshot.json'
            snapshot.write_text(json.dumps({'nodes': {'operation': {'kind': 'choice', 'options': {'a': {}}}}}),
                                encoding='utf-8')
            binary = root / 'binary'
            binary.write_bytes(b'test fixture')
            with mock.patch.object(benchmark, 'KernelPredictor', UnstablePredictor), \
                    mock.patch.object(benchmark.benchmark_process, 'windows_counters', return_value={}):
                with self.assertRaisesRegex(ValueError, 'result changed'):
                    benchmark.run(binary, data, config, snapshot, 'operation', 'id')

    @unittest.skipUnless(os.environ.get('LFG_TEST_KERNEL_BINARY'), 'requires an explicitly selected kernel binary')
    def test_actual_kernel_process_and_private_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'input.jsonl'
            data.write_text(json.dumps({'requestId': 'one', 'text': 'private test-only input'}) + '\n', encoding='utf-8')
            config = root / 'config.json'
            config.write_text(json.dumps({'schema_version': 'lfg.persistent-benchmark-config.v1',
                                          'warmup_rounds': 1, 'measured_rounds': 2, 'max_records': 5,
                                          'max_dataset_bytes': 10000, 'max_response_bytes': 65536,
                                          'timeout_seconds': 30}), encoding='utf-8')
            snapshot = Path(__file__).resolve().parents[2] / 'crates/lfg-core/config/kernel-core.v1.json'
            report = benchmark.run(Path(os.environ['LFG_TEST_KERNEL_BINARY']), data, config, snapshot,
                                   'operation', 'requestId')
            self.assertEqual(report['latency']['count'], 2)
            self.assertEqual(sum(value['count'] for value in report['latency_by_node_state'].values() if value), 2)
            self.assertNotIn('private test-only input', json.dumps(report))
            self.assertFalse(report['promotion_eligible'])


if __name__ == '__main__':
    unittest.main()
