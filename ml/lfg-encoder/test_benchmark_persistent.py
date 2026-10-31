import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import benchmark_persistent as benchmark


class PersistentBenchmarkTests(unittest.TestCase):
    def test_nearest_rank_percentiles_and_empty_samples(self):
        result = benchmark.summary(list(range(1, 101)))
        self.assertEqual(result['median_ns'], 50.5)
        self.assertEqual(result['p95_ns'], 95)
        self.assertEqual(result['p99_ns'], 99)
        with self.assertRaises(ValueError):
            benchmark.summary([])

    def test_tokenization_metadata_is_validated_and_missing_is_unknown(self):
        self.assertEqual(benchmark.tokenization_status({}), 'unknown')
        self.assertEqual(benchmark.tokenization_status({'tokenization': {'processed_tokens': 2,
                         'overflow_chunks': 0, 'truncated': False}}), 'complete')
        self.assertEqual(benchmark.tokenization_status({'tokenization': {'processed_tokens': 2,
                         'overflow_chunks': 1, 'truncated': True}}), 'truncated')
        for value in ({'processed_tokens': 2, 'overflow_chunks': 1, 'truncated': False},
                      {'processed_tokens': True, 'overflow_chunks': 0, 'truncated': False},
                      {'processed_tokens': 2, 'overflow_chunks': 0, 'truncated': 0}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'tokenization'):
                benchmark.tokenization_status({'tokenization': value})

    def test_config_and_input_limits_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((Path(__file__).parent / 'benchmark-persistent.v1.json').read_text())
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            loaded, _ = benchmark.load_config(path)
            data = root / 'examples.jsonl'
            data.write_text(json.dumps({'id': 'one', 'text': 'fixture'}) + '\n')
            self.assertEqual(len(benchmark.load_examples(data, loaded)[0]), 1)
            data.write_text((json.dumps({'id': 'one', 'text': 'fixture'}) + '\n') * 2)
            with self.assertRaisesRegex(ValueError, 'IDs'):
                benchmark.load_examples(data, loaded)
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                benchmark.load_examples(data, {**loaded, 'max_dataset_bytes': 1})
            for field, invalid in (('timeout_seconds', 0), ('max_records', True), ('measured_rounds', -1)):
                path.write_text(json.dumps({**config, field: invalid}))
                with self.assertRaises(ValueError):
                    benchmark.load_config(path)

    def test_persistent_subprocesses_match_requests_and_report_no_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((Path(__file__).parent / 'benchmark-persistent.v1.json').read_text())
            config.update(warmup_rounds=1, measured_rounds=2, max_records=2, timeout_seconds=10)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            data = root / 'examples.jsonl'
            data.write_text('\n'.join(json.dumps({'id': str(i), 'text': f'private fixture {i}'})
                                      for i in range(2)) + '\n', encoding='utf-8')
            fixture = root / 'predictor.py'
            fixture.write_text('''import hashlib,json,sys
from pathlib import Path
producer=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
for line in sys.stdin:
    x=json.loads(line)
    result={'schema_version':'lfg.evaluation-prediction.v1','id':x['id'],
      'input_sha256':hashlib.sha256(x['text'].encode()).hexdigest(),'producer_sha256':producer,
      'model':{'id':sys.argv[1],'version':'1','sha256':'a'*64},'intents':[],'capabilities':[],'risks':[]}
    if sys.argv[1]=='candidate':
      truncated=x['text'].endswith('1')
      result['tokenization']={'processed_tokens':2,'overflow_chunks':int(truncated),'truncated':truncated}
    print(json.dumps(result),flush=True)
''', encoding='utf-8')
            result = benchmark.run(Path(sys.executable), data, config_path,
                                   ['-u', str(fixture), 'baseline'], ['-u', str(fixture), 'candidate'])
            self.assertEqual(result['results']['baseline']['latency']['count'], 4)
            self.assertEqual(result['results']['candidate']['latency']['count'], 4)
            self.assertEqual(result['results']['candidate']['model']['id'], 'candidate')
            self.assertEqual(result['results']['baseline']['tokenization'], {'complete': 0, 'truncated': 0, 'unknown': 4})
            self.assertEqual(result['results']['candidate']['tokenization'], {'complete': 2, 'truncated': 2, 'unknown': 0})
            self.assertFalse(result['promotion_eligible'])
            self.assertNotIn('private fixture', json.dumps(result))
            self.assertEqual(result['dataset_sha256'], hashlib.sha256(data.read_bytes()).hexdigest())
            self.assertEqual(result['benchmark_sha256'], hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest())
            self.assertEqual(result['counter_reader_sha256'],
                             hashlib.sha256(Path(benchmark.benchmark_process.__file__).read_bytes()).hexdigest())

    def test_mismatched_prediction_digest_is_rejected(self):
        code = '''import json,sys
for line in sys.stdin:
 x=json.loads(line)
 print(json.dumps({'schema_version':'lfg.evaluation-prediction.v1','id':x['id'],
  'input_sha256':'0'*64,'producer_sha256':'a'*64,'model':{},'intents':[],
  'capabilities':[],'risks':[]}),flush=True)
'''
        predictor = benchmark.Predictor(Path(sys.executable), ['-u', '-c', code],
                                        {'max_response_bytes': 1024, 'timeout_seconds': 5})
        try:
            with self.assertRaisesRegex(ValueError, 'does not match'):
                predictor.request('fixture', 'the text')
        finally:
            predictor.close()

    def test_timeout_covers_blocked_input_and_stops_only_spawned_process(self):
        config = {'max_response_bytes': 1024, 'timeout_seconds': 1}
        predictor = benchmark.Predictor(Path(sys.executable), ['-c', 'import time; time.sleep(30)'], config)
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(ValueError, 'timed out'):
                predictor.request('fixture', 'x' * (1024 * 1024))
        finally:
            predictor.close()
        self.assertIsNotNone(predictor.process.poll())
        self.assertLess(time.monotonic() - started, 10)


if __name__ == '__main__':
    unittest.main()
