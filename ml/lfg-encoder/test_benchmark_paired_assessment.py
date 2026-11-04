import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import getpass
import subprocess
import sys
import tempfile
import threading
import time
import unittest


MODULE_PATH = Path(__file__).with_name('benchmark_paired_assessment.py')
SPEC = importlib.util.spec_from_file_location('benchmark_paired_assessment', MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


TOKEN = 'test-token-with-sufficient-entropy-0123456789'
ENGINE = {'id': 'fixture-engine', 'version': 'fixture-v1', 'sha256': 'a' * 64}
CONFIG_DIGEST = 'b' * 64


class FixtureHandler(BaseHTTPRequestHandler):
    requests = []
    get_requests = []
    sleep_owned = False
    bad_owned_token = False
    malformed_owned = False

    def log_message(self, _format, *_args):
        pass

    def respond(self, status, value):
        raw = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_GET(self):
        self.get_requests.append(self.path)
        if self.path == '/redirect':
            self.send_response(302)
            self.send_header('Location', f'http://127.0.0.1:{self.server.server_port}/status')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if self.path == '/status':
            return self.respond(200, {'phase': 'ready', 'operationsDigest': 'c' * 64,
                                      'encoder': ENGINE})
        if self.path == '/api/lfg/assess/owned/status':
            if self.headers.get('X-LFG-Session-Token') != TOKEN:
                return self.respond(401, {'error': 'unauthorized'})
            return self.respond(200, {'enabled': True, 'status': 'ready', 'engine': ENGINE,
                                      'limits': {'timeoutMs': 100}})
        return self.respond(404, {})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        try:
            value = json.loads(body)
        except Exception:
            return self.respond(400, {})
        self.requests.append((self.path, value, self.headers.get('X-LFG-Session-Token')))
        if self.path == '/api/lfg/assess':
            return self.respond(200, {'schemaVersion': 'lfg.assessment.v1',
                                      'sessionId': value['sessionId'], 'revision': value['revision'],
                                      'durationMicros': 7, 'configRevision': 'fixture-config',
                                      'configDigest': CONFIG_DIGEST})
        if self.path == '/api/lfg/assess/owned':
            if self.headers.get('X-LFG-Session-Token') != TOKEN:
                return self.respond(401, {'error': 'unauthorized'})
            if self.sleep_owned:
                time.sleep(float(self.sleep_owned))
            return self.respond(200, {'schemaVersion': ('invalid' if self.malformed_owned else 'lfg.owned-assessment.v1'), 'status': 'ok',
                                      'engine': ENGINE,
                                      'measurement': {'schemaVersion': 'lfg.owned-run-measurement.v1',
                                                      'status': 'ok', 'engineId': ENGINE['id'],
                                                      'handlerElapsedMs': 3},
                                      'outputText': 'private fixture output'})
        return self.respond(404, {})


class PairedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        FixtureHandler.requests = []
        FixtureHandler.get_requests = []
        FixtureHandler.sleep_owned = False
        FixtureHandler.bad_owned_token = False
        FixtureHandler.malformed_owned = False
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def make_config(self, *, timeout=1.0, rounds=2, warmups=1, max_response=4096):
        dataset = self.root / 'train.jsonl'
        dataset.write_text(json.dumps({'id': 'fixture-1', 'text': 'same multibyte text: café PRIVATE123'}) + '\n',
                           encoding='utf-8')
        policy = {
            'schema_version': 'lfg.private-corpus-screen.v1', 'revision': 'fixture-policy-v1',
            'limits': {'max_policy_bytes': 65536, 'max_plan_bytes': 1000,
                       'max_source_bytes': 1000, 'max_total_source_bytes': 1000,
                       'max_total_output_bytes': 1000, 'max_records': 10,
                       'max_text_bytes': 1000, 'max_matches_per_record': 10},
            'rules': [{'id': 'fixture-marker', 'pattern': 'PRIVATE[0-9]+',
                       'replacement': '[REDACTED]'}],
        }
        policy_path = self.root / 'screening-policy.json'
        policy_path.write_text(json.dumps(policy), encoding='utf-8')
        config = {
            'schema_version': benchmark.SCHEMA,
            'target_base_url': f'http://127.0.0.1:{self.server.server_port}',
            'owned_token_env': 'LFG_TEST_OWNED_TOKEN', 'owned_token_file': None,
            'dataset_path': str(dataset), 'dataset_split': 'train', 'dataset_source_ref': 'fixture:train',
            'screening_policy_path': str(policy_path),
            'selection_seed': 42, 'max_records': 5, 'max_source_records': 10,
            'max_dataset_bytes': 10000, 'max_input_bytes': 1000, 'max_request_bytes': 4000,
            'max_response_bytes': max_response, 'max_report_bytes': 100000,
            'client_timeout_seconds': timeout, 'server_timeout_overhead_ms': 200,
            'warmup_rounds': warmups, 'measured_rounds': rounds,
            'declared_subscriptions': [], 'first_order': 'deterministic_first',
        }
        path = self.root / 'config.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        os.environ['LFG_TEST_OWNED_TOKEN'] = TOKEN
        return path, config

    def test_loopback_url_requires_literal_loopback_without_redirect_components(self):
        self.assertTrue(benchmark.validate_loopback_url('http://[::1]:8080/') == 'http://[::1]:8080')
        for value in ('https://127.0.0.1', 'http://localhost:8080', 'http://0.0.0.0:3',
                      'http://127.0.0.1:8080/path', 'http://127.0.0.1:8080/?x=1',
                      'http://user@127.0.0.1:8080'):
            with self.subTest(value=value), self.assertRaises(benchmark.BenchmarkError):
                benchmark.validate_loopback_url(value)

    def test_config_extensions_are_inert_and_configured_scale_is_not_clamped(self):
        path, config = self.make_config()
        config['max_records'] = 10001
        config['max_source_records'] = 10002
        config['extension_metadata'] = {'ignored': True}
        path.write_text(json.dumps(config), encoding='utf-8')
        loaded, _, _ = benchmark.load_config(path)
        self.assertEqual(loaded['max_records'], 10001)
        self.assertNotIn('extension_metadata', loaded)
        config['dataset_split'] = 'development'
        path.write_text(json.dumps(config), encoding='utf-8')
        loaded, _, _ = benchmark.load_config(path)
        self.assertEqual(loaded['dataset_split'], 'development')
        config['dataset_split'] = 'validation'
        path.write_text(json.dumps(config), encoding='utf-8')
        with self.assertRaisesRegex(benchmark.BenchmarkError, 'development_split_required'):
            benchmark.load_config(path)
        path.write_bytes(b' ' * (benchmark.MAX_CONFIG_BYTES + 1))
        with self.assertRaisesRegex(benchmark.BenchmarkError, 'config_size_limit'):
            benchmark.load_config(path)

    def test_success_uses_identical_text_without_context_and_redacts_response(self):
        path, _ = self.make_config()
        report = benchmark.run_benchmark(path)
        self.assertTrue(report['complete'])
        self.assertEqual(report['source']['split'], 'train')
        self.assertEqual(report['measurement_scope']['cpu'], None)
        self.assertEqual(report['source']['screening_changes'],
                         {'changed_records': 1, 'redaction_counts': {'fixture-marker': 1}})
        self.assertNotEqual(report['source']['source_selected_id_text_digest'],
                            report['source']['screened_selected_id_text_digest'])
        self.assertEqual(report['paths']['deterministic']['attempts'], 3)
        self.assertEqual(report['paths']['owned']['attempts'], 3)
        measured = FixtureHandler.requests
        self.assertEqual([item[0] for item in measured[:6]], [
            '/api/lfg/assess', '/api/lfg/assess/owned',
            '/api/lfg/assess/owned', '/api/lfg/assess',
            '/api/lfg/assess', '/api/lfg/assess/owned',
        ])
        for pair_start in range(0, len(measured), 2):
            pair = {item[0]: item for item in measured[pair_start:pair_start + 2]}
            deterministic = pair['/api/lfg/assess']
            owned = pair['/api/lfg/assess/owned']
            self.assertEqual(deterministic[1]['text'], owned[1]['text'])
            self.assertNotIn('PRIVATE123', deterministic[1]['text'])
            self.assertEqual(set(owned[1]), {'text'})
            self.assertEqual(deterministic[1]['declaredSubscriptions'], [])
            self.assertIsNone(deterministic[2])
            self.assertEqual(owned[2], TOKEN)
        serialized = json.dumps(report)
        self.assertNotIn('café', serialized)
        self.assertNotIn('PRIVATE123', serialized)
        self.assertNotIn('private fixture output', serialized)
        self.assertNotIn(TOKEN, serialized)

    def test_timeout_aborts_remaining_requests_and_marks_contamination_unknown(self):
        FixtureHandler.sleep_owned = True
        path, _ = self.make_config(timeout=0.4, rounds=2, warmups=0)
        FixtureHandler.sleep_owned = 0.75
        report = benchmark.run_benchmark(path)
        self.assertFalse(report['complete'])
        self.assertEqual(report['abort_code'], 'client_timeout_contamination_unknown')
        self.assertEqual(report['contamination_state'], 'unknown; no further requests issued')
        self.assertEqual(len(FixtureHandler.requests), 2)

    def test_response_byte_limit_is_enforced_before_json_is_accepted(self):
        opener = benchmark.make_opener()
        result = benchmark.http_json(opener, f'http://127.0.0.1:{self.server.server_port}/status',
                                     method='GET', payload=None, token=None, timeout=1,
                                     max_request_bytes=128, max_response_bytes=16)
        self.assertEqual(result.response_code, 'response_too_large')
        self.assertIsNone(result.payload)

    def test_redirect_is_not_followed_and_request_limits_preflight_the_whole_pair(self):
        opener = benchmark.make_opener()
        redirect = benchmark.http_json(opener, f'http://127.0.0.1:{self.server.server_port}/redirect',
                                       method='GET', payload=None, token=None, timeout=1,
                                       max_request_bytes=128, max_response_bytes=1024)
        self.assertEqual(redirect.status, 302)
        self.assertEqual(FixtureHandler.get_requests, ['/redirect'])

        path, config = self.make_config(warmups=0, rounds=1)
        config['first_order'] = 'owned_first'
        config['max_request_bytes'] = 100
        path.write_text(json.dumps(config), encoding='utf-8')
        with self.assertRaisesRegex(benchmark.BenchmarkError, 'request_size_limit'):
            benchmark.run_benchmark(path)
        self.assertEqual(FixtureHandler.requests, [])

    def test_input_byte_limit_rejects_before_any_assessment_request(self):
        path, config = self.make_config(warmups=0, rounds=1)
        config['max_input_bytes'] = 1
        path.write_text(json.dumps(config), encoding='utf-8')
        with self.assertRaisesRegex(benchmark.BenchmarkError, 'input_size_limit'):
            benchmark.run_benchmark(path)
        self.assertEqual(FixtureHandler.requests, [])

    def test_malformed_owned_schema_is_counted_with_missing_handler_time(self):
        FixtureHandler.malformed_owned = True
        path, _ = self.make_config(warmups=0, rounds=2)
        report = benchmark.run_benchmark(path)
        owned = report['paths']['owned']
        self.assertEqual(owned['status_counts'], {'malformed_response': 2})
        self.assertEqual(owned['server_handler_time_missing'], 2)
        self.assertEqual(owned['server_handler_time_us']['denominator'], 2)

    def test_private_report_creation_is_exclusive_and_contains_no_fixture_text(self):
        path, _ = self.make_config(warmups=0, rounds=1)
        report = benchmark.run_benchmark(path)
        output_dir = self.root / 'private'
        output_dir.mkdir(mode=0o700)
        if os.name == 'posix':
            output_dir.chmod(0o700)
        elif os.name == 'nt':
            subprocess.run(['icacls.exe', str(output_dir), '/inheritance:r', '/grant:r',
                            f'{getpass.getuser()}:(OI)(CI)F'], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        output = output_dir / 'result.json'
        benchmark.write_report(output, report, 100000)
        raw = output.read_text(encoding='utf-8')
        self.assertNotIn('café', raw)
        self.assertNotIn('PRIVATE123', raw)
        self.assertNotIn('private fixture output', raw)
        self.assertNotIn(TOKEN, raw)
        with self.assertRaisesRegex(benchmark.BenchmarkError, 'report_already_exists'):
            benchmark.write_report(output, report, 100000)


if __name__ == '__main__':
    unittest.main()
