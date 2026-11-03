"""Exercises the comparison job with the real collector and a fake model server."""
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

import compare_local_models as subject

COLLECTOR_AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'safetensors'))
LEVELS = {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3, 'extreme': 3}
SYSTEM = 'Assess the following user request for routing; do not perform the request.'
FAKE_SERVER = r'''
import argparse, json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
parser = argparse.ArgumentParser()
for name in ('--weights', '--host', '--port', '--alias'):
    parser.add_argument(name)
parser.add_argument('--answers')
args = parser.parse_args()
answers = json.load(open(args.answers, encoding='utf-8'))
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass
    def reply(self, status, raw, kind='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def authorized(self):
        return self.headers.get('Authorization') == 'Bearer ' + os.environ['LLAMA_API_KEY']
    def do_GET(self):
        if self.path == '/health':
            return self.reply(200, b'{"status":"ok"}')
        if self.path == '/metrics' and self.authorized():
            return self.reply(200, b'llamacpp:tokens_predicted_total 100\nllamacpp:tokens_predicted_seconds_total 4\n'
                                   b'llamacpp:prompt_tokens_total 50\nllamacpp:prompt_seconds_total 1\n# other\n', 'text/plain')
        self.reply(401, b'{}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if not self.authorized() or body['messages'][0]['content'] != answers['system']:
            return self.reply(401, b'{}')
        content = answers['by_request'][body['messages'][1]['content']]
        self.reply(200, json.dumps({'model': args.alias, 'choices': [{'finish_reason': 'stop',
                   'message': {'content': content}}]}).encode())
HTTPServer(('127.0.0.1', int(args.port)), Handler).serve_forever()
'''


def teacher_row(number, complexity, agents):
    return {'id': f'request-{number}', 'messages': [
        {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': f'Private request number {number}.'},
        {'role': 'assistant', 'content': json.dumps({'intent': 'x', 'complexity': complexity, 'suggested_models': [],
                                                     'agent_count': agents, 'reasons': ['teacher']})}],
        'metadata': {'project': 'alpha', 'project_group': 'related', 'original_source_sha256_claim': 'a' * 64}}


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


class ScoringTests(unittest.TestCase):
    def test_scores_agreement_levels_and_agent_differences(self):
        reference = [{'id': 'a', 'answer': {'complexity': 'High', 'agent_count': 1}},
                     {'id': 'b', 'answer': {'complexity': 'low', 'agent_count': 3}},
                     {'id': 'c', 'answer': {'complexity': 'extreme', 'agent_count': 2}},
                     {'id': 'd', 'answer': {'complexity': 'medium', 'agent_count': 1}}]
        answers = {'a': {'complexity': 'high', 'agent_count': 1, 'suggested_models': []},
                   'b': {'complexity': 'high', 'agent_count': 1, 'suggested_models': ['x']},
                   'c': {'complexity': 'very high', 'agent_count': '2', 'suggested_models': []}}
        result = subject.score_model(reference, answers, LEVELS)
        self.assertEqual(result['complexity'], {'compared': 3, 'exact': 2, 'within_one_level': 2, 'unknown_labels': 0})
        self.assertEqual(result['agent_count'], {'compared': 2, 'exact': 1, 'invalid': 1, 'mean_absolute_difference': 1.0})
        self.assertEqual(result['suggested_models_empty'], 2)

    def test_reads_only_llama_metrics(self):
        text = '# HELP other\nllamacpp:tokens_predicted_total 12\nother 5\nllamacpp:rate{slot="0"} 1.5\nllamacpp:bad nan!\n'
        self.assertEqual(subject.parse_metrics(text), {'tokens_predicted_total': 12.0, 'rate': 1.5})

    def test_reads_vllm_metrics_and_adds_labelled_lines(self):
        text = ('vllm:generation_tokens_total{model_name="m"} 90\nvllm:request_decode_time_seconds_sum{model_name="m"} 3\n'
                'vllm:time_to_first_token_seconds_sum{model_name="m"} 2\nvllm:time_to_first_token_seconds_count{model_name="m"} 4\n'
                'vllm:other{kind="a"} 1\nvllm:other{kind="b"} 2\nvllm:not_finite nan\n')
        metrics = subject.parse_metrics(text)
        self.assertEqual(metrics['other'], 3.0)
        self.assertNotIn('not_finite', metrics)
        result = subject.speed([{'answered': 2, 'elapsed_seconds': 8, 'startup_seconds': 1, 'metrics': metrics,
                                 'available_memory_before_bytes': None, 'available_memory_ready_bytes': None}])
        self.assertEqual(result['generated_tokens_per_second'], 30.0)
        self.assertEqual(result['mean_time_to_first_token_seconds'], 0.5)
        self.assertEqual(result['seconds_per_answer'], 4.0)

    def test_directory_digest_ignores_hidden_download_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model.safetensors').write_bytes(b'weights')
            first = subject.weights_digest(root)
            (root / '.cache').mkdir()
            (root / '.cache' / 'lock').write_bytes(b'changes')
            self.assertEqual(subject.weights_digest(root), first)
            (root / 'model.safetensors').write_bytes(b'different')
            self.assertNotEqual(subject.weights_digest(root), first)

    def test_speed_is_empty_without_finished_segments(self):
        result = subject.speed([])
        self.assertIsNone(result['generated_tokens_per_second'])
        self.assertEqual(result['finished_segments'], 0)


class ConfigurationTests(unittest.TestCase):
    def test_rejects_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'config.json'
            good = {'schema_version': 'lfg.local-model-comparison.v1', 'port': 1,
                    'server_command': ['server', '{weights}', '127.0.0.1', '{port}', '{id}'],
                    'startup_timeout_seconds': 1, 'collection': {}, 'complexity_levels': LEVELS,
                    'models': [{'id': 'one', 'weights': str(Path(temporary).resolve() / 'w'), 'server_args': []}],
                    **{key: str(Path(temporary).resolve() / key) for key in
                       ('python', 'collector', 'policy', 'reference_dataset', 'output')}}
            path.write_text(json.dumps(good), encoding='utf-8')
            self.assertEqual(subject.load_config(path)['port'], 1)
            bad_models = [{'id': '../escape', 'weights': good['models'][0]['weights'], 'server_args': []}]
            for changes in ({'schema_version': 'x'}, {'port': 0}, {'server_command': []}, {'models': []},
                            {'server_command': ['server', '{weights}', '0.0.0.0', '{port}', '{id}']},
                            {'python': 'relative'}, {'models': bad_models}, {'models': good['models'] * 2},
                            {'complexity_levels': {'low': 'one'}}):
                path.write_text(json.dumps({**good, **changes}), encoding='utf-8')
                with self.assertRaises(ValueError, msg=changes):
                    subject.load_config(path)


@unittest.skipUnless(COLLECTOR_AVAILABLE, 'torch and safetensors required')
class ComparisonJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        rows = [teacher_row(0, 'high', 1), teacher_row(1, 'High', 2), teacher_row(2, 'low', 1), teacher_row(3, 'medium', 3)]
        self.dataset = self.root / 'train.jsonl'
        self.dataset.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        answer = lambda complexity, agents: json.dumps({'intent': 'y', 'complexity': complexity, 'suggested_models': [],
                                                        'agent_count': agents, 'reasons': ['model']})
        self.answers = self.root / 'answers.json'
        self.answers.write_text(json.dumps({'system': SYSTEM, 'by_request': {
            'Private request number 0.': answer('high', 1), 'Private request number 1.': answer('medium', 2),
            'Private request number 2.': answer('low', 2), 'Private request number 3.': 'not json'}}), encoding='utf-8')
        (self.root / 'fake_server.py').write_text(FAKE_SERVER, encoding='utf-8')
        (self.root / 'weights.gguf').write_bytes(b'weights')
        policy = self.root / 'policy.json'
        policy.write_text(json.dumps({'schema_version': 'lfg.private-corpus-screen.v1', 'revision': 'fixture',
            'limits': {'max_policy_bytes': 65536, 'max_plan_bytes': 65536, 'max_source_bytes': 100000,
                       'max_total_source_bytes': 100000, 'max_total_output_bytes': 100000, 'max_records': 100,
                       'max_text_bytes': 10000, 'max_matches_per_record': 100},
            'rules': [{'id': 'test-secret', 'pattern': r'fixture-secret-[0-9]+', 'replacement': '[REDACTED]'}]}), encoding='utf-8')
        self.config_value = {
            'schema_version': 'lfg.local-model-comparison.v1', 'python': sys.executable,
            'collector': str(Path(subject.__file__).resolve().parent / 'collect_local_teacher.py'), 'policy': str(policy),
            'reference_dataset': str(self.dataset), 'output': str(self.root / 'output'), 'port': free_port(),
            'server_command': [sys.executable, str(self.root / 'fake_server.py'), '--weights', '{weights}', '--host',
                               '127.0.0.1', '--port', '{port}', '--alias', '{id}'], 'startup_timeout_seconds': 30,
            'complexity_levels': LEVELS,
            'collection': {'schema_version': 'lfg.local-teacher-collection.v1', 'source_origin': 'owner_chat',
                           'required_fields': ['intent', 'complexity', 'suggested_models', 'agent_count', 'reasons'],
                           'json_mode': True, 'temperature': 0.0, 'max_tokens': 64, 'timeout_seconds': 10,
                           'max_requests': 10, 'max_request_bytes': 4096, 'max_http_response_bytes': 65536,
                           'max_response_bytes': 8192, 'max_manifest_bytes': 1048576, 'on_incomplete': 'skip'},
            'models': [{'id': 'small-model', 'weights': str(self.root / 'weights.gguf'),
                        'server_args': ['--answers', str(self.answers)]}]}
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps(self.config_value), encoding='utf-8')

    def test_runs_real_collector_and_scores_against_teacher(self):
        report = subject.experiment(self.config)
        result = report['models']['small-model']
        self.assertTrue(result['finished'])
        self.assertEqual((result['answered'], result['failed']), (3, 1))
        self.assertEqual(result['complexity'], {'compared': 3, 'exact': 2, 'within_one_level': 3, 'unknown_labels': 0})
        self.assertEqual(result['agent_count']['exact'], 2)
        self.assertEqual(result['speed']['generated_tokens_per_second'], 25.0)
        self.assertEqual(report['always_most_common']['complexity'], {'value': 2, 'share': 0.5})
        self.assertFalse(report['assessment_quality_evaluated'])
        self.assertEqual(report['reference_kind'], 'unreviewed_teacher_proposals')
        self.assertTrue((self.root / 'output' / 'complete.json').exists())
        written = (self.root / 'output' / 'comparison.json').read_text(encoding='utf-8')
        self.assertNotIn('Private request', written)
        identity = json.loads((self.root / 'output/models/small-model/collection/manifest.json').read_text(encoding='utf-8'))
        self.assertTrue(identity['teacher_identity'].startswith('small-model@sha256:'))

    def test_instruction_suffix_reaches_the_model(self):
        self.config_value['instruction_suffix'] = ' Use low, medium or high.'
        self.config.write_text(json.dumps(self.config_value), encoding='utf-8')
        answers = json.loads(self.answers.read_text(encoding='utf-8'))
        answers['system'] = SYSTEM + ' Use low, medium or high.'
        self.answers.write_text(json.dumps(answers), encoding='utf-8')
        report = subject.experiment(self.config)
        self.assertEqual(report['models']['small-model']['answered'], 3)
        written = json.loads((self.root / 'output/models/small-model/collection.json').read_text(encoding='utf-8'))
        self.assertEqual(written['system_prompt'], SYSTEM + ' Use low, medium or high.')

    def test_finished_model_is_not_started_again(self):
        subject.experiment(self.config)
        first = (self.root / 'output/models/small-model/segments.json').read_text(encoding='utf-8')
        report = subject.experiment(self.config)
        self.assertEqual(report['models']['small-model']['answered'], 3)
        self.assertEqual((self.root / 'output/models/small-model/segments.json').read_text(encoding='utf-8'), first)

    def test_interrupted_collection_continues_from_saved_answers(self):
        healthy = self.answers.read_text(encoding='utf-8')
        broken = json.loads(healthy)
        del broken['by_request']['Private request number 2.']
        self.answers.write_text(json.dumps(broken), encoding='utf-8')
        with self.assertRaises(subprocess.CalledProcessError):
            subject.experiment(self.config)
        collection = self.root / 'output/models/small-model/collection'
        self.assertEqual(subject.count_files(collection / 'records'), 2)
        self.assertFalse((collection / 'manifest.json').exists())
        self.answers.write_text(healthy, encoding='utf-8')
        report = subject.experiment(self.config)
        self.assertEqual(report['models']['small-model']['answered'], 3)
        self.assertEqual(json.loads((self.root / 'output/models/small-model/segments.json').read_text(encoding='utf-8'))[-1]['answered'], 1)

    def test_rejects_output_from_a_different_configuration(self):
        subject.experiment(self.config)
        self.config_value['port'] = free_port()
        self.config.write_text(json.dumps(self.config_value), encoding='utf-8')
        with self.assertRaises(ValueError):
            subject.experiment(self.config)


if __name__ == '__main__':
    unittest.main()
