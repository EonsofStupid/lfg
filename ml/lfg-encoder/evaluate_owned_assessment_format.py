#!/usr/bin/env python3
"""Compare output format, not assessment accuracy, on private development requests."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request

import benchmark_owned_generation as generation
import benchmark_persistent as persistent
import evaluate_owned_bytes as byte_eval
import screen_private_corpus as screen


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def config_file(path: Path) -> tuple[dict, bytes]:
    raw = screen._read(path, 65536, 'format_config_unavailable')
    config = screen._json(raw, 'invalid_format_config')
    integers = ('max_validation_bytes', 'max_manifest_bytes', 'max_records', 'min_prompt_bytes',
                'max_prompt_bytes', 'max_context', 'max_new_bytes', 'max_response_bytes',
                'timeout_seconds', 'startup_timeout_seconds')
    framing = ('system_prefix', 'system_suffix', 'user_prefix', 'user_suffix', 'assistant_prefix')
    if (not isinstance(config, dict) or config.get('schema_version') != 'lfg.owned-assessment-format-evaluation.v1'
            or any(type(config.get(k)) is not int or config[k] < 1 for k in integers)
            or config['min_prompt_bytes'] > config['max_prompt_bytes']
            or not isinstance(config.get('seed'), str) or not config['seed']
            or not isinstance(config.get('system_prompt'), str) or not config['system_prompt'].strip()
            or not isinstance(config.get('framing'), dict)
            or any(not isinstance(config['framing'].get(k), str) for k in framing)
            or not isinstance(config.get('required_fields'), list) or not config['required_fields']
            or any(not isinstance(k, str) or not k for k in config['required_fields'])
            or not isinstance(config.get('declared_subscriptions'), list)
            or any(not isinstance(item, str) or not item for item in config['declared_subscriptions'])
            or (config.get('stop_sequence') is not None and
                (not isinstance(config['stop_sequence'], str) or not config['stop_sequence']
                 or '\0' in config['stop_sequence'] or len(config['stop_sequence'].encode()) > config['max_new_bytes']))):
        raise ValueError('invalid format evaluation configuration')
    return config, raw


def select_validation(dataset: Path, policy_path: Path, config: dict) -> tuple[list[str], dict]:
    manifest_raw = screen._read(dataset / 'manifest.json', config['max_manifest_bytes'], 'validation_manifest_unavailable')
    manifest = screen._json(manifest_raw, 'invalid_validation_manifest')
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 'lfg.private-corpus-screen-manifest.v1'
            or not isinstance(manifest.get('outputs'), dict) or not isinstance(manifest['outputs'].get('validation'), str)):
        raise ValueError('validation source is not declared')
    raw = screen._read(dataset / 'validation.jsonl', config['max_validation_bytes'], 'validation_unavailable')
    if sha(raw) != manifest['outputs']['validation']:
        raise ValueError('validation digest mismatch')
    policy, rules, policy_sha = screen.load_policy(policy_path, config['max_manifest_bytes'])
    candidates, ids, seen = [], set(), set()
    for line in raw.splitlines():
        row = screen._json(line, 'invalid_validation_record')
        if (not isinstance(row, dict) or row.get('schema_version') != screen.OUTPUT_VERSION
                or row.get('split') != 'validation' or not isinstance(row.get('id'), str)
                or not row['id'] or row['id'] in ids or not isinstance(row.get('text'), str)
                or sha(row['text'].encode()) != row.get('screened_text_sha256')):
            raise ValueError('invalid validation record identity')
        ids.add(row['id'])
        text, _ = screen._screen(row['text'], rules, policy['limits'])
        size = len(text.encode())
        if not config['min_prompt_bytes'] <= size <= config['max_prompt_bytes'] or sha(text.encode()) in seen:
            continue
        seen.add(sha(text.encode()))
        rank = sha((config['seed'] + '\0' + row['id']).encode())
        candidates.append((rank, text))
    candidates.sort(key=lambda item: item[0])
    selected = [text for _, text in candidates[:config['max_records']]]
    if not selected:
        raise ValueError('no eligible validation requests')
    return selected, {'manifest_sha256': sha(manifest_raw), 'source_sha256': sha(raw), 'policy_sha256': policy_sha,
                      'selected': len(selected), 'source_records': len(ids)}


def prompt(text: str, config: dict) -> str:
    f = config['framing']
    result = (f['system_prefix'] + config['system_prompt'] + f['system_suffix'] + f['user_prefix']
              + text + f['user_suffix'] + f['assistant_prefix'])
    if len(result.encode()) > config['max_context']:
        raise ValueError('framed validation prompt exceeds context')
    return result


def response_format(raw: bytes, fields: list[str]) -> str:
    try:
        result = screen._json(raw, 'invalid_assessment_json')
    except (ValueError, UnicodeError):
        return 'invalid_json'
    if not isinstance(result, dict) or any(field not in result for field in fields):
        return 'missing_required_fields'
    return 'required_fields_present'


class FormatGenerator(generation.Generator):
    def __init__(self, binary: Path, config: dict, settings: dict):
        self.stop = config.get('stop_sequence')
        args = ['embedded-owned-byte-generate', '--experimental', '--max-new-bytes', str(config['max_new_bytes'])]
        if self.stop is not None:
            args += ['--stop', self.stop]
        super().__init__(binary, args, settings, config['max_new_bytes'])

    def validate_response(self, value, identifier, text):
        data = value.get('generated_bytes') if isinstance(value, dict) else None
        if (not isinstance(value, dict) or value.get('schema_version') != 'lfg.owned-byte-generation.v1'
                or value.get('id') != identifier or value.get('input_bytes') != len(text.encode())
                or not isinstance(data, list) or not data or len(data) > self.max_new_bytes
                or any(type(byte) is not int or not 0 <= byte <= 255 for byte in data)
                or value.get('stop_reason') not in ('byte_limit', 'context_limit', 'stop_sequence')
                or (value['stop_reason'] == 'byte_limit' and len(data) != self.max_new_bytes)
                or (value['stop_reason'] == 'stop_sequence' and
                    (self.stop is None or not bytes(data).endswith(self.stop.encode())))):
            raise ValueError('generation response does not match configured bounds')


def evaluate(binary: Path, dataset: Path, policy: Path, config_path: Path, output: Path) -> dict:
    config, config_raw = config_file(config_path)
    examples, identity = select_validation(dataset, policy, config)
    if output.exists():
        raise ValueError('output already exists')
    byte_eval.private_output(output)
    binary = binary.resolve(strict=True)
    binary_sha = sha(binary.read_bytes())
    settings = {'timeout_seconds': config['timeout_seconds'], 'max_response_bytes': config['max_response_bytes']}
    formats, stops, baseline_latencies, model_latencies = Counter(), Counter(), [], []
    with tempfile.TemporaryDirectory(prefix='lfg-assessment-evaluation-') as folder:
        env = {key: value for key, value in os.environ.items() if not key.startswith('LFG_')
               and key != 'MASTER_CLYFFY_UI_DIR'}
        env.update({'LFG_STATE_DIR': str(Path(folder) / 'state'), 'LFG_LEARNING': 'off'})
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); port = listener.getsockname()[1]
        server = subprocess.Popen([str(binary), 'serve', '--bind', f'127.0.0.1:{port}'], cwd=folder,
                                  env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        predictor = None
        try:
            predictor = FormatGenerator(binary, config, settings)
            base = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + config['startup_timeout_seconds']
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while True:
                try:
                    with opener.open(base + '/health', timeout=1) as response:
                        if response.status != 200: raise ValueError('baseline health failed')
                    break
                except OSError:
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise ValueError('baseline did not start') from None
                    time.sleep(.1)
            for index, text in enumerate(examples):
                body = json.dumps({'sessionId': f'format-evaluation-{index}', 'revision': 1, 'text': text,
                                   'declaredSubscriptions': config['declared_subscriptions']}).encode()
                request = urllib.request.Request(base + '/api/lfg/assess', data=body,
                                                 headers={'Content-Type': 'application/json'}, method='POST')
                started = time.perf_counter_ns()
                with opener.open(request, timeout=config['timeout_seconds']) as response:
                    baseline_raw = response.read(config['max_response_bytes'] + 1)
                if len(baseline_raw) > config['max_response_bytes']:
                    raise ValueError('baseline response exceeds configured limit')
                baseline = screen._json(baseline_raw, 'invalid_baseline_response')
                baseline_latencies.append(time.perf_counter_ns() - started)
                if (not isinstance(baseline, dict) or baseline.get('schemaVersion') != 'lfg.assessment.v1'
                        or any(key not in baseline for key in ('intent', 'complexity', 'models', 'agents'))):
                    raise ValueError('baseline returned an unexpected assessment')
                elapsed, result = predictor.request_prediction(str(index), prompt(text, config))
                model_latencies.append(elapsed)
                stops[result['stop_reason']] += 1
                formats[response_format(bytes(result['generated_bytes']), config['required_fields'])] += 1
        finally:
            try:
                if predictor is not None:
                    predictor.close()
            finally:
                if server.poll() is None:
                    server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait()
    if (sha(binary.read_bytes()) != binary_sha or config_file(config_path)[1] != config_raw
            or select_validation(dataset, policy, config)[1] != identity):
        raise ValueError('evaluation inputs changed')
    report = {'schema_version': 'lfg.owned-assessment-format-result.v1', 'binary_sha256': binary_sha,
              'config_sha256': sha(config_raw), 'source': identity, 'responses': len(examples),
              'baseline_valid_assessment_envelopes': len(baseline_latencies),
              'model_format': dict(sorted(formats.items())), 'model_stop_reasons': dict(sorted(stops.items())),
              'baseline_latency_ns': persistent.summary(baseline_latencies),
              'model_generation_latency_ns': persistent.summary(model_latencies),
              'model_code_sha256': sha(Path(__file__).read_bytes()), 'validation_is_independent': False,
              'test_opened': False, 'ground_truth_available': False, 'assessment_quality_evaluated': False,
              'promotion_eligible': False}
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write((json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'dataset', 'policy', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.binary, args.dataset, args.policy, args.config, args.output)
        print(json.dumps({k: result[k] for k in ('responses', 'baseline_valid_assessment_envelopes', 'model_format',
              'model_stop_reasons', 'assessment_quality_evaluated', 'promotion_eligible')}))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        raise SystemExit(f'Assessment format evaluation failed ({type(error).__name__})') from None
