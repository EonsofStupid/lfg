#!/usr/bin/env python3
"""Run local models on requests a teacher answered and compare their answers.

The reference is a set of unreviewed teacher proposals, so this shows how closely a
model agrees with its teacher, not whether either is correct. Validation and test
files stay closed. The job can stop at any time and continue with the next request.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import time
import urllib.error
import urllib.request

from local_teacher_service import command as collector_command
from owned_training_service import atomic_json

API_KEY_ENV = 'LFG_LOCAL_MODEL_API_KEY'
PLACEHOLDERS = ('{weights}', '{port}', '{id}')


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def weights_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    # Download tools keep changing metadata in hidden directories next to the weights.
    return sorted(item for item in path.rglob('*') if item.is_file()
                  and not any(part.startswith('.') for part in item.relative_to(path).parts))


def weights_digest(path: Path) -> str:
    if path.is_file():
        return file_sha(path)
    digest = hashlib.sha256()
    for item in weights_files(path):
        digest.update(f'{item.relative_to(path).as_posix()} {file_sha(item)}\n'.encode())
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    paths = ('python', 'collector', 'policy', 'reference_dataset', 'output')
    models = value.get('models') if isinstance(value, dict) else None
    if (not isinstance(value, dict) or value.get('schema_version') != 'lfg.local-model-comparison.v1'
            or any(not isinstance(value.get(k), str) or not Path(value[k]).is_absolute() for k in paths)
            or not isinstance(value.get('server_command'), list) or not value['server_command']
            or any(not isinstance(k, str) or not k for k in value['server_command'])
            or any(k not in value['server_command'] for k in (*PLACEHOLDERS, '127.0.0.1'))
            or type(value.get('port')) is not int or not 0 < value['port'] < 65536
            or type(value.get('startup_timeout_seconds')) not in (int, float) or value['startup_timeout_seconds'] <= 0
            or not isinstance(value.get('collection'), dict)
            or not isinstance(value.get('instruction_suffix', ''), str)
            or ('instruction' in value and (not isinstance(value['instruction'], str) or not value['instruction'].strip()))
            or not isinstance(value.get('complexity_levels'), dict) or not value['complexity_levels']
            or any(type(k) is not int for k in value['complexity_levels'].values())
            or not isinstance(models, list) or not models):
        raise ValueError('invalid local model comparison configuration')
    ids = set()
    for model in models:
        if (not isinstance(model, dict) or not isinstance(model.get('id'), str)
                or not re.fullmatch(r'[a-z0-9][a-z0-9._-]*', model['id']) or model['id'] in ids
                or not isinstance(model.get('weights'), str) or not Path(model['weights']).is_absolute()
                or not isinstance(model.get('server_args'), list)
                or any(not isinstance(k, str) for k in model['server_args'])
                or not isinstance(model.get('request_extensions', {}), dict)):
            raise ValueError('invalid local model')
        ids.add(model['id'])
    return value


def read_reference(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        messages, metadata = record['messages'], record['metadata']
        if [item['role'] for item in messages] != ['system', 'user', 'assistant']:
            raise ValueError('reference record must contain one system, user and assistant message')
        rows.append({'id': record['id'], 'system': messages[0]['content'], 'text': messages[1]['content'],
                     'answer': json.loads(messages[2]['content']), 'project': metadata['project'],
                     'project_group': metadata['project_group'],
                     'source_sha256': metadata['original_source_sha256_claim']})
    if not rows or len({row['id'] for row in rows}) != len(rows) or len({row['system'] for row in rows}) != 1:
        raise ValueError('reference must have unique requests and one system instruction')
    return rows


def server_argv(config: dict, model: dict) -> list[str]:
    values = {'{weights}': model['weights'], '{port}': str(config['port']), '{id}': model['id']}
    return [values.get(item, item) for item in (*config['server_command'], *model['server_args'])]


def write_selection(reference: list[dict], directory: Path, reference_sha256: str) -> None:
    raw = ''.join(json.dumps({'id': row['id'], 'text': row['text'], 'project': row['project'],
                              'project_group': row['project_group'],
                              'screened_text_sha256': sha(row['text'].encode()),
                              'source_text_sha256_claim': row['source_sha256'], 'split': 'train',
                              'review_status': 'unreviewed', 'outcome': 'unknown'},
                             ensure_ascii=False, sort_keys=True) + '\n' for row in reference).encode()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    (directory / 'requests.jsonl').write_bytes(raw)
    atomic_json(directory / 'manifest.json', {'schema_version': 'lfg.teacher-request-selection-result.v1',
                                              'requests': len(reference), 'requests_sha256': sha(raw),
                                              'reference_dataset_sha256': reference_sha256,
                                              'validation_opened': False, 'test_opened': False})


def write_collection_config(config: dict, model: dict, identity: str, system: str, path: Path) -> None:
    atomic_json(path, {**config['collection'], 'endpoint': f"http://127.0.0.1:{config['port']}/v1/chat/completions",
                       'model': model['id'], 'teacher_identity': identity,
                       'system_prompt': config.get('instruction', system) + config.get('instruction_suffix', ''),
                       'api_key_env': API_KEY_ENV, 'request_extensions': model.get('request_extensions', {})})


def fetch(url: str, key: str | None = None, timeout: float = 10) -> str:
    headers = {'Authorization': 'Bearer ' + key} if key else {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
        return response.read(1 << 20).decode('utf-8')


def wait_until_ready(process: subprocess.Popen, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'model server exited with status {process.returncode}')
        try:
            fetch(url + '/health', timeout=5)
            return
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    raise RuntimeError('model server did not become ready')


def parse_metrics(text: str) -> dict:
    metrics = {}
    for line in text.splitlines():
        match = re.fullmatch(r'(?:llamacpp|vllm):(\w+)(?:\{[^}]*\})? (\S+)', line.strip())
        if match:
            try:
                value = float(match[2])
            except ValueError:
                continue
            if math.isfinite(value):
                metrics[match[1]] = metrics.get(match[1], 0) + value
    return metrics


def available_memory() -> int | None:
    try:
        match = re.search(r'^MemAvailable:\s+(\d+) kB', Path('/proc/meminfo').read_text(), re.M)
    except OSError:
        return None
    return int(match[1]) * 1024 if match else None


def count_files(directory: Path) -> int:
    return sum(1 for _ in directory.glob('*.json')) if directory.is_dir() else 0


def run_model(config: dict, model: dict, selection: Path, directory: Path, system: str) -> None:
    collection = directory / 'collection'
    if (collection / 'manifest.json').exists():
        return
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    weights = Path(model['weights'])
    collection_config = directory / 'collection.json'
    write_collection_config(config, model, f"{model['id']}@sha256:{weights_digest(weights)}", system, collection_config)
    key = secrets.token_urlsafe(32)
    environment = {**os.environ, API_KEY_ENV: key, 'LLAMA_API_KEY': key, 'VLLM_API_KEY': key}
    base = f"http://127.0.0.1:{config['port']}"
    before = available_memory()
    started = time.monotonic()
    with (directory / 'server.log').open('ab') as log:
        server = subprocess.Popen(server_argv(config, model), env=environment, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_until_ready(server, base, config['startup_timeout_seconds'])
        ready, ready_memory = time.monotonic(), available_memory()
        answered_before = count_files(collection / 'records')
        argv = collector_command(Path(config['python']), Path(config['collector']), selection, collection_config,
                                 Path(config['policy']), collection)
        subprocess.run(argv, env=environment, check=True, stdout=subprocess.DEVNULL)
        try:
            metrics = parse_metrics(fetch(base + '/metrics', key))
        except (urllib.error.URLError, OSError):
            metrics = {}
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    segments_path = directory / 'segments.json'
    segments = json.loads(segments_path.read_text(encoding='utf-8')) if segments_path.exists() else []
    segments.append({'answered': count_files(collection / 'records') - answered_before,
                     'startup_seconds': ready - started, 'elapsed_seconds': time.monotonic() - ready,
                     'available_memory_before_bytes': before, 'available_memory_ready_bytes': ready_memory,
                     'metrics': metrics})
    atomic_json(segments_path, segments)


def normal_label(value: object) -> str | None:
    return re.sub(r'[\s-]+', '_', value.strip().lower()) if isinstance(value, str) and value.strip() else None


def valid_count(value: object) -> bool:
    return type(value) is int and value >= 0


def most_common(values: list) -> dict | None:
    if not values:
        return None
    value, count = Counter(values).most_common(1)[0]
    return {'value': value, 'share': count / len(values)}


def score_model(reference: list[dict], answers: dict, levels: dict) -> dict:
    exact, within, compared, unknown = 0, 0, 0, 0
    agents_exact, agents_compared, agents_invalid, difference, empty_models = 0, 0, 0, 0, 0
    for row in reference:
        answer = answers.get(row['id'])
        if answer is None:
            continue
        ours, theirs = normal_label(answer.get('complexity')), normal_label(row['answer'].get('complexity'))
        if ours in levels and theirs in levels:
            compared += 1
            exact += levels[ours] == levels[theirs]
            within += abs(levels[ours] - levels[theirs]) <= 1
        else:
            unknown += 1
        if answer.get('suggested_models') == []:
            empty_models += 1
        if not valid_count(answer.get('agent_count')) or not valid_count(row['answer'].get('agent_count')):
            agents_invalid += 1
            continue
        agents_compared += 1
        agents_exact += answer['agent_count'] == row['answer']['agent_count']
        difference += abs(answer['agent_count'] - row['answer']['agent_count'])
    return {'complexity': {'compared': compared, 'exact': exact, 'within_one_level': within, 'unknown_labels': unknown},
            'agent_count': {'compared': agents_compared, 'exact': agents_exact, 'invalid': agents_invalid,
                            'mean_absolute_difference': difference / agents_compared if agents_compared else None},
            'suggested_models_empty': empty_models}


def speed(segments: list[dict]) -> dict:
    totals = Counter()
    for segment in segments:
        totals.update(segment['metrics'])
    tokens = totals['tokens_predicted_total'] + totals['generation_tokens_total']
    seconds = totals['tokens_predicted_seconds_total'] + totals['request_decode_time_seconds_sum']
    first_count = totals['time_to_first_token_seconds_count']
    answered = sum(item['answered'] for item in segments)
    elapsed = sum(item['elapsed_seconds'] for item in segments)
    return {'generated_tokens_per_second': tokens / seconds if seconds else None,
            'mean_time_to_first_token_seconds': totals['time_to_first_token_seconds_sum'] / first_count if first_count else None,
            'seconds_per_answer': elapsed / answered if answered else None,
            'startup_seconds': segments[-1]['startup_seconds'] if segments else None,
            'available_memory_before_bytes': segments[-1]['available_memory_before_bytes'] if segments else None,
            'available_memory_ready_bytes': segments[-1]['available_memory_ready_bytes'] if segments else None,
            'finished_segments': len(segments)}


def compare(config: dict, reference: list[dict], reference_sha256: str) -> dict:
    levels = {normal_label(key): value for key, value in config['complexity_levels'].items()}
    complexity = [normal_label(row['answer'].get('complexity')) for row in reference]
    report = {'schema_version': 'lfg.local-model-comparison-result.v1', 'reference_dataset_sha256': reference_sha256,
              'reference_kind': 'unreviewed_teacher_proposals', 'requests': len(reference),
              'always_most_common': {
                  'complexity': most_common([levels[item] for item in complexity if item in levels]),
                  'agent_count': most_common([row['answer']['agent_count'] for row in reference
                                              if valid_count(row['answer'].get('agent_count'))])},
              'assessment_quality_evaluated': False, 'validation_opened': False, 'test_opened': False, 'models': {}}
    for model in config['models']:
        directory = Path(config['output']) / 'models' / model['id']
        answers = {}
        for path in sorted((directory / 'collection' / 'records').glob('*.json')):
            trace = json.loads(path.read_text(encoding='utf-8'))['trace']
            answers[trace['id']] = json.loads(trace['response'])
        segments_path = directory / 'segments.json'
        segments = json.loads(segments_path.read_text(encoding='utf-8')) if segments_path.exists() else []
        report['models'][model['id']] = {
            'finished': (directory / 'collection' / 'manifest.json').exists(), 'answered': len(answers),
            'failed': count_files(directory / 'collection' / 'failures'),
            'weights_bytes': sum(item.stat().st_size for item in weights_files(Path(model['weights']))),
            **score_model(reference, answers, levels), 'speed': speed(segments)}
    return report


def experiment(config_path: Path) -> dict:
    config = load_config(config_path)
    reference_path = Path(config['reference_dataset'])
    reference_sha256 = file_sha(reference_path)
    reference = read_reference(reference_path)
    output = Path(config['output'])
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    identity = output / 'identity.json'
    expected = {'config_sha256': file_sha(config_path), 'reference_dataset_sha256': reference_sha256}
    if identity.exists():
        if json.loads(identity.read_text(encoding='utf-8')) != expected:
            raise ValueError('output belongs to a different configuration or reference')
    else:
        atomic_json(identity, expected)
    selection = output / 'selection'
    write_selection(reference, selection, reference_sha256)
    for model in config['models']:
        run_model(config, model, selection, output / 'models' / model['id'], reference[0]['system'])
        atomic_json(output / 'comparison.json', compare(config, reference, reference_sha256))
    report = compare(config, reference, reference_sha256)
    atomic_json(output / 'comparison.json', report)
    atomic_json(output / 'complete.json', {'schema_version': 'lfg.local-model-comparison-complete.v1', **expected})
    return report


def stop(*_):
    raise SystemExit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    print(json.dumps(experiment(args.config.resolve()), indent=2, sort_keys=True))
