#!/usr/bin/env python3
"""Compare long-lived local LFG prediction processes on a private corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import queue
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import benchmark_process


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tokenization_status(prediction: dict[str, Any]) -> str:
    if 'tokenization' not in prediction:
        return 'unknown'
    value = prediction['tokenization']
    if (not isinstance(value, dict) or set(value) != {'processed_tokens', 'overflow_chunks', 'truncated'}
            or type(value['processed_tokens']) is not int or value['processed_tokens'] < 1
            or type(value['overflow_chunks']) is not int or value['overflow_chunks'] < 0
            or type(value['truncated']) is not bool or value['truncated'] != (value['overflow_chunks'] > 0)):
        raise ValueError('invalid tokenization coverage metadata')
    return 'truncated' if value['truncated'] else 'complete'


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    fields = {'warmup_rounds', 'measured_rounds', 'max_records', 'max_dataset_bytes',
              'max_response_bytes', 'timeout_seconds'}
    if not isinstance(config, dict) or set(config) != fields | {'schema_version'} or \
            config.get('schema_version') != 'lfg.persistent-benchmark-config.v1':
        raise ValueError('unsupported persistent benchmark configuration')
    if any(type(config[field]) is not int or config[field] < 1 for field in fields):
        raise ValueError('persistent benchmark limits must be positive integers')
    return config, digest(raw)


def load_examples(path: Path, config: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    if path.stat().st_size > config['max_dataset_bytes']:
        raise ValueError('dataset exceeds configured byte limit')
    raw = path.read_bytes()
    if len(raw) > config['max_dataset_bytes']:
        raise ValueError('dataset exceeds configured byte limit')
    values = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if not values or len(values) > config['max_records']:
        raise ValueError('dataset record count exceeds configured limits')
    ids: set[str] = set()
    rows = []
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get('id'), str) or not value['id'].strip() or \
                value['id'] in ids or not isinstance(value.get('text'), str) or not value['text'].strip():
            raise ValueError('dataset IDs and texts must be nonempty and IDs unique')
        ids.add(value['id'])
        rows.append({'id': value['id'], 'text': value['text']})
    return rows, digest(raw)


def memory(process: subprocess.Popen) -> dict[str, int | None]:
    if os.name != 'nt':
        return {'working_set_bytes': None, 'peak_working_set_bytes': None}
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]

    api = ctypes.WinDLL('psapi', use_last_error=True)
    api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    api.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ok = api.GetProcessMemoryInfo(int(process._handle), ctypes.byref(counters), counters.cb)
    return {'working_set_bytes': counters.WorkingSetSize if ok else None,
            'peak_working_set_bytes': counters.PeakWorkingSetSize if ok else None}


def summary(samples: list[int]) -> dict[str, float | int]:
    if not samples:
        raise ValueError('latency samples must not be empty')
    ordered = sorted(samples)
    def nearest_rank(percent: int) -> int:
        return ordered[math.ceil(percent * len(ordered) / 100) - 1]
    return {'count': len(samples), 'median_ns': statistics.median(samples),
            'p95_ns': nearest_rank(95), 'p99_ns': nearest_rank(99),
            'minimum_ns': ordered[0], 'maximum_ns': ordered[-1]}


class Predictor:
    def __init__(self, binary: Path, args: list[str], config: dict[str, Any]):
        self.config = config
        self.started = time.perf_counter_ns()
        self.process = subprocess.Popen([str(binary), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.lines: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self.outgoing: queue.Queue[bytes | None] = queue.Queue(maxsize=1)
        self.model = None
        self.producer = None
        self.failed = False
        def read_lines() -> None:
            assert self.process.stdout is not None
            while True:
                line = self.process.stdout.readline(config['max_response_bytes'] + 1)
                self.lines.put(line)
                if not line:
                    return
        def write_lines() -> None:
            assert self.process.stdin is not None
            while True:
                payload = self.outgoing.get()
                if payload is None:
                    self.process.stdin.close()
                    return
                try:
                    self.process.stdin.write(payload)
                    self.process.stdin.flush()
                except (OSError, ValueError):
                    self.lines.put(b'')
                    return
        threading.Thread(target=read_lines, daemon=True).start()
        threading.Thread(target=write_lines, daemon=True).start()

    def request(self, identifier: str, text: str) -> int:
        elapsed, _ = self.request_prediction(identifier, text)
        return elapsed

    def request_prediction(self, identifier: str, text: str) -> tuple[int, dict[str, Any]]:
        payload = (json.dumps({'id': identifier, 'text': text}, ensure_ascii=False) + '\n').encode('utf-8')
        if self.process.poll() is not None or self.process.stdin is None:
            raise ValueError('prediction process exited before request')
        started = time.perf_counter_ns()
        try:
            self.outgoing.put(payload, timeout=self.config['timeout_seconds'])
            remaining = self.config['timeout_seconds'] - (time.perf_counter_ns() - started) / 1_000_000_000
            if remaining <= 0:
                raise queue.Empty
            line = self.lines.get(timeout=remaining)
        except (queue.Empty, queue.Full):
            self.failed = True
            raise ValueError('prediction response timed out') from None
        elapsed = time.perf_counter_ns() - started
        if not line or len(line) > self.config['max_response_bytes'] or not line.endswith(b'\n'):
            raise ValueError('prediction response is missing or exceeds configured limit')
        value = json.loads(line)
        self.validate_response(value, identifier, text)
        return elapsed, value

    def validate_response(self, value: Any, identifier: str, text: str) -> None:
        if not isinstance(value, dict) or value.get('schema_version') not in ('lfg.evaluation-prediction.v1', 'lfg.evaluation-abstention.v1') or \
                value.get('id') != identifier or value.get('input_sha256') != digest(text.encode('utf-8')) or \
                not isinstance(value.get('model'), dict) or not isinstance(value.get('producer_sha256'), str):
            raise ValueError('prediction response does not match its request')
        coverage = tokenization_status(value)
        model = value['model']
        if any(not isinstance(model.get(key), str) or not model[key].strip() for key in ('id', 'version')) or \
                not isinstance(model.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', model['sha256']) or \
                not re.fullmatch(r'[0-9a-f]{64}', value['producer_sha256']):
            raise ValueError('prediction model or producer identity is invalid')
        if any(not isinstance(value.get(field), list) or
               any(not isinstance(x, str) or not x.strip() for x in value[field]) or
               len(set(value[field])) != len(value[field])
               for field in ('intents', 'capabilities', 'risks')):
            raise ValueError('prediction response has invalid labels')
        if value['schema_version'] == 'lfg.evaluation-abstention.v1':
            if (coverage != 'truncated' or value.get('abstention_reason') != 'token_limit_exceeded'
                    or any(value[field] for field in ('intents', 'capabilities', 'risks'))
                    or 'scores' in value):
                raise ValueError('invalid abstention response')
        elif 'abstention_reason' in value:
            raise ValueError('prediction cannot contain an abstention reason')
        if self.model is not None and (value['model'] != self.model or value['producer_sha256'] != self.producer):
            raise ValueError('prediction identity changed during measurement')
        self.model, self.producer = value['model'], value['producer_sha256']

    def close(self) -> None:
        if self.failed and self.process.poll() is None:
            self.process.terminate()
        try:
            self.outgoing.put_nowait(None)
        except queue.Full:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()


def run(binary: Path, dataset: Path, config_path: Path, baseline_args: list[str], candidate_args: list[str]) -> dict[str, Any]:
    config, config_hash = load_config(config_path)
    rows, dataset_hash = load_examples(dataset, config)
    binary = binary.resolve(strict=True)
    binary_hash = digest(binary.read_bytes())
    benchmark_hash = digest(Path(__file__).read_bytes())
    counter_reader_hash = digest(Path(benchmark_process.__file__).read_bytes())
    if any(not args or any(not isinstance(arg, str) or not arg for arg in args)
           for args in (baseline_args, candidate_args)):
        raise ValueError('both command argument lists must be nonempty')
    overrides = {key: Path(os.environ[key]) for key in ('LFG_PREFLIGHT_CONFIG', 'LFG_OPERATIONS_CONFIG') if key in os.environ}
    override_hashes = {key: digest(path.read_bytes()) for key, path in overrides.items()}
    processes: dict[str, Predictor] = {}
    try:
        cold = {}
        for name, args in (('baseline', baseline_args), ('candidate', candidate_args)):
            predictor = Predictor(binary, args, config)
            processes[name] = predictor
            predictor.request('warm-0', rows[0]['text'])
            cold[name] = time.perf_counter_ns() - predictor.started
            if predictor.producer != binary_hash:
                raise ValueError('prediction producer differs from binary digest')
        for n in range(1, config['warmup_rounds'] * len(rows)):
            for name in (('baseline', 'candidate') if n % 2 == 0 else ('candidate', 'baseline')):
                processes[name].request(f'warm-{n}', rows[n % len(rows)]['text'])
        before = {name: benchmark_process.windows_counters(p.process) for name, p in processes.items()}
        samples: dict[str, list[int]] = {'baseline': [], 'candidate': []}
        tokenization = {name: {'complete': 0, 'truncated': 0, 'unknown': 0} for name in processes}
        outcomes = {name: {'predicted': 0, 'abstained': 0} for name in processes}
        for n in range(config['measured_rounds'] * len(rows)):
            for name in (('baseline', 'candidate') if n % 2 == 0 else ('candidate', 'baseline')):
                elapsed, value = processes[name].request_prediction(f'measured-{n}', rows[n % len(rows)]['text'])
                samples[name].append(elapsed)
                tokenization[name][tokenization_status(value)] += 1
                outcomes[name]['abstained' if value['schema_version'] == 'lfg.evaluation-abstention.v1' else 'predicted'] += 1
        after = {name: benchmark_process.windows_counters(p.process) for name, p in processes.items()}
        memory_after = {name: memory(p.process) for name, p in processes.items()}
        if any(p.process.poll() is not None for p in processes.values()):
            raise ValueError('prediction process exited during measurement')
        if (digest(binary.read_bytes()) != binary_hash or digest(dataset.read_bytes()) != dataset_hash or
                digest(config_path.read_bytes()) != config_hash or
                digest(Path(__file__).read_bytes()) != benchmark_hash or
                digest(Path(benchmark_process.__file__).read_bytes()) != counter_reader_hash or
                {key: digest(path.read_bytes()) for key, path in overrides.items()} != override_hashes):
            raise ValueError('benchmark inputs changed during measurement')
        results = {}
        for name, predictor in processes.items():
            delta = {key: None if before[name][key] is None or after[name][key] is None
                     else after[name][key] - before[name][key] for key in before[name]}
            if any(value is not None and value < 0 for value in delta.values()):
                raise ValueError('process counters moved backwards during measurement')
            results[name] = {'model': predictor.model, 'time_to_first_response_ns': cold[name],
                             'latency': summary(samples[name]), 'measured_counter_delta': delta,
                             'memory_after': memory_after[name], 'tokenization': tokenization[name],
                             'outcomes': outcomes[name]}
        return {'schema_version': 'lfg.persistent-benchmark.v1', 'promotion_eligible': False,
                'binary_sha256': binary_hash, 'dataset_sha256': dataset_hash, 'config_sha256': config_hash,
                'benchmark_sha256': benchmark_hash, 'counter_reader_sha256': counter_reader_hash,
                'overrides_sha256': override_hashes,
                'baseline_args_sha256': digest(json.dumps(baseline_args, separators=(',', ':')).encode()),
                'candidate_args_sha256': digest(json.dumps(candidate_args, separators=(',', ':')).encode()),
                'platform': platform.platform(), 'machine': platform.machine(), 'python_version': platform.python_version(),
                'processor': platform.processor() or None, 'logical_cpus': os.cpu_count(),
                'rows': len(rows), 'warmup_rounds': config['warmup_rounds'], 'measured_rounds': config['measured_rounds'],
                'order': 'alternating-sequential', 'percentile_method': 'nearest-rank',
                'counter_source': 'Win32 process times, process cycles and working set' if os.name == 'nt' else None,
                'measurement_scope': 'one request at a time over local stdio, including IPC and response serialization',
                'results': results}
    finally:
        for predictor in processes.values():
            predictor.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--baseline-arg', action='append', required=True)
    parser.add_argument('--candidate-arg', action='append', required=True)
    args = parser.parse_args()
    try:
        result = run(args.binary, args.dataset, args.config, args.baseline_arg, args.candidate_arg)
        with args.output.open('x', encoding='utf-8') as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write('\n')
        print(json.dumps(result, allow_nan=False))
    except (OSError, ValueError, KeyError, UnicodeError, TypeError, BrokenPipeError) as error:
        raise SystemExit(f'persistent benchmark rejected: {error}') from error


if __name__ == '__main__':
    main()
