"""Measure an experimental owned generator on selected local requests."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import benchmark_persistent as persistent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class Generator(persistent.Predictor):
    def __init__(self, binary: Path, args: list[str], config: dict, max_new_bytes: int):
        self.max_new_bytes = max_new_bytes
        super().__init__(binary, args, config)

    def validate_response(self, value, identifier, text):
        data = value.get('generated_bytes') if isinstance(value, dict) else None
        if (not isinstance(value, dict) or value.get('schema_version') != 'lfg.owned-byte-generation.v1'
                or value.get('id') != identifier or value.get('input_bytes') != len(text.encode('utf-8'))
                or not isinstance(data, list) or not data or len(data) > self.max_new_bytes
                or any(type(byte) is not int or not 0 <= byte <= 255 for byte in data)
                or value.get('stop_reason') not in ('byte_limit', 'context_limit')
                or (value['stop_reason'] == 'byte_limit' and len(data) != self.max_new_bytes)):
            raise ValueError('generation response does not match the request or configured limits')


def cpu_time_ns(process) -> int | None:
    if os.name != 'nt':
        return None
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    api.GetProcessTimes.restype = wintypes.BOOL
    created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
    if not api.GetProcessTimes(int(process._handle), ctypes.byref(created), ctypes.byref(exited),
                               ctypes.byref(kernel), ctypes.byref(user)):
        return None
    def value(item): return (item.dwHighDateTime << 32) | item.dwLowDateTime
    return (value(kernel) + value(user)) * 100


def run(binary: Path, dataset: Path, config_path: Path, max_new_bytes: int,
        runtime_config: Path | None = None) -> dict:
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    fields = ('warmup_rounds', 'measured_rounds', 'max_records', 'max_dataset_bytes',
              'max_response_bytes', 'timeout_seconds')
    if (not isinstance(config, dict) or config.get('schema_version') != 'lfg.persistent-benchmark-config.v1'
            or any(type(config.get(name)) is not int or config[name] < 1 for name in fields)
            or type(max_new_bytes) is not int or max_new_bytes < 1):
        raise ValueError('invalid generation benchmark configuration')
    rows, dataset_sha = persistent.load_examples(dataset, config)
    binary = binary.resolve(strict=True)
    binary_sha = sha(binary.read_bytes())
    args = ['embedded-owned-byte-generate', '--experimental', '--max-new-bytes', str(max_new_bytes)]
    config_runtime_raw = None
    if runtime_config is not None:
        config_runtime_raw = runtime_config.read_bytes()
        args += ['--runtime-config', str(runtime_config)]
    predictor = Generator(binary, args, config, max_new_bytes)
    samples, stops, lengths = [], Counter(), Counter()
    try:
        for round_index in range(config['warmup_rounds']):
            for index, row in enumerate(rows):
                predictor.request_prediction(f'warmup-{round_index}-{index}', row['text'])
        startup = time.perf_counter_ns() - predictor.started
        cpu_before = cpu_time_ns(predictor.process)
        for round_index in range(config['measured_rounds']):
            for index, row in enumerate(rows):
                elapsed, value = predictor.request_prediction(f'measured-{round_index}-{index}', row['text'])
                samples.append(elapsed)
                stops[value['stop_reason']] += 1
                lengths[len(value['generated_bytes'])] += 1
        cpu_after = cpu_time_ns(predictor.process)
        memory = persistent.memory(predictor.process)
    finally:
        predictor.close()
    if (sha(binary.read_bytes()) != binary_sha or sha(dataset.read_bytes()) != dataset_sha
            or config_path.read_bytes() != config_raw
            or (runtime_config is not None and runtime_config.read_bytes() != config_runtime_raw)):
        raise ValueError('benchmark input changed during measurement')
    return {'schema_version': 'lfg.owned-generation-benchmark.v1', 'binary_sha256': binary_sha,
            'dataset_sha256': dataset_sha, 'config_sha256': sha(config_raw),
            'runtime_config_sha256': sha(config_runtime_raw) if config_runtime_raw is not None else None,
            'benchmark_sha256': sha(Path(__file__).read_bytes()),
            'benchmark_support_sha256': sha(Path(persistent.__file__).read_bytes()),
            'warmup_rounds': config['warmup_rounds'], 'measured_rounds': config['measured_rounds'],
            'records': len(rows), 'max_new_bytes': max_new_bytes, 'latency': persistent.summary(samples),
            'warmup_elapsed_ns': startup, 'cpu_time_ns': None if cpu_before is None or cpu_after is None else cpu_after - cpu_before,
            'memory': memory, 'stop_reasons': dict(stops), 'generated_lengths': dict(sorted(lengths.items())),
            'platform': platform.platform(), 'processor': platform.processor(),
            'native_threads_controlled': False, 'deterministic_baseline_compared': False,
            'assessment_quality_evaluated': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'dataset', 'config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-new-bytes', type=int, required=True)
    parser.add_argument('--runtime-config', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.binary, args.dataset, args.config, args.max_new_bytes, args.runtime_config), indent=2))
