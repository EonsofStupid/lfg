#!/usr/bin/env python3
"""Measure the actual deterministic kernel on operator-selected local requests."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import benchmark_persistent as persistent
import benchmark_process
import node_evaluate


class KernelPredictor(persistent.Predictor):
    def __init__(self, binary: Path, args: list[str], config: dict[str, Any], snapshot_sha: str):
        self.snapshot_sha = snapshot_sha
        super().__init__(binary, args, config)

    def validate_response(self, value: Any, identifier: str, text: str) -> None:
        if (not isinstance(value, dict) or set(value) != {'schema_version', 'id', 'input_sha256', 'result'}
                or value['schema_version'] != 'lfg.kernel-prediction.v1' or value['id'] != identifier
                or value['input_sha256'] != persistent.digest(text.encode('utf-8'))):
            raise ValueError('kernel response does not match request')
        result = value['result']
        if (not isinstance(result, dict) or result.get('schemaVersion') != 'lfg.kernel-result.v1'
                or result.get('snapshotSha256') != self.snapshot_sha):
            raise ValueError('kernel response does not match snapshot')


def load_rows(path: Path, config: dict[str, Any], id_field: str) -> tuple[list[dict[str, str]], str]:
    if path.stat().st_size > config['max_dataset_bytes']:
        raise ValueError('dataset exceeds configured byte limit')
    raw = path.read_bytes()
    if len(raw) > config['max_dataset_bytes']:
        raise ValueError('dataset exceeds configured byte limit')
    records = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if not records or len(records) > config['max_records']:
        raise ValueError('dataset record count exceeds configured limits')
    seen = set()
    rows = []
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get(id_field), str)
                or not record[id_field].strip() or record[id_field] in seen
                or not isinstance(record.get('text'), str) or not record['text'].strip()):
            raise ValueError('dataset IDs and texts must be nonempty and IDs unique')
        seen.add(record[id_field])
        rows.append({'id': record[id_field], 'text': record['text']})
    return rows, persistent.digest(raw)


def run(binary: Path, dataset: Path, config_path: Path, snapshot: Path,
        node: str, id_field: str) -> dict[str, Any]:
    config, config_hash = persistent.load_config(config_path)
    rows, dataset_hash = load_rows(dataset, config, id_field)
    binary = binary.resolve(strict=True)
    snapshot = snapshot.resolve(strict=True)
    snapshot_raw = snapshot.read_bytes()
    snapshot_hash = persistent.digest(snapshot_raw)
    spec = json.loads(snapshot_raw)
    choice_node = spec.get('nodes', {}).get(node) if isinstance(spec, dict) else None
    if (not isinstance(choice_node, dict) or choice_node.get('kind') != 'choice'
            or not isinstance(choice_node.get('options'), dict) or not choice_node['options']):
        raise ValueError('selected node must be a snapshot choice')
    options = set(choice_node['options'])
    binary_hash = node_evaluate.hash_file(binary)
    code_paths = [Path(__file__), Path(persistent.__file__), Path(benchmark_process.__file__),
                  Path(node_evaluate.__file__)]
    code_hashes = {path.name: node_evaluate.hash_file(path) for path in code_paths}
    operations = Path(os.environ['LFG_OPERATIONS_CONFIG']) if 'LFG_OPERATIONS_CONFIG' in os.environ else None
    operations_hash = node_evaluate.hash_file(operations) if operations else None
    total = len(rows) * (config['warmup_rounds'] + config['measured_rounds'])
    predictor = KernelPredictor(binary, ['kernel-predict', '--snapshot', str(snapshot),
                                         '--max-examples', str(total)], config, snapshot_hash)
    try:
        observed: dict[int, str] = {}
        samples: dict[str, list[int]] = {'selected': [], 'escalated': [], 'not_reached': []}
        cold = None
        before = None
        for index in range(total):
            if index == len(rows) * config['warmup_rounds']:
                before = benchmark_process.windows_counters(predictor.process)
            item_index = index % len(rows)
            elapsed, response = predictor.request_prediction(f'benchmark-{index}', rows[item_index]['text'])
            if index == 0:
                cold = time.perf_counter_ns() - predictor.started
            result = response['result']
            state, _ = node_evaluate.baseline_choice(result, node, options)
            result_hash = persistent.digest(json.dumps(result, sort_keys=True, separators=(',', ':'),
                                                        ensure_ascii=False).encode('utf-8'))
            if item_index in observed and observed[item_index] != result_hash:
                raise ValueError('kernel result changed for a repeated input')
            observed[item_index] = result_hash
            if index >= len(rows) * config['warmup_rounds']:
                samples[state].append(elapsed)
        after = benchmark_process.windows_counters(predictor.process)
        mem = persistent.memory(predictor.process)
        if predictor.process.poll() is not None:
            raise ValueError('kernel process exited during measurement')
        if (node_evaluate.hash_file(binary) != binary_hash or persistent.digest(dataset.read_bytes()) != dataset_hash
                or persistent.digest(config_path.read_bytes()) != config_hash
                or persistent.digest(snapshot.read_bytes()) != snapshot_hash
                or {path.name: node_evaluate.hash_file(path) for path in code_paths} != code_hashes
                or (node_evaluate.hash_file(operations) if operations else None) != operations_hash):
            raise ValueError('benchmark inputs changed during measurement')
        assert before is not None
        counters = {key: None if before[key] is None or after[key] is None else after[key] - before[key]
                    for key in before}
        if any(value is not None and value < 0 for value in counters.values()):
            raise ValueError('process counters moved backwards')
        all_samples = [value for values in samples.values() for value in values]
        return {'schema_version': 'lfg.kernel-persistent-benchmark.v1', 'promotion_eligible': False,
                'binary_sha256': binary_hash, 'dataset_sha256': dataset_hash, 'config_sha256': config_hash,
                'snapshot_sha256': snapshot_hash, 'code_sha256': code_hashes,
                'operations_config_sha256': operations_hash, 'node': node, 'rows': len(rows),
                'warmup_rounds': config['warmup_rounds'], 'measured_rounds': config['measured_rounds'],
                'platform': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor() or None,
                'logical_cpus': os.cpu_count(), 'python_version': platform.python_version(),
                'time_to_first_response_ns': cold, 'latency': persistent.summary(all_samples),
                'latency_by_node_state': {key: persistent.summary(values) if values else None
                                          for key, values in samples.items()},
                'measured_counter_delta': counters, 'memory_after': mem,
                'measurement_scope': 'sequential local stdio including IPC and response serialization'}
    finally:
        predictor.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'dataset', 'config', 'snapshot', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--node', required=True)
    parser.add_argument('--id-field', default='id')
    args = parser.parse_args()
    try:
        report = run(args.binary, args.dataset, args.config, args.snapshot, args.node, args.id_field)
        with args.output.open('x', encoding='utf-8') as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
        print(json.dumps(report, allow_nan=False))
    except (OSError, ValueError, KeyError, UnicodeError, TypeError, BrokenPipeError) as error:
        raise SystemExit(f'kernel benchmark rejected: {error}') from error


if __name__ == '__main__':
    main()
