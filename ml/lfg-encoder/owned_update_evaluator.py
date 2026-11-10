#!/usr/bin/env python3
"""Evaluate completed continual-training jobs without blocking training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import owned_training_chain as chain
import owned_training_service as supervisor


def load_config(path: Path, max_config_bytes: int = 1048576) -> tuple[dict, bytes]:
    if type(max_config_bytes) is not int or max_config_bytes < 1:
        raise ValueError('evaluator config byte limit must be positive')
    with path.open('rb') as stream:
        raw = stream.read(max_config_bytes + 1)
    if len(raw) > max_config_bytes:
        raise ValueError('evaluator configuration too large')
    value = json.loads(raw)
    paths = ('python', 'evaluator', 'jobs_root', 'dataset', 'policy', 'evaluation_config', 'report_root', 'state_root')
    integers = ('threads', 'poll_seconds', 'max_checkpoint_bytes', 'max_jobs')
    if (not isinstance(value, dict) or value.get('schema_version') != 'lfg.owned-update-evaluator.v1'
            or any(not isinstance(value.get(k), str) or not Path(value[k]).is_absolute() for k in paths)
            or any(type(value.get(k)) is not int or value[k] < 1 for k in integers)
            or value['poll_seconds'] > 3600 or value.get('device') not in ('cpu', 'cuda')):
        raise ValueError('invalid update evaluator configuration')
    return value, raw


def run(config_path: Path, *, max_config_bytes: int = 1048576) -> None:
    import fcntl
    value, raw = load_config(config_path, max_config_bytes)
    state = Path(value['state_root']); state.mkdir(mode=0o700, parents=True, exist_ok=True)
    reports = Path(value['report_root']); reports.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = open(state / 'evaluator.lock', 'a+b')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('another update evaluator is active') from None
    identity = {'config_sha256': hashlib.sha256(raw).hexdigest(), 'evaluator_sha256': supervisor.digest(Path(value['evaluator'])),
                'evaluation_config_sha256': supervisor.digest(Path(value['evaluation_config'])),
                'implementation_sha256': supervisor.digest(Path(__file__)), 'chain_sha256': supervisor.digest(Path(chain.__file__))}
    identity_path = state / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('evaluation configuration or code changed; use a new state directory')
    if not identity_path.exists():
        supervisor.atomic_json(identity_path, identity)
    status_path = state / 'status.json'
    done_path = state / 'completed.json'
    done = json.loads(done_path.read_text(encoding='utf-8')) if done_path.exists() else {}
    if not isinstance(done, dict):
        raise ValueError('invalid completed evaluation index')
    for key, item in done.items():
        if (not re.fullmatch(r'[0-9a-f]{64}', key) or not isinstance(item, dict)
                or item.get('report_sha256') != supervisor.digest(reports / key / 'report.json')
                or type(item.get('required_fields_present')) is not int or type(item.get('total')) is not int
                or not 0 <= item['required_fields_present'] <= item['total']):
            raise ValueError('completed evaluation index does not match reports')
    def report(status, **fields):
        supervisor.atomic_json(status_path, {'schema_version': 'lfg.owned-update-evaluator-status.v1',
                                             'state': status, 'updated_unix': time.time(), **fields})
    if done and not status_path.exists():
        latest = max(done, key=lambda name: (reports / name / 'report.json').stat().st_mtime_ns)
        report('complete', completed_reports=len(done), required_fields_present=done[latest]['required_fields_present'],
               total=done[latest]['total'])
    while True:
        if (config_path.read_bytes() != raw or supervisor.digest(Path(value['evaluation_config'])) != identity['evaluation_config_sha256']
                or supervisor.digest(Path(value['evaluator'])) != identity['evaluator_sha256']
                or supervisor.digest(Path(__file__)) != identity['implementation_sha256']
                or supervisor.digest(Path(chain.__file__)) != identity['chain_sha256']):
            raise ValueError('evaluation configuration or code changed')
        jobs_root = Path(value['jobs_root'])
        jobs = sorted(path for path in jobs_root.iterdir() if path.is_dir() and not path.is_symlink()
                      and re.fullmatch(r'[0-9a-f]{64}', path.name)) if jobs_root.exists() else []
        if len(jobs) > value['max_jobs']:
            raise ValueError('evaluation job limit exceeded')
        pending = []
        for job in jobs:
            if job.name in done:
                continue
            checkpoint = chain.completed_checkpoint(job / 'state', value['max_checkpoint_bytes'],
                                                    'lfg.owned-assessment-sft-result.v1')
            if checkpoint is not None:
                pending.append((job, checkpoint))
        pending.sort(key=lambda item: (item[0] / 'state' / 'complete.json').stat().st_mtime_ns)
        completed = len(done)
        for job, checkpoint in pending:
            output = reports / job.name
            if not output.exists():
                report('evaluating', completed_reports=completed)
                command = [value['python'], value['evaluator'], '--checkpoint', str(checkpoint), '--dataset', value['dataset'],
                           '--policy', value['policy'], '--config', value['evaluation_config'], '--output', str(output),
                           '--device', value['device'], '--threads', str(value['threads'])]
                result = subprocess.run(command, check=False)
                if result.returncode:
                    report('failed', completed_reports=completed, exit_code=result.returncode)
                    raise ValueError('development evaluation failed')
            path = output / 'report.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            if (data.get('schema_version') != 'lfg.owned-torch-format-evaluation.v1'
                    or data.get('weights_sha256') != supervisor.digest(checkpoint / 'weights.safetensors')
                    or data.get('config_sha256') != identity['evaluation_config_sha256']
                    or data.get('validation_source', {}).get('source_sha256') != supervisor.digest(Path(value['dataset']) / 'validation.jsonl')
                    or data.get('validation_source', {}).get('policy_sha256') != supervisor.digest(Path(value['policy']))):
                raise ValueError('development evaluation report identity mismatch')
            counts = data.get('format_counts')
            if (not isinstance(counts, dict) or any(k not in ('invalid_json', 'missing_required_fields', 'required_fields_present')
                    or type(v) is not int or v < 0 for k, v in counts.items())):
                raise ValueError('invalid development format counts')
            completed += 1
            done[job.name] = {'report_sha256': supervisor.digest(path),
                              'required_fields_present': counts.get('required_fields_present', 0),
                              'total': sum(counts.values())}
            supervisor.atomic_json(done_path, done)
            report('complete', completed_reports=completed, required_fields_present=counts.get('required_fields_present', 0),
                   total=sum(counts.values()))
        if not pending and not done:
            report('waiting', completed_reports=0)
        time.sleep(value['poll_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--max-config-bytes', type=int, default=1048576)
    args = parser.parse_args()
    try:
        run(args.config, max_config_bytes=args.max_config_bytes)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f'owned update evaluator failed ({type(error).__name__}): {error}') from None


if __name__ == '__main__':
    main()
