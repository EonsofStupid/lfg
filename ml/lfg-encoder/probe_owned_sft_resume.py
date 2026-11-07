#!/usr/bin/env python3
"""Compare uninterrupted and resumed owned SFT on an explicitly selected local dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import owned_sft_checkpoint as checkpoints
import train_owned_assessment_sft as trainer


def run(dataset: Path, checkpoint: Path, config_path: Path, output: Path, max_config_bytes: int = 65536) -> dict:
    config, _ = trainer.configuration(config_path, max_config_bytes)
    if config['steps'] < 2:
        raise ValueError('resume comparison requires at least two configured steps')
    if output.exists():
        raise ValueError('probe output already exists')
    checkpoints.private_directory(output)
    output.mkdir(mode=0o700)
    full = trainer.experiment(dataset, checkpoint, config_path, output / 'uninterrupted', max_config_bytes=max_config_bytes)
    partial = trainer.experiment(dataset, checkpoint, config_path, output / 'partial',
                                 max_steps_this_run=1, max_config_bytes=max_config_bytes)
    resumed = trainer.experiment(dataset, checkpoint, config_path, output / 'resumed',
                resume=Path(partial['latest_checkpoint']), resume_state_sha256=partial['latest_checkpoint_state_sha256'],
                max_config_bytes=max_config_bytes)
    result = {'schema_version': 'lfg.owned-assessment-sft-resume-probe.v1',
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'uninterrupted_manifest_sha256': hashlib.sha256((output / 'uninterrupted' / 'manifest.json').read_bytes()).hexdigest(),
              'resumed_manifest_sha256': hashlib.sha256((output / 'resumed' / 'manifest.json').read_bytes()).hexdigest(),
              'source_checkpoint_weights_sha256': full['source_checkpoint_weights_sha256'],
              'dataset_manifest_sha256': full['dataset_manifest_sha256'], 'training_file_sha256': full['training_file_sha256'],
              'config_sha256': full['config_sha256'], 'parameters': full['parameters'], 'steps': full['steps'],
              'device': full['device'], 'torch': full['torch'], 'python': full['python'],
              'same_weights': full['weights_sha256'] == resumed['weights_sha256'],
              'same_losses': full['losses'] == resumed['losses'],
              'same_supervised_tokens': full['supervised_tokens'] == resumed['supervised_tokens'],
              'teacher_claims_verified': False, 'assessment_quality_evaluated': False, 'promotion_eligible': False}
    if full['device'] == 'cuda':
        for name in ('cuda_device', 'cuda_device_index', 'cuda_version', 'cuda_memory_fraction',
                     'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes'):
            result[name] = full[name]
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    descriptor = os.open(output / 'report.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'checkpoint', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-config-bytes', type=int, default=65536)
    args = parser.parse_args()
    try:
        result = run(args.dataset, args.checkpoint, args.config, args.output, args.max_config_bytes)
        print(json.dumps(result))
        if not all(result[key] for key in ('same_weights', 'same_losses', 'same_supervised_tokens')):
            raise SystemExit('Resume comparison differs; inspect the private report.')
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise SystemExit(f'Owned SFT resume probe failed ({type(error).__name__}): {error}') from None
