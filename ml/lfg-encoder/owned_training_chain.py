#!/usr/bin/env python3
"""Start configured assessment fine-tuning after an owned pretraining job finishes."""
import argparse
import json
import os
from pathlib import Path
import time

import owned_training_service as supervisor


def load_config(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if len(raw) > 1048576:
        raise ValueError('chain configuration is too large')
    value = json.loads(raw)
    paths = ('python', 'supervisor', 'trainer', 'source_state', 'target_state', 'dataset', 'sft_config')
    if (not isinstance(value, dict) or value.get('schema_version') != 'lfg.owned-training-chain.v1'
            or any(not isinstance(value.get(k), str) or not Path(value[k]).is_absolute() for k in paths)
            or type(value.get('segment_steps')) is not int or value['segment_steps'] < 1
            or type(value.get('poll_seconds')) is not int or not 1 <= value['poll_seconds'] <= 3600
            or type(value.get('max_checkpoint_bytes')) is not int or value['max_checkpoint_bytes'] < 1):
        raise ValueError('invalid chain configuration')
    return value, raw


def completed_checkpoint(source_state: Path, max_checkpoint_bytes: int, schema: str) -> Path | None:
    source = source_state.resolve()
    complete = source / 'complete.json'
    if not complete.exists():
        status = source / 'status.json'
        if status.exists():
            current = json.loads(status.read_text(encoding='utf-8'))
            if current.get('state') in ('failed', 'needs_attention'):
                raise ValueError('source training needs attention')
            if current.get('state') == 'complete':
                raise ValueError('source completion marker is missing')
        return None
    completion = json.loads(complete.read_text(encoding='utf-8'))
    checkpoint = Path(completion['result_path'])
    if not checkpoint.is_absolute() or checkpoint.is_symlink():
        raise ValueError('invalid source checkpoint path')
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_relative_to((source / 'segments').resolve()):
        raise ValueError('source checkpoint is outside the source job')
    manifest_path, weights = checkpoint / 'manifest.json', checkpoint / 'weights.safetensors'
    if manifest_path.is_symlink() or weights.is_symlink() or weights.stat().st_size > max_checkpoint_bytes:
        raise ValueError('invalid source checkpoint files')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    digest = supervisor.digest(weights)
    if (manifest.get('schema_version') != schema or manifest.get('completed') is not True
            or manifest.get('steps') != completion.get('completed_steps')
            or manifest.get('weights_sha256') != digest or completion.get('weights_sha256') != digest):
        raise ValueError('source checkpoint identity mismatch')
    return checkpoint


def target_config(value: dict) -> dict | None:
    checkpoint = completed_checkpoint(Path(value['source_state']), value['max_checkpoint_bytes'],
                                      'lfg.owned-distillation-result.v1')
    if checkpoint is None:
        return None
    return {'schema_version': 'lfg.owned-training-service.v1', 'trainer_kind': 'assessment_sft',
            'python': value['python'], 'trainer': value['trainer'], 'state_dir': value['target_state'],
            'dataset': value['dataset'], 'checkpoint': str(checkpoint), 'sft_config': value['sft_config'],
            'segment_steps': value['segment_steps'], 'status_poll_seconds': value['poll_seconds']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    value, raw = load_config(args.config)
    while True:
        target = target_config(value)
        if args.config.read_bytes() != raw:
            raise ValueError('chain configuration changed')
        if target is not None:
            root = Path(value['target_state'])
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            path = root / 'generated-service-config.json'
            if path.exists():
                if json.loads(path.read_text(encoding='utf-8')) != target:
                    raise ValueError('existing target configuration changed')
            else:
                supervisor.atomic_json(path, target)
            os.execv(value['python'], [value['python'], value['supervisor'], '--config', str(path)])
        time.sleep(value['poll_seconds'])


if __name__ == '__main__':
    main()
