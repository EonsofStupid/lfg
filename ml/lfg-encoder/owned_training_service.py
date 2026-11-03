#!/usr/bin/env python3
"""Supervise one private, resumable owned-model distillation job on Linux."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix='.status-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('schema_version') != 'lfg.owned-training-service.v1':
        raise ValueError('unsupported service configuration')
    kind = value.get('trainer_kind', 'distillation')
    if kind not in ('distillation', 'assessment_sft'):
        raise ValueError('unsupported trainer kind')
    paths = ('student_config', 'distillation_config', 'runtime_config', 'corpus_plan', 'screen_policy',
             'teacher_manifest', 'teacher_weights') if kind == 'distillation' else ('dataset', 'checkpoint', 'sft_config')
    for name in ('python', 'trainer', 'state_dir', *paths):
        if not isinstance(value.get(name), str) or not Path(value[name]).is_absolute():
            raise ValueError(f'{name} must be an absolute path')
    if kind == 'distillation' and value.get('teacher_format') not in ('gpt2-byte', 'owned-byte'):
        raise ValueError('unsupported teacher format')
    if kind == 'distillation' and value['teacher_format'] == 'gpt2-byte' and not value.get('teacher_revision'):
        raise ValueError('a pinned teacher revision is required')
    if type(value.get('segment_steps')) is not int or value['segment_steps'] < 1:
        raise ValueError('segment_steps must be positive')
    if type(value.get('status_poll_seconds')) not in (int, float) or not 1 <= value['status_poll_seconds'] <= 3600:
        raise ValueError('invalid status poll interval')
    if type(value.get('max_checkpoint_state_bytes', 67108864)) is not int or value.get('max_checkpoint_state_bytes', 67108864) < 1:
        raise ValueError('invalid checkpoint state byte limit')
    return value


def latest_checkpoint(segments: Path, schema: str = 'lfg.owned-distillation-checkpoint.v1',
                      max_state_bytes: int = 67108864) -> tuple[int, Path, str] | None:
    if type(max_state_bytes) is not int or max_state_bytes < 1:
        raise ValueError('invalid checkpoint state byte limit')
    candidates = []
    for state_path in segments.glob('*/checkpoints/step-*/state.json'):
        name = state_path.parent.name
        digits = name.removeprefix('step-')
        if len(digits) == 16 and digits.isascii() and digits.isdecimal() and int(digits) > 0:
            candidates.append((int(digits), state_path))
    if not candidates:
        return None
    latest_step = max(step for step, _ in candidates)
    newest = []
    for step, state_path in candidates:
        if step != latest_step:
            continue
        try:
            with state_path.open('rb') as stream:
                raw = stream.read(max_state_bytes + 1)
            if len(raw) > max_state_bytes:
                raise ValueError('checkpoint state too large')
            state = json.loads(raw)
            if (state.get('schema_version') != schema or type(state['progress']['step']) is not int
                    or state['progress']['step'] != step):
                raise ValueError('checkpoint failed validation')
            newest.append((step, state_path.parent, hashlib.sha256(raw).hexdigest(), state))
        except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'cannot validate published checkpoint: {state_path}') from exc
    for _, path, _, state in newest:
        try:
            if (digest(path / 'weights.safetensors') != state['model_sha256']
                    or digest(path / 'optimizer.safetensors') != state['optimizer_sha256']):
                raise ValueError('checkpoint failed validation')
        except (KeyError, OSError) as exc:
            raise ValueError('cannot validate latest checkpoint tensors') from exc
    if any(item[2] != newest[0][2] for item in newest[1:]):
        raise ValueError('conflicting checkpoints at the latest step')
    return newest[-1][:3]


def run(path: Path, *, exit_on_complete: bool = False) -> None:
    import fcntl  # Linux service lock; keep read-only helpers usable on other platforms.
    value = config(path)
    root = Path(value['state_dir'])
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    segments = root / 'segments'
    segments.mkdir(mode=0o700, exist_ok=True)
    lock = open(root / 'service.lock', 'a+b')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('another supervisor owns this training directory') from None
    identity_file = root / 'service-identity.json'
    # Existing runs cannot silently switch configurations after a restart.
    identity = {'config_sha256': digest(path), 'trainer_sha256': digest(Path(value['trainer']))}
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError('service configuration or trainer changed; create a new state directory')
    if not identity_file.exists():
        atomic_json(identity_file, identity)
    status = root / 'status.json'
    def report(state: str, **extra) -> None:
        atomic_json(status, {'schema_version': 'lfg.owned-training-status.v1', 'state': state,
                             'updated_unix': time.time(), **extra})
    complete = root / 'complete.json'
    kind = value.get('trainer_kind', 'distillation')
    schema = ('lfg.owned-distillation-checkpoint.v1' if kind == 'distillation'
              else 'lfg.owned-assessment-sft-checkpoint.v1')
    while not complete.exists():
        checkpoint = latest_checkpoint(segments, schema, value.get('max_checkpoint_state_bytes', 67108864))
        student = json.loads(Path(value['student_config'] if kind == 'distillation' else value['sft_config']).read_text(encoding='utf-8'))
        total = student['steps']
        if checkpoint and checkpoint[0] > total:
            report('needs_attention', step=checkpoint[0], reason='checkpoint exceeds configured final step')
            raise ValueError('checkpoint exceeds configured final step')
        output = segments / ('segment-' + uuid.uuid4().hex)
        command = [value['python'], value['trainer']]
        options = (('student_config', 'distillation_config', 'runtime_config', 'corpus_plan', 'screen_policy',
                    'teacher_manifest', 'teacher_weights', 'teacher_format') if kind == 'distillation'
                   else ('dataset', 'checkpoint'))
        for option in options:
            command.extend(['--' + option.replace('_', '-'), value[option]])
        if kind == 'assessment_sft':
            command.extend(['--config', value['sft_config']])
        if kind == 'distillation' and value.get('teacher_revision'):
            command.extend(['--teacher-revision', value['teacher_revision']])
        command.extend(['--output', str(output), '--max-steps-this-run', str(value['segment_steps'])])
        if checkpoint:
            command.extend(['--resume', str(checkpoint[1]), '--resume-state-sha256', checkpoint[2]])
        report('training', step=checkpoint[0] if checkpoint else 0, total_steps=total)
        child = subprocess.Popen(command)
        def terminate(signum, _frame):
            if child.poll() is None:
                child.send_signal(signum)
        previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            code = child.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if code:
            report('failed', exit_code=code, step=checkpoint[0] if checkpoint else 0, total_steps=total)
            raise RuntimeError(f'training child exited with status {code}')
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if not manifest.get('completed') or digest(output / 'weights.safetensors') != manifest.get('weights_sha256'):
                raise ValueError('invalid final model result')
            atomic_json(complete, {'completed_steps': manifest['steps'], 'weights_sha256': manifest['weights_sha256'],
                                   'result_path': str(output), 'assessment_quality_evaluated': False})
            break
        next_checkpoint = latest_checkpoint(segments, schema, value.get('max_checkpoint_state_bytes', 67108864))
        if not next_checkpoint or next_checkpoint[0] <= (checkpoint[0] if checkpoint else 0):
            raise ValueError('segment completed without advancing its checkpoint')
    result = json.loads(complete.read_text(encoding='utf-8'))
    report('complete', step=result['completed_steps'], total_steps=result['completed_steps'], weights_sha256=result['weights_sha256'],
           assessment_quality_evaluated=False)
    if exit_on_complete:
        return
    # Keep the user service alive for status inspection, without repeatedly training the same data.
    while True:
        time.sleep(value['status_poll_seconds'])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--exit-on-complete', action='store_true')
    args = parser.parse_args()
    try:
        run(args.config, exit_on_complete=args.exit_on_complete)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f'LFG training service: {error}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
