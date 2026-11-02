"""Private, data-only optimizer snapshots for owned assessment fine-tuning."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from safetensors.torch import load, save
import torch


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def private_directory(directory: Path) -> None:
    """All files in a model/checkpoint output directory must be ignored and untracked."""
    directory = directory.resolve()
    ancestor = directory.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    try:
        result = subprocess.run(['git', '-C', str(ancestor), 'rev-parse', '--show-toplevel'],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            if any((item / '.git').exists() for item in (ancestor, *ancestor.parents)):
                raise ValueError('cannot verify checkpoint output privacy')
            return
        repository = Path(result.stdout.rstrip('\r\n')).resolve()
        if not directory.is_relative_to(repository):
            raise ValueError('cannot resolve checkpoint output inside Git worktree')
        relative = directory.relative_to(repository).as_posix().rstrip('/') + '/'
        ignored = subprocess.run(['git', '-C', str(repository), 'check-ignore', '-q', '--no-index', '--', relative],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        tracked = subprocess.run(['git', '-C', str(repository), 'ls-files', '--', relative],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        if ignored.returncode != 0 or tracked.returncode != 0 or tracked.stdout:
            raise ValueError('checkpoint output directory must be ignored and untracked')
    except OSError:
        raise ValueError('cannot verify checkpoint output privacy') from None


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def continuation_config(config: dict) -> str:
    # Resource limits and save frequency do not change the training objective.
    mutable = {'steps', 'checkpoint_every', 'max_parameters', 'max_attention_elements', 'max_dataset_bytes',
               'max_manifest_bytes', 'max_checkpoint_bytes', 'max_optimizer_bytes', 'max_examples', 'max_record_bytes'}
    return sha(canonical({key: value for key, value in config.items() if key not in mutable}))


def _write(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def snapshot(root: Path, model, optimizer, rng, progress: dict, identity: dict, config: dict) -> Path:
    states = optimizer.state_dict()
    tensors = {'rng.cpu': torch.get_rng_state().cpu().contiguous()}
    for index, values in states['state'].items():
        for name, tensor in values.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError('unsupported non-tensor optimizer state')
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
                raise ValueError('non-finite optimizer checkpoint state')
            tensors[f'optimizer.{index}.{name}'] = tensor.detach().cpu().contiguous()
    if config['device'] == 'cuda':
        index = config['cuda_device_index']
        tensors[f'rng.cuda.{index}'] = torch.cuda.get_rng_state(index).cpu().contiguous()
    model_state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in model_state.values()):
        raise ValueError('non-finite model checkpoint state')
    model_raw = save(model_state)
    optimizer_raw = save(tensors)
    state = {'schema_version': 'lfg.owned-assessment-sft-checkpoint.v1', 'identity': identity,
             'progress': progress, 'sampler_state': rng.getstate(), 'param_groups': states['param_groups'],
             'model_sha256': sha(model_raw), 'optimizer_sha256': sha(optimizer_raw), 'device': config['device']}
    state_raw = canonical(state)
    if (len(model_raw) > config['max_checkpoint_bytes'] or len(optimizer_raw) > config['max_optimizer_bytes']
            or len(state_raw) > config['max_manifest_bytes']):
        raise ValueError('checkpoint exceeds configured size limits')
    checkpoints = root / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    final = checkpoints / f"step-{progress['step']:016d}"
    if final.exists():
        raise ValueError('checkpoint step already exists')
    temporary = Path(tempfile.mkdtemp(prefix='.writing-', dir=checkpoints))
    _write(temporary / 'weights.safetensors', model_raw)
    _write(temporary / 'optimizer.safetensors', optimizer_raw)
    _write(temporary / 'state.json', state_raw)
    os.replace(temporary, final)
    # The checkpoint itself is authoritative. A failed pointer update leaves a
    # complete step directory that can be selected explicitly.
    pointer = canonical({'schema_version': 'lfg.owned-assessment-sft-latest.v1', 'checkpoint': final.name,
                         'step': progress['step'], 'state_sha256': sha(state_raw)})
    fd, name = tempfile.mkstemp(prefix='.latest-', dir=root)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(pointer)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, root / 'latest.json')
    return final


def _read(path: Path, limit: int) -> bytes:
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('checkpoint file exceeds configured limit')
    return raw


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def restore(path: Path, expected_state_sha256: str, model, optimizer, rng, identity: dict, config: dict) -> dict:
    state_raw = _read(path / 'state.json', config['max_manifest_bytes'])
    if not isinstance(expected_state_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_state_sha256) \
            or sha(state_raw) != expected_state_sha256:
        raise ValueError('checkpoint state digest mismatch')
    state = json.loads(state_raw)
    weights = _read(path / 'weights.safetensors', config['max_checkpoint_bytes'])
    optimizer_raw = _read(path / 'optimizer.safetensors', config['max_optimizer_bytes'])
    if (not isinstance(state, dict) or state.get('schema_version') != 'lfg.owned-assessment-sft-checkpoint.v1'
            or state.get('identity') != identity or state.get('device') != config['device']
            or state.get('model_sha256') != sha(weights) or state.get('optimizer_sha256') != sha(optimizer_raw)
            or not isinstance(state.get('progress'), dict) or type(state['progress'].get('step')) is not int
            or not 0 < state['progress']['step'] <= config['steps']
            or type(state['progress'].get('supervised_tokens')) is not int
            or state['progress']['supervised_tokens'] < 0
            or not isinstance(state['progress'].get('losses'), list)
            or len(state['progress']['losses']) != state['progress']['step']
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in state['progress']['losses'])
            or not isinstance(state.get('param_groups'), list)):
        raise ValueError('checkpoint identity, progress or digest mismatch')
    model.load_state_dict(load(weights), strict=True)
    tensors = load(optimizer_raw)
    if 'rng.cpu' not in tensors:
        raise ValueError('checkpoint is missing CPU random state')
    optimizer_states = {}
    for key, value in tensors.items():
        if key.startswith('optimizer.'):
            pieces = key.split('.', 2)
            if len(pieces) != 3 or not pieces[1].isdigit() or not pieces[2]:
                raise ValueError('invalid optimizer checkpoint key')
            if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
                raise ValueError('non-finite optimizer checkpoint state')
            optimizer_states.setdefault(int(pieces[1]), {})[pieces[2]] = value
        elif key != 'rng.cpu' and not key.startswith('rng.cuda.'):
            raise ValueError('unsupported checkpoint tensor')
    optimizer.load_state_dict({'state': optimizer_states, 'param_groups': state['param_groups']})
    rng.setstate(_tuples(state['sampler_state']))
    torch.set_rng_state(tensors['rng.cpu'])
    cuda_keys = sorted((key for key in tensors if key.startswith('rng.cuda.')),
                       key=lambda key: int(key.rsplit('.', 1)[1]))
    if config['device'] == 'cuda':
        index = config['cuda_device_index']
        if cuda_keys != [f'rng.cuda.{index}']:
            raise ValueError('checkpoint CUDA device index changed')
        torch.cuda.set_rng_state(tensors[cuda_keys[0]], device=index)
    elif cuda_keys:
        raise ValueError('unexpected CUDA random state in CPU checkpoint')
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError('checkpoint contains non-finite model weights')
    return state['progress']
