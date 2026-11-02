#!/usr/bin/env python3
"""Run a bounded, CPU-only, unqualified from-scratch next-byte experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import safetensors
from safetensors.torch import save
import torch
from torch.nn import functional as F

import owned_student
import corpus_documents
import screen_private_corpus


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_bounded(path: Path, limit: int) -> bytes:
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('input exceeds configured limit')
    return raw


def load_config(path: Path) -> tuple[dict, str]:
    raw = read_bounded(path, 65536)
    config = json.loads(raw)
    integers = {'seed', 'cpu_threads', 'batch_size', 'steps', 'max_data_bytes', 'max_parameters'}
    fields = integers | {'schema_version', 'architecture', 'learning_rate', 'weight_decay'}
    if (not isinstance(config, dict) or set(config) != fields
            or config['schema_version'] != 'lfg.owned-student-experiment.v1'
            or any(type(config[key]) is not int or config[key] < (0 if key == 'seed' else 1) for key in integers)
            or any(type(config[key]) not in (int, float) or not math.isfinite(config[key]) for key in ('learning_rate', 'weight_decay'))
            or config['learning_rate'] <= 0 or config['weight_decay'] < 0):
        raise ValueError('invalid experiment configuration')
    owned_student.validate_architecture(config['architecture'])
    return config, sha(raw)


def require_private_outputs(directory: Path) -> None:
    directory = directory.resolve()
    ancestor = directory.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    try:
        root = subprocess.run(['git', '-C', str(ancestor), 'rev-parse', '--show-toplevel'],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, text=True)
        if root.returncode:
            if any((item / '.git').exists() for item in (ancestor, *ancestor.parents)):
                raise ValueError('cannot verify output privacy')
            return
        repository = Path(root.stdout.rstrip('\r\n')).resolve()
        if not directory.is_relative_to(repository):
            raise ValueError('cannot verify output privacy')
        for name in ('weights.safetensors', 'manifest.json'):
            relative = (directory / name).relative_to(repository).as_posix()
            tracked = subprocess.run(['git', '-C', str(repository), 'ls-files', '--error-unmatch', '--', relative],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            ignored = subprocess.run(['git', '-C', str(repository), 'check-ignore', '-q', '--no-index', '--', relative],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if tracked.returncode != 1 or ignored.returncode != 0:
                raise ValueError('outputs must be ignored and untracked')
    except OSError:
        raise ValueError('cannot verify output privacy') from None


def prepare_source(source_path: Path | None, plan_path: Path | None, policy_path: Path | None, config: dict) -> dict:
    if source_path is not None and plan_path is None and policy_path is None:
        source = read_bounded(source_path, config['max_data_bytes'])
        if len(source) <= config['architecture']['context']:
            raise ValueError('source is shorter than the configured context')
        return {'documents': [source], 'source_sha256': sha(source), 'source_bytes': len(source),
                'source_path': source_path, 'corpus': None}
    if source_path is None and plan_path is not None and policy_path is not None:
        corpus = corpus_documents.load(plan_path, policy_path, config['max_data_bytes'], config['architecture']['context'])
        return {'documents': corpus['documents'], 'source_sha256': corpus['corpus_sha256'],
                'source_bytes': sum(map(len, corpus['documents'])), 'source_path': None, 'corpus': corpus}
    raise ValueError('select either one source or a corpus plan and screening policy')


def verify_source(source: dict, config: dict) -> None:
    if source['corpus'] is not None:
        corpus_documents.verify(source['corpus'])
    elif read_bounded(source['source_path'], config['max_data_bytes']) != source['documents'][0]:
        raise ValueError('training source changed')


def experiment(config_path: Path, source_path: Path | None, output: Path, *,
               corpus_plan: Path | None = None, screen_policy: Path | None = None) -> dict:
    config, config_sha = load_config(config_path)
    source = prepare_source(source_path, corpus_plan, screen_policy, config)
    context = config['architecture']['context']
    if output.exists():
        raise ValueError('output already exists')
    require_private_outputs(output)
    code_paths = (Path(__file__), Path(owned_student.__file__), Path(corpus_documents.__file__),
                  Path(screen_private_corpus.__file__))
    code_hashes = {path.name: sha(path.read_bytes()) for path in code_paths}
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed'])
    with torch.device('meta'):
        template = owned_student.OwnedByteStudent(config['architecture'])
    parameters = sum(parameter.numel() for parameter in template.parameters())
    if parameters > config['max_parameters']:
        raise ValueError('model exceeds configured parameter limit')
    del template
    model = owned_student.OwnedByteStudent(config['architecture']).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = random.Random(config['seed'])
    sampler = corpus_documents.sampler(source, context)
    losses = []
    start_time = time.perf_counter()
    for _ in range(config['steps']):
        input_rows, target_rows = corpus_documents.batch(source, generator, config['batch_size'], context, sampler)
        inputs = torch.tensor(input_rows, dtype=torch.long)
        targets = torch.tensor(target_rows, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise ValueError('non-finite training loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    duration = time.perf_counter() - start_time
    verify_source(source, config)
    if (load_config(config_path)[1] != config_sha
            or any(sha(path.read_bytes()) != code_hashes[path.name] for path in code_paths)):
        raise ValueError('experiment inputs changed during run')
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError('non-finite model weights')
    weights = save({name: tensor.detach().contiguous() for name, tensor in model.state_dict().items()})
    manifest = {'schema_version': 'lfg.owned-student-experiment-result.v1', 'architecture': config['architecture'],
                'config_sha256': config_sha, 'source_sha256': source['source_sha256'], 'source_bytes': source['source_bytes'],
                'code_sha256': code_hashes, 'python': platform.python_version(), 'torch': str(torch.__version__),
                'safetensors': safetensors.__version__, 'platform': platform.platform(), 'device': 'cpu',
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'parameters': parameters, 'steps': config['steps'], 'batch_size': config['batch_size'],
                'tokens_processed': config['steps'] * config['batch_size'] * context,
                'losses': losses, 'elapsed_seconds': duration, 'weights_sha256': sha(weights),
                'validation_performed': False, 'assessment_training_performed': False,
                'teacher_distillation_performed': False, 'promotion_eligible': False}
    if source['corpus'] is not None:
        manifest['corpus'] = source['corpus']['identity']
    output.mkdir(mode=0o700)
    for name, payload in (('weights.safetensors', weights),
                          ('manifest.json', (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())):
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path)
    inputs.add_argument('--corpus-plan', type=Path)
    parser.add_argument('--screen-policy', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = experiment(args.config, args.source, args.output, corpus_plan=args.corpus_plan,
                            screen_policy=args.screen_policy)
        print(json.dumps({key: result[key] for key in ('parameters', 'source_bytes', 'steps', 'tokens_processed',
              'weights_sha256', 'validation_performed', 'teacher_distillation_performed', 'promotion_eligible')}))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError, RuntimeError):
        raise SystemExit('Owned student experiment failed; check inputs, limits and execution environment.') from None


if __name__ == '__main__':
    main()
