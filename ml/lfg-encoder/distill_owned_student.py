#!/usr/bin/env python3
"""Experimental online Top-K distillation from a local, byte-aligned teacher."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import safetensors
from safetensors.torch import load, save
import torch
from torch.nn import functional as F

import owned_student
import train_owned_student as base


def load_config(path: Path) -> tuple[dict, str]:
    raw = base.read_bounded(path, 65536)
    value = json.loads(raw)
    integers = ('top_k', 'max_teacher_bytes', 'max_teacher_manifest_bytes', 'max_teacher_parameters')
    if (not isinstance(value, dict) or set(value) != set(integers) | {'schema_version', 'temperature', 'distillation_weight'}
            or value['schema_version'] != 'lfg.owned-distillation-config.v1'
            or any(type(value[name]) is not int or value[name] < 1 for name in integers)
            or value['top_k'] >= 256
            or any(type(value[name]) not in (int, float) or not math.isfinite(value[name])
                   for name in ('temperature', 'distillation_weight'))
            or value['temperature'] < 1 or value['distillation_weight'] <= 0):
        raise ValueError('invalid distillation configuration')
    return value, base.sha(raw)


def teacher_distribution(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get untempered top probabilities and tail mass with deterministic tie order."""
    if logits.shape[-1] != 256 or not bool(torch.isfinite(logits).all()) or not 1 <= k < logits.shape[-1]:
        raise ValueError('invalid teacher logits or Top-K')
    logits = logits.to(dtype=torch.float64)
    indices = torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :k]
    normalizer = torch.logsumexp(logits, dim=-1)
    selected = logits.gather(-1, indices)
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, indices, True)
    top_probabilities = (selected - normalizer.unsqueeze(-1)).exp()
    tail = (torch.logsumexp(logits.masked_fill(mask, -torch.inf), dim=-1) - normalizer).exp()
    return indices, top_probabilities, tail


def load_teacher(manifest_path: Path, weights_path: Path, config: dict,
                 student_context: int, model_code_sha: str) -> tuple[torch.nn.Module, dict]:
    manifest_raw = base.read_bounded(manifest_path, config['max_teacher_manifest_bytes'])
    weights_raw = base.read_bounded(weights_path, config['max_teacher_bytes'])
    manifest = json.loads(manifest_raw)
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 'lfg.owned-student-experiment-result.v1'
            or manifest.get('weights_sha256') != base.sha(weights_raw)
            or not isinstance(manifest.get('code_sha256'), dict)
            or manifest['code_sha256'].get('owned_student.py') != model_code_sha):
        raise ValueError('teacher identity or implementation does not match')
    architecture = manifest.get('architecture')
    owned_student.validate_architecture(architecture)
    if architecture['context'] < student_context:
        raise ValueError('teacher context is shorter than student context')
    with torch.device('meta'):
        template = owned_student.OwnedByteStudent(architecture)
    if sum(parameter.numel() for parameter in template.parameters()) > config['max_teacher_parameters']:
        raise ValueError('teacher exceeds configured parameter limit')
    del template
    teacher = owned_student.OwnedByteStudent(architecture)
    teacher.load_state_dict(load(weights_raw), strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in teacher.state_dict().values()):
        raise ValueError('teacher contains non-finite weights')
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, {'manifest_sha256': base.sha(manifest_raw), 'weights_sha256': base.sha(weights_raw),
                     'architecture': architecture, 'kind': 'owned-byte-experimental',
                     'quality_verified': False}


def experiment(student_config_path: Path, distill_config_path: Path, source_path: Path,
               teacher_manifest: Path, teacher_weights: Path, output: Path) -> dict:
    student_config, student_config_sha = base.load_config(student_config_path)
    distill_config, distill_config_sha = load_config(distill_config_path)
    source = base.read_bounded(source_path, student_config['max_data_bytes'])
    context = student_config['architecture']['context']
    if len(source) <= context:
        raise ValueError('source is shorter than the configured context')
    if output.exists():
        raise ValueError('output already exists')
    base.require_private_outputs(output)
    paths = (Path(__file__), Path(base.__file__), Path(owned_student.__file__))
    code = {path.name: base.sha(path.read_bytes()) for path in paths}
    torch.set_num_threads(student_config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    # Teacher construction uses random initialization before loading weights; seed
    # the student separately so initialization does not depend on teacher shape.
    teacher, teacher_id = load_teacher(teacher_manifest, teacher_weights, distill_config, context,
                                       code['owned_student.py'])
    torch.manual_seed(student_config['seed'])
    with torch.device('meta'):
        template = owned_student.OwnedByteStudent(student_config['architecture'])
    parameter_count = sum(parameter.numel() for parameter in template.parameters())
    if parameter_count > student_config['max_parameters']:
        raise ValueError('student exceeds configured parameter limit')
    del template
    student = owned_student.OwnedByteStudent(student_config['architecture']).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=student_config['learning_rate'],
                                  weight_decay=student_config['weight_decay'])
    generator = random.Random(student_config['seed'])
    history = []
    start_time = time.perf_counter()
    for _ in range(student_config['steps']):
        starts = [generator.randrange(len(source) - context) for _ in range(student_config['batch_size'])]
        inputs = torch.tensor([list(source[start:start + context]) for start in starts], dtype=torch.long)
        targets = torch.tensor([list(source[start + 1:start + context + 1]) for start in starts], dtype=torch.long)
        with torch.inference_mode():
            teacher_logits = teacher(inputs)
            indices, top_probabilities, tail_probability = teacher_distribution(teacher_logits, distill_config['top_k'])
        # Inference tensors must be copied before an autograd operation saves them.
        indices, top_probabilities, tail_probability = (value.clone() for value in (indices, top_probabilities, tail_probability))
        optimizer.zero_grad(set_to_none=True)
        student_logits = student(inputs)
        cross_entropy = F.cross_entropy(student_logits.reshape(-1, 256), targets.reshape(-1))
        distillation = owned_student.decoupled_topk_loss(student_logits, indices, top_probabilities,
                                                         tail_probability, distill_config['temperature'])
        loss = cross_entropy + distill_config['distillation_weight'] * distillation
        if not bool(torch.isfinite(loss)):
            raise ValueError('non-finite training loss')
        loss.backward()
        optimizer.step()
        history.append({'cross_entropy': float(cross_entropy.detach()),
                        'distillation': float(distillation.detach()), 'total': float(loss.detach())})
    elapsed = time.perf_counter() - start_time
    if (base.load_config(student_config_path)[1] != student_config_sha or load_config(distill_config_path)[1] != distill_config_sha
            or base.read_bounded(source_path, student_config['max_data_bytes']) != source
            or base.sha(base.read_bounded(teacher_manifest, distill_config['max_teacher_manifest_bytes'])) != teacher_id['manifest_sha256']
            or base.sha(base.read_bounded(teacher_weights, distill_config['max_teacher_bytes'])) != teacher_id['weights_sha256']
            or any(base.sha(path.read_bytes()) != code[path.name] for path in paths)):
        raise ValueError('distillation inputs changed during run')
    if any(not bool(torch.isfinite(value).all()) for value in student.state_dict().values()):
        raise ValueError('student has non-finite weights')
    weights = save({name: value.detach().contiguous() for name, value in student.state_dict().items()})
    manifest = {'schema_version': 'lfg.owned-distillation-result.v1', 'architecture': student_config['architecture'],
                'student_config_sha256': student_config_sha, 'distillation_config_sha256': distill_config_sha,
                'source_sha256': base.sha(source), 'source_bytes': len(source), 'code_sha256': code,
                'teacher': teacher_id, 'parameters': parameter_count, 'steps': student_config['steps'],
                'tokens_processed': student_config['steps'] * student_config['batch_size'] * context,
                'losses': history, 'elapsed_seconds': elapsed, 'weights_sha256': base.sha(weights),
                'python': platform.python_version(), 'platform': platform.platform(), 'torch': str(torch.__version__),
                'safetensors': safetensors.__version__, 'device': 'cpu',
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'teacher_distillation_performed': True, 'validation_performed': False,
                'assessment_training_performed': False, 'promotion_eligible': False}
    output.mkdir(mode=0o700)
    for name, raw in (('weights.safetensors', weights),
                      ('manifest.json', (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())):
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('student-config', 'distillation-config', 'source', 'teacher-manifest', 'teacher-weights', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = experiment(args.student_config, args.distillation_config, args.source,
                            args.teacher_manifest, args.teacher_weights, args.output)
        print(json.dumps({key: result[key] for key in ('parameters', 'source_bytes', 'steps', 'tokens_processed',
              'weights_sha256', 'teacher_distillation_performed', 'validation_performed', 'promotion_eligible')}))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError, RuntimeError):
        raise SystemExit('Owned distillation failed; check inputs, identities, limits and environment.') from None


if __name__ == '__main__':
    main()
