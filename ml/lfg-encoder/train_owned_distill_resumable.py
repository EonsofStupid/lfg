#!/usr/bin/env python3
"""Resume bounded owned-student byte distillation on CPU or an explicit CUDA device."""
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
from safetensors.torch import save
import torch
from torch.nn import functional as F

import distill_owned_student as distill
import owned_sft_checkpoint as checkpoints
import owned_student
import train_owned_student as base
import training_inputs

CHECKPOINT_SCHEMA = 'lfg.owned-distillation-checkpoint.v1'


def runtime_config(path: Path, limit: int = 65536) -> tuple[dict, bytes]:
    if type(limit) is not int or limit < 1:
        raise ValueError('runtime configuration byte limit must be positive')
    raw = base.read_bounded(path, limit)
    value = json.loads(raw)
    integers = ('cuda_device_index', 'checkpoint_every', 'max_checkpoint_bytes', 'max_optimizer_bytes', 'max_manifest_bytes')
    if (not isinstance(value, dict) or value.get('schema_version') != 'lfg.owned-distillation-runtime.v1'
            or value.get('device') not in ('cpu', 'cuda')
            or any(type(value.get(k)) is not int or value[k] < (0 if k == 'cuda_device_index' else 1) for k in integers)
            or type(value.get('cuda_memory_fraction')) not in (int, float)
            or not math.isfinite(value['cuda_memory_fraction']) or not 0 < value['cuda_memory_fraction'] <= 1
            or type(value.get('deterministic_algorithms')) is not bool or type(value.get('allow_tf32')) is not bool):
        raise ValueError('invalid owned distillation runtime configuration')
    return value, raw


def experiment(student_path: Path, distill_path: Path, runtime_path: Path, plan: Path, policy: Path,
               teacher_manifest: Path, teacher_weights: Path, teacher_format: str, teacher_revision: str | None,
               output: Path, *, resume: Path | None = None, resume_state_sha256: str | None = None,
               max_steps_this_run: int | None = None, max_runtime_config_bytes: int = 65536) -> dict:
    student_config, student_sha = base.load_config(student_path)
    distill_config, distill_sha = distill.load_config(distill_path)
    runtime, runtime_raw = runtime_config(runtime_path, max_runtime_config_bytes)
    if (resume is None) != (resume_state_sha256 is None):
        raise ValueError('resume and resume state digest must be supplied together')
    if max_steps_this_run is not None and (type(max_steps_this_run) is not int or max_steps_this_run < 1):
        raise ValueError('max_steps_this_run must be positive')
    source = base.prepare_source(None, plan, policy, student_config)
    context = student_config['architecture']['context']
    if output.exists():
        raise ValueError('output already exists')
    checkpoints.private_directory(output)
    code_paths = (Path(__file__), Path(base.__file__), Path(distill.__file__), Path(owned_student.__file__),
                  Path(checkpoints.__file__), Path(base.corpus_documents.__file__), Path(base.screen_private_corpus.__file__),
                  Path(training_inputs.__file__), Path(training_inputs.lock_wheelhouse.__file__))
    code = {path.name: base.sha(path.read_bytes()) for path in code_paths}
    if runtime['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('requested CUDA device is unavailable')
    if runtime['device'] == 'cuda' and runtime['cuda_device_index'] >= torch.cuda.device_count():
        raise ValueError('requested CUDA device index is unavailable')
    torch.set_num_threads(student_config['cpu_threads'])
    torch.use_deterministic_algorithms(runtime['deterministic_algorithms'])
    torch.backends.cuda.matmul.allow_tf32 = runtime['allow_tf32']
    torch.backends.cudnn.allow_tf32 = runtime['allow_tf32']
    device = torch.device('cpu')
    if runtime['device'] == 'cuda':
        device = torch.device('cuda', runtime['cuda_device_index'])
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(runtime['cuda_memory_fraction'], device=device)
        torch.cuda.reset_peak_memory_stats(device)
    if teacher_format == 'owned-byte':
        teacher, teacher_id = distill.load_teacher(teacher_manifest, teacher_weights, distill_config, context, code['owned_student.py'])
    elif teacher_format == 'gpt2-byte' and teacher_revision:
        teacher, teacher_id = distill.load_gpt2_teacher(teacher_weights, teacher_manifest, teacher_revision, distill_config, context)
    else:
        raise ValueError('unsupported teacher format or missing revision')
    teacher.to(device).eval()
    torch.manual_seed(student_config['seed'])
    if runtime['device'] == 'cuda':
        torch.cuda.manual_seed(student_config['seed'])
    with torch.device('meta'):
        template = owned_student.OwnedByteStudent(student_config['architecture'])
    parameters = sum(item.numel() for item in template.parameters())
    del template
    if parameters > student_config['max_parameters']:
        raise ValueError('student exceeds configured parameter limit')
    model = owned_student.OwnedByteStudent(student_config['architecture']).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=student_config['learning_rate'],
                                  weight_decay=student_config['weight_decay'])
    rng = random.Random(student_config['seed'])
    sampler = base.corpus_documents.sampler(source, context)
    resume_config = {**student_config, **runtime, 'distillation': distill_config}
    identity = {'kind': 'owned-byte-distillation', 'source_sha256': source['source_sha256'], 'teacher': teacher_id,
                'code_sha256': code, 'continuation_config_sha256': checkpoints.continuation_config(resume_config),
                'architecture': student_config['architecture'], 'python': platform.python_version(),
                'torch': str(torch.__version__), 'safetensors': safetensors.__version__,
                'cuda_version': torch.version.cuda if runtime['device'] == 'cuda' else None,
                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG') if runtime['device'] == 'cuda' else None,
                'cuda_devices': ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                                 if runtime['device'] == 'cuda' else [])}
    checkpoint_config = {**runtime, 'steps': student_config['steps']}
    initial_step, losses, ce_losses, distill_losses, tokens = 0, [], [], [], 0
    if resume is not None:
        progress = checkpoints.restore(resume, resume_state_sha256, model, optimizer, rng, identity,
                                       checkpoint_config, schema=CHECKPOINT_SCHEMA)
        initial_step, losses, tokens = progress['step'], progress['losses'], progress['supervised_tokens']
        ce_losses, distill_losses = progress.get('ce_losses'), progress.get('distillation_losses')
        if (initial_step > student_config['steps'] or not isinstance(ce_losses, list) or not isinstance(distill_losses, list)
                or len(ce_losses) != initial_step or len(distill_losses) != initial_step
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in ce_losses + distill_losses)):
            raise ValueError('invalid distillation checkpoint progress')
    end_step = min(student_config['steps'], initial_step + max_steps_this_run) if max_steps_this_run else student_config['steps']
    output.mkdir(mode=0o700)
    start_time, latest = time.perf_counter(), None
    for step in range(initial_step + 1, end_step + 1):
        inputs, targets = base.corpus_documents.batch(source, rng, student_config['batch_size'], context, sampler)
        input_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
        target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = teacher(input_tensor, use_cache=False).logits if teacher_format == 'gpt2-byte' else teacher(input_tensor)
            indices, top, tail = distill.teacher_distribution(logits, distill_config['top_k'])
        indices, top, tail = (item.clone() for item in (indices, top, tail))
        optimizer.zero_grad(set_to_none=True)
        student_logits = model(input_tensor)
        ce = F.cross_entropy(student_logits.reshape(-1, 256), target_tensor.reshape(-1))
        kd = owned_student.decoupled_topk_loss(student_logits, indices, top, tail, distill_config['temperature'])
        loss = ce + distill_config['distillation_weight'] * kd
        if not bool(torch.isfinite(loss)):
            raise ValueError('non-finite distillation loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ce_losses.append(float(ce.detach().cpu()))
        distill_losses.append(float(kd.detach().cpu()))
        tokens += student_config['batch_size'] * context
        if step % runtime['checkpoint_every'] == 0 or step == end_step:
            progress = {'step': step, 'losses': losses, 'ce_losses': ce_losses,
                        'distillation_losses': distill_losses, 'supervised_tokens': tokens}
            latest = checkpoints.snapshot(output, model, optimizer, rng, progress, identity,
                                          checkpoint_config, schema=CHECKPOINT_SCHEMA)
    elapsed = time.perf_counter() - start_time
    base.verify_source(source, student_config)
    if teacher_format == 'gpt2-byte':
        teacher_current = training_inputs.verify_model(teacher_weights, teacher_manifest, teacher_revision) == {
            key: teacher_id[key] for key in ('model_id', 'revision', 'manifest_sha256', 'files')}
    else:
        teacher_current = (base.sha(base.read_bounded(teacher_manifest, distill_config['max_teacher_manifest_bytes'])) == teacher_id['manifest_sha256']
                           and base.sha(base.read_bounded(teacher_weights, distill_config['max_teacher_bytes'])) == teacher_id['weights_sha256'])
    if (not teacher_current or base.load_config(student_path)[1] != student_sha or distill.load_config(distill_path)[1] != distill_sha
            or runtime_config(runtime_path, max_runtime_config_bytes)[1] != runtime_raw
            or any(base.sha(path.read_bytes()) != code[path.name] for path in code_paths)):
        raise ValueError('distillation inputs changed during execution')
    if end_step < student_config['steps']:
        partial = {'schema_version': 'lfg.owned-distillation-partial.v1', 'completed': False, 'steps': end_step,
                'tokens_processed': tokens, 'latest_checkpoint': str(latest),
                'latest_checkpoint_state_sha256': base.sha(base.read_bounded(latest / 'state.json', runtime['max_manifest_bytes'])),
                'elapsed_seconds': elapsed, 'device': runtime['device'],
                'assessment_quality_evaluated': False, 'promotion_eligible': False}
        if runtime['device'] == 'cuda':
            partial.update(cuda_device=torch.cuda.get_device_name(device), cuda_memory_fraction=runtime['cuda_memory_fraction'],
                           cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        return partial
    weights = save({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()})
    manifest = {'schema_version': 'lfg.owned-distillation-result.v1', 'completed': True, 'resumed_from_step': initial_step,
                'resume_checkpoint_state_sha256': resume_state_sha256, 'architecture': student_config['architecture'],
                'student_config_sha256': student_sha, 'distillation_config_sha256': distill_sha,
                'runtime_config_sha256': base.sha(runtime_raw), 'source_sha256': source['source_sha256'],
                'source_bytes': source['source_bytes'], 'code_sha256': code, 'teacher': teacher_id,
                'parameters': parameters, 'steps': student_config['steps'], 'tokens_processed': tokens,
                'losses': [{'cross_entropy': ce, 'distillation': kd, 'total': total}
                           for ce, kd, total in zip(ce_losses, distill_losses, losses, strict=True)],
                'elapsed_seconds': elapsed, 'weights_sha256': base.sha(weights), 'device': runtime['device'],
                'python': platform.python_version(), 'platform': platform.platform(), 'torch': str(torch.__version__),
                'safetensors': safetensors.__version__, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'allow_tf32': runtime['allow_tf32'], 'teacher_distillation_performed': True,
                'validation_performed': False, 'test_opened': False, 'assessment_training_performed': False,
                'assessment_quality_evaluated': False, 'promotion_eligible': False, 'corpus': source['corpus']['identity']}
    if runtime['device'] == 'cuda':
        manifest.update(cuda_device=torch.cuda.get_device_name(device), cuda_device_index=runtime['cuda_device_index'],
                        cuda_memory_fraction=runtime['cuda_memory_fraction'], cuda_version=torch.version.cuda,
                        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    for name, raw in (('weights.safetensors', weights),
                      ('manifest.json', (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())):
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('student-config', 'distillation-config', 'runtime-config', 'corpus-plan', 'screen-policy',
                 'teacher-manifest', 'teacher-weights', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--teacher-format', choices=('owned-byte', 'gpt2-byte'), required=True)
    parser.add_argument('--teacher-revision')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--resume-state-sha256')
    parser.add_argument('--max-steps-this-run', type=int)
    parser.add_argument('--max-runtime-config-bytes', type=int, default=65536)
    args = parser.parse_args()
    try:
        result = experiment(args.student_config, args.distillation_config, args.runtime_config, args.corpus_plan,
                args.screen_policy, args.teacher_manifest, args.teacher_weights, args.teacher_format,
                args.teacher_revision, args.output, resume=args.resume, resume_state_sha256=args.resume_state_sha256,
                max_steps_this_run=args.max_steps_this_run, max_runtime_config_bytes=args.max_runtime_config_bytes)
        print(json.dumps({key: result.get(key) for key in ('completed', 'steps', 'tokens_processed', 'weights_sha256',
             'latest_checkpoint', 'latest_checkpoint_state_sha256', 'elapsed_seconds', 'device', 'cuda_device',
             'cuda_memory_fraction', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes',
             'assessment_quality_evaluated', 'promotion_eligible')}))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError, RuntimeError) as error:
        raise SystemExit(f'Owned distillation failed ({type(error).__name__}): {error}') from None
