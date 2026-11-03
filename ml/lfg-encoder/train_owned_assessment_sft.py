#!/usr/bin/env python3
"""Fine-tune an owned byte student on explicitly prepared local teacher traces."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import time

import safetensors
from safetensors.torch import load, save
import torch
from torch.nn import functional as F

import owned_student
import owned_sft_checkpoint as checkpoints
import train_owned_student as base


FRAMING_FIELDS = ('system_prefix', 'system_suffix', 'user_prefix', 'user_suffix',
                 'assistant_prefix', 'assistant_suffix')


def framing_sha256(framing: dict) -> str:
    """Hash the exact delimiter strings used to render every SFT example."""
    canonical = json.dumps({key: framing[key] for key in FRAMING_FIELDS}, sort_keys=True,
                           ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def configuration(path: Path, max_bytes: int = 65536) -> tuple[dict, bytes]:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('max config bytes must be positive')
    raw = base.read_bounded(path, max_bytes)
    config = json.loads(raw)
    integers = ('seed', 'cpu_threads', 'batch_size', 'steps', 'context', 'max_parameters',
                'max_attention_elements', 'max_dataset_bytes', 'max_manifest_bytes',
                'max_checkpoint_bytes', 'max_optimizer_bytes', 'max_examples', 'max_record_bytes', 'checkpoint_every')
    framing_keys = ('system_prefix', 'system_suffix', 'user_prefix', 'user_suffix',
                    'assistant_prefix', 'assistant_suffix')
    weights = {'review_status_weights': ('reviewed', 'unreviewed'),
               'outcome_weights': ('success', 'unknown', 'failure')}
    if (not isinstance(config, dict) or config.get('schema_version') != 'lfg.owned-assessment-sft-config.v1'
            or any(type(config.get(k)) is not int or config[k] < (0 if k == 'seed' else 1) for k in integers)
            or config.get('device') not in ('cpu', 'cuda')
            or type(config.get('cuda_device_index')) is not int or config['cuda_device_index'] < 0
            or type(config.get('cuda_memory_fraction')) not in (int, float)
            or not math.isfinite(config['cuda_memory_fraction']) or not 0 < config['cuda_memory_fraction'] <= 1
            or type(config.get('deterministic_algorithms')) is not bool
            or type(config.get('allow_tf32')) is not bool
            or type(config.get('allow_truncated_prompt')) is not bool
            or any(type(config.get(k)) not in (int, float) or not math.isfinite(config[k])
                   for k in ('learning_rate', 'weight_decay'))
            or config['learning_rate'] <= 0 or config['weight_decay'] < 0
            or not isinstance(config.get('framing'), dict)
            or any(not isinstance(config['framing'].get(k), str) for k in framing_keys)):
        raise ValueError('invalid supervised training configuration')
    for key, names in weights.items():
        value = config.get(key)
        if (not isinstance(value, dict) or any(type(value.get(n)) not in (int, float)
                or not math.isfinite(value[n]) or value[n] < 0 for n in names)):
            raise ValueError('invalid example weights')
    return config, raw


def read_training(dataset: Path, config: dict) -> tuple[list[dict], bytes, bytes, dict]:
    manifest_raw = base.read_bounded(dataset / 'manifest.json', config['max_manifest_bytes'])
    manifest = json.loads(manifest_raw)
    files = manifest.get('files') if isinstance(manifest, dict) else None
    sources = manifest.get('sources') if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 'lfg.assessment-distillation-dataset.v1'
            or not isinstance(files, dict) or not isinstance(files.get('train.jsonl'), str)
            or any(not isinstance(name, str) or re.match(r'(?i)^(test|held[-_]?out)([._/-]|$)',
                   name.replace('\\', '/').rsplit('/', 1)[-1]) for name in files)
            or not isinstance(sources, list) or not sources
            or any(not isinstance(item, dict) or item.get('split') not in ('train', 'validation')
                   or item.get('origin') not in ('owner_chat', 'owner_project')
                   or any(not isinstance(item.get(k), str) or not item[k].strip()
                          for k in ('project', 'project_group', 'sha256'))
                   or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']) for item in sources)):
        raise ValueError('unsupported development dataset or held-out file declared')
    train_sources = {(item['sha256'], item['project'], item['project_group'], item['origin'])
                     for item in sources if item['split'] == 'train'}
    projects, groups, source_splits = {}, {}, {}
    for item in sources:
        project, group, split, source_sha = (item['project'], item['project_group'], item['split'], item['sha256'])
        if ((project in projects and projects[project] != (group, split))
                or (group in groups and groups[group] != split)
                or (source_sha in source_splits and source_splits[source_sha] != split)):
            raise ValueError('declared training and validation sources overlap')
        projects[project], groups[group], source_splits[source_sha] = (group, split), split, split
    # Only the training file is opened. A declared validation file is not used.
    train_raw = base.read_bounded(dataset / 'train.jsonl', config['max_dataset_bytes'])
    if base.sha(train_raw) != files['train.jsonl']:
        raise ValueError('training dataset digest mismatch')
    records, identifiers, counts = [], set(), Counter()
    system_sha = None
    for line in train_raw.splitlines():
        if not line.strip() or len(line) > config['max_record_bytes']:
            raise ValueError('invalid or oversized training record')
        row = json.loads(line)
        messages = row.get('messages') if isinstance(row, dict) else None
        metadata = row.get('metadata') if isinstance(row, dict) else None
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'].strip() or row['id'] in identifiers
                or not isinstance(messages, list) or len(messages) != 3
                or any(not isinstance(message, dict) or message.get('role') != role
                       or not isinstance(message.get('content'), str) or not message['content']
                       for message, role in zip(messages, ('system', 'user', 'assistant'), strict=True))
                or not isinstance(metadata, dict)
                or metadata.get('review_status_claim') not in ('reviewed', 'unreviewed')
                or metadata.get('outcome_claim') not in ('success', 'unknown', 'failure')
                or not isinstance(metadata.get('teacher'), str) or not metadata['teacher'].strip()
                or not isinstance(metadata.get('original_source_ref_claim'), str) or not metadata['original_source_ref_claim'].strip()
                or not isinstance(metadata.get('original_source_sha256_claim'), str)
                or not re.fullmatch(r'[0-9a-f]{64}', metadata['original_source_sha256_claim'])
                or metadata.get('task') != manifest.get('task')
                or tuple(metadata.get(k) for k in ('trace_file_sha256', 'project', 'project_group', 'origin')) not in train_sources):
            raise ValueError('invalid training trace')
        identifiers.add(row['id'])
        if len(identifiers) > config['max_examples']:
            raise ValueError('training example limit exceeded')
        status, outcome = metadata['review_status_claim'], metadata['outcome_claim']
        weight = config['review_status_weights'][status] * config['outcome_weights'][outcome]
        if not math.isfinite(weight):
            raise ValueError('combined example weight is not finite')
        counts[f'review_status:{status}'] += 1
        counts[f'outcome:{outcome}'] += 1
        if weight == 0:
            continue
        system, user, assistant = (message['content'] for message in messages)
        current_system_sha = base.sha(system.encode('utf-8'))
        if system_sha is not None and current_system_sha != system_sha:
            raise ValueError('training traces use different system prompts')
        system_sha = current_system_sha
        fmt = config['framing']
        prefix = (fmt['system_prefix'] + system + fmt['system_suffix'] + fmt['user_prefix'] + user
                  + fmt['user_suffix'] + fmt['assistant_prefix']).encode('utf-8')
        response = (assistant + fmt['assistant_suffix']).encode('utf-8')
        if not prefix or not response:
            raise ValueError('a training trace needs a prefix and response')
        if len(prefix) + len(response) - 1 > config['context'] and not config['allow_truncated_prompt']:
            raise ValueError('full prompt and response exceed context; choose an explicit truncation policy or larger context')
        records.append({'bytes': prefix + response, 'response_start': len(prefix), 'weight': weight})
    if not records or not math.isfinite(sum(item['weight'] for item in records)):
        raise ValueError('no positively weighted training traces')
    return records, manifest_raw, train_raw, {'selected': len(records), 'claims': dict(sorted(counts.items())),
        'system_sha256': system_sha,
        'framing_sha256': framing_sha256(config['framing']),
        'initial_prompt_exceeds_context': sum(item['response_start'] > config['context'] for item in records),
        'complete_example_exceeds_context': sum(len(item['bytes']) > config['context'] + 1 for item in records)}


def sample_batch(records: list[dict], rng: random.Random, batch_size: int, context: int):
    inputs, labels, corrections = [], [], []
    for item in rng.choices(records, weights=[entry['weight'] for entry in records], k=batch_size):
        end = rng.randrange(item['response_start'], len(item['bytes']))
        start = max(0, end - context)
        segment = item['bytes'][start:end + 1]
        inputs.append(list(segment[:-1]))
        labels.append([byte if start + index + 1 >= item['response_start'] else -100
                       for index, byte in enumerate(segment[1:])])
        # A target near the response end appears in fewer uniformly sampled
        # windows. Its inverse inclusion count gives an unbiased per-example
        # mean loss when summed over the sampled window.
        corrections.append([1.0 / min(context, len(item['bytes']) - (start + index + 1))
                            if start + index + 1 >= item['response_start'] else 0.0
                            for index in range(len(segment) - 1)])
    width = max(map(len, inputs))
    for source, target, correction in zip(inputs, labels, corrections, strict=True):
        padding = width - len(source)
        source.extend([0] * padding)
        target.extend([-100] * padding)
        correction.extend([0.0] * padding)
    return inputs, labels, corrections


def experiment(dataset: Path, checkpoint: Path, config_path: Path, output: Path, *,
               resume: Path | None = None, resume_state_sha256: str | None = None,
               max_steps_this_run: int | None = None, max_config_bytes: int = 65536) -> dict:
    config, config_raw = configuration(config_path, max_config_bytes)
    if (resume is None) != (resume_state_sha256 is None):
        raise ValueError('resume and resume state digest must be supplied together')
    if max_steps_this_run is not None and (type(max_steps_this_run) is not int or max_steps_this_run < 1):
        raise ValueError('max_steps_this_run must be positive')
    records, dataset_manifest, train_raw, coverage = read_training(dataset, config)
    checkpoint_manifest = base.read_bounded(checkpoint / 'manifest.json', config['max_manifest_bytes'])
    initial = json.loads(checkpoint_manifest)
    initial_weights = base.read_bounded(checkpoint / 'weights.safetensors', config['max_checkpoint_bytes'])
    code_paths = (Path(__file__), Path(base.__file__), Path(owned_student.__file__), Path(checkpoints.__file__))
    code = {path.name: base.sha(path.read_bytes()) for path in code_paths}
    if (not isinstance(initial, dict) or initial.get('schema_version') not in ('lfg.owned-student-experiment-result.v1', 'lfg.owned-distillation-result.v1',
                                              'lfg.owned-assessment-sft-result.v1')
            or initial.get('weights_sha256') != base.sha(initial_weights)
            or initial.get('code_sha256', {}).get('owned_student.py') != code['owned_student.py']):
        raise ValueError('owned checkpoint identity does not match')
    architecture = dict(initial['architecture'])
    owned_student.validate_architecture(architecture)
    old_context = architecture['context']
    if config['context'] < old_context:
        raise ValueError('context cannot shrink the source checkpoint')
    architecture['context'] = config['context']
    with torch.device('meta'):
        template = owned_student.OwnedByteStudent(architecture)
    parameters = sum(item.numel() for item in template.parameters())
    del template
    attention_layers = architecture['layers'] // architecture['attention_every']
    estimate = config['batch_size'] * architecture['heads'] * config['context'] ** 2 * attention_layers
    if parameters > config['max_parameters'] or estimate > config['max_attention_elements']:
        raise ValueError('configured model or attention estimate exceeds limits')
    if output.exists():
        raise ValueError('output already exists')
    checkpoints.private_directory(output)
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('requested CUDA device is unavailable')
    if config['device'] == 'cuda' and config['cuda_device_index'] >= torch.cuda.device_count():
        raise ValueError('requested CUDA device index is unavailable')
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(config['deterministic_algorithms'])
    torch.backends.cuda.matmul.allow_tf32 = config['allow_tf32']
    torch.backends.cudnn.allow_tf32 = config['allow_tf32']
    torch.manual_seed(config['seed'])
    device = torch.device('cpu')
    if config['device'] == 'cuda':
        device = torch.device('cuda', config['cuda_device_index'])
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'], device=device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.manual_seed(config['seed'])
    model = owned_student.OwnedByteStudent(architecture)
    state = load(initial_weights)
    if config['context'] > old_context:
        positions = model.position.weight.detach().clone()
        if state['position.weight'].shape != (old_context, architecture['width']):
            raise ValueError('source position embedding shape mismatch')
        positions[:old_context] = state['position.weight']
        state['position.weight'] = positions
    model.load_state_dict(state, strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError('source checkpoint has non-finite weights')
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rng, losses, supervised = random.Random(config['seed']), [], 0
    identity = {'dataset_manifest_sha256': base.sha(dataset_manifest), 'training_file_sha256': base.sha(train_raw),
                'source_checkpoint_manifest_sha256': base.sha(checkpoint_manifest),
                'source_checkpoint_weights_sha256': base.sha(initial_weights), 'code_sha256': code,
                'continuation_config_sha256': checkpoints.continuation_config(config), 'architecture': architecture,
                'python': platform.python_version(), 'torch': str(torch.__version__), 'safetensors': safetensors.__version__,
                'cuda_version': torch.version.cuda if config['device'] == 'cuda' else None,
                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG') if config['device'] == 'cuda' else None,
                'cuda_devices': ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                                 if config['device'] == 'cuda' else [])}
    initial_step = 0
    if resume is not None:
        progress = checkpoints.restore(resume, resume_state_sha256, model, optimizer, rng, identity, config)
        initial_step, losses, supervised = progress['step'], progress['losses'], progress['supervised_tokens']
        if initial_step > config['steps']:
            raise ValueError('checkpoint already reached the requested total steps')
    end_step = min(config['steps'], initial_step + max_steps_this_run) if max_steps_this_run else config['steps']
    output.mkdir(mode=0o700)
    start_time = time.perf_counter()
    latest = None
    for step in range(initial_step + 1, end_step + 1):
        input_rows, target_rows, correction_rows = sample_batch(records, rng, config['batch_size'], config['context'])
        inputs = torch.tensor(input_rows, dtype=torch.long, device=device)
        targets = torch.tensor(target_rows, dtype=torch.long, device=device)
        correction = torch.tensor(correction_rows, dtype=torch.float32, device=device)
        supervised += int((targets != -100).sum().item())
        optimizer.zero_grad(set_to_none=True)
        token_loss = F.cross_entropy(model(inputs).reshape(-1, 256), targets.reshape(-1),
                                     ignore_index=-100, reduction='none')
        loss = (token_loss * correction.reshape(-1)).sum() / config['batch_size']
        if not bool(torch.isfinite(loss)):
            raise ValueError('non-finite supervised loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % config['checkpoint_every'] == 0 or step == end_step:
            latest = checkpoints.snapshot(output, model, optimizer, rng,
                    {'step': step, 'losses': losses, 'supervised_tokens': supervised}, identity, config)
    elapsed = time.perf_counter() - start_time
    if (configuration(config_path, max_config_bytes)[1] != config_raw
            or base.read_bounded(dataset / 'manifest.json', config['max_manifest_bytes']) != dataset_manifest
            or base.read_bounded(dataset / 'train.jsonl', config['max_dataset_bytes']) != train_raw
            or base.read_bounded(checkpoint / 'manifest.json', config['max_manifest_bytes']) != checkpoint_manifest
            or base.read_bounded(checkpoint / 'weights.safetensors', config['max_checkpoint_bytes']) != initial_weights
            or any(base.sha(path.read_bytes()) != code[path.name] for path in code_paths)):
        raise ValueError('training inputs changed during execution')
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError('non-finite trained weights')
    if end_step < config['steps']:
        return {'schema_version': 'lfg.owned-assessment-sft-partial.v1', 'completed': False,
                'steps': end_step, 'parameters': parameters, 'supervised_tokens': supervised,
                'coverage': coverage, 'latest_checkpoint': str(latest),
                'latest_checkpoint_state_sha256': base.sha(base.read_bounded(latest / 'state.json', config['max_manifest_bytes'])),
                'assessment_quality_evaluated': False, 'promotion_eligible': False}
    trained = save({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()})
    manifest = {'schema_version': 'lfg.owned-assessment-sft-result.v1', 'architecture': architecture,
                'completed': True, 'resumed_from_step': initial_step,
                'resume_checkpoint_state_sha256': resume_state_sha256,
                'source_checkpoint_manifest_sha256': base.sha(checkpoint_manifest),
                'source_checkpoint_weights_sha256': base.sha(initial_weights),
                'dataset_manifest_sha256': base.sha(dataset_manifest), 'training_file_sha256': base.sha(train_raw),
                'system_sha256': coverage['system_sha256'], 'framing_sha256': coverage['framing_sha256'],
                'config_sha256': base.sha(config_raw), 'code_sha256': code, 'coverage': coverage,
                'position_rows_reused': old_context, 'position_rows_initialized': config['context'] - old_context,
                'context_clipping_possible': coverage['complete_example_exceeds_context'] > 0,
                'steps': config['steps'], 'batch_size': config['batch_size'], 'supervised_tokens': supervised,
                'losses': losses, 'elapsed_seconds': elapsed, 'parameters': parameters,
                'estimated_attention_elements': estimate, 'weights_sha256': base.sha(trained),
                'device': config['device'], 'python': platform.python_version(), 'torch': str(torch.__version__),
                'safetensors': safetensors.__version__, 'platform': platform.platform(),
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'allow_tf32': config['allow_tf32'],
                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                'teacher_claims_verified': False, 'validation_opened': False, 'test_opened': False,
                'assessment_quality_evaluated': False, 'promotion_eligible': False}
    if config['device'] == 'cuda':
        manifest['cuda_device'] = torch.cuda.get_device_name(device)
        manifest['cuda_device_index'] = config['cuda_device_index']
        manifest['cuda_memory_fraction'] = config['cuda_memory_fraction']
        manifest['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        manifest['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        manifest['cuda_version'] = torch.version.cuda
    for name, raw in (('weights.safetensors', trained),
                      ('manifest.json', (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())):
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'checkpoint', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', type=Path, help='completed private checkpoint step directory')
    parser.add_argument('--resume-state-sha256', help='expected checkpoint state digest from latest.json')
    parser.add_argument('--max-steps-this-run', type=int, help='optional bound on steps in this invocation')
    parser.add_argument('--max-config-bytes', type=int, default=65536, help='maximum configuration file size')
    args = parser.parse_args()
    try:
        result = experiment(args.dataset, args.checkpoint, args.config, args.output,
                            resume=args.resume, resume_state_sha256=args.resume_state_sha256,
                            max_steps_this_run=args.max_steps_this_run, max_config_bytes=args.max_config_bytes)
        print(json.dumps({key: result.get(key) for key in ('completed', 'latest_checkpoint', 'latest_checkpoint_state_sha256', 'weights_sha256',
              'parameters', 'steps', 'supervised_tokens', 'coverage', 'assessment_quality_evaluated', 'promotion_eligible')}))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError, RuntimeError) as error:
        raise SystemExit(f'Owned assessment SFT failed ({type(error).__name__}): {error}') from None
