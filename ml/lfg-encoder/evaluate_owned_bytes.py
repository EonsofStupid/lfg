#!/usr/bin/env python3
"""Compare byte-level next-token losses on an explicitly selected local file."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

from safetensors.torch import load
import torch
from torch.nn import functional as F

import distill_owned_student as distill
import owned_student
import train_owned_student as base
import training_inputs


def config_file(path: Path) -> tuple[dict, str]:
    raw = base.read_bounded(path, 65536)
    config = json.loads(raw)
    integers = {'context', 'cpu_threads', 'warmup_windows', 'max_windows', 'max_data_bytes',
                'max_model_bytes', 'max_model_parameters', 'max_manifest_bytes'}
    if (not isinstance(config, dict) or set(config) != integers | {'schema_version'}
            or config['schema_version'] != 'lfg.owned-byte-evaluation-config.v1'
            or any(type(config[name]) is not int or config[name] < (0 if name == 'warmup_windows' else 1)
                   for name in integers)):
        raise ValueError('invalid byte evaluation configuration')
    return config, base.sha(raw)


def private_output(path: Path) -> None:
    path = path.resolve()
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    try:
        root = subprocess.run(['git', '-C', str(ancestor), 'rev-parse', '--show-toplevel'],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
        if root.returncode:
            if any((item / '.git').exists() for item in (ancestor, *ancestor.parents)):
                raise ValueError('cannot verify output privacy')
            return
        repository = Path(root.stdout.rstrip('\r\n')).resolve()
        if not path.is_relative_to(repository):
            raise ValueError('cannot verify output privacy')
        relative = path.relative_to(repository).as_posix()
        tracked = subprocess.run(['git', '-C', str(repository), 'ls-files', '--error-unmatch', '--', relative],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        ignored = subprocess.run(['git', '-C', str(repository), 'check-ignore', '-q', '--no-index', '--', relative],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if tracked.returncode != 1 or ignored.returncode != 0:
            raise ValueError('output must be ignored and untracked')
    except OSError:
        raise ValueError('cannot verify output privacy') from None


def load_student(manifest_path: Path, weights_path: Path, schema: str, config: dict,
                 model_code_sha: str) -> tuple[torch.nn.Module, dict]:
    raw = base.read_bounded(manifest_path, config['max_manifest_bytes'])
    weights = base.read_bounded(weights_path, config['max_model_bytes'])
    manifest = json.loads(raw)
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != schema
            or manifest.get('weights_sha256') != base.sha(weights)
            or not isinstance(manifest.get('code_sha256'), dict)
            or manifest['code_sha256'].get('owned_student.py') != model_code_sha):
        raise ValueError('student identity or implementation does not match')
    architecture = manifest.get('architecture')
    owned_student.validate_architecture(architecture)
    if architecture['context'] < config['context']:
        raise ValueError('student context is too short')
    with torch.device('meta'):
        template = owned_student.OwnedByteStudent(architecture)
    if sum(value.numel() for value in template.parameters()) > config['max_model_parameters']:
        raise ValueError('student exceeds configured parameter limit')
    del template
    model = owned_student.OwnedByteStudent(architecture)
    model.load_state_dict(load(weights), strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError('student weights are not finite')
    corpus = manifest.get('corpus')
    if corpus is None:
        source_members = [manifest.get('source_sha256')]
        validation_members = []
    else:
        if (not isinstance(corpus, dict) or corpus.get('schema_version') != 'lfg.screened-document-corpus.v1'
                or not isinstance(corpus.get('sources'), list) or not corpus['sources']
                or base.sha(json.dumps(corpus, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))
                   != manifest.get('source_sha256')):
            raise ValueError('invalid training corpus identity')
        source_members = []
        for item in corpus['sources']:
            if not isinstance(item, dict) or any(not isinstance(item.get(key), str) or len(item[key]) != 64
                                                 for key in ('original_sha256', 'screened_sha256')):
                raise ValueError('invalid training source identities')
            source_members.extend((item['original_sha256'], item['screened_sha256']))
        source_members = sorted(set(source_members))
        validations = corpus.get('declared_validation_sources', [])
        if (not isinstance(validations, list) or any(not isinstance(item, dict) or
                not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64 for item in validations)):
            raise ValueError('invalid declared validation identities')
        validation_members = sorted(set(item['sha256'] for item in validations))
    return model.eval(), {'manifest_sha256': base.sha(raw), 'weights_sha256': base.sha(weights),
                          'architecture': architecture, 'source_sha256': manifest.get('source_sha256'),
                          'source_members': source_members,
                          'validation_members': validation_members,
                          'student_config_sha256': manifest.get('student_config_sha256', manifest.get('config_sha256')),
                          'teacher': manifest.get('teacher')}


def evaluate(config_path: Path, student_config_path: Path, source_path: Path, output: Path,
             candidate_manifest: Path, candidate_weights: Path, control_manifest: Path, control_weights: Path,
             teacher_manifest: Path, teacher_snapshot: Path, teacher_revision: str) -> dict:
    config, config_sha = config_file(config_path)
    student_config, student_config_sha = base.load_config(student_config_path)
    data = base.read_bounded(source_path, config['max_data_bytes'])
    if len(data) < 2 or (len(data) - 1 + config['context'] - 1) // config['context'] > config['max_windows']:
        raise ValueError('evaluation data is empty or exceeds window limit')
    if output.exists():
        raise ValueError('output already exists')
    private_output(output)
    paths = (Path(__file__), Path(owned_student.__file__), Path(base.__file__), Path(distill.__file__),
             Path(training_inputs.__file__), Path(training_inputs.lock_wheelhouse.__file__))
    code = {path.name: base.sha(path.read_bytes()) for path in paths}
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    candidate, candidate_id = load_student(candidate_manifest, candidate_weights, 'lfg.owned-distillation-result.v1',
                                           config, code['owned_student.py'])
    control, control_id = load_student(control_manifest, control_weights, 'lfg.owned-student-experiment-result.v1',
                                       config, code['owned_student.py'])
    if (candidate_id['architecture'] != control_id['architecture']
            or candidate_id['architecture'] != student_config['architecture']
            or candidate_id['student_config_sha256'] != student_config_sha
            or control_id['student_config_sha256'] != student_config_sha
            or not isinstance(candidate_id['source_sha256'], str)
            or candidate_id['source_sha256'] != control_id['source_sha256']
            or candidate_id['source_members'] != control_id['source_members']
            or candidate_id['validation_members'] != control_id['validation_members']
            or base.sha(data) in candidate_id['source_members']):
        raise ValueError('models must share a training configuration and a different source')
    if sum(value.numel() for value in candidate.parameters()) > student_config['max_parameters']:
        raise ValueError('student exceeds training parameter limit')
    torch.manual_seed(student_config['seed'])
    initial = owned_student.OwnedByteStudent(student_config['architecture']).eval()
    teacher, teacher_id = distill.load_gpt2_teacher(teacher_snapshot, teacher_manifest, teacher_revision,
        {'max_teacher_manifest_bytes': config['max_manifest_bytes'], 'max_teacher_bytes': config['max_model_bytes'],
         'max_teacher_parameters': config['max_model_parameters']}, config['context'])
    recorded_teacher = candidate_id['teacher']
    if not isinstance(recorded_teacher, dict) or any(recorded_teacher.get(key) != teacher_id[key]
            for key in ('kind', 'model_id', 'revision', 'manifest_sha256', 'weights_sha256', 'byte_id_mapping')):
        raise ValueError('candidate was not trained with the selected teacher identity')
    models = {'initial': initial, 'cross_entropy': control, 'distilled': candidate, 'teacher': teacher}
    metrics = {name: {'nll_sum': 0., 'correct': 0, 'times_ns': []} for name in models}
    windows = [(start, min(config['context'], len(data) - 1 - start))
               for start in range(0, len(data) - 1, config['context'])]
    with torch.inference_mode():
        warm_start, warm_length = windows[0]
        warm = torch.tensor([list(data[warm_start:warm_start + warm_length])], dtype=torch.long)
        for _ in range(config['warmup_windows']):
            for name, model in models.items():
                model(warm, use_cache=False) if name == 'teacher' else model(warm)
        for index, (start, length) in enumerate(windows):
            tokens = torch.tensor([list(data[start:start + length])], dtype=torch.long)
            targets = torch.tensor(list(data[start + 1:start + length + 1]), dtype=torch.long)
            names = list(models)
            names = names[index % len(names):] + names[:index % len(names)]
            for name in names:
                before = time.perf_counter_ns()
                result = models[name](tokens, use_cache=False).logits if name == 'teacher' else models[name](tokens)
                metrics[name]['times_ns'].append(time.perf_counter_ns() - before)
                if result.shape != (1, length, 256) or not bool(torch.isfinite(result).all()):
                    raise ValueError('invalid model output')
                metrics[name]['nll_sum'] += float(F.cross_entropy(result.reshape(-1, 256), targets, reduction='sum'))
                metrics[name]['correct'] += int((result.argmax(-1).reshape(-1) == targets).sum())
    count = len(data) - 1
    results = {}
    for name, values in metrics.items():
        times = sorted(values['times_ns'])
        results[name] = {'mean_nll_nats_per_byte': values['nll_sum'] / count,
                         'bits_per_byte': values['nll_sum'] / (count * math.log(2)),
                         'next_byte_accuracy': values['correct'] / count,
                         'forward_latency_ns_median': statistics.median(times),
                         'forward_latency_ns_p95': times[math.ceil(.95 * len(times)) - 1]}
    if (config_file(config_path)[1] != config_sha or base.load_config(student_config_path)[1] != student_config_sha
            or base.read_bounded(source_path, config['max_data_bytes']) != data
            or base.sha(base.read_bounded(candidate_manifest, config['max_manifest_bytes'])) != candidate_id['manifest_sha256']
            or base.sha(base.read_bounded(candidate_weights, config['max_model_bytes'])) != candidate_id['weights_sha256']
            or base.sha(base.read_bounded(control_manifest, config['max_manifest_bytes'])) != control_id['manifest_sha256']
            or base.sha(base.read_bounded(control_weights, config['max_model_bytes'])) != control_id['weights_sha256']
            or training_inputs.verify_model(teacher_snapshot, teacher_manifest, teacher_revision) != {
                key: teacher_id[key] for key in ('model_id', 'revision', 'manifest_sha256', 'files')}
            or any(base.sha(path.read_bytes()) != code[path.name] for path in paths)):
        raise ValueError('evaluation inputs changed')
    report = {'schema_version': 'lfg.owned-byte-evaluation.v1', 'evaluation_source_sha256': base.sha(data),
              'training_source_sha256': candidate_id['source_sha256'], 'evaluation_bytes': len(data),
              'predicted_bytes': count, 'windows': len(windows), 'config': config, 'config_sha256': config_sha,
              'student_config_sha256': student_config_sha, 'candidate': candidate_id, 'control': control_id,
              'teacher': teacher_id, 'code_sha256': code, 'python': platform.python_version(),
              'platform': platform.platform(), 'torch': str(torch.__version__), 'device': 'cpu', 'results': results,
              'declared_validation_match': (base.sha(data) in candidate_id['validation_members']) if candidate_id['validation_members'] else None,
              'assessment_evaluated': False, 'independent_project_verified': False, 'promotion_eligible': False}
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write((json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'student-config', 'source', 'output', 'candidate-manifest', 'candidate-weights',
                 'control-manifest', 'control-weights', 'teacher-manifest', 'teacher-snapshot'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--teacher-revision', required=True)
    args = parser.parse_args()
    try:
        report = evaluate(args.config, args.student_config, args.source, args.output, args.candidate_manifest,
                          args.candidate_weights, args.control_manifest, args.control_weights,
                          args.teacher_manifest, args.teacher_snapshot, args.teacher_revision)
        print(json.dumps({'predicted_bytes': report['predicted_bytes'], 'windows': report['windows'],
                          'results': report['results'], 'assessment_evaluated': False,
                          'promotion_eligible': False}))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError, RuntimeError):
        raise SystemExit('Byte evaluation failed; check inputs, identities, limits and environment.') from None


if __name__ == '__main__':
    main()
