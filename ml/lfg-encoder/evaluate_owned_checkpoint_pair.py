#!/usr/bin/env python3
"""Compare two compatible owned byte checkpoints on one screened local source.

This reports byte-level language-model metrics only. It does not establish
project or semantic independence, assess routing quality, or promote a model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import stat
import time

import torch
from torch.nn import functional as F

import evaluate_owned_bytes as byte_eval
import owned_student
import screen_private_corpus as screening
import train_owned_student as base


def strict_json(raw: bytes) -> object:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError('invalid JSON numeric value')

    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def _is_windows() -> bool:
    return os.name == 'nt'


def require_private_output_directory(output: Path) -> None:
    """Require a protected POSIX parent chain; Windows ACL creation is not implemented."""
    if _is_windows():
        raise ValueError('private output ACL verification is unavailable on Windows')
    absolute = Path(os.path.abspath(output))
    if absolute.exists() or absolute.is_symlink():
        raise ValueError('output already exists')
    parent = absolute.parent
    chain = []
    while True:
        chain.append(parent)
        if parent.parent == parent:
            break
        parent = parent.parent
    uid = os.getuid()
    for index, directory in enumerate(reversed(chain)):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError('output path contains a non-directory component')
        if index == len(chain) - 1:
            if info.st_uid != uid or (info.st_mode & 0o077):
                raise ValueError('output parent must be owned and private to this account')
        elif (info.st_uid not in (0, uid)
              or (info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX)):
            raise ValueError('output path has an unsafe writable ancestor')


def load_config(path: Path) -> tuple[dict, str]:
    raw = base.read_bounded(path, 65536)
    value = strict_json(raw)
    fields = {'schema_version', 'context', 'cpu_threads', 'warmup_windows', 'max_windows',
              'max_data_bytes', 'max_model_bytes', 'max_model_parameters', 'max_manifest_bytes',
              'max_policy_bytes'}
    ints = fields - {'schema_version'}
    if (not isinstance(value, dict) or set(value) != fields
            or value.get('schema_version') != 'lfg.owned-checkpoint-pair-evaluation-config.v1'
            or any(type(value[key]) is not int or value[key] < (0 if key == 'warmup_windows' else 1)
                   for key in ints)):
        raise ValueError('invalid pair evaluation configuration')
    return value, base.sha(raw)


def prepare_data(source_path: Path, policy_path: Path, config: dict) -> tuple[bytes, dict]:
    source = base.read_bounded(source_path, config['max_data_bytes'])
    policy, rules, policy_sha = screening.load_policy(policy_path, config['max_policy_bytes'])
    limits = policy['limits']
    if len(source) > limits['max_source_bytes']:
        raise ValueError('source exceeds screening policy limit')
    try:
        source_text = source.decode('utf-8', errors='strict')
    except UnicodeDecodeError:
        raise ValueError('source must be valid UTF-8 for screening') from None
    screened_text, counts = screening._screen(source_text, rules, limits)
    data = screened_text.encode('utf-8')
    if len(data) > config['max_data_bytes'] or len(data) > limits['max_text_bytes']:
        raise ValueError('screened source exceeds configured limit')
    return data, {'original_sha256': base.sha(source), 'screened_sha256': base.sha(data),
                  'original_bytes': len(source), 'screened_bytes': len(data),
                  'policy_sha256': policy_sha, 'policy_revision': policy['revision'],
                  'redaction_counts': dict(sorted(counts.items()))}


def load_checkpoint(manifest_path: Path, weights_path: Path, student_config_path: Path,
                    config: dict, code_sha: str) -> tuple[torch.nn.Module, dict]:
    student_config, student_config_sha = base.load_config(student_config_path)
    model, identity = byte_eval.load_student(manifest_path, weights_path,
        'lfg.owned-distillation-result.v1', config, code_sha)
    manifest = strict_json(base.read_bounded(manifest_path, config['max_manifest_bytes']))
    if (('completed' in manifest and manifest['completed'] is not True)
            or manifest.get('teacher_distillation_performed') is not True):
        raise ValueError('checkpoint is not complete')
    steps = manifest.get('steps')
    tokens = manifest.get('tokens_processed')
    digest_fields = ('source_sha256', 'student_config_sha256', 'distillation_config_sha256')
    if (type(steps) is not int or steps < 1 or type(tokens) is not int or tokens < 1
            or any(not isinstance(manifest.get(key), str)
                   or len(manifest[key]) != 64
                   or any(ch not in '0123456789abcdef' for ch in manifest[key]) for key in digest_fields)
            or not isinstance(manifest.get('teacher'), dict)):
        raise ValueError('checkpoint training provenance is incomplete')
    if any(not isinstance(item, str) or len(item) != 64
           or any(ch not in '0123456789abcdef' for ch in item)
           for item in identity['source_members'] + identity['validation_members']):
        raise ValueError('checkpoint source identities are invalid')
    if (manifest.get('student_config_sha256') != student_config_sha
            or student_config.get('architecture') != identity['architecture']):
        raise ValueError('checkpoint does not match its student configuration')
    # Record only whitelisted provenance, never file paths or corpus contents.
    identity.update({'schema_version': manifest['schema_version'], 'steps': steps,
                     'student_config_sha256': manifest.get('student_config_sha256'),
                     'distillation_config_sha256': manifest.get('distillation_config_sha256'),
                     'teacher_claim_sha256': base.sha(json.dumps(manifest['teacher'], sort_keys=True,
                         separators=(',', ':'), ensure_ascii=False).encode('utf-8')),
                     'tokens_processed': tokens,
                     'completed': manifest.get('completed')})
    return model, identity


def summarize_model(model: torch.nn.Module, data: bytes, context: int, warmups: int) -> dict:
    windows = [(start, min(context, len(data) - 1 - start))
               for start in range(0, len(data) - 1, context)]
    with torch.inference_mode():
        start, length = windows[0]
        warm = torch.tensor([list(data[start:start + length])], dtype=torch.long)
        for _ in range(warmups):
            model(warm)
        nll = 0.0
        correct = 0
        times = []
        for start, length in windows:
            tokens = torch.tensor([list(data[start:start + length])], dtype=torch.long)
            targets = torch.tensor(list(data[start + 1:start + length + 1]), dtype=torch.long)
            before = time.perf_counter_ns()
            logits = model(tokens)
            times.append(time.perf_counter_ns() - before)
            if logits.shape != (1, length, 256) or not bool(torch.isfinite(logits).all()):
                raise ValueError('invalid checkpoint output')
            nll += float(F.cross_entropy(logits.reshape(-1, 256), targets, reduction='sum'))
            correct += int((logits.argmax(-1).reshape(-1) == targets).sum())
    count = len(data) - 1
    mean = nll / count
    ordered = sorted(times)
    return {'mean_nll_nats_per_byte': mean, 'bits_per_byte': mean / math.log(2),
            'perplexity': math.exp(mean) if mean < math.log(float.fromhex('0x1.fffffffffffffp+1023')) else None,
            'next_byte_accuracy': correct / count, 'predicted_bytes': count, 'windows': len(windows),
            'forward_latency_ns_median': statistics.median(ordered),
            'forward_latency_ns_p95': ordered[math.ceil(.95 * len(ordered)) - 1]}


def evaluate(config_path: Path, source_path: Path, policy_path: Path, output: Path,
             left_manifest: Path, left_weights: Path, left_student_config: Path,
             right_manifest: Path, right_weights: Path, right_student_config: Path) -> dict:
    evaluation_started = time.perf_counter()
    require_private_output_directory(output)
    config, config_sha = load_config(config_path)
    byte_eval.private_output(output)
    source_raw = base.read_bounded(source_path, config['max_data_bytes'])
    policy_raw = base.read_bounded(policy_path, config['max_policy_bytes'])
    data, source_id = prepare_data(source_path, policy_path, config)
    if source_id['original_sha256'] != base.sha(source_raw) or source_id['policy_sha256'] != base.sha(policy_raw):
        raise ValueError('evaluation inputs changed during screening')
    if len(data) <= config['context']:
        raise ValueError('screened source must exceed the configured context')
    windows = (len(data) - 1 + config['context'] - 1) // config['context']
    if windows > config['max_windows']:
        raise ValueError('screened source exceeds window limit')
    code_sha = base.sha(Path(owned_student.__file__).read_bytes())
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    left, left_id = load_checkpoint(left_manifest, left_weights, left_student_config, config, code_sha)
    right, right_id = load_checkpoint(right_manifest, right_weights, right_student_config, config, code_sha)
    if left_id['architecture'] != right_id['architecture']:
        raise ValueError('checkpoint architectures are incompatible')
    data_hashes = {source_id['original_sha256'], source_id['screened_sha256']}
    for identity in (left_id, right_id):
        declared = set(identity['source_members']) | set(identity['validation_members'])
        if data_hashes & declared:
            raise ValueError('evaluation source overlaps declared checkpoint data')
    started = time.perf_counter()
    metrics = {'left': summarize_model(left, data, config['context'], config['warmup_windows']),
               'right': summarize_model(right, data, config['context'], config['warmup_windows'])}
    elapsed = time.perf_counter() - started
    deltas = {}
    for key in ('mean_nll_nats_per_byte', 'bits_per_byte', 'perplexity', 'next_byte_accuracy'):
        left_value, right_value = metrics['left'][key], metrics['right'][key]
        deltas[key + '_right_minus_left'] = (None if left_value is None or right_value is None
                                              else right_value - left_value)
    if (load_config(config_path)[1] != config_sha
            or base.read_bounded(source_path, config['max_data_bytes']) != source_raw
            or base.read_bounded(policy_path, config['max_policy_bytes']) != policy_raw
            or base.load_config(left_student_config)[1] != left_id['student_config_sha256']
            or base.load_config(right_student_config)[1] != right_id['student_config_sha256']
            or screening.load_policy(policy_path, config['max_policy_bytes'])[2] != source_id['policy_sha256']):
        raise ValueError('evaluation inputs changed')
    for path, identity, kind in ((left_manifest, left_id, 'manifest'), (left_weights, left_id, 'weights'),
                                 (right_manifest, right_id, 'manifest'), (right_weights, right_id, 'weights')):
        limit = config['max_manifest_bytes'] if kind == 'manifest' else config['max_model_bytes']
        expected = identity['manifest_sha256'] if kind == 'manifest' else identity['weights_sha256']
        if base.sha(base.read_bounded(path, limit)) != expected:
            raise ValueError('checkpoint inputs changed')
    report = {'schema_version': 'lfg.owned-checkpoint-pair-evaluation.v1',
              'config': config, 'config_sha256': config_sha,
              'evaluation_source': source_id,
              'left_checkpoint': left_id, 'right_checkpoint': right_id,
              'metrics': metrics, 'metric_deltas_right_minus_left': deltas,
              'scoring_elapsed_seconds': elapsed,
              'total_elapsed_seconds': time.perf_counter() - evaluation_started,
              'predicted_bytes_per_checkpoint': len(data) - 1,
              'device': 'cpu', 'torch': str(torch.__version__), 'python': platform.python_version(),
              'byte_modeling_only': True, 'assessment_evaluated': False,
              'project_independence': 'unknown', 'semantic_independence': 'unknown',
              'training_histories_comparison': 'not_performed', 'promotion_eligible': False}
    require_private_output_directory(output)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write((json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'source', 'screening-policy', 'output', 'left-manifest', 'left-weights',
                 'left-student-config', 'right-manifest', 'right-weights', 'right-student-config'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.config, args.source, args.screening_policy, args.output,
                          args.left_manifest, args.left_weights, args.left_student_config,
                          args.right_manifest, args.right_weights, args.right_student_config)
        print(json.dumps({'metrics': result['metrics'], 'predicted_bytes_per_checkpoint':
                          result['predicted_bytes_per_checkpoint'], 'project_independence': 'unknown',
                          'semantic_independence': 'unknown', 'training_histories_comparison': 'not_performed',
                          'promotion_eligible': False}))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError, RuntimeError):
        raise SystemExit('Pair evaluation failed; check inputs, identities, screening policy, limits and environment.') from None


if __name__ == '__main__':
    main()
