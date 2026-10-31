#!/usr/bin/env python3
"""Evaluate reviewed node outcomes against the actual deterministic kernel on validation data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import node_review


def hash_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('file changed while hashing')
    return digest.hexdigest()


def predictions(binary: Path, snapshot: Path, snapshot_sha: str, records: list[dict], timeout: float,
                max_items: int, max_input_bytes: int) -> dict[str, dict]:
    if not records:
        raise ValueError('validation split is empty')
    if len(records) > max_items:
        raise ValueError('validation split exceeds configured item limit')
    binary = binary.resolve(strict=True)
    snapshot = snapshot.resolve(strict=True)
    expected = {row['review']['itemSha256']: row for row in records}
    if len(expected) != len(records):
        raise ValueError('duplicate validation item')
    lines, size = [], 0
    for key, row in expected.items():
        line = node_review.encode({'id': key, 'text': row['item']['text']}) + b'\n'
        size += len(line)
        if size > max_input_bytes:
            raise ValueError('validation inputs exceed configured byte limit')
        lines.append(line)
    body = b''.join(lines)
    result = subprocess.run([str(binary), 'kernel-predict', '--snapshot', str(snapshot), '--max-examples', str(len(records))],
                            input=body, capture_output=True, timeout=timeout, check=False)
    if result.returncode != 0:
        # stderr can contain private inputs; do not echo it in the aggregate report.
        raise ValueError(f'kernel prediction process failed with exit code {result.returncode}')
    if not result.stdout.endswith(b'\n'):
        raise ValueError('kernel predictions are incomplete')
    seen = {}
    for line in result.stdout.splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {'schema_version', 'id', 'input_sha256', 'result'} or row['schema_version'] != 'lfg.kernel-prediction.v1':
            raise ValueError('invalid kernel prediction')
        key = row['id']
        if key not in expected or key in seen or row['input_sha256'] != expected[key]['item']['inputSha256']:
            raise ValueError('kernel prediction ID or input digest mismatch')
        kernel = row['result']
        if not isinstance(kernel, dict) or kernel.get('schemaVersion') != 'lfg.kernel-result.v1' or kernel.get('snapshotSha256') != snapshot_sha:
            raise ValueError('kernel result does not match snapshot')
        seen[key] = kernel
    if set(seen) != set(expected):
        raise ValueError('kernel predictions do not cover validation items')
    return seen


def baseline_choice(result: dict, node: str, options: set[str]) -> tuple[str, str | None]:
    hops = result.get('hops')
    if not isinstance(hops, list) or result.get('state') not in ('complete', 'escalate', 'pruned') or not isinstance(result.get('atNode'), str):
        raise ValueError('invalid kernel result state')
    found = [hop for hop in hops if isinstance(hop, dict) and hop.get('node') == node]
    if len(found) > 1:
        raise ValueError('kernel visited node more than once')
    if found:
        choice = found[0].get('value')
        if found[0].get('operation') != 'choice' or not isinstance(choice, str) or choice not in options:
            raise ValueError('invalid kernel choice hop')
        # The kernel records a provisional top choice before its threshold and
        # ambiguity checks. An escalation at this node did not commit that hop.
        if result['atNode'] == node and result['state'] == 'escalate':
            return 'escalated', None
        return 'selected', choice
    if result['atNode'] == node and result['state'] == 'escalate':
        return 'escalated', None
    return 'not_reached', None


def load_candidate(path: Path, records: list[dict], snapshot_sha: str, options: set[str]) -> tuple[dict[str, str | None], str]:
    raw = path.read_bytes()
    if not raw.endswith(b'\n'):
        raise ValueError('candidate prediction file must be newline terminated')
    expected = {row['review']['itemSha256']: row for row in records}
    values = {}
    for line in raw.splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {'schemaVersion', 'itemSha256', 'inputSha256', 'snapshotSha256', 'choice'} or row['schemaVersion'] != 'lfg.node-prediction.v1':
            raise ValueError('invalid candidate prediction record')
        key = row['itemSha256']
        if (key not in expected or key in values or row['inputSha256'] != expected[key]['item']['inputSha256']
                or row['snapshotSha256'] != snapshot_sha or (row['choice'] is not None and row['choice'] not in options)):
            raise ValueError('candidate prediction does not match validation item or snapshot')
        values[key] = row['choice']
    if set(values) != set(expected):
        raise ValueError('candidate predictions must cover exactly the validation items')
    if path.read_bytes() != raw:
        raise ValueError('candidate predictions changed while reading')
    return values, node_review.digest(raw)


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score(records: list[dict], outputs: dict[str, dict], node: str, options: set[str],
          candidate: dict[str, str | None] | None) -> dict:
    per_choice = {option: {'total': 0, 'correct': 0, 'wrong': 0, 'escalated': 0, 'notReached': 0}
                  for option in sorted(options)}
    counts = {'total': len(records), 'correct': 0, 'wrong': 0, 'escalated': 0, 'notReached': 0}
    fallback = {'eligible': 0, 'correct': 0, 'wrong': 0, 'abstained': 0} if candidate is not None else None
    for record in records:
        key = record['review']['itemSha256']
        expected = record['review']['choice']
        if expected not in options or key not in outputs:
            raise ValueError('invalid reviewed choice or missing baseline output')
        label = per_choice[expected]
        label['total'] += 1
        state, selected = baseline_choice(outputs[key], node, options)
        outcome = ('correct' if selected == expected else 'wrong') if state == 'selected' else ('escalated' if state == 'escalated' else 'notReached')
        counts[outcome] += 1
        label[outcome] += 1
        if fallback is not None and state == 'escalated':
            if key not in candidate:
                raise ValueError('candidate output missing validation item')
            fallback['eligible'] += 1
            value = candidate[key]
            fallback['abstained' if value is None else 'correct' if value == expected else 'wrong'] += 1
    selected = counts['correct'] + counts['wrong']
    result = {'baseline': {**counts, 'selectionRate': rate(selected, counts['total']),
                            'accuracyAmongSelected': rate(counts['correct'], selected),
                            'correctPerReviewedItem': rate(counts['correct'], counts['total'])},
              'perReviewedChoice': per_choice}
    if fallback is not None:
        result['fallbackOnEscalation'] = {**fallback, 'selectionRate': rate(fallback['correct'] + fallback['wrong'], fallback['eligible']),
            'accuracyAmongSelected': rate(fallback['correct'], fallback['correct'] + fallback['wrong']),
            'combinedCorrectPerReviewedItem': rate(counts['correct'] + fallback['correct'], counts['total'])}
    return result


def evaluate(args: argparse.Namespace) -> dict:
    if not 0 < args.timeout_seconds < float('inf'):
        raise ValueError('timeout must be finite and positive')
    if args.max_validation_items < 1 or args.max_input_bytes < 1:
        raise ValueError('validation item and input byte limits must be positive')
    evaluator_sha = hash_file(Path(__file__))
    binary_sha = hash_file(args.kernel_binary)
    operations_path = os.environ.get('LFG_OPERATIONS_CONFIG')
    operations_sha = hash_file(Path(operations_path)) if operations_path else None
    audit, records = node_review.audit(args.input, args.snapshot, args.policy, args.node)
    if not audit['admissionEligible']:
        raise ValueError('node admission gates are not satisfied')
    validation = [record for record in records if record['review']['split'] == 'validation']
    spec = json.loads(args.snapshot.read_bytes())
    options = set(spec['nodes'][args.node]['options'])
    candidate, candidate_sha = (load_candidate(args.candidate, validation, audit['snapshotSha256'], options)
                                if args.candidate else (None, None))
    baseline = predictions(args.kernel_binary, args.snapshot, audit['snapshotSha256'], validation, args.timeout_seconds,
                           args.max_validation_items, args.max_input_bytes)
    scores = score(validation, baseline, args.node, options, candidate)
    if hash_file(args.kernel_binary) != binary_sha:
        raise ValueError('kernel binary changed during evaluation')
    if operations_path and hash_file(Path(operations_path)) != operations_sha:
        raise ValueError('operations configuration changed during evaluation')
    after, _ = node_review.audit(args.input, args.snapshot, args.policy, args.node)
    if after != audit:
        raise ValueError('review inputs changed during evaluation')
    if args.candidate and hash_file(args.candidate) != candidate_sha:
        raise ValueError('candidate predictions changed during evaluation')
    if hash_file(Path(__file__)) != evaluator_sha:
        raise ValueError('evaluator changed during evaluation')
    return {'schemaVersion': 'lfg.node-validation-report.v1', 'split': 'validation', 'node': args.node,
            'admission': audit, 'kernelBinarySha256': binary_sha, 'candidatePredictionsSha256': candidate_sha,
            'operationsConfigSha256': operations_sha, 'timeoutSeconds': args.timeout_seconds,
            'maxValidationItems': args.max_validation_items, 'maxInputBytes': args.max_input_bytes,
            'evaluatorSha256': evaluator_sha, 'candidateExecutionVerified': False,
            'qualificationEstablished': False, 'metrics': scores}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs=2, action='append', required=True, metavar=('QUEUE', 'LEDGER'))
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--kernel-binary', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=float, required=True)
    parser.add_argument('--max-validation-items', type=int, required=True)
    parser.add_argument('--max-input-bytes', type=int, required=True)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=True, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f'node evaluation: {error}', file=sys.stderr)
        sys.exit(1)
