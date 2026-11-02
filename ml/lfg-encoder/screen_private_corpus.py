#!/usr/bin/env python3
"""Prepare private, unreviewed text-only development candidates with redactions.

This is a screening aid, not proof that secrets have been removed, a human
review, a supervised assessment corpus, or a model training or evaluation run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import unicodedata
from typing import Any

import assessment_distill
import corpus_audit
import review_corpus

POLICY_VERSION = 'lfg.private-corpus-screen.v1'
OUTPUT_VERSION = 'lfg.screened-text-candidate.v1'
LIMITS = {'max_policy_bytes', 'max_plan_bytes', 'max_source_bytes', 'max_total_source_bytes',
          'max_total_output_bytes', 'max_records', 'max_text_bytes', 'max_matches_per_record'}


class ScreenError(ValueError):
    """Contains an operational code only, never private input or matched text."""


def _read(path: Path, maximum: int, code: str) -> bytes:
    try:
        with path.open('rb') as source:
            raw = source.read(maximum + 1)
    except OSError:
        raise ScreenError(code) from None
    if len(raw) > maximum:
        raise ScreenError(code)
    return raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(raw: bytes, code: str) -> Any:
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate key')
            value[key] = item
        return value

    def invalid_constant(_value):
        raise ValueError('invalid numeric value')

    try:
        return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
    except (ValueError, UnicodeError):
        raise ScreenError(code) from None


def load_policy(path: Path, read_limit: int) -> tuple[dict, list[tuple[str, re.Pattern, str]], str]:
    if type(read_limit) is not int or read_limit < 1:
        raise ScreenError('invalid_policy_read_limit')
    raw = _read(path, read_limit, 'policy_unavailable_or_oversize')
    policy = _json(raw, 'invalid_policy')
    if (not isinstance(policy, dict) or set(policy) != {'schema_version', 'revision', 'limits', 'rules'}
            or policy['schema_version'] != POLICY_VERSION or not isinstance(policy['revision'], str)
            or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]*', policy['revision'])
            or not isinstance(policy['limits'], dict)
            or set(policy['limits']) != LIMITS
            or any(type(value) is not int or value < 1 for value in policy['limits'].values())
            or len(raw) > policy['limits']['max_policy_bytes']
            or not isinstance(policy['rules'], list) or not policy['rules']):
        raise ScreenError('invalid_policy')
    compiled, ids = [], set()
    for rule in policy['rules']:
        if (not isinstance(rule, dict) or set(rule) != {'id', 'pattern', 'replacement'}
                or any(not isinstance(rule[key], str) for key in rule)
                or not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', rule['id'])
                or rule['id'] in ids or not rule['pattern'] or not rule['replacement']):
            raise ScreenError('invalid_policy_rule')
        try:
            pattern = re.compile(rule['pattern'])
        except re.error:
            raise ScreenError('invalid_policy_pattern') from None
        if pattern.search('') is not None:
            raise ScreenError('empty_policy_match')
        ids.add(rule['id'])
        compiled.append((rule['id'], pattern, rule['replacement']))
    return policy, compiled, _sha(raw)


def _screen(text: str, rules: list[tuple[str, re.Pattern, str]], limits: dict) -> tuple[str, Counter]:
    counts: Counter = Counter()
    matches = 0
    for identifier, pattern, replacement in rules:
        found = 0
        for match in pattern.finditer(text):
            if match.start() == match.end():
                raise ScreenError('empty_policy_match')
            found += 1
            matches += 1
            if matches > limits['max_matches_per_record']:
                raise ScreenError('match_limit_exceeded')
        # Reject expansion before creating a potentially large replacement string.
        if len(text.encode('utf-8')) + found * len(replacement.encode('utf-8')) > limits['max_text_bytes'] * 2:
            raise ScreenError('redacted_text_limit_exceeded')
        if found:
            text = pattern.sub(lambda _: replacement, text)
            counts[identifier] += found
        if len(text.encode('utf-8')) > limits['max_text_bytes']:
            raise ScreenError('redacted_text_limit_exceeded')
    if not text.strip():
        raise ScreenError('redacted_text_empty')
    if any(pattern.search(text) is not None for _, pattern, _ in rules):
        raise ScreenError('residual_policy_match')
    return text, counts


def require_private_outputs(output: Path) -> None:
    """Check every output in its own working tree, before any writes."""
    output = output.resolve()
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    try:
        root = subprocess.run(['git', '-C', str(ancestor), 'rev-parse', '--show-toplevel'],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, text=True)
    except OSError:
        raise ScreenError('cannot_verify_output_privacy') from None
    if root.returncode != 0:
        if any((directory / '.git').exists() for directory in (ancestor, *ancestor.parents)):
            raise ScreenError('cannot_verify_output_privacy')
        return
    repository = Path(root.stdout.rstrip('\r\n')).resolve()
    if not output.is_relative_to(repository):
        raise ScreenError('cannot_verify_output_privacy')
    for name in ('train.jsonl', 'validation.jsonl', 'manifest.json'):
        relative = (output / name).relative_to(repository).as_posix()
        result = subprocess.run(['git', '-C', str(repository), 'check-ignore', '-q', '--no-index', '--', relative],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if result.returncode != 0:
            raise ScreenError('output_not_private')


def prepare(plan_path: Path, policy_path: Path, output: Path, policy_read_limit: int) -> dict:
    policy, rules, policy_sha = load_policy(policy_path, policy_read_limit)
    limits = policy['limits']
    plan_raw = _read(plan_path, limits['max_plan_bytes'], 'plan_unavailable_or_oversize')
    plan = _json(plan_raw, 'invalid_plan')
    if (not isinstance(plan, dict) or plan.get('schema_version') != corpus_audit.DEVELOPMENT_PLAN_SCHEMA
            or plan.get('review_ledger') is not None or not isinstance(plan.get('sources'), list)
            or not plan['sources']):
        raise ScreenError('invalid_development_plan')
    # Validate the complete split list before resolving or reading any source.
    for source in plan['sources']:
        if (not isinstance(source, dict) or source.get('split') not in ('train', 'validation')
                or any(not isinstance(source.get(key), str) or not source[key].strip()
                       for key in ('path', 'project'))
                or not isinstance(source.get('project_group', source['project']), str)
                or not source.get('project_group', source['project']).strip()):
            raise ScreenError('invalid_source_or_heldout_split')
    require_private_outputs(output)
    if output.exists():
        raise ScreenError('output_exists')

    projects, groups, families, ids, normalized = {}, {}, {}, set(), {}
    output_rows: dict[str, list[dict]] = {'train': [], 'validation': []}
    source_snapshots, rule_counts = [], Counter()
    total_source_bytes = total_output_bytes = matched_records = 0
    for source in plan['sources']:
        project, group, split = source['project'], source.get('project_group', source['project']), source['split']
        if (project in projects and projects[project] != (group, split)) or (group in groups and groups[group] != split):
            raise ScreenError('project_group_crosses_split')
        projects[project], groups[group] = (group, split), split
        path = corpus_audit.resolve(plan_path.parent, source['path'])
        raw = _read(path, limits['max_source_bytes'], 'source_unavailable_or_oversize')
        total_source_bytes += len(raw)
        if total_source_bytes > limits['max_total_source_bytes']:
            raise ScreenError('total_source_limit_exceeded')
        source_sha = _sha(raw)
        try:
            lines = raw.decode('utf-8').split('\n')
        except UnicodeError:
            raise ScreenError('invalid_source_encoding') from None
        if lines and lines[-1] == '':
            lines.pop()
        if not lines or any(not line.strip() for line in lines):
            raise ScreenError('invalid_source_records')
        source_snapshots.append((path, source_sha, len(lines)))
        for line in lines:
            row = _json(line.encode('utf-8'), 'invalid_source_records')
            if (not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                                for key in ('id', 'family', 'text'))
                    or not isinstance(row.get('source'), dict) or row['source'].get('project') != project):
                raise ScreenError('invalid_source_records')
            identifier, family, original = row['id'], row['family'], row['text']
            if identifier in ids or len(ids) >= limits['max_records']:
                raise ScreenError('duplicate_or_excess_records')
            if family in families and families[family] != split:
                raise ScreenError('family_crosses_split')
            ids.add(identifier)
            families[family] = split
            if len(original.encode('utf-8')) > limits['max_text_bytes']:
                raise ScreenError('input_text_limit_exceeded')
            # Do not silently carry sensitive material through metadata fields.
            for value in (identifier, family, project, group):
                if any(pattern.search(value) is not None for _, pattern, _ in rules):
                    raise ScreenError('sensitive_metadata_detected')
            text, matches = _screen(original, rules, limits)
            key = corpus_audit.text_key(text)
            if key in normalized and normalized[key] != split:
                raise ScreenError('screened_text_crosses_split')
            normalized[key] = split
            rule_counts.update(matches)
            matched_records += int(bool(matches))
            result = {'schema_version': OUTPUT_VERSION, 'id': identifier, 'family': family,
                      'project': project, 'project_group': group, 'split': split, 'text': text,
                      'candidate_file_sha256': source_sha,
                      'source_record_sha256': _sha(line.encode('utf-8')),
                      'source_text_sha256': _sha(original.encode('utf-8')),
                      'screened_text_sha256': _sha(text.encode('utf-8')),
                      'screen_policy_sha256': policy_sha, 'redaction_counts': dict(sorted(matches.items())),
                      'review_status': 'unreviewed', 'redaction_complete': False}
            encoded = review_corpus.encoded_rows([result])
            total_output_bytes += len(encoded)
            if total_output_bytes > limits['max_total_output_bytes']:
                raise ScreenError('total_output_limit_exceeded')
            output_rows[split].append(result)

    # Recheck source and policy identities before any output is created.
    if (_sha(_read(plan_path, limits['max_plan_bytes'], 'plan_changed')) != _sha(plan_raw)
            or _sha(_read(policy_path, policy_read_limit, 'policy_changed')) != policy_sha
            or any(_sha(_read(path, limits['max_source_bytes'], 'source_changed')) != digest
                   for path, digest, _ in source_snapshots)):
        raise ScreenError('input_changed')
    manifest = {'schema_version': 'lfg.private-corpus-screen-manifest.v1',
                'plan_sha256': _sha(plan_raw), 'policy_sha256': policy_sha,
                'policy_revision': policy['revision'],
                'screen_code_sha256': _sha(Path(__file__).read_bytes()),
                'helper_sha256': {Path(module.__file__).name: _sha(Path(module.__file__).read_bytes())
                                  for module in (assessment_distill, corpus_audit, review_corpus)},
                'runtime': {'python_implementation': platform.python_implementation(),
                            'python_version': platform.python_version(),
                            'unicode_database_version': unicodedata.unidata_version},
                'source_snapshots': [{'sha256': digest, 'records': count} for _, digest, count in source_snapshots],
                'counts': {split: len(rows) for split, rows in output_rows.items()},
                'matched_records': matched_records, 'matches_by_rule': dict(sorted(rule_counts.items())),
                'claims': {'redaction_complete': False, 'human_reviewed': False, 'original_source_verified': False,
                           'independent_split_verified': False, 'promotion_eligible': False},
                'outputs': {split: _sha(review_corpus.encoded_rows(rows)) for split, rows in output_rows.items() if rows}}
    output.mkdir(mode=0o700)
    for split, rows in output_rows.items():
        if rows:
            review_corpus.new_file(output / f'{split}.jsonl', review_corpus.encoded_rows(rows))
    review_corpus.new_file(output / 'manifest.json', (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode('utf-8'))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--policy-read-limit', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.plan, args.policy, args.output, args.policy_read_limit), indent=2, sort_keys=True))
    except (ScreenError, OSError, TypeError, ValueError, UnicodeError, OverflowError):
        raise SystemExit('Private corpus screening failed; check the inputs, bounds and split plan.') from None


if __name__ == '__main__':
    main()
