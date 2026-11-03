"""Prepare private, explicitly sourced teacher traces for exploratory SFT.

This utility does not run a teacher or train, evaluate, or promote a model.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unicodedata


PLAN_SCHEMA = 'lfg.assessment-distillation-plan.v1'
SOURCE_FIELDS = {'path', 'sha256', 'project', 'project_group', 'split', 'origin'}
RECORD_FIELDS = {'id', 'request', 'response', 'teacher', 'review_status', 'outcome', 'source_ref', 'source_sha256'}
PLAN_FIELDS = {'schema_version', 'revision', 'task', 'system_prompt', 'authorization_attestation',
               'redaction_attestation', 'forbidden_literals', 'include_outcomes', 'max_source_bytes',
               'max_total_bytes', 'max_examples', 'sources'}
REPOSITORY = Path(__file__).resolve().parents[2]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def contains_forbidden(value: object, literals: list[str]) -> bool:
    return any(literal in text for text in strings(value) for literal in literals)


def underlying_request(row: dict) -> str:
    """Return the original request after validating optional rendered-context provenance."""
    context_sha = row.get('project_context_sha256')
    context_length = row.get('project_context_byte_length')
    if 'project_context_sha256' not in row and 'project_context_byte_length' not in row:
        return row['request']
    request_bytes = row['request'].encode('utf-8')
    if (not isinstance(context_sha, str) or len(context_sha) != 64
            or any(char not in '0123456789abcdef' for char in context_sha)
            or type(context_length) is not int or context_length < 1
            or len(request_bytes) < context_length + 2
            or request_bytes[context_length:context_length + 2] != b'\n\n'
            or sha(request_bytes[:context_length]) != context_sha):
        raise ValueError('invalid project context digest extension')
    try:
        request = request_bytes[context_length + 2:].decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError('invalid project context request encoding') from None
    if not request.strip():
        raise ValueError('invalid project context request')
    return request


def private_output(path: Path) -> None:
    """Require outputs in any Git working tree to be ignored by that tree."""
    path = path.resolve()
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    root = subprocess.run(['git', '-C', str(ancestor), 'rev-parse', '--show-toplevel'],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, text=True)
    if root.returncode != 0:
        if any((directory / '.git').exists() for directory in (ancestor, *ancestor.parents)):
            raise ValueError('cannot verify Git ignore status for output')
        return
    repository = Path(root.stdout.rstrip('\r\n')).resolve()
    if not path.is_relative_to(repository):
        raise ValueError('cannot resolve output within its Git working tree')
    relative = path.relative_to(repository).as_posix() + '/train.jsonl'
    check = subprocess.run(['git', '-C', str(repository), 'check-ignore', '-q', '--no-index', '--', relative],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if check.returncode != 0:
        raise ValueError('output inside a Git working tree must be Git-ignored')


def build(plan_path: Path, output: Path) -> dict:
    raw_plan = plan_path.read_bytes()
    plan = json.loads(raw_plan)
    if (not isinstance(plan, dict) or set(plan) != PLAN_FIELDS or plan['schema_version'] != PLAN_SCHEMA
            or any(not nonempty(plan[field]) for field in ('revision', 'task', 'system_prompt',
                                                           'authorization_attestation', 'redaction_attestation'))
            or not isinstance(plan['sources'], list) or not plan['sources']
            or not isinstance(plan['forbidden_literals'], list)
            or any(not nonempty(item) for item in plan['forbidden_literals'])
            or not isinstance(plan['include_outcomes'], list) or not plan['include_outcomes']
            or len(set(plan['include_outcomes'])) != len(plan['include_outcomes'])
            or any(item not in ('success', 'unknown', 'failure') for item in plan['include_outcomes'])
            or any(type(plan[field]) is not int or plan[field] < 1
                   for field in ('max_source_bytes', 'max_total_bytes', 'max_examples'))):
        raise ValueError('invalid assessment distillation plan')
    # Reject held-out sources before resolving or reading any source path.
    for source in plan['sources']:
        if (not isinstance(source, dict) or set(source) != SOURCE_FIELDS
                or source['split'] not in ('train', 'validation')
                or source['origin'] not in ('owner_project', 'owner_chat')
                or any(not nonempty(source[field]) for field in ('path', 'project', 'project_group'))
                or not isinstance(source['sha256'], str) or len(source['sha256']) != 64
                or any(char not in '0123456789abcdef' for char in source['sha256'])):
            raise ValueError('invalid source; only selected owner-project/chat train or validation sources are supported')
    exported_plan = {key: plan[key] for key in ('system_prompt', 'task', 'revision',
                                                'authorization_attestation', 'redaction_attestation')}
    if contains_forbidden(exported_plan, plan['forbidden_literals']):
        raise ValueError('plan text contains a configured forbidden literal')
    private_output(output)
    if output.exists():
        raise ValueError('output directory must be new')

    ids, groups, projects, texts, original_sources = set(), {}, {}, {}, {}
    source_evidence, selected = [], {'train': [], 'validation': []}
    counts = {split: {'review_status': Counter(), 'outcome': Counter(), 'seen': 0, 'selected': 0}
              for split in selected}
    total_bytes = 0
    for source in plan['sources']:
        group, project, split = source['project_group'], source['project'], source['split']
        if contains_forbidden([group, project], plan['forbidden_literals']):
            raise ValueError('source metadata contains a configured forbidden literal')
        if group in groups and groups[group] != split:
            raise ValueError('project group spans train and validation')
        if project in projects and projects[project] != (group, split):
            raise ValueError('project has inconsistent group or split')
        groups[group], projects[project] = split, (group, split)
        path = (plan_path.parent / source['path']).resolve()
        with path.open('rb') as stream:
            raw = stream.read(plan['max_source_bytes'] + 1)
        total_bytes += len(raw)
        if len(raw) > plan['max_source_bytes'] or total_bytes > plan['max_total_bytes']:
            raise ValueError('source exceeds configured byte limit')
        if sha(raw) != source['sha256']:
            raise ValueError('source digest does not match plan')
        rows = raw.decode('utf-8').splitlines()
        if not rows or any(not line.strip() for line in rows):
            raise ValueError('source must contain nonempty JSONL records')
        for line in rows:
            row = json.loads(line)
            if (not isinstance(row, dict) or not RECORD_FIELDS.issubset(row)
                    or any(not nonempty(row[field]) for field in ('id', 'request', 'response', 'teacher', 'source_ref'))
                    or not isinstance(row['source_sha256'], str) or len(row['source_sha256']) != 64
                    or any(char not in '0123456789abcdef' for char in row['source_sha256'])
                    or row['review_status'] not in ('unreviewed', 'reviewed')
                    or row['outcome'] not in ('success', 'unknown', 'failure')):
                raise ValueError('invalid teacher trace')
            dedup_request = underlying_request(row)
            teacher_system_sha = row.get('teacher_system_sha256')
            if ('teacher_system_sha256' in row
                    and (not isinstance(teacher_system_sha, str) or len(teacher_system_sha) != 64
                         or any(char not in '0123456789abcdef' for char in teacher_system_sha)
                         or teacher_system_sha != sha(plan['system_prompt'].encode('utf-8')))):
                raise ValueError('teacher system prompt digest does not match distillation plan')
            if row['id'] in ids:
                raise ValueError('teacher trace IDs must be globally unique')
            ids.add(row['id'])
            original_sha = row['source_sha256']
            if original_sha in original_sources and original_sources[original_sha] != split:
                raise ValueError('declared original source spans train and validation')
            original_sources[original_sha] = split
            if len(ids) > plan['max_examples']:
                raise ValueError('trace count exceeds configured limit')
            if contains_forbidden(row, plan['forbidden_literals']):
                raise ValueError('trace contains a configured forbidden literal')
            text_key = sha(' '.join(unicodedata.normalize('NFKC', dedup_request).casefold().split()).encode('utf-8'))
            if text_key in texts:
                raise ValueError('duplicate normalized request across sources or splits')
            texts[text_key] = split
            counts[split]['seen'] += 1
            counts[split]['review_status'][row['review_status']] += 1
            counts[split]['outcome'][row['outcome']] += 1
            if row['outcome'] not in plan['include_outcomes']:
                continue
            selected[split].append({'id': row['id'], 'messages': [
                {'role': 'system', 'content': plan['system_prompt']},
                {'role': 'user', 'content': row['request']},
                {'role': 'assistant', 'content': row['response']}],
                'metadata': {'task': plan['task'], 'project': project, 'project_group': group,
                             'origin': source['origin'], 'trace_file_sha256': source['sha256'],
                             'original_source_ref_claim': row['source_ref'],
                             'original_source_sha256_claim': row['source_sha256'],
                             'teacher': row['teacher'], 'review_status_claim': row['review_status'],
                             'outcome_claim': row['outcome'],
                             'trace_extensions': {key: value for key, value in row.items()
                                                  if key not in RECORD_FIELDS}}})
            counts[split]['selected'] += 1
        source_evidence.append({'path': path, 'sha256': sha(raw), 'split': split, 'records': len(rows),
                                'project': project, 'project_group': group, 'origin': source['origin']})
    if not selected['train']:
        raise ValueError('no selected training traces')
    # An optimistic identity check, not a lock: source files must not change during preparation.
    def stable_source(item):
        with item['path'].open('rb') as stream:
            return sha(stream.read(plan['max_source_bytes'] + 1)) == item['sha256']
    if plan_path.read_bytes() != raw_plan or any(not stable_source(item) for item in source_evidence):
        raise ValueError('plan or source changed during preparation')
    files = {split + '.jsonl': b''.join(canonical(row) for row in rows)
             for split, rows in selected.items() if rows}
    report = {'schema_version': 'lfg.assessment-distillation-dataset.v1', 'plan_sha256': sha(raw_plan),
              'builder_sha256': sha(Path(__file__).read_bytes()), 'revision': plan['revision'], 'task': plan['task'],
              'authorization_attestation': plan['authorization_attestation'],
              'redaction_attestation': plan['redaction_attestation'],
              'forbidden_literal_count': len(plan['forbidden_literals']),
              'included_outcomes': plan['include_outcomes'],
              'sources': [{key: value for key, value in item.items() if key != 'path'} for item in source_evidence],
              'counts': {split: {key: dict(value) if isinstance(value, Counter) else value
                                 for key, value in data.items()} for split, data in counts.items()},
              'files': {name: sha(raw) for name, raw in files.items()},
              'validation_is_independent': False, 'test_assessed': False, 'promotion_eligible': False,
              'claims_verified': False}
    output.mkdir(mode=0o700)
    for name, raw in {**files, 'manifest.json': canonical(report)}.items():
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
    return {'schema_version': 'lfg.assessment-distillation-result.v1', 'counts': report['counts'],
            'manifest_sha256': sha(canonical(report)), 'promotion_eligible': False}
