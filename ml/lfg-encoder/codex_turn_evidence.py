#!/usr/bin/env python3
"""Recover private, unreviewed per-turn usage evidence, without inferring outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import corpus_audit
import inspect_codex_inventory
import review_corpus


CONFIG_KEYS = {'max_plan_bytes', 'max_candidate_file_bytes', 'max_total_candidate_bytes',
               'max_source_prefix_bytes', 'max_total_prefix_bytes', 'max_line_bytes',
               'max_candidates', 'max_contexts_per_session', 'max_metadata_chars'}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_bounded(path: Path, maximum: int) -> bytes:
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('input exceeds configured byte limit')
    return raw


def require_private_output(path: Path) -> None:
    path = path.resolve()
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    try:
        root = subprocess.run(['git', '-C', str(ancestor), 'rev-parse', '--show-toplevel'],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
        if root.returncode:
            if any((directory / '.git').exists() for directory in (ancestor, *ancestor.parents)):
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
            raise ValueError('output must be untracked and ignored')
    except OSError:
        raise ValueError('cannot verify output privacy') from None


def load_config(path: Path) -> tuple[dict, str]:
    raw = read_bounded(path, 65536)
    config = json.loads(raw)
    if (not isinstance(config, dict) or set(config) != CONFIG_KEYS | {'schema_version'}
            or config['schema_version'] != 'lfg.codex-turn-evidence-config.v1'
            or any(type(config[key]) is not int or config[key] < 1 for key in CONFIG_KEYS)):
        raise ValueError('invalid evidence configuration')
    return config, sha(raw)


def metadata(value, maximum: int) -> str | None:
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum
            or any(ord(character) < 32 or ord(character) == 127 for character in value)):
        return None
    return value


def prepare(plan_path: Path, config_path: Path, output: Path) -> dict:
    modules = (inspect_codex_inventory, corpus_audit, review_corpus)
    code = {Path(module.__file__).name: sha(Path(module.__file__).read_bytes()) for module in modules}
    code[Path(__file__).name] = sha(Path(__file__).read_bytes())
    config, config_hash = load_config(config_path)
    raw_plan = read_bounded(plan_path, config['max_plan_bytes'])
    plan = json.loads(raw_plan)
    sources = plan.get('sources') if isinstance(plan, dict) else None
    if (not isinstance(plan, dict) or plan.get('schema_version') != corpus_audit.DEVELOPMENT_PLAN_SCHEMA
            or not isinstance(sources, list) or not sources
            or any(not isinstance(source, dict) or source.get('split') not in ('train', 'validation')
                   or not isinstance(source.get('project'), str) or not source['project']
                   or not isinstance(source.get('path'), str) or not source['path'] for source in sources)):
        raise ValueError('development plan must contain only train and validation sources')
    require_private_output(output)
    if output.exists():
        raise ValueError('output already exists')
    groups, source_hashes, ids, projects, project_groups, families, texts = {}, [], set(), {}, {}, {}, {}
    total_bytes = 0
    for source in sources:
        project, split = source['project'], source['split']
        group = source.get('project_group', project)
        if (not isinstance(group, str) or not group
                or (project in projects and projects[project] != (group, split))
                or (group in project_groups and project_groups[group] != split)):
            raise ValueError('declared project group crosses split')
        projects[project], project_groups[group] = (group, split), split
        path = corpus_audit.resolve(plan_path.parent, source['path'])
        raw = read_bounded(path, config['max_candidate_file_bytes'])
        total_bytes += len(raw)
        if total_bytes > config['max_total_candidate_bytes']:
            raise ValueError('candidate files exceed configured total')
        source_hashes.append((path, sha(raw)))
        for line in raw.splitlines():
            row = json.loads(line)
            provenance = row.get('source') if isinstance(row, dict) else None
            if not isinstance(provenance, dict) or provenance.get('project') != source['project']:
                raise ValueError('candidate source does not match project')
            if provenance.get('kind') != 'codex-session-user-text':
                continue
            if (not isinstance(row.get('id'), str) or not row['id'] or row['id'] in ids
                    or not isinstance(row.get('family'), str) or not row['family']
                    or not isinstance(row.get('text'), str)):
                raise ValueError('invalid candidate identity')
            family, text_key = row['family'], corpus_audit.text_key(row['text'])
            if (family in families and families[family] != split) or (text_key in texts and texts[text_key] != split):
                raise ValueError('declared family or normalized text crosses split')
            families[family], texts[text_key] = split, split
            ids.add(row['id'])
            if len(ids) > config['max_candidates']:
                raise ValueError('candidate count exceeds configured limit')
            record = {'schema_version': 'lfg.unreviewed-codex-message.v2', **provenance}
            inspect_codex_inventory.verify_record(record)
            if not Path(record['source_file']).is_absolute():
                raise ValueError('source path must be absolute')
            groups.setdefault(record['source_file'], []).append((row, source['split'], record))
    if not groups:
        raise ValueError('development plan has no Codex candidates')
    if sum(max(record['source_prefix_bytes'] for _, _, record in group) for group in groups.values()) > config['max_total_prefix_bytes']:
        raise ValueError('session prefixes exceed configured total')
    rows = []
    for name, group in sorted(groups.items()):
        ordered = sorted(enumerate(group), key=lambda item: item[1][2]['source_line'])
        if len({item[2]['source_line'] for _, item in ordered}) != len(ordered):
            raise ValueError('duplicate selected session line')
        contexts = {}
        def prior(number: int, line: bytes) -> None:
            event = json.loads(line)
            if not isinstance(event, dict) or event.get('type') != 'turn_context':
                return
            payload = event.get('payload')
            if not isinstance(payload, dict):
                return
            turn = metadata(payload.get('turn_id'), config['max_metadata_chars'])
            if turn is None:
                return
            if turn not in contexts and len(contexts) >= config['max_contexts_per_session']:
                raise ValueError('session exceeds context count limit')
            item = {'model': metadata(payload.get('model'), config['max_metadata_chars']),
                    'effort': metadata(payload.get('effort'), config['max_metadata_chars']),
                    'context_line': number, 'context_line_sha256': sha(line.rstrip(b'\r\n')),
                    'consistent_contexts': 1}
            if turn not in contexts:
                contexts[turn] = item
            elif contexts[turn] is not None:
                previous = contexts[turn]
                if (previous['model'], previous['effort']) != (item['model'], item['effort']):
                    contexts[turn] = None
                else:
                    item['consistent_contexts'] = previous['consistent_contexts'] + 1
                    contexts[turn] = item
        def selected(line: bytes, record: dict) -> dict:
            text = inspect_codex_inventory.message_text(line, record)
            event = json.loads(line)
            meta = event['payload']['internal_chat_message_metadata_passthrough']
            turn = metadata(meta.get('turn_id'), config['max_metadata_chars'])
            return {'text_sha256': sha(text.encode('utf-8')), 'turn': turn,
                    'context': contexts.get(turn) if turn is not None else None,
                    'ambiguous': turn in contexts and contexts[turn] is None if turn is not None else False}
        verified = inspect_codex_inventory.verify_group(
            Path(name), [(index, item[2]) for index, item in ordered],
            config['max_source_prefix_bytes'], config['max_line_bytes'], prior, selected)
        for index, evidence in verified:
            row, split, record = group[index]
            context = evidence['context'] if sha(row['text'].encode('utf-8')) == evidence['text_sha256'] else None
            status = ('candidate_text_changed' if sha(row['text'].encode('utf-8')) != evidence['text_sha256'] else
                      'ambiguous_turn_context' if evidence['ambiguous'] else
                      'matched_turn_context' if context is not None else 'turn_context_missing')
            rows.append({'schema_version': 'lfg.codex-turn-evidence.v1', 'id': row['id'],
                         'project': record['project'], 'family': row['family'], 'split': split,
                         'source_prefix_sha256': record['source_prefix_sha256'],
                         'source_line_sha256': record['source_line_sha256'],
                         'text_sha256': evidence['text_sha256'], 'status': status,
                         'turn_id_sha256': sha(evidence['turn'].encode('utf-8')) if evidence['turn'] else None,
                         'context': context, 'outcome': 'unknown', 'agent_count': None,
                         'review_status': 'unreviewed', 'promotion_eligible': False})
    if (sha(read_bounded(plan_path, config['max_plan_bytes'])) != sha(raw_plan)
            or load_config(config_path)[1] != config_hash
            or any(sha(read_bounded(path, config['max_candidate_file_bytes'])) != digest for path, digest in source_hashes)
            or any(sha(Path(module.__file__).read_bytes()) != code[Path(module.__file__).name] for module in modules)
            or sha(Path(__file__).read_bytes()) != code[Path(__file__).name]):
        raise ValueError('plan, configuration or candidate files changed')
    provenance = {'plan_sha256': sha(raw_plan), 'config_sha256': config_hash,
                  'code_sha256': code, 'python_version': platform.python_version()}
    for row in rows:
        row['extraction'] = provenance
    payload = review_corpus.encoded_rows(sorted(rows, key=lambda row: row['id']))
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(payload)
    return {'schema_version': 'lfg.codex-turn-evidence-summary.v1', 'records': len(rows),
            'statuses': dict(sorted(Counter(row['status'] for row in rows).items())),
            'matched_model': sum(row['context'] is not None and row['context']['model'] is not None for row in rows),
            'matched_effort': sum(row['context'] is not None and row['context']['effort'] is not None for row in rows),
            'output_sha256': sha(payload), 'plan_sha256': sha(raw_plan), 'config_sha256': config_hash,
            'source_verifier_sha256': code[Path(inspect_codex_inventory.__file__).name],
            'evidence_code_sha256': code[Path(__file__).name], 'promotion_eligible': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.plan, args.config, args.output), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, OverflowError):
        raise SystemExit('Turn evidence preparation failed; check source, configuration and limits.') from None


if __name__ == '__main__':
    main()
