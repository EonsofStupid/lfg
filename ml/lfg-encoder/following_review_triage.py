#!/usr/bin/env python3
"""Inspect bounded later user messages for markers in a training-only workspace."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import corpus_audit
import inspect_codex_inventory as inspect
import review_corpus
import train


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    limits = {'max_session_bytes', 'max_line_bytes', 'max_messages', 'max_context_bytes', 'max_scan_lines'}
    if (not isinstance(value, dict) or set(value) != limits | {'schema_version', 'revision', 'markers'}
            or value['schema_version'] != 'lfg.following-review-triage-config.v1'
            or not isinstance(value['revision'], str) or not value['revision'].strip()
            or any(type(value[key]) is not int or value[key] < 1 for key in limits)
            or not isinstance(value['markers'], list) or not value['markers']):
        raise ValueError('invalid following triage configuration')
    ids = set()
    for marker in value['markers']:
        if (not isinstance(marker, dict) or set(marker) != {'id', 'text', 'ignore_case'}
                or not isinstance(marker['id'], str) or not marker['id'].strip() or marker['id'] in ids
                or not isinstance(marker['text'], str) or not marker['text'].strip()
                or type(marker['ignore_case']) is not bool):
            raise ValueError('invalid following triage marker')
        ids.add(marker['id'])
    if path.read_bytes() != raw:
        raise ValueError('following triage configuration changed')
    return value, sha(raw)


def signature(path: Path) -> tuple[int, int, int]:
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_ino


def session_scan(path: Path, candidates: list[dict], config: dict) -> list[dict]:
    """Verify selected prefixes and inspect following messages in one bounded pass."""
    initial = signature(path)
    records = [{**candidate['source'], 'schema_version': 'lfg.unreviewed-codex-message.v2'} for candidate in candidates]
    for record in records:
        inspect.verify_record(record)
        if record['source_prefix_bytes'] > config['max_session_bytes']:
            raise ValueError('source prefix exceeds configured limit')
    if len({record['source_line'] for record in records}) != len(records):
        raise ValueError('duplicate source positions')
    positions = {record['source_line']: (record, candidate) for record, candidate in zip(records, candidates, strict=True)}
    active: dict[int, dict] = {}
    completed: list[dict] = []
    seen = set()
    total = 0
    hasher = hashlib.sha256()

    def finish(key: int, reason: str) -> None:
        state = active.pop(key)
        completed.append({**state['candidate']['report'], 'hits': state['hits'], 'context_end': reason})

    with path.open('rb') as source:
        line_number = 0
        while True:
            line = source.readline(config['max_line_bytes'] + 1)
            if not line:
                break
            if len(line) > config['max_line_bytes']:
                raise ValueError('source line exceeds configured limit')
            total += len(line)
            if total > config['max_session_bytes']:
                raise ValueError('source scan exceeds configured limit')
            line_number += 1
            hasher.update(line)
            for key in list(active):
                if line_number - key > config['max_scan_lines']:
                    finish(key, 'scan_limit')
            if active:
                row = json.loads(line)
                payload = row.get('payload') if isinstance(row, dict) else None
                content = payload.get('content') if isinstance(payload, dict) else None
                metadata = payload.get('internal_chat_message_metadata_passthrough') if isinstance(payload, dict) else None
                if (isinstance(row, dict) and row.get('type') == 'response_item'
                        and isinstance(payload, dict) and payload.get('type') == 'message' and payload.get('role') == 'user'
                        and isinstance(metadata, dict) and metadata.get('content_item_kinds') == ['user.text']
                        and isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict)
                        and content[0].get('type') == 'input_text' and isinstance(content[0].get('text'), str)
                        and content[0]['text'].strip()):
                    text = content[0]['text']
                    size = len(text.encode('utf-8'))
                    for key in list(active):
                        state = active[key]
                        if state['bytes'] + size > config['max_context_bytes']:
                            finish(key, 'context_byte_limit')
                            continue
                        state['bytes'] += size
                        state['messages'] += 1
                        for marker in config['markers']:
                            candidate_text = text.casefold() if marker['ignore_case'] else text
                            literal = marker['text'].casefold() if marker['ignore_case'] else marker['text']
                            if literal in candidate_text:
                                state['hits'].append({'marker_id': marker['id'], 'source_line': line_number,
                                    'source_line_sha256': sha(line.rstrip(b'\r\n')), 'observed_prefix_sha256': hasher.hexdigest()})
                        if state['messages'] >= config['max_messages']:
                            finish(key, 'message_limit')
            if line_number in positions:
                record, candidate = positions[line_number]
                selected = line.rstrip(b'\r\n')
                if (total != record['source_prefix_bytes'] or hasher.hexdigest() != record['source_prefix_sha256']
                        or sha(selected) != record['source_line_sha256']
                        or inspect.message_text(selected, record) != candidate['text']):
                    raise ValueError('source prefix or review text does not match')
                active[line_number] = {'candidate': candidate, 'hits': [], 'messages': 0, 'bytes': 0}
                seen.add(line_number)
            if len(seen) == len(positions) and not active:
                break
    for key in list(active):
        finish(key, 'end_of_file')
    if seen != set(positions):
        raise ValueError('selected source position disappeared')
    if signature(path) != initial:
        # Appends are common in live sessions. Recheck exactly the bytes read;
        # a changed prefix still fails, and an observed EOF is no longer final.
        remaining = total
        recheck = hashlib.sha256()
        with path.open('rb') as source:
            while remaining:
                chunk = source.read(min(remaining, 1048576))
                if not chunk:
                    raise ValueError('source changed during context scan')
                recheck.update(chunk)
                remaining -= len(chunk)
        if recheck.hexdigest() != hasher.hexdigest():
            raise ValueError('source changed during context scan')
        for result in completed:
            if result['context_end'] == 'end_of_file':
                result['context_end'] = 'source_changed_after_eof'
    return completed


def triage(workspace: Path, config_path: Path) -> dict:
    workspace = workspace.resolve(strict=True)
    config, config_sha = load_config(config_path)
    code_paths = [Path(__file__), Path(inspect.__file__), Path(review_corpus.__file__),
                  Path(corpus_audit.__file__), Path(train.__file__), Path(review_corpus.evaluate.__file__),
                  Path(review_corpus.assessment_targets.__file__)]
    code_hashes = {path.name: sha(path.read_bytes()) for path in code_paths}
    raw_plan = review_corpus.private_path(workspace, 'plan.json').read_bytes()
    candidate_plan = json.loads(raw_plan)
    if (not isinstance(candidate_plan, dict) or not isinstance(candidate_plan.get('sources'), list)
            or not candidate_plan['sources'] or any(not isinstance(s, dict) or s.get('split') != 'train'
                                                     for s in candidate_plan['sources'])):
        raise ValueError('following triage accepts only training sources')
    plan, audit, ledger = review_corpus.load_workspace(workspace)
    if sha(raw_plan) != audit['plan_sha256']:
        raise ValueError('workspace plan changed after split check')
    grouped = defaultdict(list)
    counts = Counter()
    for source in plan['sources']:
        rows, _ = train.load_dataset(review_corpus.private_path(workspace, source['path']), allow_unlabeled_review=True)
        for row in rows:
            decision = ledger[row['id']]
            if decision['decision'] != 'pending':
                continue
            counts['pending'] += 1
            annotation = corpus_audit.annotation_digest(row, source['project'], source['split'], audit['taxonomy_sha256'])
            if decision['annotation_sha256'] != annotation or row['verified']:
                counts['stale'] += 1
                continue
            origin = row.get('source')
            if not isinstance(origin, dict) or origin.get('kind') != 'codex-session-user-text':
                counts['unsupported_source'] += 1
                continue
            if row.get('review_transforms'):
                counts['transformed'] += 1
                continue
            if not isinstance(origin.get('source_file'), str):
                counts['invalid_source'] += 1
                continue
            grouped[origin['source_file']].append({'source': origin, 'text': row['text'],
                'report': {'id': row['id'], 'project': source['project'], 'annotation_sha256': annotation}})
    results = []
    source_errors = Counter()
    for path, candidates in grouped.items():
        try:
            results.extend(session_scan(Path(path), candidates, config))
        except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
            counts['source_unavailable'] += len(candidates)
            reason = str(error)
            allowed = {'selected source position disappeared', 'source changed during context scan', 'source prefix exceeds configured limit',
                       'source scan exceeds configured limit', 'source line exceeds configured limit',
                       'source prefix or review text does not match', 'duplicate source positions'}
            source_errors[reason if reason in allowed else type(error).__name__] += len(candidates)
    for result in results:
        counts['checked'] += 1
        counts['context_' + result['context_end']] += 1
    hits = [result for result in results if result['hits']]
    if (review_corpus.load_workspace(workspace)[1] != audit or load_config(config_path)[1] != config_sha
            or any(sha(path.read_bytes()) != code_hashes[path.name] for path in code_paths)):
        raise ValueError('workspace, config or code changed during triage')
    return {'schema_version': 'lfg.following-review-triage.v1', 'scope': 'training-pending-only',
            'workspace_plan_sha256': audit['plan_sha256'], 'workspace_ledger_sha256': audit['review_ledger_sha256'],
            'taxonomy_sha256': audit['taxonomy_sha256'], 'workspace_sources': audit['sources'],
            'config_sha256': config_sha, 'code_sha256': code_hashes, 'counts': dict(sorted(counts.items())),
            'source_errors': dict(sorted(source_errors.items())), 'flagged': len(hits), 'items': hits, 'review_decisions_made': False,
            'notice': 'Optimistic read; matches are not labels, and no match does not prove clean context.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        workspace = args.workspace.resolve(strict=True)
        if output.exists() or output == workspace or output.is_relative_to(workspace):
            raise ValueError('output must be outside the review workspace')
        report = triage(workspace, args.config)
        raw = (json.dumps(report, sort_keys=True, indent=2) + '\n').encode()
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
        print(json.dumps({'counts': report['counts'], 'flagged': report['flagged'], 'report_sha256': sha(raw)}))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        raise SystemExit(f'following triage rejected: {error}') from error


if __name__ == '__main__':
    main()
