#!/usr/bin/env python3
"""Select and rescreen real train requests for a local assessment teacher."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import unicodedata

import screen_private_corpus as screen
from owned_sft_checkpoint import private_directory


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def select(dataset: Path, policy_path: Path, config_path: Path, output: Path) -> dict:
    raw_config = screen._read(config_path, 65536, 'selection_config_unavailable')
    config = screen._json(raw_config, 'invalid_selection_config')
    fields = ('max_manifest_bytes', 'max_train_bytes', 'max_record_bytes', 'max_candidates',
              'max_per_project', 'max_selected', 'min_text_bytes', 'max_text_bytes')
    if (not isinstance(config, dict) or config.get('schema_version') != 'lfg.teacher-request-selection.v1'
            or any(type(config.get(k)) is not int or config[k] < 1 for k in fields)
            or config['min_text_bytes'] > config['max_text_bytes']
            or not isinstance(config.get('seed'), str) or not config['seed']):
        raise ValueError('invalid selection configuration')
    excluded = config.get('exclude_ids', [])
    if (not isinstance(excluded, list) or any(not isinstance(item, str) or not item for item in excluded)
            or len(set(excluded)) != len(excluded)):
        raise ValueError('invalid excluded request identifiers')
    excluded = set(excluded)
    policy, rules, policy_sha = screen.load_policy(policy_path, config['max_manifest_bytes'])
    manifest_raw = screen._read(dataset / 'manifest.json', config['max_manifest_bytes'], 'dataset_manifest_unavailable')
    manifest = screen._json(manifest_raw, 'invalid_dataset_manifest')
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 'lfg.private-corpus-screen-manifest.v1'
            or not isinstance(manifest.get('outputs'), dict) or not isinstance(manifest['outputs'].get('train'), str)):
        raise ValueError('invalid screened dataset manifest')
    train_raw = screen._read(dataset / 'train.jsonl', config['max_train_bytes'], 'train_source_unavailable')
    if sha(train_raw) != manifest['outputs']['train']:
        raise ValueError('screened train digest mismatch')
    if output.exists():
        raise ValueError('output already exists')
    private_directory(output)
    by_project, seen_ids, seen_text, redactions = defaultdict(list), set(), set(), Counter()
    lines = train_raw.splitlines()
    if not lines or len(lines) > config['max_candidates']:
        raise ValueError('invalid candidate count')
    for line in lines:
        if not line.strip() or len(line) > config['max_record_bytes']:
            raise ValueError('invalid candidate record')
        row = screen._json(line, 'invalid_candidate')
        if (not isinstance(row, dict) or row.get('schema_version') != screen.OUTPUT_VERSION
                or row.get('split') != 'train' or row.get('review_status') != 'unreviewed'
                or any(not isinstance(row.get(k), str) or not row[k].strip()
                       for k in ('id', 'project', 'project_group', 'family', 'text', 'screened_text_sha256', 'source_text_sha256'))
                or sha(row['text'].encode()) != row['screened_text_sha256']
                or len(row['source_text_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['source_text_sha256'])
                or row['id'] in seen_ids):
            raise ValueError('invalid train candidate identity')
        seen_ids.add(row['id'])
        if row['id'] in excluded:
            continue
        text, counts = screen._screen(row['text'], rules, policy['limits'])
        size = len(text.encode())
        if not config['min_text_bytes'] <= size <= config['max_text_bytes']:
            continue
        key = sha(' '.join(unicodedata.normalize('NFKC', text).casefold().split()).encode())
        if key in seen_text:
            continue
        seen_text.add(key)
        rank = sha((config['seed'] + '\0' + row['id']).encode())
        by_project[row['project']].append((rank, row, text, counts))
    # Round-robin across projects so a large repository cannot consume the pilot.
    for entries in by_project.values():
        entries.sort(key=lambda item: item[0])
    selected = []
    for index in range(config['max_per_project']):
        for project in sorted(by_project):
            if index < len(by_project[project]) and len(selected) < config['max_selected']:
                _, row, text, counts = by_project[project][index]
                redactions.update(counts)
                selected.append({'id': row['id'], 'project': project, 'project_group': row['project_group'],
                                 'family': row['family'], 'text': text, 'screened_text_sha256': sha(text.encode()),
                                 'source_text_sha256_claim': row['source_text_sha256'],
                                 'candidate_file_sha256_claim': row.get('candidate_file_sha256'),
                                 'review_status': 'unreviewed', 'outcome': 'unknown', 'split': 'train'})
    if not selected:
        raise ValueError('no eligible train requests')
    if (screen._read(config_path, 65536, 'selection_config_changed') != raw_config
            or screen._read(dataset / 'manifest.json', config['max_manifest_bytes'], 'dataset_manifest_changed') != manifest_raw
            or screen._read(dataset / 'train.jsonl', config['max_train_bytes'], 'train_source_changed') != train_raw
            or screen.load_policy(policy_path, config['max_manifest_bytes'])[2] != policy_sha):
        raise ValueError('selection inputs changed')
    records = b''.join(encoded(row) for row in selected)
    report = {'schema_version': 'lfg.teacher-request-selection-result.v1', 'selection_config_sha256': sha(raw_config),
              'screened_manifest_sha256': sha(manifest_raw), 'screened_train_sha256': sha(train_raw),
              'screen_policy_sha256': policy_sha, 'code_sha256': sha(Path(__file__).read_bytes()),
              'requests_sha256': sha(records), 'requests': len(selected),
              'projects': dict(sorted(Counter(row['project'] for row in selected).items())),
              'redaction_counts': dict(sorted(redactions.items())), 'validation_opened': False, 'test_opened': False,
              'redaction_complete': False, 'human_reviewed': False, 'promotion_eligible': False}
    output.mkdir(mode=0o700)
    for name, raw in (('requests.jsonl', records), ('manifest.json', encoded(report))):
        fd = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'policy', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = select(args.dataset, args.policy, args.config, args.output)
        print(json.dumps({k: result[k] for k in ('requests', 'projects', 'redaction_counts', 'validation_opened', 'test_opened')}))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        raise SystemExit(f'Teacher request selection failed ({type(error).__name__})') from None
