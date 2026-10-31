#!/usr/bin/env python3
"""Flag configurable review markers in training-only workspaces without editing them."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import combine_reviewed_workspaces as combine
import corpus_audit
import review_corpus
import train


def load_config(path: Path):
    raw = path.read_bytes()
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'revision', 'max_text_bytes', 'patterns'}
            or value['schema_version'] != 'lfg.review-triage-config.v1'
            or not isinstance(value['revision'], str) or not value['revision'].strip()
            or type(value['max_text_bytes']) is not int or value['max_text_bytes'] < 1
            or not isinstance(value['patterns'], list) or not value['patterns']):
        raise ValueError('invalid triage configuration')
    compiled, ids = [], set()
    for item in value['patterns']:
        if (not isinstance(item, dict) or set(item) != {'id', 'expression', 'ignore_case'}
                or not isinstance(item['id'], str) or not item['id'].strip() or item['id'] in ids
                or item['id'] in ('scan-limit-exceeded', 'stale-annotation', 'pending')
                or not isinstance(item['expression'], str) or not item['expression']
                or type(item['ignore_case']) is not bool):
            raise ValueError('invalid triage pattern')
        ids.add(item['id'])
        compiled.append((item['id'], re.compile(item['expression'], re.IGNORECASE if item['ignore_case'] else 0)))
    if path.read_bytes() != raw:
        raise ValueError('triage configuration changed')
    return value, compiled, hashlib.sha256(raw).hexdigest()


def triage(paths: list[Path], config_path: Path) -> dict:
    workspaces = combine.workspace_paths(paths)
    config, patterns, config_hash = load_config(config_path)
    code_paths = [Path(__file__), Path(combine.__file__), Path(corpus_audit.__file__),
                  Path(review_corpus.__file__), Path(train.__file__), Path(review_corpus.evaluate.__file__)]
    code_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}
    flagged, sources, counters = [], [], Counter()
    with contextlib.ExitStack() as stack:
        for path in sorted(workspaces, key=lambda p: os.path.normcase(str(p))):
            stack.enter_context(train.run_lock(path / 'review'))
        raw_plans = []
        for workspace in workspaces:
            raw = review_corpus.private_path(workspace, 'plan.json').read_bytes()
            plan = json.loads(raw)
            if (not isinstance(plan, dict) or not isinstance(plan.get('sources'), list) or not plan['sources']
                    or any(not isinstance(source, dict) or source.get('split') != 'train' for source in plan['sources'])):
                raise ValueError('triage accepts only training sources')
            raw_plans.append(raw)
        for index, (workspace, raw_plan) in enumerate(zip(workspaces, raw_plans, strict=True)):
            plan, audit, ledger = review_corpus.load_workspace(workspace)
            if hashlib.sha256(raw_plan).hexdigest() != audit['plan_sha256']:
                raise ValueError('workspace plan changed after split check')
            pending = [key for key, item in ledger.items() if item['decision'] == 'pending']
            offsets = {key: offset for offset, key in enumerate(pending)}
            sources.append({'workspace_index': index, 'plan_sha256': audit['plan_sha256'],
                            'ledger_sha256': audit['review_ledger_sha256'], 'taxonomy_sha256': audit['taxonomy_sha256'],
                            'source_sha256': [item['sha256'] for item in audit['sources']], 'pending': len(pending)})
            counters['pending'] += len(pending)
            for source in plan['sources']:
                rows, _ = train.load_dataset(review_corpus.private_path(workspace, source['path']), allow_unlabeled_review=True)
                for row in rows:
                    item = ledger[row['id']]
                    if item['decision'] != 'pending':
                        continue
                    reasons = []
                    annotation = corpus_audit.annotation_digest(row, source['project'], source['split'], audit['taxonomy_sha256'])
                    if item['annotation_sha256'] != annotation or row['verified']:
                        reasons.append('stale-annotation')
                    if len(row['text'].encode('utf-8')) > config['max_text_bytes']:
                        reasons.append('scan-limit-exceeded')
                    else:
                        reasons.extend(key for key, pattern in patterns if pattern.search(row['text']))
                    if reasons:
                        flagged.append({'workspace_index': index, 'id': row['id'], 'project': source['project'],
                                        'annotation_sha256': annotation, 'pending_offset': offsets[row['id']],
                                        'reasons': reasons})
                        counters.update(reasons)
            if review_corpus.load_workspace(workspace)[1] != audit:
                raise ValueError('workspace changed during triage')
        if load_config(config_path)[2] != config_hash or any(
                hashlib.sha256(path.read_bytes()).hexdigest() != code_hashes[path.name] for path in code_paths):
            raise ValueError('triage code or configuration changed')
    return {'schema_version': 'lfg.training-review-triage.v1', 'scope': 'training-candidates-only',
            'configuration_revision': config['revision'], 'configuration_sha256': config_hash,
            'code_sha256': code_hashes, 'workspaces': sources, 'pending': counters.pop('pending'),
            'flagged': len(flagged), 'reason_counts': dict(sorted(counters.items())), 'items': flagged,
            'review_decisions_made': False, 'promotion_eligible': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, action='append', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('output already exists')
        output = args.output.resolve()
        workspaces = combine.workspace_paths(args.workspace)
        if any(output == workspace or output.is_relative_to(workspace) for workspace in workspaces):
            raise ValueError('output must be outside review workspaces')
        report = triage(workspaces, args.config)
        raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as file:
            file.write(raw)
        print(json.dumps({'pending': report['pending'], 'flagged': report['flagged'],
                          'output_sha256': hashlib.sha256(raw).hexdigest(), 'review_decisions_made': False}))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, re.error) as error:
        raise SystemExit(f'triage rejected: {error}') from error


if __name__ == '__main__':
    main()
