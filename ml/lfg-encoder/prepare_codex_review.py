#!/usr/bin/env python3
"""Create a private, unlabeled review workspace from verified session records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import corpus_audit
import inspect_codex_inventory
import review_corpus


def prepare(inventory: Path, assignments: Path, taxonomy: Path, output: Path,
            max_bytes: int, max_line_bytes: int) -> dict:
    raw = inventory.read_bytes()
    config = json.loads(assignments.read_bytes())
    if not isinstance(config, dict) or set(config) != {'schema_version', 'projects'} or \
            config['schema_version'] != 'lfg.codex-review-assignments.v1' or not isinstance(config['projects'], dict) or not config['projects']:
        raise ValueError('invalid project assignments')
    settings = config['projects']
    groups = {}
    for project, assignment in settings.items():
        if (not isinstance(project, str) or not project.strip() or not isinstance(assignment, dict)
                or set(assignment) != {'project_group', 'split'}
                or not isinstance(assignment['project_group'], str) or not assignment['project_group'].strip()
                or assignment['split'] not in corpus_audit.SPLITS):
            raise ValueError('invalid project assignment')
        group = assignment['project_group']
        if group in groups and groups[group] != assignment['split']:
            raise ValueError('project group crosses assigned splits')
        groups[group] = assignment['split']
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    selected = [row for row in rows if isinstance(row, dict) and row.get('project') in settings]
    if not selected:
        raise ValueError('no records match the assigned projects')
    texts = inspect_codex_inventory.verify_many(selected, max_bytes, max_line_bytes)
    if inventory.read_bytes() != raw:
        raise ValueError('inventory changed during verification')
    inventory_hash = hashlib.sha256(raw).hexdigest()
    taxonomy_raw = taxonomy.read_bytes()
    review_corpus.evaluate.load_taxonomy(taxonomy)
    by_project: dict[str, list[dict]] = {}
    seen: dict[str, str] = {}
    excluded = 0
    for item, text in zip(selected, texts, strict=True):
        project = item['project']
        split = settings[project]['split']
        normalized = corpus_audit.text_key(text)
        if normalized in seen:
            if seen[normalized] != split:
                raise ValueError('normalized duplicate crosses assigned splits')
            excluded += 1
            continue
        seen[normalized] = split
        session = hashlib.sha256(os.fsencode(item['source_file'])).hexdigest()
        identifier = hashlib.sha256((item['source_prefix_sha256'] + ':' + str(item['source_line'])).encode()).hexdigest()
        source = {'project': project, 'kind': 'codex-session-user-text',
                  'inventory_sha256': inventory_hash, **{key: item[key] for key in (
                      'source_file', 'source_file_sha256', 'source_line', 'source_line_sha256',
                      'source_prefix_sha256', 'source_prefix_bytes', 'text_sha256')}}
        by_project.setdefault(project, []).append({
            'id': f'codex-{identifier}', 'text': text, 'family': f'codex-session-{session}',
            'intents': [], 'capabilities': [], 'risks': [], 'labels_pending': True,
            'verified': False, 'source': source,
            'review_note': 'No labels proposed. Verify source, surrounding context, privacy, and every label.'})
    if taxonomy.read_bytes() != taxonomy_raw:
        raise ValueError('taxonomy changed during preparation')
    output.mkdir(mode=0o700)
    review_corpus.new_file(output / 'taxonomy.json', taxonomy_raw)
    sources = []
    for index, (project, records) in enumerate(sorted(by_project.items())):
        name = f'source-{index:04}.jsonl'
        review_corpus.new_file(output / name, review_corpus.encoded_rows(records))
        sources.append({'path': name, 'project': project, **settings[project]})
    plan = {'schema_version': corpus_audit.PLAN_SCHEMA, 'taxonomy': 'taxonomy.json',
            'review_ledger': None, 'sources': sources}
    review_corpus.new_file(output / 'plan.json', (json.dumps(plan, indent=2) + '\n').encode())
    workspace = review_corpus.init_workspace(output / 'plan.json', output / 'workspace')
    return {'selected_source_messages': len(selected), 'normalized_duplicates_excluded': excluded,
            'pending_review': workspace['examples'], 'pending_by_project': {k: len(v) for k, v in by_project.items()},
            'approved': 0, 'promotion_eligible': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--assignments', type=Path, required=True)
    parser.add_argument('--taxonomy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new private directory; will contain raw messages')
    parser.add_argument('--max-session-bytes', type=int, default=512 * 1024 * 1024)
    parser.add_argument('--max-line-bytes', type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()
    try:
        if args.max_session_bytes < 1 or args.max_line_bytes < 1:
            raise ValueError('byte limits must be positive')
        print(json.dumps(prepare(args.inventory, args.assignments, args.taxonomy, args.output,
                                 args.max_session_bytes, args.max_line_bytes), indent=2))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        raise SystemExit(f'review preparation rejected: {error}') from error


if __name__ == '__main__':
    main()
