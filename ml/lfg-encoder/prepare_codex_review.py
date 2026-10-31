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
import train


def prepare(inventory: Path, assignments: Path, taxonomy: Path, output: Path,
            max_bytes: int, max_line_bytes: int, reference_plans: list[Path] | None = None) -> dict:
    raw = inventory.read_bytes()
    assignment_raw = assignments.read_bytes()
    config = json.loads(assignment_raw)
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
    reference_plans = reference_plans or []
    reference_reports = []
    reference_texts = set()
    reference_families: dict[str, set[str]] = {}
    for path in reference_plans:
        report, _ = corpus_audit.audit(path)
        plan_raw = path.read_bytes()
        if hashlib.sha256(plan_raw).hexdigest() != report['plan_sha256']:
            raise ValueError('reference plan changed during preparation')
        if hashlib.sha256(taxonomy_raw).hexdigest() != report['taxonomy_sha256']:
            raise ValueError('reference and candidate taxonomies differ')
        plan = json.loads(plan_raw)
        for source, evidence in zip(plan['sources'], report['sources'], strict=True):
            for project, assignment in settings.items():
                if ((source['project'] == project or source.get('project_group', source['project']) == assignment['project_group'])
                        and source['split'] != assignment['split']):
                    raise ValueError('candidate project or group crosses a reference split')
            ref_rows, digest = train.load_dataset(corpus_audit.resolve(path.parent, source['path']),
                                                   allow_unlabeled_review=True)
            if digest != evidence['sha256']:
                raise ValueError('reference source changed during preparation')
            for row in ref_rows:
                reference_texts.add(corpus_audit.text_key(row['text']))
                reference_families.setdefault(row['family'], set()).add(source['split'])
        reference_reports.append(report)
    by_project: dict[str, list[dict]] = {}
    seen: dict[str, str] = {}
    excluded = 0
    reference_excluded = 0
    for item, text in zip(selected, texts, strict=True):
        project = item['project']
        split = settings[project]['split']
        session = hashlib.sha256(os.fsencode(item['source_file'])).hexdigest()
        family = f'codex-session-{session}'
        if family in reference_families and reference_families[family] != {split}:
            raise ValueError('candidate session crosses a reference split')
        normalized = corpus_audit.text_key(text)
        if normalized in reference_texts:
            reference_excluded += 1
            continue
        if normalized in seen:
            if seen[normalized] != split:
                raise ValueError('normalized duplicate crosses assigned splits')
            excluded += 1
            continue
        seen[normalized] = split
        identifier = hashlib.sha256((item['source_prefix_sha256'] + ':' + str(item['source_line'])).encode()).hexdigest()
        source = {'project': project, 'kind': 'codex-session-user-text',
                  'inventory_sha256': inventory_hash, **{key: item[key] for key in (
                      'source_file', 'source_file_sha256', 'source_line', 'source_line_sha256',
                      'source_prefix_sha256', 'source_prefix_bytes', 'text_sha256')}}
        by_project.setdefault(project, []).append({
            'id': f'codex-{identifier}', 'text': text, 'family': family,
            'intents': [], 'capabilities': [], 'risks': [], 'labels_pending': True,
            'verified': False, 'source': source,
            'review_note': 'No labels proposed. Verify source, surrounding context, privacy, and every label.'})
    if not by_project:
        raise ValueError('no candidate records remain')
    if (taxonomy.read_bytes() != taxonomy_raw or assignments.read_bytes() != assignment_raw
            or inventory.read_bytes() != raw or any(corpus_audit.audit(path)[0] != report
                for path, report in zip(reference_plans, reference_reports, strict=True))):
        raise ValueError('inputs changed during preparation')
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
    summary = {'selected_source_messages': len(selected), 'normalized_duplicates_excluded': excluded,
               'reference_overlap_excluded': reference_excluded, 'reference_plan_count': len(reference_plans),
               'pending_review': workspace['examples'], 'pending_by_project': {k: len(v) for k, v in by_project.items()},
               'approved': 0, 'promotion_eligible': False}
    names = ['plan.json', 'taxonomy.json', 'reviews.jsonl'] + [source['path'] for source in sources]
    manifest = {'schema_version': 'lfg.codex-review-preparation.v1',
                'inventory_sha256': hashlib.sha256(raw).hexdigest(),
                'assignments_sha256': hashlib.sha256(assignment_raw).hexdigest(),
                'max_session_bytes': max_bytes, 'max_line_bytes': max_line_bytes,
                'reference_audit_sha256': [hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':'),
                                                                     allow_nan=False).encode()).hexdigest()
                                           for report in reference_reports],
                'preparer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'source_verifier_sha256': hashlib.sha256(Path(inspect_codex_inventory.__file__).read_bytes()).hexdigest(),
                'workspace_sha256': {name: hashlib.sha256((output / 'workspace' / name).read_bytes()).hexdigest()
                                     for name in names}, 'summary': summary}
    review_corpus.new_file(output / 'preparation-manifest.json',
                           (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--assignments', type=Path, required=True)
    parser.add_argument('--taxonomy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new private directory; will contain raw messages')
    parser.add_argument('--max-session-bytes', type=int, default=512 * 1024 * 1024)
    parser.add_argument('--max-line-bytes', type=int, default=16 * 1024 * 1024)
    parser.add_argument('--reference-plan', type=Path, action='append', default=[],
                        help='existing plan to check for overlap; may be repeated')
    args = parser.parse_args()
    try:
        if args.max_session_bytes < 1 or args.max_line_bytes < 1:
            raise ValueError('byte limits must be positive')
        print(json.dumps(prepare(args.inventory, args.assignments, args.taxonomy, args.output,
                                 args.max_session_bytes, args.max_line_bytes, args.reference_plan), indent=2))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        raise SystemExit(f'review preparation rejected: {error}') from error


if __name__ == '__main__':
    main()
