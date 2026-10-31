#!/usr/bin/env python3
"""Prepare pending test-review records from a verified checkpoint history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import corpus_audit
import inventory_checkpoint_prompts as inventory
import review_corpus
import train


def prepare(repo: Path, revision: str, project: str, project_group: str,
            source_inventory: Path, exclusions: Path, reference_plans: list[Path],
            taxonomy: Path, output: Path, max_history_bytes: int) -> dict:
    if not project.strip() or not project_group.strip():
        raise ValueError('project and group must be nonempty')
    raw_history = inventory.read_history(repo, revision, max_history_bytes)
    _, recovered = inventory.parse_history(raw_history, project, revision)
    inventory_raw = source_inventory.read_bytes()
    recorded = [json.loads(line) for line in inventory_raw.splitlines() if line.strip()]
    if recovered != recorded or not recovered:
        raise ValueError('inventory does not match the pinned source history')
    exclusion_raw = exclusions.read_bytes()
    prior = [json.loads(line) for line in exclusion_raw.splitlines() if line.strip()]
    if not prior:
        raise ValueError('at least one prior-inspection exclusion is required')
    excluded_ids = set()
    for row in prior:
        source = row.get('source') if isinstance(row, dict) else None
        if not isinstance(source, dict) or source.get('project') != project or not inventory.OID.fullmatch(source.get('commit', '')):
            raise ValueError('prior inspection requires source project and commit')
        excluded_ids.add(source['commit'])
    by_id = {row['id']: row for row in recovered}
    if len(by_id) != len(recovered) or not excluded_ids <= by_id.keys():
        raise ValueError('prior-inspection commit is absent or source IDs are duplicated')
    excluded_sessions = {by_id[oid]['session_sha256'] for oid in excluded_ids}
    if None in excluded_sessions:
        raise ValueError('cannot isolate an inspected record with no session identity')
    excluded_rows = [row for row in recovered if row['session_sha256'] in excluded_sessions]
    forbidden = {corpus_audit.text_key(row['text']) for row in excluded_rows}

    if not reference_plans:
        raise ValueError('at least one reference plan is required')
    taxonomy_raw = taxonomy.read_bytes()
    review_corpus.evaluate.load_taxonomy(taxonomy)
    reference_reports = []
    reference_keys = set()
    for reference_plan in reference_plans:
        reference_report, _ = corpus_audit.audit(reference_plan)
        reference_reports.append(reference_report)
        reference_raw = reference_plan.read_bytes()
        if hashlib.sha256(reference_raw).hexdigest() != reference_report['plan_sha256']:
            raise ValueError('reference plan changed')
        reference = json.loads(reference_raw)
        if any(source['project'] == project or source.get('project_group', source['project']) == project_group
               for source in reference['sources']):
            raise ValueError('candidate project and group must be absent from the reference plan')
        if hashlib.sha256(taxonomy_raw).hexdigest() != reference_report['taxonomy_sha256']:
            raise ValueError('reference and candidate taxonomies differ')
        for source, evidence in zip(reference['sources'], reference_report['sources'], strict=True):
            rows, digest = train.load_dataset(corpus_audit.resolve(reference_plan.parent, source['path']),
                                              allow_unlabeled_review=True)
            if digest != evidence['sha256']:
                raise ValueError('reference source changed')
            reference_keys.update(corpus_audit.text_key(row['text']) for row in rows)

    candidates = []
    seen = set()
    skipped_excluded_session = skipped_related_text = skipped_reference = skipped_duplicate = 0
    for row in recovered:
        if row['session_sha256'] in excluded_sessions:
            skipped_excluded_session += 1
            continue
        key = corpus_audit.text_key(row['text'])
        if key in forbidden:
            skipped_related_text += 1
            continue
        if key in reference_keys:
            skipped_reference += 1
            continue
        if key in seen:
            skipped_duplicate += 1
            continue
        seen.add(key)
        session = row['session_sha256'] or 'unknown-session'
        source = {**row['source'], 'text_sha256': row['text_sha256'],
                  'source_inventory_sha256': hashlib.sha256(inventory_raw).hexdigest(),
                  'history_stream_sha256': hashlib.sha256(raw_history).hexdigest(),
                  'extractor_sha256': hashlib.sha256(Path(inventory.__file__).read_bytes()).hexdigest()}
        candidates.append({'id': 'checkpoint-' + row['id'], 'text': row['text'],
                           'family': 'checkpoint-session-' + session,
                           'intents': [], 'capabilities': [], 'risks': [], 'labels_pending': True,
                           'verified': False, 'source': source,
                           'review_note': 'No labels proposed. Check attribution, context, rights, privacy, and every label.'})
    if not candidates:
        raise ValueError('no candidate records remain')
    if (source_inventory.read_bytes() != inventory_raw or exclusions.read_bytes() != exclusion_raw
            or taxonomy.read_bytes() != taxonomy_raw
            or any(corpus_audit.audit(path)[0] != report for path, report in zip(reference_plans, reference_reports, strict=True))):
        raise ValueError('inputs changed during preparation')
    output.mkdir(mode=0o700)
    review_corpus.new_file(output / 'taxonomy.json', taxonomy_raw)
    review_corpus.new_file(output / 'source-0000.jsonl', review_corpus.encoded_rows(candidates))
    plan = {'schema_version': corpus_audit.PLAN_SCHEMA, 'taxonomy': 'taxonomy.json', 'review_ledger': None,
            'sources': [{'path': 'source-0000.jsonl', 'project': project,
                         'project_group': project_group, 'split': 'test'}]}
    review_corpus.new_file(output / 'plan.json', (json.dumps(plan, indent=2) + '\n').encode())
    workspace = review_corpus.init_workspace(output / 'plan.json', output / 'workspace')
    summary = {'recovered': len(recovered), 'excluded_inspected_sessions': skipped_excluded_session,
               'excluded_text_from_inspected_sessions': skipped_related_text,
               'excluded_reference_overlap': skipped_reference, 'excluded_normalized_duplicates': skipped_duplicate,
               'pending_review': workspace['examples'], 'approved': 0,
               'assignment': 'provisional-test', 'promotion_eligible': False}
    manifest = {'schema_version': 'lfg.checkpoint-review-preparation.v1', 'project': project,
                'project_group': project_group, 'revision': revision, 'max_history_bytes': max_history_bytes,
                'history_stream_sha256': hashlib.sha256(raw_history).hexdigest(),
                'source_inventory_sha256': hashlib.sha256(inventory_raw).hexdigest(),
                'prior_inspections_sha256': hashlib.sha256(exclusion_raw).hexdigest(),
                'reference_audit_sha256': [hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':'),
                                                                     allow_nan=False).encode()).hexdigest()
                                           for report in reference_reports],
                'preparer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'extractor_sha256': hashlib.sha256(Path(inventory.__file__).read_bytes()).hexdigest(),
                'workspace_sha256': {name: hashlib.sha256((output / 'workspace' / name).read_bytes()).hexdigest()
                                     for name in ('plan.json', 'taxonomy.json', 'source-0000.jsonl', 'reviews.jsonl')},
                'summary': summary}
    review_corpus.new_file(output / 'preparation-manifest.json',
                           (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'source-inventory', 'exclusions', 'taxonomy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reference-plan', type=Path, action='append', required=True)
    for name in ('revision', 'project', 'project-group'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-history-bytes', type=int, required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.repo, args.revision, args.project, args.project_group, args.source_inventory,
                         args.exclusions, args.reference_plan, args.taxonomy, args.output, args.max_history_bytes)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        raise SystemExit(f'checkpoint review preparation rejected: {error}') from error


if __name__ == '__main__':
    main()
