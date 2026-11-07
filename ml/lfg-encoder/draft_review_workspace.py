#!/usr/bin/env python3
"""Create a pending-only training review workspace with explicit AI label drafts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import corpus_audit
import evaluate
import review_corpus
import review_web
import train


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def prepare(workspace: Path, proposals: Path, guide: Path, output: Path) -> dict:
    workspace = workspace.resolve(strict=True)
    output = output.resolve()
    if output.exists() or output.is_relative_to(workspace) or not output.parent.is_dir():
        raise ValueError('select a new output directory outside the input workspace')
    raw_drafts = proposals.read_bytes()
    drafts = json.loads(raw_drafts)
    fields = {'schema_version', 'taxonomy_sha256', 'guide_sha256', 'author', 'items'}
    if (not isinstance(drafts, dict) or set(drafts) != fields
            or drafts['schema_version'] != 'lfg.ai-label-drafts.v1'
            or not isinstance(drafts['author'], str) or not drafts['author'].strip()
            or not isinstance(drafts['items'], list) or not drafts['items']):
        raise ValueError('invalid AI label drafts')
    with train.run_lock(workspace / 'review'):
        # Do not open record sources or ledgers from test/validation workspaces.
        plan_raw = review_corpus.private_path(workspace, 'plan.json').read_bytes()
        candidate = json.loads(plan_raw)
        if (not isinstance(candidate, dict) or not isinstance(candidate.get('sources'), list)
                or not candidate['sources'] or any(not isinstance(source, dict) or source.get('split') != 'train'
                                                  for source in candidate['sources'])):
            raise ValueError('AI drafts may only be prepared for training sources')
        plan, report, ledger = review_corpus.load_workspace(workspace)
        if digest(plan_raw) != report['plan_sha256']:
            raise ValueError('workspace plan changed after split check')
        if any(item['decision'] != 'pending' for item in ledger.values()):
            raise ValueError('input workspace must contain only pending decisions')
        taxonomy_path = review_corpus.private_path(workspace, plan['taxonomy'])
        taxonomy, taxonomy_sha = evaluate.load_taxonomy(taxonomy_path)
        _, guide_sha = review_web.load_guide(guide, taxonomy)
        if drafts['taxonomy_sha256'] != taxonomy_sha or drafts['guide_sha256'] != guide_sha:
            raise ValueError('draft taxonomy or guide changed')
        items = {}
        for item in drafts['items']:
            if (not isinstance(item, dict)
                    or set(item) != {'id', 'annotation_sha256', 'labels', 'note'}
                    or not isinstance(item['id'], str) or item['id'] in items
                    or item['id'] not in ledger or item['annotation_sha256'] != ledger[item['id']]['annotation_sha256']
                    or not isinstance(item['note'], str) or not item['note'].strip()
                    or not isinstance(item['labels'], dict) or set(item['labels']) != set(evaluate.FIELDS)):
                raise ValueError('invalid, stale or duplicate draft item')
            for field in evaluate.FIELDS:
                values = item['labels'][field]
                if (not isinstance(values, list) or any(not isinstance(v, str) for v in values)
                        or len(values) != len(set(values)) or set(values) - taxonomy[field]):
                    raise ValueError('draft label is invalid for the taxonomy')
            if not item['labels']['intents']:
                raise ValueError('draft requires an intent; omit uncertain examples')
            items[item['id']] = item
        transformed = []
        matched = set()
        for source in plan['sources']:
            rows, _ = train.load_dataset(review_corpus.private_path(workspace, source['path']), allow_unlabeled_review=True)
            for row in rows:
                if row['verified']:
                    raise ValueError('input records must be unverified')
                item = items.get(row['id'])
                if item is None:
                    continue
                if (not row.get('labels_pending') or corpus_audit.annotation_digest(
                        row, source['project'], source['split'], taxonomy_sha) != item['annotation_sha256']):
                    raise ValueError('AI drafts only apply to unlabeled records')
                for field in evaluate.FIELDS:
                    row[field] = sorted(item['labels'][field])
                row.pop('labels_pending')
                previous_note = row.get('review_note')
                if previous_note is not None and not isinstance(previous_note, str):
                    raise ValueError('invalid existing review note')
                row['review_note'] = ('AI-drafted labels, not reviewed. Check every label, implied negative, '
                                      'source, context, privacy and rights. ' + item['note']
                                      + (' Previous note: ' + previous_note if previous_note else ''))
                row['ai_label_draft'] = {'schema_version': 'lfg.ai-label-draft.v1', 'author': drafts['author'],
                                         'original_annotation_sha256': item['annotation_sha256'],
                                         'guide_sha256': guide_sha, 'proposal_sha256': digest(raw_drafts)}
                matched.add(row['id'])
            transformed.append((source, rows))
        if matched != set(items) or review_corpus.load_workspace(workspace)[1] != report:
            raise ValueError('draft items or workspace changed')
        if proposals.read_bytes() != raw_drafts or review_web.load_guide(guide, taxonomy)[1] != guide_sha:
            raise ValueError('draft inputs changed')
        output.mkdir(mode=0o700)
        review_corpus.new_file(output / 'taxonomy.json', taxonomy_path.read_bytes())
        sources = []
        for index, (source, rows) in enumerate(transformed):
            name = f'source-{index:04}.jsonl'
            review_corpus.new_file(output / name, review_corpus.encoded_rows(rows))
            sources.append({**source, 'path': name})
        new_plan = {**plan, 'taxonomy': 'taxonomy.json', 'review_ledger': None, 'sources': sources}
        review_corpus.new_file(output / 'plan.json', (json.dumps(new_plan, indent=2) + '\n').encode())
        result = review_corpus.init_workspace(output / 'plan.json', output / 'workspace')
        if (review_corpus.load_workspace(workspace)[1] != report or proposals.read_bytes() != raw_drafts
                or review_web.load_guide(guide, taxonomy)[1] != guide_sha):
            raise ValueError('draft inputs changed during preparation')
        manifest = {'schema_version': 'lfg.ai-draft-workspace.v1', 'input_plan_sha256': report['plan_sha256'],
                    'input_ledger_sha256': report['review_ledger_sha256'], 'taxonomy_sha256': taxonomy_sha,
                    'guide_sha256': guide_sha, 'proposal_sha256': digest(raw_drafts),
                    'preparer_sha256': digest(Path(__file__).read_bytes()), 'drafts': len(matched),
                    'pending': result['examples'], 'approved': 0, 'promotion_eligible': False}
        review_corpus.new_file(output / 'draft-manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'proposals', 'guide', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.workspace, args.proposals, args.guide, args.output), indent=2))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        raise SystemExit(f'AI draft preparation rejected: {error}') from error


if __name__ == '__main__':
    main()
