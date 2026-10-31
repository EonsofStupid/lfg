#!/usr/bin/env python3
"""Combine independently reviewed workspaces into one audited private corpus."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path

import corpus_audit
import review_corpus
import train


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def report_digest(value: dict) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def workspace_paths(paths: list[Path]) -> list[Path]:
    if not paths:
        raise ValueError('at least one workspace is required')
    result = [path.resolve(strict=True) for path in paths]
    if len({os.path.normcase(str(path)) for path in result}) != len(result):
        raise ValueError('workspaces must be distinct')
    if any(not path.is_dir() for path in result):
        raise ValueError('workspace must be a directory')
    return result


def inspect(workspaces: list[Path]) -> tuple[list[dict], dict]:
    reports = []
    counts = {split: {'examples': 0, 'ready': 0} for split in corpus_audit.SPLITS}
    decisions = {'pending': 0, 'approved': 0, 'rejected': 0}
    taxonomy = None
    for workspace in workspaces:
        _, report, ledger = review_corpus.load_workspace(workspace)
        if taxonomy is not None and report['taxonomy_sha256'] != taxonomy:
            raise ValueError('workspace taxonomies differ')
        taxonomy = report['taxonomy_sha256']
        reports.append(report)
        for split in corpus_audit.SPLITS:
            counts[split]['examples'] += report['counts'][split].get('examples', 0)
            counts[split]['ready'] += report['counts'][split].get('ready_for_reviewed_subset', 0)
        for item in ledger.values():
            decisions[item['decision']] += 1
    return reports, {'counts': counts, 'decisions': decisions, 'promotion_eligible': False}


def run(paths: list[Path], output: Path | None, policy_path: Path = train.POLICY,
        scope: str = 'complete') -> dict:
    if scope not in ('complete', 'development'):
        raise ValueError('invalid assembly scope')
    workspaces = workspace_paths(paths)
    policy_raw = policy_path.read_bytes()
    minimum = train.parse_promotion_policy(policy_raw)['minimum_human_verified_examples']
    if output is not None:
        output = output.resolve()
        parent = output.parent.resolve(strict=True)
        if not parent.is_dir() or any(output == path or output.is_relative_to(path) for path in workspaces):
            raise ValueError('output must be outside the review workspaces')
        if output.exists():
            raise FileExistsError('output already exists')
    with contextlib.ExitStack() as locks:
        for path in sorted(workspaces, key=lambda path: os.path.normcase(str(path))):
            locks.enter_context(train.run_lock(path / 'review'))
        if scope == 'development':
            # Read all plans before opening any source or review ledger.
            for workspace in workspaces:
                plan = json.loads(review_corpus.private_path(workspace, 'plan.json').read_bytes())
                sources = plan.get('sources') if isinstance(plan, dict) else None
                if (not isinstance(sources, list) or not sources
                        or any(not isinstance(source, dict) or source.get('split') not in ('train', 'validation')
                               for source in sources)):
                    raise ValueError('development assembly permits only train and validation sources')
        initial, summary = inspect(workspaces)
        if scope == 'development':
            summary.update(scope='development', holdout_assessment='not_performed')
        summary['sample_gate'] = {'policy_sha256': digest(policy_raw), 'minimum_train_examples': minimum,
                                  'approved_train_shortfall': max(0, minimum - summary['counts']['train']['ready']),
                                  'candidate_train_shortfall': max(0, minimum - summary['counts']['train']['examples'])}
        if output is None:
            return summary
        required = ('train', 'validation') if scope == 'development' else corpus_audit.SPLITS
        if any(summary['counts'][split]['ready'] == 0 for split in required):
            raise ValueError('each split needs at least one current approval before combining')
        with tempfile.TemporaryDirectory(prefix='.lfg-combine-', dir=parent) as temp:
            stage = Path(temp).resolve(strict=True)
            if not stage.is_relative_to(parent) or stage == parent:
                raise ValueError('temporary directory escaped output parent')
            exports = []
            for index, workspace in enumerate(workspaces):
                # A supplied workspace with no current approvals is an error, not a silent omission.
                export = stage / f'export-{index:04}'
                review_corpus.export_approved(workspace, export)
                exports.append(export)
            combined = stage / 'combined'
            combined.mkdir(mode=0o700)
            taxonomy = (exports[0] / 'taxonomy.json').read_bytes()
            review_corpus.new_file(combined / 'taxonomy.json', taxonomy)
            sources, ledger = [], []
            for export in exports:
                plan = json.loads((export / 'plan.json').read_bytes())
                if (export / 'taxonomy.json').read_bytes() != taxonomy:
                    raise ValueError('exported taxonomies differ')
                items, _ = corpus_audit.read_ledger(export / 'reviews.jsonl')
                ledger.extend(items.values())
                for source in plan['sources']:
                    name = f'source-{len(sources):04}.jsonl'
                    review_corpus.new_file(combined / name, (export / source['path']).read_bytes())
                    sources.append({**source, 'path': name})
            review_corpus.new_file(combined / 'reviews.jsonl', review_corpus.encoded_rows(ledger))
            plan = {'schema_version': (corpus_audit.DEVELOPMENT_PLAN_SCHEMA if scope == 'development'
                                       else corpus_audit.PLAN_SCHEMA), 'taxonomy': 'taxonomy.json',
                    'review_ledger': 'reviews.jsonl', 'sources': sources}
            review_corpus.new_file(combined / 'plan.json', (json.dumps(plan, indent=2) + '\n').encode())
            training, report = corpus_audit.assemble_training(combined / 'plan.json')
            review_corpus.new_file(combined / 'training.jsonl', training)
            if inspect(workspaces)[0] != initial or policy_path.read_bytes() != policy_raw:
                raise ValueError('review workspaces changed during assembly')
            code = [Path(__file__), Path(review_corpus.assessment_targets.__file__), Path(review_corpus.__file__), Path(corpus_audit.__file__),
                    Path(train.__file__), Path(review_corpus.evaluate.__file__)]
            files = ['plan.json', 'taxonomy.json', 'reviews.jsonl', 'training.jsonl'] + [source['path'] for source in sources]
            manifest = {'schema_version': 'lfg.combined-reviewed-corpus.v1',
                        'input_audit_sha256': [report_digest(value) for value in initial],
                        'code_sha256': {path.name: digest(path.read_bytes()) for path in code},
                        'files_sha256': {name: digest((combined / name).read_bytes()) for name in files},
                        'counts': report['counts'], 'sample_gate': summary['sample_gate'], 'promotion_eligible': False}
            if scope == 'development':
                manifest.update(scope='development', holdout_assessment='not_performed')
            review_corpus.new_file(combined / 'assembly-manifest.json',
                                   (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
            output.mkdir(mode=0o700)
            for name in [*files, 'assembly-manifest.json']:
                review_corpus.new_file(output / name, (combined / name).read_bytes())
            checked_training, checked = corpus_audit.assemble_training(output / 'plan.json')
            if (checked != report or checked_training != training or inspect(workspaces)[0] != initial
                    or policy_path.read_bytes() != policy_raw
                    or any(digest((output / name).read_bytes()) != expected
                           for name, expected in manifest['files_sha256'].items())
                    or any(digest(path.read_bytes()) != manifest['code_sha256'][path.name] for path in code)):
                raise ValueError('inputs or combined output changed during assembly')
            result = {'counts': checked['counts'], 'sample_gate': summary['sample_gate'], 'promotion_eligible': False}
            if scope == 'development':
                result.update(scope='development', holdout_assessment='not_performed')
            return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, action='append', required=True)
    parser.add_argument('--promotion-policy', type=Path, default=train.POLICY,
                        help='sample gate policy selected for the prospective training run')
    parser.add_argument('--scope', choices=('complete', 'development'), default='complete')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.workspace, args.output, args.promotion_policy, args.scope), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        raise SystemExit(f'combined review rejected: {error}') from error


if __name__ == '__main__':
    main()
