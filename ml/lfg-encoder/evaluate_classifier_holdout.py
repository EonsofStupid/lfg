#!/usr/bin/env python3
"""Freeze a reviewed classifier before opening a separately committed holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import corpus_audit
import evaluate_reviewed_native as native
import train


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def commitment(plan_path: Path) -> dict:
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    if (not isinstance(plan, dict) or plan.get('schema_version') != corpus_audit.PLAN_SCHEMA
            or not isinstance(plan.get('sources'), list) or not plan['sources']
            or any(not isinstance(source, dict) or source.get('split') != 'test' for source in plan['sources'])):
        raise ValueError('holdout plan must contain only test sources')
    report, _ = corpus_audit.audit(plan_path)
    counts = report['counts']['test']
    if (not report['review_ledger_sha256'] or not counts.get('examples')
            or counts.get('ready_for_reviewed_subset', 0) != counts['examples']
            or report['duplicate_within_split']):
        raise ValueError('holdout requires unique, currently reviewed test records')
    if sha(plan_path.read_bytes()) != report['plan_sha256'] or corpus_audit.audit(plan_path)[0] != report:
        raise ValueError('holdout changed during commitment')
    return {'schema_version': 'lfg.classifier-holdout-commitment.v1',
            'plan_sha256': report['plan_sha256'], 'taxonomy_sha256': report['taxonomy_sha256'],
            'ledger_sha256': report['review_ledger_sha256'], 'sources': report['sources'],
            'records': counts['examples']}


def load_bound(path: Path, expected: str, schema: str) -> dict:
    raw = path.read_bytes()
    if not re.fullmatch(r'[0-9a-f]{64}', expected) or sha(raw) != expected:
        raise ValueError('bound file digest mismatch')
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get('schema_version') != schema or path.read_bytes() != raw:
        raise ValueError('invalid or changed bound file')
    return value


def development_loader(plan: Path, dataset: Path, manifest: dict) -> tuple[dict, list[dict], dict]:
    report, validation, taxonomy, _ = native.audited_run(plan, dataset, manifest, development=True)
    return report, validation, taxonomy


def freeze(args: argparse.Namespace) -> dict:
    committed = load_bound(args.commitment, args.commitment_sha256, 'lfg.classifier-holdout-commitment.v1')
    if (not isinstance(committed.get('taxonomy_sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', committed['taxonomy_sha256'])
            or type(committed.get('records')) is not int or committed['records'] < 1):
        raise ValueError('invalid holdout commitment')
    code_sha = native.file_hash(Path(__file__))
    validation = native.compare(args.plan, args.training_dataset, args.model, args.artifact, args.binary,
                                args.runtime_config, args.config, args.mode, args.calibration_config,
                                loader=development_loader)
    if validation['audit']['taxonomy_sha256'] != committed['taxonomy_sha256']:
        raise ValueError('development taxonomy differs from committed holdout')
    if (native.file_hash(Path(__file__)) != code_sha
            or load_bound(args.commitment, args.commitment_sha256, 'lfg.classifier-holdout-commitment.v1') != committed):
        raise ValueError('freeze inputs changed')
    return {'schema_version': 'lfg.classifier-freeze.v1', 'commitment_sha256': args.commitment_sha256,
            'validation': validation, 'evaluator_sha256': code_sha,
            'holdout_inputs_accepted': False, 'qualification_established': False}


def combined_loader(holdout_plan: Path, committed: dict):
    def load(plan: Path, dataset: Path, manifest: dict) -> tuple[dict, list[dict], dict]:
        development, _, taxonomy, _ = native.audited_run(plan, dataset, manifest, development=True)
        if commitment(holdout_plan) != committed:
            raise ValueError('holdout differs from commitment')
        dev_raw, test_raw = plan.read_bytes(), holdout_plan.read_bytes()
        dev, test = json.loads(dev_raw), json.loads(test_raw)
        if sha(dev_raw) != development['plan_sha256'] or sha(test_raw) != committed['plan_sha256']:
            raise ValueError('plan changed before separation check')
        def paths(value: dict, root: Path) -> set[Path]:
            return {corpus_audit.resolve(root, source['path']) for source in value['sources']} | {
                corpus_audit.resolve(root, value['review_ledger'])}
        dev_paths, test_paths = paths(dev, plan.parent), paths(test, holdout_plan.parent)
        if dev_paths & test_paths or len(dev_paths) != len(dev['sources']) + 1 or len(test_paths) != len(test['sources']) + 1:
            raise ValueError('development and holdout sources and ledgers must be distinct')
        dev_taxonomy = corpus_audit.resolve(plan.parent, dev['taxonomy']).read_bytes()
        test_taxonomy = corpus_audit.resolve(holdout_plan.parent, test['taxonomy']).read_bytes()
        if sha(dev_taxonomy) != development['taxonomy_sha256'] or dev_taxonomy != test_taxonomy:
            raise ValueError('holdout taxonomy differs from development')
        with tempfile.TemporaryDirectory(prefix='lfg-holdout-audit-') as directory:
            root = Path(directory)
            (root / 'taxonomy.json').write_bytes(dev_taxonomy)
            dev_ledger = corpus_audit.resolve(plan.parent, dev['review_ledger']).read_bytes()
            test_ledger = corpus_audit.resolve(holdout_plan.parent, test['review_ledger']).read_bytes()
            (root / 'reviews.jsonl').write_bytes(dev_ledger.rstrip(b'\r\n') + b'\n' + test_ledger)
            sources = [{**source, 'path': str(corpus_audit.resolve(base, source['path']))}
                       for value, base in ((dev, plan.parent), (test, holdout_plan.parent))
                       for source in value['sources']]
            merged = {'schema_version': corpus_audit.PLAN_SCHEMA, 'taxonomy': 'taxonomy.json',
                      'review_ledger': 'reviews.jsonl', 'sources': sources}
            merged_path = root / 'plan.json'
            merged_path.write_bytes(encode(merged))
            report, _ = corpus_audit.audit(merged_path)
            if report['duplicate_within_split']:
                raise ValueError('combined corpus has duplicate normalized text')
            for split in corpus_audit.SPLITS:
                count = report['counts'][split]
                if not count.get('examples') or count.get('ready_for_reviewed_subset', 0) != count['examples']:
                    raise ValueError('combined corpus contains unreviewed records')
            rows = []
            for source, evidence in zip(sources, report['sources'], strict=True):
                if source['split'] == 'test':
                    items, digest = train.load_dataset(Path(source['path']))
                    if digest != evidence['sha256']:
                        raise ValueError('holdout source changed')
                    rows.extend(items)
            if corpus_audit.audit(merged_path)[0] != report:
                raise ValueError('combined corpus changed during separation check')
        if (plan.read_bytes() != dev_raw or holdout_plan.read_bytes() != test_raw
                or commitment(holdout_plan) != committed
                or native.audited_run(plan, dataset, manifest, development=True)[0] != development):
            raise ValueError('corpus changed during separation check')
        return report, rows, taxonomy
    return load


def final(args: argparse.Namespace, frozen: dict) -> dict:
    if freeze(args) != frozen:
        raise ValueError('candidate differs from frozen development state')
    committed = load_bound(args.commitment, args.commitment_sha256, 'lfg.classifier-holdout-commitment.v1')
    result = native.compare(args.plan, args.training_dataset, args.model, args.artifact, args.binary,
                            args.runtime_config, args.config, args.mode, args.calibration_config,
                            loader=combined_loader(args.holdout_plan, committed))
    if freeze(args) != frozen or commitment(args.holdout_plan) != committed:
        raise ValueError('candidate or holdout changed during evaluation')
    return {'schema_version': 'lfg.classifier-holdout-report.v1', 'freeze_sha256': args.freeze_sha256,
            'commitment_sha256': args.commitment_sha256, 'test': result,
            'candidate_authenticated': False, 'qualification_established': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    commit = sub.add_parser('commit-holdout')
    commit.add_argument('--holdout-plan', type=Path, required=True)
    commit.add_argument('--output', type=Path, required=True)
    for name in ('freeze', 'test'):
        command = sub.add_parser(name)
        for field in ('plan', 'training-dataset', 'model', 'artifact', 'binary', 'runtime-config', 'config', 'commitment', 'output'):
            command.add_argument('--' + field, type=Path, required=True)
        command.add_argument('--commitment-sha256', required=True)
        command.add_argument('--mode', choices=('embedded', 'external'), required=True)
        command.add_argument('--calibration-config', type=Path)
        if name == 'test':
            command.add_argument('--holdout-plan', type=Path, required=True)
            command.add_argument('--freeze', type=Path, required=True)
            command.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output already exists')
    if args.command == 'commit-holdout':
        result = commitment(args.holdout_plan)
    elif args.command == 'freeze':
        result = freeze(args)
    else:
        frozen = load_bound(args.freeze, args.freeze_sha256, 'lfg.classifier-freeze.v1')
        result = final(args, frozen)
        if load_bound(args.freeze, args.freeze_sha256, 'lfg.classifier-freeze.v1') != frozen:
            raise ValueError('freeze changed during evaluation')
    raw = encode(result) + b'\n'
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as output:
        output.write(raw)
    print(json.dumps({'output_sha256': sha(raw), 'qualification_established': False}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        print(f'classifier holdout rejected: {error}', file=sys.stderr)
        sys.exit(1)
