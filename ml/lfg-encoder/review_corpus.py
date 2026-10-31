#!/usr/bin/env python3
"""Review a private corpus locally; approvals are explicit human attestations."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import corpus_audit as audit
import evaluate
import train


def encoded_rows(rows: list[dict[str, Any]]) -> bytes:
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
                   for row in rows).encode('utf-8')


def new_file(path: Path, content: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(content)


def atomic_replace(path: Path, content: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix='.review-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def private_path(root: Path, value: Any) -> Path:
    path = audit.resolve(root, value)
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError('workspace paths must stay inside the workspace')
    return path


def init_workspace(plan_path: Path, workspace: Path) -> dict[str, Any]:
    report, queue = audit.audit(plan_path)
    raw = plan_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != report['plan_sha256']:
        raise ValueError('plan changed during initialization')
    plan = json.loads(raw)
    if plan.get('review_ledger') is not None:
        raise ValueError('initialize from a plan without an existing review ledger')
    taxonomy_path = audit.resolve(plan_path.parent, plan['taxonomy'])
    taxonomy = taxonomy_path.read_bytes()
    if hashlib.sha256(taxonomy).hexdigest() != report['taxonomy_sha256']:
        raise ValueError('taxonomy changed during initialization')
    copies = []
    for source, evidence in zip(plan['sources'], report['sources'], strict=True):
        source_path = audit.resolve(plan_path.parent, source['path'])
        rows, digest = train.load_dataset(source_path)
        raw_source = source_path.read_bytes()
        if digest != evidence['sha256'] or hashlib.sha256(raw_source).hexdigest() != digest:
            raise ValueError('source changed during initialization')
        if any(row['verified'] for row in rows):
            raise ValueError('initialize only from unverified source records')
        copies.append(raw_source)
    if audit.audit(plan_path)[0] != report:
        raise ValueError('inputs changed during initialization')
    workspace.mkdir(mode=0o700)
    new_file(workspace / 'taxonomy.json', taxonomy)
    for i, (source, raw_source) in enumerate(zip(plan['sources'], copies, strict=True)):
        name = f'source-{i:04}.jsonl'
        new_file(workspace / name, raw_source)
        source['path'] = name
    new_file(workspace / 'reviews.jsonl', encoded_rows(queue))
    plan['taxonomy'] = 'taxonomy.json'
    plan['review_ledger'] = 'reviews.jsonl'
    new_file(workspace / 'plan.json', (json.dumps(plan, indent=2) + '\n').encode())
    created, _ = audit.audit(workspace / 'plan.json')
    return {'examples': len(queue), 'approved': 0, 'counts': created['counts']}


def load_workspace(workspace: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    workspace = workspace.resolve(strict=True)
    plan_path = private_path(workspace, 'plan.json')
    raw_plan = plan_path.read_bytes()
    plan = json.loads(raw_plan)
    if plan.get('review_ledger') != 'reviews.jsonl' or plan.get('taxonomy') != 'taxonomy.json':
        raise ValueError('invalid workspace plan paths')
    private_path(workspace, plan['review_ledger'])
    private_path(workspace, plan['taxonomy'])
    for source in plan['sources']:
        private_path(workspace, source['path'])
    report, queue = audit.audit(plan_path)
    if hashlib.sha256(raw_plan).hexdigest() != report['plan_sha256']:
        raise ValueError('workspace plan changed while loading')
    ledger, ledger_hash = audit.read_ledger(private_path(workspace, plan['review_ledger']))
    if ledger_hash != report['review_ledger_sha256']:
        raise ValueError('workspace ledger changed while loading')
    if set(ledger) != {item['id'] for item in queue}:
        raise ValueError('workspace ledger must cover exactly the source records')
    return plan, report, ledger


def visible(value: str, multiline: bool = False) -> str:
    """Prevent control characters in source text from driving the terminal."""
    return ''.join(c if (multiline and c in '\n\t') or not unicodedata.category(c).startswith('C')
                   else f'\\u{ord(c):04x}' for c in value)


def status(workspace: Path) -> dict[str, Any]:
    _, report, ledger = load_workspace(workspace)
    return {'counts': report['counts'], 'decisions': {key: sum(item['decision'] == key for item in ledger.values())
                                                       for key in ('pending', 'approved', 'rejected')},
            'promotion_eligible': False}


def find_row(workspace: Path, plan: dict[str, Any], identifier: str) -> tuple[Path, list[dict[str, Any]], int, dict[str, Any]]:
    for source in plan['sources']:
        path = private_path(workspace, source['path'])
        rows, _ = train.load_dataset(path)
        for index, row in enumerate(rows):
            if row['id'] == identifier:
                return path, rows, index, source
    raise ValueError('review record not found')


def record_decision(workspace: Path, identifier: str, reviewer: str, decision: str,
                    expected_digest: str, proposed: dict[str, list[str]] | None = None) -> None:
    if not isinstance(reviewer, str) or not reviewer.strip() or '\n' in reviewer or '\r' in reviewer:
        raise ValueError('reviewer must be a nonempty single-line identifier')
    if decision not in ('approved', 'rejected'):
        raise ValueError('decision must be approved or rejected')
    plan, report, ledger = load_workspace(workspace)
    item = ledger.get(identifier)
    if item is None or item['decision'] != 'pending':
        raise ValueError('only pending records can be reviewed')
    source_path, rows, index, source = find_row(workspace, plan, identifier)
    row = rows[index]
    taxonomy, taxonomy_hash = evaluate.load_taxonomy(private_path(workspace, plan['taxonomy']))
    current_digest = audit.annotation_digest(row, source['project'], source['split'], taxonomy_hash)
    if current_digest != expected_digest:
        raise ValueError('record changed after display; review it again')
    if not row['verified'] and item['annotation_sha256'] != current_digest:
        raise ValueError('pending record changed; reinitialize review')
    changed = dict(row)
    if proposed is not None:
        if decision != 'approved' or set(proposed) != set(evaluate.FIELDS):
            raise ValueError('label edits require an approved record and all label groups')
        for field in evaluate.FIELDS:
            values = proposed[field]
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values) or \
                    len(values) != len(set(values)) or set(values) - taxonomy[field]:
                raise ValueError(f'invalid {field} labels')
            changed[field] = sorted(values)
        if not changed['intents']:
            raise ValueError('at least one intent is required')
    if audit.audit(workspace / 'plan.json')[0] != report:
        raise ValueError('workspace changed during review')
    if decision == 'approved' or row['verified']:
        changed['verified'] = decision == 'approved'
        rows[index] = changed
        # A crash before the ledger update leaves no matching approval.
        atomic_replace(source_path, encoded_rows(rows))
    item['annotation_sha256'] = audit.annotation_digest(changed, source['project'], source['split'], taxonomy_hash)
    item.update(decision=decision, reviewer=reviewer.strip(), reviewed_at=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_replace(private_path(workspace, plan['review_ledger']), encoded_rows(list(ledger.values())))


def interactive_review(workspace: Path, reviewer: str) -> None:
    plan, _, ledger = load_workspace(workspace)
    taxonomy, taxonomy_hash = evaluate.load_taxonomy(private_path(workspace, plan['taxonomy']))
    for identifier, item in ledger.items():
        if item['decision'] != 'pending':
            continue
        _, rows, index, source = find_row(workspace, plan, identifier)
        row = rows[index]
        displayed_digest = audit.annotation_digest(row, source['project'], source['split'], taxonomy_hash)
        if row['verified']:
            print('This pending record has an incomplete earlier write. Review its current contents again.')
        proposed = {field: list(row[field]) for field in evaluate.FIELDS}
        while True:
            print(f'\nID: {visible(identifier)}  project: {visible(source["project"])}  split: {source["split"]}')
            print('SOURCE:', json.dumps(row['source'], ensure_ascii=True, sort_keys=True))
            print('REQUEST:\n' + visible(row['text'], multiline=True))
            if 'review_note' in row:
                print('REVIEW NOTE:', json.dumps(row['review_note'], ensure_ascii=True))
            print('PROPOSED LABELS:', json.dumps(proposed, ensure_ascii=True, sort_keys=True))
            print('Absent labels are negatives. Check attribution, context, permissions and sensitive data before approving.')
            action = input('[a]pprove, [e]dit labels, [r]eject, [s]kip, [q]uit: ').strip().lower()
            if action == 'e':
                for field in evaluate.FIELDS:
                    print(f'{field}: ' + ', '.join(sorted(taxonomy[field])))
                    value = input(f'{field} labels (comma-separated; blank means none): ')
                    proposed[field] = [part.strip() for part in value.split(',') if part.strip()]
                continue
            if action == 'a':
                confirm = input('Type APPROVE to attest you reviewed the text, source and all labels: ').strip()
                if confirm == 'APPROVE':
                    record_decision(workspace, identifier, reviewer, 'approved', displayed_digest, proposed)
                    break
            elif action == 'r':
                confirm = input('Type REJECT to exclude this record from approved exports: ').strip()
                if confirm == 'REJECT':
                    record_decision(workspace, identifier, reviewer, 'rejected', displayed_digest)
                    break
            elif action == 's':
                break
            elif action == 'q':
                return


def export_approved(workspace: Path, output: Path) -> dict[str, Any]:
    plan, report, ledger = load_workspace(workspace)
    taxonomy_bytes = private_path(workspace, plan['taxonomy']).read_bytes()
    _, taxonomy_hash = evaluate.load_taxonomy(private_path(workspace, plan['taxonomy']))
    if hashlib.sha256(taxonomy_bytes).hexdigest() != taxonomy_hash or taxonomy_hash != report['taxonomy_sha256']:
        raise ValueError('taxonomy changed during export')
    selected = []
    approved_items = []
    for source in plan['sources']:
        rows, _ = train.load_dataset(private_path(workspace, source['path']))
        keep = []
        for row in rows:
            digest = audit.annotation_digest(row, source['project'], source['split'], taxonomy_hash)
            item = ledger[row['id']]
            if row['verified'] and audit.approved(item, digest):
                keep.append(row)
                approved_items.append(item)
        if keep:
            selected.append((source, keep))
    if not approved_items:
        raise ValueError('no current approvals to export')
    if audit.audit(workspace / 'plan.json')[0] != report:
        raise ValueError('workspace changed during export')
    output.mkdir(mode=0o700)
    new_file(output / 'taxonomy.json', taxonomy_bytes)
    sources = []
    for i, (source, rows) in enumerate(selected):
        name = f'source-{i:04}.jsonl'
        new_file(output / name, encoded_rows(rows))
        sources.append({**source, 'path': name})
    new_file(output / 'reviews.jsonl', encoded_rows(approved_items))
    exported_plan = {**plan, 'sources': sources, 'taxonomy': 'taxonomy.json', 'review_ledger': 'reviews.jsonl'}
    new_file(output / 'plan.json', (json.dumps(exported_plan, indent=2) + '\n').encode())
    checked, _ = audit.audit(output / 'plan.json')
    return {'exported': len(approved_items), 'counts': checked['counts'], 'promotion_eligible': False}


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--plan', required=True, type=Path)
    init.add_argument('--workspace', required=True, type=Path)
    for command in ('status', 'review', 'export'):
        child = sub.add_parser(command)
        child.add_argument('--workspace', required=True, type=Path)
        if command == 'review':
            child.add_argument('--reviewer', required=True)
        if command == 'export':
            child.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'init':
            result = init_workspace(args.plan, args.workspace)
        else:
            with train.run_lock(args.workspace / 'review'):
                if args.command == 'review':
                    interactive_review(args.workspace, args.reviewer)
                    result = status(args.workspace)
                elif args.command == 'export':
                    result = export_approved(args.workspace, args.output)
                else:
                    result = status(args.workspace)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, EOFError) as error:
        raise SystemExit(f'review rejected: {error}') from error


if __name__ == '__main__':
    main()
