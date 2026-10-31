#!/usr/bin/env python3
"""Human-attested, snapshot-bound review and admission for kernel choice observations."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata

SPLITS = ('train', 'validation', 'test')
QUEUE_FIELDS = {'schemaVersion', 'status', 'requestId', 'text', 'inputSha256',
                'sourceFileSha256', 'snapshotSha256', 'snapshotRevision', 'journalTailSha256',
                'observationSequence', 'node', 'options', 'baselineState', 'baselineChoice',
                'reviewedChoice', 'reviewer', 'project', 'family', 'split'}
HEX = re.compile(r'^[0-9a-f]{64}$')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def identity(item: dict) -> str:
    return digest(encode(item))


def nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not any(c in value for c in '\r\n')


def load_queue(queue: Path, snapshot: Path) -> tuple[list[dict], str, str]:
    snapshot_raw = snapshot.read_bytes()
    spec = json.loads(snapshot_raw)
    snapshot_sha = digest(snapshot_raw)
    if not isinstance(spec, dict) or spec.get('schemaVersion') != 'lfg.kernel-snapshot.v1' or not isinstance(spec.get('nodes'), dict):
        raise ValueError('invalid kernel snapshot')
    raw = queue.read_bytes()
    if not raw.endswith(b'\n'):
        raise ValueError('queue must have complete newline-terminated records')
    rows, seen, seen_nodes = [], set(), set()
    for line in raw.splitlines():
        item = json.loads(line)
        if not isinstance(item, dict) or set(item) != QUEUE_FIELDS or item['schemaVersion'] != 'lfg.node-review-queue-item.v1' or item['status'] != 'pending_review':
            raise ValueError('invalid pending queue record')
        if any(item[key] is not None for key in ('reviewedChoice', 'reviewer', 'project', 'family', 'split')):
            raise ValueError('queue must not contain review decisions or assignments')
        if not nonempty(item['requestId']) or not isinstance(item['text'], str) or not item['text'].strip() or digest(item['text'].encode('utf-8')) != item['inputSha256']:
            raise ValueError('invalid request identity or text digest')
        if any(not isinstance(item[key], str) or not HEX.fullmatch(item[key]) for key in ('inputSha256', 'sourceFileSha256', 'snapshotSha256', 'journalTailSha256')):
            raise ValueError('invalid source or journal digest')
        if item['snapshotSha256'] != snapshot_sha or item['snapshotRevision'] != spec.get('revision'):
            raise ValueError('queue does not match snapshot bytes and revision')
        if type(item['observationSequence']) is not int or item['observationSequence'] < 1 or not nonempty(item['node']):
            raise ValueError('invalid observation reference')
        node = spec['nodes'].get(item['node'])
        if not isinstance(node, dict) or node.get('kind') != 'choice' or not isinstance(node.get('options'), dict) or not node['options']:
            raise ValueError('queue node is not a snapshot choice')
        options = sorted(node['options'])
        if item['options'] != options or (item['baselineChoice'] is not None and item['baselineChoice'] not in options) or not nonempty(item['baselineState']):
            raise ValueError('queue options or baseline do not match snapshot')
        key = (item['journalTailSha256'], item['observationSequence'], item['node'])
        item_id = identity(item)
        if item_id in seen or key in seen_nodes:
            raise ValueError('duplicate queue item or observation node')
        seen.add(item_id)
        seen_nodes.add(key)
        rows.append(item)
    if not rows:
        raise ValueError('queue is empty')
    if snapshot.read_bytes() != snapshot_raw or queue.read_bytes() != raw:
        raise ValueError('queue or snapshot changed while reading')
    return rows, digest(raw), snapshot_sha


def create_new(path: Path, content: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def init(queue: Path, snapshot: Path, ledger: Path) -> dict:
    items, queue_sha, snapshot_sha = load_queue(queue, snapshot)
    result = {'schemaVersion': 'lfg.node-review-ledger.v1', 'queueSha256': queue_sha,
              'snapshotSha256': snapshot_sha,
              'records': [{'itemSha256': identity(item), 'decision': 'pending'} for item in items]}
    create_new(ledger, encode(result) + b'\n')
    return {'items': len(items), 'pending': len(items), 'approved': 0, 'queueSha256': queue_sha}


def load_ledger(queue: Path, snapshot: Path, ledger: Path) -> tuple[list[dict], dict, bytes]:
    items, queue_sha, snapshot_sha = load_queue(queue, snapshot)
    raw = ledger.read_bytes()
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) != {'schemaVersion', 'queueSha256', 'snapshotSha256', 'records'}
            or value['schemaVersion'] != 'lfg.node-review-ledger.v1' or value['queueSha256'] != queue_sha
            or value['snapshotSha256'] != snapshot_sha or not isinstance(value['records'], list)):
        raise ValueError('ledger is invalid or queue/snapshot changed')
    by_id = {identity(item): item for item in items}
    seen = set()
    for row in value['records']:
        if not isinstance(row, dict) or row.get('itemSha256') not in by_id or row['itemSha256'] in seen:
            raise ValueError('ledger contains unknown or duplicate item')
        seen.add(row['itemSha256'])
        if row.get('decision') == 'pending':
            if set(row) != {'itemSha256', 'decision'}:
                raise ValueError('pending records cannot contain decisions')
        elif row.get('decision') == 'approved':
            fields = {'itemSha256', 'decision', 'reviewer', 'reviewedAt', 'choice', 'project', 'projectGroup', 'family', 'split', 'attested'}
            if (set(row) != fields or row['attested'] is not True or row['choice'] not in by_id[row['itemSha256']]['options']
                    or row['split'] not in SPLITS or not all(nonempty(row[k]) for k in ('reviewer', 'project', 'projectGroup', 'family'))):
                raise ValueError('invalid approved review')
        elif row.get('decision') == 'rejected':
            if (set(row) != {'itemSha256', 'decision', 'reviewer', 'reviewedAt', 'attested'}
                    or row['attested'] is not True or not nonempty(row['reviewer'])):
                raise ValueError('invalid rejected review')
        else:
            raise ValueError('unknown review decision')
        if row['decision'] != 'pending':
            try:
                timestamp = dt.datetime.fromisoformat(row['reviewedAt'])
                if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                    raise ValueError('timestamp requires timezone')
            except (ValueError, TypeError) as error:
                raise ValueError('invalid review timestamp') from error
    if seen != set(by_id):
        raise ValueError('ledger must cover the queue exactly')
    if ledger.read_bytes() != raw:
        raise ValueError('ledger changed while reading')
    return items, value, raw


def decide(args: argparse.Namespace) -> dict:
    if not args.attest:
        raise ValueError('explicit --attest is required')
    if not nonempty(args.reviewer):
        raise ValueError('reviewer is required')
    if args.decision == 'approved' and (not all(nonempty(getattr(args, key)) for key in ('project', 'project_group', 'family', 'choice')) or args.split not in SPLITS):
        raise ValueError('approval requires choice, project, project group, family and split')
    if args.decision == 'rejected' and any(getattr(args, key) is not None for key in ('project', 'project_group', 'family', 'choice', 'split')):
        raise ValueError('rejection must not assign a choice or split')
    lock = args.ledger.with_name(args.ledger.name + '.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.close(fd)
        items, ledger, original = load_ledger(args.queue, args.snapshot, args.ledger)
        item = next((value for value in items if identity(value) == args.item_sha256), None)
        if item is None or (args.decision == 'approved' and args.choice not in item['options']):
            raise ValueError('unknown item or invalid choice')
        row = next(value for value in ledger['records'] if value['itemSha256'] == args.item_sha256)
        if row['decision'] != 'pending':
            raise ValueError('only pending items may be reviewed')
        row.update(decision=args.decision, reviewer=args.reviewer, reviewedAt=dt.datetime.now(dt.timezone.utc).isoformat(), attested=True)
        if args.decision == 'approved':
            row.update(choice=args.choice, project=args.project, projectGroup=args.project_group, family=args.family, split=args.split)
        if args.ledger.read_bytes() != original:
            raise ValueError('ledger changed during review')
        load_queue(args.queue, args.snapshot)
        if digest(args.queue.read_bytes()) != ledger['queueSha256'] or digest(args.snapshot.read_bytes()) != ledger['snapshotSha256']:
            raise ValueError('queue or snapshot changed during review')
        descriptor, name = tempfile.mkstemp(prefix='.node-review-', dir=args.ledger.parent)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(encode(ledger) + b'\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, args.ledger)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return {'decision': args.decision, 'itemSha256': args.item_sha256}
    finally:
        lock.unlink(missing_ok=True)


def normalized_text(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def load_policy(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    policy = json.loads(raw)
    if (not isinstance(policy, dict) or set(policy) != {'schemaVersion', 'minimumPerChoice', 'minimumProjectGroups'}
            or policy['schemaVersion'] != 'lfg.node-admission-policy.v1'):
        raise ValueError('invalid node admission policy')
    for key in ('minimumPerChoice', 'minimumProjectGroups'):
        if not isinstance(policy[key], dict) or set(policy[key]) != set(SPLITS) or any(type(policy[key][s]) is not int or policy[key][s] < 1 for s in SPLITS):
            raise ValueError('policy requires positive thresholds for each split')
    return policy, digest(raw)


def audit(inputs: list[list[Path]], snapshot: Path, policy_path: Path, node: str) -> tuple[dict, list[dict]]:
    policy, policy_sha = load_policy(policy_path)
    snapshot_raw = snapshot.read_bytes()
    spec = json.loads(snapshot_raw)
    node_spec = spec.get('nodes', {}).get(node, {})
    if node_spec.get('kind') != 'choice' or not isinstance(node_spec.get('options'), dict) or not node_spec['options']:
        raise ValueError('target node must be a snapshot choice')
    choices = sorted(node_spec['options'])
    counts = {s: {choice: 0 for choice in choices} for s in SPLITS}
    groups = {s: set() for s in SPLITS}
    relations = {key: {} for key in ('project', 'projectGroup', 'family', 'text', 'request')}
    project_groups, identities = {}, set()
    conflicts = set()
    approved, pending, rejected, records, sources = 0, 0, 0, [], []
    seen_split_text = {s: set() for s in SPLITS}
    for queue, ledger_path in inputs:
        items, ledger, raw_ledger = load_ledger(queue, snapshot, ledger_path)
        by_id = {identity(item): item for item in items}
        sources.append({'queueSha256': ledger['queueSha256'], 'ledgerSha256': digest(raw_ledger)})
        for decision in ledger['records']:
            item = by_id[decision['itemSha256']]
            if decision['itemSha256'] in identities:
                raise ValueError('duplicate item across queues')
            identities.add(decision['itemSha256'])
            if decision['decision'] == 'pending':
                pending += int(item['node'] == node)
                continue
            if decision['decision'] == 'rejected':
                rejected += int(item['node'] == node)
                continue
            split = decision['split']
            project_groups.setdefault(decision['project'], set()).add(decision['projectGroup'])
            keys = {'project': decision['project'], 'projectGroup': decision['projectGroup'],
                    'family': decision['family'], 'text': digest(normalized_text(item['text']).encode('utf-8')),
                    'request': item['requestId']}
            for kind, key in keys.items():
                relations[kind].setdefault(key, set()).add(split)
            if item['node'] != node:
                continue
            approved += 1
            counts[split][decision['choice']] += 1
            groups[split].add(decision['projectGroup'])
            if keys['text'] in seen_split_text[split]:
                conflicts.add('withinSplitDuplicateText')
            seen_split_text[split].add(keys['text'])
            records.append({'item': item, 'review': decision})
    for kind, mapping in relations.items():
        if any(len(splits) > 1 for splits in mapping.values()):
            conflicts.add(kind)
    if any(len(value) > 1 for value in project_groups.values()):
        conflicts.add('projectGroupAssignment')
    support_ok = all(counts[s][choice] >= policy['minimumPerChoice'][s] for s in SPLITS for choice in choices)
    groups_ok = all(len(groups[s]) >= policy['minimumProjectGroups'][s] for s in SPLITS)
    ready = approved > 0 and pending == 0 and not conflicts and support_ok and groups_ok
    report = {'schemaVersion': 'lfg.node-admission-report.v1', 'node': node,
              'snapshotSha256': digest(snapshot_raw), 'policySha256': policy_sha,
              'toolSha256': digest(Path(__file__).read_bytes()), 'pythonVersion': sys.version,
              'unicodeVersion': unicodedata.unidata_version, 'sources': sources,
              'pending': pending, 'approved': approved, 'rejected': rejected, 'counts': counts,
              'projectGroups': {s: len(groups[s]) for s in SPLITS}, 'conflicts': sorted(conflicts),
              'supportGatePassed': support_ok, 'projectGroupGatePassed': groups_ok,
              'admissionEligible': ready, 'qualificationEstablished': False}
    if snapshot.read_bytes() != snapshot_raw or digest(policy_path.read_bytes()) != policy_sha:
        raise ValueError('snapshot or policy changed during audit')
    for (queue, ledger_path), source in zip(inputs, sources, strict=True):
        if digest(queue.read_bytes()) != source['queueSha256'] or digest(ledger_path.read_bytes()) != source['ledgerSha256']:
            raise ValueError('queue or ledger changed during audit')
    return report, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('init', 'show', 'decide'):
        command = sub.add_parser(name)
        command.add_argument('--queue', type=Path, required=True)
        command.add_argument('--snapshot', type=Path, required=True)
        command.add_argument('--ledger', type=Path, required=True)
        if name in ('show', 'decide'):
            command.add_argument('--item-sha256', required=True)
        if name == 'show':
            command.add_argument('--include-text', action='store_true')
            command.add_argument('--include-baseline', action='store_true')
        if name == 'decide':
            command.add_argument('--decision', choices=('approved', 'rejected'), required=True)
            for flag in ('reviewer', 'choice', 'project', 'project-group', 'family'):
                command.add_argument('--' + flag, required=(flag == 'reviewer'))
            command.add_argument('--split', choices=SPLITS)
            command.add_argument('--attest', action='store_true')
    for name in ('audit', 'export'):
        command = sub.add_parser(name)
        command.add_argument('--input', type=Path, nargs=2, action='append', required=True, metavar=('QUEUE', 'LEDGER'))
        command.add_argument('--snapshot', type=Path, required=True)
        command.add_argument('--policy', type=Path, required=True)
        command.add_argument('--node', required=True)
        if name == 'export':
            command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('list')
    command.add_argument('--queue', type=Path, required=True)
    command.add_argument('--snapshot', type=Path, required=True)
    command.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'init':
        result = init(args.queue, args.snapshot, args.ledger)
    elif args.command == 'decide':
        result = decide(args)
    elif args.command in ('show', 'list'):
        items, ledger, _ = load_ledger(args.queue, args.snapshot, args.ledger)
        decisions = {row['itemSha256']: row['decision'] for row in ledger['records']}
        if args.command == 'list':
            result = [{'itemSha256': identity(item), 'node': item['node'], 'decision': decisions[identity(item)]} for item in items]
        else:
            item = next((item for item in items if identity(item) == args.item_sha256), None)
            if item is None:
                raise ValueError('unknown item')
            result = dict(item)
            if not args.include_text:
                result['text'] = None
            if not args.include_baseline:
                result.pop('baselineChoice')
                result.pop('baselineState')
            result['itemSha256'] = identity(item)
            result['decision'] = decisions[identity(item)]
    else:
        result, records = audit(args.input, args.snapshot, args.policy, args.node)
        if args.command == 'export':
            if not result['admissionEligible']:
                raise ValueError('node admission gates are not satisfied')
            data = {'schemaVersion': 'lfg.node-training-input.v1', 'audit': result,
                    'train': [], 'validation': []}
            for record in records:
                split = record['review']['split']
                if split in ('train', 'validation'):
                    data[split].append({'itemSha256': record['review']['itemSha256'],
                                        'text': record['item']['text'], 'choice': record['review']['choice'],
                                        'project': record['review']['project'],
                                        'projectGroup': record['review']['projectGroup'],
                                        'family': record['review']['family']})
            raw = encode(data) + b'\n'
            create_new(args.output, raw)
            result = {'schemaVersion': 'lfg.node-training-export.v1', 'sha256': digest(raw),
                      'train': len(data['train']), 'validation': len(data['validation']),
                      'qualificationEstablished': False}
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f'node review: {error}', file=sys.stderr)
        sys.exit(1)
