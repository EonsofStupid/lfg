#!/usr/bin/env python3
"""Review a private corpus locally; approvals are explicit human attestations."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import threading
import unicodedata
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import corpus_audit as audit
import assessment_targets
import evaluate
import train

PARTIAL_REVIEW_SCHEMA = 'lfg.partial-label-review.v1'
PARTIAL_REVIEW_FILE = 'partial-reviews.v1.jsonl'
PARTIAL_JUDGMENTS = {'positive', 'negative', 'unknown'}
PARTIAL_PROPOSAL_ACTIONS = {'accepted', 'rejected'}
SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_PARTIAL_LOCKS: dict[str, threading.RLock] = {}
_PARTIAL_LOCKS_GUARD = threading.Lock()


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
        rows, digest = train.load_dataset(source_path, allow_unlabeled_review=True)
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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _partial_operation_lock(workspace: Path) -> threading.RLock:
    key = str(workspace.resolve(strict=True))
    with _PARTIAL_LOCKS_GUARD:
        return _PARTIAL_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def partial_review_write_lock(workspace: Path):
    with _partial_operation_lock(workspace):
        yield


@contextmanager
def review_mutation_lock(workspace: Path):
    """Serialize review writes across threads and processes, reusing an active CLI/server lock."""
    with partial_review_write_lock(workspace):
        root = workspace.resolve(strict=True)
        marker = root / 'review.lock'
        if marker.exists() or marker.is_symlink():
            try:
                info = marker.lstat()
                if marker.is_symlink() or not stat.S_ISREG(info.st_mode) or \
                        marker.read_text(encoding='utf-8') != f'pid={os.getpid()}\n':
                    raise ValueError('another process holds the workspace review lock')
            except OSError as error:
                raise ValueError('cannot verify the workspace review lock') from error
            yield
        else:
            # Direct library callers acquire the same interprocess marker used by
            # CLI and server entry points. A competing writer fails closed.
            with train.run_lock(root / 'review'):
                yield


def _partial_event_sha256(event: dict[str, Any]) -> str:
    return _sha256(json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                              allow_nan=False).encode('utf-8'))


def _ensure_private_sidecar_location(workspace: Path, sidecar: Path) -> None:
    try:
        root_result = subprocess.run(['git', '-C', str(workspace), 'rev-parse', '--show-toplevel'],
                                     capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError('cannot verify private partial review storage location') from error
    if root_result.returncode != 0:
        if 'not a git repository' in root_result.stderr.casefold():
            return
        raise ValueError('cannot verify private partial review storage location')
    repo_root = Path(root_result.stdout.strip()).resolve()
    relative = sidecar.resolve().relative_to(repo_root).as_posix()
    ignored = subprocess.run(['git', '-C', str(repo_root), 'check-ignore', '--quiet', '--no-index', '--', relative],
                             capture_output=True, timeout=5)
    tracked = subprocess.run(['git', '-C', str(repo_root), 'ls-files', '--error-unmatch', '--', relative],
                             capture_output=True, timeout=5)
    if ignored.returncode != 0 or tracked.returncode == 0:
        raise ValueError('partial review sidecar must be ignored by Git and untracked')


def _partial_binding(workspace: Path, identifier: str,
                     allow_verified_recovery: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, report, ledger = load_workspace(workspace)
    item = ledger.get(identifier)
    if item is None or item['decision'] != 'pending':
        raise ValueError('partial review is limited to pending records')
    _, rows, index, source = find_row(workspace, plan, identifier)
    row = rows[index]
    if row.get('verified') is True and not allow_verified_recovery:
        raise ValueError('partial review cannot alter a verified record')
    taxonomy, taxonomy_sha = evaluate.load_taxonomy(private_path(workspace, plan['taxonomy']))
    annotation_sha = audit.annotation_digest(row, source['project'], source['split'], taxonomy_sha)
    if item['annotation_sha256'] != annotation_sha and not (allow_verified_recovery and row.get('verified') is True):
        raise ValueError('pending record changed; reinitialize review')
    source_path = private_path(workspace, source['path'])
    source_bytes = source_path.read_bytes()
    if _sha256(source_bytes) != report['sources'][plan['sources'].index(source)]['sha256']:
        raise ValueError('source changed during partial review')
    proposal = {field: row.get(field, []) for field in evaluate.FIELDS}
    binding = {
        'source_sha256': _sha256(source_bytes),
        'annotation_sha256': annotation_sha,
        'taxonomy_sha256': taxonomy_sha,
        'text_sha256': _sha256(row['text'].encode('utf-8')),
        'proposal_sha256': _sha256(json.dumps(proposal, ensure_ascii=False, sort_keys=True,
                                              separators=(',', ':')).encode('utf-8')),
    }
    return {'plan': plan, 'report': report, 'ledger': ledger, 'row': row, 'source': source,
            'taxonomy': taxonomy, 'binding': binding}, binding


def _read_partial_events(path: Path, max_bytes: int) -> list[dict[str, Any]]:
    if path.is_symlink():
        raise ValueError('partial review sidecar cannot be a symlink')
    if not path.exists():
        return []
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size > max_bytes:
        raise ValueError('partial review sidecar is unsafe or exceeds its configured limit')
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError('partial review sidecar is truncated or changed')
    return _parse_partial_events(raw, max_bytes)


def _parse_partial_events(raw: bytes, max_bytes: int) -> list[dict[str, Any]]:
    if not isinstance(raw, bytes) or len(raw) > max_bytes or (raw and not raw.endswith(b'\n')):
        raise ValueError('partial review sidecar is truncated or exceeds its configured limit')
    events = []
    seen_event_ids = set()
    previous_by_id: dict[str, dict[str, Any]] = {}
    for line in raw.splitlines():
        value = json.loads(line)
        base_fields = {'schema_version', 'event_id', 'id', 'reviewer_claim', 'reviewed_at', 'binding',
                       'judgments', 'proposal_actions', 'context_checks', 'attestation'}
        revision_fields = {'revision', 'event_type', 'supersedes_event_id', 'supersedes_event_sha256'}
        if not isinstance(value, dict) or value.get('schema_version') != PARTIAL_REVIEW_SCHEMA or \
                set(value) not in (base_fields, base_fields | revision_fields):
            raise ValueError('unsupported partial review event')
        if set(value) == base_fields:
            # Read the original v1 event format as revision one.
            value = {**value, 'revision': 1, 'event_type': 'decision',
                     'supersedes_event_id': None, 'supersedes_event_sha256': None}
        identifier = value.get('id')
        event_id = value.get('event_id')
        if not isinstance(identifier, str) or not identifier or \
                not isinstance(event_id, str) or not re.fullmatch(r'[0-9a-f]{32}', event_id) or event_id in seen_event_ids:
            raise ValueError('duplicate or malformed partial review event')
        prior = previous_by_id.get(identifier)
        expected_revision = prior['revision'] + 1 if prior else 1
        expected_parent_id = prior['event_id'] if prior else None
        expected_parent_digest = _partial_event_sha256(prior) if prior else None
        if type(value.get('revision')) is not int or value['revision'] != expected_revision or \
                value.get('supersedes_event_id') != expected_parent_id or \
                value.get('supersedes_event_sha256') != expected_parent_digest or \
                value.get('event_type') not in ('decision', 'retraction'):
            raise ValueError('partial review revision chain is invalid')
        reviewer = value.get('reviewer_claim')
        timestamp = value.get('reviewed_at')
        binding = value.get('binding')
        expected_binding = {'source_sha256', 'annotation_sha256', 'taxonomy_sha256', 'text_sha256',
                            'proposal_sha256', 'guide_sha256'}
        event_attestation = ('Reviewer explicitly retracted the prior partial review.'
                             if value['event_type'] == 'retraction' else
                             'Reviewer explicitly recorded partial label judgments; unknown cells remain unknown.'
                             if value['revision'] == 1 else
                             'Reviewer explicitly revised partial label judgments; unknown cells remain unknown.')
        if not isinstance(reviewer, str) or not reviewer.strip() or '\n' in reviewer or '\r' in reviewer or \
                not isinstance(timestamp, str) or not isinstance(binding, dict) or set(binding) != expected_binding or \
                any(not isinstance(binding[key], str) or not SHA256_RE.fullmatch(binding[key]) for key in expected_binding - {'guide_sha256'}) or \
                (binding['guide_sha256'] is not None and (not isinstance(binding['guide_sha256'], str) or
                                                          not SHA256_RE.fullmatch(binding['guide_sha256']))) or \
                value.get('attestation') != event_attestation:
            raise ValueError('malformed partial review provenance')
        try:
            parsed = dt.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError as error:
            raise ValueError('malformed partial review timestamp') from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError('partial review timestamp requires a timezone')
        for key, allowed in (('judgments', PARTIAL_JUDGMENTS), ('proposal_actions', PARTIAL_PROPOSAL_ACTIONS)):
            values = value.get(key)
            if not isinstance(values, dict) or set(values) - set(evaluate.FIELDS) or any(
                    not isinstance(labels, dict) or any(not isinstance(label, str) or not isinstance(state, str) or state not in allowed
                                                        for label, state in labels.items())
                    for labels in values.values()):
                raise ValueError('malformed partial review label data')
        has_content = any(value.get('judgments', {}).values()) or any(value.get('proposal_actions', {}).values())
        if (value['event_type'] == 'decision' and not has_content) or \
                (value['event_type'] == 'retraction' and has_content):
            raise ValueError('partial review event content does not match its operation')
        contexts = value.get('context_checks')
        if not isinstance(contexts, list) or any(not isinstance(entry, dict) or
                set(entry) != {'kind', 'status', 'count', 'digest'} or
                not isinstance(entry['kind'], str) or not isinstance(entry['status'], str) or
                type(entry['count']) is not int or entry['count'] < 0 or
                not isinstance(entry['digest'], str) or not SHA256_RE.fullmatch(entry['digest'])
                for entry in contexts):
            raise ValueError('malformed partial review context provenance')
        seen_event_ids.add(event_id)
        previous_by_id[identifier] = value
        events.append(value)
    return events


def _secure_existing_partial_file(path: Path) -> None:
    if path.is_symlink():
        raise ValueError('partial review sidecar cannot be a symlink')
    if path.exists():
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('partial review sidecar is not a regular file')
        _secure_partial_file(path)


def partial_review_state(workspace: Path, identifier: str, max_bytes: int,
                         guide_sha256: str | None = None) -> dict[str, Any]:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('partial review requires a positive configured byte limit')
    # Read-only recovery can inspect partial history for a pending ledger row
    # whose source was already marked verified before an interrupted write.
    current, binding = _partial_binding(workspace, identifier, allow_verified_recovery=True)
    sidecar = workspace.resolve(strict=True) / PARTIAL_REVIEW_FILE
    _ensure_private_sidecar_location(workspace.resolve(strict=True), sidecar)
    _secure_existing_partial_file(sidecar)
    events = _read_partial_events(sidecar, max_bytes)
    history = [item for item in events if item['id'] == identifier]
    event = history[-1] if history else None
    if event is not None:
        same_binding = event.get('binding') == {**binding, 'guide_sha256': guide_sha256}
        if same_binding:
            for field, labels in event['judgments'].items():
                if set(labels) - current['taxonomy'][field]:
                    raise ValueError('partial review contains a label outside the current taxonomy')
            for field, labels in event['proposal_actions'].items():
                if set(labels) - set(current['row'].get(field, [])):
                    raise ValueError('partial review proposal no longer matches the source record')
    expected = {**binding, 'guide_sha256': guide_sha256}
    if event is None:
        return {'enabled': True, 'state': 'not_reviewed', 'binding': expected, 'revision': None,
                'event': None, 'history': []}
    event_digest = _partial_event_sha256(event)
    state = 'retracted' if event['event_type'] == 'retraction' else ('current' if same_binding else 'stale')
    return {'enabled': True, 'state': state, 'binding': expected,
            'revision': {'event_id': event['event_id'], 'event_sha256': event_digest,
                         'revision': event['revision'], 'event_type': event['event_type']},
            'event': event, 'history': history}


def _secure_partial_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
        if os.name == 'nt':
            # Replace inherited ACLs with an explicit current-user-only ACL.
            identity = subprocess.run(['whoami', '/user', '/fo', 'csv', '/nh'], capture_output=True,
                                      text=True, check=True, timeout=5).stdout.strip().split(',')[-1].strip('"')
            principal = subprocess.run(['whoami'], capture_output=True, text=True,
                                       check=True, timeout=5).stdout.strip()
            if not re.fullmatch(r'S-1-[0-9-]+', identity) or not principal:
                raise ValueError('cannot establish owner-only partial review permissions')
            subprocess.run(['icacls', str(path), '/inheritance:r', '/grant:r', f'*{identity}:(F)'], capture_output=True,
                           text=True, check=True, timeout=5)
            acl = subprocess.run(['icacls', str(path)], capture_output=True, text=True,
                                 check=True, timeout=5).stdout
            principals = set()
            for line in acl.splitlines():
                line = line.strip()
                if line.startswith(str(path)):
                    line = line[len(str(path)):].strip()
                if '(' not in line:
                    continue
                ace_principal = line.split('(', 1)[0].strip().removesuffix(':')
                principals.add(ace_principal)
                if '(I)' in line or ace_principal.casefold() != principal.casefold():
                    raise ValueError('partial review sidecar ACL is not owner-private')
            if {value.casefold() for value in principals} != {principal.casefold()}:
                raise ValueError('partial review sidecar ACL is not owner-private')
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError('cannot establish owner-only partial review permissions') from error


def record_partial_decision(workspace: Path, identifier: str, reviewer: str, expected_binding: dict[str, Any],
                             judgments: dict[str, dict[str, str]], proposal_actions: dict[str, dict[str, str]],
                             guide_sha256: str | None, context_checks: list[dict[str, Any]],
                             max_bytes: int, attestation: str, operation: str = 'create',
                             supersedes_event_id: str | None = None,
                             supersedes_event_sha256: str | None = None,
                             guide_path: Path | None = None) -> None:
    with review_mutation_lock(workspace):
        _record_partial_decision(workspace, identifier, reviewer, expected_binding, judgments, proposal_actions,
                                 guide_sha256, context_checks, max_bytes, attestation, operation,
                                 supersedes_event_id, supersedes_event_sha256, guide_path)


def _record_partial_decision(workspace: Path, identifier: str, reviewer: str, expected_binding: dict[str, Any],
                             judgments: dict[str, dict[str, str]], proposal_actions: dict[str, dict[str, str]],
                             guide_sha256: str | None, context_checks: list[dict[str, Any]],
                             max_bytes: int, attestation: str, operation: str,
                             supersedes_event_id: str | None, supersedes_event_sha256: str | None,
                             guide_path: Path | None) -> None:
    if not isinstance(reviewer, str) or not reviewer.strip() or '\n' in reviewer or '\r' in reviewer:
        raise ValueError('reviewer must be a nonempty single-line identifier')
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('partial review requires a positive configured byte limit')
    expected_attestation = {'create': 'REVIEW PARTIAL LABELS', 'revise': 'REVISE PARTIAL REVIEW',
                            'retract': 'RETRACT PARTIAL REVIEW'}.get(operation)
    if expected_attestation is None or not isinstance(attestation, str) or attestation != expected_attestation:
        raise ValueError('explicit partial review operation attestation required')
    lock = workspace.resolve(strict=True) / 'review.lock'
    if lock.is_symlink() or not lock.is_file() or lock.read_text(encoding='utf-8') != f'pid={os.getpid()}\n':
        raise ValueError('partial review requires the active workspace review lock')
    current, binding = _partial_binding(workspace, identifier)
    if not isinstance(expected_binding, dict) or set(expected_binding) != set(binding) | {'guide_sha256'} or \
            {key: expected_binding[key] for key in binding} != binding or \
            expected_binding.get('guide_sha256') != guide_sha256:
        raise ValueError('record, source, taxonomy, guide or proposal changed; review again')
    if guide_sha256 is not None and not (isinstance(guide_sha256, str) and SHA256_RE.fullmatch(guide_sha256)):
        raise ValueError('invalid label guide digest')
    taxonomy = current['taxonomy']
    if not isinstance(judgments, dict) or set(judgments) - set(evaluate.FIELDS):
        raise ValueError('invalid partial label judgments')
    for field, values in judgments.items():
        if not isinstance(values, dict) or set(values) - taxonomy[field] or any(state not in PARTIAL_JUDGMENTS for state in values.values()):
            raise ValueError(f'invalid {field} judgment')
    if not isinstance(proposal_actions, dict) or set(proposal_actions) - set(evaluate.FIELDS):
        raise ValueError('invalid proposal actions')
    row = current['row']
    for field, values in proposal_actions.items():
        if not isinstance(values, dict) or set(values) - set(row.get(field, [])) or \
                any(action not in PARTIAL_PROPOSAL_ACTIONS for action in values.values()):
            raise ValueError(f'invalid {field} proposal action')
    if operation != 'retract' and not any(judgments.values()) and not any(proposal_actions.values()):
        raise ValueError('partial review must contain an explicit judgment or proposal action')
    if operation == 'retract' and (any(judgments.values()) or any(proposal_actions.values())):
        raise ValueError('retraction cannot contain active labels or proposal actions')
    safe_context = []
    if not isinstance(context_checks, list):
        raise ValueError('invalid context provenance')
    for entry in context_checks:
        if not isinstance(entry, dict) or set(entry) != {'kind', 'status', 'count', 'digest'} or \
                not isinstance(entry['kind'], str) or not isinstance(entry['status'], str) or \
                type(entry['count']) is not int or entry['count'] < 0 or \
                not isinstance(entry['digest'], str) or not SHA256_RE.fullmatch(entry['digest']):
            raise ValueError('invalid context provenance')
        safe_context.append(entry)
    sidecar = workspace.resolve(strict=True) / PARTIAL_REVIEW_FILE
    _ensure_private_sidecar_location(workspace.resolve(strict=True), sidecar)
    _secure_existing_partial_file(sidecar)
    existing = _read_partial_events(sidecar, max_bytes)
    history = [event for event in existing if event['id'] == identifier]
    previous = history[-1] if history else None
    previous_digest = _partial_event_sha256(previous) if previous else None
    if operation == 'create':
        if previous is not None or supersedes_event_id is not None or supersedes_event_sha256 is not None:
            raise ValueError('partial review already exists; revise or retract the current revision')
    elif previous is None or supersedes_event_id != previous['event_id'] or supersedes_event_sha256 != previous_digest:
        raise ValueError('partial review revision is stale or conflicts with a newer event')
    event_type = 'retraction' if operation == 'retract' else 'decision'
    event_attestation = ('Reviewer explicitly retracted the prior partial review.' if operation == 'retract' else
                         'Reviewer explicitly revised partial label judgments; unknown cells remain unknown.' if operation == 'revise' else
                         'Reviewer explicitly recorded partial label judgments; unknown cells remain unknown.')
    event = {'schema_version': PARTIAL_REVIEW_SCHEMA, 'event_id': uuid.uuid4().hex, 'id': identifier,
             'reviewer_claim': reviewer.strip(), 'reviewed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
             'binding': {**binding, 'guide_sha256': guide_sha256}, 'judgments': judgments,
             'proposal_actions': proposal_actions, 'context_checks': safe_context, 'revision': previous['revision'] + 1 if previous else 1,
             'event_type': event_type, 'supersedes_event_id': previous['event_id'] if previous else None,
             'supersedes_event_sha256': previous_digest, 'attestation': event_attestation}
    line = encoded_rows([event])
    old = sidecar.read_bytes() if sidecar.exists() else b''
    if len(old) + len(line) > max_bytes:
        raise ValueError('partial review sidecar exceeds its configured byte limit')
    fd, name = tempfile.mkstemp(prefix='.partial-review-', dir=sidecar.parent)
    try:
        os.close(fd)
        temp_path = Path(name)
        _secure_partial_file(temp_path)
        _ensure_private_sidecar_location(workspace.resolve(strict=True), temp_path)
        flags = os.O_WRONLY | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)
        with os.fdopen(os.open(temp_path, flags), 'wb') as stream:
            stream.write(old + line)
            stream.flush()
            os.fsync(stream.fileno())
        if sidecar.exists() and (sidecar.is_symlink() or not sidecar.is_file()):
            raise ValueError('partial review sidecar path is unsafe')
        latest, _ = _partial_binding(workspace, identifier)
        if latest['binding'] != binding or audit.audit(workspace.resolve(strict=True) / 'plan.json')[0] != current['report']:
            raise ValueError('workspace changed during partial review; review again')
        if guide_path is not None and _sha256(guide_path.read_bytes()) != guide_sha256:
            raise ValueError('label guide changed during partial review; review again')
        latest_events = _read_partial_events(sidecar, max_bytes)
        latest_history = [item for item in latest_events if item['id'] == identifier]
        latest_prior = latest_history[-1] if latest_history else None
        if ((latest_prior['event_id'] if latest_prior else None) != (previous['event_id'] if previous else None) or
                (_partial_event_sha256(latest_prior) if latest_prior else None) != previous_digest):
            raise ValueError('partial review revision is stale or conflicts with a concurrent event')
        os.replace(temp_path, sidecar)
        _secure_partial_file(sidecar)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def find_row(workspace: Path, plan: dict[str, Any], identifier: str) -> tuple[Path, list[dict[str, Any]], int, dict[str, Any]]:
    for source in plan['sources']:
        path = private_path(workspace, source['path'])
        rows, _ = train.load_dataset(path, allow_unlabeled_review=True)
        for index, row in enumerate(rows):
            if row['id'] == identifier:
                return path, rows, index, source
    raise ValueError('review record not found')


def apply_redactions(text: str, operations: list[dict[str, Any]]) -> str:
    """Apply reviewer-supplied replacements to ranges in the evolving text."""
    if not isinstance(operations, list):
        raise ValueError('redactions must be a list')
    for operation in operations:
        if (not isinstance(operation, dict) or set(operation) != {'start', 'end', 'replacement'}
                or type(operation['start']) is not int or type(operation['end']) is not int
                or not isinstance(operation['replacement'], str)
                or not 0 <= operation['start'] < operation['end'] <= len(text)):
            raise ValueError('invalid redaction range')
        start, end = operation['start'], operation['end']
        if text[start:end] == operation['replacement']:
            raise ValueError('redaction must change the selected text')
        text = text[:start] + operation['replacement'] + text[end:]
    if not text.strip():
        raise ValueError('redacted request must not be empty')
    return text


def _review_write_content_digest(row: dict[str, Any], project: str, split: str,
                                taxonomy_sha256: str) -> str:
    content = {key: value for key, value in row.items() if key not in ('verified', 'review_write_provenance')}
    return audit.annotation_digest(content, project, split, taxonomy_sha256)


def _validate_interrupted_review_write(row: dict[str, Any], item: dict[str, Any], project: str,
                                       split: str, taxonomy_sha256: str,
                                       reconciliation: dict[str, Any] | None) -> dict[str, Any]:
    marker = row.get('review_write_provenance')
    legacy_fields = {'schema_version', 'operation_id', 'decision', 'reviewer', 'created_at',
                     'prior_annotation_sha256', 'written_annotation_sha256', 'partial_reconciliation'}
    current_fields = legacy_fields | {'recovery_mode', 'legacy_recovery_acknowledgment'}
    if isinstance(marker, dict) and set(marker) == legacy_fields:
        # The first marker revision already bound both the ledger entry and
        # written content; absent recovery fields mean a normal source write.
        marker = {**marker, 'recovery_mode': 'normal', 'legacy_recovery_acknowledgment': None}
    if not isinstance(marker, dict) or set(marker) != current_fields or \
            marker.get('schema_version') != 'lfg.review-write-provenance.v1' or \
            not isinstance(marker.get('operation_id'), str) or not re.fullmatch(r'[0-9a-f]{32}', marker['operation_id']) or \
            marker.get('decision') != 'approved' or not isinstance(marker.get('reviewer'), str) or \
            not marker['reviewer'].strip() or '\n' in marker['reviewer'] or '\r' in marker['reviewer'] or \
            not isinstance(marker.get('prior_annotation_sha256'), str) or \
            not SHA256_RE.fullmatch(marker['prior_annotation_sha256']) or \
            marker['prior_annotation_sha256'] != item.get('annotation_sha256') or \
            not isinstance(marker.get('written_annotation_sha256'), str) or \
            not SHA256_RE.fullmatch(marker['written_annotation_sha256']) or \
            marker.get('partial_reconciliation') != reconciliation or \
            marker.get('recovery_mode') not in ('normal', 'legacy_explicit_rereview') or \
            (marker['recovery_mode'] == 'normal' and marker.get('legacy_recovery_acknowledgment') is not None) or \
            (marker['recovery_mode'] == 'legacy_explicit_rereview' and
             marker.get('legacy_recovery_acknowledgment') != 'REVIEW CURRENT INTERRUPTED RECORD'):
        raise ValueError('interrupted review write provenance is missing or does not match; reinitialize review')
    try:
        timestamp = dt.datetime.fromisoformat(marker['created_at'].replace('Z', '+00:00'))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError('interrupted review write provenance has an invalid timestamp') from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None or \
            marker['written_annotation_sha256'] != _review_write_content_digest(row, project, split, taxonomy_sha256):
        raise ValueError('interrupted review source content changed; review it again before recovery')
    return marker


def record_decision(workspace: Path, identifier: str, reviewer: str, decision: str,
                    expected_digest: str, proposed: dict[str, list[str]] | None = None,
                    redactions: list[dict[str, Any]] | None = None,
                    max_result_bytes: int | None = None, guide_sha256: str | None = None,
                    assessment: dict[str, Any] | None = None,
                    assessment_config_path: Path | None = None,
                    max_assessment_profile_bytes: int | None = None,
                    partial_reconciliation: dict[str, Any] | None = None,
                    max_partial_review_bytes: int | None = None,
                    interrupted_recovery_ack: str | None = None) -> None:
    # The same lock serializes partial writes and full approvals. The reentrant
    # lock permits the web handler's broader request transaction to call here.
    with review_mutation_lock(workspace):
        _record_decision(workspace, identifier, reviewer, decision, expected_digest, proposed,
                         redactions, max_result_bytes, guide_sha256, assessment,
                         assessment_config_path, max_assessment_profile_bytes,
                         partial_reconciliation, max_partial_review_bytes, interrupted_recovery_ack)


def _record_decision(workspace: Path, identifier: str, reviewer: str, decision: str,
                     expected_digest: str, proposed: dict[str, list[str]] | None = None,
                     redactions: list[dict[str, Any]] | None = None,
                     max_result_bytes: int | None = None, guide_sha256: str | None = None,
                     assessment: dict[str, Any] | None = None,
                     assessment_config_path: Path | None = None,
                     max_assessment_profile_bytes: int | None = None,
                     partial_reconciliation: dict[str, Any] | None = None,
                     max_partial_review_bytes: int | None = None,
                     interrupted_recovery_ack: str | None = None) -> None:
    if not isinstance(reviewer, str) or not reviewer.strip() or '\n' in reviewer or '\r' in reviewer:
        raise ValueError('reviewer must be a nonempty single-line identifier')
    if decision not in ('approved', 'rejected'):
        raise ValueError('decision must be approved or rejected')
    reconciliation_record = None
    sidecar = workspace.resolve(strict=True) / PARTIAL_REVIEW_FILE
    if decision == 'approved' and sidecar.exists():
        if type(max_partial_review_bytes) is not int or max_partial_review_bytes < 1:
            raise ValueError('configure a partial-review byte limit to reconcile the saved partial review')
        partial = partial_review_state(workspace, identifier, max_partial_review_bytes, guide_sha256)
        revision = partial.get('revision')
        if revision is None:
            if partial_reconciliation is not None:
                raise ValueError('there is no partial review revision to reconcile')
        else:
            if (not isinstance(partial_reconciliation, dict) or
                    set(partial_reconciliation) != {'event_id', 'event_sha256', 'confirmation'} or
                    partial_reconciliation.get('event_id') != revision['event_id'] or
                    partial_reconciliation.get('event_sha256') != revision['event_sha256'] or
                    partial_reconciliation.get('confirmation') != 'RECONCILE PARTIAL REVIEW'):
                raise ValueError('inspect and explicitly reconcile the saved partial review before approval')
            reconciliation_record = {
                'event_id': revision['event_id'], 'event_sha256': revision['event_sha256'],
                'event_type': revision['event_type'], 'revision': revision['revision'],
                'acknowledgment': 'RECONCILE PARTIAL REVIEW',
            }
    elif partial_reconciliation is not None:
        raise ValueError('there is no partial review revision to reconcile')
    if guide_sha256 is not None and (decision != 'approved' or not isinstance(guide_sha256, str)
                                     or not re.fullmatch(r'[0-9a-f]{64}', guide_sha256)):
        raise ValueError('guide digest requires an approval and a SHA-256 value')
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
    interrupted_marker = None
    if decision == 'approved' and row.get('verified') is True:
        if 'review_write_provenance' in row:
            if interrupted_recovery_ack is not None:
                raise ValueError('do not provide a legacy recovery acknowledgment for a bound interrupted write')
            interrupted_marker = _validate_interrupted_review_write(
                row, item, source['project'], source['split'], taxonomy_hash, reconciliation_record)
        elif interrupted_recovery_ack != 'REVIEW CURRENT INTERRUPTED RECORD':
            raise ValueError('legacy interrupted approval requires explicit re-review: type REVIEW CURRENT INTERRUPTED RECORD')
    elif interrupted_recovery_ack is not None:
        raise ValueError('interrupted recovery acknowledgment is only valid for a legacy interrupted approval')
    changed = dict(row)
    if decision == 'approved' and row.get('assessment_targets') is not None and assessment is None:
        raise ValueError('existing assessment targets require review with a matching project profile')
    if assessment is not None:
        if (decision != 'approved' or assessment_config_path is None or
                type(max_assessment_profile_bytes) is not int or max_assessment_profile_bytes < 1):
            raise ValueError('assessment targets require approval and a profile')
        assessment_targets.validate_stored(assessment, source['project'])
        changed['assessment_targets'] = assessment
    if guide_sha256 is not None:
        changed['review_guide_sha256'] = guide_sha256
    if redactions is not None and (decision != 'approved' or not isinstance(redactions, list)):
        raise ValueError('redactions require an approved record')
    if redactions:
        result = apply_redactions(row['text'], redactions)
        if max_result_bytes is not None and len(result.encode('utf-8')) > max_result_bytes:
            raise ValueError('redacted request exceeds configured display limit')
        result_key = audit.text_key(result)
        for candidate_source in plan['sources']:
            candidate_rows, _ = train.load_dataset(private_path(workspace, candidate_source['path']), allow_unlabeled_review=True)
            if any(candidate['id'] != identifier and audit.text_key(candidate['text']) == result_key
                   for candidate in candidate_rows):
                raise ValueError('redacted request duplicates another workspace record')
        previous = row.get('review_transforms', [])
        if not isinstance(previous, list):
            raise ValueError('invalid prior review transformations')
        changed['text'] = result
        changed['review_transforms'] = [*previous, {
            'schema_version': 'lfg.review-redaction.v1',
            'before_sha256': hashlib.sha256(row['text'].encode('utf-8')).hexdigest(),
            'after_sha256': hashlib.sha256(result.encode('utf-8')).hexdigest(),
            'operations': redactions,
        }]
    if decision == 'approved' and row.get('labels_pending') and proposed is None:
        raise ValueError('unlabeled records require explicitly supplied labels')
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
        changed.pop('labels_pending', None)
    if audit.audit(workspace / 'plan.json')[0] != report:
        raise ValueError('workspace changed during review')
    if assessment is not None:
        config, config_sha = assessment_targets.load_config(assessment_config_path, max_assessment_profile_bytes)
        if (config_sha != assessment['config_sha256'] or config['revision'] != assessment['config_revision']
                or assessment_targets.project_profile(config, source['project']) != assessment['project_profile']):
            raise ValueError('assessment profile changed during review')
    if decision == 'approved' or row['verified']:
        changed['verified'] = decision == 'approved'
        if decision == 'approved':
            content_digest = _review_write_content_digest(changed, source['project'], source['split'], taxonomy_hash)
            if interrupted_marker is not None:
                if content_digest != interrupted_marker['written_annotation_sha256']:
                    raise ValueError('interrupted review source content changed; review it again before recovery')
            else:
                changed['review_write_provenance'] = {
                    'schema_version': 'lfg.review-write-provenance.v1',
                    'operation_id': uuid.uuid4().hex,
                    'decision': 'approved',
                    'reviewer': reviewer.strip(),
                    'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                    'prior_annotation_sha256': item['annotation_sha256'],
                    'written_annotation_sha256': content_digest,
                    'partial_reconciliation': reconciliation_record,
                    'recovery_mode': 'legacy_explicit_rereview' if interrupted_recovery_ack else 'normal',
                    'legacy_recovery_acknowledgment': interrupted_recovery_ack,
                }
        rows[index] = changed
        # A crash before the ledger update leaves no matching approval.
        atomic_replace(source_path, encoded_rows(rows))
    item['annotation_sha256'] = audit.annotation_digest(changed, source['project'], source['split'], taxonomy_hash)
    item.update(decision=decision, reviewer=reviewer.strip(), reviewed_at=dt.datetime.now(dt.timezone.utc).isoformat())
    if reconciliation_record is not None:
        item['partial_reconciliation'] = reconciliation_record
    atomic_replace(private_path(workspace, plan['review_ledger']), encoded_rows(list(ledger.values())))


def interactive_review(workspace: Path, reviewer: str, max_partial_review_bytes: int | None = None) -> None:
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
            if row.get('labels_pending'):
                print('LABELS NOT YET PROVIDED: select all applicable labels before approving.')
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
                    reconciliation = None
                    if (workspace.resolve(strict=True) / PARTIAL_REVIEW_FILE).exists():
                        if type(max_partial_review_bytes) is not int or max_partial_review_bytes < 1:
                            raise ValueError('configure --max-partial-review-bytes to reconcile the saved partial review')
                        partial = partial_review_state(workspace, identifier, max_partial_review_bytes)
                        revision = partial.get('revision')
                        if revision is not None:
                            print('SAVED PARTIAL REVIEW HISTORY:', json.dumps(partial['history'], ensure_ascii=True, sort_keys=True))
                            print('Latest event:', revision['event_id'], 'SHA-256:', revision['event_sha256'])
                            if input('Type RECONCILE PARTIAL REVIEW after inspecting this history: ').strip() == 'RECONCILE PARTIAL REVIEW':
                                reconciliation = {'event_id': revision['event_id'],
                                    'event_sha256': revision['event_sha256'],
                                    'confirmation': 'RECONCILE PARTIAL REVIEW'}
                            else:
                                continue
                    recovery_ack = None
                    if row.get('verified') is True and 'review_write_provenance' not in row:
                        if input('Review the current record contents. Type REVIEW CURRENT INTERRUPTED RECORD to explicitly re-review this unbound legacy interrupted write: ').strip() == 'REVIEW CURRENT INTERRUPTED RECORD':
                            recovery_ack = 'REVIEW CURRENT INTERRUPTED RECORD'
                        else:
                            continue
                    record_decision(workspace, identifier, reviewer, 'approved', displayed_digest, proposed,
                                    partial_reconciliation=reconciliation,
                                    max_partial_review_bytes=max_partial_review_bytes,
                                    interrupted_recovery_ack=recovery_ack)
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
        rows, _ = train.load_dataset(private_path(workspace, source['path']), allow_unlabeled_review=True)
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
            child.add_argument('--max-partial-review-bytes', type=int,
                               help='positive sidecar read limit when reconciling saved partial reviews')
        if command == 'export':
            child.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'init':
            result = init_workspace(args.plan, args.workspace)
        else:
            with train.run_lock(args.workspace / 'review'):
                if args.command == 'review':
                    if args.max_partial_review_bytes is not None and args.max_partial_review_bytes < 1:
                        raise ValueError('--max-partial-review-bytes must be positive')
                    interactive_review(args.workspace, args.reviewer, args.max_partial_review_bytes)
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
