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
import unicodedata
import uuid
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


def _partial_binding(workspace: Path, identifier: str) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, report, ledger = load_workspace(workspace)
    item = ledger.get(identifier)
    if item is None or item['decision'] != 'pending':
        raise ValueError('partial review is limited to pending records')
    _, rows, index, source = find_row(workspace, plan, identifier)
    row = rows[index]
    if row.get('verified') is True:
        raise ValueError('partial review cannot alter a verified record')
    taxonomy, taxonomy_sha = evaluate.load_taxonomy(private_path(workspace, plan['taxonomy']))
    annotation_sha = audit.annotation_digest(row, source['project'], source['split'], taxonomy_sha)
    if item['annotation_sha256'] != annotation_sha:
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
    if len(raw) != info.st_size or len(raw) > max_bytes or (raw and not raw.endswith(b'\n')):
        raise ValueError('partial review sidecar is truncated or changed')
    events = []
    seen_ids = set()
    seen_event_ids = set()
    for line in raw.splitlines():
        value = json.loads(line)
        fields = {'schema_version', 'event_id', 'id', 'reviewer_claim', 'reviewed_at', 'binding',
                  'judgments', 'proposal_actions', 'context_checks', 'attestation'}
        if not isinstance(value, dict) or set(value) != fields or value.get('schema_version') != PARTIAL_REVIEW_SCHEMA:
            raise ValueError('unsupported partial review event')
        identifier = value.get('id')
        event_id = value.get('event_id')
        if not isinstance(identifier, str) or not identifier or identifier in seen_ids or \
                not isinstance(event_id, str) or not re.fullmatch(r'[0-9a-f]{32}', event_id) or event_id in seen_event_ids:
            raise ValueError('duplicate or malformed partial review event')
        reviewer = value.get('reviewer_claim')
        timestamp = value.get('reviewed_at')
        binding = value.get('binding')
        expected_binding = {'source_sha256', 'annotation_sha256', 'taxonomy_sha256', 'text_sha256',
                            'proposal_sha256', 'guide_sha256'}
        if not isinstance(reviewer, str) or not reviewer.strip() or '\n' in reviewer or '\r' in reviewer or \
                not isinstance(timestamp, str) or not isinstance(binding, dict) or set(binding) != expected_binding or \
                any(not isinstance(binding[key], str) or not SHA256_RE.fullmatch(binding[key]) for key in expected_binding - {'guide_sha256'}) or \
                (binding['guide_sha256'] is not None and (not isinstance(binding['guide_sha256'], str) or
                                                          not SHA256_RE.fullmatch(binding['guide_sha256']))) or \
                value.get('attestation') != 'Reviewer explicitly recorded partial label judgments; unknown cells remain unknown.':
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
        if not any(value.get('judgments', {}).values()) and not any(value.get('proposal_actions', {}).values()):
            raise ValueError('empty partial review event')
        contexts = value.get('context_checks')
        if not isinstance(contexts, list) or any(not isinstance(entry, dict) or
                set(entry) != {'kind', 'status', 'count', 'digest'} or
                not isinstance(entry['kind'], str) or not isinstance(entry['status'], str) or
                type(entry['count']) is not int or entry['count'] < 0 or
                not isinstance(entry['digest'], str) or not SHA256_RE.fullmatch(entry['digest'])
                for entry in contexts):
            raise ValueError('malformed partial review context provenance')
        seen_ids.add(identifier)
        seen_event_ids.add(event_id)
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
    current, binding = _partial_binding(workspace, identifier)
    sidecar = workspace.resolve(strict=True) / PARTIAL_REVIEW_FILE
    _ensure_private_sidecar_location(workspace.resolve(strict=True), sidecar)
    _secure_existing_partial_file(sidecar)
    events = _read_partial_events(sidecar, max_bytes)
    event = next((item for item in events if item['id'] == identifier), None)
    if event is not None:
        for field, labels in event['judgments'].items():
            if set(labels) - current['taxonomy'][field]:
                raise ValueError('partial review contains a label outside the current taxonomy')
        for field, labels in event['proposal_actions'].items():
            if set(labels) - set(current['row'].get(field, [])):
                raise ValueError('partial review proposal no longer matches the source record')
    expected = {**binding, 'guide_sha256': guide_sha256}
    if event is None:
        return {'enabled': True, 'state': 'not_reviewed', 'binding': expected}
    return {'enabled': True, 'state': 'current' if event.get('binding') == expected else 'stale',
            'binding': expected, 'event': event if event.get('binding') == expected else None}


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
                             max_bytes: int, attestation: str) -> None:
    if not isinstance(reviewer, str) or not reviewer.strip() or '\n' in reviewer or '\r' in reviewer:
        raise ValueError('reviewer must be a nonempty single-line identifier')
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('partial review requires a positive configured byte limit')
    if not isinstance(attestation, str) or attestation != 'REVIEW PARTIAL LABELS':
        raise ValueError('explicit partial review attestation required')
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
    if not any(judgments.values()) and not any(proposal_actions.values()):
        raise ValueError('partial review must contain an explicit judgment or proposal action')
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
    if any(event['id'] == identifier for event in existing):
        raise ValueError('a partial review already exists for this record')
    event = {'schema_version': PARTIAL_REVIEW_SCHEMA, 'event_id': uuid.uuid4().hex, 'id': identifier,
             'reviewer_claim': reviewer.strip(), 'reviewed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
             'binding': {**binding, 'guide_sha256': guide_sha256}, 'judgments': judgments,
             'proposal_actions': proposal_actions, 'context_checks': safe_context,
             'attestation': 'Reviewer explicitly recorded partial label judgments; unknown cells remain unknown.'}
    line = encoded_rows([event])
    old = sidecar.read_bytes() if sidecar.exists() else b''
    if len(old) + len(line) > max_bytes:
        raise ValueError('partial review sidecar exceeds its configured byte limit')
    fd, name = tempfile.mkstemp(prefix='.partial-review-', dir=sidecar.parent)
    try:
        os.close(fd)
        temp_path = Path(name)
        _secure_partial_file(temp_path)
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


def record_decision(workspace: Path, identifier: str, reviewer: str, decision: str,
                    expected_digest: str, proposed: dict[str, list[str]] | None = None,
                    redactions: list[dict[str, Any]] | None = None,
                    max_result_bytes: int | None = None, guide_sha256: str | None = None,
                    assessment: dict[str, Any] | None = None,
                    assessment_config_path: Path | None = None,
                    max_assessment_profile_bytes: int | None = None) -> None:
    if not isinstance(reviewer, str) or not reviewer.strip() or '\n' in reviewer or '\r' in reviewer:
        raise ValueError('reviewer must be a nonempty single-line identifier')
    if decision not in ('approved', 'rejected'):
        raise ValueError('decision must be approved or rejected')
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
