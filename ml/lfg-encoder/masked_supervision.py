#!/usr/bin/env python3
"""Private opt-in partial-label export and masked experimental training helpers.

Unknown target cells remain nullable in the export and are excluded from loss
and metrics. This path does not qualify or replace the LFG-owned model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import uuid
from collections import Counter
from typing import Any

import corpus_audit
import evaluate
import review_corpus
import screen_private_corpus
import train

CONFIG_SCHEMA = 'lfg.masked-supervision-config.v1'
EXPORT_SCHEMA = 'lfg.masked-supervision-export.v1'
TRAINING_CONFIG_SCHEMA = 'lfg.masked-training-config.v1'
ROW_SCHEMA = 'lfg.masked-supervision-row.v1'
CONFIG_LIMIT = 2 * 1024 * 1024
SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
HEX32_RE = re.compile(r'^[0-9a-f]{32}$')
FIELDS = tuple(evaluate.FIELDS)

CONFIG_REQUIRED = {
    'schema_version', 'workspace', 'output', 'screening_policy', 'guide', 'selected_labels',
    'allow_unreviewed_proposals', 'unreviewed_complete_array_fields', 'rejected_record_policy',
    'stale_retracted_event_policy', 'weights', 'limits', 'near_duplicate',
}
LIMIT_FIELDS = {
    'max_plan_bytes', 'max_source_file_bytes', 'max_total_source_bytes', 'max_ledger_bytes',
    'max_sidecar_bytes', 'max_guide_bytes', 'max_screening_policy_bytes', 'max_taxonomy_bytes',
    'max_records', 'max_text_bytes',
    'max_total_output_bytes', 'max_near_duplicate_pairs', 'max_near_duplicate_character_product',
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _claim_digest(kind: str, value: str) -> str:
    return _sha((f'lfg.masked-supervision.{kind}.v1\0' + value).encode('utf-8'))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def _load_json_bytes(raw: bytes, description: str) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate key in {description}')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError(f'invalid number in {description}')
    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f'invalid {description}') from error


def _read_bounded(path: Path, maximum: int, description: str) -> bytes:
    if type(maximum) is not int or maximum < 1:
        raise ValueError(f'{description} requires a positive configured byte limit')
    if path.is_symlink():
        raise ValueError(f'{description} cannot be a symlink')
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError(f'{description} is unsafe or exceeds its configured byte limit')
        with path.open('rb') as stream:
            raw = stream.read(maximum + 1)
    except OSError as error:
        raise ValueError(f'{description} is unavailable') from error
    if len(raw) != info.st_size or len(raw) > maximum:
        raise ValueError(f'{description} changed or exceeds its configured byte limit')
    return raw


def _positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _process_is_active(pid: int) -> bool:
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError as error:
            raise ValueError('original review-lock process liveness is unknown') from error
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        error = ctypes.get_last_error()
        if error in (5, 1314):
            raise ValueError('original review-lock process liveness is unknown')
        if error in (87, 1168):
            return False
        raise OSError(error, 'cannot verify original review-lock process liveness')
    try:
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise OSError(ctypes.get_last_error(), 'cannot verify original review-lock process liveness')
        return exit_code.value == 259  # STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


def validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict) or not CONFIG_REQUIRED <= set(config) or \
            config.get('schema_version') != CONFIG_SCHEMA:
        raise ValueError('unsupported masked-supervision configuration')
    # Unknown extensions are accepted but inert. They are retained only in the
    # config digest and never override recognized fields.
    for key in ('workspace', 'output', 'screening_policy'):
        if not isinstance(config[key], str) or not config[key].strip() or not Path(config[key]).is_absolute():
            raise ValueError(f'{key} must be an absolute path')
    if config['guide'] is not None and (not isinstance(config['guide'], str) or
                                        not config['guide'].strip() or not Path(config['guide']).is_absolute()):
        raise ValueError('guide must be null or an absolute path')
    limits = config['limits']
    if not isinstance(limits, dict) or set(limits) != LIMIT_FIELDS:
        raise ValueError('masked-supervision limits must specify the exact supported fields')
    for key, value in limits.items():
        _positive_integer(value, key)
    if type(config['allow_unreviewed_proposals']) is not bool:
        raise ValueError('allow_unreviewed_proposals must be a boolean')
    complete_fields = config['unreviewed_complete_array_fields']
    if not isinstance(complete_fields, list) or any(field not in FIELDS for field in complete_fields) or \
            len(complete_fields) != len(set(complete_fields)):
        raise ValueError('unreviewed_complete_array_fields must be unique taxonomy fields')
    if config['rejected_record_policy'] not in ('exclude', 'block'):
        raise ValueError('rejected_record_policy must be exclude or block')
    if config['stale_retracted_event_policy'] not in ('omit_record', 'block'):
        raise ValueError('stale_retracted_event_policy must be omit_record or block')
    selected = config['selected_labels']
    if not isinstance(selected, dict) or set(selected) != set(FIELDS):
        raise ValueError('selected_labels must name every taxonomy field')
    for field in FIELDS:
        values = selected[field]
        if not isinstance(values, list) or any(not isinstance(label, str) or not label.strip() for label in values) or \
                len(values) != len(set(values)):
            raise ValueError(f'selected_labels.{field} must be a unique string array')
    weights = config['weights']
    if not isinstance(weights, dict) or set(weights) != {'default', 'per_label'} or \
            not isinstance(weights['default'], (int, float)) or isinstance(weights['default'], bool) or \
            not math.isfinite(weights['default']) or weights['default'] <= 0 or not isinstance(weights['per_label'], dict):
        raise ValueError('weights require a finite positive default and per_label object')
    for field, labels in weights['per_label'].items():
        if field not in FIELDS or not isinstance(labels, dict) or set(labels) - set(selected[field]):
            raise ValueError('per_label weights must name selected taxonomy labels')
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or
               not math.isfinite(value) or value <= 0 for value in labels.values()):
            raise ValueError('per-label weights must be finite and positive')
    near = config['near_duplicate']
    if not isinstance(near, dict) or set(near) != {'similarity'} or \
            not isinstance(near['similarity'], (int, float)) or isinstance(near['similarity'], bool) or \
            not math.isfinite(near['similarity']) or not 0 <= near['similarity'] <= 1:
        raise ValueError('near_duplicate.similarity must be within zero and one')
    return config


def load_config(path: Path) -> tuple[dict[str, Any], bytes, str]:
    raw = _read_bounded(path, CONFIG_LIMIT, 'masked-supervision configuration')
    config = validate_config(_load_json_bytes(raw, 'masked-supervision configuration'))
    return config, raw, _sha(raw)


def _verify_private_output(output: Path, allow_existing: bool = False) -> Path:
    output = output.resolve(strict=False)
    if output.exists() and not allow_existing:
        raise ValueError('private export directory already exists')
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise ValueError('existing training output must be a real directory')
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    try:
        result = subprocess.run(['git', '-C', str(ancestor), 'rev-parse', '--show-toplevel'],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError('cannot verify that export output is outside Git worktrees') from error
    if result.returncode == 0:
        root = Path(result.stdout.strip()).resolve()
        if output.is_relative_to(root):
            raise ValueError('private export must be outside every Git worktree')
    elif any((parent / '.git').exists() for parent in (ancestor, *ancestor.parents)):
        raise ValueError('cannot verify that export output is outside Git worktrees')
    if output.exists():
        _verify_owner_private_access(output, is_directory=True)
    return output


def _windows_current_sid() -> str:
    result = subprocess.run(['whoami', '/user', '/fo', 'csv', '/nh'], capture_output=True,
                            text=True, check=True, timeout=5)
    sid = result.stdout.strip().split(',')[-1].strip('"')
    if not re.fullmatch(r'S-1-[0-9-]+', sid):
        raise ValueError('cannot establish current Windows user SID')
    return sid


def _windows_current_logon_principal() -> str:
    result = subprocess.run(['whoami', '/logonid'], capture_output=True, text=True,
                            check=True, timeout=5)
    match = re.fullmatch(r'S-1-5-5-([0-9]+)-([0-9]+)', result.stdout.strip())
    if not match:
        raise ValueError('cannot establish current Windows logon SID')
    return f'NT AUTHORITY\\LogonSessionId_{match.group(1)}_{match.group(2)}'


def _is_allowed_owner_principal(name: str, owner: str, logon_principal: str) -> bool:
    return name.casefold() in (owner.casefold(), logon_principal.casefold())


def _windows_owner_sid(path: Path) -> str:
    path_literal = str(path).replace("'", "''")
    command = (f"(New-Object System.Security.Principal.NTAccount((Get-Acl -LiteralPath '{path_literal}').Owner))."
               'Translate([System.Security.Principal.SecurityIdentifier]).Value')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, text=True, check=True, timeout=5)
    sid = result.stdout.strip()
    if not re.fullmatch(r'S-1-[0-9-]+', sid):
        raise ValueError('cannot establish private path owner SID')
    return sid


def _verify_posix_owner(uid: int, expected_uid: int) -> None:
    if uid != expected_uid:
        raise ValueError('private workspace input is not owned by the current user')


def _verify_owner_sid(actual_sid: str, current_sid: str) -> None:
    if actual_sid.casefold() != current_sid.casefold():
        raise ValueError('private path owner is not the current user')


def _secure_private_path(path: Path, is_directory: bool) -> None:
    try:
        os.chmod(path, 0o700 if is_directory else 0o600)
        if os.name != 'nt':
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode != (0o700 if is_directory else 0o600):
                raise ValueError('export path permissions are not owner-only')
            return
        sid = _windows_current_sid()
        principal = subprocess.run(['whoami'], capture_output=True, text=True,
                                   check=True, timeout=5).stdout.strip()
        logon_principal = _windows_current_logon_principal()
        if not principal:
            raise ValueError('cannot establish owner-only export permissions')
        subprocess.run(['icacls', str(path), '/inheritance:r', '/grant:r', f'*{sid}:(F)'],
                       capture_output=True, text=True, check=True, timeout=5)
        output = subprocess.run(['icacls', str(path)], capture_output=True, text=True,
                                 check=True, timeout=5).stdout
        entries = []
        for line in output.splitlines():
            line = line.strip()
            if line.startswith(str(path)):
                line = line[len(str(path)):].strip()
            if '(' not in line:
                continue
            entries.append(line.split('(', 1)[0].strip().removesuffix(':'))
        def owner_principal(name: str) -> bool:
            return _is_allowed_owner_principal(name, principal, logon_principal)
        for name in set(entries):
            if owner_principal(name):
                continue
            subprocess.run(['icacls', str(path), '/remove:g', name], capture_output=True,
                           text=True, check=True, timeout=5)
            subprocess.run(['icacls', str(path), '/remove:d', name], capture_output=True,
                           text=True, check=True, timeout=5)
        output = subprocess.run(['icacls', str(path)], capture_output=True, text=True,
                                 check=True, timeout=5).stdout
        entries = []
        for line in output.splitlines():
            line = line.strip()
            if line.startswith(str(path)):
                line = line[len(str(path)):].strip()
            if '(' in line:
                entries.append(line.split('(', 1)[0].strip().removesuffix(':'))
        if not entries or any(not owner_principal(name) for name in entries):
            raise ValueError('export ACL is not owner-only')
        _verify_owner_sid(_windows_owner_sid(path), sid)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError('cannot establish owner-only export permissions') from error


def _verify_owner_private_access(path: Path, is_directory: bool) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or (is_directory and not stat.S_ISDIR(info.st_mode)) or \
            (not is_directory and not stat.S_ISREG(info.st_mode)):
        raise ValueError('private workspace input has an unsafe filesystem type')
    if os.name != 'nt':
        _verify_posix_owner(info.st_uid, os.geteuid())
        expected = 0o700 if is_directory else 0o600
        if stat.S_IMODE(info.st_mode) != expected:
            raise ValueError('private workspace input permissions are not owner-only')
        return
    try:
        owner = subprocess.run(['whoami'], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        current_sid = _windows_current_sid()
        logon_principal = _windows_current_logon_principal()
        output = subprocess.run(['icacls', str(path)], capture_output=True, text=True,
                                check=True, timeout=5).stdout
        owner_sid = _windows_owner_sid(path)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError('cannot verify owner-only workspace input ACL') from error
    principals = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith(str(path)):
            line = line[len(str(path)):].strip()
        if '(' in line:
            principal = line.split('(', 1)[0].strip().removesuffix(':')
            principals.append(principal)
            if re.search(r'(?<![A-Z])\(I\)', line) and principal.casefold() != owner.casefold():
                raise ValueError('workspace input ACL grants inherited access outside current user')
    def owner_principal(name: str) -> bool:
        return _is_allowed_owner_principal(name, owner, logon_principal)
    if not owner or not principals or any(not owner_principal(value) for value in principals):
        raise ValueError('workspace input ACL is not owner-only')
    _verify_owner_sid(owner_sid, current_sid)


def _new_private_directory(output: Path) -> None:
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    _secure_private_path(output, is_directory=True)


def _write_private_file(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        if os.name == 'nt':
            _secure_private_path(path, is_directory=False)
        with os.fdopen(descriptor, 'wb') as stream:
            descriptor = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _selected_columns(config: dict[str, Any], taxonomy: dict[str, set[str]]) -> list[dict[str, str]]:
    columns = []
    for field in FIELDS:
        labels = config['selected_labels'][field]
        if set(labels) - taxonomy[field]:
            raise ValueError(f'selected labels for {field} are outside the current taxonomy')
        columns.extend({'field': field, 'label': label} for label in labels)
    if not columns:
        raise ValueError('masked export requires at least one selected target label')
    return columns


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino != 0 and left.st_ino == right.st_ino


def _read_owner_private_sidecar(path: Path, max_bytes: int) -> tuple[bytes, list[dict[str, Any]], bool]:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('partial-review sidecar requires a positive configured byte limit')
    if path.is_symlink():
        raise ValueError('partial-review sidecar cannot be a symlink')
    try:
        before = path.lstat()
    except FileNotFoundError:
        return b'', [], False
    if stat.S_ISLNK(before.st_mode):
        raise ValueError('partial-review sidecar cannot be a symlink')
    if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
        raise ValueError('partial-review sidecar is unsafe or exceeds its configured byte limit')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError('partial-review sidecar could not be opened without following links') from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > max_bytes or \
                not _same_file_identity(before, opened):
            raise ValueError('partial-review sidecar changed before secure open')
        if os.name == 'nt':
            _verify_owner_private_access(path, is_directory=False)
            if not _same_file_identity(opened, path.lstat()):
                raise ValueError('partial-review sidecar changed during owner verification')
        else:
            _verify_posix_owner(opened.st_uid, os.geteuid())
            if stat.S_IMODE(opened.st_mode) != 0o600:
                raise ValueError('partial-review sidecar permissions are not owner-only')
        chunks, total = [], 0
        while True:
            chunk = os.read(descriptor, min(65536, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError('partial-review sidecar exceeds its configured byte limit')
        raw = b''.join(chunks)
        after = os.fstat(descriptor)
        current_path = path.lstat()
        if len(raw) != opened.st_size or after.st_size != opened.st_size or \
                after.st_mtime_ns != opened.st_mtime_ns or after.st_ctime_ns != opened.st_ctime_ns or \
                not _same_file_identity(opened, current_path):
            raise ValueError('partial-review sidecar changed during secure read')
    finally:
        os.close(descriptor)
    events = review_corpus._parse_partial_events(raw, max_bytes)
    return raw, events, True


def _event_binding(row: dict[str, Any], source_sha256: str, annotation_sha256: str,
                   taxonomy_sha256: str, guide_sha256: str | None) -> dict[str, Any]:
    proposal = {field: row.get(field, []) for field in FIELDS}
    return {
        'source_sha256': source_sha256,
        'annotation_sha256': annotation_sha256,
        'taxonomy_sha256': taxonomy_sha256,
        'text_sha256': _sha(row['text'].encode('utf-8')),
        'proposal_sha256': _sha(json.dumps(proposal, ensure_ascii=False, sort_keys=True,
                                           separators=(',', ':')).encode('utf-8')),
        'guide_sha256': guide_sha256,
    }


def _record_targets(row: dict[str, Any], columns: list[dict[str, str]], config: dict[str, Any],
                    event: dict[str, Any] | None, full_review: bool,
                    reviewer_claim: str | None) -> tuple[list[int | None], list[bool], list[float], list[str], list[str | None]]:
    values: list[int | None] = []
    known: list[bool] = []
    weights: list[float] = []
    origins: list[str] = []
    reviewer_claims: list[str | None] = []
    completeness = row.get('target_completeness', {})
    if not isinstance(completeness, dict) or any(type(value) is not bool for value in completeness.values()):
        raise ValueError('target_completeness must map fields to booleans')
    if type(row.get('labels_pending', False)) is not bool:
        raise ValueError('labels_pending must be a boolean when present')
    judgments = event.get('judgments', {}) if event else {}
    proposal_actions = event.get('proposal_actions', {}) if event else {}
    for column in columns:
        field, label = column['field'], column['label']
        weight = config['weights']['per_label'].get(field, {}).get(label, config['weights']['default'])
        explicit = judgments.get(field, {}).get(label)
        action = proposal_actions.get(field, {}).get(label)
        if full_review:
            value = int(label in row[field])
            origin, claim = 'full_reviewed_legacy', reviewer_claim
            is_known = True
        elif explicit in ('positive', 'negative', 'unknown'):
            is_known = explicit != 'unknown'
            value = 1 if explicit == 'positive' else 0 if explicit == 'negative' else None
            origin = f'partial_explicit_{explicit}'
            claim = event.get('reviewer_claim') if event else None
        elif action == 'rejected':
            is_known, value = False, None
            origin, claim = 'rejected_proposal_unknown', event.get('reviewer_claim') if event else None
        else:
            values_in_row = row.get(field, [])
            if label in values_in_row and config['allow_unreviewed_proposals']:
                is_known, value = True, 1
                origin, claim = 'owner_authorized_unreviewed_positive', None
            else:
                complete, completeness_origin = _source_completeness(row, field, config)
                if complete and config['allow_unreviewed_proposals']:
                    is_known, value = True, 0
                    origin = 'source_attested_complete_array_negative' if completeness_origin == 'source_target_completeness_claim' else 'operator_assumed_complete_array_negative'
                    claim = completeness_origin
                else:
                    is_known, value = False, None
                    origin = 'unspecified_unknown'
                    claim = None
        values.append(value)
        known.append(is_known)
        weights.append(float(weight) if is_known else 0.0)
        origins.append(origin)
        reviewer_claims.append(claim)
    return values, known, weights, origins, reviewer_claims


def _safe_source_path(workspace: Path, relative: str, maximum: int) -> tuple[Path, bytes, str]:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError('plan source path must be a nonempty string')
    candidate = workspace / relative
    if candidate.is_symlink():
        raise ValueError('plan source path cannot be a symlink')
    path = candidate.resolve(strict=True)
    if not path.is_relative_to(workspace.resolve()) or path == workspace.resolve():
        raise ValueError('plan source path escapes workspace')
    raw = _read_bounded(path, maximum, 'source artifact')
    return path, raw, _sha(raw)


def _validate_source_copy_claim(config: dict[str, Any], plan_sha256: str,
                                artifacts: dict[str, str], taxonomy_sha256: str,
                                ledger_sha256: str, sidecar_sha256: str | None) -> dict[str, Any] | None:
    claim = config.get('source_copy_claim')
    if claim is None:
        return None
    required = {'original_workspace', 'plan_sha256', 'taxonomy_sha256', 'ledger_sha256',
                'source_artifacts_sha256', 'sidecar_sha256', 'original_lock_marker',
                'original_lock_process_active'}
    if not isinstance(claim, dict) or not required <= set(claim) or \
            not isinstance(claim['original_workspace'], str) or not Path(claim['original_workspace']).is_absolute() or \
            claim['plan_sha256'] != plan_sha256 or claim['taxonomy_sha256'] != taxonomy_sha256 or \
            claim['ledger_sha256'] != ledger_sha256 or claim['source_artifacts_sha256'] != artifacts or \
            claim['sidecar_sha256'] != sidecar_sha256 or \
            not isinstance(claim['original_lock_marker'], str) or claim['original_lock_process_active'] is not False:
        raise ValueError('source-copy claim does not match the verified private snapshot')
    original = Path(claim['original_workspace']).resolve(strict=True)
    original_marker = original / 'review.lock'
    marker_raw = _read_bounded(original_marker, 128, 'original workspace review-lock marker')
    if marker_raw.decode('ascii', errors='strict').splitlines() != [claim['original_lock_marker']]:
        raise ValueError('original review-lock marker changed since copy verification')
    match = re.fullmatch(r'pid=([1-9][0-9]*)', claim['original_lock_marker'])
    if not match:
        raise ValueError('original review-lock marker is malformed')
    if _process_is_active(int(match.group(1))):
        raise ValueError('original review-lock process is active; do not snapshot it')
    return {**claim, 'original_lock_process_active': False,
            'original_lock_stale': True, 'original_lock_modified': False,
            'original_workspace_access_scope': 'local_read_only_hash_reference; owner ACL not enforced'}


def _hash_declared_workspace(workspace: Path, limits: dict[str, int],
                             require_private: bool = True) -> dict[str, Any]:
    """Hash a previously approved train/development plan and only its allowed sources."""
    if require_private:
        _verify_owner_private_access(workspace, is_directory=True)
    plan_raw = _read_bounded(workspace / 'plan.json', limits['max_plan_bytes'], 'source plan')
    plan = _load_json_bytes(plan_raw, 'source plan')
    if not isinstance(plan, dict) or plan.get('schema_version') not in (
            corpus_audit.PLAN_SCHEMA, corpus_audit.DEVELOPMENT_PLAN_SCHEMA) or \
            not isinstance(plan.get('sources'), list) or not plan['sources']:
        raise ValueError('unsupported source workspace plan')
    if any(not isinstance(source, dict) or source.get('split') not in ('train', 'validation')
           for source in plan['sources']):
        raise ValueError('source plan declares a test, holdout or unsupported split; no source opened')
    artifacts = {}
    total = 0
    for source in plan['sources']:
        name = source.get('path')
        candidate = workspace / name if isinstance(name, str) else None
        if candidate is not None and candidate.is_symlink():
            raise ValueError('source plan path cannot be a symlink')
        path = candidate.resolve(strict=True) if candidate is not None else None
        if path is None or not path.is_relative_to(workspace.resolve()) or path == workspace.resolve():
            raise ValueError('source plan path escapes workspace')
        if require_private:
            _verify_owner_private_access(path, is_directory=False)
        raw = _read_bounded(path, limits['max_source_file_bytes'], 'source artifact')
        total += len(raw)
        if total > limits['max_total_source_bytes']:
            raise ValueError('source files exceed configured aggregate byte limit')
        artifacts[name] = _sha(raw)
    if not isinstance(plan.get('taxonomy'), str) or not plan['taxonomy'].strip() or \
            not isinstance(plan.get('review_ledger'), str) or not plan['review_ledger'].strip():
        raise ValueError('source plan must name its taxonomy and review ledger')
    taxonomy_candidate = workspace / plan['taxonomy']
    ledger_candidate = workspace / plan['review_ledger']
    if taxonomy_candidate.is_symlink() or ledger_candidate.is_symlink():
        raise ValueError('source plan taxonomy or ledger cannot be symlinks')
    taxonomy_path = taxonomy_candidate.resolve(strict=True)
    ledger_path = ledger_candidate.resolve(strict=True)
    if not taxonomy_path.is_relative_to(workspace.resolve()) or not ledger_path.is_relative_to(workspace.resolve()):
        raise ValueError('source plan taxonomy or ledger escapes workspace')
    if require_private:
        _verify_owner_private_access(taxonomy_path, is_directory=False)
        _verify_owner_private_access(ledger_path, is_directory=False)
    taxonomy_raw = _read_bounded(taxonomy_path, limits['max_taxonomy_bytes'], 'taxonomy')
    ledger_raw = _read_bounded(ledger_path, limits['max_ledger_bytes'], 'review ledger')
    sidecar_path = workspace / review_corpus.PARTIAL_REVIEW_FILE
    sidecar_raw, _, sidecar_present = _read_owner_private_sidecar(sidecar_path, limits['max_sidecar_bytes'])
    return {'plan': plan, 'plan_raw': plan_raw, 'plan_sha256': _sha(plan_raw),
            'taxonomy_sha256': _sha(taxonomy_raw), 'ledger_sha256': _sha(ledger_raw),
            'source_artifacts_sha256': artifacts,
            'sidecar_sha256': _sha(sidecar_raw) if sidecar_present else None}


def _verify_source_copy_claim(config: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any] | None:
    claim = _validate_source_copy_claim(config, snapshot['plan_sha256'],
                                        snapshot['source_artifacts_sha256'],
                                        snapshot['taxonomy_sha256'], snapshot['ledger_sha256'],
                                        snapshot['sidecar_sha256'])
    if claim is None:
        return None
    original = _hash_declared_workspace(Path(claim['original_workspace']).resolve(strict=True),
                                        config['limits'], require_private=False)
    expected = {key: snapshot[key] for key in (
        'plan_sha256', 'taxonomy_sha256', 'ledger_sha256', 'source_artifacts_sha256', 'sidecar_sha256')}
    observed = {key: original[key] for key in expected}
    if observed != expected:
        raise ValueError('protected workspace copy no longer matches original train-only artifacts')
    return {**claim, 'original_hash_comparison': 'matched_before_export',
            'original_workspace_lock_scope': 'stale_marker_preserved; no live lock acquired'}


def _hash_source_snapshot_again(config: dict[str, Any], snapshot: dict[str, Any],
                                screen_rules: list[Any], screen_limits: dict[str, Any],
                                guide_sha256: str | None) -> dict[str, Any]:
    workspace = Path(config['workspace']).resolve(strict=True)
    _verify_owner_private_access(workspace, is_directory=True)
    artifacts = _hash_declared_workspace(workspace, config['limits'])
    policy_raw = _read_bounded(Path(config['screening_policy']),
                               config['limits']['max_screening_policy_bytes'], 'screening policy')
    policy, _, policy_sha = screen_private_corpus.load_policy_bytes(
        policy_raw, config['limits']['max_screening_policy_bytes'])
    if policy_sha != snapshot['screen_policy_sha256']:
        raise ValueError('screening policy changed during masked export')
    if config['guide'] is None:
        observed_guide_sha = None
    else:
        guide_raw = _read_bounded(Path(config['guide']), config['limits']['max_guide_bytes'], 'label guide')
        observed_guide_sha = _sha(guide_raw)
    if observed_guide_sha != guide_sha256:
        raise ValueError('label guide changed during masked export')
    current = {'plan_sha256': artifacts['plan_sha256'],
               'taxonomy_sha256': artifacts['taxonomy_sha256'],
               'ledger_sha256': artifacts['ledger_sha256'],
               'sidecar_sha256': artifacts['sidecar_sha256'],
               'guide_sha256': guide_sha256}
    if current != snapshot['source_manifest']:
        return current
    _verify_source_copy_claim(config, {
        **artifacts, 'source_artifacts_sha256': artifacts['source_artifacts_sha256']})
    return current


def _snapshot_workspace(config: dict[str, Any], screen_rules: list[Any], screen_limits: dict[str, Any],
                        guide_sha256: str | None) -> dict[str, Any]:
    workspace = Path(config['workspace']).resolve(strict=True)
    plan_raw = _read_bounded(workspace / 'plan.json', config['limits']['max_plan_bytes'], 'source plan')
    plan = _load_json_bytes(plan_raw, 'source plan')
    if not isinstance(plan, dict) or plan.get('schema_version') not in (
            corpus_audit.PLAN_SCHEMA, corpus_audit.DEVELOPMENT_PLAN_SCHEMA) or \
            not isinstance(plan.get('sources'), list) or not plan['sources']:
        raise ValueError('unsupported source workspace plan')
    # This check is deliberately before resolving or reading any source path.
    if any(not isinstance(source, dict) or source.get('split') not in ('train', 'validation')
           for source in plan['sources']):
        raise ValueError('source plan declares a test, holdout or unsupported split; no source opened')
    if plan.get('taxonomy') != 'taxonomy.json' or plan.get('review_ledger') != 'reviews.jsonl':
        raise ValueError('source review workspace must use its in-workspace taxonomy and ledger')
    _verify_owner_private_access(workspace / plan['taxonomy'], is_directory=False)
    _verify_owner_private_access(workspace / plan['review_ledger'], is_directory=False)
    taxonomy_raw = _read_bounded(workspace / plan['taxonomy'], config['limits']['max_taxonomy_bytes'], 'taxonomy')
    taxonomy_json = _load_json_bytes(taxonomy_raw, 'taxonomy')
    if not isinstance(taxonomy_json, dict) or taxonomy_json.get('schema_version') != 'lfg.encoder-evaluation-taxonomy.v1':
        raise ValueError('unsupported taxonomy')
    taxonomy = {field: evaluate.labels(taxonomy_json, field) for field in FIELDS}
    if any(not group for group in taxonomy.values()) or sum(map(len, taxonomy.values())) != len(set.union(*taxonomy.values())):
        raise ValueError('taxonomy fields must be nonempty with unique labels')
    taxonomy_sha256 = _sha(taxonomy_raw)
    columns = _selected_columns(config, taxonomy)
    ledger_raw = _read_bounded(workspace / plan['review_ledger'], config['limits']['max_ledger_bytes'], 'review ledger')
    ledger = _parse_ledger(ledger_raw)
    if not ledger:
        raise ValueError('review ledger is empty')
    total_source_bytes = 0
    all_rows = []
    source_artifacts = []
    source_digests: dict[str, str] = {}
    seen_ids: set[str] = set()
    source_file_for_id = {}
    project_for_id = {}
    for source in plan['sources']:
        if not isinstance(source.get('path'), str) or not source['path'].strip() or \
                not isinstance(source.get('project'), str) or not source['project'].strip():
            raise ValueError('source plan entry has invalid path or project')
        project_group = source.get('project_group', source['project'])
        if not isinstance(project_group, str) or not project_group.strip():
            raise ValueError('source plan project group is invalid')
        path, raw, source_sha256 = _safe_source_path(workspace, source['path'], config['limits']['max_source_file_bytes'])
        _verify_owner_private_access(path, is_directory=False)
        total_source_bytes += len(raw)
        if total_source_bytes > config['limits']['max_total_source_bytes']:
            raise ValueError('source files exceed configured aggregate byte limit')
        rows = _parse_jsonl(raw, 'source artifact')
        if len(all_rows) + len(rows) > config['limits']['max_records']:
            raise ValueError('source record count exceeds configured limit')
        for row in rows:
            _validate_row(row, taxonomy, seen_ids, len(all_rows) + 1)
            if row['source']['project'] != source['project']:
                raise ValueError('source row project does not match plan')
            row['_masked_source'] = {'path': source['path'], 'sha256': source_sha256,
                                     'split': source['split'], 'project': source['project'],
                                     'project_group': project_group}
            source_file_for_id[row['id']] = source['path']
            project_for_id[row['id']] = source
            all_rows.append(row)
        source_digests[source['path']] = source_sha256
        source_artifacts.append({'path': source['path'], 'split': source['split'],
                                 'project_claim_sha256': _claim_digest('project', source['project']),
                                 'project_group_claim_sha256': _claim_digest('project-group', project_group),
                                 'sha256': source_sha256, 'bytes': len(raw), 'records': len(rows)})
    if set(ledger) != seen_ids:
        raise ValueError('review ledger IDs do not exactly match the selected train/development records')
    sidecar_path = workspace / review_corpus.PARTIAL_REVIEW_FILE
    review_corpus._ensure_private_sidecar_location(workspace, sidecar_path)
    sidecar_raw, events, sidecar_present = _read_owner_private_sidecar(
        sidecar_path, config['limits']['max_sidecar_bytes'])
    guide_raw = b''
    if config['guide'] is not None:
        guide_raw = _read_bounded(Path(config['guide']), config['limits']['max_guide_bytes'], 'label guide')
    observed_guide_sha = _sha(guide_raw) if guide_raw else None
    if observed_guide_sha != guide_sha256:
        raise ValueError('configured guide changed during snapshot')
    latest_by_id = {event['id']: event for event in events}
    source_manifest = {
        'plan_sha256': _sha(plan_raw), 'taxonomy_sha256': taxonomy_sha256,
        'ledger_sha256': _sha(ledger_raw), 'sidecar_sha256': _sha(sidecar_raw) if sidecar_present else None,
        'guide_sha256': guide_sha256,
    }
    origin_claim = _verify_source_copy_claim(config, {
        **source_manifest, 'source_artifacts_sha256': source_digests})
    return {'workspace': workspace, 'plan': plan, 'plan_raw': plan_raw, 'ledger': ledger,
            'taxonomy': taxonomy, 'taxonomy_raw': taxonomy_raw, 'taxonomy_sha256': taxonomy_sha256,
            'rows': all_rows, 'source_manifest': source_manifest, 'source_artifacts': source_artifacts,
            'source_digests': source_digests, 'source_file_for_id': source_file_for_id,
            'project_for_id': project_for_id, 'events': latest_by_id, 'event_count': len(events),
            'sidecar_raw': sidecar_raw, 'guide_raw': guide_raw, 'origin_claim': origin_claim,
            'columns': columns, 'source_bytes': total_source_bytes}


def _make_export_rows(snapshot: dict[str, Any], config: dict[str, Any],
                      screen_rules: list[Any], screen_limits: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {'train': [], 'validation': []}
    counts = {
        'input_records': len(snapshot['rows']), 'exported_records': 0,
        'rejected_records_omitted': 0, 'stale_or_retracted_records_omitted': 0,
        'full_reviewed_records': 0, 'unreviewed_records': 0,
        'records_with_current_partial_review': 0, 'records_with_any_known_target': 0,
        'all_unknown_records': 0, 'screened_records_changed': 0,
        'screening_matches_by_rule': Counter(), 'sidecar_events': snapshot['event_count'],
        'sidecar_events_without_source_record': 0,
    }
    support = {f"{column['field']}:{column['label']}": {
        'positive': 0, 'negative': 0, 'unknown': 0, 'origins': Counter()} for column in snapshot['columns']}
    event_ids = set(snapshot['events'])
    source_ids = {row['id'] for row in snapshot['rows']}
    orphan_ids = event_ids - source_ids
    if orphan_ids:
        raise ValueError('partial sidecar contains events without a selected source record')
    project_splits: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    family_splits: dict[str, str] = {}
    text_splits: dict[str, str] = {}
    for row in snapshot['rows']:
        metadata = row.pop('_masked_source')
        split = metadata['split']
        project, group, family = metadata['project'], metadata['project_group'], row['family']
        for mapping, value, kind in ((project_splits, project, 'project'),
                                     (group_splits, group, 'project group'),
                                     (family_splits, family, 'family')):
            if value in mapping and mapping[value] != split:
                raise ValueError(f'{kind} crosses train and development validation')
            mapping[value] = split
        if len(row['text'].encode('utf-8')) > screen_limits['max_text_bytes']:
            raise ValueError('source text exceeds the screening policy text limit')
        screened_text, redaction_counts = screen_private_corpus._screen(row['text'], screen_rules, screen_limits)
        raw_text_sha = _sha(row['text'].encode('utf-8'))
        screened_text_sha = _sha(screened_text.encode('utf-8'))
        normalized = ' '.join(screened_text.casefold().split())
        if normalized in text_splits and text_splits[normalized] != split:
            raise ValueError('exact screened text crosses train and development validation')
        text_splits[normalized] = split
        for rule, number in redaction_counts.items():
            counts['screening_matches_by_rule'][rule] += number
        if redaction_counts:
            counts['screened_records_changed'] += 1
        item = snapshot['ledger'][row['id']]
        taxonomy_hash = snapshot['taxonomy_sha256']
        annotation_sha = corpus_audit.annotation_digest(row, project, split, taxonomy_hash)
        if item['decision'] == 'rejected':
            if config['rejected_record_policy'] == 'block':
                raise ValueError('a source record is rejected by the current review ledger')
            counts['rejected_records_omitted'] += 1
            continue
        if row['verified'] is True:
            if item['decision'] != 'approved' or not corpus_audit.approved(item, annotation_sha):
                raise ValueError('verified source record lacks a matching current approval')
            full_review = True
            reviewer_claim = item['reviewer']
        else:
            if item['decision'] != 'pending':
                raise ValueError('unverified source record must remain pending in the review ledger')
            if item['annotation_sha256'] != annotation_sha:
                raise ValueError('pending source changed after review-workspace initialization')
            full_review = False
            reviewer_claim = None
        event = snapshot['events'].get(row['id'])
        event_current = False
        event_state = None
        if event is not None and not full_review:
            source_binding = _event_binding(row, metadata['sha256'], annotation_sha,
                                            taxonomy_hash, snapshot['source_manifest']['guide_sha256'])
            event_current, event_state = _validate_event(event, source_binding, snapshot['taxonomy'], screened_text)
            if not event_current:
                if config['stale_retracted_event_policy'] == 'block':
                    raise ValueError(f'partial review is {event_state}')
                counts['stale_or_retracted_records_omitted'] += 1
                continue
            for field, label_map in event['proposal_actions'].items():
                if set(label_map) - set(row.get(field, [])):
                    raise ValueError('partial review proposal action does not match the current draft')
            counts['records_with_current_partial_review'] += 1
        if full_review:
            event = None  # Complete approval supersedes partial targets.
            counts['full_reviewed_records'] += 1
        else:
            counts['unreviewed_records'] += 1
        target_values, known, weights, origins, reviewer_claims = _record_targets(
            row, snapshot['columns'], config, event, full_review, reviewer_claim)
        if any(known):
            counts['records_with_any_known_target'] += 1
        else:
            counts['all_unknown_records'] += 1
        for index, column in enumerate(snapshot['columns']):
            entry = support[f"{column['field']}:{column['label']}"]
            if not known[index]:
                entry['unknown'] += 1
            elif target_values[index] == 1:
                entry['positive'] += 1
            else:
                entry['negative'] += 1
            entry['origins'][origins[index]] += 1
        record = {
            'schema_version': ROW_SCHEMA, 'id': _claim_digest('record-id', row['id']), 'text': screened_text,
            'family': _claim_digest('family', family), 'split': split,
            'source_identity': {'artifact_path': metadata['path'], 'artifact_sha256': metadata['sha256'],
                                'project_claim_sha256': _claim_digest('project', project),
                                'project_group_claim_sha256': _claim_digest('project-group', group),
                                'record_id_sha256': _claim_digest('record-id', row['id']),
                                'source_claim_sha256': _sha(json.dumps(row.get('source', {}), sort_keys=True,
                                    ensure_ascii=True, separators=(',', ':')).encode('utf-8'))},
            'text_sha256': screened_text_sha, 'original_text_sha256': raw_text_sha,
            'screening': {'policy_sha256': snapshot['screen_policy_sha256'],
                          'redaction_counts': dict(sorted(redaction_counts.items())),
                          'text_changed': raw_text_sha != screened_text_sha},
            'columns': snapshot['columns'], 'targets': target_values,
            'known_mask': known, 'weights': weights, 'target_origins': origins,
            'target_reviewer_claims': [_claim_digest('reviewer', claim) if claim is not None else None
                                       for claim in reviewer_claims],
            'verified_source_claim': bool(full_review),
            'partial_review_provenance': ({'event_id': event['event_id'],
                'event_sha256': review_corpus._partial_event_sha256(event),
                'revision': event['revision'], 'event_type': event['event_type'],
                'reviewer_claim_sha256': _claim_digest('reviewer', event['reviewer_claim']),
                'reviewed_at': event['reviewed_at'],
                'binding': event['binding']} if event_current else None),
            'full_review_provenance': ({'reviewer_claim_sha256': _claim_digest('reviewer', item.get('reviewer')),
                'reviewed_at': item.get('reviewed_at'),
                'reconciled_partial_review': item.get('partial_reconciliation')} if full_review else None),
        }
        rows_by_split[split].append(record)
        counts['exported_records'] += 1
    train_rows, validation_rows = rows_by_split['train'], rows_by_split['validation']
    if not train_rows:
        raise ValueError('train split has no exportable records')
    near_pairs = near_work = 0
    for train_row in train_rows:
        left = ' '.join(train_row['text'].casefold().split())
        for validation_row in validation_rows:
            right = ' '.join(validation_row['text'].casefold().split())
            near_pairs += 1
            near_work += len(left) * len(right)
            if near_pairs > config['limits']['max_near_duplicate_pairs'] or \
                    near_work > config['limits']['max_near_duplicate_character_product']:
                raise ValueError('train/validation near-duplicate audit exceeds configured work bounds')
            if __import__('difflib').SequenceMatcher(None, left, right, autojunk=False).ratio() >= config['near_duplicate']['similarity']:
                raise ValueError('near-duplicate screened text crosses train and validation')
    counts['sidecar_events_without_source_record'] = len(orphan_ids)
    summary = {
        'counts': {**counts, 'screening_matches_by_rule': dict(sorted(counts['screening_matches_by_rule'].items()))},
        'target_support': {name: {**entry, 'origins': dict(sorted(entry['origins'].items()))}
                           for name, entry in support.items()},
        'near_duplicate_checks': {'pairs': near_pairs, 'character_product_work': near_work},
    }
    return rows_by_split, summary


def load_export(export_dir: Path, *, train_only: bool = False) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    export_dir = export_dir.resolve(strict=True)
    if not export_dir.is_dir() or export_dir.is_symlink():
        raise ValueError('masked export path must be a private directory')
    _verify_owner_private_access(export_dir, is_directory=True)
    config_raw = _read_bounded(export_dir / 'config.json', CONFIG_LIMIT, 'masked export configuration')
    config = validate_config(_load_json_bytes(config_raw, 'masked export configuration'))
    _verify_owner_private_access(export_dir / 'config.json', is_directory=False)
    manifest_raw = _read_bounded(export_dir / 'manifest.json', config['limits']['max_total_output_bytes'],
                                 'masked export manifest')
    _verify_owner_private_access(export_dir / 'manifest.json', is_directory=False)
    if len(config_raw) + len(manifest_raw) > config['limits']['max_total_output_bytes']:
        raise ValueError('masked export metadata exceeds configured aggregate byte limit')
    manifest = _load_json_bytes(manifest_raw, 'masked export manifest')
    if not isinstance(manifest, dict) or manifest.get('schema_version') != EXPORT_SCHEMA:
        raise ValueError('unsupported masked export manifest')
    files = manifest.get('output_files')
    if not isinstance(files, dict) or 'train.jsonl' not in files:
        raise ValueError('masked export manifest must contain train.jsonl')
    if set(files) - {'train.jsonl', 'validation.jsonl'}:
        raise ValueError('masked export file manifest is malformed')
    if manifest.get('config_sha256') != _sha(config_raw):
        raise ValueError('masked export configuration digest does not match manifest')
    rows_by_split = {}
    columns = manifest.get('selected_columns')
    if not isinstance(columns, list) or not columns or any(
            not isinstance(column, dict) or set(column) != {'field', 'label'} or
            column.get('field') not in FIELDS or not isinstance(column.get('label'), str) or
            not column['label'].strip() for column in columns) or \
            len({(column['field'], column['label']) for column in columns}) != len(columns):
        raise ValueError('masked export has no target columns')
    n_labels = len(columns)
    seen_all_split_ids: set[str] = set()
    total_bytes = len(config_raw) + len(manifest_raw)
    for filename, details in files.items():
        # This branch is deliberately before any ACL, stat, or open of the
        # validation artifact. Train-only consumers must not probe held-out data.
        if train_only and filename != 'train.jsonl':
            continue
        if filename not in ('train.jsonl', 'validation.jsonl') or \
                not isinstance(details, dict) or set(details) != {'sha256', 'bytes'} or \
                not isinstance(details['sha256'], str) or not SHA256_RE.fullmatch(details['sha256']) or \
                type(details['bytes']) is not int or details['bytes'] < 1:
            raise ValueError('masked export file manifest is malformed')
        total_bytes += details['bytes']
        if total_bytes > config['limits']['max_total_output_bytes']:
            raise ValueError('masked export splits exceed configured aggregate byte limit')
        _verify_owner_private_access(export_dir / filename, is_directory=False)
        raw = _read_bounded(export_dir / filename, details['bytes'], 'masked export split')
        if len(raw) != details['bytes'] or _sha(raw) != details['sha256']:
            raise ValueError('masked export split digest does not match manifest')
        rows = _parse_jsonl(raw, 'masked export split')
        if not rows:
            raise ValueError('masked export split is empty')
        seen_ids = set()
        expected_split = 'train' if filename == 'train.jsonl' else 'validation'
        for row in rows:
            if row.get('schema_version') != ROW_SCHEMA or not isinstance(row.get('id'), str) or \
                    not SHA256_RE.fullmatch(row['id']) or row['id'] in seen_ids or not isinstance(row.get('text'), str) or \
                    not row['text'].strip() or not isinstance(row.get('family'), str) or \
                    not SHA256_RE.fullmatch(row['family']) or row.get('split') != expected_split or \
                    row['id'] in seen_all_split_ids:
                raise ValueError('masked export row identity or text is invalid')
            source_identity = row.get('source_identity')
            if not isinstance(source_identity, dict) or \
                    not isinstance(source_identity.get('artifact_path'), str) or \
                    Path(source_identity['artifact_path']).is_absolute() or '..' in Path(source_identity['artifact_path']).parts or \
                    any(not isinstance(source_identity.get(name), str) or not SHA256_RE.fullmatch(source_identity[name])
                        for name in ('artifact_sha256', 'project_claim_sha256', 'project_group_claim_sha256',
                                     'record_id_sha256', 'source_claim_sha256')) or \
                    source_identity['record_id_sha256'] != row.get('id') or \
                    not isinstance(row.get('text_sha256'), str) or \
                    row['text_sha256'] != _sha(row['text'].encode('utf-8')):
                raise ValueError('masked export source or text identity is invalid')
            seen_ids.add(row['id'])
            seen_all_split_ids.add(row['id'])
            if row.get('columns') != columns:
                raise ValueError('masked export row columns differ from manifest')
            targets, known, weights = row.get('targets'), row.get('known_mask'), row.get('weights')
            if not isinstance(targets, list) or len(targets) != n_labels or \
                    not isinstance(known, list) or len(known) != n_labels or any(type(flag) is not bool for flag in known) or \
                    not isinstance(weights, list) or len(weights) != n_labels or \
                    not isinstance(row.get('target_origins'), list) or len(row['target_origins']) != n_labels or \
                    any(not isinstance(origin, str) or not origin for origin in row['target_origins']) or \
                    not isinstance(row.get('target_reviewer_claims'), list) or len(row['target_reviewer_claims']) != n_labels or \
                    any(claim is not None and (not isinstance(claim, str) or not SHA256_RE.fullmatch(claim))
                        for claim in row['target_reviewer_claims']):
                raise ValueError('masked target arrays have inconsistent shapes')
            for target, is_known, weight in zip(targets, known, weights, strict=True):
                if is_known:
                    if type(target) is not int or target not in (0, 1) or \
                            not isinstance(weight, (int, float)) or isinstance(weight, bool) or \
                            not math.isfinite(weight) or weight <= 0:
                        raise ValueError('known target must be binary with a finite positive weight')
                elif target is not None or weight != 0:
                    raise ValueError('unknown target must remain null with zero loss weight')
        rows_by_split['train' if filename == 'train.jsonl' else 'validation'] = rows
    return manifest, rows_by_split


def _masked_bce_components(logits: Any, targets: Any, known_mask: Any, weights: Any) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as functional
    if logits.ndim != 2 or targets.shape != logits.shape or known_mask.shape != logits.shape or weights.shape != logits.shape:
        raise ValueError('masked BCE logits, targets, mask and weights must have identical rank-two shapes')
    if known_mask.dtype != torch.bool:
        raise ValueError('known_mask must be boolean')
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError('loss weights must be finite and nonnegative')
    if bool((weights.masked_select(~known_mask) != 0).any()):
        raise ValueError('unknown target cells must have zero loss weight')
    if bool(((targets != 0) & (targets != 1) & known_mask).any()):
        raise ValueError('known targets must be binary')
    effective = weights.to(dtype=torch.float32) * known_mask.to(dtype=torch.float32)
    if bool((known_mask & (effective <= 0)).any()):
        raise ValueError('known target loss weights are not representable as positive float32 values')
    denominator = effective.sum()
    safe_targets = torch.where(known_mask, targets, torch.zeros_like(targets))
    cell_loss = functional.binary_cross_entropy_with_logits(
        logits.to(dtype=torch.float32), safe_targets.to(dtype=torch.float32), reduction='none')
    numerator = (cell_loss * effective).sum()
    return numerator, denominator


def masked_bce_with_logits(logits: Any, targets: Any, known_mask: Any, weights: Any) -> Any:
    """Weighted BCE over known cells only; unknown cells have exactly zero gradient."""
    numerator, denominator = _masked_bce_components(logits, targets, known_mask, weights)
    if float(denominator.detach().item()) <= 0.0:
        return logits.sum() * 0.0
    return numerator / denominator


def masked_eval_batch_loss(numerator: Any, batch_size: int, total_examples: int,
                           total_known_weight: float) -> Any:
    """Scale a numerator for Transformers' example-weighted evaluation reducer."""
    if type(batch_size) is not int or batch_size < 1 or type(total_examples) is not int or \
            total_examples < batch_size or not math.isfinite(total_known_weight) or total_known_weight <= 0:
        raise ValueError('masked validation loss aggregation inputs are invalid')
    return numerator * (total_examples / (total_known_weight * batch_size))


def masked_metrics(targets: Any, predictions: Any, known_mask: Any) -> dict[str, Any]:
    """Compute per-label metrics only over explicitly known target cells."""
    import numpy as np
    truth = np.asarray(targets)
    predicted = np.asarray(predictions)
    known = np.asarray(known_mask)
    if truth.ndim != 2 or predicted.shape != truth.shape or known.shape != truth.shape or known.dtype.kind != 'b':
        raise ValueError('masked metric arrays must be matching rank-two matrices with a boolean mask')
    if np.any(known & ~np.isin(truth, [0, 1])) or np.any(known & ~np.isin(predicted, [0, 1])):
        raise ValueError('known target/prediction cells must be binary')
    by_label = {}
    macro_values = {'precision': [], 'recall': [], 'f1': []}
    micro = Counter()
    for index in range(truth.shape[1]):
        mask = known[:, index]
        actual, output = truth[mask, index].astype(int), predicted[mask, index].astype(int)
        tp = int(((actual == 1) & (output == 1)).sum())
        fp = int(((actual == 0) & (output == 1)).sum())
        fn = int(((actual == 1) & (output == 0)).sum())
        tn = int(((actual == 0) & (output == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        by_label[str(index)] = {'known_support': int(mask.sum()),
                                'positive_support': int((actual == 1).sum()),
                                'negative_support': int((actual == 0).sum()),
                                'unknown_support': int((~mask).sum()),
                                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                                'precision': precision, 'recall': recall, 'f1': f1}
        for key, value in (('precision', precision), ('recall', recall), ('f1', f1)):
            if value is not None:
                macro_values[key].append(value)
        micro.update(tp=tp, fp=fp, fn=fn, tn=tn)
    micro_precision = micro['tp'] / (micro['tp'] + micro['fp']) if micro['tp'] + micro['fp'] else None
    micro_recall = micro['tp'] / (micro['tp'] + micro['fn']) if micro['tp'] + micro['fn'] else None
    micro_f1 = (2 * micro['tp'] / (2 * micro['tp'] + micro['fp'] + micro['fn'])
                if 2 * micro['tp'] + micro['fp'] + micro['fn'] else None)
    return {'known_cells': int(known.sum()), 'unknown_cells': int(known.size - known.sum()),
            'per_label': by_label,
            'macro': {key: sum(values) / len(values) if values else None for key, values in macro_values.items()},
            'micro': {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1,
                      'tp': micro['tp'], 'fp': micro['fp'], 'fn': micro['fn'], 'tn': micro['tn']}}


class MaskedTextDataset:
    """Small HF-compatible dataset; unknown target placeholders are always masked."""
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        if type(max_length) is not int or max_length < 1:
            raise ValueError('max_length must be positive')
        texts = [row['text'] for row in rows]
        train.token_coverage([{'text': text} for text in texts], tokenizer, max_length, 'masked')
        self.tokens = []
        self.labels = []
        self.known = []
        self.weights = []
        for row in rows:
            encoded = tokenizer(row['text'], truncation=False, padding=False)
            input_ids = encoded.get('input_ids')
            if not isinstance(input_ids, list) or not input_ids or len(input_ids) > max_length:
                raise ValueError('tokenizer returned an invalid masked input')
            feature = {key: value for key, value in encoded.items() if key in ('input_ids', 'attention_mask', 'token_type_ids')}
            feature.setdefault('attention_mask', [1] * len(input_ids))
            self.tokens.append(feature)
            self.labels.append([float(value) if flag else 0.0 for value, flag in zip(row['targets'], row['known_mask'], strict=True)])
            self.known.append(list(row['known_mask']))
            self.weights.append(list(row['weights']))

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {**self.tokens[index], 'labels': self.labels[index],
                'known_mask': self.known[index], 'cell_weights': self.weights[index]}


class MaskedDataCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        if not features:
            raise ValueError('cannot collate an empty batch')
        token_features = [{key: value for key, value in item.items()
                           if key not in ('labels', 'known_mask', 'cell_weights')} for item in features]
        padded = self.tokenizer.pad(token_features, padding=True,
                                    pad_to_multiple_of=self.pad_to_multiple_of, return_tensors='pt')
        labels = [item['labels'] for item in features]
        known = [item['known_mask'] for item in features]
        weights = [item['cell_weights'] for item in features]
        if any(not (len(y) == len(mask) == len(weight)) for y, mask, weight in zip(labels, known, weights, strict=True)):
            raise ValueError('masked batch target shapes differ')
        padded['labels'] = torch.tensor(labels, dtype=torch.float32)
        padded['known_mask'] = torch.tensor(known, dtype=torch.bool)
        padded['cell_weights'] = torch.tensor(weights, dtype=torch.float32)
        return padded


def make_masked_trainer_class(trainer_base: Any, gradient_accumulation_steps: int = 1,
                              world_size: int = 1) -> Any:
    if gradient_accumulation_steps != 1:
        raise ValueError('masked training currently requires gradient_accumulation_steps=1 for exact known-cell normalization')
    if world_size != 1:
        raise ValueError('masked training currently rejects distributed execution; loss normalization is single-process only')

    class MaskedTrainer(trainer_base):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
                         num_items_in_batch: Any = None):
            batch = dict(inputs)
            targets = batch.pop('labels')
            known_mask = batch.pop('known_mask')
            weights = batch.pop('cell_weights')
            outputs = model(**batch)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            if not model.training and getattr(self, 'masked_validation_weight_total', None) is not None:
                numerator, _ = _masked_bce_components(logits, targets, known_mask, weights)
                loss = masked_eval_batch_loss(numerator, int(logits.shape[0]),
                                              self.masked_validation_examples,
                                              self.masked_validation_weight_total)
                return (loss, outputs) if return_outputs else loss
            loss = masked_bce_with_logits(logits, targets, known_mask, weights)
            return (loss, outputs) if return_outputs else loss

    return MaskedTrainer


TRAINING_REQUIRED = {
    'schema_version', 'export_dir', 'export_manifest_sha256', 'output', 'base_model', 'model_revision', 'model_manifest',
    'seed', 'epochs', 'max_length', 'batch_size', 'checkpoint_steps', 'device',
    'gradient_accumulation_steps', 'allow_mismatched_sizes', 'allow_unreviewed_targets',
    'allow_partial_review_targets', 'threshold', 'resume',
}


def validate_training_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict) or not TRAINING_REQUIRED <= set(config) or \
            config.get('schema_version') != TRAINING_CONFIG_SCHEMA:
        raise ValueError('unsupported masked-training configuration')
    for name in ('export_dir', 'output'):
        if not isinstance(config[name], str) or not config[name].strip() or not Path(config[name]).is_absolute():
            raise ValueError(f'{name} must be an absolute path')
    if not isinstance(config['export_manifest_sha256'], str) or \
            not SHA256_RE.fullmatch(config['export_manifest_sha256']):
        raise ValueError('export_manifest_sha256 must pin the selected masked export')
    if not isinstance(config['base_model'], str) or not config['base_model'].strip() or \
            not isinstance(config['model_revision'], str) or not re.fullmatch(r'[0-9a-f]{40}', config['model_revision']):
        raise ValueError('masked training requires a model identity and full immutable revision')
    if config['model_manifest'] is not None and (not isinstance(config['model_manifest'], str) or
                                                  not Path(config['model_manifest']).is_absolute()):
        raise ValueError('model_manifest must be null or an absolute path')
    if Path(config['base_model']).exists() and config['model_manifest'] is None:
        raise ValueError('local base_model paths require a verified model manifest')
    if type(config['seed']) is not int or type(config['max_length']) is not int or \
            type(config['batch_size']) is not int or type(config['checkpoint_steps']) is not int or \
            config['max_length'] < 1 or config['batch_size'] < 1 or config['checkpoint_steps'] < 1:
        raise ValueError('masked training integer settings are invalid')
    if not isinstance(config['epochs'], (int, float)) or isinstance(config['epochs'], bool) or \
            not math.isfinite(config['epochs']) or config['epochs'] <= 0:
        raise ValueError('epochs must be finite and positive')
    if config['device'] not in ('auto', 'cuda') or type(config['gradient_accumulation_steps']) is not int or \
            type(config['allow_mismatched_sizes']) is not bool or type(config['allow_unreviewed_targets']) is not bool or \
            type(config['allow_partial_review_targets']) is not bool or type(config['resume']) is not bool:
        raise ValueError('masked training policy values have invalid types')
    if config['gradient_accumulation_steps'] != 1:
        raise ValueError('masked training currently requires gradient_accumulation_steps=1')
    if not isinstance(config['threshold'], (int, float)) or isinstance(config['threshold'], bool) or \
            not math.isfinite(config['threshold']) or not 0 <= config['threshold'] <= 1:
        raise ValueError('threshold must be finite and within zero and one')
    return config


def _load_training_config(path: Path) -> tuple[dict[str, Any], bytes, str]:
    raw = _read_bounded(path, CONFIG_LIMIT, 'masked-training configuration')
    config = validate_training_config(_load_json_bytes(raw, 'masked-training configuration'))
    return config, raw, _sha(raw)


def _verify_pinned_export(export_dir: Path, expected_manifest_sha256: str):
    config_raw = _read_bounded(export_dir / 'config.json', CONFIG_LIMIT, 'masked export configuration')
    export_config = validate_config(_load_json_bytes(config_raw, 'masked export configuration'))
    manifest_raw = _read_bounded(export_dir / 'manifest.json',
                                 export_config['limits']['max_total_output_bytes'], 'masked export manifest')
    observed_sha256 = _sha(manifest_raw)
    if observed_sha256 != expected_manifest_sha256:
        raise ValueError('masked export manifest differs from the operator-pinned digest')
    manifest, partitions = load_export(export_dir)
    final_raw = _read_bounded(export_dir / 'manifest.json',
                              export_config['limits']['max_total_output_bytes'], 'masked export manifest')
    if _sha(final_raw) != observed_sha256:
        raise ValueError('masked export manifest changed while loading training input')
    return manifest, partitions, observed_sha256


def _apply_training_target_policy(rows: list[dict[str, Any]], config: dict[str, Any],
                                  require_known: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, support = [], Counter()
    for source in rows:
        row = dict(source)
        values, known, weights, origins = (list(row['targets']), list(row['known_mask']),
                                           list(row['weights']), list(row['target_origins']))
        reviewer_claims = list(row['target_reviewer_claims'])
        for index, origin in enumerate(origins):
            use = (origin == 'full_reviewed_legacy' or
                   (origin.startswith('partial_explicit_') and config['allow_partial_review_targets']) or
                   (origin.startswith(('owner_authorized_unreviewed_', 'source_attested_complete_array_',
                                       'operator_assumed_complete_array_')) and config['allow_unreviewed_targets']))
            if not use:
                values[index], known[index], weights[index], reviewer_claims[index] = None, False, 0.0, None
                support['excluded_target_cells'] += 1
            elif known[index]:
                support['known_target_cells'] += 1
                support['known_positive_cells' if values[index] == 1 else 'known_negative_cells'] += 1
        row['targets'], row['known_mask'], row['weights'] = values, known, weights
        row['target_reviewer_claims'] = reviewer_claims
        if any(known):
            selected.append(row)
        else:
            if require_known:
                support['all_unknown_rows_excluded'] += 1
            else:
                support['all_unknown_rows_retained'] += 1
                selected.append(row)
    if require_known and not any(any(row['known_mask']) for row in selected):
        raise ValueError('no known train target cells remain under the configured source policy')
    support['rows_with_known_targets'] = sum(any(row['known_mask']) for row in selected)
    support['rows_in_dataset'] = len(selected)
    per_target = {}
    for row in selected:
        columns = row.get('columns')
        if not isinstance(columns, list) or len(columns) != len(row['targets']):
            raise ValueError('masked training row columns do not match target shape')
        for index, column in enumerate(columns):
            name = f"{column['field']}:{column['label']}"
            entry = per_target.setdefault(name, {'known': 0, 'positive': 0, 'negative': 0, 'unknown': 0})
            if row['known_mask'][index]:
                entry['known'] += 1
                entry['positive' if row['targets'][index] == 1 else 'negative'] += 1
            else:
                entry['unknown'] += 1
    support['per_target_support'] = per_target
    return selected, dict(support)


def _hf_metrics(labels: list[dict[str, Any]], threshold: float):
    def compute(eval_prediction: Any) -> dict[str, float]:
        import numpy as np
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        label_ids = eval_prediction.label_ids
        if not isinstance(label_ids, tuple) or len(label_ids) != 2:
            raise ValueError('masked validation must return labels and known masks')
        targets, known = label_ids
        probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
        result = masked_metrics(targets, (probabilities >= threshold).astype(int), known)
        output = {'known_cells': float(result['known_cells']),
                  'unknown_cells': float(result['unknown_cells'])}
        for name in ('precision', 'recall', 'f1'):
            if result['micro'][name] is not None:
                output[f'micro_{name}'] = float(result['micro'][name])
            if result['macro'][name] is not None:
                output[f'macro_{name}'] = float(result['macro'][name])
        for index, column in enumerate(labels):
            record = result['per_label'][str(index)]
            output[f'label_{index}_known_support'] = float(record['known_support'])
            output[f'label_{index}_positive_support'] = float(record['positive_support'])
            output[f'label_{index}_negative_support'] = float(record['negative_support'])
            output[f'label_{index}_unknown_support'] = float(record['unknown_support'])
            for count_name in ('tp', 'fp', 'fn', 'tn'):
                output[f'label_{index}_{count_name}'] = float(record[count_name])
            for metric_name in ('precision', 'recall', 'f1'):
                if record[metric_name] is not None:
                    output[f'label_{index}_{metric_name}'] = float(record[metric_name])
        return output
    return compute


def train_masked(config_path: Path) -> dict[str, Any]:
    config, config_raw, config_sha = _load_training_config(config_path)
    output = _verify_private_output(Path(config['output']), allow_existing=config['resume'])
    export_dir = Path(config['export_dir']).resolve(strict=True)
    export_manifest, partitions, pinned_export_sha = _verify_pinned_export(
        export_dir, config['export_manifest_sha256'])
    train_rows, train_support = _apply_training_target_policy(partitions['train'], config)
    validation_rows, validation_support = _apply_training_target_policy(partitions.get('validation', []), config, require_known=False) \
        if partitions.get('validation') else ([], {'known_target_cells': 0, 'known_positive_cells': 0,
                                                   'known_negative_cells': 0, 'excluded_target_cells': 0,
                                                   'all_unknown_rows_excluded': 0, 'all_unknown_rows_retained': 0,
                                                   'rows_with_known_targets': 0, 'rows_in_dataset': 0,
                                                   'per_target_support': {}})
    validation_has_known = any(any(row['known_mask']) for row in validation_rows)
    if not validation_has_known:
        validation_rows = []
    try:
        import torch
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                                  TrainingArguments, TrainerCallback)
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as error:
        raise SystemExit('Install the locked LFG training dependencies before masked training.') from error
    try:
        args = argparse.Namespace(output=output, seed=config['seed'], epochs=config['epochs'],
                                  batch_size=config['batch_size'], checkpoint_steps=config['checkpoint_steps'],
                                  device=config['device'])
        training_args, cuda_available = train.make_training_arguments(args, torch, TrainingArguments,
                                                                       bool(validation_rows))
    except ValueError as error:
        raise SystemExit(f'masked training device rejected: {error}') from error
    if training_args.device.type != 'cuda' or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('masked training supports one CUDA process only')
    if getattr(training_args, 'n_gpu', 1) != 1:
        raise ValueError('masked training supports one CUDA device only')
    training_args.gradient_accumulation_steps = 1
    training_args.remove_unused_columns = False
    training_args.label_names = ['labels', 'known_mask']
    training_args.metric_for_best_model = 'eval_loss'
    training_args.greater_is_better = False
    try:
        from training_inputs import verify_model
    except ImportError:
        verify_model = None
    model_evidence = None
    if config['model_manifest'] is not None:
        if verify_model is None:
            raise ValueError('local model manifest verification is unavailable')
        model_evidence = verify_model(Path(config['base_model']), Path(config['model_manifest']), config['model_revision'])
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'], revision=config['model_revision'],
                                              local_files_only=bool(model_evidence))
    train_data = MaskedTextDataset(train_rows, tokenizer, config['max_length'])
    validation_data = MaskedTextDataset(validation_rows, tokenizer, config['max_length']) if validation_rows else None
    labels = export_manifest['selected_columns']
    model_digest = _sha(json.dumps({'base_model': config['base_model'], 'revision': config['model_revision'],
                                    'manifest': model_evidence}, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    run_manifest = {
        'schema_version': 'lfg.masked-training-run.v1',
        'training_config_sha256': config_sha,
        'export_manifest_sha256': pinned_export_sha,
        'export_source_artifacts': export_manifest['source_artifacts'],
        'masked_export_schema': EXPORT_SCHEMA,
        'trainer_sha256': _sha(Path(__file__).read_bytes()),
        'legacy_trainer_helpers_sha256': _sha(Path(train.__file__).read_bytes()),
        'screening_policy_sha256': export_manifest['screening_policy_sha256'],
        'partial_review_code_sha256': export_manifest['partial_review_code_sha256'],
        'base_model': config['base_model'], 'model_revision': config['model_revision'],
        'model_identity_sha256': model_digest, 'model_snapshot_evidence': model_evidence,
        'labels': labels, 'train_target_support': train_support,
        'validation_target_support': validation_support,
        'validation_split_present': bool(partitions.get('validation')),
        'validation_configured': bool(validation_rows),
        'gradient_accumulation_steps': 1, 'world_size': 1,
        'normalization': ('train: per-batch weighted mean over known target cells; one optimizer step per batch; '
                          'validation: global weighted-cell mean via example-weighted reducer scaling'),
        'unknown_targets': 'nullable targets with false known_mask and zero cell weight; excluded from loss and metrics',
        'metric_threshold': config['threshold'],
        'allow_unreviewed_targets': config['allow_unreviewed_targets'],
        'allow_partial_review_targets': config['allow_partial_review_targets'],
        'promotion_eligible': False, 'model_is_lfg_owned': False,
        'python': sys.version, 'platform': platform.platform(),
        'packages': train.package_versions(('torch', 'transformers', 'datasets', 'accelerate', 'numpy',
                                            'scikit-learn', 'scipy', 'tokenizers', 'huggingface-hub', 'safetensors')),
        'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
        'selected_device': str(training_args.device), 'cuda_available': cuda_available,
    }
    with train.training_run_lock(output):
        if not output.exists():
            _new_private_directory(output)
        train.prepare_run(output, run_manifest, config['resume'])
        checkpoint = get_last_checkpoint(str(output)) if config['resume'] else None
        if config['resume'] and checkpoint is None:
            raise ValueError('resume requested but no masked checkpoint exists')
        if not config['resume']:
            (output / 'run-manifest.json').write_text(json.dumps(run_manifest, indent=2) + '\n', encoding='utf-8')
        tokenizer_class_kwargs = {'revision': config['model_revision'], 'num_labels': len(labels),
                                  'problem_type': 'multi_label_classification',
                                  'ignore_mismatched_sizes': config['allow_mismatched_sizes'],
                                  'local_files_only': bool(model_evidence)}
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(
                config['base_model'], **tokenizer_class_kwargs)
        trainer_class = make_masked_trainer_class(Trainer, 1, 1)
        trainer = trainer_class(model_init=model_init, args=training_args, train_dataset=train_data,
                                eval_dataset=validation_data,
                                data_collator=MaskedDataCollator(tokenizer),
                                compute_metrics=_hf_metrics(labels, config['threshold']) if validation_data else None,
                                callbacks=[train.make_final_evaluation_callback(TrainerCallback, bool(validation_data))])
        if validation_data:
            trainer.masked_validation_examples = len(validation_rows)
            trainer.masked_validation_weight_total = sum(
                weight for row in validation_rows for weight, known in
                zip(row['weights'], row['known_mask'], strict=True) if known)
            if trainer.masked_validation_weight_total <= 0:
                raise ValueError('masked validation has no known weighted target cells')
        if trainer.args.device.type != 'cuda' or next(trainer.model.parameters()).device.type != 'cuda':
            raise ValueError('masked training device changed after model initialization')
        _verify_pinned_export(export_dir, config['export_manifest_sha256'])
        trainer.train(resume_from_checkpoint=checkpoint)
        if validation_data:
            train.validate_selected_checkpoint(trainer.state)
        trainer.save_model(output)
        tokenizer.save_pretrained(output)
        (output / 'labels.json').write_text(json.dumps(labels, indent=2) + '\n', encoding='utf-8')
        trainer.save_state()
    return {'output': str(output), 'run_manifest_sha256': _sha((output / 'run-manifest.json').read_bytes()),
            'train_target_support': train_support, 'validation_target_support': validation_support,
            'promotion_eligible': False, 'model_is_lfg_owned': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export_parser = sub.add_parser('export')
    export_parser.add_argument('--config', type=Path, required=True)
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export(args.config) if args.command == 'export' else train_masked(args.config)
        print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f'masked supervision {args.command} failed: {error}') from error


def export(config_path: Path) -> dict[str, Any]:
    config, config_raw, config_sha = load_config(config_path)
    output = _verify_private_output(Path(config['output']))
    policy, rules, policy_sha = screen_private_corpus.load_policy(
        Path(config['screening_policy']), config['limits']['max_screening_policy_bytes'])
    policy_code_sha = _sha(Path(screen_private_corpus.__file__).read_bytes())
    taxonomy_code_sha = _sha(Path(evaluate.__file__).read_bytes())
    review_code_sha = _sha(Path(review_corpus.__file__).read_bytes())
    exporter_code_sha = _sha(Path(__file__).read_bytes())
    guide_sha = None
    if config['guide'] is not None:
        guide_raw = _read_bounded(Path(config['guide']), config['limits']['max_guide_bytes'], 'label guide')
        guide_sha = _sha(guide_raw)
    workspace = Path(config['workspace']).resolve(strict=True)
    with review_corpus.review_mutation_lock(workspace):
        snapshot = _snapshot_workspace(config, rules, policy['limits'], guide_sha)
        snapshot['screen_policy_sha256'] = policy_sha
        rows_by_split, summary = _make_export_rows(snapshot, config, rules, policy['limits'])
        files: dict[str, bytes] = {}
        for split, rows in rows_by_split.items():
            if rows:
                filename = 'train.jsonl' if split == 'train' else 'validation.jsonl'
                files[filename] = b''.join(_json_bytes(row) for row in rows)
        output_total = sum(map(len, files.values()))
        if output_total > config['limits']['max_total_output_bytes']:
            raise ValueError('masked export exceeds configured aggregate output byte limit')
        file_sha = {name: _sha(raw) for name, raw in files.items()}
        source_manifest_final = _hash_source_snapshot_again(config, snapshot, rules, policy['limits'], guide_sha)
        manifest = {
            'schema_version': EXPORT_SCHEMA,
            'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'config_sha256': config_sha, 'screening_policy_sha256': policy_sha,
            'screening_code_sha256': policy_code_sha, 'taxonomy_loader_sha256': taxonomy_code_sha,
            'partial_review_code_sha256': review_code_sha, 'exporter_code_sha256': exporter_code_sha,
            'comparison_semantics_code_sha256': _sha(Path(__file__).with_name('classifier_comparison.py').read_bytes()),
            'workspace_plan_sha256': snapshot['source_manifest']['plan_sha256'],
            'taxonomy_sha256': snapshot['taxonomy_sha256'], 'review_ledger_sha256': snapshot['source_manifest']['ledger_sha256'],
            'partial_sidecar_sha256': snapshot['source_manifest']['sidecar_sha256'],
            'label_guide_sha256': guide_sha, 'source_artifacts': snapshot['source_artifacts'],
            'source_copy_claim': snapshot['origin_claim'], 'selected_columns': snapshot['columns'],
            'target_policy': {
                'allow_unreviewed_proposals': config['allow_unreviewed_proposals'],
                'unreviewed_complete_array_fields': config['unreviewed_complete_array_fields'],
                'rejected_record_policy': config['rejected_record_policy'],
                'stale_retracted_event_policy': config['stale_retracted_event_policy'],
                'weights': config['weights'],
                'proposal_action_semantics': 'rejection suppresses draft positive but is not a negative; acceptance alone is not a target',
                'unknown_semantics': 'null target and false known_mask; zero loss weight',
            },
            'claims': {
                'owner_authorized_unreviewed_proposals': config['allow_unreviewed_proposals'],
                'unreviewed_complete_array_assumption': config['unreviewed_complete_array_fields'],
                'screening_recomputed_locally': True,
                'labels_independently_verified': False,
                'reviewer_and_source_claims_independently_verified': False,
                'quality_evaluation_performed': False,
                'promotion_eligible': False,
                'model_training_performed': False,
                'model_is_lfg_owned': False,
                'heldout_accessed': False,
            },
            'counts': summary['counts'], 'target_support': summary['target_support'],
            'near_duplicate_checks': summary['near_duplicate_checks'],
            'output_files': {name: {'sha256': file_sha[name], 'bytes': len(files[name])} for name in files},
            'source_snapshot_reverified_before_write': source_manifest_final == snapshot['source_manifest'],
            'validation_split_present': bool(rows_by_split['validation']),
            'promotion_eligible': False,
        }
        if source_manifest_final != snapshot['source_manifest']:
            raise ValueError('source workspace changed while building masked snapshot')
        manifest_raw = _json_bytes(manifest)
        if len(manifest_raw) + output_total + len(config_raw) > config['limits']['max_total_output_bytes']:
            raise ValueError('masked export and manifest exceed configured output byte limit')
        _new_private_directory(output)
        _write_private_file(output / 'config.json', config_raw)
        for filename, raw in files.items():
            _write_private_file(output / filename, raw)
        _write_private_file(output / 'manifest.json', manifest_raw)
        # Check exact private objects and source identities after writing.
        for filename, expected_sha in file_sha.items():
            if _sha(_read_bounded(output / filename, config['limits']['max_total_output_bytes'], 'export file')) != expected_sha:
                raise ValueError('masked export changed during private write')
        if _sha(_read_bounded(output / 'manifest.json', config['limits']['max_total_output_bytes'], 'export manifest')) != _sha(manifest_raw):
            raise ValueError('masked export manifest changed during private write')
        after = _hash_source_snapshot_again(config, snapshot, rules, policy['limits'], guide_sha)
        if after != snapshot['source_manifest']:
            raise ValueError('source workspace changed during masked export')
        result = {'schema_version': EXPORT_SCHEMA, 'output': str(output), 'manifest_sha256': _sha(manifest_raw),
                  'counts': summary['counts'], 'target_support': summary['target_support'],
                  'promotion_eligible': False, 'model_training_performed': False}
        return result


def _parse_jsonl(raw: bytes, description: str) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        value = _load_json_bytes(line.encode('utf-8'), f'{description} row {index}')
        if not isinstance(value, dict):
            raise ValueError(f'{description} row {index} must be an object')
        rows.append(value)
    return rows


def _validate_row(row: dict[str, Any], taxonomy: dict[str, set[str]], seen_ids: set[str],
                  record_number: int) -> None:
    required = {'id', 'text', 'intents', 'capabilities', 'risks', 'verified', 'family'}
    if required - set(row):
        raise ValueError(f'source record {record_number} is missing required fields')
    identifier = row['id']
    if not isinstance(identifier, str) or not identifier.strip() or identifier in seen_ids:
        raise ValueError('source record IDs must be nonempty and unique')
    seen_ids.add(identifier)
    if not isinstance(row['text'], str) or not row['text'].strip():
        raise ValueError('source record text must be nonempty')
    if type(row['verified']) is not bool:
        raise ValueError('source record verified must be boolean')
    if not isinstance(row['family'], str) or not row['family'].strip():
        raise ValueError('source record family must be nonempty')
    for field in FIELDS:
        labels = row[field]
        if not isinstance(labels, list) or any(not isinstance(label, str) or not label.strip() for label in labels) or \
                len(labels) != len(set(labels)) or set(labels) - taxonomy[field]:
            raise ValueError(f'source record has invalid {field} labels')
    pending = row.get('labels_pending', False)
    if type(pending) is not bool:
        raise ValueError('source record labels_pending must be boolean')
    if pending and row['verified']:
        raise ValueError('verified source record cannot have pending labels')
    if 'source' not in row or not isinstance(row['source'], dict) or \
            not isinstance(row['source'].get('project'), str) or not row['source']['project'].strip():
        raise ValueError('source record requires project provenance')


def _parse_ledger(raw: bytes) -> dict[str, dict[str, Any]]:
    ledger = {}
    for item in _parse_jsonl(raw, 'review ledger'):
        identifier = item.get('id')
        if item.get('schema_version') != corpus_audit.REVIEW_SCHEMA or not isinstance(identifier, str) or \
                not identifier.strip() or identifier in ledger or item.get('decision') not in ('pending', 'approved', 'rejected'):
            raise ValueError('review ledger has an invalid or duplicate entry')
        ledger[identifier] = item
    return ledger


def _validate_event(event: dict[str, Any], binding: dict[str, Any], taxonomy: dict[str, set[str]],
                    screened_text: str) -> tuple[bool, str]:
    if event['event_type'] == 'retraction':
        return False, 'retracted'
    event_binding = event['binding']
    for field, labels in event['judgments'].items():
        if set(labels) - taxonomy[field]:
            raise ValueError('partial event contains labels outside current taxonomy')
    if event_binding != binding:
        return False, 'stale_source_binding'
    if event_binding['text_sha256'] != _sha(screened_text.encode('utf-8')):
        return False, 'stale_screened_text_binding'
    return True, 'current'


def _source_completeness(row: dict[str, Any], field: str, config: dict[str, Any]) -> tuple[bool, str]:
    completeness = row.get('target_completeness', {})
    if not isinstance(completeness, dict) or any(type(value) is not bool for value in completeness.values()):
        raise ValueError('target_completeness must map fields to booleans')
    claim = completeness.get(field)
    if claim is True:
        return True, 'source_target_completeness_claim'
    if claim is False:
        return False, 'source_explicitly_incomplete'
    if field in config['unreviewed_complete_array_fields']:
        return True, 'operator_configured_complete_array_assumption'
    return False, 'completeness_unknown'


if __name__ == '__main__':
    main()
