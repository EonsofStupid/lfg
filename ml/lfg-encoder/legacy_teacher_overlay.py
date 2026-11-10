#!/usr/bin/env python3
"""Validate and join source-bound, non-exhaustive local teacher proposals.

The module preserves teacher decisions as unreviewed positive claims. It does
not infer negatives, assess semantic quality, or authorize model promotion.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import csv
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

import masked_supervision as masked

CONFIG_SCHEMA = 'lfg.legacy-teacher-overlay-config.v1'
REPORT_SCHEMA = 'lfg.train-teacher-proposals.v1'
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_PROPOSALS_BYTES = 2 * 1024 * 1024 * 1024
SHA256_RE = __import__('re').compile(r'^[0-9a-f]{64}$')
STATUSES = {'answered', 'abstain', 'malformed', 'invalid_schema', 'refusal', 'truncated',
            'incomplete', 'identity_mismatch', 'http_error', 'timeout', 'transport_error',
            'protocol_error', 'request_too_large_not_started', 'not_attempted_after_abort'}
ORIGIN = 'unreviewed_local_teacher_positive'


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def _digest(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _has_reparse_attribute(path: Path, info: os.stat_result) -> bool:
    reparse_flag = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
    attributes = getattr(info, 'st_file_attributes', 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag)


def _verify_report_directory(path: Path) -> bool:
    """Match the producer's owner/SYSTEM/Administrators report-directory policy.

    Files remain separately owner-private through masked_supervision's strict
    file verifier. This checks only the publisher-authorized parent ACL.
    """
    try:
        path = Path(path)
        chain = (path, *path.parents)
        for item in chain:
            info = item.lstat()
            if _has_reparse_attribute(item, info) or (item / '.git').exists():
                return False
        if not path.is_dir():
            return False
        if os.name == 'posix':
            info = path.stat(follow_symlinks=False)
            return info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o700
        if os.name != 'nt':
            return False
        who = subprocess.run(['whoami', '/user', '/fo', 'csv', '/nh'], check=True,
                             capture_output=True, text=True, timeout=10, shell=False).stdout
        row = next(csv.reader([who.strip()]), [])
        if len(row) != 2 or not re.fullmatch(r'S-1-[0-9-]+', row[1]):
            return False
        user_name, sid = row
        acl = subprocess.run(['icacls', str(path)], check=True, capture_output=True,
                             text=True, timeout=10, shell=False).stdout
        lines = acl.splitlines()
        first = lines[0].partition(str(path))[2].strip() if lines else ''
        entries = []
        for line in [first] + [item.strip() for item in lines[1:]]:
            match = re.fullmatch(r'(.+?):((?:\([^)]+\))+)', line)
            if match:
                entries.append(match.groups())
        allowed = {user_name.casefold(), 'builtin\\administrators', 'nt authority\\system'}
        principals = {name.strip().casefold() for name, _rights in entries}
        if (principals != allowed or any('F' not in rights for _name, rights in entries)
                or '(I)' in acl):
            return False
        literal = str(path).replace("'", "''")
        command = (f"(New-Object System.Security.Principal.NTAccount((Get-Acl -LiteralPath '{literal}').Owner))."
                   'Translate([System.Security.Principal.SecurityIdentifier]).Value')
        owner = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                               check=True, capture_output=True, text=True, timeout=10, shell=False).stdout.strip()
        return owner.casefold() == sid.casefold()
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _read_private(path: Path, maximum: int, what: str) -> bytes:
    try:
        if path.is_symlink():
            raise ValueError(f'{what} cannot be a symlink')
        if not _verify_report_directory(path.parent):
            raise ValueError(f'{what} parent directory is not producer-private')
        masked._verify_owner_private_access(path, is_directory=False)
        before = path.lstat()
        if _has_reparse_attribute(path, before) or not stat.S_ISREG(before.st_mode) or \
                before.st_nlink != 1 or before.st_size > maximum:
            raise ValueError(f'{what} exceeds its configured bound')
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_nlink) != (
                    before.st_dev, before.st_ino, before.st_size, before.st_nlink):
                raise ValueError(f'{what} changed during open')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                raw = stream.read(maximum + 1)
            after = path.lstat()
            if len(raw) > maximum or (after.st_dev, after.st_ino, after.st_size, after.st_nlink) != (
                    before.st_dev, before.st_ino, before.st_size, before.st_nlink):
                raise ValueError(f'{what} changed during read')
            return raw
        finally:
            os.close(fd)
    except OSError:
        raise ValueError(f'{what} is unavailable or unsafe') from None


def validate_config(config: Any, selected_columns: list[dict[str, str]]) -> dict[str, Any]:
    required = {'schema_version', 'report_manifest', 'report_manifest_sha256',
                'proposals_file', 'proposals_sha256', 'default_weight',
                'per_label_weights', 'conflict_policy', 'max_manifest_bytes', 'max_proposals_bytes'}
    if not isinstance(config, dict) or not required <= set(config) or config.get('schema_version') != CONFIG_SCHEMA:
        raise ValueError('unsupported legacy teacher overlay configuration')
    extensions = {key: value for key, value in config.items() if key not in required}
    try:
        if len(_canonical(extensions)) > 64 * 1024:
            raise ValueError('overlay extension fields exceed configured bound')
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError('overlay extension fields are not bounded JSON data') from error
    for key in ('report_manifest', 'proposals_file'):
        if not isinstance(config[key], str) or not Path(config[key]).is_absolute():
            raise ValueError(f'{key} must be an absolute private path')
    for key in ('report_manifest_sha256', 'proposals_sha256'):
        if not _digest(config[key]):
            raise ValueError(f'{key} must be a SHA-256 digest')
    for key, maximum in (('max_manifest_bytes', MAX_MANIFEST_BYTES), ('max_proposals_bytes', MAX_PROPOSALS_BYTES)):
        if type(config[key]) is not int or not 1 <= config[key] <= maximum:
            raise ValueError(f'{key} is outside the supported bound')
    if config['conflict_policy'] != 'retain_source_and_skip_conflicting_teacher_positive':
        raise ValueError('unsupported legacy teacher conflict policy')
    default = config['default_weight']
    if isinstance(default, bool) or not isinstance(default, (int, float)) or not math.isfinite(default) or default <= 0:
        raise ValueError('teacher default weight must be finite and positive')
    weights = config['per_label_weights']
    labels = [f"{item['field']}:{item['label']}" for item in selected_columns]
    if len(labels) != len(set(labels)):
        raise ValueError('selected columns have ambiguous legacy label identities')
    if not isinstance(weights, dict) or set(weights) - set(labels) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError('teacher per-label weights must be finite, positive and selected')
    return config


def load_and_join(config: dict[str, Any], source_rows: list[dict[str, Any]], *,
                  source_manifest_sha256: str, source_config_sha256: str,
                  source_train_sha256: str, taxonomy_sha256: str,
                  selected_columns: list[dict[str, str]], expected_source_lineage: dict[str, Any],
                  max_rows: int) -> dict[str, Any]:
    """Verify the pinned package and return valid teacher-positive candidates.

    Unknown and every non-answered state are omitted from candidates. Returned
    counts are integrity/coverage metadata, never a quality result.
    """
    config = validate_config(config, selected_columns)
    manifest_path, proposals_path = Path(config['report_manifest']), Path(config['proposals_file'])
    if manifest_path.resolve(strict=False) == proposals_path.resolve(strict=False):
        raise ValueError('teacher report manifest and proposal file must be distinct inputs')
    manifest_raw = _read_private(manifest_path, config['max_manifest_bytes'], 'teacher report manifest')
    proposal_raw = _read_private(proposals_path, config['max_proposals_bytes'], 'teacher proposal rows')
    if _sha(manifest_raw) != config['report_manifest_sha256'] or _sha(proposal_raw) != config['proposals_sha256']:
        raise ValueError('pinned teacher report package digest mismatch')
    try:
        report = masked._load_json_bytes(manifest_raw, 'teacher report manifest')
    except ValueError as error:
        raise ValueError('teacher report manifest is invalid JSON') from error
    if not isinstance(report, dict) or report.get('schema_version') != REPORT_SCHEMA or \
            report.get('status') not in ('complete', 'partial_aborted_ambiguous') or \
            report.get('promotion_eligible') is not False:
        raise ValueError('teacher report schema or operational status is unsupported')
    if report.get('proposals_sha256') != _sha(proposal_raw) or \
            report.get('proposals_bytes') != len(proposal_raw) or \
            report.get('rows_written') != len(proposal_raw.splitlines()) or \
            not _digest(report.get('publication_id')):
        raise ValueError('teacher report does not bind the proposal package bytes')
    claims = report.get('claims')
    if not isinstance(claims, dict) or claims.get('teacher_suggestions_are_unreviewed') is not True or \
            claims.get('labels_independently_verified') is not False or claims.get('promotion_eligible') is not False or \
            claims.get('heldout_accessed') is not False:
        raise ValueError('teacher report claims are incomplete')
    uncertainty = report.get('uncertainty')
    if not isinstance(uncertainty, dict) or uncertainty.get('teacher_labels_are_claims') is not True or \
            uncertainty.get('omitted_labels_are_negative') is not False or \
            uncertainty.get('source_and_review_claims_independently_verified') is not False or \
            not isinstance(uncertainty.get('selection_claim'), str) or not uncertainty['selection_claim']:
        raise ValueError('teacher report does not prove non-exhaustive uncertainty semantics')
    provenance = report.get('provenance')
    counts = report.get('counts')
    if not isinstance(provenance, dict) or not isinstance(counts, dict) or \
            type(provenance.get('selected_row_count')) is not int or provenance['selected_row_count'] < 1 or \
            type(counts.get('selected_rows')) is not int or counts['selected_rows'] < 1:
        raise ValueError('teacher report provenance is missing')
    digest_fields = ('proposal_config_sha256', 'proposal_config_origin_sha256', 'proposal_config_identity_sha256',
                     'comparison_config_sha256', 'adapter_code_sha256', 'comparator_code_sha256',
                     'export_manifest_sha256',
                     'export_config_sha256', 'train_file_sha256', 'taxonomy_sha256',
                     'selected_row_ids_sha256', 'selected_text_digests_sha256', 'target_mask_sha256',
                     'proposal_schema_sha256', 'prompt_contract_sha256', 'progress_store_id_sha256',
                     'progress_store_events_sha256', 'model_manifest_sha256', 'server_binary_sha256')
    if any(not _digest(provenance.get(name)) for name in digest_fields) or \
            not isinstance(provenance.get('model_name'), str) or not provenance['model_name'] or \
            type(provenance.get('terminal_worker_result_verified')) is not bool or \
            type(provenance.get('progress_resumed')) is not bool:
        raise ValueError('teacher report provenance is incomplete')
    for optional_digest in ('worker_config_sha256', 'payload_sha256'):
        value = provenance.get(optional_digest)
        if value is not None and not _digest(value):
            raise ValueError('teacher report optional worker provenance is malformed')
    if report['status'] == 'complete' and (provenance.get('terminal_worker_result_verified') is not True or
            not _digest(provenance.get('worker_config_sha256')) or not _digest(provenance.get('payload_sha256'))):
        raise ValueError('complete teacher report lacks verified worker provenance')
    publication_id = _sha(_canonical({'proposal_sha256': _sha(proposal_raw),
        'progress_events_sha256': provenance['progress_store_events_sha256'],
        'status': report['status']}).rstrip(b'\n'))
    if report['publication_id'] != publication_id:
        raise ValueError('teacher report publication identity mismatch')
    report_lineage = provenance.get('source_lineage')
    if not isinstance(report_lineage, dict) or \
            report_lineage.get('selected_columns') != selected_columns or \
            (provenance.get('export_manifest_sha256') != source_manifest_sha256 or
            provenance.get('export_config_sha256') != source_config_sha256 or
            provenance.get('train_file_sha256') != source_train_sha256 or
            provenance.get('taxonomy_sha256') != taxonomy_sha256 or
            provenance.get('selected_row_count') != counts.get('selected_rows') or
            report_lineage.get('taxonomy_sha256') != taxonomy_sha256 or
            report_lineage.get('export_config_sha256') != source_config_sha256):
        raise ValueError('teacher report source, taxonomy or selected-column identity mismatch')
    expected_lineage = dict(expected_source_lineage)
    artifact_hashes = expected_lineage.get('source_artifacts_sha256')
    if not isinstance(artifact_hashes, dict):
        raise ValueError('pinned source artifact inventory is malformed')
    # The producer has two established serializers: terminal-verified reports
    # use json.dumps' default ensure_ascii=True; recovered terminal-unverified
    # reports use its ensure_ascii=False canonical serializer.
    if provenance['terminal_worker_result_verified']:
        inventory_raw = json.dumps(artifact_hashes, sort_keys=True, separators=(',', ':')).encode('utf-8')
    else:
        inventory_raw = _canonical(artifact_hashes).rstrip(b'\n')
    expected_lineage['source_artifact_inventory_sha256'] = _sha(inventory_raw)
    if report_lineage != expected_lineage:
        raise ValueError('teacher report source-lineage claims differ from the pinned export manifest')
    lines = proposal_raw.splitlines(keepends=True)
    if len(lines) != provenance['selected_row_count'] or len(lines) > max_rows:
        raise ValueError('teacher proposal row count differs from report selection')
    parsed: list[tuple[dict[str, Any], str, int]] = []
    ids: list[str] = []
    for index, line in enumerate(lines):
        if not line.endswith(b'\n'):
            raise ValueError('teacher proposal rows are truncated')
        try:
            row = masked._load_json_bytes(line, 'teacher proposal row')
        except ValueError as error:
            raise ValueError('teacher proposal row is invalid') from error
        if not isinstance(row, dict) or row.get('row_id') is None:
            raise ValueError('teacher proposal row identity is missing')
        row_id = row['row_id']
        if not _digest(row_id) or row_id in ids:
            raise ValueError('teacher proposal row identity is malformed or duplicated')
        ids.append(row_id)
        parsed.append((row, _sha(line), index))
    ids_sha = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    if ids_sha != provenance['selected_row_ids_sha256']:
        raise ValueError('teacher proposal ordered selection identity mismatch')
    if (_sha(json.dumps([row.get('text_sha256') for row, _, _ in parsed],
                        separators=(',', ':')).encode()) != provenance['selected_text_digests_sha256'] or
            _sha(json.dumps([row.get('source_known_mask') for row, _, _ in parsed],
                        separators=(',', ':')).encode()) != provenance['target_mask_sha256']):
        raise ValueError('teacher report ordered text or target-mask identity mismatch')
    labels = [f"{column['field']}:{column['label']}" for column in selected_columns]
    schema = {'type': 'object', 'additionalProperties': False,
        'properties': {'labels': {'type': 'array', 'items': {'type': 'string', 'enum': labels},
                                  'uniqueItems': True},
                       'abstain': {'type': 'boolean'}},
        'required': ['labels', 'abstain']}
    if _sha(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()) != provenance['proposal_schema_sha256']:
        raise ValueError('teacher proposal schema differs from selected label identities')
    source_by_id = {row.get('id'): row for row in source_rows}
    if len(source_by_id) != len(source_rows) or any(not _digest(key) for key in source_by_id):
        raise ValueError('source row identity is malformed or duplicated')
    candidates: list[dict[str, Any]] = []
    counts = {'teacher_positive': 0, 'unknown': 0, 'source_positive_agreement': 0,
              'source_negative_conflict': 0, 'nonanswered': 0}
    status_counts: dict[str, int] = {}
    label_names = labels
    for proposal, line_sha, index in parsed:
        source = source_by_id.get(proposal['row_id'])
        if source is None:
            raise ValueError('teacher selected row is absent from the pinned source train split')
        for key, proposal_key, source_key in (
                ('family_sha256', 'family', 'family'), ('text_sha256', 'text_sha256', 'text_sha256'),
                ('original_text_sha256', 'original_text_sha256', 'original_text_sha256'),
                ('screening', 'screening', 'screening'), ('source_identity', 'source_identity', 'source_identity'),
                ('columns', 'columns', 'columns'), ('source_targets', 'targets', 'targets'),
                ('source_known_mask', 'known_mask', 'known_mask'),
                ('source_target_origins', 'target_origins', 'target_origins'),
                ('source_reviewer_claims', 'target_reviewer_claims', 'target_reviewer_claims'),
                ('verified_source_claim', 'verified_source_claim', 'verified_source_claim')):
            if proposal.get(key) != source.get(source_key):
                raise ValueError('teacher proposal differs from pinned source row evidence')
        if proposal.get('provenance') != 'unreviewed_local_teacher_claim' or proposal.get('columns') != selected_columns:
            raise ValueError('teacher claim origin or selected columns are invalid')
        states = proposal.get('proposal_states')
        if not isinstance(states, dict) or set(states) != set(label_names):
            raise ValueError('teacher proposal state map differs from selected columns')
        status = proposal.get('status')
        if status not in STATUSES:
            raise ValueError('teacher proposal status is unsupported')
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != 'answered' and any(value != 'unknown' for value in states.values()):
            raise ValueError('non-answered teacher row contains a target claim')
        conflicts = []
        for col_index, label in enumerate(label_names):
            state = states[label]
            if state not in ('unknown', 'proposed_positive'):
                raise ValueError('non-exhaustive teacher report contains a non-positive target')
            if state == 'proposed_positive':
                counts['teacher_positive'] += 1
                if source['known_mask'][col_index]:
                    if source['targets'][col_index] == 1:
                        counts['source_positive_agreement'] += 1
                    else:
                        counts['source_negative_conflict'] += 1
                        conflicts.append(label)
                else:
                    candidates.append({'source': source, 'column_index': col_index,
                                       'label_name': label, 'proposal_sha256': line_sha,
                                       'proposal_index': index, 'report_manifest_sha256': config['report_manifest_sha256'],
                                       'proposals_sha256': config['proposals_sha256'],
                                       'provenance': provenance, 'status': status})
            else:
                counts['unknown'] += 1
        if proposal.get('conflict_labels') != conflicts:
            raise ValueError('teacher/source conflict claims do not match source evidence')
        if status != 'answered':
            counts['nonanswered'] += 1
    report_counts = report.get('counts', {})
    expected_status_counts = dict(sorted(status_counts.items()))
    failures = sum(number for status, number in status_counts.items() if status not in ('answered', 'abstain'))
    if report_counts.get('statuses') != expected_status_counts or \
            report_counts.get('answered_rows') != status_counts.get('answered', 0) or \
            report_counts.get('abstentions') != status_counts.get('abstain', 0) or \
            report_counts.get('failures') != failures or \
            report_counts.get('not_attempted_after_abort') != status_counts.get('not_attempted_after_abort', 0):
        raise ValueError('teacher report status counts differ from proposal rows')
    return {'candidates': candidates, 'report': report, 'provenance': provenance,
            'report_manifest_sha256': config['report_manifest_sha256'],
            'proposals_sha256': config['proposals_sha256'], 'proposal_bytes': len(proposal_raw),
            'counts': counts, 'weight_policy': {'default_weight': float(config['default_weight']),
                'per_label_weights': {key: float(value) for key, value in config['per_label_weights'].items()},
                'conflict_policy': config['conflict_policy']},
            'config_sha256': _sha(_canonical(config))}


def reverify_package(config: dict[str, Any], result: dict[str, Any]) -> None:
    """Re-read pinned inputs before publication to detect post-join replacement."""
    manifest = _read_private(Path(config['report_manifest']), config['max_manifest_bytes'], 'teacher report manifest')
    proposals = _read_private(Path(config['proposals_file']), config['max_proposals_bytes'], 'teacher proposal rows')
    if (_sha(manifest) != result['report_manifest_sha256'] or
            _sha(proposals) != result['proposals_sha256'] or
            len(proposals) != result['proposal_bytes']):
        raise ValueError('teacher report package changed before training artifact publication')
