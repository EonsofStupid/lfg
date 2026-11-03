#!/usr/bin/env python3
"""Build and consume source-bound user-requested routing observations.

Observations are bounded, caller-interpreted source claims only. They remain
unreviewed and do not represent actual preference, teacher outputs, provider
entitlement, execution, outcomes, quality, or training labels. This module
returns bounded values in memory and does not write observations.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

import inspect_codex_inventory
import screen_private_corpus
import source_bound_snapshot


SCHEMA = 'lfg.user-requested-routing-observation.v1'
SUMMARY_SCHEMA = 'lfg.user-requested-routing-observation-summary.v1'
CONFIG_SCHEMA = 'lfg.user-requested-routing-observation-config.v1'
MANIFEST_SCHEMA = 'lfg.reserved-session-groups.v1'
MAX_CONFIG_BYTES = 64 * 1024
CONFIG_LIMITS = {
    'max_source_prefix_bytes', 'max_line_bytes', 'max_user_message_bytes',
    'max_span_count', 'max_span_bytes', 'max_extension_bytes',
    'max_extension_nodes', 'max_extension_depth', 'max_extension_string_bytes',
    'max_observation_bytes', 'max_reserved_manifest_bytes', 'max_reserved_project_labels',
    'max_claims_per_field',
}
SPAN_FIELDS = {'phase', 'model', 'effort'}
RESERVED_EXTENSION_KEYS = {
    'schema_version', 'kind', 'source', 'interpretation', 'preference', 'state',
    'extensions', 'teacher', 'teacher_proposal', 'request', 'response', 'outcome',
    'success', 'failure', 'review_status', 'promotion_eligible', 'entitlement',
    'agent_count', 'total_agent_count', 'model_id', 'provider', 'recommendation',
    'source_file', 'source_path', 'source_locator', 'prompt', 'raw_prompt', 'text',
}
SENSITIVE_EXTENSION_KEY = re.compile(
    r'(?:^|[_-])(?:secret|password|credential|access[_-]?token|refresh[_-]?token|'
    r'auth[_-]?token|session[_-]?token|id[_-]?token|bearer[_-]?token|jwt|'
    r'token|tokens|api[_-]?key|access[_-]?key)(?:$|[_-])', re.I)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            + '\n').encode('utf-8')


def _strict_json(raw: bytes) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def bad_constant(_value):
        raise ValueError('invalid JSON constant')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def _valid_sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def load_config(path: Path) -> tuple[dict, str]:
    """Load a small exact-key config; resource bounds remain operator chosen."""
    with path.open('rb') as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError('observation config exceeds bootstrap limit')
    try:
        value = _strict_json(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError('invalid observation config') from None
    validate_config(value)
    return value, sha(raw)


def validate_config(config: object) -> None:
    if (not isinstance(config, dict) or set(config) != CONFIG_LIMITS | {
            'schema_version', 'phase_labels', 'claim_polarities'}
            or config.get('schema_version') != CONFIG_SCHEMA
            or any(type(config.get(key)) is not int or config[key] < 1 for key in CONFIG_LIMITS)
            or not isinstance(config.get('phase_labels'), list) or not config['phase_labels']
            or any(not isinstance(label, str)
                   or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', label)
                   for label in config['phase_labels'])
            or len(set(config['phase_labels'])) != len(config['phase_labels'])
            or not isinstance(config.get('claim_polarities'), list) or not config['claim_polarities']
            or any(not isinstance(label, str)
                   or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', label)
                   for label in config['claim_polarities'])
            or len(set(config['claim_polarities'])) != len(config['claim_polarities'])):
        raise ValueError('invalid observation config')


def read_reserved_manifest(path: Path, config: dict, *, screen_policy_path: Path | None = None) -> tuple[list[str], str, str, list[dict]]:
    """Read an owner-only bounded POSIX manifest; unknown/partial is not complete."""
    if os.name == 'nt':
        raise ValueError('reserved manifests are unavailable on Windows without verified no-follow traversal')
    validate_config(config)
    directory_fd = file_fd = -1
    try:
        directory_fd, name, _ = source_bound_snapshot._open_private_parent(path)
        source_bound_snapshot._verify_parent_binding(path, directory_fd, name)
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) & 0o077
                or before.st_size > config['max_reserved_manifest_bytes']):
            raise ValueError('reserved manifest must be a private regular file')
        chunks, remaining = [], config['max_reserved_manifest_bytes'] + 1
        while remaining:
            chunk = os.read(file_fd, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b''.join(chunks)
        after = os.fstat(file_fd)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (len(raw) > config['max_reserved_manifest_bytes'] or len(raw) != after.st_size
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino)
                or not stat.S_ISREG(named.st_mode) or named.st_nlink != 1):
            raise ValueError('reserved manifest changed during read')
        source_bound_snapshot._verify_parent_binding(path, directory_fd, name)
    except OSError:
        raise ValueError('reserved manifest unavailable or changed during read') from None
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if directory_fd >= 0:
            os.close(directory_fd)
    try:
        manifest = _strict_json(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError('invalid reserved manifest') from None
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != MANIFEST_SCHEMA
            or manifest.get('coverage') not in ('unknown', 'partial')
            or not isinstance(manifest.get('groups'), list)
            or 'identities' not in manifest
            or len(manifest['groups']) > config['max_reserved_manifest_bytes'] // 64
            or any(not _valid_sha(group) for group in manifest['groups'])
            or len(set(manifest['groups'])) != len(manifest['groups'])):
        raise ValueError('invalid reserved manifest')
    source_bound_snapshot._require_supported_reserved_groups(manifest['groups'])
    identities = source_bound_snapshot._validate_reserved_source_identities(
        manifest['identities'], screen_policy_path=screen_policy_path)
    if len(canonical(identities)) > config['max_reserved_manifest_bytes']:
        raise ValueError('reserved source identities exceed configured manifest limit')
    return manifest['groups'], sha(raw), manifest['coverage'], identities


def pin_reserved_manifest(path: Path, config: dict, *, screen_policy_path: Path | None = None) -> dict:
    groups, digest, coverage, identities = read_reserved_manifest(path, config, screen_policy_path=screen_policy_path)
    return {'_locator': str(path.absolute()), 'manifest_sha256': digest,
            'groups': groups, 'identities': identities, 'coverage': coverage}


def _recheck_reserved(reserved: dict | None, config: dict, *, screen_policy_path: Path | None = None) -> tuple[list[str], str | None, str, list[dict]]:
    if reserved is None:
        return [], None, 'unknown', []
    if (not isinstance(reserved, dict) or not isinstance(reserved.get('_locator'), str)
            or not reserved['_locator'] or not _valid_sha(reserved.get('manifest_sha256'))
            or not isinstance(reserved.get('groups'), list)
            or not isinstance(reserved.get('identities'), list)
            or reserved.get('coverage') not in ('unknown', 'partial')):
        raise ValueError('invalid pinned reserved manifest')
    groups, digest, coverage, identities = read_reserved_manifest(
        Path(reserved['_locator']), config, screen_policy_path=screen_policy_path)
    if (groups != reserved['groups'] or digest != reserved['manifest_sha256']
            or coverage != reserved['coverage'] or identities != reserved['identities']):
        raise ValueError('reserved manifest changed')
    return groups, digest, coverage, identities


def _project_key(value: str) -> str:
    return ''.join(character for character in value.casefold() if character.isalnum())


def _safe_extensions(value: object, rules: list, policy_limits: dict, config: dict,
                     *, require_already_screened: bool = False,
                     forbidden_literals: tuple[str, ...] = ()) -> object:
    nodes = 0

    def visit(item: object, depth: int) -> object:
        nonlocal nodes
        nodes += 1
        if depth > config['max_extension_depth'] or nodes > config['max_extension_nodes']:
            raise ValueError('extension bounds exceeded')
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                normalized_key = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', key) if isinstance(key, str) else ''
                if (not isinstance(key, str) or not key or len(key.encode('utf-8')) > config['max_extension_string_bytes']
                        or normalized_key.casefold() in RESERVED_EXTENSION_KEYS
                        or SENSITIVE_EXTENSION_KEY.search(normalized_key)
                        or source_bound_snapshot.SENSITIVE_KEY.search(normalized_key)):
                    raise ValueError('extension key is invalid or collides with a reserved field')
                if any(literal and literal in key for literal in forbidden_literals):
                    raise ValueError('extension key copies selected source text')
                clean_key, _ = screen_private_corpus._screen(key, rules, policy_limits)
                if clean_key != key or clean_key in result:
                    raise ValueError('extension key requires redaction or collides')
                result[key] = visit(child, depth + 1)
            return result
        if isinstance(item, list):
            return [visit(child, depth + 1) for child in item]
        if isinstance(item, str):
            if len(item.encode('utf-8')) > config['max_extension_string_bytes']:
                raise ValueError('extension string exceeds configured limit')
            if any(literal and literal in item for literal in forbidden_literals):
                raise ValueError('extension value copies selected source text')
            clean, _ = screen_private_corpus._screen(item, rules, policy_limits)
            if require_already_screened and clean != item:
                raise ValueError('extension value requires screening')
            return clean
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            if len(str(item)) > config['max_extension_string_bytes']:
                raise ValueError('extension integer exceeds configured limit')
            return item
        if type(item) is float and __import__('math').isfinite(item):
            return item
        raise ValueError('invalid extension value')

    result = visit(value, 0)
    try:
        extension_size = len(canonical(result))
    except (RecursionError, ValueError, TypeError, OverflowError):
        raise ValueError('extension cannot be encoded within configured bounds') from None
    if extension_size > config['max_extension_bytes']:
        raise ValueError('extension payload exceeds configured byte limit')
    return result


def _span_map(text: str, spans: object, config: dict) -> tuple[dict, list[dict]]:
    if not isinstance(spans, list) or len(spans) < 1 or len(spans) > config['max_span_count']:
        raise ValueError('invalid source span list')
    raw = text.encode('utf-8')
    parsed, outputs = {}, []
    for span in spans:
        if not isinstance(span, dict) or set(span) != {'id', 'field', 'start_byte', 'end_byte'}:
            raise ValueError('invalid source span shape')
        identifier, field = span['id'], span['field']
        start, end = span['start_byte'], span['end_byte']
        if (not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', identifier)
                or identifier in parsed or not isinstance(field, str) or field not in SPAN_FIELDS
                or type(start) is not int or type(end) is not int or start < 0 or end <= start
                or end > len(raw) or end - start > config['max_span_bytes']):
            raise ValueError('invalid source span bounds or identity')
        span_raw = raw[start:end]
        try:
            span_text = span_raw.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError('source span splits a UTF-8 code point') from None
        if not span_text.strip():
            raise ValueError('source span is empty')
        digest = sha(span_raw)
        output_id = f'span-{len(outputs) + 1}'
        parsed[identifier] = {'field': field, 'text': span_text, 'start_byte': start,
                              'end_byte': end, 'sha256': digest, 'output_id': output_id}
        outputs.append({'id': output_id, 'field': field, 'start_byte': start,
                        'end_byte': end, 'sha256': digest})
    return parsed, outputs


def _source_claims(claims: object, field: str, spans: dict, config: dict) -> dict:
    if not isinstance(claims, list) or len(claims) > config['max_claims_per_field']:
        raise ValueError(f'invalid source-linked {field} claims')
    if not claims:
        return {'status': 'unknown'}
    results = []
    used_spans = set()
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {'text', 'span_id', 'polarity'}:
            raise ValueError(f'invalid caller-interpreted {field} claim')
        text, span_id, polarity = claim['text'], claim['span_id'], claim['polarity']
        if (not isinstance(text, str) or not isinstance(span_id, str) or span_id not in spans
                or not isinstance(polarity, str) or polarity not in config['claim_polarities']):
            raise ValueError(f'incomplete caller-interpreted {field} claim')
        span = spans[span_id]
        if span['field'] != field or span['text'] != text or span_id in used_spans:
            raise ValueError(f'{field} claim does not match a unique source span')
        used_spans.add(span_id)
        results.append({'status': 'unreviewed_source_claim', 'polarity': polarity,
                        'evidence_span_id': span['output_id'], 'text_sha256': span['sha256']})
    return {'status': 'unreviewed_source_claims', 'items': results,
            'review_status': 'unreviewed'}


def build_observation(record: dict, spans: list[dict], interpretation: dict, config: dict,
                      screen_policy_path: Path, *, reserved: dict | None = None,
                      reserved_project_labels: list[str] | None = None,
                      extensions: dict | None = None) -> dict:
    """Verify exactly one selected user message and return a text-free observation."""
    if os.name == 'nt':
        raise ValueError('source-bound observations are unavailable on Windows without verified no-follow traversal')
    validate_config(config)
    inspect_codex_inventory.verify_record(record)
    if (not isinstance(record.get('project'), str) or not record['project'].strip()
            or not Path(record['source_file']).is_absolute()
            or type(record.get('source_line')) is not int or record['source_line'] < 1
            or type(record.get('source_prefix_bytes')) is not int or record['source_prefix_bytes'] < 1
            or record['source_prefix_bytes'] > config['max_source_prefix_bytes']
            or any(not _valid_sha(record.get(key)) for key in (
                'source_file_sha256', 'source_prefix_sha256', 'source_line_sha256', 'text_sha256'))):
        raise ValueError('invalid or over-limit source inventory identity')
    reserved_groups, reserved_sha, reserved_coverage, reserved_identities = _recheck_reserved(
        reserved, config, screen_policy_path=screen_policy_path)
    labels = reserved_project_labels or []
    if (not isinstance(labels, list) or len(labels) > config['max_reserved_project_labels']
            or any(not isinstance(label, str) or not label.strip()
                   or len(label.encode('utf-8')) > config['max_extension_string_bytes'] for label in labels)):
        raise ValueError('invalid reserved project labels')
    group_id = source_bound_snapshot.source_group_id(record)
    if (group_id in set(reserved_groups)
            or _project_key(record['project']) in {_project_key(label) for label in labels}
            or source_bound_snapshot._matches_reserved_identity(record, reserved_identities)):
        raise ValueError('selected source belongs to a reserved group')

    # This opens only the single caller-selected source after reservation checks.
    message = source_bound_snapshot.verify_selected_source(
        record, max_session_bytes=config['max_source_prefix_bytes'],
        max_line_bytes=config['max_line_bytes'])
    message_bytes = message.encode('utf-8')
    if len(message_bytes) > config['max_user_message_bytes']:
        raise ValueError('selected user message exceeds configured limit')
    if not isinstance(interpretation, dict) or set(interpretation) != {
            'phase_label', 'phase_span_id', 'model_claims', 'effort_claims'}:
        raise ValueError('invalid caller interpretation')
    phase_label = interpretation['phase_label']
    if (not isinstance(phase_label, str) or phase_label not in config['phase_labels']):
        raise ValueError('invalid phase interpretation label')
    parsed_spans, output_spans = _span_map(message, spans, config)
    phase_id = interpretation['phase_span_id']
    if not isinstance(phase_id, str) or phase_id not in parsed_spans or parsed_spans[phase_id]['field'] != 'phase':
        raise ValueError('phase interpretation requires a source evidence span')
    model_claims = _source_claims(interpretation['model_claims'], 'model', parsed_spans, config)
    effort_claims = _source_claims(interpretation['effort_claims'], 'effort', parsed_spans, config)
    referenced = {phase_id}
    for claim_key in ('model_claims', 'effort_claims'):
        referenced.update(claim['span_id'] for claim in interpretation[claim_key])
    if referenced != set(parsed_spans):
        raise ValueError('source spans must be limited to the interpreted request')
    ranges = sorted((item['start_byte'], item['end_byte']) for item in parsed_spans.values())
    if any(current[0] < previous[1] for previous, current in zip(ranges, ranges[1:])):
        raise ValueError('interpreted source spans must not overlap')

    policy, rules, policy_sha = screen_private_corpus.load_policy(screen_policy_path, 1024 * 1024)
    baseline, baseline_rules, _ = screen_private_corpus.load_policy(
        source_bound_snapshot._default_screen_policy_path(), 1024 * 1024)
    policies = [(baseline, baseline_rules), (policy, rules)]
    labels_to_screen = [phase_label, *(item['polarity'] for item in interpretation['model_claims']),
                        *(item['polarity'] for item in interpretation['effort_claims'])]
    for label in labels_to_screen:
        if source_bound_snapshot._screen_text_policies(label, policies)[0] != label:
            raise ValueError('caller interpretation label requires redaction')
    if extensions is not None and not isinstance(extensions, dict):
        raise ValueError('extensions must be an object')
    forbidden = tuple([message, record['source_file'], *(span['text'] for span in parsed_spans.values())])
    safe_extensions = extensions if extensions is not None else {}
    for current_policy, current_rules in policies:
        safe_extensions = _safe_extensions(safe_extensions, current_rules, current_policy['limits'], config,
                                           forbidden_literals=forbidden)
    for current_policy, current_rules in policies:
        _safe_extensions(safe_extensions, current_rules, current_policy['limits'], config,
                         require_already_screened=True, forbidden_literals=forbidden)
    observation = {
        'schema_version': SCHEMA,
        'kind': 'user_requested',
        'source': {
            'role': 'user',
            'project_label_sha256': sha(record['project'].encode('utf-8')),
            'source_group_sha256': group_id,
            'source_file_sha256_at_inventory': record['source_file_sha256'],
            'source_prefix_bytes': record['source_prefix_bytes'],
            'source_prefix_sha256': record['source_prefix_sha256'],
            'source_line': record['source_line'],
            'source_line_sha256': record['source_line_sha256'],
            'user_text_sha256': record['text_sha256'],
            'user_text_utf8_bytes': len(message_bytes),
            'spans': output_spans,
        },
        'reservation': {
            'exclusion_checked': True,
            'manifest_sha256': reserved_sha,
            'coverage': reserved_coverage,
            'groups_checked': len(reserved_groups),
            'source_identities_checked': len(reserved_identities),
            'project_labels_checked': len(labels),
            'heldout_separation': 'unknown',
        },
        'interpretation': {
            'origin': 'caller_supplied_interpretation',
            'phase_label': phase_label,
            'phase_evidence_span_id': parsed_spans[phase_id]['output_id'],
            'model_claims': model_claims,
            'effort_claims': effort_claims,
            'claim_review_status': 'unreviewed',
        },
        'state': {
            'provider_entitlement': 'unknown',
            'executor_effort': 'unknown',
            'execution': 'unknown',
            'outcome': 'unknown',
            'total_agent_count': None,
            'review_status': 'unreviewed',
            'promotion_eligible': False,
        },
        'screen_policy_sha256': policy_sha,
        'config_sha256': sha(canonical(config)),
        'extensions': safe_extensions,
    }
    if len(canonical(observation)) > config['max_observation_bytes']:
        raise ValueError('observation exceeds configured byte limit')
    # The reservation is pinned through in-memory materialization. This does not
    # lock it after return; the owner-only manifest must not be edited concurrently.
    validate_observation(observation, config, screen_policy_path)
    final_message = source_bound_snapshot.verify_selected_source(
        record, max_session_bytes=config['max_source_prefix_bytes'],
        max_line_bytes=config['max_line_bytes'])
    if (final_message.encode('utf-8') != message_bytes
            or sha(final_message.encode('utf-8')) != record['text_sha256']):
        raise ValueError('selected source changed during observation materialization')
    _recheck_reserved(reserved, config, screen_policy_path=screen_policy_path)
    return observation


def validate_observation(value: object, config: dict, screen_policy_path: Path) -> None:
    """Boundedly validate the isolated observation schema without reading sources."""
    validate_config(config)
    expected_top_fields = {'schema_version', 'kind', 'source', 'reservation', 'interpretation',
                           'state', 'screen_policy_sha256', 'config_sha256', 'extensions'}
    if (not isinstance(value, dict) or set(value) != expected_top_fields
            or value.get('schema_version') != SCHEMA or value.get('kind') != 'user_requested'):
        raise ValueError('invalid user-requested observation schema')
    source, reservation = value.get('source'), value.get('reservation')
    interpretation, state = value.get('interpretation'), value.get('state')
    source_fields = {'role', 'project_label_sha256', 'source_group_sha256', 'source_file_sha256_at_inventory',
                     'source_prefix_bytes', 'source_prefix_sha256', 'source_line', 'source_line_sha256',
                     'user_text_sha256', 'user_text_utf8_bytes', 'spans'}
    if not isinstance(source, dict) or set(source) != source_fields or source.get('role') != 'user':
        raise ValueError('observation must have a user source')
    for key in ('project_label_sha256', 'source_group_sha256', 'source_file_sha256_at_inventory',
                'source_prefix_sha256', 'source_line_sha256', 'user_text_sha256'):
        if not _valid_sha(source.get(key)):
            raise ValueError('invalid source identity digest')
    if (type(source.get('source_prefix_bytes')) is not int or source['source_prefix_bytes'] < 1
            or source['source_prefix_bytes'] > config['max_source_prefix_bytes']
            or type(source.get('source_line')) is not int or source['source_line'] < 1
            or type(source.get('user_text_utf8_bytes')) is not int or source['user_text_utf8_bytes'] < 1
            or source['user_text_utf8_bytes'] > config['max_user_message_bytes']):
        raise ValueError('invalid source identity bounds')
    spans = source.get('spans')
    if not isinstance(spans, list) or not 1 <= len(spans) <= config['max_span_count']:
        raise ValueError('invalid observation spans')
    span_map = {}
    for index, span in enumerate(spans, start=1):
        if not isinstance(span, dict) or set(span) != {'id', 'field', 'start_byte', 'end_byte', 'sha256'}:
            raise ValueError('invalid observation span')
        if (span['id'] != f'span-{index}'
                or span['id'] in span_map or not isinstance(span['field'], str) or span['field'] not in SPAN_FIELDS
                or type(span['start_byte']) is not int or type(span['end_byte']) is not int
                or span['start_byte'] < 0 or span['end_byte'] <= span['start_byte']
                or span['end_byte'] > source['user_text_utf8_bytes']
                or span['end_byte'] - span['start_byte'] > config['max_span_bytes']
                or not _valid_sha(span['sha256'])):
            raise ValueError('invalid observation span bounds or digest')
        span_map[span['id']] = span
    reservation_fields = {'exclusion_checked', 'manifest_sha256', 'coverage', 'groups_checked',
                          'source_identities_checked',
                          'project_labels_checked', 'heldout_separation'}
    if (not isinstance(reservation, dict) or set(reservation) != reservation_fields
            or reservation.get('exclusion_checked') is not True
            or reservation.get('coverage') not in ('unknown', 'partial')
            or reservation.get('heldout_separation') != 'unknown'
            or type(reservation.get('groups_checked')) is not int or reservation['groups_checked'] != 0
            or type(reservation.get('source_identities_checked')) is not int
            or reservation['source_identities_checked'] < 0
            or reservation['source_identities_checked'] > config['max_reserved_manifest_bytes'] // 200
            or type(reservation.get('project_labels_checked')) is not int or reservation['project_labels_checked'] < 0
            or reservation['project_labels_checked'] > config['max_reserved_project_labels']
            or (reservation.get('manifest_sha256') is not None
                and not _valid_sha(reservation['manifest_sha256']))
            or (reservation.get('manifest_sha256') is None
                and (reservation['coverage'] != 'unknown' or reservation['source_identities_checked'] != 0))):
        raise ValueError('invalid reserved-set claims')
    interpretation_fields = {'origin', 'phase_label', 'phase_evidence_span_id',
                             'model_claims', 'effort_claims', 'claim_review_status'}
    if (not isinstance(interpretation, dict) or set(interpretation) != interpretation_fields
            or interpretation.get('origin') != 'caller_supplied_interpretation'
            or interpretation.get('claim_review_status') != 'unreviewed'
            or not isinstance(interpretation.get('phase_label'), str)
            or interpretation['phase_label'] not in config['phase_labels']):
        raise ValueError('invalid caller interpretation')
    phase_id = interpretation.get('phase_evidence_span_id')
    if (not isinstance(phase_id, str) or phase_id not in span_map
            or span_map[phase_id]['field'] != 'phase'):
        raise ValueError('missing phase evidence')
    referenced = {phase_id}
    for key, field in (('model_claims', 'model'), ('effort_claims', 'effort')):
        claims = interpretation.get(key)
        if not isinstance(claims, dict):
            raise ValueError(f'invalid {field} claim collection')
        if claims == {'status': 'unknown'}:
            continue
        if (set(claims) != {'status', 'items', 'review_status'}
                or claims.get('status') != 'unreviewed_source_claims'
                or claims.get('review_status') != 'unreviewed'
                or not isinstance(claims.get('items'), list)
                or len(claims['items']) > config['max_claims_per_field']
                or not claims['items']):
            raise ValueError(f'invalid unreviewed {field} claims')
        for claim in claims['items']:
            if (not isinstance(claim, dict)
                    or set(claim) != {'status', 'polarity', 'evidence_span_id', 'text_sha256'}
                    or claim.get('status') != 'unreviewed_source_claim'
                    or claim.get('polarity') not in config['claim_polarities']
                    or not isinstance(claim.get('evidence_span_id'), str)
                    or claim['evidence_span_id'] not in span_map
                    or span_map[claim['evidence_span_id']]['field'] != field
                    or claim.get('text_sha256') != span_map[claim['evidence_span_id']]['sha256']
                    or claim['evidence_span_id'] in referenced):
                raise ValueError(f'invalid unreviewed {field} claim')
            referenced.add(claim['evidence_span_id'])
    if referenced != set(span_map):
        raise ValueError('observation has unassigned source spans')
    expected_state = {
        'provider_entitlement': 'unknown', 'executor_effort': 'unknown', 'execution': 'unknown',
        'outcome': 'unknown', 'total_agent_count': None, 'review_status': 'unreviewed',
        'promotion_eligible': False,
    }
    if not isinstance(state, dict) or canonical(state) != canonical(expected_state):
        raise ValueError('user-requested observation state must remain unknown and unreviewed')
    policy, rules, policy_sha = screen_private_corpus.load_policy(screen_policy_path, 1024 * 1024)
    if (value.get('screen_policy_sha256') != policy_sha
            or value.get('config_sha256') != sha(canonical(config))):
        raise ValueError('observation configuration provenance does not match')
    if not isinstance(value.get('extensions'), dict):
        raise ValueError('observation extensions must be an object')
    baseline, baseline_rules, _ = screen_private_corpus.load_policy(
        source_bound_snapshot._default_screen_policy_path(), 1024 * 1024)
    policies = [(baseline, baseline_rules), (policy, rules)]
    labels_to_screen = [interpretation['phase_label']]
    for key in ('model_claims', 'effort_claims'):
        labels_to_screen.extend(item['polarity'] for item in interpretation[key].get('items', []))
    for label in labels_to_screen:
        source_bound_snapshot._require_screened_text(label, policies)
    for current_policy, current_rules in policies:
        clean_extensions = _safe_extensions(value['extensions'], current_rules, current_policy['limits'], config,
                                            require_already_screened=True)
        if clean_extensions != value['extensions']:
            raise ValueError('observation extensions are not in canonical screened form')
    try:
        observation_bytes = len(canonical(value))
    except (RecursionError, ValueError, TypeError, OverflowError):
        raise ValueError('observation cannot be encoded within the configured schema') from None
    if observation_bytes > config['max_observation_bytes']:
        raise ValueError('observation exceeds configured byte limit')


def consume_observation(value: object, config: dict, screen_policy_path: Path) -> dict:
    """Return an isolated unreviewed-claim summary; never emit labels or weights."""
    validate_observation(value, config, screen_policy_path)
    return {
        'schema_version': SUMMARY_SCHEMA,
        'kind': 'user_requested',
        'interpretation': value['interpretation'],
        'source_evidence': {
            'user_text_sha256': value['source']['user_text_sha256'],
            'source_prefix_sha256': value['source']['source_prefix_sha256'],
            'source_line_sha256': value['source']['source_line_sha256'],
            'spans': value['source']['spans'],
        },
        'reservation': value['reservation'],
        'state': value['state'],
        'extensions': value['extensions'],
    }
