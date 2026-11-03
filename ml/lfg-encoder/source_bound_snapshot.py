#!/usr/bin/env python3
"""Validate a bounded, private snapshot of a selected Codex request and provider list.

This module records provenance claims only.  It does not infer task outcomes,
subscription entitlement, historical effort, or an evaluation split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

import inspect_codex_inventory
import screen_private_corpus
import project_training_context

SCHEMA = 'lfg.source-bound-model-context.v1'
MAX_CACHE_BYTES = 8 * 1024 * 1024
MAX_MODELS = 2048
MAX_EXTENSION_NODES = 20_000
MAX_EXTENSION_DEPTH = 48
MAX_METADATA_BYTES = 64 * 1024
MAX_RESERVED_MANIFEST_BYTES = 1024 * 1024
SENSITIVE_KEY = re.compile(
    r'(?:^|[_-])(?:secret|password|credential|access[_-]?token|refresh[_-]?token|'
    r'auth[_-]?token|session[_-]?token|id[_-]?token|bearer[_-]?token|jwt|'
    r'api[_-]?key|access[_-]?key)(?:$|[_-])', re.I)


def _project_key(value: str) -> str:
    return ''.join(character for character in value.casefold() if character.isalnum())


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: object) -> bytes:
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
        raise ValueError('invalid JSON numeric constant')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def _default_screen_policy_path() -> Path:
    return Path(__file__).with_name('private-corpus-screen.example.json')


def _safe_extensions(value: object, rules, limits: dict) -> object:
    """Validate extensions and screen every persisted string value and key."""
    count = 0

    def visit(current, depth):
        nonlocal count
        count += 1
        if depth > MAX_EXTENSION_DEPTH or count > MAX_EXTENSION_NODES:
            raise ValueError('provider extension limits exceeded')
        if isinstance(current, dict):
            result = {}
            for key, item in current.items():
                if not isinstance(key, str) or not key or len(key.encode('utf-8')) > 256:
                    raise ValueError('invalid provider extension key')
                if SENSITIVE_KEY.search(key):
                    raise ValueError('sensitive provider extension key')
                screened_key, _ = screen_private_corpus._screen(key, rules, limits)
                if screened_key != key or screened_key in result:
                    raise ValueError('provider extension key requires redaction or collides')
                result[key] = visit(item, depth + 1)
            return result
        if isinstance(current, list):
            return [visit(item, depth + 1) for item in current]
        if isinstance(current, str):
            if len(current.encode('utf-8')) > MAX_METADATA_BYTES:
                raise ValueError('provider metadata exceeds limit')
            if not current:
                return current
            screened, _ = screen_private_corpus._screen(current, rules, limits)
            return screened
        if current is None or type(current) in (bool, int, float):
            if isinstance(current, float) and not __import__('math').isfinite(current):
                raise ValueError('invalid provider extension number')
            return current
        raise ValueError('invalid provider extension value')

    return visit(value, 0)


def _metadata(value: object, name: str, *, required: bool = False, rules=None,
              limits: dict | None = None, identity: bool = False) -> str | None:
    if value is None and not required:
        return None
    if (not isinstance(value, str) or (required and not value.strip())
            or len(value.encode('utf-8')) > MAX_METADATA_BYTES
            or any(ord(char) < 32 and char not in '\t\r\n' for char in value)):
        raise ValueError(f'invalid provider {name}')
    if value and rules is not None and limits is not None:
        screened, _ = screen_private_corpus._screen(value, rules, limits)
        if identity and screened != value:
            raise ValueError(f'provider {name} requires redaction')
        return screened
    return value


def read_provider_cache(path: Path, *, max_bytes: int = MAX_CACHE_BYTES,
                        screen_policy_path: Path | None = None) -> tuple[dict, bytes, dict]:
    """Read stable cache bytes and retain the provider's listed metadata as inert claims."""
    if type(max_bytes) is not int or max_bytes < 1 or max_bytes > MAX_CACHE_BYTES:
        raise ValueError('invalid provider cache byte limit')
    screen_policy_path = screen_policy_path or _default_screen_policy_path()
    policy, rules, policy_sha = screen_private_corpus.load_policy(screen_policy_path, 1024 * 1024)
    before = path.stat()
    if not path.is_file() or before.st_size > max_bytes:
        raise ValueError('provider cache unavailable or too large')
    with path.open('rb') as stream:
        raw = stream.read(max_bytes + 1)
    after = path.stat()
    if (len(raw) > max_bytes or (before.st_size, before.st_mtime_ns, before.st_ino)
            != (after.st_size, after.st_mtime_ns, after.st_ino) or len(raw) != after.st_size):
        raise ValueError('provider cache changed during read')
    try:
        cache = _strict_json(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError('invalid provider cache JSON') from None
    if not isinstance(cache, dict) or not isinstance(cache.get('models'), list) or len(cache['models']) > MAX_MODELS:
        raise ValueError('invalid provider cache structure')
    fetched_at = _metadata(cache.get('fetched_at'), 'fetched_at', rules=rules, limits=policy['limits'])
    listed = []
    seen = set()
    for model in cache['models']:
        if not isinstance(model, dict):
            raise ValueError('invalid provider model record')
        if model.get('visibility') != 'list':
            continue
        slug = _metadata(model.get('slug'), 'model ID', required=True, rules=rules,
                         limits=policy['limits'], identity=True)
        if slug in seen:
            raise ValueError('duplicate provider model ID')
        seen.add(slug)
        label = _metadata(model.get('display_name'), 'display name', rules=rules, limits=policy['limits'])
        description = _metadata(model.get('description'), 'description', rules=rules, limits=policy['limits'])
        priority = model.get('priority')
        if priority is not None and (type(priority) is not int or priority < 0):
            raise ValueError('invalid provider priority')
        levels = model.get('supported_reasoning_levels')
        if levels is not None and not isinstance(levels, list):
            raise ValueError('invalid provider effort list')
        efforts = []
        effort_extensions = []
        for level in levels or []:
            if not isinstance(level, dict):
                raise ValueError('invalid provider effort record')
            effort = _metadata(level.get('effort'), 'effort', rules=rules,
                               limits=policy['limits'], identity=True)
            if effort is not None:
                efforts.append(effort)
            extra = {key: value for key, value in level.items() if key != 'effort'}
            extra = _safe_extensions(extra, rules, policy['limits'])
            effort_extensions.append(extra)
        extensions = {key: value for key, value in model.items()
                      if key not in {'slug', 'display_name', 'description', 'priority',
                                     'supported_reasoning_levels', 'visibility'}}
        extensions = _safe_extensions(extensions, rules, policy['limits'])
        listed.append({'model_id': slug, 'display_name': label or slug,
                       'description': description or '', 'priority': priority,
                       'provider_supported_efforts': efforts, 'effort_extensions': effort_extensions,
                       'extensions': extensions})
    if not listed:
        raise ValueError('provider cache contains no listed models')
    root_extensions = {key: value for key, value in cache.items() if key not in {'models', 'fetched_at'}}
    root_extensions = _safe_extensions(root_extensions, rules, policy['limits'])
    metadata = {'fetched_at': fetched_at, 'models': listed, 'extensions': root_extensions,
                'cache_sha256': sha(raw), 'cache_bytes': len(raw),
                'cache_mtime_utc': datetime.fromtimestamp(after.st_mtime, timezone.utc).isoformat(),
                'screen_policy_sha256': policy_sha,
                'screen_policy_locator': str(screen_policy_path.resolve())}
    return metadata, raw, after


def source_group_id(record: dict) -> str:
    """Project-label-independent session key without exposing the source path."""
    path = os.path.normcase(os.path.realpath(os.path.abspath(record['source_file'])))
    return sha(('codex-session-path-v1\0' + path).encode('utf-8'))


def read_reserved_session_manifest(path: Path) -> tuple[list[str], str, str]:
    """Read an owner-only POSIX manifest; completeness remains unknown.

    Windows direct reads are disabled because this module does not verify
    Windows reparse points or effective ACLs. The CLI refuses Windows before
    opening any inputs; this guard gives direct callers the same boundary for
    this private manifest.
    """
    if os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW'):
        raise ValueError('reserved-session manifest reads require POSIX no-follow file handles')
    directory_fd = -1
    file_fd = -1
    name = ''
    try:
        directory_fd, name, _actual_parent = _open_private_parent(path)
        _verify_parent_binding(path, directory_fd, name)
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) & 0o077
                or before.st_size > MAX_RESERVED_MANIFEST_BYTES):
            raise ValueError('reserved-session manifest unavailable or not owner-only')
        chunks = []
        remaining = MAX_RESERVED_MANIFEST_BYTES + 1
        while remaining:
            chunk = os.read(file_fd, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b''.join(chunks)
        after = os.fstat(file_fd)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (len(raw) > MAX_RESERVED_MANIFEST_BYTES or len(raw) != after.st_size
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino)
                or not stat.S_ISREG(named.st_mode) or named.st_nlink != 1):
            raise ValueError('reserved-session manifest changed during read')
        _verify_parent_binding(path, directory_fd, name)
    except OSError:
        raise ValueError('reserved-session manifest unavailable or changed during read') from None
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if directory_fd >= 0:
            os.close(directory_fd)
    try:
        value = _strict_json(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError('invalid reserved-session manifest') from None
    if (not isinstance(value, dict) or value.get('schema_version') != 'lfg.reserved-session-groups.v1'
            or not isinstance(value.get('groups'), list)
            or value.get('coverage') not in ('unknown', 'partial')):
        raise ValueError('invalid reserved-session manifest')
    groups = value['groups']
    if (len(groups) > 100_000 or any(not isinstance(item, str) or len(item) != 64
            or any(char not in '0123456789abcdef' for char in item) for item in groups)
            or len(set(groups)) != len(groups)):
        raise ValueError('invalid reserved-session identity')
    return groups, sha(raw), value['coverage']


def _recheck_reserved_manifest(locator: str | None, manifest_sha: str | None,
                               groups: list[str], coverage: str) -> None:
    if manifest_sha is None:
        if locator is not None:
            raise ValueError('unexpected reserved-session manifest locator')
        return
    if not isinstance(locator, str) or not locator:
        raise ValueError('reserved-session manifest identity is incomplete')
    current_groups, current_sha, current_coverage = read_reserved_session_manifest(Path(locator))
    if current_sha != manifest_sha or current_groups != groups or current_coverage != coverage:
        raise ValueError('reserved-session manifest changed')


def verify_selected_source(record: dict, *, max_session_bytes: int, max_line_bytes: int) -> str:
    """Verify exactly the recorded prefix and line while allowing append-only suffix growth.

    The source is a live session log. Its whole-file digest can change as later
    events append; those bytes are irrelevant to the selected line. The exact
    prefix digest and selected line digest must still match.
    """
    inspect_codex_inventory.verify_record(record)
    if (type(max_session_bytes) is not int or max_session_bytes < 1
            or type(max_line_bytes) is not int or max_line_bytes < 1
            or record['source_prefix_bytes'] > max_session_bytes):
        raise ValueError('invalid source verification bounds')
    path = Path(record['source_file'])
    before = path.stat()
    if before.st_size < record['source_prefix_bytes']:
        raise ValueError('source prefix is unavailable')
    digest = hashlib.sha256()
    total = 0
    line = b''
    with path.open('rb') as stream:
        for number in range(1, record['source_line'] + 1):
            line = stream.readline(max_line_bytes + 1)
            if not line or len(line) > max_line_bytes:
                raise ValueError('selected source line is unavailable or too large')
            total += len(line)
            if total > max_session_bytes:
                raise ValueError('selected source prefix exceeds limit')
            digest.update(line)
            if number == record['source_line']:
                selected = line.rstrip(b'\r\n')
    if (total != record['source_prefix_bytes'] or digest.hexdigest() != record['source_prefix_sha256']
            or sha(selected) != record['source_line_sha256']):
        raise ValueError('selected source prefix or line changed')
    after = path.stat()
    if before.st_ino != after.st_ino or after.st_size < total:
        raise ValueError('source was replaced during verification')
    try:
        text = inspect_codex_inventory.message_text(selected, record)
    except (ValueError, KeyError, TypeError, UnicodeError):
        raise ValueError('selected source is not a verified user-text message') from None
    return text


def build_snapshot(record: dict, cache_path: Path, *, split_policy: str,
                   known_reserved_group_sha256: tuple[str, ...] | list[str] = (),
                   known_reserved_manifest_sha256: str | None = None,
                   known_reserved_manifest_locator: str | None = None,
                   known_reserved_groups_coverage: str = 'unknown',
                   known_reserved_project_labels: tuple[str, ...] | list[str] = (),
                   screen_policy_path: Path | None = None,
                   max_session_bytes: int = 512 * 1024 * 1024,
                   max_line_bytes: int = 16 * 1024 * 1024,
                   max_cache_bytes: int = MAX_CACHE_BYTES,
                   observed_at: str | None = None) -> dict:
    """Reverify the exact session line/prefix and cache; keep access/split unknowns explicit.

    ``exploratory_train`` is an explicit collection assignment, not a claim of
    holdout independence. ``reserved_train`` is intentionally unavailable until
    a separately verifiable split attestation is wired into this contract.
    """
    if split_policy not in {'exploratory_train', 'reserved_train'}:
        raise ValueError('split policy must be explicit')
    if split_policy == 'reserved_train':
        raise ValueError('reserved train split needs verifiable split evidence')
    if not isinstance(known_reserved_group_sha256, (tuple, list)):
        raise ValueError('invalid reserved-group list')
    if any(not isinstance(item, str) or len(item) != 64
           or any(char not in '0123456789abcdef' for char in item)
           for item in known_reserved_group_sha256):
        raise ValueError('invalid reserved-group identity')
    if (known_reserved_manifest_sha256 is not None and
            (not isinstance(known_reserved_manifest_sha256, str)
             or len(known_reserved_manifest_sha256) != 64
             or any(char not in '0123456789abcdef' for char in known_reserved_manifest_sha256))):
        raise ValueError('invalid reserved-session manifest digest')
    if ((known_reserved_manifest_sha256 is None) != (known_reserved_manifest_locator is None)
            or (known_reserved_manifest_locator is not None
                and (not isinstance(known_reserved_manifest_locator, str) or not known_reserved_manifest_locator))):
        raise ValueError('reserved-session manifest identity is incomplete')
    if known_reserved_groups_coverage not in ('unknown', 'partial'):
        raise ValueError('reserved-session coverage cannot be asserted complete')
    _recheck_reserved_manifest(known_reserved_manifest_locator, known_reserved_manifest_sha256,
                               list(known_reserved_group_sha256), known_reserved_groups_coverage)
    if (not isinstance(known_reserved_project_labels, (tuple, list))
            or any(not isinstance(item, str) or not item.strip() for item in known_reserved_project_labels)):
        raise ValueError('invalid reserved-project list')
    inspect_codex_inventory.verify_record(record)
    group_id = source_group_id(record)
    reserved_projects = {_project_key(item) for item in known_reserved_project_labels}
    if group_id in set(known_reserved_group_sha256) or _project_key(record['project']) in reserved_projects:
        # This check intentionally precedes opening the selected session file.
        raise ValueError('selected source belongs to a known reserved group')
    try:
        selected_text = verify_selected_source(record, max_session_bytes=max_session_bytes,
                                               max_line_bytes=max_line_bytes)
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
        raise ValueError('selected Codex source identity no longer verifies') from None
    if not selected_text or sha(selected_text.encode('utf-8')) != record['text_sha256']:
        raise ValueError('selected Codex user text does not verify')
    cache, _raw, _stat = read_provider_cache(cache_path, max_bytes=max_cache_bytes,
                                              screen_policy_path=screen_policy_path)
    if observed_at is None:
        observed_at = datetime.now(timezone.utc).isoformat()
    try:
        stamp = datetime.fromisoformat(observed_at.replace('Z', '+00:00'))
    except (TypeError, ValueError):
        raise ValueError('invalid observation timestamp') from None
    if stamp.tzinfo is None:
        raise ValueError('observation timestamp must include timezone')
    return {
        'schema_version': SCHEMA,
        'observed_at': stamp.astimezone(timezone.utc).isoformat(),
        'source': {
            'kind': 'codex-session-user-text',
            'project': record['project'],
            'project_association': 'inventory_label_unverified',
            'source_locator_sha256': sha(os.path.normcase(os.path.realpath(
                os.path.abspath(record['source_file']))).encode('utf-8')),
            'source_group_sha256': group_id,
            'selected_line': record['source_line'],
            'source_prefix_bytes': record['source_prefix_bytes'],
            'source_prefix_sha256': record['source_prefix_sha256'],
            'source_line_sha256': record['source_line_sha256'],
            'user_text_sha256': record['text_sha256'],
            'user_text_utf8_bytes': len(selected_text.encode('utf-8')),
            'source_file_sha256_at_inventory': record['source_file_sha256'],
            'current_whole_file_sha256': 'unknown',
            'source_locator': record['source_file'],
            'inventory_identity': {key: record[key] for key in (
                'schema_version', 'project', 'source_file', 'source_line', 'source_line_sha256',
                'source_prefix_sha256', 'source_prefix_bytes', 'text_sha256', 'source_file_sha256')},
            'verified': True,
        },
        'split': {'assignment': 'train', 'policy': split_policy,
                  'heldout_separation': 'unknown',
                  'known_reserved_group_exclusion': 'checked',
                  'known_reserved_groups_checked': len(known_reserved_group_sha256),
                  'known_reserved_group_sha256': list(known_reserved_group_sha256),
                  'known_reserved_manifest_sha256': known_reserved_manifest_sha256,
                  'known_reserved_groups_coverage': known_reserved_groups_coverage,
                  'known_reserved_project_labels': list(known_reserved_project_labels)},
        'provider_list': {**cache, 'source_locator': str(cache_path.resolve())},
        'access_claims': {
            'provider_listing_is_entitlement': False,
            'subscription_entitlement': 'unknown',
            'historical_session_effort': 'unknown',
            'historical_model_access': 'unknown',
        },
        'training_claims': {'task_outcome': 'unknown', 'teacher_proposal': 'none',
                            'review_status': 'unreviewed', 'promotion_eligible': False},
        'reserved_session_manifest_locator': known_reserved_manifest_locator,
        'extensions': {},
    }


def revalidate_snapshot(value: dict, *, max_session_bytes: int = 512 * 1024 * 1024,
                        max_line_bytes: int = 16 * 1024 * 1024,
                        max_cache_bytes: int = MAX_CACHE_BYTES) -> None:
    """Recheck the source line/prefix and exact provider-cache bytes before collection."""
    if not isinstance(value, dict) or value.get('schema_version') != SCHEMA:
        raise ValueError('invalid source-bound snapshot')
    source, provider, split = value.get('source'), value.get('provider_list'), value.get('split')
    if not isinstance(source, dict) or not isinstance(provider, dict) or not isinstance(split, dict):
        raise ValueError('invalid source-bound snapshot')
    record = source.get('inventory_identity')
    source_path, cache_path = source.get('source_locator'), provider.get('source_locator')
    if (not isinstance(record, dict) or not isinstance(source_path, str) or record.get('source_file') != source_path
            or not isinstance(cache_path, str) or split.get('policy') != 'exploratory_train'):
        raise ValueError('snapshot source identity is incomplete')
    manifest_sha = split.get('known_reserved_manifest_sha256')
    manifest_locator = value.get('reserved_session_manifest_locator')
    policy_path = Path(provider.get('screen_policy_locator') or _default_screen_policy_path())
    current = build_snapshot(record, Path(cache_path), split_policy='exploratory_train',
                             known_reserved_group_sha256=split.get('known_reserved_group_sha256', []),
                             known_reserved_manifest_sha256=split.get('known_reserved_manifest_sha256'),
                             known_reserved_manifest_locator=manifest_locator,
                             known_reserved_groups_coverage=split.get('known_reserved_groups_coverage', 'unknown'),
                             known_reserved_project_labels=split.get('known_reserved_project_labels', []),
                             screen_policy_path=policy_path,
                             max_session_bytes=max_session_bytes, max_line_bytes=max_line_bytes,
                             max_cache_bytes=max_cache_bytes, observed_at=value.get('observed_at'))
    # The byte digests alone do not bind the separately serialized projection:
    # compare all reconstructed source, provider, split, and access fields so a
    # caller cannot edit model rows or claims while retaining the original hash.
    if (current['source'] != source or current['provider_list'] != provider
            or current['split'] != split or current['access_claims'] != value.get('access_claims')
            or current['training_claims'] != value.get('training_claims')):
        raise ValueError('snapshot source or provider cache changed')


def render_bound_context(value: dict, request: str, *, max_context_bytes: int,
                         policy_path: Path) -> tuple[str, str]:
    """Render only the verified current listing and bind it to the exact source prompt."""
    if not isinstance(request, str) or not request:
        raise ValueError('request must be non-empty text')
    source, provider = value.get('source'), value.get('provider_list')
    if not isinstance(source, dict) or not isinstance(provider, dict):
        raise ValueError('invalid source-bound snapshot')
    if sha(request.encode('utf-8')) != source.get('user_text_sha256'):
        raise ValueError('request does not match verified source text')
    models = provider.get('models')
    if not isinstance(models, list) or not models:
        raise ValueError('provider listing is unavailable')
    request_terms = set(re.findall(r'[a-z0-9]+', request.casefold()))
    model_aliases = []
    alias_counts: dict[str, int] = {}
    for model in models:
        if not isinstance(model, dict):
            raise ValueError('invalid provider listing')
        aliases = {term for term in re.findall(
            r'[a-z0-9]+', (str(model.get('model_id', '')) + ' ' + str(model.get('display_name', ''))).casefold())
                   if len(term) >= 4 and not term.isdigit()}
        model_aliases.append(aliases)
        for alias in aliases:
            alias_counts[alias] = alias_counts.get(alias, 0) + 1
    rendered_models = []
    for model, aliases in zip(models, model_aliases):
        # Keep exact, relatively specific family mentions, including ambiguous
        # variants of that family. Shared generic terms across the catalog do
        # not expand the candidate set.
        if any(alias_counts[alias] <= 2 for alias in aliases & request_terms):
            rendered_models.append({key: model[key] for key in
                                    ('model_id', 'display_name', 'description', 'priority',
                                     'provider_supported_efforts') if key in model})
    if not rendered_models:
        raise ValueError('request does not explicitly mention a listed provider model')
    fetched = provider.get('fetched_at') or 'unknown'
    header = (f'Provider-list observation v1; observed={value.get("observed_at")}; fetched={fetched}; '
              f'cache_sha256={provider.get("cache_sha256")}; entitlement=unknown; historical_effort=unknown; '
              'lexically mentioned model rows are candidates, not recommendations; heldout_separation=unknown')
    rows = [json.dumps([model.get('model_id'), model.get('display_name'), model.get('description', ''),
                        model.get('priority'), model.get('provider_supported_efforts', [])],
                       ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            for model in rendered_models]
    rendered = header + '\n' + '\n'.join(rows)
    policy, rules, _policy_sha = screen_private_corpus.load_policy(policy_path, 1024 * 1024)
    rendered, _redaction_counts = screen_private_corpus._screen(rendered, rules, policy['limits'])
    if type(max_context_bytes) is not int or max_context_bytes < 1 or len(rendered.encode('utf-8')) > max_context_bytes:
        raise ValueError('source-bound context exceeds configured limit')
    return rendered, sha(rendered.encode('utf-8'))


def prepare_single_request(value: dict, *, policy_path: Path, max_request_bytes: int,
                           max_context_bytes: int) -> tuple[dict, dict, dict, bytes, bytes, bytes]:
    """Create one directly source-bound, exploratory train request and context file."""
    revalidate_snapshot(value)
    source = value['source']
    record = source['inventory_identity']
    request = verify_selected_source(record, max_session_bytes=512 * 1024 * 1024,
                                     max_line_bytes=16 * 1024 * 1024)
    if sha(request.encode('utf-8')) != source['user_text_sha256']:
        raise ValueError('source prompt changed before selection')
    _recheck_reserved_manifest(value.get('reserved_session_manifest_locator'),
                               value['split'].get('known_reserved_manifest_sha256'),
                               value['split'].get('known_reserved_group_sha256', []),
                               value['split'].get('known_reserved_groups_coverage', 'unknown'))
    policy, rules, policy_sha = screen_private_corpus.load_policy(policy_path, 1024 * 1024)
    if value.get('provider_list', {}).get('screen_policy_sha256') != policy_sha:
        raise ValueError('snapshot provider extensions use a different screening policy')
    context_text, context_sha = render_bound_context(value, request, max_context_bytes=max_context_bytes,
                                                     policy_path=policy_path)
    screened_request, redactions = screen_private_corpus._screen(request, rules, policy['limits'])
    if len(screened_request.encode('utf-8')) > max_request_bytes:
        raise ValueError('screened source request exceeds configured limit')
    rendered_teacher_request = project_training_context.render(
        context_text, screened_request, max_context_bytes=max_context_bytes, expected_sha256=context_sha)
    if len(rendered_teacher_request.encode('utf-8')) > max_context_bytes:
        raise ValueError('fully rendered teacher request exceeds configured context limit')
    snapshot_raw = json_bytes(value)
    snapshot_sha = sha(snapshot_raw)
    group = source['source_group_sha256']
    request_id = sha((group + '\0' + source['user_text_sha256'] + '\0' + snapshot_sha).encode('utf-8'))
    row = {
        'schema_version': 'lfg.source-bound-teacher-request.v1',
        'id': request_id,
        'project': source['project'],
        'project_group': group,
        'family': group,
        'text': screened_request,
        'screened_text_sha256': sha(screened_request.encode('utf-8')),
        'source_text_sha256_claim': source['user_text_sha256'],
        'candidate_file_sha256_claim': snapshot_sha,
        'source_bound_snapshot_sha256': snapshot_sha,
        'source_prefix_sha256_claim': source['source_prefix_sha256'],
        'source_line_sha256_claim': source['source_line_sha256'],
        'split': 'train',
        'split_policy': 'exploratory_train',
        'heldout_separation': 'unknown',
        'review_status': 'unreviewed',
        'outcome': 'unknown',
    }
    request_raw = (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                   + '\n').encode('utf-8')
    context_file_value = {
        'schema_version': 'lfg.project-training-context.v1',
        'source_bound_snapshot_sha256': snapshot_sha,
        'materialized_from_verified_snapshot': True,
        'projects': {source['project']: {
            'text': context_text,
            'sha256': context_sha,
            'source_bound_snapshot_sha256': snapshot_sha,
            'source_text_sha256_claim': source['user_text_sha256'],
            'provider_cache_sha256': value['provider_list']['cache_sha256'],
            'split_policy': 'exploratory_train',
            'heldout_separation': 'unknown',
        }},
    }
    context_raw = (json.dumps(context_file_value, sort_keys=True, ensure_ascii=False,
                              separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    manifest = {
        'schema_version': 'lfg.teacher-request-selection-result.v1',
        'selection_kind': 'direct_source_bound_single_request',
        'source_bound_snapshot_sha256': snapshot_sha,
        'selection_policy_sha256': policy_sha,
        'requests_sha256': sha(request_raw),
        'requests': 1,
        'projects': {source['project']: 1},
        'skipped': {},
        'redaction_counts': dict(sorted(redactions.items())),
        'validation_opened': False,
        'test_opened': False,
        'heldout_separation': 'unknown',
        'human_reviewed': False,
        'promotion_eligible': False,
    }
    manifest_raw = (json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                               separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    projection, bundle = stage_snapshot_projection(value, context_text)
    bundle.update({'requests_sha256': sha(request_raw),
                   'selection_manifest_sha256': sha(manifest_raw),
                   'project_context_sha256': sha(context_raw),
                   'screen_policy_sha256': policy_sha})
    # Re-read after all rendering and serialization, immediately before
    # returning the staged bytes. This narrows (but cannot eliminate) a race
    # with a concurrent manifest edit after this final check.
    _recheck_reserved_manifest(value.get('reserved_session_manifest_locator'),
                               value['split'].get('known_reserved_manifest_sha256'),
                               value['split'].get('known_reserved_group_sha256', []),
                               value['split'].get('known_reserved_groups_coverage', 'unknown'))
    return (manifest, row, context_file_value, request_raw, manifest_raw, context_raw,
            json_bytes(projection), json_bytes(bundle))


def validate_staged_artifacts(snapshot_raw: bytes, request_raw: bytes,
                              manifest_raw: bytes, context_raw: bytes,
                              bundle_raw: bytes | None = None,
                              policy_raw: bytes | None = None) -> dict:
    """Validate copied private artifacts without reopening heldout or original host files."""
    try:
        source_snapshot = _strict_json(snapshot_raw)
        request = _strict_json(request_raw.rstrip(b'\r\n'))
        manifest = _strict_json(manifest_raw)
        context_file = _strict_json(context_raw)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError('invalid staged source-bound artifacts') from None
    if (not isinstance(source_snapshot, dict) or source_snapshot.get('schema_version') != SCHEMA
            or not isinstance(request, dict) or not isinstance(manifest, dict)
            or not isinstance(context_file, dict)):
        raise ValueError('invalid staged source-bound artifacts')
    snapshot_copy_sha = sha(snapshot_raw)
    materialization = source_snapshot.get('materialization', {})
    snapshot_sha = (materialization.get('origin_snapshot_sha256')
                    if isinstance(materialization, dict) else None) or snapshot_copy_sha
    bundle = None
    if bundle_raw is not None:
        try:
            bundle = _strict_json(bundle_raw)
        except (ValueError, UnicodeError, RecursionError):
            raise ValueError('invalid staged bundle manifest') from None
        if (not isinstance(bundle, dict) or bundle.get('schema_version') != 'lfg.source-bound-staging-bundle.v1'
                or bundle.get('origin_snapshot_sha256') != snapshot_sha
                or bundle.get('staged_snapshot_sha256') != snapshot_copy_sha
                or bundle.get('requests_sha256') != sha(request_raw)
                or bundle.get('selection_manifest_sha256') != sha(manifest_raw)
                or bundle.get('project_context_sha256') != sha(context_raw)):
            raise ValueError('staged bundle digest mismatch')
        if policy_raw is not None and bundle.get('screen_policy_sha256') != sha(policy_raw):
            raise ValueError('staged screening policy digest mismatch')
    source, split = source_snapshot.get('source'), source_snapshot.get('split')
    provider = source_snapshot.get('provider_list')
    if (not isinstance(source, dict) or not isinstance(split, dict) or not isinstance(provider, dict)
            or split.get('assignment') != 'train' or split.get('policy') != 'exploratory_train'
            or split.get('heldout_separation') != 'unknown' or source.get('verified') is not True
            or source.get('project_association') != 'inventory_label_unverified'
            or not isinstance(source.get('source_prefix_sha256'), str)
            or not isinstance(source.get('source_line_sha256'), str)
            or not isinstance(source.get('user_text_sha256'), str)
            or not isinstance(provider.get('cache_sha256'), str)
            or not isinstance(provider.get('screen_policy_sha256'), str)):
        raise ValueError('staged source provenance is incomplete')
    if source_snapshot.get('training_claims') != {
            'task_outcome': 'unknown', 'teacher_proposal': 'none',
            'review_status': 'unreviewed', 'promotion_eligible': False}:
        raise ValueError('staged training claims must remain unknown and unreviewed')
    if isinstance(materialization, dict) and materialization.get('kind') == 'origin-verified-private-projection':
        if (materialization.get('source_and_cache_verified_on_origin_host') is not True
                or materialization.get('full_source_paths_transferred') is not False
                or materialization.get('provider_extension_payloads_transferred') is not False):
            raise ValueError('staged projection provenance is invalid')
    if policy_raw is not None and provider.get('screen_policy_sha256') != sha(policy_raw):
        raise ValueError('staged provider screening policy digest mismatch')
    if (manifest.get('schema_version') != 'lfg.teacher-request-selection-result.v1'
            or manifest.get('requests') != 1 or manifest.get('test_opened') is not False
            or manifest.get('validation_opened') is not False or manifest.get('heldout_separation') != 'unknown'
            or manifest.get('human_reviewed') is not False
            or manifest.get('promotion_eligible') is not False
            or manifest.get('source_bound_snapshot_sha256') != snapshot_sha
            or manifest.get('requests_sha256') != sha(request_raw)):
        raise ValueError('staged selection claims or digests do not match snapshot')
    if (request.get('split') != 'train' or request.get('split_policy') != 'exploratory_train'
            or request.get('heldout_separation') != 'unknown' or request.get('review_status') != 'unreviewed'
            or request.get('outcome') != 'unknown' or request.get('source_bound_snapshot_sha256') != snapshot_sha
            or request.get('source_text_sha256_claim') != source.get('user_text_sha256')
            or request.get('source_prefix_sha256_claim') != source.get('source_prefix_sha256')
            or request.get('source_line_sha256_claim') != source.get('source_line_sha256')
            or not isinstance(request.get('text'), str)
            or request.get('screened_text_sha256') != sha(request['text'].encode('utf-8'))):
        raise ValueError('staged request does not match selected source')
    if (not isinstance(context_file.get('projects'), dict) or context_file.get('source_bound_snapshot_sha256') != snapshot_sha
            or context_file.get('materialized_from_verified_snapshot') is not True):
        raise ValueError('staged context does not match snapshot')
    entry = context_file['projects'].get(request.get('project'))
    if (not isinstance(entry, dict) or entry.get('source_bound_snapshot_sha256') != snapshot_sha
            or entry.get('source_text_sha256_claim') != source.get('user_text_sha256')
            or entry.get('provider_cache_sha256') != provider.get('cache_sha256')
            or entry.get('heldout_separation') != 'unknown'
            or not isinstance(entry.get('text'), str)
            or entry.get('sha256') != sha(entry['text'].encode('utf-8'))):
        raise ValueError('staged project context does not match snapshot')
    return {'snapshot_sha256': snapshot_sha, 'staged_snapshot_sha256': snapshot_copy_sha,
            'provider_cache_sha256': provider['cache_sha256'],
            'request_sha256': request['screened_text_sha256'], 'context_sha256': entry['sha256'],
            'project': request['project'], 'split': 'train', 'heldout_separation': 'unknown',
            'requests': 1, 'test_opened': False, 'validation_opened': False}


def stage_snapshot_projection(value: dict, context_text: str) -> tuple[dict, dict]:
    """Create a minimal copy manifest and snapshot projection for private staging."""
    origin_raw = json_bytes(value)
    origin_sha = sha(origin_raw)
    if not isinstance(context_text, str) or not context_text:
        raise ValueError('source-bound context is required for staging')
    selected_rows = []
    for line in context_text.splitlines()[1:]:
        try:
            row = _strict_json(line.encode('utf-8'))
        except (ValueError, UnicodeError, RecursionError):
            raise ValueError('invalid rendered provider context') from None
        if not isinstance(row, list) or len(row) != 5 or not isinstance(row[0], str):
            raise ValueError('invalid rendered provider model row')
        selected_rows.append({'model_id': row[0], 'display_name': row[1], 'description': row[2],
                              'priority': row[3], 'provider_supported_efforts': row[4]})
    if not selected_rows:
        raise ValueError('staging context has no source-bound provider rows')
    source, provider = value['source'], value['provider_list']
    projection = {
        'schema_version': SCHEMA,
        'observed_at': value['observed_at'],
        'materialization': {'kind': 'origin-verified-private-projection',
                            'origin_snapshot_sha256': origin_sha,
                            'source_and_cache_verified_on_origin_host': True,
                            'full_source_paths_transferred': False,
                            'provider_extension_payloads_transferred': False},
        'source': {key: source[key] for key in (
            'kind', 'project', 'project_association', 'source_locator_sha256', 'source_group_sha256', 'selected_line',
            'source_prefix_bytes', 'source_prefix_sha256', 'source_line_sha256', 'user_text_sha256',
            'user_text_utf8_bytes', 'source_file_sha256_at_inventory', 'current_whole_file_sha256', 'verified')},
        'split': value['split'],
        'provider_list': {
            'fetched_at': provider['fetched_at'],
            'cache_sha256': provider['cache_sha256'],
            'cache_bytes': provider['cache_bytes'],
            'cache_mtime_utc': provider['cache_mtime_utc'],
            'screen_policy_sha256': provider['screen_policy_sha256'],
            'source_locator_sha256': sha(provider['source_locator'].encode('utf-8')),
            'models': selected_rows,
        },
        'access_claims': value['access_claims'],
        'training_claims': value['training_claims'],
        'extensions': {},
    }
    projected_raw = json_bytes(projection)
    bundle = {'schema_version': 'lfg.source-bound-staging-bundle.v1',
              'origin_snapshot_sha256': origin_sha,
              'staged_snapshot_sha256': sha(projected_raw),
              'projection_kind': projection['materialization']['kind'],
              'full_source_paths_transferred': False,
              'provider_extension_payloads_transferred': False}
    return projection, bundle


def _require_ignored_untracked(path: Path) -> None:
    absolute = path.resolve()
    ancestor = absolute.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    try:
        root = subprocess.run(['git', '-C', str(ancestor), 'rev-parse', '--show-toplevel'],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
    except OSError:
        raise ValueError('cannot verify private snapshot location') from None
    if root.returncode:
        if any((item / '.git').exists() for item in (ancestor, *ancestor.parents)):
            raise ValueError('cannot verify private snapshot location')
        return
    repository = Path(root.stdout.rstrip('\r\n')).resolve()
    if not absolute.is_relative_to(repository):
        raise ValueError('cannot verify private snapshot location')
    relative = absolute.relative_to(repository).as_posix()
    ignored = subprocess.run(['git', '-C', str(repository), 'check-ignore', '-q', '--no-index', '--', relative],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    tracked = subprocess.run(['git', '-C', str(repository), 'ls-files', '--error-unmatch', '--', relative],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if ignored.returncode != 0 or tracked.returncode != 1:
        raise ValueError('snapshot must be ignored and untracked')


def _open_private_parent(path: Path) -> tuple[int, str, str]:
    """Open an existing POSIX parent without following replaceable path components."""
    if os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'supports_dir_fd'):
        raise ValueError('safe private output requires POSIX no-follow directory handles')
    absolute = os.path.abspath(os.fspath(path))
    parent, name = os.path.split(absolute)
    if not name or name in {'.', '..'}:
        raise ValueError('invalid private output path')
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | os.O_NOFOLLOW
    descriptor = os.open(os.path.sep, flags)
    try:
        current = os.path.sep
        for component in parent.split(os.path.sep):
            if not component:
                continue
            child = os.open(component, flags, dir_fd=descriptor)
            try:
                info = os.fstat(child)
                mode = stat.S_IMODE(info.st_mode)
                trusted_owner = info.st_uid in {0, os.geteuid()}
                sticky_shared = bool(info.st_mode & stat.S_ISVTX) and info.st_uid == 0
                if not trusted_owner or (mode & 0o022 and not sticky_shared):
                    raise ValueError('private output path has an untrusted writable ancestor')
                os.close(descriptor)
                descriptor = child
                current = os.path.join(current, component)
            except Exception:
                os.close(child)
                raise
        info = os.fstat(descriptor)
        if info.st_uid not in {0, os.geteuid()} or stat.S_IMODE(info.st_mode) & 0o022:
            raise ValueError('private output parent is not protected from other writers')
        actual_parent = os.readlink(f'/proc/self/fd/{descriptor}')
        return descriptor, name, actual_parent
    except Exception:
        os.close(descriptor)
        raise


def _verify_parent_binding(path: Path, descriptor: int, name: str) -> None:
    """Reject path swaps after preflight while retaining the opened directory handle."""
    try:
        check, check_name, _actual = _open_private_parent(path)
    except (OSError, ValueError):
        raise ValueError('private output path changed during write') from None
    try:
        if check_name != name:
            raise ValueError('private output path changed during write')
        held, current = os.fstat(descriptor), os.fstat(check)
        if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError('private output path changed during write')
    finally:
        os.close(check)


def _write_private_posix(path: Path, raw: bytes) -> None:
    directory_fd = -1
    file_fd = -1
    name = ''
    created = False
    try:
        directory_fd, name, actual_parent = _open_private_parent(path)
        actual_path = Path(actual_parent) / name
        _require_ignored_untracked(actual_path)
        _verify_parent_binding(path, directory_fd, name)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        file_fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
        created = True
        info = os.fstat(file_fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o077):
            raise ValueError('private output file is not owner-only')
        with os.fdopen(file_fd, 'wb') as stream:
            file_fd = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _verify_parent_binding(path, directory_fd, name)
    except Exception:
        if created and directory_fd >= 0:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except OSError:
                pass
        raise
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if directory_fd >= 0:
            os.close(directory_fd)


def write_private_snapshot(path: Path, snapshot: dict) -> None:
    """Write a new ignored snapshot on POSIX; Windows path safety is not proven."""
    if os.name == 'nt':
        raise ValueError('Windows private snapshot output is disabled until no-follow ACL verification is available')
    raw = json_bytes(snapshot)
    _write_private_posix(path, raw)


def write_private_bytes(path: Path, raw: bytes) -> None:
    """Write private POSIX bytes; Windows path safety is not proven."""
    if not isinstance(raw, bytes):
        raise ValueError('private output must be bytes')
    if os.name == 'nt':
        raise ValueError('Windows private snapshot output is disabled until no-follow ACL verification is available')
    _write_private_posix(path, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--record', type=int, required=True)
    parser.add_argument('--provider-cache', type=Path, required=True)
    parser.add_argument('--screen-policy', type=Path, default=_default_screen_policy_path())
    parser.add_argument('--split-policy', choices=('exploratory_train', 'reserved_train'), required=True)
    parser.add_argument('--reserved-project', action='append', default=[])
    parser.add_argument('--reserved-session-ids', type=Path,
                        help='private lfg.reserved-session-groups.v1 manifest; coverage remains unknown or partial')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-session-bytes', type=int, default=512 * 1024 * 1024)
    parser.add_argument('--max-line-bytes', type=int, default=16 * 1024 * 1024)
    parser.add_argument('--max-cache-bytes', type=int, default=MAX_CACHE_BYTES)
    args = parser.parse_args()
    if os.name == 'nt':
        raise SystemExit('Snapshot validation failed; Windows private snapshot output is disabled.')
    try:
        if args.record < 1:
            raise ValueError('invalid inventory record number')
        with args.inventory.open('rb') as stream:
            for _ in range(args.record - 1):
                if not stream.readline(1024 * 1024 + 1):
                    raise ValueError('inventory record missing')
            inventory_line = stream.readline(1024 * 1024 + 1)
        if not inventory_line or len(inventory_line) > 1024 * 1024:
            raise ValueError('inventory record unavailable')
        record = _strict_json(inventory_line)
        if not isinstance(record, dict):
            raise ValueError('invalid inventory record')
        if args.reserved_session_ids is None:
            reserved_groups, reserved_manifest_sha, reserved_manifest_locator, reserved_coverage = [], None, None, 'unknown'
        else:
            reserved_groups, reserved_manifest_sha, reserved_coverage = read_reserved_session_manifest(
                args.reserved_session_ids)
            reserved_manifest_locator = str(args.reserved_session_ids.resolve())
        result = build_snapshot(record, args.provider_cache, split_policy=args.split_policy,
                                known_reserved_group_sha256=reserved_groups,
                                known_reserved_manifest_sha256=reserved_manifest_sha,
                                known_reserved_manifest_locator=reserved_manifest_locator,
                                known_reserved_groups_coverage=reserved_coverage,
                                known_reserved_project_labels=args.reserved_project,
                                screen_policy_path=args.screen_policy,
                                max_session_bytes=args.max_session_bytes, max_line_bytes=args.max_line_bytes,
                                max_cache_bytes=args.max_cache_bytes)
        write_private_snapshot(args.output, result)
        print(json.dumps({'status': 'verified', 'split_assignment': result['split']['assignment'],
                          'heldout_separation': result['split']['heldout_separation'],
                          'provider_models': len(result['provider_list']['models']),
                          'reserved_session_groups_checked': result['split']['known_reserved_groups_checked'],
                          'reserved_session_groups_coverage': result['split']['known_reserved_groups_coverage'],
                          'entitlement': result['access_claims']['subscription_entitlement']}))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, OverflowError):
        raise SystemExit('Snapshot validation failed; no source text or metadata was emitted.') from None


if __name__ == '__main__':
    main()
