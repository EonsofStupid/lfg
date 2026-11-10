"""Inspect bounded, unambiguous descendant chains in Claude session files."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import inspect_claude_inventory as inspect
import inventory_claude_sessions as inventory


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stamp(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def session_scan(path: Path, candidates: list[dict], config: dict) -> list[dict]:
    """Verify originals and follow only unique, same-session child links."""
    before = stamp(path)
    records = [{**candidate['source'], 'schema_version': inventory.SCHEMA} for candidate in candidates]
    for record in records:
        inspect.verify_record(record)
        if record['source_prefix_bytes'] > config['max_session_bytes']:
            raise ValueError('source prefix exceeds configured limit')
    if (len({record['source_line'] for record in records}) != len(records)
            or len({record['project_root'] for record in records}) != 1):
        raise ValueError('duplicate positions or mixed project roots')
    project_root = records[0]['project_root']
    positions = {record['source_line']: (record, candidate) for record, candidate in zip(records, candidates, strict=True)}
    last_line = max(positions) + config['max_scan_lines']
    nodes: dict[str, dict] = {}
    children: dict[tuple[str, str], list[dict]] = defaultdict(list)
    selected = {}
    hasher = hashlib.sha256()
    total = 0
    identities = 0
    with path.open('rb') as stream:
        for number in range(1, last_line + 1):
            line = stream.readline(config['max_line_bytes'] + 1)
            if not line:
                break
            if len(line) > config['max_line_bytes']:
                raise ValueError('source line exceeds configured limit')
            total += len(line)
            if total > config['max_session_bytes']:
                raise ValueError('source scan exceeds configured limit')
            hasher.update(line)
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            uid, parent, session = row.get('uuid'), row.get('parentUuid'), row.get('sessionId')
            identifiable = (isinstance(uid, str) and bool(uid) and isinstance(session, str) and bool(session)
                            and 'parentUuid' in row and (parent is None or isinstance(parent, str) and bool(parent)))
            if number in positions:
                record, candidate = positions[number]
                raw = line.rstrip(b'\r\n')
                if (total != record['source_prefix_bytes'] or hasher.hexdigest() != record['source_prefix_sha256']
                        or sha(raw) != record['source_line_sha256']
                        or inspect.message_text(raw, record) != candidate['text'] or not identifiable):
                    raise ValueError('source prefix, review text or identity does not match')
                selected[number] = (uid, session, candidate)
            if not identifiable:
                continue
            identities += 1
            if identities > config['max_ancestry_records']:
                raise ValueError('source exceeds ancestry record limit')
            text = inventory.message_text(row, project_root)
            encoded = text.encode('utf-8') if text is not None else b''
            identity = (parent, session, row.get('type'), row.get('turnOrigin'), sha(encoded) if text is not None else None)
            hit_ids = []
            if text is not None:
                for marker in config['markers']:
                    value = text.casefold() if marker['ignore_case'] else text
                    literal = marker['text'].casefold() if marker['ignore_case'] else marker['text']
                    if literal in value:
                        hit_ids.append(marker['id'])
            existing = nodes.get(uid)
            if existing is None:
                nodes[uid] = {'identity': identity, 'ambiguous': False}
            elif existing['identity'] != identity:
                existing['ambiguous'] = True
            if parent is not None:
                children[(session, parent)].append({'id': uid, 'line': number, 'bytes': len(encoded),
                    'hit_ids': hit_ids, 'line_sha256': sha(line.rstrip(b'\r\n')) if hit_ids else None,
                    'prefix_sha256': hasher.hexdigest() if hit_ids else None,
                    'human': text is not None})
    if set(selected) != set(positions):
        raise ValueError('selected source position disappeared')
    changed = stamp(path) != before
    if changed:
        remaining = total
        check = hashlib.sha256()
        with path.open('rb') as stream:
            while remaining:
                chunk = stream.read(min(remaining, 1048576))
                if not chunk:
                    raise ValueError('source changed during context scan')
                check.update(chunk)
                remaining -= len(chunk)
        if check.hexdigest() != hasher.hexdigest():
            raise ValueError('source changed during context scan')
    results = []
    for origin_line, (uid, session, candidate) in selected.items():
        hits = []
        seen = {uid}
        current_line = origin_line
        message_count = size = 0
        reason = 'no_observed_child'
        while True:
            node = nodes.get(uid)
            if node is None or node['ambiguous'] or node['identity'][1] != session:
                reason = 'ambiguous_identity'
                break
            options_by_id = {}
            for edge in children.get((session, uid), ()):
                if current_line < edge['line'] <= origin_line + config['max_scan_lines']:
                    options_by_id.setdefault(edge['id'], edge)
            options = list(options_by_id.values())
            if not options:
                break
            if len(options) != 1:
                reason = 'ambiguous_branch'
                break
            child = options[0]
            if nodes[child['id']]['ambiguous'] or child['id'] in seen:
                reason = 'ambiguous_identity'
                break
            seen.add(child['id'])
            uid, current_line = child['id'], child['line']
            if not child['human']:
                continue
            if size + child['bytes'] > config['max_context_bytes']:
                reason = 'context_byte_limit'
                break
            size += child['bytes']
            message_count += 1
            hits.extend({'marker_id': marker, 'source_line': child['line'],
                         'source_line_sha256': child['line_sha256'], 'observed_prefix_sha256': child['prefix_sha256']}
                        for marker in child['hit_ids'])
            if message_count >= config['max_messages']:
                reason = 'message_limit'
                break
        if changed and reason == 'no_observed_child':
            reason = 'source_changed_after_scan'
        results.append({**candidate['report'], 'hits': hits, 'context_end': reason})
    return results
