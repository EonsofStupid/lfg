#!/usr/bin/env python3
"""Verify private Claude inventory entries against original session prefixes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import inspect_codex_inventory
import inventory_claude_sessions


def verify_record(record: dict) -> None:
    if (not isinstance(record, dict) or record.get('schema_version') != inventory_claude_sessions.SCHEMA
            or not isinstance(record.get('source_file'), str) or not os.path.isabs(record['source_file'])
            or not isinstance(record.get('project_root'), str) or not os.path.isabs(record['project_root'])
            or type(record.get('source_line')) is not int or record['source_line'] < 1
            or type(record.get('source_prefix_bytes')) is not int or record['source_prefix_bytes'] < 1
            or any(not isinstance(record.get(name), str) or len(record[name]) != 64
                   or any(character not in '0123456789abcdef' for character in record[name])
                   for name in ('source_file_sha256', 'source_prefix_sha256', 'source_line_sha256', 'text_sha256'))):
        raise ValueError('invalid Claude inventory entry')


def message_text(selected: bytes, record: dict) -> str:
    row = json.loads(selected)
    text = inventory_claude_sessions.message_text(row, record['project_root'])
    if text is None or hashlib.sha256(text.encode('utf-8')).hexdigest() != record['text_sha256']:
        raise ValueError('source is not the inventoried human-marked single-text message')
    return text


def verify_many(records: list[dict], max_bytes: int, max_line_bytes: int) -> list[str]:
    if min(max_bytes, max_line_bytes) < 1:
        raise ValueError('byte limits must be positive')
    groups: dict[str, list[tuple[int, dict]]] = {}
    for index, record in enumerate(records):
        verify_record(record)
        groups.setdefault(record['source_file'], []).append((index, record))
    results = [''] * len(records)
    for name, group in groups.items():
        ordered = sorted(group, key=lambda item: item[1]['source_line'])
        if len({item['source_line'] for _, item in ordered}) != len(ordered):
            raise ValueError('duplicate source line')
        for index, text in inspect_codex_inventory.verify_group(
                Path(name), ordered, max_bytes, max_line_bytes, parse_message=message_text):
            results[index] = text
    return results


def verify(record: dict, max_bytes: int, max_line_bytes: int) -> str:
    return verify_many([record], max_bytes, max_line_bytes)[0]


def prior_user_context(record: dict, max_bytes: int, max_line_bytes: int,
                       max_messages: int, max_context_bytes: int,
                       max_ancestry_records: int) -> tuple[str, list[dict], bool]:
    """Follow bounded same-session parent links after verifying the source prefix."""
    verify_record(record)
    if any(type(value) is not int or value < 1 for value in
           (max_bytes, max_line_bytes, max_messages, max_context_bytes, max_ancestry_records)):
        raise ValueError('context limits must be positive integers')
    path = Path(record['source_file'])
    before = path.stat()
    nodes = {}
    offset = 0
    selected = {}

    missing_parent = object()

    def signature(row):
        if not isinstance(row, dict):
            return None
        uid, parent, session = row.get('uuid'), row.get('parentUuid', missing_parent), row.get('sessionId')
        if not isinstance(uid, str) or not uid:
            return None
        text = inventory_claude_sessions.message_text(row, record['project_root'])
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest() if text is not None else None
        length = len(text.encode('utf-8')) if text is not None else 0
        return uid, (parent, session, row.get('type'), row.get('turnOrigin'), text_hash, length)

    def consume(line_number: int, line: bytes) -> None:
        nonlocal offset
        parsed = signature(json.loads(line))
        if parsed is not None:
            uid, identity = parsed
            if uid not in nodes:
                if len(nodes) >= max_ancestry_records:
                    raise ValueError('source exceeds configured ancestry record limit')
                nodes[uid] = (identity, line_number, offset, len(line), hashlib.sha256(line).hexdigest())
            elif nodes[uid] is not None and nodes[uid][0] != identity:
                nodes[uid] = None
        offset += len(line)

    def selected_message(raw: bytes, entry: dict) -> str:
        text = message_text(raw, entry)
        identity = signature(json.loads(raw))
        if identity is None:
            raise ValueError('selected source has no record identity')
        selected['identity'] = identity
        return text

    found = inspect_codex_inventory.verify_group(path, [(0, record)], max_bytes, max_line_bytes,
                                                consume, parse_message=selected_message)
    selected_id, identity = selected['identity']
    parent, session, _, _, _, _ = identity
    if not isinstance(session, str) or not session:
        raise ValueError('selected source has no session identity')
    if selected_id in nodes and (nodes[selected_id] is None or nodes[selected_id][0] != identity):
        raise ValueError('selected record identity is ambiguous')
    selected_nodes = []
    seen = {selected_id}
    size = 0
    truncated = False
    while parent is not None:
        if not isinstance(parent, str) or not parent or parent in seen:
            raise ValueError('invalid or cyclic source ancestry')
        seen.add(parent)
        if parent not in nodes or nodes[parent] is None:
            raise ValueError('source ancestry is missing or ambiguous')
        node = nodes[parent]
        node_parent, node_session, _, _, text_hash, text_length = node[0]
        if node_session != session:
            raise ValueError('source ancestry crosses session identity')
        if text_hash is not None:
            if len(selected_nodes) >= max_messages or size + text_length > max_context_bytes:
                truncated = True
                break
            selected_nodes.append(node)
            size += text_length
        parent = node_parent
    after_scan = path.stat()
    stamp = lambda item: (item.st_size, item.st_mtime_ns, item.st_ino)
    if stamp(before) != stamp(after_scan):
        raise ValueError('source changed during context inspection')
    messages = []
    with path.open('rb') as stream:
        for identity, line_number, start, length, line_hash in reversed(selected_nodes):
            stream.seek(start)
            raw = stream.read(length)
            if hashlib.sha256(raw).hexdigest() != line_hash:
                raise ValueError('source changed during context inspection')
            row = json.loads(raw)
            text = inventory_claude_sessions.message_text(row, record['project_root'])
            if text is None or signature(row)[1] != identity:
                raise ValueError('source ancestry changed during context inspection')
            messages.append({'source_line': line_number, 'text': text})
    if stamp(before) != stamp(path.stat()):
        raise ValueError('source changed during context inspection')
    return found[0][1], messages, truncated
