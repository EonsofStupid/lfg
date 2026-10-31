#!/usr/bin/env python3
"""Verify private Claude inventory entries against original session prefixes."""
from __future__ import annotations

import hashlib
import json
import os
from collections import deque
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
                       max_messages: int, max_context_bytes: int) -> tuple[str, list[dict], bool]:
    """Return a bounded file-order suffix, not a verified conversation ancestry."""
    verify_record(record)
    if any(type(value) is not int or value < 1 for value in (max_bytes, max_line_bytes, max_messages, max_context_bytes)):
        raise ValueError('context limits must be positive integers')
    messages = deque()
    size = 0
    truncated = False

    def consume(line_number: int, line: bytes) -> None:
        nonlocal size, truncated
        text = inventory_claude_sessions.message_text(json.loads(line), record['project_root'])
        if text is None:
            return
        length = len(text.encode('utf-8'))
        if length > max_context_bytes:
            messages.clear()
            size = 0
            truncated = True
            return
        messages.append((line_number, text, length))
        size += length
        while len(messages) > max_messages or size > max_context_bytes:
            _, _, removed = messages.popleft()
            size -= removed
            truncated = True

    found = inspect_codex_inventory.verify_group(Path(record['source_file']), [(0, record)], max_bytes,
                                                max_line_bytes, consume, parse_message=message_text)
    return found[0][1], [{'source_line': line, 'text': text} for line, text, _ in messages], truncated
