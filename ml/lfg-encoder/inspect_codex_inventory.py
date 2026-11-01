#!/usr/bin/env python3
"""Verify one local inventory entry and optionally display its original text."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import sys
from pathlib import Path


def verify_record(record: dict) -> None:
    if (record.get('schema_version') != 'lfg.unreviewed-codex-message.v2'
            or not isinstance(record.get('source_file'), str)
            or type(record.get('source_line')) is not int or record['source_line'] < 1
            or type(record.get('source_prefix_bytes')) is not int or record['source_prefix_bytes'] < 1
            or not isinstance(record.get('source_file_sha256'), str)
            or not isinstance(record.get('source_prefix_sha256'), str)
            or not isinstance(record.get('source_line_sha256'), str)
            or not isinstance(record.get('text_sha256'), str)):
        raise ValueError('invalid inventory entry')


def verify_many(records: list[dict], max_bytes: int, max_line_bytes: int) -> list[str]:
    """Verify selected records in source order, reading each source prefix once."""
    groups: dict[str, list[tuple[int, dict]]] = {}
    for index, record in enumerate(records):
        verify_record(record)
        groups.setdefault(record['source_file'], []).append((index, record))
    results = [''] * len(records)
    for name, group in groups.items():
        ordered = sorted(group, key=lambda item: item[1]['source_line'])
        if len({item['source_line'] for _, item in ordered}) != len(ordered):
            raise ValueError('duplicate source line')
        for index, text in verify_group(Path(name), ordered, max_bytes, max_line_bytes):
            results[index] = text
    return results


def verify_group(path: Path, ordered: list[tuple[int, dict]], max_bytes: int, max_line_bytes: int,
                 on_prior_line=None, parse_message=None) -> list[tuple[int, str]]:
    before = path.stat()
    if any(record['source_prefix_bytes'] > max_bytes or before.st_size < record['source_prefix_bytes']
           for _, record in ordered):
        raise ValueError('source prefix is unavailable or exceeds byte limit')
    hasher = hashlib.sha256()
    total = 0
    found = []
    with path.open('rb') as source:
        line_number = 0
        pending = iter(ordered)
        current = next(pending, None)
        while True:
            line = source.readline(max_line_bytes + 1)
            if not line:
                break
            if len(line) > max_line_bytes:
                raise ValueError('source line exceeds byte limit')
            total += len(line)
            if total > max_bytes:
                raise ValueError('source grew beyond byte limit')
            hasher.update(line)
            line_number += 1
            if on_prior_line is not None and current and line_number < current[1]['source_line']:
                on_prior_line(line_number, line)
            if current and line_number == current[1]['source_line']:
                index, record = current
                selected = line.rstrip(b'\r\n')
                if (hasher.hexdigest() != record['source_prefix_sha256']
                        or total != record['source_prefix_bytes']):
                    raise ValueError('source prefix no longer matches the inventory')
                if hashlib.sha256(selected).hexdigest() != record['source_line_sha256']:
                    raise ValueError('source line does not match the inventory')
                found.append((index, (parse_message or message_text)(selected, record)))
                current = next(pending, None)
                if current is None:
                    break
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('source changed during read')
    if len(found) != len(ordered):
        raise ValueError('source line is missing')
    return found


def message_text(selected: bytes, record: dict) -> str:
    try:
        row = json.loads(selected)
        payload = row['payload']
        meta = payload['internal_chat_message_metadata_passthrough']
        content = payload['content']
        if (row['type'] != 'response_item' or payload['type'] != 'message'
                or payload['role'] != 'user' or meta['content_item_kinds'] != ['user.text']
                or len(content) != 1 or content[0]['type'] != 'input_text'
                or not isinstance(content[0]['text'], str)):
            raise ValueError('source is not a single user-text message')
        text = content[0]['text']
    except (KeyError, TypeError, IndexError, AttributeError, UnicodeError) as error:
        raise ValueError('invalid source message') from error
    if hashlib.sha256(text.encode('utf-8')).hexdigest() != record['text_sha256']:
        raise ValueError('message text does not match the inventory')
    return text


def verify(record: dict, max_bytes: int, max_line_bytes: int) -> str:
    return verify_many([record], max_bytes, max_line_bytes)[0]


def prior_user_context(record: dict, max_bytes: int, max_line_bytes: int,
                       max_messages: int, max_context_bytes: int) -> tuple[str, list[dict], bool]:
    """Return a bounded suffix of explicitly marked prior user messages after prefix verification."""
    verify_record(record)
    if any(type(value) is not int or value < 1 for value in (max_bytes, max_line_bytes, max_messages, max_context_bytes)):
        raise ValueError('context limits must be positive integers')
    messages = deque()
    size = 0
    truncated = False

    def consume(line_number: int, line: bytes) -> None:
        nonlocal size, truncated
        row = json.loads(line)
        if not isinstance(row, dict) or row.get('type') != 'response_item':
            return
        payload = row.get('payload')
        if not isinstance(payload, dict) or payload.get('type') != 'message' or payload.get('role') != 'user':
            return
        metadata, content = payload.get('internal_chat_message_metadata_passthrough'), payload.get('content')
        if (not isinstance(metadata, dict) or metadata.get('content_item_kinds') != ['user.text']
                or not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict)
                or content[0].get('type') != 'input_text' or not isinstance(content[0].get('text'), str)):
            return
        text = content[0]['text']
        if not text.strip():
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

    found = verify_group(Path(record['source_file']), [(0, record)], max_bytes, max_line_bytes, consume)
    return found[0][1], [{'source_line': line, 'text': text} for line, text, _ in messages], truncated


def following_user_context(record: dict, max_bytes: int, max_line_bytes: int,
                           max_messages: int, max_context_bytes: int, max_scan_lines: int) -> tuple[str, list[dict], bool, int, str]:
    """Verify the original prefix, then inspect a bounded suffix of the current session."""
    verify_record(record)
    if any(type(value) is not int or value < 1 for value in
           (max_bytes, max_line_bytes, max_messages, max_context_bytes, max_scan_lines)):
        raise ValueError('context limits must be positive integers')
    path = Path(record['source_file'])
    before = path.stat()
    if record['source_prefix_bytes'] > max_bytes or before.st_size < record['source_prefix_bytes']:
        raise ValueError('source prefix is unavailable or exceeds byte limit')
    hasher = hashlib.sha256()
    total = 0
    messages = []
    size = 0
    truncated = False
    through_line = 0
    with path.open('rb') as source:
        for line_number in range(1, record['source_line'] + 1):
            line = source.readline(max_line_bytes + 1)
            if not line or len(line) > max_line_bytes:
                raise ValueError('source line missing or exceeds byte limit')
            total += len(line)
            if total > max_bytes:
                raise ValueError('source prefix exceeds byte limit')
            hasher.update(line)
            through_line = line_number
        selected = line.rstrip(b'\r\n')
        if (total != record['source_prefix_bytes'] or hasher.hexdigest() != record['source_prefix_sha256']
                or hashlib.sha256(selected).hexdigest() != record['source_line_sha256']):
            raise ValueError('source prefix or line no longer matches the inventory')
        text = message_text(selected, record)
        for offset in range(1, max_scan_lines + 1):
            line = source.readline(max_line_bytes + 1)
            if not line:
                break
            if len(line) > max_line_bytes:
                raise ValueError('source line exceeds byte limit')
            if total + len(line) > max_bytes:
                truncated = True
                break
            total += len(line)
            hasher.update(line)
            through_line = record['source_line'] + offset
            row = json.loads(line)
            if not isinstance(row, dict) or row.get('type') != 'response_item':
                continue
            payload = row.get('payload')
            if not isinstance(payload, dict) or payload.get('type') != 'message' or payload.get('role') != 'user':
                continue
            metadata, content = payload.get('internal_chat_message_metadata_passthrough'), payload.get('content')
            if (not isinstance(metadata, dict) or metadata.get('content_item_kinds') != ['user.text']
                    or not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict)
                    or content[0].get('type') != 'input_text' or not isinstance(content[0].get('text'), str)
                    or not content[0]['text'].strip()):
                continue
            message = content[0]['text']
            length = len(message.encode('utf-8'))
            if len(messages) >= max_messages or size + length > max_context_bytes:
                truncated = True
                break
            messages.append({'source_line': through_line, 'text': message})
            size += length
        else:
            truncated = True
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('source changed during read')
    return text, messages, truncated, through_line, hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--record', type=int, required=True, help='one-based inventory row number')
    parser.add_argument('--show-text', action='store_true', help='print the private original message after verification')
    parser.add_argument('--max-session-bytes', type=int, default=512 * 1024 * 1024)
    parser.add_argument('--max-line-bytes', type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()
    try:
        if args.record < 1 or args.max_session_bytes < 1 or args.max_line_bytes < 1:
            raise ValueError('record and byte limits must be positive')
        with args.inventory.open(encoding='utf-8') as source:
            line = next((line for index, line in enumerate(source, 1) if index == args.record), None)
        if line is None:
            raise ValueError('inventory record does not exist')
        record = json.loads(line)
        text = verify(record, args.max_session_bytes, args.max_line_bytes)
        result = {'status': 'verified_source_prefix_match', 'project': record['project'],
                  'source_line': record['source_line'], 'human_reviewed': False}
        if args.show_text:
            result['text'] = text
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        print(f'inspection failed: {error}', file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
