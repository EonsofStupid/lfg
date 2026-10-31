#!/usr/bin/env python3
"""Verify one local inventory entry and optionally display its original text."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def verify(record: dict, max_bytes: int, max_line_bytes: int) -> str:
    if (record.get('schema_version') != 'lfg.unreviewed-codex-message.v2'
            or not isinstance(record.get('source_file'), str)
            or type(record.get('source_line')) is not int or record['source_line'] < 1
            or type(record.get('source_prefix_bytes')) is not int or record['source_prefix_bytes'] < 1
            or not isinstance(record.get('source_file_sha256'), str)
            or not isinstance(record.get('source_prefix_sha256'), str)
            or not isinstance(record.get('source_line_sha256'), str)
            or not isinstance(record.get('text_sha256'), str)):
        raise ValueError('invalid inventory entry')
    path = Path(record['source_file'])
    before = path.stat()
    if record['source_prefix_bytes'] > max_bytes or before.st_size < record['source_prefix_bytes']:
        raise ValueError('source prefix is unavailable or exceeds byte limit')
    hasher = hashlib.sha256()
    selected = None
    total = 0
    with path.open('rb') as source:
        line_number = 0
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
            if line_number == record['source_line']:
                selected = line.rstrip(b'\r\n')
                break
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('source changed during read')
    if (hasher.hexdigest() != record['source_prefix_sha256'] or selected is None
            or total != record['source_prefix_bytes']):
        raise ValueError('source prefix no longer matches the inventory')
    if hashlib.sha256(selected).hexdigest() != record['source_line_sha256']:
        raise ValueError('source line does not match the inventory')
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
