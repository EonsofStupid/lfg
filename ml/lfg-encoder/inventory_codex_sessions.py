#!/usr/bin/env python3
"""Inventory explicitly marked user text in selected local Codex project sessions.

The optional inventory contains source paths, line numbers and hashes, never the
message text. All records are unreviewed and have no label or split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(path: str) -> str:
    return os.path.normcase(os.path.abspath(path)).rstrip(os.sep)


def project_for(cwd: str, roots: dict[str, str], exact_cwd: bool = False) -> str | None:
    if not isinstance(cwd, str) or not cwd or not os.path.isabs(cwd):
        return None
    value = normalize(cwd)
    matches = [name for name, root in roots.items()
               if value == root or (not exact_cwd and value.startswith(root + os.sep))]
    if len(matches) > 1:
        raise ValueError("project roots overlap")
    return matches[0] if matches else None


def scan_file(path: Path, roots: dict[str, str], max_bytes: int, max_line_bytes: int,
              exact_cwd: bool = False) -> tuple[str, list[dict]]:
    """Read the header before examining any messages in an unmatched session."""
    before = path.stat()
    with path.open('rb') as source:
        header = source.readline(1024 * 1024 + 1)
        if len(header) > 1024 * 1024 or not header.endswith(b'\n'):
            return 'invalid_header', []
        try:
            first = json.loads(header)
            if first.get('type') != 'session_meta':
                return 'invalid_header', []
            project = project_for(first.get('payload', {}).get('cwd', ''), roots, exact_cwd)
        except (ValueError, TypeError, AttributeError):
            return 'invalid_header', []
        if project is None:
            return 'outside_scope', []
        if before.st_size > max_bytes:
            return 'too_large', []
        hasher = hashlib.sha256(header)
        total_bytes = len(header)
        candidates = []
        line_number = 1
        try:
            while True:
                line = source.readline(max_line_bytes + 1)
                if not line:
                    break
                if len(line) > max_line_bytes:
                    return 'line_too_large', []
                line_number += 1
                total_bytes += len(line)
                if total_bytes > max_bytes:
                    return 'changed_during_read', []
                hasher.update(line)
                row = json.loads(line)
                payload = row.get('payload', {})
                if row.get('type') != 'response_item' or payload.get('type') != 'message' or payload.get('role') != 'user':
                    continue
                meta = payload.get('internal_chat_message_metadata_passthrough')
                content = payload.get('content')
                if not isinstance(meta, dict) or meta.get('content_item_kinds') != ['user.text']:
                    continue
                if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict):
                    continue
                item = content[0]
                text = item.get('text')
                if item.get('type') != 'input_text' or not isinstance(text, str) or not text.strip():
                    continue
                encoded = text.encode('utf-8')
                candidates.append({
                    'schema_version': 'lfg.unreviewed-codex-message.v2',
                    'project': project, 'source_file': str(path), 'source_line': line_number,
                    'source_line_sha256': digest(line.rstrip(b'\r\n')), 'text_sha256': digest(encoded),
                    'source_prefix_sha256': hasher.hexdigest(), 'source_prefix_bytes': total_bytes,
                    'text_utf8_bytes': len(encoded), 'verified': False,
                    'labels': None, 'split': None,
                })
        except (ValueError, TypeError, AttributeError, UnicodeError):
            return 'invalid_json', []
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or total_bytes != before.st_size:
        return 'changed_during_read', []
    file_hash = hasher.hexdigest()
    for candidate in candidates:
        candidate['source_file_sha256'] = file_hash
    return 'scanned', candidates


def inventory(root: Path, roots: dict[str, str], max_bytes: int, max_line_bytes: int = 16 * 1024 * 1024,
              exact_cwd: bool = False) -> tuple[dict, list[dict]]:
    root = root.resolve(strict=True)
    status: Counter[str] = Counter()
    projects: Counter[str] = Counter()
    candidates: list[dict] = []
    for path in sorted(root.rglob('*.jsonl')):
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root) or not resolved.is_file():
                status['outside_source_root'] += 1
                continue
            state, rows = scan_file(resolved, roots, max_bytes, max_line_bytes, exact_cwd)
        except OSError:
            state, rows = 'read_error', []
        status[state] += 1
        candidates.extend(rows)
        projects.update(row['project'] for row in rows)
    hashes = {row['text_sha256'] for row in candidates}
    return ({'schema_version': 'lfg.codex-inventory-summary.v1', 'file_status': dict(status),
             'candidate_messages': len(candidates), 'projects': dict(projects),
             'exact_repetitions': len(candidates) - len(hashes),
             'verified': 0, 'labeled': 0, 'split_assigned': 0}, candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions-root', type=Path, required=True)
    parser.add_argument('--project-root', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--max-session-bytes', type=int, default=512 * 1024 * 1024)
    parser.add_argument('--max-line-bytes', type=int, default=16 * 1024 * 1024)
    parser.add_argument('--exact-cwd', action='store_true',
                        help='include only sessions whose working directory exactly matches a project root')
    parser.add_argument('--output', type=Path, help='new private inventory file; source paths and hashes are sensitive')
    args = parser.parse_args()
    try:
        if args.max_session_bytes < 1 or args.max_line_bytes < 1:
            raise ValueError('session and line byte limits must be positive')
        roots = {}
        for binding in args.project_root:
            name, separator, path = binding.partition('=')
            if not separator or not name.strip() or not path or not os.path.isabs(path) or name in roots:
                raise ValueError('each project root must be a unique NAME=absolute-path')
            roots[name] = normalize(path)
        values = list(roots.values())
        if any(a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)
               for i, a in enumerate(values) for b in values[i + 1:]):
            raise ValueError('project roots overlap')
        summary, candidates = inventory(args.sessions_root, roots, args.max_session_bytes, args.max_line_bytes, args.exact_cwd)
        if args.output:
            with args.output.open('x', encoding='utf-8', newline='\n') as output:
                for row in candidates:
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(summary, indent=2))
    except (OSError, ValueError) as error:
        print(f'inventory failed: {error}', file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
