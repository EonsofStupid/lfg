#!/usr/bin/env python3
"""Inventory explicitly human-marked local Claude text without storing prompt text."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import inventory_codex_sessions

SCHEMA = 'lfg.unreviewed-claude-message.v1'


def message_text(row: object, project_root: str) -> str | None:
    """Accept only explicit human-origin, main-session, single-text records."""
    if (not isinstance(row, dict) or row.get('type') != 'user' or row.get('turnOrigin') != 'human'
            or row.get('isSidechain') is not False or row.get('isMeta') or row.get('isCompactSummary')
            or row.get('isVisibleInTranscriptOnly') or 'sourceToolAssistantUUID' in row
            or 'toolUseResult' in row or row.get('promptSource') == 'system'):
        return None
    cwd = row.get('cwd')
    if (not isinstance(cwd, str) or not os.path.isabs(cwd)
            or inventory_codex_sessions.normalize(cwd) != inventory_codex_sessions.normalize(project_root)):
        return None
    message = row.get('message')
    if not isinstance(message, dict) or message.get('role') != 'user':
        return None
    content = message.get('content')
    if isinstance(content, str):
        text = content
    elif (isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict)
          and content[0].get('type') == 'text' and isinstance(content[0].get('text'), str)):
        text = content[0]['text']
    else:
        return None
    return text if text.strip() else None


def scan(path: Path, project: str, project_root: str, max_bytes: int, max_line_bytes: int) -> list[dict]:
    before = path.stat()
    if before.st_size > max_bytes:
        raise ValueError(f'session exceeds byte limit: {path}')
    hasher = hashlib.sha256()
    total = 0
    candidates = []
    with path.open('rb') as stream:
        line_number = 0
        while True:
            line = stream.readline(max_line_bytes + 1)
            if not line:
                break
            if len(line) > max_line_bytes:
                raise ValueError(f'session line exceeds byte limit: {path}')
            total += len(line)
            if total > max_bytes:
                raise ValueError(f'session grew beyond byte limit: {path}')
            hasher.update(line)
            line_number += 1
            row = json.loads(line)
            text = message_text(row, project_root)
            if text is None:
                continue
            candidates.append({'schema_version': SCHEMA, 'project': project, 'project_root': project_root,
                               'source_file': str(path), 'source_line': line_number,
                               'source_line_sha256': hashlib.sha256(line.rstrip(b'\r\n')).hexdigest(),
                               'source_prefix_sha256': hasher.hexdigest(), 'source_prefix_bytes': total,
                               'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                               'text_utf8_bytes': len(text.encode('utf-8')),
                               'verified': False, 'labels': None, 'split': None})
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or total != before.st_size:
        raise ValueError(f'session changed during inventory: {path}')
    for row in candidates:
        row['source_file_sha256'] = hasher.hexdigest()
    return candidates


def inventory(sources: dict[str, tuple[Path, str]], max_bytes: int, max_line_bytes: int) -> tuple[dict, list[dict]]:
    if not sources or min(max_bytes, max_line_bytes) < 1:
        raise ValueError('sources and positive byte limits are required')
    candidates = []
    counts = Counter()
    seen = set()
    for project, (directory, project_root) in sources.items():
        if not project.strip() or not os.path.isabs(project_root) or not directory.is_absolute():
            raise ValueError('project names, source directories and project roots must be explicit')
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('source directory must be a real directory')
        root = directory.resolve(strict=True)
        for path in sorted(root.glob('*.jsonl')):
            if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
                raise ValueError('source session must be a regular file inside its selected directory')
            resolved = path.resolve(strict=True)
            if resolved in seen:
                raise ValueError('source session selected more than once')
            seen.add(resolved)
            rows = scan(resolved, project, project_root, max_bytes, max_line_bytes)
            candidates.extend(rows)
            counts[project] += len(rows)
    return {'schema_version': 'lfg.claude-inventory-summary.v1', 'source_files': len(seen),
            'candidate_messages': len(candidates), 'projects': dict(counts),
            'unique_exact_texts': len({row['text_sha256'] for row in candidates}),
            'verified': 0, 'labeled': 0, 'split_assigned': 0}, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True, metavar='NAME=ABSOLUTE_SESSION_DIRECTORY')
    parser.add_argument('--project-root', action='append', required=True, metavar='NAME=ABSOLUTE_PROJECT_ROOT')
    parser.add_argument('--max-session-bytes', type=int, required=True)
    parser.add_argument('--max-line-bytes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new private metadata inventory')
    args = parser.parse_args()
    try:
        def bindings(values):
            result = {}
            for value in values:
                name, separator, path = value.partition('=')
                if not separator or not name.strip() or not os.path.isabs(path) or name in result:
                    raise ValueError('bindings require distinct NAME=absolute-path values')
                result[name] = path
            return result
        dirs, roots = bindings(args.source), bindings(args.project_root)
        if set(dirs) != set(roots):
            raise ValueError('source and project-root names must match exactly')
        summary, rows = inventory({name: (Path(dirs[name]), roots[name]) for name in dirs},
                                  args.max_session_bytes, args.max_line_bytes)
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(summary, indent=2))
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        raise SystemExit(f'Claude inventory rejected: {error}') from error


if __name__ == '__main__':
    main()
