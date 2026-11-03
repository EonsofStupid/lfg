"""Bounded, digest-checked project context shared by teacher and owned inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = 'lfg.project-training-context.v1'
SEPARATOR = '\n\n'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def strings(value):
    stack, visited = [(value, 0)], 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if depth > 64 or visited > 10000:
            raise ValueError('project context extension structure exceeds limits')
        if isinstance(current, str):
            yield current
        elif isinstance(current, dict):
            for key, item in current.items():
                if not isinstance(key, str):
                    raise ValueError('invalid project context extension key')
                yield key
                stack.append((item, depth + 1))
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def render(context: str | None, request: str, *, max_context_bytes: int,
           expected_sha256: str | None = None) -> str:
    """Join optional context and request with the cross-runtime canonical separator."""
    if not isinstance(request, str) or not request:
        raise ValueError('request must be non-empty text')
    if context is None:
        if expected_sha256 is not None:
            raise ValueError('context digest supplied without context')
        return request
    if not isinstance(context, str) or not context or len(context.encode('utf-8')) > max_context_bytes:
        raise ValueError('project context exceeds configured limit or is empty')
    digest = sha(context.encode('utf-8'))
    if expected_sha256 != digest:
        raise ValueError('project context digest mismatch')
    return context + SEPARATOR + request


def load(path: Path, *, max_file_bytes: int, max_context_bytes: int) -> tuple[dict, str, dict]:
    with path.open('rb') as stream:
        raw = stream.read(max_file_bytes + 1)
    if len(raw) > max_file_bytes:
        raise ValueError('project context file exceeds configured limit')
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError('invalid project context file') from None
    if not isinstance(value, dict) or value.get('schema_version') != SCHEMA:
        raise ValueError('invalid project context file')
    contexts = value.get('projects')
    if not isinstance(contexts, dict) or not contexts:
        raise ValueError('invalid project context file')
    validated = {}
    for project, entry in contexts.items():
        if (not isinstance(project, str) or not project or not isinstance(entry, dict)
                or not isinstance(entry.get('text'), str) or not entry['text']
                or len(entry['text'].encode('utf-8')) > max_context_bytes
                or entry.get('sha256') != sha(entry['text'].encode('utf-8'))):
            raise ValueError('invalid project context entry')
        extensions = {key: item for key, item in entry.items() if key not in ('text', 'sha256')}
        try:
            json.dumps(extensions, allow_nan=False)
            list(strings(extensions))
        except (TypeError, ValueError):
            raise ValueError('invalid project context extension') from None
        validated[project] = {'text': entry['text'], 'sha256': entry['sha256'],
                              'extensions': extensions}
    top_extensions = {key: item for key, item in value.items() if key not in ('schema_version', 'projects')}
    try:
        json.dumps(top_extensions, allow_nan=False)
        list(strings(top_extensions))
    except (TypeError, ValueError):
        raise ValueError('invalid project context extension') from None
    return validated, sha(raw), top_extensions
