#!/usr/bin/env python3
"""Accept one bounded private assessment dataset on standard input."""
import argparse
import base64
import binascii
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def receive(raw: bytes, inbox: Path, max_payload_bytes: int, max_dataset_bytes: int) -> dict:
    if (type(max_payload_bytes) is not int or max_payload_bytes < 1 or
            type(max_dataset_bytes) is not int or max_dataset_bytes < 1 or len(raw) > max_payload_bytes):
        raise ValueError('invalid transfer size')
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get('schema_version') != 'lfg.teacher-dataset-transfer.v1':
        raise ValueError('invalid transfer format')
    try:
        manifest = base64.b64decode(value['manifest_base64'], validate=True)
        train = base64.b64decode(value['train_base64'], validate=True)
    except (KeyError, TypeError, ValueError, binascii.Error):
        raise ValueError('invalid transfer encoding') from None
    if len(manifest) + len(train) > max_dataset_bytes:
        raise ValueError('dataset exceeds configured size')
    metadata = json.loads(manifest)
    sources = metadata.get('sources') if isinstance(metadata, dict) else None
    if (not isinstance(metadata, dict) or metadata.get('schema_version') != 'lfg.assessment-distillation-dataset.v1'
            or metadata.get('test_assessed') is not False or not isinstance(metadata.get('files'), dict)
            or set(metadata['files']) != {'train.jsonl'} or metadata['files']['train.jsonl'] != sha(train)
            or not isinstance(sources, list) or not sources
            or any(not isinstance(item, dict) or item.get('split') != 'train'
                   or item.get('origin') not in ('owner_chat', 'owner_project') for item in sources)):
        raise ValueError('invalid train-only dataset manifest')
    identifiers = set()
    for line in train.splitlines():
        row = json.loads(line)
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']
                or row['id'] in identifiers):
            raise ValueError('invalid training record identity')
        identifiers.add(row['id'])
    if not identifiers:
        raise ValueError('empty training dataset')
    dataset_id = sha(manifest + b'\0' + train)
    inbox.mkdir(mode=0o700, parents=True, exist_ok=True)
    if inbox.is_symlink() or not inbox.is_dir():
        raise ValueError('invalid inbox')
    final = inbox / dataset_id
    if final.exists():
        if (final.is_symlink() or not final.is_dir() or (final / 'manifest.json').read_bytes() != manifest
                or (final / 'train.jsonl').read_bytes() != train):
            raise ValueError('existing inbox dataset does not match')
        return {'dataset_id': dataset_id, 'records': len(identifiers), 'already_present': True}
    temporary = Path(tempfile.mkdtemp(prefix='.incoming-', dir=inbox))
    try:
        for name, data in (('manifest.json', manifest), ('train.jsonl', train)):
            descriptor = os.open(temporary / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        os.rename(temporary, final)
    except Exception:
        # Preserve an incomplete temporary directory for inspection; it is never
        # selected as a published inbox dataset.
        raise
    return {'dataset_id': dataset_id, 'records': len(identifiers), 'already_present': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inbox', type=Path, required=True)
    parser.add_argument('--max-payload-bytes', type=int, required=True)
    parser.add_argument('--max-dataset-bytes', type=int, required=True)
    args = parser.parse_args()
    if not args.inbox.is_absolute():
        parser.error('inbox must be absolute')
    if args.max_payload_bytes < 1 or args.max_dataset_bytes < 1:
        parser.error('size limits must be positive')
    try:
        raw = sys.stdin.buffer.read(args.max_payload_bytes + 1)
        result = receive(raw, args.inbox, args.max_payload_bytes, args.max_dataset_bytes)
        print(json.dumps(result), flush=True)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        raise SystemExit('teacher dataset transfer rejected') from None


if __name__ == '__main__':
    main()
