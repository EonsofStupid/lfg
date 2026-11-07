#!/usr/bin/env python3
"""Create an exact hash-checked pip requirements file from a local wheelhouse."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import zipfile
from email.parser import BytesParser
from pathlib import Path


def lock(wheelhouse: Path) -> bytes:
    directory = wheelhouse.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError('wheelhouse must be a directory')
    wheels = sorted(directory.iterdir())
    if not wheels or any(path.is_symlink() or not path.is_file() or path.suffix != '.whl' for path in wheels):
        raise ValueError('wheelhouse must contain only regular wheel files')
    entries = {}
    for path in wheels:
        before = path.stat()
        with zipfile.ZipFile(path) as wheel:
            names = [name for name in wheel.namelist()
                     if name.endswith('.dist-info/METADATA') and name.count('/') == 1]
            if len(names) != 1 or wheel.getinfo(names[0]).file_size > 4 * 1024 * 1024:
                raise ValueError('wheel must contain one bounded metadata file')
            meta = BytesParser().parsebytes(wheel.read(names[0]), headersonly=True)
        name, version = meta.get('Name'), meta.get('Version')
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name)
                or not isinstance(version, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.!+_-]*', version)):
            raise ValueError('wheel metadata name or version is invalid')
        name = re.sub(r'[-_.]+', '-', name).lower()
        if name in entries:
            raise ValueError(f'multiple wheels for package {name}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('wheel changed during hashing')
        entries[name] = f'{name}=={version} --hash=sha256:{digest.hexdigest()}'
    return ('# Generated from wheel metadata and bytes; install using --no-index and --require-hashes.\n'
            + '\n'.join(entries[name] for name in sorted(entries)) + '\n').encode('utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheelhouse', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new requirements file')
    args = parser.parse_args()
    try:
        data = lock(args.wheelhouse)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as output:
            output.write(data)
        print(f'locked {len(data.splitlines()) - 1} wheels; sha256={hashlib.sha256(data).hexdigest()}')
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(f'wheelhouse lock rejected: {error}') from error


if __name__ == '__main__':
    main()
