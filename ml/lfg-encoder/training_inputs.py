#!/usr/bin/env python3
"""Bind a training run to a wheelhouse and a local model snapshot."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import sysconfig
import zipfile
from pathlib import Path

import lock_wheelhouse

MODEL_SCHEMA = 'lfg.training-model-snapshot.v1'


def sha_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('model file changed while hashing')
    return digest.hexdigest()


def snapshot(model_dir: Path) -> tuple[list[str], dict[str, str]]:
    def link_or_reparse(path: Path) -> bool:
        attributes = getattr(path.lstat(), 'st_file_attributes', 0)
        return path.is_symlink() or bool(attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0))

    if link_or_reparse(model_dir) or not model_dir.is_dir():
        raise ValueError('model snapshot must be a real directory')
    root = model_dir.resolve(strict=True)
    directories, files = [], {}
    for path in sorted(root.rglob('*')):
        if link_or_reparse(path):
            raise ValueError('model snapshot cannot contain links or reparse points')
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.append(relative)
        elif path.is_file():
            files[relative] = sha_file(path)
        else:
            raise ValueError('model snapshot has an unsupported filesystem entry')
    if not files:
        raise ValueError('model snapshot contains no files')
    return directories, files


def model_manifest(model_dir: Path, model_id: str, revision: str) -> bytes:
    if not isinstance(model_id, str) or not model_id.strip() or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('model ID and full lowercase revision are required')
    directories, files = snapshot(model_dir)
    value = {'schema_version': MODEL_SCHEMA, 'model_id': model_id, 'revision': revision,
             'directories': directories, 'files': files}
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8')


def verify_model(model_dir: Path, manifest: Path, revision: str) -> dict:
    raw = manifest.read_bytes()
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'model_id', 'revision', 'directories', 'files'}
            or value['schema_version'] != MODEL_SCHEMA or not isinstance(value['model_id'], str)
            or not value['model_id'].strip() or value['revision'] != revision
            or not re.fullmatch(r'[0-9a-f]{40}', revision)):
        raise ValueError('invalid model manifest or revision')
    directories, files = snapshot(model_dir)
    if value['directories'] != directories or value['files'] != files:
        raise ValueError('model snapshot does not match its manifest')
    if manifest.read_bytes() != raw:
        raise ValueError('model manifest changed during verification')
    return {'model_id': value['model_id'], 'revision': revision,
            'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'files': len(files)}


def normalize_name(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
        raise ValueError('invalid distribution name')
    return re.sub(r'[-_.]+', '-', name).lower()


def verify_wheel_payload(wheelhouse: Path) -> dict:
    """Compare non-generated installed wheel payloads with the supplied wheel bytes."""
    locations = {name: Path(value).resolve(strict=True) for name, value in sysconfig.get_paths().items()
                 if name in ('purelib', 'platlib', 'data')}
    if set(locations) != {'purelib', 'platlib', 'data'}:
        raise ValueError('interpreter does not expose required installation paths')
    digest = hashlib.sha256()
    checked_files = checked_bytes = skipped_records = skipped_scripts = 0
    destinations: set[str] = set()
    for path in sorted(wheelhouse.resolve(strict=True).iterdir()):
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len(set(names)) != len(names):
                raise ValueError('wheel contains duplicate paths')
            wheels = [item for item in infos if item.filename.endswith('.dist-info/WHEEL')
                      and item.filename.count('/') == 1]
            if len(wheels) != 1 or wheels[0].file_size > 4 * 1024 * 1024:
                raise ValueError('wheel requires one bounded WHEEL metadata file')
            metadata = BytesParser().parsebytes(archive.read(wheels[0]), headersonly=True)
            pure = metadata.get('Root-Is-Purelib', '').lower()
            if pure not in ('true', 'false'):
                raise ValueError('wheel has invalid Root-Is-Purelib metadata')
            for item in infos:
                if item.is_dir():
                    continue
                name = item.filename
                parts = name.split('/')
                if (not name or '\\' in name or any(part in ('', '.', '..') or ':' in part for part in parts)
                        or stat.S_ISLNK(item.external_attr >> 16)):
                    raise ValueError('wheel contains an unsafe file path')
                if len(parts) == 2 and parts[0].endswith('.dist-info') and parts[1] == 'RECORD':
                    skipped_records += 1
                    continue
                if parts[0].endswith('.data'):
                    if len(parts) < 3:
                        raise ValueError('wheel data path is incomplete')
                    kind = parts[1]
                    if kind == 'scripts':
                        skipped_scripts += 1
                        continue
                    if kind not in locations:
                        raise ValueError('wheel has unsupported data scheme')
                    relative = parts[2:]
                else:
                    kind = 'purelib' if pure == 'true' else 'platlib'
                    relative = parts
                root = locations[kind]
                target = root.joinpath(*relative)
                resolved = target.resolve(strict=True)
                if not resolved.is_relative_to(root) or not resolved.is_file():
                    raise ValueError('installed wheel file is outside its expected directory')
                identity = os.path.normcase(str(resolved))
                if identity in destinations:
                    raise ValueError('wheel payloads map to the same installed file')
                destinations.add(identity)
                actual = hashlib.sha256()
                expected = hashlib.sha256()
                with target.open('rb') as installed, archive.open(item) as original:
                    for chunk in iter(lambda: original.read(1024 * 1024), b''):
                        expected.update(chunk)
                    for chunk in iter(lambda: installed.read(1024 * 1024), b''):
                        actual.update(chunk)
                if actual.digest() != expected.digest():
                    raise ValueError(f'installed wheel payload differs: {path.name} / {name}')
                checked_files += 1
                checked_bytes += item.file_size
                digest.update(json.dumps([path.name, name, expected.hexdigest()], ensure_ascii=True,
                                         separators=(',', ':')).encode() + b'\n')
    if not checked_files:
        raise ValueError('no installed wheel payload files were checked')
    return {'schema_version': 'lfg.installed-wheel-payload.v1', 'files': checked_files,
            'bytes': checked_bytes, 'payload_sha256': digest.hexdigest(),
            'excluded_wheel_records': skipped_records, 'excluded_wheel_scripts': skipped_scripts}


def verify_environment(lock: Path, wheelhouse: Path, allowed_extra: list[str], verify_payload: bool = False) -> dict:
    raw = lock.read_bytes()
    try:
        actual = lock_wheelhouse.lock(wheelhouse)
    except zipfile.BadZipFile as error:
        raise ValueError('wheelhouse contains an invalid wheel') from error
    if actual != raw:
        raise ValueError('wheelhouse does not match the lock')
    expected = {}
    for line in raw.decode('utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([a-z0-9-]+)==([A-Za-z0-9][A-Za-z0-9.!+_-]*) --hash=sha256:([0-9a-f]{64})', line)
        if not match or match.group(1) in expected:
            raise ValueError('invalid wheelhouse lock entry')
        expected[match.group(1)] = match.group(2)
    if not expected:
        raise ValueError('wheelhouse lock is empty')
    extras = {normalize_name(name) for name in allowed_extra}
    if len(extras) != len(allowed_extra) or extras & expected.keys():
        raise ValueError('allowed extra names must be unique and absent from the lock')
    installed = {}
    for dist in importlib.metadata.distributions():
        name = normalize_name(dist.metadata.get('Name'))
        if name in installed:
            raise ValueError('multiple installed distributions share a name')
        installed[name] = dist.version
    if any(installed.get(name) != version for name, version in expected.items()):
        raise ValueError('installed package versions do not match the lock')
    if set(installed) - set(expected) - extras:
        raise ValueError('unlisted installed distributions are not allowed')
    try:
        payload = verify_wheel_payload(wheelhouse) if verify_payload else None
        if verify_payload and lock_wheelhouse.lock(wheelhouse) != raw:
            raise ValueError('wheelhouse changed during installed payload verification')
    except zipfile.BadZipFile as error:
        raise ValueError('wheelhouse changed or contains an invalid wheel') from error
    if lock.read_bytes() != raw:
        raise ValueError('wheelhouse lock changed during verification')
    fingerprint = json.dumps(installed, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'lock_sha256': hashlib.sha256(raw).hexdigest(),
            'installed_versions_sha256': hashlib.sha256(fingerprint).hexdigest(),
            'allowed_extra': sorted(extras), 'installed_files_verified': False,
            'locked_wheel_payload': payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True, help='new file outside the snapshot directory')
    args = parser.parse_args()
    try:
        if args.output.resolve().is_relative_to(args.model_dir.resolve()):
            raise ValueError('model manifest must be outside the snapshot directory')
        data = model_manifest(args.model_dir, args.model_id, args.revision)
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as output:
            output.write(data)
        print(json.dumps({'manifest_sha256': hashlib.sha256(data).hexdigest()}))
    except (OSError, ValueError) as error:
        raise SystemExit(f'model snapshot rejected: {error}') from error


if __name__ == '__main__':
    main()
