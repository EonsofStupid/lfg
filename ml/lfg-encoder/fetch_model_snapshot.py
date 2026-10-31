#!/usr/bin/env python3
"""Fetch selected public model files at an immutable revision and verify Hub identities."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import training_inputs


def fetch(model_id: str, revision: str, files: list[str], cache: Path,
          output: Path, manifest: Path, provenance: Path, endpoint: str = 'https://huggingface.co') -> dict:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*', model_id):
        raise ValueError('model ID must be an owner/name pair')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('revision must be a full lowercase commit hash')
    url = urlsplit(endpoint)
    if (url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment
            or '\\' in endpoint or any(part in ('.', '..') for part in url.path.split('/'))):
        raise ValueError('model endpoint must be an HTTPS URL without credentials, query, or fragment')
    endpoint = endpoint.rstrip('/')
    if not files or len(set(files)) != len(files):
        raise ValueError('select unique model files')
    for name in files:
        p = PurePosixPath(name)
        if (not name or p.is_absolute() or '\\' in name or ':' in name
                or any(part in ('', '.', '..') for part in name.split('/'))):
            raise ValueError('unsafe model file path')
    output = output.resolve()
    manifest, provenance = manifest.resolve(), provenance.resolve()
    cache = cache.resolve()
    if (manifest == provenance or any(p == output or p.is_relative_to(output) for p in (manifest, provenance))
            or cache == output or cache.is_relative_to(output) or output.is_relative_to(cache)
            or output.exists() or manifest.exists() or provenance.exists()):
        raise ValueError('output and evidence paths must be new and outside the snapshot')
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi(endpoint=endpoint).model_info(model_id, revision=revision, token=False, files_metadata=True)
    if info.sha != revision or info.private or info.gated:
        raise ValueError('model revision must resolve to a public, ungated repository')
    metadata = {item.rfilename: item for item in info.siblings}
    if len(metadata) != len(info.siblings):
        raise ValueError('model source returned duplicate file names')
    if any(name not in metadata for name in files):
        raise ValueError('selected model file is absent from the revision')
    output.mkdir(mode=0o700)
    evidence = {}
    for name in files:
        item = metadata[name]
        if type(item.size) is not int or item.size < 0 or not re.fullmatch(r'[0-9a-f]{40}', item.blob_id or ''):
            raise ValueError('model source metadata is incomplete')
        lfs_sha = item.lfs.sha256 if item.lfs else None
        if item.lfs and (item.lfs.size != item.size or not re.fullmatch(r'[0-9a-f]{64}', lfs_sha or '')):
            raise ValueError('invalid model LFS metadata')
        source = Path(hf_hub_download(repo_id=model_id, filename=name, revision=revision,
                                      cache_dir=str(cache), token=False, endpoint=endpoint))
        destination = output.joinpath(*PurePosixPath(name).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as source_file, destination.open('xb') as target:
            shutil.copyfileobj(source_file, target, length=1024 * 1024)
        if destination.stat().st_size != item.size:
            raise ValueError('downloaded model file size differs from source metadata')
        sha = hashlib.sha256()
        git = hashlib.sha1(f'blob {item.size}\0'.encode())
        with destination.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                sha.update(chunk)
                git.update(chunk)
        if (lfs_sha and sha.hexdigest() != lfs_sha) or (not lfs_sha and git.hexdigest() != item.blob_id):
            raise ValueError('downloaded model file differs from source metadata')
        evidence[name] = {'size': item.size, 'sha256': sha.hexdigest(),
                          'source_git_blob': item.blob_id, 'source_lfs_sha256': lfs_sha}
    manifest_bytes = training_inputs.model_manifest(output, model_id, revision)
    with manifest.open('xb') as stream:
        stream.write(manifest_bytes)
    proof = {'schema_version': 'lfg.hub-model-provenance.v1', 'endpoint': endpoint, 'model_id': model_id, 'revision': revision,
             'files': evidence, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
             'fetcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with provenance.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(proof, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return {'revision': revision, 'files': len(files), 'manifest_sha256': proof['manifest_sha256']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--endpoint', default='https://huggingface.co', help='HTTPS model repository endpoint')
    parser.add_argument('--file', action='append', required=True)
    for name in ('cache', 'output', 'manifest', 'provenance'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(fetch(args.model_id, args.revision, args.file, args.cache, args.output,
                               args.manifest, args.provenance, args.endpoint), indent=2))
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(f'model snapshot rejected: {error}') from error


if __name__ == '__main__':
    main()
