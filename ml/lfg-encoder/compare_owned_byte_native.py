"""Compare a native owned byte probe with its PyTorch checkpoint in memory."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file

import owned_student


def compare(binary: Path, checkpoint: Path, source: Path, lengths: list[int], *,
            absolute_tolerance: float, relative_tolerance: float, timeout: float,
            artifact: Path | None = None, manifest_sha256: str | None = None,
            runtime_config: Path | None = None) -> dict:
    if (not lengths or any(type(n) is not int or n < 1 for n in lengths) or len(set(lengths)) != len(lengths)
            or any(not math.isfinite(value) or value < 0 for value in (absolute_tolerance, relative_tolerance))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('invalid comparison limits')
    manifest_raw = (checkpoint / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    weights_path = checkpoint / 'weights.safetensors'
    weights = weights_path.read_bytes()
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    if (sha(weights) != manifest['weights_sha256'] or
            sha(Path(owned_student.__file__).read_bytes()) != manifest['code_sha256']['owned_student.py']):
        raise ValueError('checkpoint weights or model code do not match')
    model = owned_student.OwnedByteStudent(manifest['architecture']).eval()
    if max(lengths) > model.context:
        raise ValueError('comparison length exceeds context')
    model.load_state_dict(load_file(str(weights_path), device='cpu'))
    with source.open('rb') as stream:
        raw = stream.read(max(lengths))
    if len(raw) != max(lengths):
        raise ValueError('comparison source is too short')
    try:
        texts = [raw[:length].decode('utf-8') for length in lengths]
    except UnicodeDecodeError:
        raise ValueError('each selected byte prefix must end at a UTF-8 boundary') from None
    args = [str(binary)]
    if artifact is None:
        if manifest_sha256 is not None:
            raise ValueError('embedded mode does not accept a manifest digest')
        args += ['embedded-owned-byte-probe']
    else:
        if not manifest_sha256 or runtime_config is None:
            raise ValueError('directory mode requires manifest digest and runtime config')
        args += ['owned-byte-probe', '--artifact', str(artifact), '--manifest-sha256', manifest_sha256]
    if runtime_config is not None:
        args += ['--runtime-config', str(runtime_config)]
    args += ['--experimental', '--include-logits']
    payload = ''.join(json.dumps({'id': str(i), 'text': text}, ensure_ascii=False) + '\n'
                      for i, text in enumerate(texts))
    run = subprocess.run(args, input=payload.encode('utf-8'), capture_output=True, timeout=timeout)
    if run.returncode:
        raise ValueError(f'native probe failed with status {run.returncode}: {run.stderr.decode("utf-8", "replace").strip()}')
    rows = [json.loads(line) for line in run.stdout.splitlines()]
    if len(rows) != len(lengths):
        raise ValueError('native output count differs')
    maximum = 0.0
    with torch.inference_mode():
        for index, (length, row) in enumerate(zip(lengths, rows, strict=True)):
            if row.get('id') != str(index) or row.get('input_bytes') != length:
                raise ValueError('native output does not match input')
            expected = model(torch.tensor([list(raw[:length])], dtype=torch.long))[0, -1]
            actual = torch.tensor(row['logits'], dtype=expected.dtype)
            torch.testing.assert_close(actual, expected, rtol=relative_tolerance, atol=absolute_tolerance)
            if row['next_byte'] != int(expected.argmax()):
                raise ValueError('native next-byte choice differs')
            maximum = max(maximum, float((actual - expected).abs().max()))
    if weights_path.read_bytes() != weights or (checkpoint / 'manifest.json').read_bytes() != manifest_raw:
        raise ValueError('checkpoint changed during comparison')
    return {'schema_version': 'lfg.owned-byte-native-parity.v1', 'binary_sha256': sha(binary.read_bytes()),
            'source_prefix_sha256': sha(raw), 'checkpoint_manifest_sha256': sha(manifest_raw),
            'lengths': lengths, 'max_absolute_difference': maximum, 'embedded': artifact is None,
            'assessment_quality_checked': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'checkpoint', 'source'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--length', type=int, action='append', required=True)
    parser.add_argument('--absolute-tolerance', type=float, required=True)
    parser.add_argument('--relative-tolerance', type=float, required=True)
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('--artifact', type=Path)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--runtime-config', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.binary, args.checkpoint, args.source, args.length,
          absolute_tolerance=args.absolute_tolerance, relative_tolerance=args.relative_tolerance,
          timeout=args.timeout, artifact=args.artifact, manifest_sha256=args.manifest_sha256,
          runtime_config=args.runtime_config)))
