"""Measure a private owned checkpoint's full-prefix and cached CPU inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

import owned_student
from owned_cached_decode import DecodeSession


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(checkpoint: Path, source: Path, length: int, threads: int, warmup: int, repeats: int) -> dict:
    if any(type(value) is not int or value < 1 for value in (length, threads, repeats)) or type(warmup) is not int or warmup < 0:
        raise ValueError('length, threads and repeats must be positive; warmup must be nonnegative')
    manifest_path, weights_path = checkpoint / 'manifest.json', checkpoint / 'weights.safetensors'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    weights_sha = sha(weights_path)
    model_code_sha = sha(Path(owned_student.__file__))
    if weights_sha != manifest['weights_sha256'] or model_code_sha != manifest['code_sha256']['owned_student.py']:
        raise ValueError('checkpoint weights or model implementation do not match manifest')
    model = owned_student.OwnedByteStudent(manifest['architecture']).eval()
    if length > model.context:
        raise ValueError('length exceeds model context')
    model.load_state_dict(load_file(str(weights_path), device='cpu'))
    with source.open('rb') as stream:
        raw = stream.read(length)
    if len(raw) != length:
        raise ValueError('source is shorter than the selected length')
    tokens = torch.tensor([list(raw)], dtype=torch.long)
    torch.set_num_threads(threads)

    def full_prefix():
        return torch.stack([model(tokens[:, :index + 1])[:, -1] for index in range(length)], dim=1)

    def cached():
        session = DecodeSession(model, 1)
        return torch.stack([session.step(tokens[:, index]) for index in range(length)], dim=1)

    samples = {'full_prefix_ms': [], 'cached_ms': []}
    with torch.inference_mode():
        expected, actual = full_prefix(), cached()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        difference = float((actual - expected).abs().max())
        for _ in range(warmup):
            full_prefix()
            cached()
        for index in range(repeats):
            order = (('full_prefix_ms', full_prefix), ('cached_ms', cached))
            if index % 2:
                order = tuple(reversed(order))
            for name, operation in order:
                start = time.perf_counter_ns()
                operation()
                samples[name].append((time.perf_counter_ns() - start) / 1e6)

    if (manifest_path.read_bytes() != manifest_bytes or sha(weights_path) != weights_sha
            or sha(Path(owned_student.__file__)) != model_code_sha):
        raise ValueError('checkpoint or model implementation changed during measurement')

    def summary(values):
        return {'median': statistics.median(values), 'p95': sorted(values)[math.ceil(.95 * len(values)) - 1]}
    return {'schema_version': 'lfg.owned-cached-decode-benchmark.v1',
            'checkpoint_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'weights_sha256': weights_sha,
            'source_prefix_sha256': hashlib.sha256(raw).hexdigest(), 'model_code_sha256': model_code_sha,
            'cached_code_sha256': sha(Path(__file__).with_name('owned_cached_decode.py')),
            'benchmark_code_sha256': sha(Path(__file__)), 'architecture': manifest['architecture'],
            'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'length_bytes': length, 'batch_size': 1, 'threads': threads, 'warmup': warmup, 'repeats': repeats,
            'max_absolute_logit_difference': difference,
            'elapsed_ms_for_all_prefixes': {name: summary(values) for name, values in samples.items()},
            'platform': platform.platform(), 'processor': platform.processor(), 'torch': torch.__version__,
            'native_runtime_measured': False, 'assessment_quality_measured': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    for name in ('length', 'threads', 'warmup', 'repeats'):
        parser.add_argument('--' + name, type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.source, args.length, args.threads, args.warmup, args.repeats), indent=2))
