#!/usr/bin/env python3
"""Exercise assessment-head CUDA mechanics using fixtures, never project data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import assessment_train
import training_inputs


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'model-manifest', 'config', 'environment-lock', 'wheelhouse', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--device-index', type=int, required=True)
    parser.add_argument('--cuda-memory-fraction', type=float, required=True)
    parser.add_argument('--allowed-extra', action='append', default=[])
    args = parser.parse_args()
    try:
        if (args.device_index < 0 or not math.isfinite(args.cuda_memory_fraction)
                or not 0 < args.cuda_memory_fraction <= 1):
            raise ValueError('invalid CUDA device or allocator fraction')
        if args.output.exists() or args.output.resolve().is_relative_to(args.model.resolve()):
            raise ValueError('probe output must be new and outside the model snapshot')
        code = [Path(__file__), Path(assessment_train.__file__), Path(training_inputs.__file__),
                Path(training_inputs.lock_wheelhouse.__file__)]
        code_hashes = {item.name: sha(item) for item in code}
        config, config_sha = assessment_train.load_config(args.config)
        config['device'] = 'cuda'
        model = training_inputs.verify_model(args.model, args.model_manifest, args.revision)
        environment = training_inputs.verify_environment(args.environment_lock, args.wheelhouse,
                                                          args.allowed_extra, verify_payload=True)
        import torch
        if not torch.cuda.is_available() or args.device_index >= torch.cuda.device_count():
            raise ValueError('requested CUDA device is unavailable')
        torch.cuda.set_device(args.device_index)
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, args.device_index)
        torch.cuda.reset_peak_memory_stats(args.device_index)

        queries = ['fixture query alpha', 'fixture query beta']
        context = 'fixture project context'
        candidates = ['fixture option alpha', 'fixture option beta']
        choices = [(queries[0], context, candidates[0], 1), (queries[0], context, candidates[1], 0),
                   (queries[1], context, candidates[0], 0), (queries[1], context, candidates[1], 1)]
        examples = {'complexity': choices, 'models': choices, 'effort': choices,
                    'agent_count': [(queries[0], context, None, 0), (queries[1], context, None, 2)]}
        started = time.perf_counter()
        encoder = assessment_train.Encoder(args.model, config)
        vectors = encoder.encode(sorted(set(queries + [context] + candidates)))
        trained = assessment_train.fit(examples, vectors, config)
        torch.cuda.synchronize(args.device_index)
        elapsed_seconds = time.perf_counter() - started
        if any(not math.isfinite(weight) for weights in trained['weights'].values() for weight in weights):
            raise ValueError('fixture heads produced non-finite weights')
        if (training_inputs.verify_model(args.model, args.model_manifest, args.revision) != model
                or assessment_train.load_config(args.config)[1] != config_sha
                or any(sha(item) != code_hashes[item.name] for item in code)
                or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != encoder.versions['cublas_workspace_config']
                or os.environ.get('CUDA_VISIBLE_DEVICES') != encoder.versions['cuda_visible_devices']):
            raise ValueError('model, configuration, code or CUDA environment changed during probe')
        after = training_inputs.verify_environment(args.environment_lock, args.wheelhouse, args.allowed_extra)
        for key in ('lock_sha256', 'installed_versions_sha256', 'allowed_extra'):
            if after[key] != environment[key]:
                raise ValueError('environment identity changed during probe')
        report = {'schema_version': 'lfg.assessment-cuda-fixture-probe.v1', 'model': model,
                  'environment_before_probe': environment, 'config_sha256': config_sha,
                  'device_override': 'cuda', 'device_index': args.device_index,
                  'cuda_memory_fraction': args.cuda_memory_fraction,
                  'cuda_device_name': torch.cuda.get_device_name(args.device_index),
                  'cuda_capability': torch.cuda.get_device_capability(args.device_index),
                  'encoder_versions': encoder.versions, 'code_sha256': code_hashes,
                  'dimension': trained['dimension'], 'heads': sorted(trained['weights']),
                  'elapsed_seconds': elapsed_seconds,
                  'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(args.device_index),
                  'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(args.device_index),
                  'fixture_only': True, 'weights_saved': False, 'quality_evidence': False,
                  'promotion_eligible': False}
        raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
        print(json.dumps({'schema_version': 'lfg.assessment-cuda-fixture-result.v1',
                          'report_sha256': hashlib.sha256(raw).hexdigest(), 'fixture_only': True,
                          'quality_evidence': False, 'promotion_eligible': False}, sort_keys=True))
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        raise SystemExit(f'assessment CUDA fixture probe rejected: {error}') from error


if __name__ == '__main__':
    main()
