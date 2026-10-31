#!/usr/bin/env python3
"""Compare an offline native encoder probe against the selected frozen Python encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import export_onnx
import node_evaluate
import node_exemplars
import node_review
import training_inputs


def run(args: argparse.Namespace) -> dict:
    if args.max_items < 1 or args.max_input_bytes < 1 or not 0 < args.timeout_seconds < float('inf'):
        raise ValueError('invalid input or timeout limits')
    config, config_sha = node_exemplars.read_config(args.config)
    export_config, export_sha = export_onnx.read_config(args.export_config)
    items, ledger, ledger_raw = node_review.load_ledger(args.queue, args.snapshot, args.ledger)
    if len(items) > args.max_items:
        raise ValueError('queue exceeds configured item limit')
    model_proof = training_inputs.verify_model(args.model, args.model_manifest, args.revision)
    manifest_raw = (args.artifact / 'manifest.json').read_bytes()
    if node_review.digest(manifest_raw) != args.manifest_sha256:
        raise ValueError('manifest digest mismatch')
    manifest = json.loads(manifest_raw)
    if (manifest.get('schemaVersion') != 'lfg.node-onnx-encoder.v1' or manifest.get('promotionEligible') is not False
            or manifest.get('sourceModelManifestSha256') != model_proof['manifest_sha256']
            or manifest.get('encoderConfigSha256') != node_exemplars.encoder_config_sha(config)
            or manifest.get('exportConfigSha256') != export_sha):
        raise ValueError('artifact does not match selected model or configuration')
    model_hash = node_evaluate.hash_file(args.artifact / 'model.onnx')
    tokenizer_hash = node_evaluate.hash_file(args.artifact / 'tokenizer.json')
    binary_hash = node_evaluate.hash_file(args.binary)
    runtime_hash = node_evaluate.hash_file(args.runtime_config)
    tool_hash = node_evaluate.hash_file(Path(__file__))
    encoder_hash = node_evaluate.hash_file(Path(node_exemplars.__file__))
    verifier_hash = node_evaluate.hash_file(Path(node_evaluate.__file__))
    if model_hash != manifest['modelSha256'] or tokenizer_hash != manifest['tokenizerSha256']:
        raise ValueError('artifact file digest mismatch')
    body = bytearray()
    expected = {}
    for item in items:
        item_id = node_review.identity(item)
        expected[item_id] = item
        line = node_review.encode({'id': item_id, 'text': item['text']}) + b'\n'
        if len(body) + len(line) > args.max_input_bytes:
            raise ValueError('parity input exceeds configured byte limit')
        body.extend(line)
    command = [str(args.binary.resolve(strict=True)), 'onnx-node-embedding', '--artifact', str(args.artifact),
               '--manifest-sha256', args.manifest_sha256, '--runtime-config', str(args.runtime_config), '--include-embedding']
    result = subprocess.run(command, input=bytes(body), capture_output=True, timeout=args.timeout_seconds, check=False)
    if result.returncode != 0:
        raise ValueError(f'native encoder failed with exit code {result.returncode}')
    if not result.stdout.endswith(b'\n'):
        raise ValueError('native encoder output is incomplete')
    outputs = {}
    for line in result.stdout.splitlines():
        row = json.loads(line)
        if (not isinstance(row, dict) or set(row) != {'schemaVersion', 'id', 'inputSha256', 'manifestSha256',
                                                       'runtimeConfigSha256', 'dimension', 'embedding'}
                or row['schemaVersion'] != 'lfg.node-embedding-probe.v1'):
            raise ValueError('invalid native encoder output')
        key = row['id']
        if (key not in expected or key in outputs or row['inputSha256'] != expected[key]['inputSha256']
                or row['manifestSha256'] != args.manifest_sha256 or row['runtimeConfigSha256'] != runtime_hash
                or row['dimension'] != manifest['dimension']):
            raise ValueError('native encoder output identity mismatch')
        outputs[key] = row['embedding']
    if set(outputs) != set(expected):
        raise ValueError('native output does not cover the queue')
    import numpy as np
    encoder = node_exemplars.Encoder(args.model, config)
    maximum_error = 0.0
    lengths = []
    for key, item in expected.items():
        vectors, counts = encoder.encode([item['text']])
        python_vector = np.asarray(vectors[0], dtype=np.float32)
        native_vector = np.asarray(outputs[key], dtype=np.float32)
        if (native_vector.shape != python_vector.shape or not np.isfinite(native_vector).all()
                or not np.allclose(native_vector, python_vector, atol=export_config['absolute_tolerance'],
                                   rtol=export_config['relative_tolerance'])):
            raise ValueError('native vector differs from frozen Python encoder')
        maximum_error = max(maximum_error, float(np.max(np.abs(native_vector - python_vector))))
        lengths.extend(counts)
    if (training_inputs.verify_model(args.model, args.model_manifest, args.revision) != model_proof
            or node_review.load_ledger(args.queue, args.snapshot, args.ledger)[2] != ledger_raw
            or node_exemplars.read_config(args.config)[1] != config_sha
            or export_onnx.read_config(args.export_config)[1] != export_sha
            or (args.artifact / 'manifest.json').read_bytes() != manifest_raw
            or node_evaluate.hash_file(args.artifact / 'model.onnx') != model_hash
            or node_evaluate.hash_file(args.artifact / 'tokenizer.json') != tokenizer_hash
            or node_evaluate.hash_file(args.binary) != binary_hash
            or node_evaluate.hash_file(args.runtime_config) != runtime_hash
            or node_evaluate.hash_file(Path(__file__)) != tool_hash
            or node_evaluate.hash_file(Path(node_exemplars.__file__)) != encoder_hash
            or node_evaluate.hash_file(Path(node_evaluate.__file__)) != verifier_hash):
        raise ValueError('inputs changed during parity check')
    return {'schemaVersion': 'lfg.node-native-parity.v1', 'items': len(items), 'dimension': manifest['dimension'],
            'minTokens': min(lengths), 'maxTokens': max(lengths), 'maxAbsoluteError': maximum_error,
            'absoluteTolerance': export_config['absolute_tolerance'], 'relativeTolerance': export_config['relative_tolerance'],
            'queueSha256': ledger['queueSha256'], 'manifestSha256': args.manifest_sha256,
            'binarySha256': binary_hash, 'runtimeConfigSha256': runtime_hash, 'toolSha256': tool_hash,
            'encoderToolSha256': encoder_hash, 'hashVerifierSha256': verifier_hash,
            'labelsUsed': False, 'vectorsSaved': False, 'qualityEvidence': False, 'promotionEligible': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('queue', 'ledger', 'snapshot', 'model', 'model-manifest', 'config', 'export-config',
                 'artifact', 'binary', 'runtime-config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--max-items', type=int, required=True)
    parser.add_argument('--max-input-bytes', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=float, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=True, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f'node encoder parity: {error}', file=sys.stderr)
        sys.exit(1)
