#!/usr/bin/env python3
"""Run an unpromoted native exemplar candidate against the LFG kernel on reviewed validation inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import node_evaluate
import node_exemplars
import node_review


def run(args: argparse.Namespace) -> dict:
    if args.max_items < 1 or args.max_input_bytes < 1 or not 0 < args.timeout_seconds < float('inf'):
        raise ValueError('invalid validation limits')
    tool_sha = node_evaluate.hash_file(Path(__file__))
    baseline_tool_sha = node_evaluate.hash_file(Path(node_evaluate.__file__))
    exemplar_tool_sha = node_evaluate.hash_file(Path(node_exemplars.__file__))
    binary_sha = node_evaluate.hash_file(args.binary)
    operations_path = os.environ.get('LFG_OPERATIONS_CONFIG')
    operations_sha = node_evaluate.hash_file(Path(operations_path)) if operations_path else None
    manifest_raw = (args.artifact / 'manifest.json').read_bytes()
    if node_review.digest(manifest_raw) != args.manifest_sha256:
        raise ValueError('encoder manifest digest mismatch')
    manifest = json.loads(manifest_raw)
    model_sha = node_evaluate.hash_file(args.artifact / 'model.onnx')
    tokenizer_sha = node_evaluate.hash_file(args.artifact / 'tokenizer.json')
    runtime_sha = node_evaluate.hash_file(args.runtime_config)
    config, config_sha = node_exemplars.read_config(args.exemplar_config)
    pack_raw = node_exemplars.bounded_read(args.pack, config['maxPackBytes'])
    pack_sha = node_review.digest(pack_raw)
    if (manifest.get('modelSha256') != model_sha or manifest.get('tokenizerSha256') != tokenizer_sha
            or pack_sha != args.pack_sha256):
        raise ValueError('encoder artifact or pack digest mismatch')
    audit, records = node_review.audit(args.input, args.snapshot, args.policy, args.node)
    if not audit['admissionEligible']:
        raise ValueError('node admission gates are not satisfied')
    pack = json.loads(pack_raw)
    if not isinstance(pack, dict) or not isinstance(pack.get('admission'), dict) or not isinstance(pack.get('exemplars'), list):
        raise ValueError('invalid exemplar pack')
    # Python build strings vary across hosts; Unicode and all audited inputs,
    # counts, decisions, policy and tool identities must still match.
    comparable = lambda value: {key: item for key, item in value.items() if key != 'pythonVersion'}
    expected_training = sorted((record['review']['itemSha256'], record['review']['choice'])
                               for record in records if record['review']['split'] == 'train')
    if (pack.get('schemaVersion') != 'lfg.node-exemplar-pack.v1' or pack.get('promotionEligible') is not False
            or pack.get('node') != args.node or pack.get('snapshotSha256') != audit['snapshotSha256']
            or comparable(pack['admission']) != comparable(audit)
            or pack.get('encoderConfigSha256') != node_exemplars.encoder_config_sha(config)
            or pack.get('encoderConfigSha256') != manifest.get('encoderConfigSha256')
            or pack.get('model', {}).get('manifest_sha256') != manifest.get('sourceModelManifestSha256')
            or sorted((entry['itemSha256'], entry['choice']) for entry in pack['exemplars']) != expected_training):
        raise ValueError('exemplar pack differs from the audited training inputs or encoder')
    validation = [record for record in records if record['review']['split'] == 'validation']
    if not validation or len(validation) > args.max_items:
        raise ValueError('invalid validation item count')
    baseline = node_evaluate.predictions(args.binary, args.snapshot, audit['snapshotSha256'], validation,
                                         args.timeout_seconds, args.max_items, args.max_input_bytes)
    expected = {record['review']['itemSha256']: record for record in validation}
    body = bytearray()
    for key, record in expected.items():
        line = node_review.encode({'id': key, 'text': record['item']['text']}) + b'\n'
        if len(body) + len(line) > args.max_input_bytes:
            raise ValueError('validation inputs exceed configured byte limit')
        body.extend(line)
    command = [str(args.binary.resolve(strict=True)), 'onnx-node-suggest', '--experimental', '--prediction-only',
        '--snapshot', str(args.snapshot), '--artifact', str(args.artifact), '--manifest-sha256', args.manifest_sha256,
        '--runtime-config', str(args.runtime_config), '--exemplar-pack', str(args.pack), '--pack-sha256', args.pack_sha256,
        '--exemplar-config', str(args.exemplar_config)]
    result = subprocess.run(command, input=bytes(body), capture_output=True, timeout=args.timeout_seconds, check=False)
    if result.returncode != 0:
        raise ValueError(f'native suggestion process failed with exit code {result.returncode}')
    if not result.stdout.endswith(b'\n'):
        raise ValueError('native suggestion output is incomplete')
    options = set(json.loads(args.snapshot.read_bytes())['nodes'][args.node]['options'])
    candidate = {}
    for line in result.stdout.splitlines():
        row = json.loads(line)
        if (not isinstance(row, dict) or set(row) != {'schemaVersion', 'itemSha256', 'inputSha256', 'snapshotSha256', 'choice'}
                or row['schemaVersion'] != 'lfg.node-prediction.v1'):
            raise ValueError('invalid native suggestion record')
        key = row['itemSha256']
        if (key not in expected or key in candidate or row['inputSha256'] != expected[key]['item']['inputSha256']
                or row['snapshotSha256'] != audit['snapshotSha256']
                or (row['choice'] is not None and row['choice'] not in options)):
            raise ValueError('native suggestion identity mismatch')
        candidate[key] = row['choice']
    if set(candidate) != set(expected):
        raise ValueError('native suggestion output does not cover validation items')
    metrics = node_evaluate.score(validation, baseline, args.node, options, candidate)
    after, _ = node_review.audit(args.input, args.snapshot, args.policy, args.node)
    if (after != audit or node_evaluate.hash_file(args.binary) != binary_sha
            or (args.artifact / 'manifest.json').read_bytes() != manifest_raw
            or node_evaluate.hash_file(args.artifact / 'model.onnx') != model_sha
            or node_evaluate.hash_file(args.artifact / 'tokenizer.json') != tokenizer_sha
            or node_exemplars.bounded_read(args.pack, config['maxPackBytes']) != pack_raw
            or node_evaluate.hash_file(args.runtime_config) != runtime_sha
            or node_exemplars.read_config(args.exemplar_config)[1] != config_sha
            or node_evaluate.hash_file(Path(__file__)) != tool_sha
            or node_evaluate.hash_file(Path(node_evaluate.__file__)) != baseline_tool_sha
            or node_evaluate.hash_file(Path(node_exemplars.__file__)) != exemplar_tool_sha
            or (operations_path and node_evaluate.hash_file(Path(operations_path)) != operations_sha)):
        raise ValueError('inputs changed during native evaluation')
    return {'schemaVersion': 'lfg.native-node-validation.v1', 'split': 'validation', 'node': args.node,
            'admission': audit, 'binarySha256': binary_sha, 'encoderManifestSha256': args.manifest_sha256,
            'packSha256': pack_sha, 'runtimeConfigSha256': runtime_sha, 'exemplarConfigSha256': config_sha,
            'toolSha256': tool_sha, 'baselineToolSha256': baseline_tool_sha, 'exemplarToolSha256': exemplar_tool_sha,
            'operationsConfigSha256': operations_sha, 'maxItems': args.max_items,
            'maxInputBytes': args.max_input_bytes, 'timeoutSeconds': args.timeout_seconds,
            'candidateProcessObserved': True, 'candidateAuthenticated': False,
            'qualificationEstablished': False, 'metrics': metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs=2, action='append', required=True, metavar=('QUEUE', 'LEDGER'))
    for name in ('snapshot', 'policy', 'binary', 'artifact', 'runtime-config', 'pack', 'exemplar-config'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('node', 'manifest-sha256', 'pack-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-items', type=int, required=True)
    parser.add_argument('--max-input-bytes', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=float, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=True, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f'native exemplar evaluation: {error}', file=sys.stderr)
        sys.exit(1)
