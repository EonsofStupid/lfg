#!/usr/bin/env python3
"""Compare an explicitly untrained probe artifact in native LFG and ONNX Runtime."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--runtime-config', type=Path, required=True)
    parser.add_argument('--unknown-repetitions', type=int, action='append', required=True)
    parser.add_argument('--score-tolerance', type=float, required=True)
    parser.add_argument('--timeout-seconds', type=float, required=True)
    args = parser.parse_args()
    if (not args.unknown_repetitions or any(n < 1 for n in args.unknown_repetitions)
            or not math.isfinite(args.score_tolerance) or args.score_tolerance < 0
            or not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0):
        raise SystemExit('invalid probe limits')
    manifest_bytes = (args.artifact / 'manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != args.manifest_sha256:
        raise SystemExit('probe manifest digest mismatch')
    manifest = json.loads(manifest_bytes)
    if manifest.get('probe_only') is not True or manifest.get('promotion_eligible') is not False:
        raise SystemExit('this tool requires an explicitly untrained probe artifact')
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.artifact, local_files_only=True, trust_remote_code=False)
    if tokenizer.unk_token is None:
        raise SystemExit('probe tokenizer has no unknown token')
    cases = [{'id': f'probe-{i}', 'text': ' '.join([tokenizer.unk_token] * n)}
             for i, n in enumerate(args.unknown_repetitions)]
    command = [str(args.binary), 'onnx-predict', '--artifact', str(args.artifact),
               '--manifest-sha256', args.manifest_sha256, '--runtime-config', str(args.runtime_config),
               '--include-scores']
    result = subprocess.run(command, input=''.join(json.dumps(row) + '\n' for row in cases),
                            text=True, capture_output=True, timeout=args.timeout_seconds, check=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    if len(rows) != len(cases):
        raise SystemExit('native response count differs from probe count')
    options = ort.SessionOptions()
    runtime_config = json.loads(args.runtime_config.read_text(encoding='utf-8'))
    options.intra_op_num_threads = options.inter_op_num_threads = runtime_config['cpuThreads']
    session = ort.InferenceSession(str(args.artifact / 'model.onnx'), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    names = json.loads((args.artifact / 'labels.json').read_text(encoding='utf-8'))
    config = json.loads((args.artifact / 'prediction-config.json').read_text(encoding='utf-8'))
    thresholds = np.array([config['label_thresholds'].get(name, config['default_threshold']) for name in names])
    observations = []
    for case, row in zip(cases, rows, strict=True):
        encoded = tokenizer.backend_tokenizer.encode(case['text'], add_special_tokens=True)
        values = {'input_ids': encoded.ids, 'attention_mask': encoded.attention_mask,
                  'token_type_ids': encoded.type_ids}
        inputs = {name: np.array([values[name]], dtype=np.int64) for name in manifest['inputs']}
        logits = session.run(['logits'], inputs)[0][0]
        reference = 1 / (1 + np.exp(-logits))
        scores = row['scores']['values']
        if row['id'] != case['id'] or set(scores) != set(names):
            raise SystemExit('native response identity or labels differ')
        actual = np.array([scores[name] for name in names])
        error = float(np.max(np.abs(reference - actual)))
        same = bool(np.array_equal(reference >= thresholds, actual >= thresholds))
        if not np.isfinite(reference).all() or not np.isfinite(actual).all() or error > args.score_tolerance or not same:
            raise SystemExit('native scores or threshold decisions differ from ONNX Runtime')
        observations.append({'tokens': len(encoded.ids), 'max_absolute_score_error': error,
                             'threshold_decisions_match': same})
    print(json.dumps({'probe_only': True, 'label_count': len(names), 'cases': observations}))


if __name__ == '__main__':
    main()
