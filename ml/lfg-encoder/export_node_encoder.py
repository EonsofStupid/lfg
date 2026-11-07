#!/usr/bin/env python3
"""Export an unpromoted frozen node encoder and check local CPU parity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import export_onnx
import node_evaluate
import node_exemplars
import node_review
import training_inputs


def run(args: argparse.Namespace) -> dict:
    config, config_sha = node_exemplars.read_config(args.config)
    export_config, export_sha = export_onnx.read_config(args.export_config)
    if args.max_parity_items < 1 or export_config['cpu_threads'] != 1:
        raise ValueError('positive parity limit and single-threaded ONNX runtime are required')
    items, ledger, ledger_raw = node_review.load_ledger(args.queue, args.snapshot, args.ledger)
    if len(items) > args.max_parity_items:
        raise ValueError('parity queue exceeds configured item limit')
    if args.output.exists() or args.output.resolve().is_relative_to(args.model.resolve()):
        raise ValueError('output must be a new directory outside the model snapshot')
    proof = training_inputs.verify_model(args.model, args.model_manifest, args.revision)
    exporter_sha = node_review.digest(Path(__file__).read_bytes())
    encoder_sha = node_review.digest(Path(node_exemplars.__file__).read_bytes())
    verifier_sha = node_review.digest(Path(node_evaluate.__file__).read_bytes())
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    encoder = node_exemplars.Encoder(args.model, config)
    token = encoder.tokenizer.unk_token
    if token is None:
        raise ValueError('tokenizer must provide an unknown token for graph tracing')
    trace = encoder.tokenizer([token], padding=True, truncation=False, return_tensors='pt')
    inputs = list(trace.keys())
    if not inputs or 'input_ids' not in inputs or 'attention_mask' not in inputs or set(inputs) - {'input_ids', 'token_type_ids', 'attention_mask'}:
        raise ValueError('unsupported tokenizer input contract')

    class Embedding(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *values):
            batch = dict(zip(inputs, values, strict=True))
            hidden = self.model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return pooled / torch.linalg.vector_norm(pooled, dim=1, keepdim=True)

    args.output.mkdir(mode=0o700)
    model_path = args.output / 'model.onnx'
    torch.onnx.export(Embedding(encoder.model).eval(), tuple(trace[name] for name in inputs), str(model_path),
        input_names=inputs, output_names=['embedding'],
        dynamic_axes={**{name: {1: 'tokens'} for name in inputs}, 'embedding': {}},
        opset_version=export_config['opset_version'], dynamo=False, external_data=False)
    if model_path.stat().st_size > export_config['max_model_bytes']:
        raise ValueError('exported model exceeds configured byte limit')
    model_sha = node_evaluate.hash_file(model_path)
    graph = onnx.load(str(model_path), load_external_data=False)
    export_onnx.reject_external_tensors(graph, onnx)
    onnx.checker.check_model(graph, full_check=True)
    del graph
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=['CPUExecutionProvider'])
    if [value.name for value in session.get_inputs()] != inputs or [value.name for value in session.get_outputs()] != ['embedding']:
        raise ValueError('ONNX graph input/output names differ from the export contract')
    max_error = 0.0
    token_lengths = []
    dimension = None
    for item in items:
        vectors, lengths = encoder.encode([item['text']])
        batch = encoder.tokenizer([item['text']], padding=True, truncation=False, return_tensors='pt')
        native = session.run(['embedding'], {name: batch[name].numpy() for name in inputs})[0]
        expected = np.asarray(vectors, dtype=np.float32)
        if native.shape != expected.shape or not np.isfinite(native).all():
            raise ValueError('ONNX output has invalid shape or non-finite values')
        if not np.allclose(native, expected, atol=export_config['absolute_tolerance'], rtol=export_config['relative_tolerance']):
            raise ValueError('ONNX embedding does not match the frozen Python encoder')
        max_error = max(max_error, float(np.max(np.abs(native - expected))))
        token_lengths.extend(lengths)
        dimension = expected.shape[1]
    encoder.tokenizer.backend_tokenizer.no_truncation()
    encoder.tokenizer.backend_tokenizer.no_padding()
    tokenizer_raw = encoder.tokenizer.backend_tokenizer.to_str().encode('utf-8')
    node_review.create_new(args.output / 'tokenizer.json', tokenizer_raw)
    if (training_inputs.verify_model(args.model, args.model_manifest, args.revision) != proof
            or node_review.load_ledger(args.queue, args.snapshot, args.ledger)[2] != ledger_raw
            or node_exemplars.read_config(args.config)[1] != config_sha
            or export_onnx.read_config(args.export_config)[1] != export_sha
            or node_evaluate.hash_file(model_path) != model_sha
            or node_review.digest(Path(__file__).read_bytes()) != exporter_sha
            or node_review.digest(Path(node_exemplars.__file__).read_bytes()) != encoder_sha
            or node_review.digest(Path(node_evaluate.__file__).read_bytes()) != verifier_sha):
        raise ValueError('input changed during export')
    manifest = {'schemaVersion': 'lfg.node-onnx-encoder.v1', 'promotionEligible': False,
                'modelSha256': model_sha, 'tokenizerSha256': node_review.digest(tokenizer_raw),
                'sourceModelManifestSha256': proof['manifest_sha256'],
                'encoderConfigSha256': node_exemplars.encoder_config_sha(config),
                'exportConfigSha256': export_sha, 'inputs': inputs, 'output': 'embedding',
                'dimension': dimension, 'maxTokens': config['maxTokens'], 'pooling': config['pooling']}
    manifest_raw = node_review.encode(manifest) + b'\n'
    node_review.create_new(args.output / 'manifest.json', manifest_raw)
    return {'schemaVersion': 'lfg.node-encoder-onnx-probe.v1', 'manifestSha256': node_review.digest(manifest_raw),
            'modelSha256': manifest['modelSha256'], 'sourceModel': proof, 'queueSha256': ledger['queueSha256'],
            'configSha256': config_sha, 'encoderConfigSha256': manifest['encoderConfigSha256'],
            'exportConfigSha256': export_sha, 'exporterSha256': exporter_sha, 'hashVerifierSha256': verifier_sha,
            'items': len(items), 'minTokens': min(token_lengths), 'maxTokens': max(token_lengths), 'dimension': dimension,
            'maxAbsoluteError': max_error, 'absoluteTolerance': export_config['absolute_tolerance'],
            'relativeTolerance': export_config['relative_tolerance'], 'torch': torch.__version__, 'onnx': onnx.__version__,
            'onnxruntime': ort.__version__, 'labelsUsed': False, 'vectorsSaved': False,
            'qualityEvidence': False, 'promotionEligible': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('queue', 'ledger', 'snapshot', 'model', 'model-manifest', 'config', 'export-config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--max-parity-items', type=int, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=True, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as error:
        print(f'node encoder export: {error}', file=sys.stderr)
        sys.exit(1)
