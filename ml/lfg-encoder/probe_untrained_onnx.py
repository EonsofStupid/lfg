#!/usr/bin/env python3
"""Probe an untrained local classifier's ONNX compatibility; never qualify it."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import evaluate
import export_onnx
import predict_model
import training_inputs


def shapes(values: list[str], maximum: int) -> list[tuple[int, int]]:
    result = []
    for value in values:
        parts = value.split(':')
        if len(parts) != 2 or not all(part.isascii() and part.isdigit() for part in parts):
            raise ValueError('shapes must be BATCH:TOKENS')
        batch, tokens = map(int, parts)
        if batch < 1 or tokens < 1 or tokens > maximum or (batch, tokens) in result:
            raise ValueError('shapes must be unique, positive and within the token limit')
        result.append((batch, tokens))
    if not result:
        raise ValueError('at least one shape is required')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'model-manifest', 'taxonomy', 'prediction-config', 'export-config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--max-length', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--shape', action='append', required=True)
    args = parser.parse_args()
    try:
        if args.max_length < 1 or args.seed < 0:
            raise ValueError('max length must be positive and seed nonnegative')
        cases = shapes(args.shape, args.max_length)
        model_proof = training_inputs.verify_model(args.model, args.model_manifest, args.revision)
        taxonomy, taxonomy_hash = evaluate.load_taxonomy(args.taxonomy)
        prediction, prediction_hash = predict_model.prediction_config(args.prediction_config)
        config, config_hash = export_onnx.read_config(args.export_config)
        if any(batch not in config['parity_batch_sizes'] for batch, _ in cases):
            raise ValueError('probe batch sizes must be selected in the export configuration')
        names = sorted(set.union(*taxonomy.values()))
        if set(prediction['label_thresholds']) - set(names):
            raise ValueError('prediction thresholds exceed taxonomy')
        if args.output.resolve().is_relative_to(args.model.resolve()) or args.output.exists():
            raise ValueError('output must be a new directory outside the model snapshot')
        import numpy as np
        import onnx
        os.environ['ORT_DISABLE_TELEMETRY'] = '1'  # Set before import, or model names and hashes go to Microsoft.
        import onnxruntime as ort
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(config['cpu_threads'])
        torch.manual_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        if tokenizer.unk_token is None or tokenizer.unk_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError('probe requires explicit unknown and padding tokens')
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(names), problem_type='multi_label_classification',
            local_files_only=True, trust_remote_code=False, ignore_mismatched_sizes=False).to('cpu').eval()
        example = tokenizer([tokenizer.unk_token], truncation=True, padding=True,
                            max_length=args.max_length, return_tensors='pt')
        inputs = list(example.keys())
        if not inputs or 'input_ids' not in inputs or set(inputs) - {'input_ids', 'token_type_ids', 'attention_mask'}:
            raise ValueError('unsupported tokenizer input contract')

        class Logits(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, *values):
                return self.inner(**dict(zip(inputs, values, strict=True))).logits

        wrapper = Logits(model).eval()
        args.output.mkdir(mode=0o700)
        onnx_path = args.output / 'model.onnx'
        axes = {name: {0: 'batch', 1: 'tokens'} for name in inputs}
        axes['logits'] = {0: 'batch'}
        torch.onnx.export(wrapper, tuple(example[name] for name in inputs), str(onnx_path),
                          input_names=inputs, output_names=['logits'], dynamic_axes=axes,
                          opset_version=config['opset_version'], dynamo=False, external_data=False)
        if onnx_path.stat().st_size > config['max_model_bytes']:
            raise ValueError('probe model exceeds configured size limit')
        graph = onnx.load(str(onnx_path), load_external_data=False)
        export_onnx.reject_external_tensors(graph, onnx)
        onnx.checker.check_model(graph, full_check=True)
        del graph
        options = ort.SessionOptions()
        options.intra_op_num_threads = config['cpu_threads']
        options.inter_op_num_threads = config['cpu_threads']
        session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])
        if [value.name for value in session.get_inputs()] != inputs:
            raise ValueError('ONNX inputs differ from tokenizer contract')
        thresholds = np.array([prediction['label_thresholds'].get(name, prediction['default_threshold']) for name in names])
        observed = []
        for batch, length in cases:
            ids = torch.full((batch, length), tokenizer.unk_token_id, dtype=torch.int64)
            mask = torch.ones_like(ids)
            if batch > 1 and length > 1:
                ids[-1, length // 2:] = tokenizer.pad_token_id
                mask[-1, length // 2:] = 0
            values = {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': torch.zeros_like(ids)}
            model_inputs = {name: values[name] for name in inputs}
            with torch.inference_mode():
                reference = model(**model_inputs).logits.numpy()
            actual = session.run(['logits'], {name: model_inputs[name].numpy() for name in inputs})[0]
            if (actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all()
                    or not np.allclose(actual, reference, atol=config['absolute_tolerance'], rtol=config['relative_tolerance'])):
                raise ValueError('ONNX probe differs from the source model')
            a = torch.sigmoid(torch.from_numpy(actual)).numpy() >= thresholds
            b = torch.sigmoid(torch.from_numpy(reference)).numpy() >= thresholds
            if not np.array_equal(a, b):
                raise ValueError('ONNX probe changes threshold decisions')
            observed.append({'batch': batch, 'tokens': length, 'padded': batch > 1 and length > 1,
                             'max_absolute_logit_error': float(np.max(np.abs(reference - actual)))})
        if (training_inputs.verify_model(args.model, args.model_manifest, args.revision) != model_proof
                or evaluate.load_taxonomy(args.taxonomy)[1] != taxonomy_hash
                or predict_model.prediction_config(args.prediction_config)[1] != prediction_hash
                or export_onnx.read_config(args.export_config)[1] != config_hash):
            raise ValueError('probe inputs changed during export')
        tokenizer.save_pretrained(args.output)
        (args.output / 'labels.json').write_text(json.dumps(names), encoding='utf-8')
        (args.output / 'taxonomy.json').write_bytes(args.taxonomy.read_bytes())
        (args.output / 'prediction-config.json').write_bytes(args.prediction_config.read_bytes())
        required = ['model.onnx', 'tokenizer.json', 'labels.json', 'taxonomy.json', 'prediction-config.json']
        manifest = {'schema_version': 'lfg.encoder-onnx-artifact.v2', 'promotion_eligible': False,
                    'probe_only': True, 'reason': 'untrained classification head; compatibility check only',
                    'source_model_manifest_sha256': model_proof['manifest_sha256'],
                    'export_config_sha256': config_hash, 'prediction_config_sha256': prediction_hash,
                    'taxonomy_sha256': taxonomy_hash, 'max_length': args.max_length,
                    'padding_side': tokenizer.padding_side, 'truncation_side': tokenizer.truncation_side,
                    'inputs': inputs, 'outputs': ['logits'], 'seed': args.seed, 'observed_shapes': observed,
                    'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'files_sha256': {name: export_onnx.file_digest(args.output / name) for name in required}}
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({'manifest_sha256': export_onnx.file_digest(args.output / 'manifest.json'),
                          'observed_shapes': observed, 'probe_only': True, 'promotion_eligible': False}))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as error:
        raise SystemExit(f'untrained ONNX probe rejected: {error}') from error


if __name__ == '__main__':
    main()
