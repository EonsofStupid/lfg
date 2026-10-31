#!/usr/bin/env python3
"""Check a local model's execution path without training data or quality claims."""
import argparse
import hashlib
import json
from pathlib import Path

import evaluate
import training_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-manifest', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--taxonomy', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--tokens', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, help='new file for the aggregate probe report')
    args = parser.parse_args()
    try:
        if args.threads < 1 or args.tokens < 1 or args.seed < 0:
            raise ValueError('threads and tokens must be positive and seed must be nonnegative')
        if args.output and (args.output.resolve().is_relative_to(args.model.resolve()) or args.output.exists()):
            raise ValueError('probe output must be a new file outside the model snapshot')
        model_proof = training_inputs.verify_model(args.model, args.model_manifest, args.revision)
        taxonomy, taxonomy_hash = evaluate.load_taxonomy(args.taxonomy)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        device = torch.device(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.pad_token_id
        if token_id is None:
            raise ValueError('tokenizer exposes no usable token ID')
        labels = sorted(set.union(*taxonomy.values()))
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(labels), problem_type='multi_label_classification',
            local_files_only=True, trust_remote_code=False, ignore_mismatched_sizes=False)
        model.to(device).train()
        inputs = torch.full((1, args.tokens), token_id, dtype=torch.long, device=device)
        outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs))
        loss = outputs.logits.float().square().mean()
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        finite = bool(torch.isfinite(outputs.logits).all().item() and torch.isfinite(loss).item())
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not finite or not gradients or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
            raise ValueError('model forward/backward did not produce finite values and gradients')
        if (training_inputs.verify_model(args.model, args.model_manifest, args.revision) != model_proof
                or evaluate.load_taxonomy(args.taxonomy)[1] != taxonomy_hash):
            raise ValueError('model or taxonomy changed during runtime probe')
        result = {'schema_version': 'lfg.model-runtime-probe.v1', 'model': model_proof,
                  'taxonomy_sha256': taxonomy_hash, 'device': str(device), 'torch': torch.__version__,
                  'cuda_runtime': torch.version.cuda, 'threads': args.threads, 'tokens': args.tokens,
                  'seed': args.seed, 'output_shape': list(outputs.logits.shape), 'finite': finite,
                  'parameters': sum(parameter.numel() for parameter in model.parameters()),
                  'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                  'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'quality_evidence': False, 'promotion_eligible': False}
        encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if args.output:
            with args.output.open('x', encoding='utf-8', newline='\n') as output:
                output.write(encoded)
        print(encoded, end='')
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        raise SystemExit(f'model runtime probe rejected: {error}') from error


if __name__ == '__main__':
    main()
