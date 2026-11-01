#!/usr/bin/env python3
"""Check a local model's execution path without training data or quality claims."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import evaluate
import training_inputs


def cgroup_limits() -> dict | None:
    """Report the current cgroup v2 limits where available; not ancestor limits."""
    try:
        entries = Path('/proc/self/cgroup').read_text().splitlines()
        relative = next(line[3:] for line in entries if line.startswith('0::'))
        root = Path('/sys/fs/cgroup') / relative.lstrip('/')
        return {name: (root / name).read_text().strip() for name in ('memory.max', 'cpu.max', 'pids.max')}
    except (OSError, StopIteration):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-manifest', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--taxonomy', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--tokens', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--cuda-memory-fraction', type=float,
                        help='optional PyTorch CUDA allocator fraction; not isolation from other GPU processes')
    parser.add_argument('--optimizer-learning-rate', type=float,
                        help='run one disposable AdamW step with generated labels at this learning rate')
    parser.add_argument('--optimizer-weight-decay', type=float,
                        help='explicit AdamW weight decay when requesting an optimizer step')
    parser.add_argument('--output', type=Path, help='new file for the aggregate probe report')
    args = parser.parse_args()
    try:
        if args.threads < 1 or args.tokens < 1 or args.batch_size < 1 or args.seed < 0:
            raise ValueError('threads and tokens must be positive and seed must be nonnegative')
        if args.cuda_memory_fraction is not None and (not re.fullmatch(r'cuda(?::[0-9]+)?', args.device) or
                not math.isfinite(args.cuda_memory_fraction) or not 0 < args.cuda_memory_fraction <= 1):
            raise ValueError('CUDA memory fraction requires cuda and must be in (0, 1]')
        if ((args.optimizer_learning_rate is None) != (args.optimizer_weight_decay is None) or
                (args.optimizer_learning_rate is not None and
                 (not math.isfinite(args.optimizer_learning_rate) or args.optimizer_learning_rate <= 0 or
                  not math.isfinite(args.optimizer_weight_decay) or args.optimizer_weight_decay < 0))):
            raise ValueError('optimizer step requires a positive learning rate and nonnegative weight decay')
        if args.output and (args.output.resolve().is_relative_to(args.model.resolve()) or args.output.exists()):
            raise ValueError('probe output must be a new file outside the model snapshot')
        model_proof = training_inputs.verify_model(args.model, args.model_manifest, args.revision)
        taxonomy, taxonomy_hash = evaluate.load_taxonomy(args.taxonomy)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.init()
            if device.index is None:
                device = torch.device('cuda', torch.cuda.current_device())
            if args.cuda_memory_fraction is not None:
                torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
            torch.cuda.reset_peak_memory_stats(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.pad_token_id
        if token_id is None:
            raise ValueError('tokenizer exposes no usable token ID')
        labels = sorted(set.union(*taxonomy.values()))
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(labels), problem_type='multi_label_classification',
            local_files_only=True, trust_remote_code=False, ignore_mismatched_sizes=False)
        model.to(device).train()
        inputs = torch.full((args.batch_size, args.tokens), token_id, dtype=torch.long, device=device)
        optimizer = None if args.optimizer_learning_rate is None else torch.optim.AdamW(
            model.parameters(), lr=args.optimizer_learning_rate, weight_decay=args.optimizer_weight_decay)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        if optimizer is None:
            outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs))
            loss = outputs.logits.float().square().mean()
        else:
            targets = torch.zeros((args.batch_size, len(labels)), dtype=torch.float32, device=device)
            outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=targets)
            loss = outputs.loss
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        finite = bool(torch.isfinite(outputs.logits).all().item() and torch.isfinite(loss).item())
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not finite or not gradients or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
            raise ValueError('model forward/backward did not produce finite values and gradients')
        if optimizer is not None:
            optimizer.step()
            if not optimizer.state:
                raise ValueError('optimizer did not initialize state')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started
        if (training_inputs.verify_model(args.model, args.model_manifest, args.revision) != model_proof
                or evaluate.load_taxonomy(args.taxonomy)[1] != taxonomy_hash):
            raise ValueError('model or taxonomy changed during runtime probe')
        result = {'schema_version': 'lfg.model-runtime-probe.v1', 'model': model_proof,
                  'taxonomy_sha256': taxonomy_hash, 'device': str(device), 'torch': torch.__version__,
                  'cuda_runtime': torch.version.cuda, 'threads': args.threads, 'tokens': args.tokens,
                  'batch_size': args.batch_size, 'cuda_memory_fraction': args.cuda_memory_fraction,
                  'optimizer_step': optimizer is not None, 'optimizer_learning_rate': args.optimizer_learning_rate,
                  'optimizer_weight_decay': args.optimizer_weight_decay, 'step_elapsed_seconds': elapsed_seconds,
                  'seed': args.seed, 'output_shape': list(outputs.logits.shape), 'finite': finite,
                  'parameters': sum(parameter.numel() for parameter in model.parameters()),
                  'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                  'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
                  'cuda_device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  'cgroup_limits': cgroup_limits(),
                  'generated_inputs_only': True, 'weights_saved': False,
                  'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'model_verifier_sha256': hashlib.sha256(Path(training_inputs.__file__).read_bytes()).hexdigest(),
                  'taxonomy_loader_sha256': hashlib.sha256(Path(evaluate.__file__).read_bytes()).hexdigest(),
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
