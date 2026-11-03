#!/usr/bin/env python3
"""Measure owned-checkpoint output format on private development requests."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load

import evaluate_owned_assessment_format as evaluation
import owned_sft_checkpoint
import owned_student


def evaluate(checkpoint: Path, dataset: Path, policy: Path, config_path: Path, output: Path,
             device: str, threads: int) -> dict:
    if device not in ('cpu', 'cuda') or type(threads) is not int or threads < 1:
        raise ValueError('invalid evaluation device or thread limit')
    config, config_raw = evaluation.config_file(config_path)
    requests, source = evaluation.select_validation(dataset, policy, config)
    if output.exists():
        raise ValueError('output already exists')
    owned_sft_checkpoint.private_directory(output)
    manifest_raw = (checkpoint / 'manifest.json').read_bytes()
    weights_raw = (checkpoint / 'weights.safetensors').read_bytes()
    manifest = json.loads(manifest_raw)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    if (sha(weights_raw) != manifest.get('weights_sha256')
            or sha(Path(owned_student.__file__).read_bytes()) != manifest.get('code_sha256', {}).get('owned_student.py')):
        raise ValueError('checkpoint identity mismatch')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    torch.set_num_threads(threads)
    model = owned_student.OwnedByteStudent(manifest['architecture']).to(device).eval()
    model.load_state_dict(load(weights_raw))
    outcomes, stops, elapsed = Counter(), Counter(), []
    with torch.inference_mode():
        for request in requests:
            prompt = evaluation.prompt(request, config).encode('utf-8')
            if len(prompt) > model.context:
                raise ValueError('prompt exceeds model context')
            context, generated = bytearray(prompt), bytearray()
            started = time.perf_counter()
            stop = config.get('stop_sequence')
            while len(generated) < config['max_new_bytes'] and len(context) <= model.context:
                tokens = torch.tensor([list(context)], device=device, dtype=torch.long)
                token = int(model(tokens)[0, -1].argmax())
                context.append(token)
                generated.append(token)
                if stop and generated.endswith(stop.encode('utf-8')):
                    break
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - started)
            stops['stop_sequence' if stop and generated.endswith(stop.encode()) else
                  'context_limit' if len(context) > model.context else 'byte_limit'] += 1
            outcomes[evaluation.response_format(bytes(generated), config['required_fields'])] += 1
    if ((checkpoint / 'manifest.json').read_bytes() != manifest_raw or
            (checkpoint / 'weights.safetensors').read_bytes() != weights_raw or
            evaluation.config_file(config_path)[1] != config_raw):
        raise ValueError('evaluation inputs changed')
    result = {'schema_version': 'lfg.owned-torch-format-evaluation.v1',
              'checkpoint_manifest_sha256': sha(manifest_raw), 'weights_sha256': sha(weights_raw),
              'config_sha256': sha(config_raw), 'validation_source': source, 'device': device,
              'format_counts': dict(outcomes), 'stop_counts': dict(stops), 'elapsed_seconds': elapsed,
              'assessment_accuracy_evaluated': False, 'test_opened': False, 'promotion_eligible': False}
    output.mkdir(mode=0o700)
    (output / 'report.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'dataset', 'policy', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--threads', type=int, required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.dataset, args.policy, args.config, args.output, args.device, args.threads)
    print(json.dumps({k: result[k] for k in ('format_counts', 'stop_counts', 'assessment_accuracy_evaluated', 'test_opened')}))


if __name__ == '__main__':
    main()
