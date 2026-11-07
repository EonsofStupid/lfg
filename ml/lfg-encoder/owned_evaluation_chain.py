#!/usr/bin/env python3
"""Run one private development format evaluation after assessment training."""
import argparse
import json
import os
from pathlib import Path
import time

import owned_training_chain as training_chain
import owned_training_service as supervisor


def load_config(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if len(raw) > 1048576:
        raise ValueError('evaluation chain configuration is too large')
    value = json.loads(raw)
    paths = ('python', 'evaluator', 'source_state', 'dataset', 'policy', 'evaluation_config', 'output')
    if (not isinstance(value, dict) or value.get('schema_version') != 'lfg.owned-evaluation-chain.v1'
            or any(not isinstance(value.get(k), str) or not Path(value[k]).is_absolute() for k in paths)
            or value.get('device') not in ('cpu', 'cuda')
            or type(value.get('threads')) is not int or value['threads'] < 1
            or type(value.get('poll_seconds')) is not int or not 1 <= value['poll_seconds'] <= 3600
            or type(value.get('max_checkpoint_bytes')) is not int or value['max_checkpoint_bytes'] < 1):
        raise ValueError('invalid evaluation chain configuration')
    return value, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    value, raw = load_config(args.config)
    while True:
        checkpoint = training_chain.completed_checkpoint(Path(value['source_state']), value['max_checkpoint_bytes'],
                                                         'lfg.owned-assessment-sft-result.v1')
        if args.config.read_bytes() != raw:
            raise ValueError('evaluation chain configuration changed')
        if checkpoint is not None:
            output = Path(value['output'])
            if output.exists():
                report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
                if (report.get('schema_version') != 'lfg.owned-torch-format-evaluation.v1'
                        or report.get('weights_sha256') != supervisor.digest(checkpoint / 'weights.safetensors')
                        or report.get('config_sha256') != supervisor.digest(Path(value['evaluation_config']))
                        or report.get('validation_source', {}).get('source_sha256') != supervisor.digest(
                            Path(value['dataset']) / 'validation.jsonl')
                        or report.get('validation_source', {}).get('policy_sha256') != supervisor.digest(Path(value['policy']))):
                    raise ValueError('existing evaluation identity changed')
                return
            argv = [value['python'], value['evaluator'], '--checkpoint', str(checkpoint), '--dataset', value['dataset'],
                    '--policy', value['policy'], '--config', value['evaluation_config'], '--output', value['output'],
                    '--device', value['device'], '--threads', str(value['threads'])]
            os.execv(value['python'], argv)
        time.sleep(value['poll_seconds'])


if __name__ == '__main__':
    main()
