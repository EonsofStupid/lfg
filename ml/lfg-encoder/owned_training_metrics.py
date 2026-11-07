"""Read bounded aggregate metrics from the latest completed private checkpoint."""
import json
import math
from pathlib import Path
import re


def metrics(root: Path, max_state_bytes: int, loss_window: int) -> dict:
    if type(max_state_bytes) is not int or max_state_bytes < 1 or type(loss_window) is not int or loss_window < 1:
        raise ValueError('invalid metric limits')
    candidates = []
    for state in root.glob('*/checkpoints/step-*/state.json'):
        match = re.fullmatch(r'step-([0-9]{16})', state.parent.name)
        if match and state.is_file() and not state.is_symlink():
            candidates.append((int(match.group(1)), state))
    if not candidates:
        return {'state': 'pending'}
    step, path = max(candidates, key=lambda item: item[0])
    if sum(candidate_step == step for candidate_step, _ in candidates) != 1:
        return {'state': 'unavailable'}
    try:
        with path.open('rb') as stream:
            raw = stream.read(max_state_bytes + 1)
        if len(raw) > max_state_bytes:
            raise ValueError('checkpoint metrics exceed size limit')
        value = json.loads(raw)
        progress = value['progress']
        losses, positions = progress['losses'], progress['supervised_tokens']
        if (value.get('schema_version') not in ('lfg.owned-distillation-checkpoint.v1',
                                                 'lfg.owned-assessment-sft-checkpoint.v1')
                or progress.get('step') != step or type(positions) is not int or positions < 0
                or not isinstance(losses, list) or len(losses) != step):
            raise ValueError('invalid checkpoint progress')
        def mean_tail(values):
            tail = values[-loss_window:]
            if not tail or any(type(v) not in (int, float) or not math.isfinite(v) for v in tail):
                raise ValueError('invalid loss values')
            return sum(tail) / len(tail)
        result = {'state': 'available', 'step': step, 'byte_positions': positions,
                  'mean_training_loss': mean_tail(losses), 'loss_steps': min(loss_window, len(losses))}
        cross_entropy = progress.get('ce_losses')
        if cross_entropy is not None:
            if not isinstance(cross_entropy, list) or len(cross_entropy) != step:
                raise ValueError('invalid cross-entropy progress')
            result['mean_cross_entropy'] = mean_tail(cross_entropy)
        return result
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return {'state': 'unavailable'}
