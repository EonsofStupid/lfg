#!/usr/bin/env python3
"""Continuously fine-tune owned checkpoints on newly delivered train datasets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import owned_training_chain as chain
import owned_training_service as supervisor


def encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()


def dataset_records(dataset: Path, max_manifest_bytes: int, max_dataset_bytes: int, *, check_inbox_identity: bool = False) -> dict[str, str]:
    if dataset.is_symlink() or not dataset.is_dir():
        raise ValueError('dataset must be a regular directory')
    def bounded(path, limit):
        if path.is_symlink():
            raise ValueError('dataset file must not be a symlink')
        with path.open('rb') as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError('dataset exceeds configured limits')
        return raw
    manifest_raw = bounded(dataset / 'manifest.json', max_manifest_bytes)
    manifest = json.loads(manifest_raw)
    raw = bounded(dataset / 'train.jsonl', max_dataset_bytes)
    sources = manifest.get('sources') if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 'lfg.assessment-distillation-dataset.v1'
            or manifest.get('test_assessed') is not False or manifest.get('files') != {'train.jsonl': hashlib.sha256(raw).hexdigest()}
            or not isinstance(sources, list) or not sources
            or any(not isinstance(item, dict) or item.get('split') != 'train'
                   or item.get('origin') not in ('owner_chat', 'owner_project') for item in sources)):
        raise ValueError('invalid train-only dataset')
    if check_inbox_identity and dataset.name != hashlib.sha256(manifest_raw + b'\0' + raw).hexdigest():
        raise ValueError('inbox dataset identity mismatch')
    result = {}
    for line in raw.splitlines():
        item = json.loads(line)
        if (not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'] or item['id'] in result):
            raise ValueError('invalid dataset record identity')
        result[item['id']] = hashlib.sha256(encoded(item)).hexdigest()
    if not result:
        raise ValueError('empty training dataset')
    return result


def choose_dataset(inbox: Path, previous: dict[str, str], max_manifest_bytes: int,
                   max_dataset_bytes: int, max_datasets: int, allow_revisions: bool):
    candidates = []
    entries = sorted(path for path in inbox.iterdir() if not path.name.startswith('.')) if inbox.exists() else []
    if len(entries) > max_datasets:
        raise ValueError('inbox exceeds configured dataset limit')
    for path in entries:
        records = dataset_records(path, max_manifest_bytes, max_dataset_bytes, check_inbox_identity=True)
        if not previous.keys() <= records.keys():
            continue
        changed = any(records[key] != old for key, old in previous.items())
        if changed and not allow_revisions:
            raise ValueError('dataset revises existing records without configured permission')
        if changed or len(records) > len(previous):
            candidates.append((path, records))
    if not candidates:
        return None
    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    winner = candidates[0]
    for _, other in candidates[1:]:
        if not other.keys() <= winner[1].keys() or any(winner[1][k] != v for k, v in other.items()):
            raise ValueError('inbox contains competing dataset updates')
    return winner


def fit_sft_config(base: dict, dataset: Path, checkpoint: Path, max_context: int, multiple: int) -> dict:
    if type(max_context) is not int or max_context < 1 or type(multiple) is not int or multiple < 1:
        raise ValueError('invalid context sizing limits')
    framing = base.get('framing')
    if not isinstance(framing, dict):
        raise ValueError('fine-tuning framing is missing')
    maximum = 0
    with (dataset / 'train.jsonl').open('rb') as stream:
        for line in stream:
            row = json.loads(line)
            messages = row.get('messages') if isinstance(row, dict) else None
            if (not isinstance(messages, list) or len(messages) != 3
                    or any(not isinstance(item, dict) or not isinstance(item.get('content'), str) for item in messages)):
                raise ValueError('invalid fine-tuning message framing')
            text = (framing['system_prefix'] + messages[0]['content'] + framing['system_suffix']
                    + framing['user_prefix'] + messages[1]['content'] + framing['user_suffix']
                    + framing['assistant_prefix'] + messages[2]['content'] + framing['assistant_suffix'])
            maximum = max(maximum, len(text.encode('utf-8')) - 1)
    source = json.loads((checkpoint / 'manifest.json').read_text(encoding='utf-8'))
    arch = source['architecture']
    needed = max(base['context'], arch['context'], ((maximum + multiple - 1) // multiple) * multiple)
    estimate = base['batch_size'] * arch['heads'] * needed ** 2 * (arch['layers'] // arch['attention_every'])
    if needed > max_context or estimate > base['max_attention_elements']:
        raise ValueError('dataset requires context beyond configured resource limits')
    return {**base, 'context': needed}


def load_config(path: Path, max_config_bytes: int = 1048576) -> tuple[dict, bytes]:
    if type(max_config_bytes) is not int or max_config_bytes < 1:
        raise ValueError('queue config byte limit must be positive')
    with path.open('rb') as stream:
        raw = stream.read(max_config_bytes + 1)
    if len(raw) > max_config_bytes:
        raise ValueError('queue configuration too large')
    value = json.loads(raw)
    paths = ('python', 'supervisor', 'trainer', 'sft_config', 'initial_state', 'baseline_dataset', 'inbox', 'queue_state')
    integers = ('segment_steps', 'poll_seconds', 'max_manifest_bytes', 'max_dataset_bytes',
                'max_checkpoint_bytes', 'max_datasets')
    if (not isinstance(value, dict) or value.get('schema_version') != 'lfg.owned-dataset-queue.v1'
            or any(not isinstance(value.get(k), str) or not Path(value[k]).is_absolute() for k in paths)
            or any(type(value.get(k)) is not int or value[k] < 1 for k in integers)
            or value['poll_seconds'] > 3600 or type(value.get('allow_record_revisions')) is not bool
            or type(value.get('progress_poll_seconds', value['poll_seconds'])) is not int
            or not 1 <= value.get('progress_poll_seconds', value['poll_seconds']) <= 3600
            or type(value.get('fit_context_to_data', False)) is not bool
            or (value.get('fit_context_to_data', False) and
                (type(value.get('max_context')) is not int or value['max_context'] < 1
                 or type(value.get('context_multiple')) is not int or value['context_multiple'] < 1))):
        raise ValueError('invalid dataset queue configuration')
    return value, raw


def run(config_path: Path, *, max_config_bytes: int = 1048576) -> None:
    import fcntl
    value, raw_config = load_config(config_path, max_config_bytes)
    root = Path(value['queue_state']); root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    lock = open(root / 'queue.lock', 'a+b')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('another dataset queue is active') from None
    identity = {'config_sha256': hashlib.sha256(raw_config).hexdigest(),
                'trainer_sha256': supervisor.digest(Path(value['trainer'])),
                'supervisor_sha256': supervisor.digest(Path(value['supervisor'])),
                'queue_sha256': supervisor.digest(Path(__file__)), 'chain_sha256': supervisor.digest(Path(chain.__file__)),
                'sft_config_sha256': supervisor.digest(Path(value['sft_config']))}
    identity_path = root / 'queue-identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('queue configuration or code changed; use a new state directory')
    if not identity_path.exists():
        supervisor.atomic_json(identity_path, identity)
    state_path, status_path = root / 'state.json', root / 'status.json'
    def report(status, **fields):
        supervisor.atomic_json(status_path, {'schema_version': 'lfg.owned-dataset-queue-status.v1',
                                             'state': status, 'updated_unix': time.time(), **fields})
    while True:
        if config_path.read_bytes() != raw_config:
            raise ValueError('queue configuration changed')
        if supervisor.digest(Path(value['sft_config'])) != identity['sft_config_sha256']:
            raise ValueError('queue fine-tuning configuration changed')
        if (supervisor.digest(Path(__file__)) != identity['queue_sha256']
                or supervisor.digest(Path(chain.__file__)) != identity['chain_sha256']):
            raise ValueError('queue implementation changed')
        if not state_path.exists():
            checkpoint = chain.completed_checkpoint(Path(value['initial_state']), value['max_checkpoint_bytes'],
                                                    'lfg.owned-assessment-sft-result.v1')
            if checkpoint is None:
                report('waiting_for_initial_model')
                time.sleep(value['poll_seconds'])
                continue
            baseline = dataset_records(Path(value['baseline_dataset']), value['max_manifest_bytes'], value['max_dataset_bytes'])
            source_manifest = json.loads((checkpoint / 'manifest.json').read_text(encoding='utf-8'))
            if source_manifest.get('training_file_sha256') != supervisor.digest(Path(value['baseline_dataset']) / 'train.jsonl'):
                raise ValueError('initial model was not trained on the configured baseline dataset')
            state = {'schema_version': 'lfg.owned-dataset-queue-state.v1', 'checkpoint_state': value['initial_state'],
                     'records': baseline, 'completed_updates': 0}
            supervisor.atomic_json(state_path, state)
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if (state.get('schema_version') != 'lfg.owned-dataset-queue-state.v1' or not isinstance(state.get('records'), dict)
                or type(state.get('completed_updates')) is not int):
            raise ValueError('invalid queue state')
        candidate = choose_dataset(Path(value['inbox']), state['records'], value['max_manifest_bytes'],
                                   value['max_dataset_bytes'], value['max_datasets'], value['allow_record_revisions'])
        if candidate is None:
            report('idle', records=len(state['records']), completed_updates=state['completed_updates'])
            time.sleep(value['poll_seconds'])
            continue
        checkpoint_state = Path(state['checkpoint_state'])
        if (not checkpoint_state.is_absolute() or checkpoint_state.is_symlink()
                or (checkpoint_state.resolve() != Path(value['initial_state']).resolve()
                    and not checkpoint_state.resolve().is_relative_to((root / 'jobs').resolve()))):
            raise ValueError('queue checkpoint state is outside configured jobs')
        checkpoint = chain.completed_checkpoint(checkpoint_state, value['max_checkpoint_bytes'],
                                                'lfg.owned-assessment-sft-result.v1')
        if checkpoint is None:
            raise ValueError('current owned checkpoint is incomplete')
        dataset, records = candidate
        key = hashlib.sha256(encoded({'checkpoint_weights': supervisor.digest(checkpoint / 'weights.safetensors'),
                                      'records': records})).hexdigest()
        job = root / 'jobs' / key; job.mkdir(mode=0o700, parents=True, exist_ok=True)
        sft_config = json.loads(Path(value['sft_config']).read_text(encoding='utf-8'))
        if value.get('fit_context_to_data', False):
            sft_config = fit_sft_config(sft_config, dataset, checkpoint, value['max_context'], value['context_multiple'])
        sft_path = job / 'sft.json'
        if sft_path.exists() and json.loads(sft_path.read_text(encoding='utf-8')) != sft_config:
            raise ValueError('queued fine-tuning configuration changed')
        if not sft_path.exists():
            supervisor.atomic_json(sft_path, sft_config)
        service_config = {'schema_version': 'lfg.owned-training-service.v1', 'trainer_kind': 'assessment_sft',
                          'python': value['python'], 'trainer': value['trainer'], 'state_dir': str(job / 'state'),
                          'dataset': str(dataset.resolve()), 'checkpoint': str(checkpoint), 'sft_config': str(sft_path),
                          'segment_steps': value['segment_steps'], 'status_poll_seconds': value['poll_seconds']}
        service_path = job / 'service.json'
        if service_path.exists() and json.loads(service_path.read_text()) != service_config:
            raise ValueError('queued job identity changed')
        if not service_path.exists():
            supervisor.atomic_json(service_path, service_config)
        report('training', records=len(records), completed_updates=state['completed_updates'])
        child = subprocess.Popen([value['python'], value['supervisor'], '--config', str(service_path), '--exit-on-complete'])
        while child.poll() is None:
            progress = job / 'state' / 'status.json'
            if progress.exists():
                current = json.loads(progress.read_text(encoding='utf-8'))
                if (current.get('schema_version') == 'lfg.owned-training-status.v1'
                        and type(current.get('step')) is int and type(current.get('total_steps')) is int
                        and 0 <= current['step'] <= current['total_steps']):
                    report('training', records=len(records), completed_updates=state['completed_updates'],
                           step=current['step'], total_steps=current['total_steps'])
            time.sleep(value.get('progress_poll_seconds', value['poll_seconds']))
        if child.returncode:
            report('failed', records=len(records), completed_updates=state['completed_updates'], exit_code=child.returncode)
            raise ValueError('queued training failed')
        completed = chain.completed_checkpoint(job / 'state', value['max_checkpoint_bytes'],
                                              'lfg.owned-assessment-sft-result.v1')
        if (completed is None or dataset_records(dataset, value['max_manifest_bytes'], value['max_dataset_bytes'],
                                                 check_inbox_identity=True) != records):
            raise ValueError('queued training result or dataset changed')
        completed_manifest = json.loads((completed / 'manifest.json').read_text(encoding='utf-8'))
        if completed_manifest.get('training_file_sha256') != supervisor.digest(dataset / 'train.jsonl'):
            raise ValueError('completed model does not identify the queued dataset')
        supervisor.atomic_json(state_path, {'schema_version': 'lfg.owned-dataset-queue-state.v1',
            'checkpoint_state': str(job / 'state'), 'records': records, 'completed_updates': state['completed_updates'] + 1})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--max-config-bytes', type=int, default=1048576)
    args = parser.parse_args()
    try:
        run(args.config, max_config_bytes=args.max_config_bytes)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f'owned dataset queue failed ({type(error).__name__}): {error}') from None


if __name__ == '__main__':
    main()
