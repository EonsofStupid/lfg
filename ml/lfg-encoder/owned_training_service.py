#!/usr/bin/env python3
"""Supervise one private, resumable owned-model distillation job on Linux."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid


SFT_CHECKPOINT_SCHEMAS = {
    'lfg.owned-assessment-sft-config.v1': 'lfg.owned-assessment-sft-checkpoint.v1',
    'lfg.owned-user-request-sft-config.v1': 'lfg.owned-user-request-sft-checkpoint.v1',
    'lfg.owned-multilabel-sft-config.v1': 'lfg.owned-multilabel-sft-checkpoint.v1',
}
SFT_RESULT_SCHEMAS = {
    'lfg.owned-assessment-sft-config.v1': 'lfg.owned-assessment-sft-result.v1',
    'lfg.owned-user-request-sft-config.v1': 'lfg.owned-user-request-sft-result.v1',
    'lfg.owned-multilabel-sft-config.v1': 'lfg.owned-multilabel-sft-result.v1',
}
SFT_MUTABLE_CONFIG_FIELDS = {'steps', 'checkpoint_every', 'max_parameters', 'max_attention_elements',
    'max_dataset_bytes', 'max_manifest_bytes', 'max_checkpoint_bytes', 'max_optimizer_bytes',
    'max_examples', 'max_record_bytes'}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def bounded_bytes(path: Path, limit: int) -> bytes:
    if type(limit) is not int or limit < 1:
        raise ValueError('invalid configured input byte limit')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('configured input exceeds byte limit')
    return raw


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                        allow_nan=False) + '\n').encode('utf-8')


def sft_source_linkage(manifest: dict, schema_config: str, sft: dict) -> tuple[str, str]:
    if manifest.get('schema_version') not in SFT_RESULT_SCHEMAS.values():
        return 'not_applicable', 'not_applicable'
    coverage = manifest.get('coverage')
    if not isinstance(coverage, dict):
        raise ValueError('configured SFT source lacks coverage provenance')
    unlinked_allowed = (schema_config == 'lfg.owned-user-request-sft-config.v1'
        and sft.get('allow_unlinked_legacy_sft_checkpoint', False) is True
        and manifest.get('schema_version') == 'lfg.owned-assessment-sft-result.v1')
    links = []
    for name in ('system_sha256', 'framing_sha256'):
        top_present, coverage_present = name in manifest, name in coverage
        if not top_present:
            if not unlinked_allowed or coverage_present:
                raise ValueError('configured SFT source has unsupported unlinked provenance')
            links.append('unknown_unlinked')
            continue
        value = manifest[name]
        if (not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
                or not coverage_present or coverage[name] != value):
            raise ValueError('configured SFT source linkage is invalid')
        links.append('verified_linked')
    if len(set(links)) != 1:
        raise ValueError('configured SFT source linkage is inconsistent')
    return links[0], links[1]


MULTILABEL_IDENTITY_VALIDATOR = r'''
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

try:
    request = json.load(sys.stdin)
    import torch
    import safetensors
    trainer_path = Path(request['trainer'])
    trainer_dir = trainer_path.parent
    sys.path.insert(0, str(trainer_dir))
    spec = importlib.util.spec_from_file_location('_lfg_identity_trainer', trainer_path)
    if spec is None or spec.loader is None:
        raise ValueError('trainer module unavailable')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    config_path, dataset, source = Path(request['config']), Path(request['dataset']), Path(request['source'])
    config, config_raw = trainer.configuration(config_path, request['max_config_bytes'])
    if sha(config_raw) != request['config_sha256']:
        raise ValueError('config changed')
    _, manifest_raw, training_raw, coverage = trainer.read_training(dataset, config)
    source_manifest_path = source / 'manifest.json'
    source_weights_path = source / 'weights.safetensors'
    source_manifest_raw = trainer.read_bounded(source_manifest_path, config['max_manifest_bytes'])
    source_manifest_sha256 = sha(source_manifest_raw)
    source_weights_sha256 = sha_file(source_weights_path)
    code = {name: sha_file(trainer_dir / name) for name in (
        'train_owned_assessment_sft.py', 'train_owned_student.py', 'owned_student.py',
        'owned_sft_checkpoint.py', 'owned_multilabel_contract.py', 'masked_supervision.py')}
    coverage.update({'config_sha256': sha(config_raw),
        'trainer_code_sha256': code['train_owned_assessment_sft.py'], 'code_sha256': code,
        'source_checkpoint_manifest_sha256': source_manifest_sha256,
        'source_checkpoint_weights_sha256': source_weights_sha256,
        'live_masked_supervision_code_sha256': code['masked_supervision.py']})
    if (trainer.read_bounded(config_path, request['max_config_bytes']) != config_raw
            or trainer.read_bounded(dataset / 'manifest.json', config['max_manifest_bytes']) != manifest_raw
            or trainer.read_bounded(dataset / 'train.jsonl', config['max_dataset_bytes']) != training_raw
            or trainer.read_bounded(source_manifest_path, config['max_manifest_bytes']) != source_manifest_raw
            or sha_file(source_weights_path) != source_weights_sha256):
        raise ValueError('input changed')
    result = {'coverage_sha256': sha(trainer.canonical(coverage)),
        'dataset_manifest_sha256': coverage['dataset_manifest_sha256'],
        'training_file_sha256': coverage['training_file_sha256'],
        'source_checkpoint_manifest_sha256': source_manifest_sha256,
        'source_checkpoint_weights_sha256': source_weights_sha256}
    print(json.dumps(result, separators=(',', ':')))
except BaseException:
    raise SystemExit(2)
'''


def trainer_multilabel_coverage(python_path: Path, trainer_path: Path, dataset: Path, config_path: Path,
                                config_raw: bytes, max_config_bytes: int, source_path: Path,
                                expected: dict) -> str:
    """Compute the exact coverage identity in the configured trainer environment."""
    request = {'trainer': str(trainer_path), 'dataset': str(dataset), 'config': str(config_path),
               'source': str(source_path), 'max_config_bytes': max_config_bytes,
               'config_sha256': hashlib.sha256(config_raw).hexdigest()}
    try:
        child = subprocess.run([str(python_path), '-c', MULTILABEL_IDENTITY_VALIDATOR],
                               input=json.dumps(request), capture_output=True, text=True, check=False)
        if child.returncode != 0:
            raise ValueError('trainer identity process failed')
        result = json.loads(child.stdout)
        if (not isinstance(result, dict)
                or any(result.get(name) != expected[name] for name in (
                    'dataset_manifest_sha256', 'training_file_sha256',
                    'source_checkpoint_manifest_sha256', 'source_checkpoint_weights_sha256'))
                or not isinstance(result.get('coverage_sha256'), str)
                or not re.fullmatch(r'[0-9a-f]{64}', result['coverage_sha256'])):
            raise ValueError('trainer identity result mismatch')
        return result['coverage_sha256']
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise ValueError('cannot validate multilabel checkpoint identity in configured trainer environment') from None


def sft_context(value: dict) -> dict:
    """Resolve the trainer's explicit schema and stable continuation identity before launch."""
    config_path = Path(value['sft_config'])
    config_raw = bounded_bytes(config_path, value.get('max_sft_config_bytes', 65536))
    sft = json.loads(config_raw)
    if not isinstance(sft, dict) or sft.get('schema_version') not in SFT_CHECKPOINT_SCHEMAS:
        raise ValueError('unsupported SFT configuration schema')
    schema_config = sft['schema_version']
    steps = sft.get('steps')
    if type(steps) is not int or steps < 1:
        raise ValueError('SFT steps must be positive')
    dataset = Path(value['dataset'])
    if schema_config == 'lfg.owned-user-request-sft-config.v1':
        dataset_raw = bounded_bytes(dataset, sft['max_dataset_bytes'])
        envelope = json.loads(dataset_raw)
        if not isinstance(envelope, dict) or not isinstance(envelope.get('train_records'), list):
            raise ValueError('invalid user-request SFT dataset envelope')
        training_raw = b''.join(canonical_json(row) for row in envelope['train_records'])
    else:
        dataset_raw = bounded_bytes(dataset / 'manifest.json', sft['max_manifest_bytes'])
        training_raw = bounded_bytes(dataset / 'train.jsonl', sft['max_dataset_bytes'])
        if (schema_config == 'lfg.owned-multilabel-sft-config.v1'
                and sft.get('expected_manifest_sha256') != hashlib.sha256(dataset_raw).hexdigest()):
            raise ValueError('multilabel dataset manifest differs from configured identity')
    source = Path(value['checkpoint'])
    source_manifest_raw = bounded_bytes(source / 'manifest.json', sft['max_manifest_bytes'])
    source_manifest = json.loads(source_manifest_raw)
    source_weights_sha256 = digest(source / 'weights.safetensors')
    trainer_dir = Path(value['trainer']).parent
    if (not isinstance(source_manifest, dict)
            or source_manifest.get('schema_version') not in {
                'lfg.owned-student-experiment-result.v1', 'lfg.owned-distillation-result.v1',
                *SFT_RESULT_SCHEMAS.values()}
            or source_manifest.get('weights_sha256') != source_weights_sha256
            or not isinstance(source_manifest.get('architecture'), dict)
            or not isinstance(source_manifest.get('code_sha256'), dict)
            or source_manifest['code_sha256'].get('owned_student.py') != digest(trainer_dir / 'owned_student.py')):
        raise ValueError('configured source checkpoint identity is invalid')
    system_linkage, framing_linkage = sft_source_linkage(source_manifest, schema_config, sft)
    expected = {
        'dataset_manifest_sha256': hashlib.sha256(dataset_raw).hexdigest(),
        'training_file_sha256': hashlib.sha256(training_raw).hexdigest(),
        'source_checkpoint_manifest_sha256': hashlib.sha256(source_manifest_raw).hexdigest(),
        'source_checkpoint_weights_sha256': source_weights_sha256,
        'source_checkpoint_schema': source_manifest.get('schema_version'),
        'source_checkpoint_system_linkage': system_linkage,
        'source_checkpoint_framing_linkage': framing_linkage,
        'continuation_config_sha256': hashlib.sha256(canonical_json(
            {key: item for key, item in sft.items() if key not in SFT_MUTABLE_CONFIG_FIELDS})).hexdigest(),
    }
    if schema_config == 'lfg.owned-multilabel-sft-config.v1':
        expected['training_config_sha256'] = hashlib.sha256(config_raw).hexdigest()
    if ('allow_unlinked_legacy_sft_checkpoint' in sft
            and (type(sft['allow_unlinked_legacy_sft_checkpoint']) is not bool
                 or (sft['allow_unlinked_legacy_sft_checkpoint']
                     and schema_config != 'lfg.owned-user-request-sft-config.v1'))):
        raise ValueError('invalid unlinked SFT source policy')
    architecture = dict(source_manifest['architecture'])
    if type(sft.get('context')) is not int or sft['context'] < 1:
        raise ValueError('SFT context must be positive')
    architecture['context'] = sft['context']
    expected['architecture'] = architecture
    code_names = ['train_owned_assessment_sft.py', 'train_owned_student.py',
                  'owned_student.py', 'owned_sft_checkpoint.py']
    if schema_config == 'lfg.owned-multilabel-sft-config.v1':
        code_names.extend(['owned_multilabel_contract.py', 'masked_supervision.py'])
        expected['multilabel_coverage_sha256'] = trainer_multilabel_coverage(
            Path(value['python']), Path(value['trainer']), dataset, config_path, config_raw,
            value.get('max_sft_config_bytes', 65536), source, expected)
    expected['code_sha256'] = {name: digest(trainer_dir / name) for name in code_names}
    return {'schema_config': schema_config, 'checkpoint_schema': SFT_CHECKPOINT_SCHEMAS[schema_config],
            'result_schema': SFT_RESULT_SCHEMAS[schema_config], 'steps': steps,
            'config_sha256': hashlib.sha256(config_raw).hexdigest(),
            'identity': expected}


def validate_sft_state_identity(state: dict, expected: dict) -> None:
    identity = state.get('identity')
    required = {'dataset_manifest_sha256', 'training_file_sha256', 'source_checkpoint_manifest_sha256',
                'source_checkpoint_weights_sha256', 'source_checkpoint_schema',
                'source_checkpoint_system_linkage', 'source_checkpoint_framing_linkage', 'code_sha256',
                'continuation_config_sha256', 'architecture', 'python', 'torch', 'safetensors',
                'cuda_version', 'cublas_workspace_config', 'cuda_devices'}
    if expected['schema_config'] == 'lfg.owned-multilabel-sft-config.v1':
        required.add('training_config_sha256')
    if expected['schema_config'] == 'lfg.owned-multilabel-sft-config.v1':
        required.add('multilabel_coverage_sha256')
    if not isinstance(identity, dict) or not required <= set(identity):
        raise ValueError('checkpoint identity is incomplete for configured SFT schema')
    for name, wanted in expected['identity'].items():
        if wanted is not None and identity.get(name) != wanted:
            raise ValueError(f'checkpoint identity does not match configured SFT {name}')
    for name in ('dataset_manifest_sha256', 'training_file_sha256', 'source_checkpoint_manifest_sha256',
                 'source_checkpoint_weights_sha256', 'continuation_config_sha256'):
        if not isinstance(identity.get(name), str) or not re.fullmatch(r'[0-9a-f]{64}', identity[name]):
            raise ValueError('checkpoint identity digest is malformed')
    if expected['schema_config'] == 'lfg.owned-multilabel-sft-config.v1':
        value = identity.get('training_config_sha256')
        if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('checkpoint training configuration identity is malformed')
    if expected['schema_config'] == 'lfg.owned-multilabel-sft-config.v1':
        value = identity.get('multilabel_coverage_sha256')
        if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('checkpoint multilabel coverage identity is malformed')


def validate_sft_result(sft: dict, manifest: object, weights_path: Path) -> None:
    identity = sft['identity']
    expected = {'schema_version': sft['result_schema'], 'steps': sft['steps'],
        'config_sha256': sft['config_sha256'],
        'dataset_manifest_sha256': identity['dataset_manifest_sha256'],
        'training_file_sha256': identity['training_file_sha256'],
        'source_checkpoint_manifest_sha256': identity['source_checkpoint_manifest_sha256'],
        'source_checkpoint_weights_sha256': identity['source_checkpoint_weights_sha256'],
        'source_checkpoint_schema': identity['source_checkpoint_schema'],
        'source_checkpoint_system_linkage': identity['source_checkpoint_system_linkage'],
        'source_checkpoint_framing_linkage': identity['source_checkpoint_framing_linkage'],
        'code_sha256': identity['code_sha256']}
    if (not isinstance(manifest, dict) or manifest.get('completed') is not True
            or any(manifest.get(key) != expected_value for key, expected_value in expected.items())
            or digest(weights_path) != manifest.get('weights_sha256')):
        raise ValueError('invalid SFT result identity')


def atomic_json(path: Path, value: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix='.status-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('schema_version') != 'lfg.owned-training-service.v1':
        raise ValueError('unsupported service configuration')
    kind = value.get('trainer_kind', 'distillation')
    if kind not in ('distillation', 'assessment_sft'):
        raise ValueError('unsupported trainer kind')
    paths = ('student_config', 'distillation_config', 'runtime_config', 'corpus_plan', 'screen_policy',
             'teacher_manifest', 'teacher_weights') if kind == 'distillation' else ('dataset', 'checkpoint', 'sft_config')
    for name in ('python', 'trainer', 'state_dir', *paths):
        if not isinstance(value.get(name), str) or not Path(value[name]).is_absolute():
            raise ValueError(f'{name} must be an absolute path')
    if kind == 'distillation' and value.get('teacher_format') not in ('gpt2-byte', 'owned-byte'):
        raise ValueError('unsupported teacher format')
    if kind == 'distillation' and value['teacher_format'] == 'gpt2-byte' and not value.get('teacher_revision'):
        raise ValueError('a pinned teacher revision is required')
    if ('initial_checkpoint' in value and (kind != 'distillation' or not isinstance(value['initial_checkpoint'], str)
                                            or not Path(value['initial_checkpoint']).is_absolute())):
        raise ValueError('initial checkpoint must be an absolute distillation checkpoint path')
    if type(value.get('segment_steps')) is not int or value['segment_steps'] < 1:
        raise ValueError('segment_steps must be positive')
    if type(value.get('status_poll_seconds')) not in (int, float) or not 1 <= value['status_poll_seconds'] <= 3600:
        raise ValueError('invalid status poll interval')
    if type(value.get('max_checkpoint_state_bytes', 67108864)) is not int or value.get('max_checkpoint_state_bytes', 67108864) < 1:
        raise ValueError('invalid checkpoint state byte limit')
    if 'max_sft_config_bytes' in value and (type(value['max_sft_config_bytes']) is not int
                                             or value['max_sft_config_bytes'] < 1):
        raise ValueError('invalid SFT configuration byte limit')
    return value


def latest_checkpoint(segments: Path, schema: str = 'lfg.owned-distillation-checkpoint.v1',
                      max_state_bytes: int = 67108864, sft: dict | None = None) -> tuple[int, Path, str] | None:
    if type(max_state_bytes) is not int or max_state_bytes < 1:
        raise ValueError('invalid checkpoint state byte limit')
    candidates = []
    for state_path in segments.glob('*/checkpoints/step-*/state.json'):
        name = state_path.parent.name
        digits = name.removeprefix('step-')
        if len(digits) == 16 and digits.isascii() and digits.isdecimal() and int(digits) > 0:
            candidates.append((int(digits), state_path))
    if not candidates:
        return None
    latest_step = max(step for step, _ in candidates)
    newest = []
    for step, state_path in candidates:
        if step != latest_step:
            continue
        try:
            with state_path.open('rb') as stream:
                raw = stream.read(max_state_bytes + 1)
            if len(raw) > max_state_bytes:
                raise ValueError('checkpoint state too large')
            state = json.loads(raw)
            if (not isinstance(state, dict) or state.get('schema_version') != schema
                    or type(state['progress']['step']) is not int
                    or state['progress']['step'] != step):
                raise ValueError('checkpoint failed validation')
            if sft is not None:
                validate_sft_state_identity(state, sft)
            newest.append((step, state_path.parent, hashlib.sha256(raw).hexdigest(), state))
        except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'cannot validate published checkpoint: {state_path}') from exc
    for _, path, _, state in newest:
        try:
            if (digest(path / 'weights.safetensors') != state['model_sha256']
                    or digest(path / 'optimizer.safetensors') != state['optimizer_sha256']):
                raise ValueError('checkpoint failed validation')
        except (KeyError, OSError) as exc:
            raise ValueError('cannot validate latest checkpoint tensors') from exc
    if any(item[2] != newest[0][2] for item in newest[1:]):
        raise ValueError('conflicting checkpoints at the latest step')
    return newest[-1][:3]


def run(path: Path, *, exit_on_complete: bool = False) -> None:
    import fcntl  # Linux service lock; keep read-only helpers usable on other platforms.
    value = config(path)
    root = Path(value['state_dir'])
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    segments = root / 'segments'
    segments.mkdir(mode=0o700, exist_ok=True)
    lock = open(root / 'service.lock', 'a+b')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('another supervisor owns this training directory') from None
    identity_file = root / 'service-identity.json'
    # Existing runs cannot silently switch configurations after a restart.
    identity = {'config_sha256': digest(path), 'trainer_sha256': digest(Path(value['trainer']))}
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError('service configuration or trainer changed; create a new state directory')
    if not identity_file.exists():
        atomic_json(identity_file, identity)
    status = root / 'status.json'
    def report(state: str, **extra) -> None:
        atomic_json(status, {'schema_version': 'lfg.owned-training-status.v1', 'state': state,
                             'updated_unix': time.time(), **extra})
    complete = root / 'complete.json'
    kind = value.get('trainer_kind', 'distillation')
    while not complete.exists():
        sft = sft_context(value) if kind == 'assessment_sft' else None
        schema = 'lfg.owned-distillation-checkpoint.v1' if sft is None else sft['checkpoint_schema']
        checkpoint = latest_checkpoint(segments, schema, value.get('max_checkpoint_state_bytes', 67108864), sft)
        student = (json.loads(Path(value['student_config']).read_text(encoding='utf-8'))
                   if sft is None else {'steps': sft['steps']})
        total = student['steps']
        if checkpoint and checkpoint[0] > total:
            report('needs_attention', step=checkpoint[0], reason='checkpoint exceeds configured final step')
            raise ValueError('checkpoint exceeds configured final step')
        output = segments / ('segment-' + uuid.uuid4().hex)
        command = [value['python'], value['trainer']]
        options = (('student_config', 'distillation_config', 'runtime_config', 'corpus_plan', 'screen_policy',
                    'teacher_manifest', 'teacher_weights', 'teacher_format') if kind == 'distillation'
                   else ('dataset', 'checkpoint'))
        for option in options:
            command.extend(['--' + option.replace('_', '-'), value[option]])
        if kind == 'assessment_sft':
            command.extend(['--config', value['sft_config'], '--max-config-bytes',
                            str(value.get('max_sft_config_bytes', 65536))])
        if kind == 'distillation' and value.get('teacher_revision'):
            command.extend(['--teacher-revision', value['teacher_revision']])
        if kind == 'distillation' and value.get('initial_checkpoint'):
            command.extend(['--initial-checkpoint', value['initial_checkpoint']])
        command.extend(['--output', str(output), '--max-steps-this-run', str(value['segment_steps'])])
        if checkpoint:
            command.extend(['--resume', str(checkpoint[1]), '--resume-state-sha256', checkpoint[2]])
        report('training', step=checkpoint[0] if checkpoint else 0, total_steps=total)
        try:
            child = subprocess.Popen(command)
        except BaseException:
            lock.close()
            raise
        def terminate(signum, _frame):
            if child.poll() is None:
                child.send_signal(signum)
        previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            code = child.wait()
        except BaseException:
            lock.close()
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if code:
            report('failed', exit_code=code, step=checkpoint[0] if checkpoint else 0, total_steps=total)
            raise RuntimeError(f'training child exited with status {code}')
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if sft is not None:
                validate_sft_result(sft, manifest, output / 'weights.safetensors')
            elif (not isinstance(manifest, dict) or manifest.get('completed') is not True
                  or digest(output / 'weights.safetensors') != manifest.get('weights_sha256')):
                raise ValueError('invalid final model result')
            atomic_json(complete, {'completed_steps': manifest['steps'], 'weights_sha256': manifest['weights_sha256'],
                                   'result_path': str(output), 'assessment_quality_evaluated': False})
            break
        next_checkpoint = latest_checkpoint(segments, schema, value.get('max_checkpoint_state_bytes', 67108864), sft)
        if not next_checkpoint or next_checkpoint[0] <= (checkpoint[0] if checkpoint else 0):
            lock.close()
            raise ValueError('segment completed without advancing its checkpoint')
    result = json.loads(complete.read_text(encoding='utf-8'))
    report('complete', step=result['completed_steps'], total_steps=result['completed_steps'], weights_sha256=result['weights_sha256'],
           assessment_quality_evaluated=False)
    if exit_on_complete:
        lock.close()
        return
    # Keep the user service alive for status inspection, without repeatedly training the same data.
    while True:
        time.sleep(value['status_poll_seconds'])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--exit-on-complete', action='store_true')
    args = parser.parse_args()
    try:
        run(args.config, exit_on_complete=args.exit_on_complete)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f'LFG training service: {error}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
