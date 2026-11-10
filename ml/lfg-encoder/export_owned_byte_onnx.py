"""Export a private owned byte checkpoint to ONNX and check CPU parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import owned_student


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_bounded(path: Path, limit: int) -> bytes:
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('input exceeds configured limit')
    return data


def private_output(directory: Path) -> None:
    directory = directory.resolve()
    parent = directory.parent
    while not parent.exists():
        parent = parent.parent
    try:
        root = subprocess.run(['git', '-C', str(parent), 'rev-parse', '--show-toplevel'],
                              capture_output=True, text=True, check=False)
        if root.returncode:
            if any((part / '.git').exists() for part in (parent, *parent.parents)):
                raise ValueError('cannot verify output privacy')
            return
        repository = Path(root.stdout.rstrip('\r\n')).resolve()
        for name in ('model.onnx', 'manifest.json'):
            relative = (directory / name).relative_to(repository).as_posix()
            ignored = subprocess.run(['git', '-C', str(repository), 'check-ignore', '-q', '--no-index', '--', relative],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            tracked = subprocess.run(['git', '-C', str(repository), 'ls-files', '--error-unmatch', '--', relative],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if ignored.returncode != 0 or tracked.returncode != 1:
                raise ValueError('export outputs must be ignored and untracked')
    except OSError:
        raise ValueError('cannot verify output privacy') from None


def reject_external(message, onnx):
    if message.DESCRIPTOR.full_name == 'onnx.TensorProto' and (message.external_data or message.data_location == onnx.TensorProto.EXTERNAL):
        raise ValueError('external ONNX tensor data is not supported')
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            for child in (value if field.is_repeated else [value]):
                reject_external(child, onnx)


def export(checkpoint: Path, config_path: Path, source_path: Path, output: Path) -> dict:
    import numpy as np
    import onnx
    os.environ['ORT_DISABLE_TELEMETRY'] = '1'  # Set before import, or model names and hashes go to Microsoft.
    import onnxruntime as ort
    import torch
    from safetensors.torch import load_file

    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    integer_fields = ('opset', 'cpu_threads', 'max_model_bytes', 'max_checkpoint_bytes', 'max_manifest_bytes')
    if (not isinstance(config, dict) or config.get('schema_version') != 'lfg.owned-byte-onnx-export.v1'
            or any(type(config.get(key)) is not int or config[key] < 1 for key in integer_fields)
            or not isinstance(config.get('parity_lengths'), list) or not config['parity_lengths']
            or any(type(n) is not int or n < 1 for n in config['parity_lengths'])
            or len(set(config['parity_lengths'])) != len(config['parity_lengths'])
            or any(type(config.get(key)) not in (float, int) or not math.isfinite(config[key]) or config[key] < 0
                   for key in ('absolute_tolerance', 'relative_tolerance'))):
        raise ValueError('invalid export configuration')
    manifest_path, weights_path = checkpoint / 'manifest.json', checkpoint / 'weights.safetensors'
    manifest_raw = read_bounded(manifest_path, config['max_manifest_bytes'])
    manifest = json.loads(manifest_raw)
    weights_raw = read_bounded(weights_path, config['max_checkpoint_bytes'])
    code_sha = digest(Path(owned_student.__file__).read_bytes())
    if (digest(weights_raw) != manifest['weights_sha256']
            or code_sha != manifest['code_sha256']['owned_student.py']):
        raise ValueError('checkpoint does not match recorded weights or model code')
    model = owned_student.OwnedByteStudent(manifest['architecture']).eval()
    if max(config['parity_lengths']) > model.context:
        raise ValueError('parity length exceeds model context')
    with source_path.open('rb') as stream:
        source = stream.read(max(config['parity_lengths']))
    if len(source) != max(config['parity_lengths']):
        raise ValueError('parity source is too short')
    if output.exists():
        raise ValueError('output directory must be new')
    private_output(output)
    model.load_state_dict(load_file(str(weights_path), device='cpu'))
    torch.set_num_threads(config['cpu_threads'])
    output.mkdir(mode=0o700)
    model_path = output / 'model.onnx'
    example = torch.tensor([list(source)], dtype=torch.long)
    with torch.inference_mode():
        torch.onnx.export(model, (example,), str(model_path), input_names=['tokens'], output_names=['logits'],
                          opset_version=config['opset'], dynamo=False,
                          dynamic_axes={'tokens': {1: 'length'}, 'logits': {1: 'length'}})
    onnx_raw = read_bounded(model_path, config['max_model_bytes'])
    graph = onnx.load_from_string(onnx_raw)
    reject_external(graph, onnx)
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = config['cpu_threads']
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_raw, sess_options=options, providers=['CPUExecutionProvider'])
    max_diff = 0.0
    with torch.inference_mode():
        for length in config['parity_lengths']:
            token = torch.tensor([list(source[:length])], dtype=torch.long)
            expected = model(token).numpy()
            actual = session.run(['logits'], {'tokens': token.numpy()})[0]
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError('invalid ONNX output')
            np.testing.assert_allclose(actual, expected, rtol=config['relative_tolerance'], atol=config['absolute_tolerance'])
            max_diff = max(max_diff, float(np.max(np.abs(actual - expected))))
    with source_path.open('rb') as stream:
        source_after = stream.read(len(source))
    if (source_after != source or config_path.read_bytes() != config_raw or read_bounded(manifest_path, config['max_manifest_bytes']) != manifest_raw
            or read_bounded(weights_path, config['max_checkpoint_bytes']) != weights_raw
            or digest(Path(owned_student.__file__).read_bytes()) != code_sha):
        raise ValueError('export inputs changed during export')
    result = {'schema_version': 'lfg.owned-byte-onnx.v1', 'promotion_eligible': False,
              'model_sha256': digest(onnx_raw), 'source_manifest_sha256': digest(manifest_raw),
              'source_weights_sha256': digest(weights_raw), 'source_model_code_sha256': code_sha,
              'export_config_sha256': digest(config_raw), 'export_code_sha256': digest(Path(__file__).read_bytes()),
              'source_prefix_sha256': digest(source), 'context': model.context, 'vocabulary': 256,
              'input': 'tokens', 'output': 'logits', 'parity_lengths': config['parity_lengths'],
              'max_absolute_difference': max_diff, 'torch': torch.__version__, 'onnx': onnx.__version__,
              'onnxruntime': ort.__version__, 'native_runtime_checked': False, 'assessment_quality_checked': False}
    raw = (json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()
    descriptor = os.open(output / 'manifest.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
    return {'manifest_sha256': digest(raw), 'model_sha256': result['model_sha256'], 'max_absolute_difference': max_diff}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'config', 'source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.config, args.source, args.output)))
