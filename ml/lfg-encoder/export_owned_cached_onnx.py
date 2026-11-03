"""Export separate cache-aware prefill/decode ONNX graphs for an owned byte student.

The export is experimental. It preserves the full-prefix artifact and weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

import owned_student
from owned_cached_decode import DecodeSession

MAX_CONFIG_BYTES = 1 << 20


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_bounded(path: Path, maximum: int) -> bytes:
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('input exceeds configured limit')
    return raw


def private_output(directory: Path) -> None:
    resolved = directory.resolve()
    parent = resolved.parent
    while not parent.exists():
        parent = parent.parent
    try:
        result = subprocess.run(['git', '-C', str(parent), 'rev-parse', '--show-toplevel'],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            if any((part / '.git').exists() for part in (parent, *parent.parents)):
                raise ValueError('cannot verify output privacy')
            return
        repository = Path(result.stdout.rstrip('\r\n')).resolve()
        for name in ('prefill.onnx', 'decode.onnx', 'manifest.json', 'measurement.json'):
            relative = (resolved / name).relative_to(repository).as_posix()
            ignored = subprocess.run(['git', '-C', str(repository), 'check-ignore', '-q', '--no-index', '--', relative],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            tracked = subprocess.run(['git', '-C', str(repository), 'ls-files', '--error-unmatch', '--', relative],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if ignored.returncode != 0 or tracked.returncode != 1:
                raise ValueError('export outputs must be ignored and untracked')
    except OSError:
        raise ValueError('cannot verify output privacy') from None


def cache_layout(architecture: dict) -> list[dict]:
    layout = []
    for index in range(architecture['layers']):
        if architecture['conv_kernel'] > 1:
            layout.append({'name': f'conv_{index}', 'kind': 'depthwise-convolution-history', 'layer': index,
                           'dtype': 'float32', 'shape': ['batch', architecture['width'], architecture['conv_kernel'] - 1],
                           'capacity': architecture['conv_kernel'] - 1})
        if (index + 1) % architecture['attention_every'] == 0:
            head_dim = architecture['width'] // architecture['heads']
            for kind in ('key', 'value'):
                layout.append({'name': f'{kind}_{index}', 'kind': f'attention-{kind}', 'layer': index,
                               'dtype': 'float32', 'shape': ['batch', architecture['heads'], 'position', head_dim],
                               'capacity': architecture['context']})
    return layout


class PrefillGraph(torch.nn.Module):
    def __init__(self, model: owned_student.OwnedByteStudent, layout: list[dict]):
        super().__init__()
        self.model = model
        self.layout = layout

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.model.embed(tokens) + self.model.position(positions)[None, :, :]
        cache_values = []
        for index, block in enumerate(self.model.blocks):
            content, gate = block.input_gate(block.norm_conv(x)).chunk(2, dim=-1)
            convolved = block.conv(F.pad(content.transpose(1, 2), (block.kernel - 1, 0))).transpose(1, 2)
            x = x + block.conv_out(convolved * torch.sigmoid(gate))
            if block.kernel > 1:
                history = F.pad(content.transpose(1, 2), (block.kernel - 1, 0))
                cache_values.append(history[:, :, -(block.kernel - 1):])
            if block.attention:
                batch, length, width = x.shape
                q, k, v = block.qkv(block.norm_attn(x)).chunk(3, dim=-1)

                def split(tensor):
                    return tensor.reshape(batch, length, block.heads, width // block.heads).transpose(1, 2)

                q, k, v = split(q), split(k), split(v)
                cache_values.extend((k, v))
                attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                x = x + block.attn_out(attended.transpose(1, 2).contiguous().reshape(batch, length, width))
            x = x + block.ff(block.norm_ff(x))
        logits = self.model.output(self.model.norm(x[:, -1, :]))
        return (logits, *cache_values)


class DecodeGraph(torch.nn.Module):
    def __init__(self, model: owned_student.OwnedByteStudent, layout: list[dict]):
        super().__init__()
        self.model = model
        self.layout = layout

    def forward(self, token, position, *cache_values):
        x = self.model.embed(token).unsqueeze(1) + self.model.position(position)[None, None, :]
        cache_by_name = {entry['name']: value for entry, value in zip(self.layout, cache_values)}
        updated = []
        for index, block in enumerate(self.model.blocks):
            content, gate = block.input_gate(block.norm_conv(x)).chunk(2, dim=-1)
            content_channels = content.transpose(1, 2)
            if block.kernel > 1:
                window = torch.cat((cache_by_name[f'conv_{index}'], content_channels), dim=2)
                conv_cache = window[:, :, 1:]
            else:
                window = content_channels
            convolved = block.conv(window).transpose(1, 2)
            x = x + block.conv_out(convolved * torch.sigmoid(gate))
            if block.kernel > 1:
                updated.append(conv_cache)
            if block.attention:
                batch, _, width = x.shape
                q, k, v = block.qkv(block.norm_attn(x)).chunk(3, dim=-1)

                def split(tensor):
                    return tensor.reshape(batch, 1, block.heads, width // block.heads).transpose(1, 2)

                q, k, v = split(q), split(k), split(v)
                keys = torch.cat((cache_by_name[f'key_{index}'], k), dim=2)
                values = torch.cat((cache_by_name[f'value_{index}'], v), dim=2)
                attended = F.scaled_dot_product_attention(q, keys, values, is_causal=False)
                x = x + block.attn_out(attended.transpose(1, 2).contiguous().reshape(batch, 1, width))
                updated.extend((keys, values))
            x = x + block.ff(block.norm_ff(x))
        logits = self.model.output(self.model.norm(x)).squeeze(1)
        return (logits, *updated)


def export(checkpoint: Path, config_path: Path, source: Path, output: Path,
           additional_sources: list[Path] | None = None, decode_steps: int = 32,
           warmup: int = 1, repeats: int = 3) -> dict:
    import numpy as np
    import onnx
    import onnxruntime as ort

    config_raw = read_bounded(config_path, MAX_CONFIG_BYTES)
    config = json.loads(config_raw)
    integer_fields = ('opset', 'cpu_threads', 'max_model_bytes', 'max_checkpoint_bytes', 'max_manifest_bytes')
    if (not isinstance(config, dict) or config.get('schema_version') != 'lfg.owned-byte-onnx-export.v1'
            or any(type(config.get(key)) is not int or config[key] < 1 for key in integer_fields)
            or not isinstance(config.get('parity_lengths'), list) or not config['parity_lengths']
            or any(type(length) is not int or length < 1 for length in config['parity_lengths'])
            or len(set(config['parity_lengths'])) != len(config['parity_lengths'])
            or any(type(config.get(key)) not in (float, int) or not math.isfinite(config[key]) or config[key] < 0
                   for key in ('absolute_tolerance', 'relative_tolerance'))
            or type(decode_steps) is not int or decode_steps < 1
            or type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1):
        raise ValueError('invalid cached export configuration')
    manifest_path, weights_path = checkpoint / 'manifest.json', checkpoint / 'weights.safetensors'
    manifest_raw = read_bounded(manifest_path, config['max_manifest_bytes'])
    checkpoint_manifest = json.loads(manifest_raw)
    weights_raw = read_bounded(weights_path, config['max_checkpoint_bytes'])
    model_code_path = Path(owned_student.__file__)
    model_code_raw = read_bounded(model_code_path, config['max_manifest_bytes'])
    model_code_sha = digest(model_code_raw)
    if (digest(weights_raw) != checkpoint_manifest['weights_sha256']
            or model_code_sha != checkpoint_manifest['code_sha256']['owned_student.py']):
        raise ValueError('checkpoint weights or model implementation do not match manifest')
    model = owned_student.OwnedByteStudent(checkpoint_manifest['architecture']).eval()
    if max(config['parity_lengths']) > model.context:
        raise ValueError('parity length exceeds model context')
    sources = [source, *(additional_sources or [])]
    source_raws = [read_bounded(path, model.context) for path in sources]
    if any(not raw for raw in source_raws):
        raise ValueError('development inputs must be nonempty')
    if max(config['parity_lengths']) > len(source_raws[0]):
        raise ValueError('first development input is shorter than configured parity lengths')
    if output.exists():
        raise ValueError('output directory must be new')
    private_output(output)
    model.load_state_dict(load_file(str(weights_path), device='cpu'))
    torch.set_num_threads(config['cpu_threads'])
    layout = cache_layout(checkpoint_manifest['architecture'])
    prefill = PrefillGraph(model, layout).eval()
    decode = DecodeGraph(model, layout).eval()
    trace_length = min(max(config['parity_lengths']), model.context)
    trace_tokens = torch.tensor([list(source_raws[0][:trace_length])], dtype=torch.long)
    with torch.inference_mode():
        traced_prefill = prefill(trace_tokens)
        prefill_state = list(traced_prefill[1:])
        traced_decode = decode(trace_tokens[:, -1], torch.tensor(trace_length, dtype=torch.long), *prefill_state)
    output.mkdir(mode=0o700)
    prefill_path, decode_path = output / 'prefill.onnx', output / 'decode.onnx'
    state_names = [entry['name'] for entry in layout]
    prefill_output_names = ['logits', *state_names]
    decode_input_names = ['token', 'position', *state_names]
    decode_output_names = ['logits', *[f'{name}_out' for name in state_names]]
    prefill_dynamic_axes = {'tokens': {0: 'batch', 1: 'sequence'}, 'logits': {0: 'batch'}}
    decode_dynamic_axes = {'token': {0: 'batch'}, 'logits': {0: 'batch'}}
    for entry in layout:
        name = entry['name']
        prefill_dynamic_axes[name] = {0: 'batch'}
        decode_dynamic_axes[name] = {0: 'batch'}
        decode_dynamic_axes[f'{name}_out'] = {0: 'batch'}
        if entry['kind'].startswith('attention-'):
            prefill_dynamic_axes[name][2] = 'sequence'
            decode_dynamic_axes[name][2] = 'past_length'
            decode_dynamic_axes[f'{name}_out'][2] = 'present_length'
    with torch.inference_mode():
        torch.onnx.export(prefill, (trace_tokens,), str(prefill_path), input_names=['tokens'],
                          output_names=prefill_output_names, dynamic_axes=prefill_dynamic_axes,
                          opset_version=config['opset'], dynamo=False)
        torch.onnx.export(decode, (trace_tokens[:, -1], torch.tensor(trace_length, dtype=torch.long), *prefill_state),
                          str(decode_path), input_names=decode_input_names, output_names=decode_output_names,
                          dynamic_axes=decode_dynamic_axes, opset_version=config['opset'], dynamo=False)
    prefill_raw = read_bounded(prefill_path, config['max_model_bytes'])
    decode_raw = read_bounded(decode_path, config['max_model_bytes'])
    for raw in (prefill_raw, decode_raw):
        graph = onnx.load_from_string(raw)
        onnx.checker.check_model(graph)
        for tensor in graph.graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
                raise ValueError('external ONNX tensors are not supported')
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = config['cpu_threads']
    session_options.inter_op_num_threads = 1
    prefill_session = ort.InferenceSession(prefill_raw, sess_options=session_options, providers=['CPUExecutionProvider'])
    decode_session = ort.InferenceSession(decode_raw, sess_options=session_options, providers=['CPUExecutionProvider'])
    max_logit_diff = 0.0
    max_cache_diff = 0.0
    exact_greedy = True
    source_measurements = []
    for source_index, raw in enumerate(source_raws):
        source_tokens = torch.tensor([list(raw)], dtype=torch.long)
        prefix_length = len(raw)
        if source_index == 0:
            for parity_length in config['parity_lengths']:
                parity_tokens = source_tokens[:, :parity_length]
                parity_outputs = prefill_session.run(None, {'tokens': parity_tokens.numpy()})
                with torch.inference_mode():
                    parity_expected = model(parity_tokens)[:, -1, :].detach().numpy()
                np.testing.assert_allclose(parity_outputs[0], parity_expected,
                                           rtol=config['relative_tolerance'], atol=config['absolute_tolerance'])
                parity_cache = DecodeSession(model, 1)
                with torch.inference_mode():
                    parity_python_logits = None
                    for token_index in range(parity_length):
                        parity_python_logits = parity_cache.step(parity_tokens[:, token_index])
                np.testing.assert_allclose(parity_outputs[0], parity_python_logits.detach().cpu().numpy(),
                                           rtol=config['relative_tolerance'], atol=config['absolute_tolerance'])
                for entry, state_value in zip(layout, parity_outputs[1:]):
                    reference = parity_cache._states[entry['layer']]
                    kind = entry['kind']
                    reference_key = 'conv' if kind == 'depthwise-convolution-history' else ('keys' if kind == 'attention-key' else 'values')
                    expected_state = reference[reference_key].detach().cpu().numpy()
                    if state_value.shape != expected_state.shape or not np.isfinite(state_value).all():
                        raise ValueError('invalid parity cache state')
                    np.testing.assert_allclose(state_value, expected_state,
                                               rtol=config['relative_tolerance'], atol=config['absolute_tolerance'])
                    max_cache_diff = max(max_cache_diff,
                                         float(np.max(np.abs(state_value - expected_state))) if state_value.size else 0.0)
                max_logit_diff = max(max_logit_diff, float(np.max(np.abs(parity_outputs[0] - parity_expected))))
        prefill_start = time.perf_counter_ns()
        onnx_state = prefill_session.run(None, {'tokens': source_tokens.numpy()})
        prefill_elapsed = time.perf_counter_ns() - prefill_start
        expected = model(source_tokens)[:, -1, :].detach().numpy()
        actual = onnx_state[0]
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError('invalid cache prefill output')
        np.testing.assert_allclose(actual, expected, rtol=config['relative_tolerance'], atol=config['absolute_tolerance'])
        max_logit_diff = max(max_logit_diff, float(np.max(np.abs(actual - expected))))
        python_cache = DecodeSession(model, 1)
        with torch.inference_mode():
            python_logits = torch.stack([python_cache.step(source_tokens[:, index]) for index in range(prefix_length)], dim=1)[:, -1, :].numpy()
        np.testing.assert_allclose(actual, python_logits, rtol=config['relative_tolerance'], atol=config['absolute_tolerance'])
        for entry, state_value in zip(layout, onnx_state[1:]):
            reference = python_cache._states[entry['layer']]
            kind = entry['kind']
            reference_key = 'conv' if kind == 'depthwise-convolution-history' else ('keys' if kind == 'attention-key' else 'values')
            expected_state = reference[reference_key].detach().cpu().numpy()
            if state_value.shape != expected_state.shape or not np.isfinite(state_value).all():
                raise ValueError('invalid cache state output')
            np.testing.assert_allclose(state_value, expected_state, rtol=config['relative_tolerance'], atol=config['absolute_tolerance'])
            max_cache_diff = max(max_cache_diff, float(np.max(np.abs(state_value - expected_state))) if state_value.size else 0.0)
        bytes_out = []
        decode_samples = []
        decode_cpu_samples = []
        state_values = list(onnx_state[1:])
        logits = actual
        measured_steps = min(decode_steps, model.context - prefix_length)
        if measured_steps < 1:
            raise ValueError('development input leaves no context for decode measurement')
        for step_index in range(measured_steps):
            next_byte = int(np.argmax(logits[0]))
            bytes_out.append(next_byte)
            token = np.asarray([next_byte], dtype=np.int64)
            position = np.asarray(prefix_length + step_index, dtype=np.int64)
            feeds = {'token': token, 'position': position}
            feeds.update({name: value for name, value in zip(state_names, state_values)})
            start = time.perf_counter_ns()
            cpu_start = time.process_time_ns()
            decoded = decode_session.run(None, feeds)
            decode_samples.append(time.perf_counter_ns() - start)
            decode_cpu_samples.append(time.process_time_ns() - cpu_start)
            logits, *state_values = decoded
            with torch.inference_mode():
                full_tokens = torch.tensor([list(raw) + bytes_out], dtype=torch.long)
                expected_next = model(full_tokens)[:, -1, :].detach().numpy()
                cached_reference = python_cache.step(torch.tensor([next_byte], dtype=torch.long)).detach().cpu().numpy()
            if logits.shape != expected_next.shape or not np.isfinite(logits).all():
                raise ValueError('invalid cache decode logits')
            np.testing.assert_allclose(logits, expected_next, rtol=config['relative_tolerance'], atol=config['absolute_tolerance'])
            np.testing.assert_allclose(logits, cached_reference, rtol=config['relative_tolerance'], atol=config['absolute_tolerance'])
            max_logit_diff = max(max_logit_diff, float(np.max(np.abs(logits - expected_next))))
            for entry, state_value in zip(layout, state_values):
                reference = python_cache._states[entry['layer']]
                kind = entry['kind']
                reference_key = 'conv' if kind == 'depthwise-convolution-history' else ('keys' if kind == 'attention-key' else 'values')
                expected_state = reference[reference_key].detach().cpu().numpy()
                if state_value.shape != expected_state.shape or not np.isfinite(state_value).all():
                    raise ValueError('invalid cache decode state')
                np.testing.assert_allclose(state_value, expected_state, rtol=config['relative_tolerance'], atol=config['absolute_tolerance'])
                max_cache_diff = max(max_cache_diff, float(np.max(np.abs(state_value - expected_state))) if state_value.size else 0.0)
            if int(np.argmax(logits[0])) != int(np.argmax(expected_next[0])):
                exact_greedy = False
        # Warmed, alternating repeats measure prefill and a decode batch without exposing tokens or text.
        for _ in range(warmup):
            prefill_session.run(None, {'tokens': source_tokens.numpy()})
        prefill_samples, decode_batch_samples = [], []
        prefill_cpu_samples, decode_batch_cpu_samples = [], []
        for repeat_index in range(repeats):
            start = time.perf_counter_ns()
            cpu_start = time.process_time_ns()
            warmed_state = prefill_session.run(None, {'tokens': source_tokens.numpy()})
            prefill_samples.append(time.perf_counter_ns() - start)
            prefill_cpu_samples.append(time.process_time_ns() - cpu_start)
            feed_state = list(warmed_state[1:])
            logits_current = warmed_state[0]
            generated = []
            start = time.perf_counter_ns()
            cpu_start = time.process_time_ns()
            for index in range(measured_steps):
                byte = int(np.argmax(logits_current[0]))
                generated.append(byte)
                feed = {'token': np.asarray([byte], dtype=np.int64),
                        'position': np.asarray(prefix_length + index, dtype=np.int64)}
                feed.update({name: value for name, value in zip(state_names, feed_state)})
                result = decode_session.run(None, feed)
                logits_current, *feed_state = result
            decode_batch_samples.append(time.perf_counter_ns() - start)
            decode_batch_cpu_samples.append(time.process_time_ns() - cpu_start)
        source_measurements.append({
            'source_index': source_index, 'source_bytes': prefix_length,
            'source_sha256': digest(raw), 'decode_steps': measured_steps,
            'prefill_ms_first': prefill_elapsed / 1e6,
            'prefill_ms_median': statistics.median(prefill_samples) / 1e6,
            'prefill_cpu_ms_median': statistics.median(prefill_cpu_samples) / 1e6,
            'decode_total_ms_median': statistics.median(decode_batch_samples) / 1e6,
            'decode_cpu_ms_median': statistics.median(decode_batch_cpu_samples) / 1e6,
            'decode_ms_per_byte_median': statistics.median(decode_batch_samples) / measured_steps / 1e6,
            'decode_step_ms_median': statistics.median(decode_samples) / 1e6,
            'decode_step_cpu_ms_median': statistics.median(decode_cpu_samples) / 1e6,
            'full_prefix_cached_parity_checked': True,
        })
    if not exact_greedy:
        raise ValueError('cached ONNX greedy output differs from full-prefix output')
    if (read_bounded(config_path, MAX_CONFIG_BYTES) != config_raw
            or read_bounded(manifest_path, config['max_manifest_bytes']) != manifest_raw
            or read_bounded(weights_path, config['max_checkpoint_bytes']) != weights_raw
            or read_bounded(model_code_path, config['max_manifest_bytes']) != model_code_raw
            or any(read_bounded(path, model.context) != raw for path, raw in zip(sources, source_raws))):
        raise ValueError('export inputs changed during the run')
    model_manifest = {
        'schema_version': 'lfg.owned-byte-cached-onnx.v1',
        'artifact_version': 1,
        'promotion_eligible': False,
        'model_sha256': digest(weights_raw),
        'source_manifest_sha256': digest(manifest_raw),
        'source_model_code_sha256': model_code_sha,
        'export_code_sha256': digest(Path(__file__).read_bytes()),
        'export_config_sha256': digest(config_raw),
        'prefill_onnx_sha256': digest(prefill_raw),
        'decode_onnx_sha256': digest(decode_raw),
        'prefill': {'file': 'prefill.onnx', 'inputs': ['tokens'], 'outputs': prefill_output_names,
                    'input_dtype': 'int64', 'logit_dtype': 'float32', 'input_shape': ['batch', 'sequence'],
                    'sequence_min': 1, 'sequence_max': model.context},
        'decode': {'file': 'decode.onnx', 'inputs': decode_input_names, 'outputs': decode_output_names,
                   'token_dtype': 'int64', 'position_dtype': 'int64', 'logit_dtype': 'float32',
                   'token_shape': ['batch'], 'position_shape': [], 'consumes_one_token': True},
        'cache': {'states': layout, 'position_semantics': 'number of bytes already consumed; absolute positions are preserved',
                  'attention_cache_length_must_equal_position': True, 'capacity': model.context,
                  'maximum_state_bytes_per_batch': 4 * sum(
                      (entry['shape'][1] * model.context * entry['shape'][3]
                       if entry['kind'].startswith('attention-')
                       else entry['shape'][1] * entry['shape'][2])
                      for entry in layout),
                  'overflow': 'reject', 'eviction': 'none',
                  'initial_attention_cache_shape': ['batch', 'heads', 0, 'head_dim'],
                  'initial_convolution_cache': 'zeros with declared fixed shape',
                  'consumer_must_validate': ['input bytes are in [0,255]',
                                             'prefill sequence length is 1..context',
                                             'decode position is less than context',
                                             'attention cache length equals position',
                                             'cache batch, dtype, rank, and fixed dimensions match states']},
        'architecture': checkpoint_manifest['architecture'],
        'context': model.context,
        'vocabulary': 256,
        'dtype': 'float32',
        'opset': config['opset'],
        'parity_lengths': config['parity_lengths'],
        'prefill_parity_checked': True,
        'max_absolute_logit_difference': max_logit_diff,
        'max_absolute_cache_difference': max_cache_diff,
        'greedy_bytes_exact': exact_greedy,
        'torch': torch.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__,
        'native_runtime_checked': False, 'assessment_quality_checked': False,
    }
    measurement = {
        'schema_version': 'lfg.owned-cached-onnx-measurement.v1',
        'artifact_manifest_sha256': digest((json.dumps(model_manifest, sort_keys=True, separators=(',', ':')) + '\n').encode()),
        'checkpoint_manifest_sha256': digest(manifest_raw), 'weights_sha256': digest(weights_raw),
        'model_code_sha256': model_code_sha,
        'cached_reference_code_sha256': digest(Path(__file__).with_name('owned_cached_decode.py').read_bytes()),
        'export_code_sha256': digest(Path(__file__).read_bytes()), 'export_config_sha256': digest(config_raw),
        'cpu_threads': config['cpu_threads'], 'warmup_runs': warmup, 'repeats': repeats,
        'decode_steps': decode_steps, 'samples': source_measurements,
        'platform': platform.platform(), 'processor': platform.processor(),
        'parity': {'max_absolute_logit_difference': max_logit_diff,
                   'max_absolute_cache_difference': max_cache_diff, 'greedy_bytes_exact': exact_greedy},
        'assessment_quality_evaluated': False,
    }
    manifest_bytes = (json.dumps(model_manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()
    measurement_bytes = (json.dumps(measurement, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()
    for path, data in ((output / 'manifest.json', manifest_bytes), (output / 'measurement.json', measurement_bytes)):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
    return {'status': 'verified', 'source_count': len(source_raws), 'parity_lengths': len(config['parity_lengths']),
            'decode_steps': decode_steps, 'greedy_bytes_exact': exact_greedy}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'config', 'source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--additional-source', type=Path, action='append', default=[])
    parser.add_argument('--decode-steps', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    arguments = parser.parse_args()
    result = export(arguments.checkpoint, arguments.config, arguments.source, arguments.output,
                    arguments.additional_source, arguments.decode_steps, arguments.warmup, arguments.repeats)
    # Do not print private paths, inputs, model output, or artifact digests.
    print(json.dumps(result, separators=(',', ':')))
