"""Compare prefilling and incremental caches without exposing model values.

The diagnostic is CPU-only and read-only. It reports aggregate differences for
the ONNX prefill graph, the pinned PyTorch PrefillGraph, and the pinned
incremental DecodeSession. It does not export, promote, deploy, or evaluate
task quality. Host-level resource limits remain an independent launch duty.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import types

_CORE_PATH = Path(__file__).with_name('diagnose_owned_cached_parity.py')
_CORE_MODULE = None
_PINNED_CORE_RAW = None
_PINNED_CORE_DIGEST = None


class CorePinError(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class StartupPinError(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _valid_sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _captured_bytes_unchanged(path, raw, maximum):
    try:
        target = Path(path)
        if target.stat().st_size > maximum:
            return False
        observed = target.read_bytes()
    except OSError:
        return False
    return len(observed) <= maximum and hashlib.sha256(observed).digest() == hashlib.sha256(raw).digest()


def _execute_pinned_core(path, expected_sha256):
    """Execute the exact verified core bytes without module-name resolution."""
    if not _valid_sha256(expected_sha256):
        raise CorePinError('core_digest_pin_missing_or_invalid')
    try:
        raw = Path(path).read_bytes()
    except OSError:
        raise CorePinError('core_file_unavailable') from None
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise CorePinError('core_digest_mismatch')
    module = types.ModuleType('_lfg_prefill_isolation_pinned_core')
    module.__file__ = str(path)
    try:
        exec(compile(raw, str(path), 'exec'), module.__dict__)
    except Exception:
        raise CorePinError('core_execution_failed') from None
    return module, raw, digest


def _load_pinned_core(expected_sha256=None):
    global _CORE_MODULE, _PINNED_CORE_RAW, _PINNED_CORE_DIGEST
    expected = (expected_sha256 if expected_sha256 is not None else
                os.environ.get('LFG_PREFILL_CORE_EXPECTED_SHA256'))
    if not _valid_sha256(expected):
        raise CorePinError('core_digest_pin_missing_or_invalid')
    if _CORE_MODULE is not None:
        if _PINNED_CORE_DIGEST != expected:
            raise CorePinError('core_digest_mismatch')
        return _CORE_MODULE
    _CORE_MODULE, _PINNED_CORE_RAW, _PINNED_CORE_DIGEST = _execute_pinned_core(
        _CORE_PATH, expected)
    return _CORE_MODULE


class _PinnedCoreProxy:
    def __getattr__(self, name):
        return getattr(_load_pinned_core(), name)


core = _PinnedCoreProxy()

SESSION_VARIANTS = ('current_default', 'disable_all')
VARIANT_LEVELS = {
    'current_default': 'ORT_ENABLE_ALL',
    'disable_all': 'ORT_DISABLE_ALL',
}
def _tensor_values(torch, values):
    return [value.detach().to(device='cpu').numpy() for value in values]


def _incremental_values(torch, logits, cache, layout):
    result = [logits.detach().to(device='cpu').numpy()]
    for entry in layout:
        state = cache._states[entry['layer']]
        key = ('conv' if entry['kind'] == 'depthwise-convolution-history'
               else 'keys' if entry['kind'] == 'attention-key' else 'values')
        value = state[key]
        if value is None:
            raise core.DiagnosticAbort('cache_capture', 'incremental_cache_missing')
        result.append(value.detach().to(device='cpu').numpy())
    return result


def _pair_summary(actual, reference, labels, atol, rtol):
    if len(actual) != len(reference) or len(actual) != len(labels):
        return {'status': 'invalid', 'reason': 'output_identity_count_mismatch',
                'logits': None, 'cache_tensors': {}}
    metrics = [core.compare_arrays(a, b, atol, rtol) for a, b in zip(actual, reference)]
    cache = {label: metric for label, metric in zip(labels[1:], metrics[1:])}
    status = ('nonfinite' if any(item['status'] == 'nonfinite' for item in metrics)
              else 'invalid' if any(item['status'] != 'measured' for item in metrics)
              else 'measured')
    return {'status': status, 'logits': metrics[0], 'cache_tensors': cache}


def three_way_comparisons(onnx_values, full_values, incremental_values, labels, atol, rtol):
    """Return the three directed comparisons used for prefill isolation."""
    return {
        'onnx_vs_torch_full_prefill': _pair_summary(onnx_values, full_values, labels, atol, rtol),
        'torch_full_prefill_vs_torch_incremental': _pair_summary(
            full_values, incremental_values, labels, atol, rtol),
        'onnx_vs_torch_incremental': _pair_summary(
            onnx_values, incremental_values, labels, atol, rtol),
    }


def _metric_invalid(metric):
    if metric is None:
        return True
    if metric.get('status') != 'measured':
        return True
    if metric.get('nonfinite_count', 0) > 0:
        return True
    return False


def _pair_invalid(summary):
    if summary.get('status') != 'measured' or _metric_invalid(summary.get('logits')):
        return True
    return any(_metric_invalid(item) for item in summary.get('cache_tensors', {}).values())


def _pair_original_failures(summary):
    metrics = [summary.get('logits'), *summary.get('cache_tensors', {}).values()]
    return sum(int(item.get('failed_original_count', 0))
               for item in metrics if isinstance(item, dict))


def _trace_specs(model):
    specs = []
    for layer, block in enumerate(model.blocks):
        specs.append((layer, 'block_input_norm', block.norm_conv, None))
        specs.append((layer, 'input_gate_content', block.input_gate, model.embed.embedding_dim))
        specs.append((layer, 'convolution_branch', block.conv_out, None))
        if block.attention:
            specs.append((layer, 'post_convolution_residual_norm', block.norm_attn, None))
            specs.append((layer, 'attention_branch', block.attn_out, None))
        specs.append((layer, 'post_attention_residual_norm', block.norm_ff, None))
        specs.append((layer, 'feed_forward_branch', block.ff, None))
    return specs


def _trace_byte_estimate(model, length):
    width = model.embed.embedding_dim
    return len(_trace_specs(model)) * length * width * 4 * 2


def _attach_trace_hooks(specs, trace_values, active_path):
    handles = []
    for layer, operation, module, content_width in specs:
        key = f'layer_{layer}.{operation}'

        def capture(_module, _inputs, output, *, trace_key=key, width=content_width):
            import torch
            if not isinstance(output, torch.Tensor):
                trace_values[active_path[0]][trace_key].append(None)
                return
            tensor = output[..., :width] if width is not None else output
            trace_values[active_path[0]][trace_key].append(tensor.detach().to('cpu').clone())

        handles.append(module.register_forward_hook(capture))
    return handles


def _trace_summary(trace_values, specs, length, atol, rtol):
    entries = []
    earliest_difference = None
    earliest_tolerance_failure = None
    for layer, operation, _module, _width in specs:
        key = f'layer_{layer}.{operation}'
        full = trace_values['full'].get(key, [])
        incremental = trace_values['incremental'].get(key, [])
        if (not full or any(value is None for value in full)
                or len(incremental) != length or any(value is None for value in incremental)):
            metric = {'status': 'invalid', 'reason': 'intermediate_capture_incomplete'}
        else:
            try:
                import torch
                full_values = torch.cat(full, dim=1) if len(full) > 1 else full[0]
                incremental_values = torch.cat(incremental, dim=1)
                metric = core.compare_arrays(full_values.numpy(), incremental_values.numpy(), atol, rtol)
            except Exception:
                metric = {'status': 'invalid', 'reason': 'intermediate_shape_invalid'}
        row = {'layer': layer, 'operation': operation, 'full_vs_incremental': metric}
        entries.append(row)
        location = {'layer': layer, 'operation': operation,
                    'status': metric.get('status'),
                    'failed_original_count': metric.get('failed_original_count')}
        maximum = metric.get('max_absolute_error')
        if earliest_difference is None and (
                _metric_invalid(metric) or (maximum is not None and maximum > 0)):
            earliest_difference = location
        if earliest_tolerance_failure is None and (
                _metric_invalid(metric) or metric.get('failed_original_count', 0) > 0):
            earliest_tolerance_failure = location
    status = 'invalid' if any(_metric_invalid(row['full_vs_incremental']) for row in entries) else 'measured'
    return {'status': status, 'operations': entries,
            'earliest_numerical_difference': earliest_difference,
            'earliest_tolerance_divergence': earliest_tolerance_failure}


def _new_report(policy_raw, selection_raw, policy, source_count, lengths,
                cpu_threads, max_bytes, max_work_items, max_prefix_tokens,
                max_intermediate_bytes, max_seconds, variants):
    return {
        'schema_version': 'lfg.owned-cached-prefill-isolation-report.v1',
        'status': 'running', 'original_export_status': 'failed',
        'assessment_quality_evaluated': False, 'promotion_eligible': False,
        'policy_sha256': core.sha256(policy_raw),
        'selection_manifest_sha256': core.sha256(selection_raw),
        'original_failure': core._failure_summary(policy),
        'pins': {name: policy['pins'][name] for name in core.PIN_NAMES},
        'screening_claim_status': 'operator_claim_not_independently_verified',
        'source_count': source_count,
        'requested_work_items': source_count * len(lengths),
        'completed_work_items': 0,
        'invalid_work_items': 0, 'incomplete_work_items': 0,
        'crashed_work_items': 0, 'in_progress_work_items': 0,
        'not_tested_work_items': source_count * len(lengths),
        'requested_prefix_tokens': source_count * sum(lengths) * (2 + len(variants)),
        'requested_lengths': list(lengths),
        'additional_decode_steps_after_prefill': 0,
        'cpu_device': 'cpu', 'cpu_threads_requested': cpu_threads,
        'max_bytes_per_file': max_bytes,
        'max_sample_work_items': max_work_items,
        'max_prefix_tokens': max_prefix_tokens,
        'max_intermediate_bytes_per_sample': max_intermediate_bytes,
        'intermediate_capture_budget_semantics': 'estimate_only_not_a_process_memory_limit',
        'cooperative_time_limit_seconds': max_seconds,
        'session_variants': [
            {'name': name, 'graph_optimization_level': VARIANT_LEVELS[name],
             'intra_op_num_threads': cpu_threads, 'inter_op_num_threads': 1,
             'provider': 'CPUExecutionProvider'} for name in variants],
        'execution_control': {
            'timer_kind': 'cooperative_in_process',
            'hard_wall_time_enforced': False,
            'hard_cpu_or_memory_limit_enforced': False,
            'host_supervisor_verified_by_runner': False,
            'external_host_control_prerequisite': 'independent CPU, memory, pid, and wall supervision required'},
        'layer_intermediate_scope': 'PyTorch module outputs only; ONNX graph intermediates unavailable; no graph was modified',
        'samples': [], 'input_mutation_detected': False,
        'isolation_tool_expected_sha256': os.environ.get('LFG_PREFILL_ISOLATION_EXPECTED_SHA256'),
        'shared_core_sha256': _PINNED_CORE_DIGEST,
        'shared_core_expected_sha256': os.environ.get('LFG_PREFILL_CORE_EXPECTED_SHA256'),
        'launcher_prerequisite': 'external_sha256_bootstrap_must_verify_tool_and_core_raw_bytes_before_execution',
        'failure_stage': None, 'failure_code': None,
    }


def _refresh_work_counts(report):
    statuses = [sample.get('status') for sample in report.get('samples', [])]
    report['completed_work_items'] = statuses.count('measured')
    report['invalid_work_items'] = statuses.count('invalid')
    report['incomplete_work_items'] = statuses.count('incomplete')
    report['crashed_work_items'] = statuses.count('crash')
    report['in_progress_work_items'] = statuses.count('running')
    report['not_tested_work_items'] = statuses.count('not_tested')
    report['requested_work_items'] = len(statuses)
    assert report['requested_work_items'] == sum(report[key] for key in (
        'completed_work_items', 'invalid_work_items', 'incomplete_work_items',
        'crashed_work_items', 'in_progress_work_items', 'not_tested_work_items'))


def diagnose(checkpoint: Path, export_config: Path, graphs: Path, policy_path: Path,
             selection_manifest_path: Path, source_paths: dict[str, Path],
             screening_manifest_paths: dict[str, Path], output: Path, *,
             lengths: list[int], cpu_threads: int, max_bytes: int,
             max_work_items: int, max_prefix_tokens: int,
             max_intermediate_bytes: int, max_seconds: float,
             optimization_variants: tuple[str, ...] = SESSION_VARIANTS) -> dict:
    started = time.monotonic()
    cpu_started = time.process_time()
    module_stack = ExitStack()
    raw_inputs = {}
    safe_sources = {}
    output_dir = None
    policy = None
    report = {
        'schema_version': 'lfg.owned-cached-prefill-isolation-report.v1',
        'status': 'running', 'original_export_status': 'failed',
        'assessment_quality_evaluated': False, 'promotion_eligible': False,
        'samples': [], 'requested_work_items': None, 'completed_work_items': 0,
        'invalid_work_items': 0, 'incomplete_work_items': 0,
        'crashed_work_items': 0, 'in_progress_work_items': 0,
        'not_tested_work_items': 0, 'input_mutation_detected': False,
        'execution_control': {'timer_kind': 'cooperative_in_process',
            'hard_wall_time_enforced': False, 'hard_cpu_or_memory_limit_enforced': False,
            'host_supervisor_verified_by_runner': False,
            'external_host_control_prerequisite': 'independent CPU, memory, pid, and wall supervision required'}}
    try:
        tool_path = Path(__file__)
        expected_tool_sha256 = os.environ.get('LFG_PREFILL_ISOLATION_EXPECTED_SHA256')
        expected_core_sha256 = os.environ.get('LFG_PREFILL_CORE_EXPECTED_SHA256')
        if not _valid_sha256(expected_tool_sha256):
            raise StartupPinError('tool_digest_pin_missing_or_invalid')
        try:
            tool_raw = tool_path.read_bytes()
        except OSError:
            raise StartupPinError('tool_file_unavailable') from None
        actual_tool_sha256 = hashlib.sha256(tool_raw).hexdigest()
        if expected_tool_sha256 != actual_tool_sha256:
            raise StartupPinError('external_tool_digest_mismatch')
        _load_pinned_core(expected_core_sha256)
        raw_inputs['isolation_tool'] = (tool_path, tool_raw, max_bytes)
        raw_inputs['shared_core'] = (_CORE_PATH, _PINNED_CORE_RAW, max_bytes)
        import numpy as np
        import onnx
        import onnxruntime as ort
        import safetensors
        import torch
        from safetensors.torch import load as load_safetensors

        policy_raw = core.read_bounded(policy_path, core.MAX_POLICY_BYTES)
        raw_inputs['policy'] = (policy_path, policy_raw, core.MAX_POLICY_BYTES)
        policy = core._load_json(policy_raw)
        core._validate_policy(policy)
        if (not isinstance(lengths, list) or not lengths
                or any(type(item) is not int or item < 1 for item in lengths)
                or len(set(lengths)) != len(lengths)
                or type(cpu_threads) is not int or cpu_threads < 1
                or type(max_bytes) is not int or max_bytes < 1
                or type(max_work_items) is not int or not 1 <= max_work_items <= core.MAX_WORK_ITEMS
                or type(max_prefix_tokens) is not int or not 1 <= max_prefix_tokens <= core.MAX_PREFIX_TOKENS
                or type(max_intermediate_bytes) is not int or max_intermediate_bytes < 1
                or not core._finite_number(max_seconds) or max_seconds <= 0):
            raise core.DiagnosticAbort('preflight', 'execution_bounds_invalid')
        variants = tuple(optimization_variants)
        if (len(variants) != len(SESSION_VARIANTS)
                or set(variants) != set(SESSION_VARIANTS)):
            raise core.DiagnosticAbort('preflight', 'optimization_variants_invalid')
        if set(source_paths) != set(screening_manifest_paths) or not source_paths:
            raise core.DiagnosticAbort('preflight', 'source_screening_identity_mismatch')
        if len(source_paths) * len(lengths) > max_work_items:
            raise core.DiagnosticAbort('preflight', 'sample_work_limit')
        prefix_work_tokens = len(source_paths) * sum(lengths) * (2 + len(variants))
        if prefix_work_tokens > max_prefix_tokens:
            raise core.DiagnosticAbort('preflight', 'prefix_token_limit')

        def capture(label, path, limit, pin_name=None):
            raw = core.read_bounded(path, limit)
            raw_inputs[label] = (path, raw, limit)
            if pin_name and core.sha256(raw) != policy['pins'][pin_name]:
                raise core.DiagnosticAbort('preflight', 'pinned_input_mismatch')
            return raw

        selection_raw = capture('selection', selection_manifest_path,
                                 core.MAX_POLICY_BYTES, 'selection_manifest_sha256')
        selection = core._load_json(selection_raw)
        selection_entries = core._validate_selection(selection)
        if set(selection_entries) != set(source_paths):
            raise core.DiagnosticAbort('preflight', 'source_selection_identity_mismatch')
        manifest_raw = capture('checkpoint_manifest', checkpoint / 'manifest.json',
                               max_bytes, 'checkpoint_manifest_sha256')
        weights_raw = capture('weights', checkpoint / 'weights.safetensors',
                              max_bytes, 'weights_sha256')
        model_code_path = Path(__file__).with_name('owned_student.py')
        cache_code_path = Path(__file__).with_name('owned_cached_decode.py')
        exporter_code_path = Path(__file__).with_name('export_owned_cached_onnx.py')
        model_code_raw = capture('model_code', model_code_path, max_bytes, 'model_code_sha256')
        cache_code_raw = capture('cached_reference_code', cache_code_path, max_bytes,
                                 'cached_reference_code_sha256')
        exporter_code_raw = capture('exporter_code', exporter_code_path, max_bytes,
                                    'exporter_code_sha256')
        config_raw = capture('export_config', export_config, core.MAX_POLICY_BYTES,
                             'export_config_sha256')
        prefill_raw = capture('prefill_graph', graphs / 'prefill.onnx', max_bytes,
                              'prefill_onnx_sha256')
        decode_raw = capture('decode_graph', graphs / 'decode.onnx', max_bytes,
                             'decode_onnx_sha256')
        config = core._load_json(config_raw)
        if (not isinstance(config, dict)
                or config.get('schema_version') != 'lfg.owned-byte-onnx-export.v1'
                or config.get('absolute_tolerance') != core.ORIGINAL_ATOL
                or config.get('relative_tolerance') != core.ORIGINAL_RTOL):
            raise core.DiagnosticAbort('preflight', 'export_config_contract_mismatch')
        manifest = core._load_json(manifest_raw)
        if (not isinstance(manifest, dict)
                or manifest.get('weights_sha256') != core.sha256(weights_raw)
                or manifest.get('code_sha256', {}).get('owned_student.py') != core.sha256(model_code_raw)):
            raise core.DiagnosticAbort('preflight', 'checkpoint_manifest_contract_mismatch')
        screen_raws = {}
        for source_id, screen_path in screening_manifest_paths.items():
            raw = capture('screening_' + source_id, screen_path, core.MAX_POLICY_BYTES)
            if core.sha256(raw) != selection_entries[source_id]['screening_manifest_sha256']:
                raise core.DiagnosticAbort('preflight', 'screening_manifest_digest_mismatch')
            screen_raws[source_id] = core._load_json(raw)
        for source_id, source_path in source_paths.items():
            raw = capture('source_' + source_id, source_path, max_bytes)
            entry = selection_entries[source_id]
            screen = screen_raws[source_id]
            if (core.sha256(raw) != entry['source_sha256']
                    or not isinstance(screen, dict)
                    or screen.get('schema_version') != 'lfg.train-source-screening-claim.v1'
                    or screen.get('source_id_sha256') != source_id
                    or screen.get('source_sha256') != core.sha256(raw)
                    or screen.get('claim_status') != 'screened_train_only'):
                raise core.DiagnosticAbort('preflight', 'source_screening_claim_mismatch')
            safe_sources[source_id] = raw

        # Register captured model dependencies before executing the captured exporter.
        model_module = core._load_pinned_module(module_stack, 'owned_student',
                                                model_code_raw, model_code_path)
        cache_module = core._load_pinned_module(module_stack, 'owned_cached_decode',
                                                 cache_code_raw, cache_code_path)
        exporter_name = '_lfg_prefill_isolation_exporter_' + policy['pins']['exporter_code_sha256'][:12]
        exporter = core._load_pinned_module(module_stack, exporter_name,
                                             exporter_code_raw, exporter_code_path)
        report = _new_report(policy_raw, selection_raw, policy, len(safe_sources), lengths,
                             cpu_threads, max_bytes, max_work_items, max_prefix_tokens,
                             max_intermediate_bytes, max_seconds, variants)
        for source_id, source_raw in safe_sources.items():
            for length in lengths:
                report['samples'].append({'source_id_sha256': source_id,
                          'source_sha256': core.sha256(source_raw),
                          'screening_manifest_sha256': selection_entries[source_id]['screening_manifest_sha256'],
                          'screening_claim_status': 'operator_claim_not_independently_verified',
                          'prefix_length': length, 'status': 'not_tested',
                          'torch_full_vs_torch_incremental': None,
                          'ort_variants': [], 'layer_trace': None,
                          'earliest_numerical_difference': None,
                          'earliest_tolerance_divergence': None})
        _refresh_work_counts(report)
        report['torch_version'] = str(torch.__version__)
        report['onnxruntime_version'] = str(ort.__version__)
        report['onnx_version'] = str(onnx.__version__)
        report['numpy_version'] = str(np.__version__)
        report['safetensors_version'] = str(safetensors.__version__)
        report['python_version'] = platform.python_version()
        report['isolation_tool_sha256'] = actual_tool_sha256
        report['shared_core_sha256'] = _PINNED_CORE_DIGEST
        report['shared_core_expected_sha256'] = expected_core_sha256
        report['isolation_tool_expected_sha256'] = os.environ.get(
            'LFG_PREFILL_ISOLATION_EXPECTED_SHA256')
        report['launcher_digest_match'] = (
            report['isolation_tool_expected_sha256'] == report['isolation_tool_sha256'])
        exporter.create_private_output(output)
        output_dir = output
        if not core._atomic_report(output_dir, report):
            raise core.DiagnosticAbort('report', 'report_size_limit')

        architecture = manifest['architecture']
        model = model_module.OwnedByteStudent(architecture).cpu().eval()
        model.load_state_dict(load_safetensors(weights_raw))
        torch.set_num_threads(cpu_threads)
        layout = exporter.cache_layout(architecture)
        graph = exporter.PrefillGraph(model, layout).eval()
        options = {}
        for variant in variants:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = cpu_threads
            session_options.inter_op_num_threads = 1
            level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if variant == 'current_default'
                     else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
            session_options.graph_optimization_level = level
            options[variant] = session_options
        sessions = {variant: ort.InferenceSession(
            prefill_raw, sess_options=options[variant], providers=['CPUExecutionProvider'])
                    for variant in variants}
        # The exporter's pinned validator checks the retained prefill contract; decode
        # structure is checked by the existing shared helper as well.
        decode_session = ort.InferenceSession(
            decode_raw, sess_options=options[variants[0]], providers=['CPUExecutionProvider'])
        for session in sessions.values():
            core._validate_graph_contract(session, decode_session, layout, architecture)
        labels = ['logits', *[entry['name'] for entry in layout]]
        atol = config['absolute_tolerance']
        rtol = config['relative_tolerance']
        specs = _trace_specs(model)

        _refresh_work_counts(report)
        for sample in report['samples']:
            source_id = sample['source_id_sha256']
            source_raw = safe_sources[source_id]
            length = sample['prefix_length']
            if core._timed_out(started, max_seconds):
                sample.update(status='not_tested', failure_code='cooperative_time_limit')
                if report['failure_code'] is None:
                    report['failure_stage'], report['failure_code'] = 'execution', 'cooperative_time_limit'
            elif length > len(source_raw) or length > model.context:
                sample.update(status='invalid', failure_code='prefix_out_of_bounds')
            else:
                sample.update(status='running')
                _refresh_work_counts(report)
                hook_handles = []
                try:
                    if not core._atomic_report(output_dir, report):
                        raise core.DiagnosticAbort('report', 'report_size_limit')
                    token_ids = torch.tensor([list(source_raw[:length])], dtype=torch.long, device='cpu')
                    estimated_trace_bytes = _trace_byte_estimate(model, length)
                    trace_allowed = estimated_trace_bytes <= max_intermediate_bytes
                    trace_values = {'full': {}, 'incremental': {}}
                    active_path = ['full']
                    if trace_allowed:
                        from collections import defaultdict
                        trace_values = {'full': defaultdict(list), 'incremental': defaultdict(list)}
                        hook_handles = _attach_trace_hooks(specs, trace_values, active_path)
                    with torch.inference_mode():
                        full_values = list(graph(token_ids))
                        active_path[0] = 'incremental'
                        cache = cache_module.DecodeSession(model, 1)
                        incremental_logits = None
                        for position in range(length):
                            if core._timed_out(started, max_seconds):
                                raise core.DiagnosticAbort('execution', 'cooperative_time_limit')
                            incremental_logits = cache.step(token_ids[:, position])
                    for handle in hook_handles:
                        handle.remove()
                    hook_handles = []
                    full_numpy = _tensor_values(torch, full_values)
                    incremental_numpy = _incremental_values(torch, incremental_logits, cache, layout)
                    if trace_allowed:
                        sample['layer_trace'] = _trace_summary(trace_values, specs, length, atol, rtol)
                        sample['layer_trace']['estimated_capture_bytes'] = estimated_trace_bytes
                    else:
                        sample['layer_trace'] = {'status': 'not_tested',
                            'reason': 'intermediate_capture_budget_exceeded',
                            'estimated_capture_bytes': estimated_trace_bytes}
                    full_incremental = _pair_summary(full_numpy, incremental_numpy, labels, atol, rtol)
                    sample['torch_full_vs_torch_incremental'] = full_incremental
                    invalid_comparisons = []
                    tolerance_failure_count = _pair_original_failures(full_incremental)
                    for variant in variants:
                        if core._timed_out(started, max_seconds):
                            raise core.DiagnosticAbort('execution', 'cooperative_time_limit')
                        session = sessions[variant]
                        ort_values = session.run(None, {'tokens': token_ids.numpy()})
                        ort_numpy = [np.asarray(item) for item in ort_values]
                        three_way = three_way_comparisons(
                            ort_numpy, full_numpy, incremental_numpy, labels, atol, rtol)
                        row = {'name': variant, 'graph_optimization_level': VARIANT_LEVELS[variant],
                               'intra_op_num_threads': cpu_threads, 'inter_op_num_threads': 1,
                               'provider': 'CPUExecutionProvider',
                               'ort_vs_torch_full_prefill': three_way['onnx_vs_torch_full_prefill'],
                               'ort_vs_torch_incremental': three_way['onnx_vs_torch_incremental']}
                        sample['ort_variants'].append(row)
                        invalid_comparisons.extend((_pair_invalid(row['ort_vs_torch_full_prefill']),
                                                    _pair_invalid(row['ort_vs_torch_incremental'])))
                        tolerance_failure_count += (
                            _pair_original_failures(row['ort_vs_torch_full_prefill'])
                            + _pair_original_failures(row['ort_vs_torch_incremental']))
                    trace = sample.get('layer_trace')
                    if trace and trace.get('status') == 'measured':
                        sample['earliest_numerical_difference'] = trace.get('earliest_numerical_difference')
                        sample['earliest_tolerance_divergence'] = trace.get('earliest_tolerance_divergence')
                        tolerance_failure_count += sum(
                            int(row['full_vs_incremental'].get('failed_original_count', 0))
                            for row in trace.get('operations', []))
                    elif trace and trace.get('status') not in ('not_tested', 'measured'):
                        invalid_comparisons.append(True)
                    invalid_comparisons.append(_pair_invalid(full_incremental))
                    sample['original_tolerance_failure_count'] = tolerance_failure_count
                    sample['original_tolerance_status'] = (
                        'invalid' if any(invalid_comparisons) else
                        'failed' if tolerance_failure_count else 'within_tolerance')
                    sample['status'] = 'invalid' if any(invalid_comparisons) else 'measured'
                    if any(invalid_comparisons):
                        sample['failure_code'] = 'invalid_or_nonfinite_comparison'
                except core.DiagnosticAbort as exc:
                    sample.update(status='incomplete', failure_code=exc.code)
                except Exception:
                    sample.update(status='crash', failure_code='sanitized_execution_exception')
                finally:
                    for handle in hook_handles:
                        handle.remove()
            _refresh_work_counts(report)
            if sample['status'] in ('invalid', 'incomplete', 'crash') and report['failure_code'] is None:
                report['failure_stage'] = 'sample'
                report['failure_code'] = sample.get('failure_code', sample['status'])
            if not core._atomic_report(output_dir, report):
                raise core.DiagnosticAbort('report', 'report_size_limit')
            if core._timed_out(started, max_seconds):
                report['failure_stage'], report['failure_code'] = 'execution', 'cooperative_time_limit'
                break
            if report.get('failure_code') == 'cooperative_time_limit':
                break

        mutation_count = 0
        for _label, (path, original, maximum) in raw_inputs.items():
            if _label in ('isolation_tool', 'shared_core'):
                unchanged = _captured_bytes_unchanged(path, original, maximum)
            else:
                unchanged = core._safe_digest(path, maximum) == core.sha256(original)
            if not unchanged:
                mutation_count += 1
        report['input_mutation_count'] = mutation_count
        report['input_mutation_detected'] = mutation_count > 0
        if mutation_count:
            report['failure_stage'], report['failure_code'] = 'postcondition', 'input_mutation_detected'
    except StartupPinError as exc:
        report['failure_stage'], report['failure_code'] = 'launcher_preflight', exc.code
    except CorePinError as exc:
        report['failure_stage'], report['failure_code'] = 'core_preflight', exc.code
    except core.DiagnosticAbort as exc:
        report['failure_stage'], report['failure_code'] = exc.stage, exc.code
    except Exception:
        report['failure_stage'], report['failure_code'] = 'setup', 'sanitized_setup_exception'
    finally:
        module_stack.close()

    if output_dir is None:
        return {'status': 'failed', 'report_written': False,
                'failure_stage': report.get('failure_stage') or 'output_preflight',
                'failure_code': report.get('failure_code') or 'private_output_unavailable'}
    report['elapsed_seconds'] = time.monotonic() - started
    report['cpu_seconds'] = time.process_time() - cpu_started
    report['measurement_scope'] = 'CPU prefill isolation; includes preflight, session/model setup, hooks, and comparisons'
    complete = (report['requested_work_items'] == report['completed_work_items']
                and report['failure_code'] is None and not report['input_mutation_detected'])
    report['status'] = 'diagnostic_complete' if complete else 'incomplete'
    _refresh_work_counts(report)
    try:
        report_written = core._atomic_report(output_dir, report)
    except Exception:
        report_written = False
    if not report_written:
        report['status'] = 'failed'
        report['failure_stage'] = 'report'
        report['failure_code'] = 'final_report_write_failed'
    return {'status': report['status'], 'report_written': report_written,
            'completed_work_items': report['completed_work_items'],
            'not_tested_work_items': report['not_tested_work_items'],
            'failure_stage': report['failure_stage'], 'failure_code': report['failure_code'],
            'original_export_status': 'failed'}


def main(argv=None) -> int:
    # The launcher must hash the raw script and execute those exact captured bytes.
    # Rechecking here detects a stale/mismatched launcher pin before any input path is opened.
    expected_tool_sha256 = os.environ.get('LFG_PREFILL_ISOLATION_EXPECTED_SHA256')
    expected_core_sha256 = os.environ.get('LFG_PREFILL_CORE_EXPECTED_SHA256')
    failure = None
    if not _valid_sha256(expected_tool_sha256):
        failure = ('launcher_preflight', 'tool_digest_pin_missing_or_invalid')
    try:
        if failure is None:
            tool_raw = Path(__file__).read_bytes()
    except OSError:
        failure = ('launcher_preflight', 'tool_file_unavailable')
    if failure is None and expected_tool_sha256 != hashlib.sha256(tool_raw).hexdigest():
        failure = ('launcher_preflight', 'external_tool_digest_mismatch')
    if failure is None:
        try:
            _load_pinned_core(expected_core_sha256)
        except CorePinError as exc:
            failure = ('core_preflight', exc.code)
        except Exception:
            failure = ('core_preflight', 'core_setup_failed')
    if failure:
        print(json.dumps({'status': 'failed', 'report_written': False,
                          'failure_stage': failure[0], 'failure_code': failure[1]},
                         separators=(',', ':'), allow_nan=False))
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'export-config', 'graphs', 'policy',
                 'selection-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source', action='append', default=[], metavar='SOURCE_ID_SHA256=PATH')
    parser.add_argument('--screening-manifest', action='append', default=[],
                        metavar='SOURCE_ID_SHA256=PATH')
    parser.add_argument('--length', action='append', type=int, required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--max-bytes', type=int, required=True)
    parser.add_argument('--max-work-items', type=int, required=True)
    parser.add_argument('--max-prefix-tokens', type=int, required=True)
    parser.add_argument('--max-intermediate-bytes', type=int, required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--optimization-variant', choices=SESSION_VARIANTS,
                        action='append', required=True)
    args = parser.parse_args(argv)
    if (len(args.optimization_variant) != len(SESSION_VARIANTS)
            or set(args.optimization_variant) != set(SESSION_VARIANTS)):
        parser.error('specify each optimization variant exactly once')

    def parse_map(items, label):
        result = {}
        for item in items:
            identity, separator, path = item.partition('=')
            if not separator or not core._digest(identity) or identity in result:
                parser.error(f'--{label} requires unique SHA256=PATH entries')
            result[identity] = Path(path)
        return result

    result = diagnose(args.checkpoint, args.export_config, args.graphs, args.policy,
        args.selection_manifest, parse_map(args.source, 'source'),
        parse_map(args.screening_manifest, 'screening-manifest'), args.output,
        lengths=args.length, cpu_threads=args.threads, max_bytes=args.max_bytes,
        max_work_items=args.max_work_items, max_prefix_tokens=args.max_prefix_tokens,
        max_intermediate_bytes=args.max_intermediate_bytes, max_seconds=args.max_seconds,
        optimization_variants=tuple(args.optimization_variant))
    print(json.dumps(result, separators=(',', ':'), allow_nan=False))
    return 0 if result['status'] == 'diagnostic_complete' and result['report_written'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
