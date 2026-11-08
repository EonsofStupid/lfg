"""Read-only numerical diagnostics for an exact, already-exported ONNX pair.

This is a CPU diagnostic only. It does not export, promote, deploy, or assess
task quality. Reports contain bounded aggregates and opaque digests, never input
bytes, logits, cache values, tokens, generated text, paths, or exception text.
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
import sys
import time
import types

MAX_POLICY_BYTES = 1 << 20
MAX_REPORT_BYTES = 8 << 20
MAX_WORK_ITEMS = 64
MAX_PREFIX_TOKENS = 100_000
MAX_DECODE_TRANSITIONS = 512
MAX_DECODE_PER_SAMPLE = 64
ORIGINAL_ATOL = 1e-5
ORIGINAL_RTOL = 1e-5
ORT_GRAPH_OPTIMIZATION_LEVELS = (
    'ORT_DISABLE_ALL', 'ORT_ENABLE_BASIC', 'ORT_ENABLE_EXTENDED', 'ORT_ENABLE_ALL')
PIN_NAMES = (
    'checkpoint_manifest_sha256', 'weights_sha256', 'model_code_sha256',
    'cached_reference_code_sha256', 'exporter_code_sha256',
    'export_config_sha256', 'prefill_onnx_sha256', 'decode_onnx_sha256',
    'selection_manifest_sha256',
)


class DiagnosticAbort(Exception):
    """Internal typed abort; `code` is a fixed safe identifier."""
    def __init__(self, stage: str, code: str):
        super().__init__(code)
        self.stage = stage
        self.code = code


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_bounded(path: Path, maximum: int) -> bytes:
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise DiagnosticAbort('preflight', 'input_byte_limit')
    return raw


def _digest(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _create_ort_session_options(ort, cpu_threads: int, config: dict):
    """Build diagnostic options locally so old policy-pinned exporters still work."""
    if 'graph_optimization_level' not in config:
        requested = 'unspecified'
    else:
        requested = config['graph_optimization_level']
        if not isinstance(requested, str) or requested not in ORT_GRAPH_OPTIMIZATION_LEVELS:
            raise ValueError('invalid ONNX Runtime graph optimization level')
    options = ort.SessionOptions()
    options.intra_op_num_threads = cpu_threads
    options.inter_op_num_threads = 1
    if requested != 'unspecified':
        level_api = getattr(ort, 'GraphOptimizationLevel', None)
        level = getattr(level_api, requested, None) if level_api is not None else None
        if level is None:
            raise ValueError('configured ONNX Runtime graph optimization level is unavailable')
        options.graph_optimization_level = level
    return options, requested


def _reported_ort_graph_level(ort, value) -> str:
    if isinstance(value, str) and value in ORT_GRAPH_OPTIMIZATION_LEVELS:
        return value
    level_api = getattr(ort, 'GraphOptimizationLevel', None)
    if level_api is not None:
        for name in ORT_GRAPH_OPTIMIZATION_LEVELS:
            expected = getattr(level_api, name, None)
            if expected is not None:
                try:
                    if value == expected or int(value) == int(expected):
                        return name
                except (TypeError, ValueError, OverflowError):
                    pass
    return 'unknown'


def _ort_session_provenance(ort, sessions: dict, requested: str,
                            cpu_threads: int, scope: str) -> dict:
    """Record read-back session settings without depending on exporter version."""
    reported = []
    for name, session in sessions.items():
        effective_level = 'unknown'
        intra_threads = 'unknown'
        inter_threads = 'unknown'
        providers = 'unknown'
        try:
            options = session.get_session_options()
            effective_level = _reported_ort_graph_level(
                ort, getattr(options, 'graph_optimization_level', None))
            intra_threads = getattr(options, 'intra_op_num_threads', 'unknown')
            inter_threads = getattr(options, 'inter_op_num_threads', 'unknown')
        except Exception:
            pass
        try:
            observed_providers = session.get_providers()
            if isinstance(observed_providers, (list, tuple)) and observed_providers:
                providers = list(observed_providers)
        except Exception:
            pass
        reported.append({'name': name,
                         'effective_graph_optimization_level_reported': effective_level,
                         'intra_op_num_threads_reported': intra_threads,
                         'inter_op_num_threads_reported': inter_threads,
                         'providers_reported': providers})
    return {'backend': 'onnxruntime',
            'scope': scope,
            'onnxruntime_version': str(getattr(ort, '__version__', 'unknown')),
            'requested_graph_optimization_level': requested,
            'cpu_threads_requested': cpu_threads,
            'inter_op_num_threads_requested': 1,
            'sessions': reported}


def _reject_json_constant(_value):
    raise ValueError('nonstandard_json_constant')


def _load_json(raw: bytes):
    return json.loads(raw, parse_constant=_reject_json_constant)


def compare_arrays(actual, reference, atol: float, rtol: float,
                   exploratory_tolerance: tuple[float, float] | None = None) -> dict:
    """Summarize one tensor pair; tolerance matches assert_allclose on dtype."""
    import numpy as np

    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        return {'status': 'invalid', 'reason': 'shape_or_dtype_mismatch',
                'actual_shape': list(actual.shape), 'reference_shape': list(reference.shape),
                'actual_dtype': str(actual.dtype), 'reference_dtype': str(reference.dtype)}
    count = int(actual.size)
    base = {'status': 'measured', 'elements': count,
            'failed_original_count': 0, 'failed_original_denominator': count,
            'max_absolute_error': None,
            'max_relative_error_nonzero_reference': None, 'zero_reference_count': 0,
            'nonfinite_count': 0}
    if count == 0:
        base.update(status='not_tested', reason='empty_tensor', failed_original_denominator=0)
        return base

    # numpy.testing.assert_allclose uses same-kind elementwise arithmetic and
    # defaults equal_nan=True. Keep these operands in their incoming dtype for
    # the original pass/fail count; float64 is used only for summaries.
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        close = np.isclose(actual, reference, atol=atol, rtol=rtol, equal_nan=True)
        finite_mask = np.isfinite(actual) & np.isfinite(reference)
    difference = np.full(actual.shape, np.nan, dtype=np.float64)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        difference[finite_mask] = np.abs(actual[finite_mask].astype(np.float64)
                                          - reference[finite_mask].astype(np.float64))
    ref_abs = np.abs(reference.astype(np.float64))
    relative_mask = (ref_abs > 0) & finite_mask & np.isfinite(difference)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        relative = difference[relative_mask] / ref_abs[relative_mask]
    descriptive_finite = finite_mask & np.isfinite(difference)
    difference_unrepresentable = bool(np.any(finite_mask & ~np.isfinite(difference)))
    relative_unrepresentable = bool(np.any(~np.isfinite(relative)))
    base.update(failed_original_count=int(count - np.count_nonzero(close)),
                max_absolute_error=(float(np.max(difference[descriptive_finite]))
                                    if np.any(descriptive_finite) and not difference_unrepresentable else None),
                max_relative_error_nonzero_reference=(float(np.max(relative))
                    if relative.size and not difference_unrepresentable and not relative_unrepresentable else None),
                zero_reference_count=int(np.count_nonzero((ref_abs == 0) & finite_mask)),
                nonfinite_count=int(count - np.count_nonzero(finite_mask)))
    if difference_unrepresentable:
        base['descriptive_status'] = 'unrepresentable_finite_difference'
    elif relative_unrepresentable:
        base['descriptive_status'] = 'unrepresentable_relative_error'
    if base['nonfinite_count']:
        base.update(status='nonfinite', reason='nonfinite_tensor')
    if exploratory_tolerance is not None:
        exp_at, exp_rt = exploratory_tolerance
        with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
            exp_close = np.isclose(actual, reference, atol=exp_at, rtol=exp_rt, equal_nan=True)
        base['exploratory_only'] = {
            'absolute_tolerance': exp_at, 'relative_tolerance': exp_rt,
            'failed_count': int(count - np.count_nonzero(exp_close)),
            'denominator': count, 'status': 'exploratory_only'}
    return base


def greedy_summary(actual, reference) -> dict:
    """Compare one-batch greedy decisions and top-two margins without tokens."""
    import numpy as np

    if (actual.shape != reference.shape or actual.dtype != reference.dtype
            or actual.ndim != 2 or actual.shape[0] != 1 or actual.shape[1] < 2):
        return {'status': 'invalid', 'reason': 'greedy_shape_or_dtype_mismatch',
                'agreement': None, 'reference_margin': None, 'actual_margin': None}
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return {'status': 'nonfinite', 'reason': 'nonfinite_logits', 'agreement': None,
                'reference_margin': None, 'actual_margin': None}

    def margin(logits):
        top = np.partition(logits[0].astype(np.float64), -2)[-2:]
        with np.errstate(invalid='ignore', over='ignore'):
            value = float(top.max() - top.min())
        return value if math.isfinite(value) else None

    reference_margin = margin(reference)
    actual_margin = margin(actual)
    return {'status': 'measured', 'agreement': bool(np.argmax(actual[0]) == np.argmax(reference[0])),
            'reference_margin': reference_margin, 'actual_margin': actual_margin,
            'margin_status': ('unrepresentable' if reference_margin is None or actual_margin is None
                              else 'measured')}


def decision_stability(index: int, python_full_logits, onnx_free_logits,
                       python_cache_logits, full_python_history_logits,
                       full_onnx_history_logits) -> dict:
    """Keep independent trajectories distinct from each path's same-history check."""
    return {'decision_index': index,
            'python_full_vs_onnx_independent_trajectory': greedy_summary(
                python_full_logits, onnx_free_logits),
            'onnx_vs_full_same_onnx_history': greedy_summary(
                onnx_free_logits, full_onnx_history_logits),
            'python_cache_vs_full_same_python_history': greedy_summary(
                python_cache_logits, full_python_history_logits)}


def _first_divergence(current, index: int, agreement) -> int | None:
    if current is None and agreement is False:
        return index
    return current


def _logits_health(logits) -> str:
    import numpy as np
    if (not isinstance(logits, np.ndarray) or logits.shape != (1, 256)
            or logits.dtype != np.dtype('float32')):
        return 'invalid'
    if not np.isfinite(logits).all():
        return 'nonfinite'
    return 'measured'


def _metric_problem(metrics: dict) -> str | None:
    statuses = [item.get('status') for item in metrics.values() if isinstance(item, dict)]
    if 'nonfinite' in statuses:
        return 'nonfinite'
    if any(status in ('invalid', 'not_tested') for status in statuses):
        return 'invalid'
    return None


def _decision_problem(row: dict) -> str | None:
    statuses = [value.get('status') for value in row.values() if isinstance(value, dict)]
    if 'nonfinite' in statuses:
        return 'nonfinite'
    if any(status != 'measured' for status in statuses):
        return 'invalid'
    return None


def _record_tensors(bucket: dict, actuals, references, labels, atol, rtol, exploratory=None) -> None:
    if len(actuals) != len(references) or len(actuals) != len(labels):
        raise DiagnosticAbort('comparison', 'tensor_identity_count_mismatch')
    for label, actual, reference in zip(labels, actuals, references):
        bucket[label] = compare_arrays(actual, reference, atol, rtol, exploratory)


def _np(tensor):
    return tensor.detach().cpu().numpy()


def _cache_reference(cache, layout):
    values = []
    for item in layout:
        layer_state = cache._states[item['layer']]
        key = ('conv' if item['kind'] == 'depthwise-convolution-history'
               else 'keys' if item['kind'] == 'attention-key' else 'values')
        values.append(_np(layer_state[key]))
    return values


def _validate_policy(policy: dict) -> None:
    if (not isinstance(policy, dict)
            or policy.get('schema_version') != 'lfg.owned-cached-parity-diagnostic-policy.v2'
            or not isinstance(policy.get('pins'), dict)
            or any(not _digest(policy['pins'].get(name)) for name in PIN_NAMES)):
        raise DiagnosticAbort('preflight', 'invalid_policy_or_digest_pins')
    failure = policy.get('original_failure')
    if (not isinstance(failure, dict) or failure.get('status') != 'failed'
            or failure.get('absolute_tolerance') != ORIGINAL_ATOL
            or failure.get('relative_tolerance') != ORIGINAL_RTOL):
        raise DiagnosticAbort('preflight', 'original_failure_record_invalid')


def _validate_selection(selection: dict) -> dict:
    if (not isinstance(selection, dict)
            or selection.get('schema_version') != 'lfg.screened-train-selection.v1'
            or not isinstance(selection.get('entries'), list) or not selection['entries']
            or len(selection['entries']) > MAX_WORK_ITEMS):
        raise DiagnosticAbort('preflight', 'selection_manifest_invalid')
    entries = {}
    for item in selection['entries']:
        if (not isinstance(item, dict) or not _digest(item.get('source_id_sha256'))
                or not _digest(item.get('source_sha256'))
                or not _digest(item.get('screening_manifest_sha256'))
                or item.get('screening_status') != 'screened_train_only'
                or item['source_id_sha256'] in entries):
            raise DiagnosticAbort('preflight', 'selection_entry_invalid')
        entries[item['source_id_sha256']] = item
    return entries


def _load_module_bytes(name: str, raw: bytes, filename: Path):
    module = types.ModuleType(name)
    module.__file__ = str(filename)
    module.__package__ = ''
    sys.modules[name] = module
    exec(compile(raw, str(filename), 'exec'), module.__dict__)
    return module


_MISSING_MODULE = object()


def _restore_module_binding(name: str, previous) -> None:
    if previous is _MISSING_MODULE:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


def _load_pinned_module(stack: ExitStack, name: str, raw: bytes, filename: Path):
    previous = sys.modules.get(name, _MISSING_MODULE)
    stack.callback(_restore_module_binding, name, previous)
    return _load_module_bytes(name, raw, filename)


def _shape_matches(observed, expected) -> bool:
    if len(observed) != len(expected):
        return False
    for actual, wanted in zip(observed, expected):
        if wanted is None:
            if actual is not None and not isinstance(actual, str):
                return False
        elif actual != wanted:
            return False
    return True


def _validate_graph_contract(prefill, decode, layout, architecture) -> None:
    width = architecture['width']
    heads = architecture['heads']
    head_dim = width // heads
    expected_prefill_inputs = [('tokens', 'tensor(int64)', [None, None])]
    expected_prefill_outputs = [('logits', 'tensor(float)', [None, 256])]
    expected_decode_inputs = [('token', 'tensor(int64)', [None]), ('position', 'tensor(int64)', [])]
    expected_decode_outputs = [('logits', 'tensor(float)', [None, 256])]
    for item in layout:
        if item['kind'] == 'depthwise-convolution-history':
            # TorchScript ONNX export leaves Slice-produced prefill cache
            # dimensions symbolic; actual values are checked against the
            # pinned Python cache tensor at runtime. Decode cache inputs/outputs
            # have fixed convolution dimensions in the exporter contract.
            shape = [None, None, None]
            decode_shape = [None, width, architecture['conv_kernel'] - 1]
        else:
            # Tracer leaves the static head and head-width dimensions symbolic
            # in prefill outputs; the recurrent graph has fixed dimensions.
            shape = [None, None, 'sequence', None]
            decode_shape = [None, heads, 'past_length', head_dim]
        expected_prefill_outputs.append((item['name'], 'tensor(float)', shape))
        expected_decode_inputs.append((item['name'], 'tensor(float)', decode_shape))
        output_shape = ([None, heads, 'present_length', head_dim]
                        if item['kind'].startswith('attention-') else decode_shape)
        expected_decode_outputs.append((item['name'] + '_out', 'tensor(float)', output_shape))

    def validate(observed, expected):
        if len(observed) != len(expected):
            return False
        for actual, (name, dtype, shape) in zip(observed, expected):
            if actual.name != name or actual.type != dtype or not _shape_matches(actual.shape, shape):
                return False
        return True

    for valid, code in (
        (validate(prefill.get_inputs(), expected_prefill_inputs), 'prefill_input_contract'),
        (validate(prefill.get_outputs(), expected_prefill_outputs), 'prefill_output_contract'),
        (validate(decode.get_inputs(), expected_decode_inputs), 'decode_input_contract'),
        (validate(decode.get_outputs(), expected_decode_outputs), 'decode_output_contract')):
        if not valid:
            raise DiagnosticAbort('graph_validation', code)


def _atomic_report(directory: Path, report: dict) -> bool:
    encoded = (json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                          allow_nan=False) + '\n').encode()
    if len(encoded) > MAX_REPORT_BYTES:
        return False
    temporary = directory / f'.diagnostic-{os.getpid()}-{time.monotonic_ns()}.tmp'
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / 'diagnostic.json')
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return True


def _compact_report(report: dict, code: str = 'report_size_limit') -> dict:
    compact = {key: value for key, value in report.items() if key not in
               ('samples', 'failure_stage', 'failure_code', 'status')}
    compact['samples'] = [{key: sample[key] for key in (
        'source_id_sha256', 'source_sha256', 'screening_manifest_sha256',
        'prefix_length', 'decode_transitions_requested', 'decode_transitions_planned',
        'decode_transitions_completed', 'greedy_decisions_completed', 'status',
        'failure_code', 'first_independent_trajectory_divergence') if key in sample}
        for sample in report.get('samples', [])]
    compact['status'] = 'incomplete'
    compact['failure_stage'] = 'report'
    compact['failure_code'] = code
    compact['report_compacted'] = True
    return compact


def _failure_summary(policy: dict) -> dict:
    failure = policy['original_failure']
    return {'status': 'operator_reported_failed', 'absolute_tolerance': ORIGINAL_ATOL,
            'relative_tolerance': ORIGINAL_RTOL,
            'claim_sha256': sha256(json.dumps(failure, sort_keys=True, separators=(',', ':'),
                                              allow_nan=False).encode())}


def _timed_out(start: float, limit: float) -> bool:
    return time.monotonic() - start >= limit


def _safe_digest(path: Path, maximum: int) -> str | None:
    try:
        return sha256(read_bounded(path, maximum))
    except Exception:
        return None


def diagnose(checkpoint: Path, export_config: Path, graphs: Path, policy_path: Path,
             selection_manifest_path: Path, source_paths: dict[str, Path],
             screening_manifest_paths: dict[str, Path], output: Path, *,
             lengths: list[int], decode_steps: int, cpu_threads: int,
             max_bytes: int, max_work_items: int, max_prefix_tokens: int,
             max_seconds: float, exploratory_tolerance: tuple[float, float] | None = None) -> dict:
    """Run only pinned in-memory bytes, writing a bounded private partial report."""
    started = time.monotonic()
    cpu_started = time.process_time()
    output_dir = None
    report = {'schema_version': 'lfg.owned-cached-parity-diagnostic-report.v2',
              'status': 'running', 'original_export_status': 'failed',
              'assessment_quality_evaluated': False, 'promotion_eligible': False,
              'samples': [], 'requested_work_items': None, 'completed_work_items': 0,
              'not_tested_work_items': 0, 'failure_stage': None, 'failure_code': None,
              'input_mutation_detected': False,
              'execution_control': {'timer_kind': 'cooperative_in_process',
                  'hard_wall_time_enforced': False, 'hard_cpu_or_memory_limit_enforced': False,
                  'host_supervisor_verified_by_runner': False,
                  'real_input_guard_enforced_by_runner': False,
                  'owner_authorization_verified_by_runner': False,
                  'screening_claim_independently_verified_by_runner': False,
                  'external_prerequisite': 'fresh_scoped_handoff plus independently verified authorization, screening, and hard host limits'}}
    raw_inputs: dict[str, tuple[Path, bytes, int]] = {}
    safe_sources: dict[str, bytes] = {}
    policy = None
    module_stack = ExitStack()
    report_error = None
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
        from safetensors.torch import load as load_safetensors

        policy_raw = read_bounded(policy_path, MAX_POLICY_BYTES)
        raw_inputs['policy'] = (policy_path, policy_raw, MAX_POLICY_BYTES)
        policy = _load_json(policy_raw)
        _validate_policy(policy)
        if (not lengths or any(type(value) is not int or value < 1 for value in lengths)
                or len(set(lengths)) != len(lengths) or type(decode_steps) is not int
                or decode_steps < 1 or decode_steps > MAX_DECODE_PER_SAMPLE
                or type(cpu_threads) is not int or cpu_threads < 1
                or type(max_bytes) is not int or max_bytes < 1
                or type(max_work_items) is not int or not 1 <= max_work_items <= MAX_WORK_ITEMS
                or type(max_prefix_tokens) is not int or not 1 <= max_prefix_tokens <= MAX_PREFIX_TOKENS
                or not _finite_number(max_seconds) or max_seconds <= 0):
            raise DiagnosticAbort('preflight', 'execution_bounds_invalid')
        if exploratory_tolerance is not None and (len(exploratory_tolerance) != 2
                or any(not _finite_number(value) or value < 0 for value in exploratory_tolerance)):
            raise DiagnosticAbort('preflight', 'exploratory_tolerance_invalid')
        if set(source_paths) != set(screening_manifest_paths) or not source_paths:
            raise DiagnosticAbort('preflight', 'source_screening_identity_mismatch')
        if len(source_paths) * len(lengths) > max_work_items:
            raise DiagnosticAbort('preflight', 'sample_work_limit')

        def capture(label: str, path: Path, limit: int, pin_name: str | None = None) -> bytes:
            raw = read_bounded(path, limit)
            raw_inputs[label] = (path, raw, limit)
            if pin_name and sha256(raw) != policy['pins'][pin_name]:
                raise DiagnosticAbort('preflight', 'pinned_input_mismatch')
            return raw

        selection_raw = capture('selection', selection_manifest_path, MAX_POLICY_BYTES,
                                'selection_manifest_sha256')
        selection = _load_json(selection_raw)
        selection_entries = _validate_selection(selection)
        if set(source_paths) != set(selection_entries):
            raise DiagnosticAbort('preflight', 'source_selection_identity_mismatch')

        manifest_path = checkpoint / 'manifest.json'
        weights_path = checkpoint / 'weights.safetensors'
        model_code_path = Path(__file__).with_name('owned_student.py')
        cache_code_path = Path(__file__).with_name('owned_cached_decode.py')
        exporter_code_path = Path(__file__).with_name('export_owned_cached_onnx.py')
        manifest_raw = capture('checkpoint_manifest', manifest_path, max_bytes,
                               'checkpoint_manifest_sha256')
        weights_raw = capture('weights', weights_path, max_bytes, 'weights_sha256')
        model_code_raw = capture('model_code', model_code_path, max_bytes, 'model_code_sha256')
        cache_code_raw = capture('cached_reference_code', cache_code_path, max_bytes,
                                 'cached_reference_code_sha256')
        exporter_code_raw = capture('exporter_code', exporter_code_path, max_bytes,
                                    'exporter_code_sha256')
        config_raw = capture('export_config', export_config, MAX_POLICY_BYTES, 'export_config_sha256')
        prefill_raw = capture('prefill_graph', graphs / 'prefill.onnx', max_bytes, 'prefill_onnx_sha256')
        decode_raw = capture('decode_graph', graphs / 'decode.onnx', max_bytes, 'decode_onnx_sha256')
        screen_raws = {}
        for source_id, screen_path in screening_manifest_paths.items():
            screen_raws[source_id] = capture('screening_' + source_id, screen_path,
                                             MAX_POLICY_BYTES)
            if sha256(screen_raws[source_id]) != selection_entries[source_id]['screening_manifest_sha256']:
                raise DiagnosticAbort('preflight', 'screening_manifest_digest_mismatch')
        for source_id, source_path in source_paths.items():
            raw = capture('source_' + source_id, source_path, max_bytes)
            if sha256(raw) != selection_entries[source_id]['source_sha256']:
                raise DiagnosticAbort('preflight', 'selected_source_digest_mismatch')
            screen = _load_json(screen_raws[source_id])
            if (not isinstance(screen, dict)
                    or screen.get('schema_version') != 'lfg.train-source-screening-claim.v1'
                    or screen.get('source_id_sha256') != source_id
                    or screen.get('source_sha256') != sha256(raw)
                    or screen.get('claim_status') != 'screened_train_only'):
                raise DiagnosticAbort('preflight', 'screening_claim_mismatch')
            safe_sources[source_id] = raw

        manifest = _load_json(manifest_raw)
        config = _load_json(config_raw)
        if (not isinstance(config, dict)
                or config.get('schema_version') != 'lfg.owned-byte-onnx-export.v1'
                or config.get('absolute_tolerance') != ORIGINAL_ATOL
                or config.get('relative_tolerance') != ORIGINAL_RTOL):
            raise DiagnosticAbort('preflight', 'export_config_contract_mismatch')
        if (not isinstance(manifest, dict) or manifest.get('weights_sha256') != sha256(weights_raw)
                or manifest.get('code_sha256', {}).get('owned_student.py') != sha256(model_code_raw)):
            raise DiagnosticAbort('preflight', 'checkpoint_manifest_contract_mismatch')

        # Register the two dependencies from the captured bytes before any
        # pinned exporter code executes an import by module name. A preexisting
        # sys.modules entry is temporarily shadowed and restored at exit.
        model_module = _load_pinned_module(module_stack, 'owned_student',
                                           model_code_raw, model_code_path)
        cache_module = _load_pinned_module(module_stack, 'owned_cached_decode',
                                           cache_code_raw, cache_code_path)
        exporter_module_name = '_lfg_pinned_exporter_' + policy['pins']['exporter_code_sha256'][:12]
        exporter = _load_pinned_module(module_stack, exporter_module_name,
                                       exporter_code_raw, exporter_code_path)
        try:
            prefill_options, requested_ort_level = _create_ort_session_options(
                ort, cpu_threads, config)
            decode_options, _ = _create_ort_session_options(
                ort, cpu_threads, config)
        except ValueError:
            raise DiagnosticAbort('preflight', 'ort_graph_optimization_level_invalid_or_unavailable') from None
        # Establish private output only after all executable project code bytes
        # have been validated and registered. Failures from this point can be
        # represented by a bounded sanitized partial report.
        exporter.create_private_output(output)
        output_dir = output
        report.update({'policy_sha256': sha256(policy_raw),
                       'original_failure': _failure_summary(policy),
                       'pins': {name: policy['pins'][name] for name in PIN_NAMES},
                       'selection_manifest_sha256': sha256(selection_raw),
                       'screening_claim_status': 'operator_claim_not_independently_verified',
                       'external_host_control_prerequisite': 'hard CPU, memory, and wall limits require independent host supervision',
                       'authorization_prerequisite': 'real inputs require owner authorization and a fresh scoped handoff',
                       'source_count': len(safe_sources),
                       'cpu_threads_requested': cpu_threads,
                       'max_bytes_per_file': max_bytes,
                       'cooperative_time_limit_seconds': max_seconds,
                       'max_sample_work_items': max_work_items,
                       'max_prefix_work_tokens': max_prefix_tokens,
                       'elapsed_seconds_includes': 'preflight, model/session setup, and inference',
                       'peak_rss_bytes': None,
                       'platform': platform.platform(), 'processor': platform.processor(),
                       'torch_version': None, 'onnxruntime_version': None,
                       'onnxruntime_validation': _ort_session_provenance(
                           ort, {}, requested_ort_level, cpu_threads,
                           'cached-parity-diagnostic-validation-and-decode')})
        if not _atomic_report(output_dir, report):
            raise DiagnosticAbort('report', 'report_size_limit')

        architecture = manifest['architecture']
        model = model_module.OwnedByteStudent(architecture).eval()
        model.load_state_dict(load_safetensors(weights_raw))
        torch.set_num_threads(cpu_threads)
        layout = exporter.cache_layout(architecture)
        prefill = ort.InferenceSession(prefill_raw, sess_options=prefill_options,
                                       providers=['CPUExecutionProvider'])
        decode = ort.InferenceSession(decode_raw, sess_options=decode_options,
                                      providers=['CPUExecutionProvider'])
        report['onnxruntime_validation'] = _ort_session_provenance(
            ort, {'prefill': prefill, 'decode': decode}, requested_ort_level,
            cpu_threads, 'cached-parity-diagnostic-validation-and-decode')
        _validate_graph_contract(prefill, decode, layout, architecture)
        state_names = [item['name'] for item in layout]
        prefill_output_names = ['logits', *state_names]
        decode_output_names = ['logits', *[name + '_out' for name in state_names]]
        labels = ['logits', *state_names]
        atol, rtol = config['absolute_tolerance'], config['relative_tolerance']
        context = architecture['context']
        planned_prefix_work = len(safe_sources) * sum(
            4 * length + 3 * sum(length + step for step in
                range(1, min(decode_steps, max(0, context - length)) + 1))
            + 4 * min(decode_steps, max(0, context - length)) for length in lengths)
        if planned_prefix_work > max_prefix_tokens:
            raise DiagnosticAbort('preflight', 'prefix_work_limit')
        planned_transitions = sum(
            min(decode_steps, max(0, context - length))
            for _identity in safe_sources for length in lengths)
        if planned_transitions > MAX_DECODE_TRANSITIONS:
            raise DiagnosticAbort('preflight', 'decode_work_limit')
        report['requested_work_items'] = len(safe_sources) * len(lengths)
        report['requested_prefix_work_tokens'] = planned_prefix_work
        report['requested_decode_transitions'] = len(safe_sources) * len(lengths) * decode_steps
        report['planned_decode_transitions'] = planned_transitions
        report['torch_version'] = str(torch.__version__)
        report['onnxruntime_version'] = str(ort.__version__)
        if not _atomic_report(output_dir, report):
            raise DiagnosticAbort('report', 'report_size_limit')

        def deadline_check():
            if _timed_out(started, max_seconds):
                raise DiagnosticAbort('execution', 'cooperative_time_limit')

        for source_id, source_raw in safe_sources.items():
            for length in lengths:
                sample = {'source_id_sha256': source_id,
                          'source_sha256': sha256(source_raw),
                          'screening_manifest_sha256': selection_entries[source_id]['screening_manifest_sha256'],
                          'screening_claim_status': 'operator_claim_not_independently_verified',
                          'prefix_length': length, 'decode_transitions_requested': decode_steps,
                          'decode_transitions_planned': min(decode_steps, max(0, context - length)),
                          'decode_transitions_completed': 0,
                          'teacher_forced_transitions_attempted': 0,
                          'teacher_forced_transitions_completed': 0,
                          'free_trajectory_transitions_attempted': 0,
                          'free_trajectory_transitions_completed': 0,
                          'greedy_decisions_completed': 0,
                          'status': 'not_tested', 'prefill_comparisons': {},
                          'teacher_forced_decode': [], 'decision_stability': [],
                          'first_independent_trajectory_divergence': None}
                report['samples'].append(sample)
                try:
                    if length > len(source_raw) or length >= context:
                        sample.update(status='invalid', failure_code='prefix_out_of_bounds')
                    elif _timed_out(started, max_seconds):
                        sample.update(status='not_tested', failure_code='cooperative_time_limit')
                    else:
                        tokens = torch.tensor([list(source_raw[:length])], dtype=torch.long)
                        with torch.inference_mode():
                            deadline_check()
                            reference_logits = _np(model(tokens)[:, -1, :])
                        teacher_cache = cache_module.DecodeSession(model, 1)
                        free_cache = cache_module.DecodeSession(model, 1)
                        with torch.inference_mode():
                            for index in range(length):
                                deadline_check()
                                token = tokens[:, index]
                                teacher_logits = teacher_cache.step(token)
                                free_logits_tensor = free_cache.step(token)
                        py_logits = _np(teacher_logits)
                        with torch.inference_mode():
                            deadline_check()
                            onnx_prefill = prefill.run(prefill_output_names, {'tokens': tokens.numpy()})
                        onnx_free_state = list(onnx_prefill[1:])
                        onnx_teacher_state = list(onnx_prefill[1:])
                        python_cache_values = _cache_reference(teacher_cache, layout)
                        _record_tensors(sample['prefill_comparisons'],
                            [onnx_prefill[0], *onnx_prefill[1:]],
                            [reference_logits, *python_cache_values], labels, atol, rtol,
                            exploratory_tolerance)
                        _record_tensors(sample['prefill_comparisons'], [py_logits],
                            [reference_logits], ['python_cached_logits'], atol, rtol,
                            exploratory_tolerance)
                        _record_tensors(sample['prefill_comparisons'], [onnx_prefill[0]],
                            [py_logits], ['onnx_vs_python_cached_logits'], atol, rtol,
                            exploratory_tolerance)

                        full_reference_bytes = list(source_raw[:length])
                        full_python_bytes = list(source_raw[:length])
                        full_onnx_bytes = list(source_raw[:length])
                        reference_current = reference_logits
                        python_free_logits = _np(free_logits_tensor)
                        onnx_free_logits = onnx_prefill[0]
                        initial_decision = decision_stability(
                            0, reference_logits, onnx_free_logits, python_free_logits,
                            reference_logits, reference_logits)
                        sample['decision_stability'] = [initial_decision]
                        prefill_agreement = initial_decision[
                            'python_full_vs_onnx_independent_trajectory'].get('agreement')
                        sample['first_independent_trajectory_divergence'] = _first_divergence(
                            None, 0, prefill_agreement)
                        sample['trajectory_agreement_count'] = int(prefill_agreement is True)
                        sample['trajectory_mismatch_count'] = int(prefill_agreement is False)
                        prefill_problem = _metric_problem(sample['prefill_comparisons'])
                        prefill_decision_problem = _decision_problem(initial_decision)
                        if prefill_problem or prefill_decision_problem:
                            issue = prefill_problem or prefill_decision_problem
                            initial_decision.update(status=issue,
                                                    reason=f'{issue}_prefill')
                            sample.update(status='incomplete', failure_code=f'{issue}_prefill')
                        else:
                            sample['greedy_decisions_completed'] = 1
                        free_python_full = reference_logits
                        free_onnx_full = reference_logits
                        transition_limit = (sample['decode_transitions_planned']
                                            if not (prefill_problem or prefill_decision_problem) else 0)
                        for transition in range(transition_limit):
                            deadline_check()
                            health = [_logits_health(value) for value in
                                      (reference_current, free_python_full, onnx_free_logits,
                                       python_free_logits)]
                            if any(value != 'measured' for value in health):
                                issue = 'nonfinite' if 'nonfinite' in health else 'invalid'
                                sample.update(status='incomplete', failure_code=f'{issue}_step_logits')
                                break
                            reference_token = int(np.argmax(reference_current[0]))
                            python_token = int(np.argmax(free_python_full[0]))
                            onnx_token = int(np.argmax(onnx_free_logits[0]))
                            # Teacher-forced comparison follows the full-reference stream.
                            feed = {'token': np.asarray([reference_token], dtype=np.int64),
                                    'position': np.asarray(length + transition, dtype=np.int64)}
                            feed.update({name: value for name, value in
                                         zip(state_names, onnx_teacher_state)})
                            onnx_teacher = decode.run(decode_output_names, feed)
                            with torch.inference_mode():
                                python_teacher_logits = _np(teacher_cache.step(
                                    torch.tensor([reference_token], dtype=torch.long)))
                            full_reference_bytes.append(reference_token)
                            with torch.inference_mode():
                                deadline_check()
                                full_reference_next = _np(model(torch.tensor(
                                    [full_reference_bytes], dtype=torch.long))[:, -1, :])
                            teacher_stats = {}
                            next_teacher_state = list(onnx_teacher[1:])
                            _record_tensors(teacher_stats,
                                [onnx_teacher[0], *next_teacher_state],
                                [full_reference_next, *_cache_reference(teacher_cache, layout)],
                                labels, atol, rtol, exploratory_tolerance)
                            _record_tensors(teacher_stats, [python_teacher_logits],
                                [full_reference_next], ['python_cached_logits'], atol, rtol,
                                exploratory_tolerance)
                            _record_tensors(teacher_stats, [onnx_teacher[0]],
                                [python_teacher_logits], ['onnx_vs_python_cached_logits'],
                                atol, rtol, exploratory_tolerance)
                            teacher_problem = _metric_problem(teacher_stats)
                            sample['teacher_forced_decode'].append({
                                'transition': transition,
                                'status': teacher_problem or 'measured',
                                'tensor_comparisons': teacher_stats})
                            sample['teacher_forced_transitions_attempted'] += 1
                            if teacher_problem:
                                sample.update(status='incomplete',
                                    failure_code=f'{teacher_problem}_teacher_forced_step')
                                break
                            sample['teacher_forced_transitions_completed'] += 1
                            sample['decode_transitions_completed'] = min(
                                sample['teacher_forced_transitions_completed'],
                                sample['free_trajectory_transitions_completed'])
                            onnx_teacher_state = next_teacher_state
                            reference_current = full_reference_next

                            # Advance each free-running path with its own choice.
                            free_feed = {'token': np.asarray([onnx_token], dtype=np.int64),
                                         'position': np.asarray(length + transition, dtype=np.int64)}
                            free_feed.update({name: value for name, value in
                                              zip(state_names, onnx_free_state)})
                            sample['free_trajectory_transitions_attempted'] += 1
                            onnx_free = decode.run(decode_output_names, free_feed)
                            onnx_free_logits, onnx_free_state = onnx_free[0], list(onnx_free[1:])
                            with torch.inference_mode():
                                python_free_logits = _np(free_cache.step(
                                    torch.tensor([python_token], dtype=torch.long)))
                            full_python_bytes.append(python_token)
                            full_onnx_bytes.append(onnx_token)
                            with torch.inference_mode():
                                deadline_check()
                                free_python_full = _np(model(torch.tensor(
                                    [full_python_bytes], dtype=torch.long))[:, -1, :])
                                free_onnx_full = _np(model(torch.tensor(
                                    [full_onnx_bytes], dtype=torch.long))[:, -1, :])
                            step_decision = decision_stability(
                                transition + 1, free_python_full, onnx_free_logits,
                                python_free_logits, free_python_full, free_onnx_full)
                            agreement = step_decision['python_full_vs_onnx_independent_trajectory'].get('agreement')
                            sample['first_independent_trajectory_divergence'] = _first_divergence(
                                sample['first_independent_trajectory_divergence'], transition + 1, agreement)
                            if step_decision['python_full_vs_onnx_independent_trajectory'].get('status') == 'measured':
                                sample['trajectory_agreement_count'] += int(agreement is True)
                                sample['trajectory_mismatch_count'] += int(agreement is False)
                            sample['decision_stability'].append(step_decision)
                            if _decision_problem(step_decision):
                                issue = _decision_problem(step_decision)
                                step_decision.update(status=issue,
                                                     reason=f'{issue}_step_decision')
                                sample.update(status='incomplete', failure_code=f'{issue}_step_decision')
                                break
                            sample['free_trajectory_transitions_completed'] += 1
                            sample['decode_transitions_completed'] = min(
                                sample['teacher_forced_transitions_completed'],
                                sample['free_trajectory_transitions_completed'])
                            sample['greedy_decisions_completed'] += 1
                        sample['decode_transitions_completed'] = min(
                            sample['teacher_forced_transitions_completed'],
                            sample['free_trajectory_transitions_completed'])
                        context_truncated = decode_steps > sample['decode_transitions_planned']
                        completed_plan = (sample['teacher_forced_transitions_completed']
                                          == sample['decode_transitions_planned']
                                          and sample['free_trajectory_transitions_completed']
                                          == sample['decode_transitions_planned'])
                        sample['status'] = ('incomplete' if context_truncated or not completed_plan
                                            or sample.get('failure_code') else 'measured')
                        if context_truncated:
                            sample['failure_code'] = sample.get('failure_code', 'context_truncated')
                except DiagnosticAbort as exc:
                    sample.update(status='incomplete', failure_stage=exc.stage,
                                  failure_code=exc.code)
                except Exception:
                    sample.update(status='crash', failure_stage='sample',
                                  failure_code='sanitized_execution_exception')
                reason = sample.get('failure_code') or (
                    'context_truncated' if sample['decode_transitions_planned'] < decode_steps
                    else 'not_completed')
                existing_teacher = {row.get('transition') for row in sample['teacher_forced_decode']}
                for missing in range(decode_steps):
                    if missing not in existing_teacher:
                        sample['teacher_forced_decode'].append({'transition': missing,
                            'status': 'not_tested', 'reason': reason})
                existing_decisions = {row.get('decision_index') for row in sample['decision_stability']}
                for decision_index in range(decode_steps + 1):
                    if decision_index not in existing_decisions:
                        sample['decision_stability'].append({'decision_index': decision_index,
                            'status': 'not_tested', 'reason': reason})
                report['completed_work_items'] += int(sample['status'] == 'measured')
                report['not_tested_work_items'] = report['requested_work_items'] - report['completed_work_items']
                if sample['status'] in ('incomplete', 'crash', 'invalid') and report['failure_code'] is None:
                    report['failure_stage'] = sample.get('failure_stage', 'sample')
                    report['failure_code'] = sample.get('failure_code', sample['status'])
                if not _atomic_report(output_dir, report):
                    sample.update(status='incomplete', failure_code='report_size_limit')
                    report = _compact_report(report)
                    _atomic_report(output_dir, report)
                    break
                if _timed_out(started, max_seconds):
                    report['failure_stage'], report['failure_code'] = 'execution', 'cooperative_time_limit'
                    break
            if report.get('failure_code') == 'cooperative_time_limit' or report.get('failure_code') == 'report_size_limit':
                break

        # All operational bytes came from the captured buffers above. Recheck
        # path identities only to report mutations; a swap cannot change bytes
        # already passed to the model loader or ONNX Runtime.
        mutation_count = 0
        for _label, (path, original, limit) in raw_inputs.items():
            if _safe_digest(path, limit) != sha256(original):
                mutation_count += 1
        report['input_mutation_detected'] = mutation_count > 0
        report['input_mutation_count'] = mutation_count
        if mutation_count:
            report['failure_stage'], report['failure_code'] = 'postcondition', 'input_mutation_detected'
    except DiagnosticAbort as exc:
        report['failure_stage'], report['failure_code'] = exc.stage, exc.code
    except Exception:
        report['failure_stage'], report['failure_code'] = 'setup', 'sanitized_setup_exception'

    module_stack.close()

    if output_dir is None:
        return {'status': 'failed', 'report_written': False,
                'failure_stage': report['failure_stage'] or 'output_preflight',
                'failure_code': report['failure_code'] or 'private_output_unavailable',
                'original_export_status': 'failed'}
    report['elapsed_seconds'] = time.monotonic() - started
    report['cpu_seconds'] = time.process_time() - cpu_started
    report['measurement_scope'] = 'CPU diagnostic; elapsed and CPU include preflight/model/session setup and measured work'
    all_complete = (report.get('requested_work_items') == report.get('completed_work_items')
                    and not report['failure_code'] and not report['input_mutation_detected'])
    if all_complete:
        report['status'] = 'diagnostic_complete'
    elif report.get('failure_stage') in ('setup', 'preflight', 'graph_validation'):
        report['status'] = 'failed'
    else:
        report['status'] = 'incomplete'
    report['not_tested_work_items'] = max(0, (report.get('requested_work_items') or 0)
                                         - report.get('completed_work_items', 0))
    written = False
    try:
        written = _atomic_report(output_dir, report)
    except Exception:
        written = False
    if not written:
        report = _compact_report(report)
        try:
            written = _atomic_report(output_dir, report)
        except Exception:
            written = False
    return {'status': report['status'], 'report_written': written,
            'completed_work_items': report.get('completed_work_items', 0),
            'not_tested_work_items': report['not_tested_work_items'],
            'original_export_status': 'failed', 'failure_stage': report.get('failure_stage'),
            'failure_code': report.get('failure_code')}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'export-config', 'graphs', 'policy',
                 'selection-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source', action='append', default=[], metavar='SOURCE_ID_SHA256=PATH')
    parser.add_argument('--screening-manifest', action='append', default=[],
                        metavar='SOURCE_ID_SHA256=PATH')
    parser.add_argument('--length', action='append', type=int, required=True)
    parser.add_argument('--decode-steps', type=int, required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--max-bytes', type=int, required=True)
    parser.add_argument('--max-work-items', type=int, required=True)
    parser.add_argument('--max-prefix-tokens', type=int, required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--exploratory-atol', type=float)
    parser.add_argument('--exploratory-rtol', type=float)
    args = parser.parse_args(argv)

    def parse_map(items, label):
        result = {}
        for item in items:
            identity, separator, path = item.partition('=')
            if not separator or not _digest(identity) or identity in result:
                parser.error(f'--{label} requires unique SHA256=PATH entries')
            result[identity] = Path(path)
        return result

    sources = parse_map(args.source, 'source')
    screenings = parse_map(args.screening_manifest, 'screening-manifest')
    exploratory = None
    if args.exploratory_atol is not None or args.exploratory_rtol is not None:
        if args.exploratory_atol is None or args.exploratory_rtol is None:
            parser.error('both exploratory tolerances are required')
        exploratory = (args.exploratory_atol, args.exploratory_rtol)
    result = diagnose(args.checkpoint, args.export_config, args.graphs, args.policy,
        args.selection_manifest, sources, screenings, args.output,
        lengths=args.length, decode_steps=args.decode_steps, cpu_threads=args.threads,
        max_bytes=args.max_bytes, max_work_items=args.max_work_items,
        max_prefix_tokens=args.max_prefix_tokens, max_seconds=args.max_seconds,
        exploratory_tolerance=exploratory)
    print(json.dumps(result, separators=(',', ':'), allow_nan=False))
    return 0 if result['status'] == 'diagnostic_complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
