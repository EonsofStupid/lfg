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
import stat
import statistics
import subprocess
import sys
import time

class OutputSanitizationUnavailable(Exception):
    pass


_CLI_MODE = __name__ == '__main__'
_CLI_STATUS_STDOUT_FD = None
_CLI_STATUS_STDERR_FD = None
_CLI_OUTPUT_SINK = None
_CLI_OUTPUT_BOUNDARY_READY = False
_CLI_OUTPUT_BOUNDARY_FAILED = False


def _duplicate_status_descriptor(source_fd: int) -> int | None:
    """Duplicate a status stream above the standard descriptors before redirecting them."""
    temporary_descriptors = []
    try:
        descriptor = os.dup(source_fd)
        while descriptor <= 2:
            temporary_descriptors.append(descriptor)
            descriptor = os.dup(descriptor)
        return descriptor
    except OSError:
        return None
    finally:
        for descriptor in temporary_descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _start_cli_output_boundary() -> None:
    """Keep native fd 1/2 discarded until process exit; save status channels."""
    global _CLI_STATUS_STDOUT_FD, _CLI_STATUS_STDERR_FD
    global _CLI_OUTPUT_SINK, _CLI_OUTPUT_BOUNDARY_READY, _CLI_OUTPUT_BOUNDARY_FAILED
    _CLI_STATUS_STDOUT_FD = _duplicate_status_descriptor(1)
    _CLI_STATUS_STDERR_FD = _duplicate_status_descriptor(2)
    try:
        sink = open(os.devnull, 'w', encoding='utf-8')
    except OSError:
        sink = None
    _CLI_OUTPUT_SINK = sink
    redirect_ok = sink is not None
    if sink is not None:
        for target_fd in (1, 2):
            try:
                os.dup2(sink.fileno(), target_fd)
            except OSError:
                redirect_ok = False
                try:
                    os.close(target_fd)
                except OSError:
                    pass
        sys.stdout = sink
        sys.stderr = sink
    else:
        redirect_ok = False
        for target_fd in (1, 2):
            try:
                os.close(target_fd)
            except OSError:
                pass
    _CLI_OUTPUT_BOUNDARY_READY = bool(
        redirect_ok and _CLI_STATUS_STDOUT_FD is not None and _CLI_STATUS_STDERR_FD is not None)
    _CLI_OUTPUT_BOUNDARY_FAILED = not _CLI_OUTPUT_BOUNDARY_READY


def _emit_status(result: dict, *, to_stderr: bool = False) -> bool:
    """Write a bounded fixed CLI status through the saved caller channel."""
    try:
        payload = (json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    except Exception:
        payload = b'{"status":"status_serialization_failed","failure_detail_suppressed":true}\n'
    if len(payload) > 4096:
        payload = b'{"status":"status_size_limit","failure_detail_suppressed":true}\n'
    return _emit_payload(payload, to_stderr=to_stderr)


def _emit_payload(payload: bytes, *, to_stderr: bool = False) -> bool:
    """Deliver a bounded fixed payload and report whether every byte was written."""
    if not isinstance(payload, bytes) or len(payload) > 4096:
        return False
    if _CLI_MODE:
        descriptor = _CLI_STATUS_STDERR_FD if to_stderr else _CLI_STATUS_STDOUT_FD
        if descriptor is None:
            descriptor = _CLI_STATUS_STDOUT_FD if to_stderr else _CLI_STATUS_STDERR_FD
        if descriptor is None:
            return False
        written = 0
        try:
            while written < len(payload):
                try:
                    count = os.write(descriptor, payload[written:])
                except InterruptedError:
                    continue
                if count <= 0:
                    return False
                written += count
        except OSError:
            return False
        return True
    else:
        stream = sys.stderr if to_stderr else sys.stdout
        try:
            message = payload.decode('utf-8')
            count = stream.write(message)
            if count is not None and count != len(message):
                return False
            stream.flush()
            return True
        except Exception:
            return False


if _CLI_MODE:
    _start_cli_output_boundary()


_DEPENDENCY_IMPORT_FAILED = False
_STARTUP_OUTPUT_SANITIZATION_FAILED = _CLI_OUTPUT_BOUNDARY_FAILED
try:
    if _CLI_MODE and not _CLI_OUTPUT_BOUNDARY_READY:
        raise OutputSanitizationUnavailable()
    import torch
    from torch.nn import functional as F
    from safetensors.torch import load_file

    import owned_student
    from owned_cached_decode import DecodeSession
except OutputSanitizationUnavailable:
    _DEPENDENCY_IMPORT_FAILED = True
    _STARTUP_OUTPUT_SANITIZATION_FAILED = True
    torch = F = load_file = owned_student = DecodeSession = None
except Exception:
    # Keep module import and the CLI's sanitized failure boundary available if
    # an optional ML dependency is absent or fails during its own import.
    _DEPENDENCY_IMPORT_FAILED = True
    torch = F = load_file = owned_student = DecodeSession = None
_TorchModule = torch.nn.Module if not _DEPENDENCY_IMPORT_FAILED else object

MAX_CONFIG_BYTES = 1 << 20
MAX_FAILURE_REPORT_BYTES = 8192
FAILURE_REPORT_NAME = 'failure-report.json'
FAILURE_REPORT_DIRFD_SUPPORTED = bool(getattr(os, 'supports_dir_fd', None)) and all(
    function in os.supports_dir_fd for function in (os.open, os.stat))
FAILURE_REPORT_STAT_NOFOLLOW_SUPPORTED = (
    os.stat in getattr(os, 'supports_follow_symlinks', set()))
ORT_GRAPH_OPTIMIZATION_LEVELS = (
    'ORT_DISABLE_ALL', 'ORT_ENABLE_BASIC', 'ORT_ENABLE_EXTENDED', 'ORT_ENABLE_ALL')

FAILURE_PHASES = frozenset((
    'configured_length_prefill', 'full_source_prefill', 'autoregressive_decode'))
FAILURE_COMPARISONS = frozenset((
    'onnx_prefill_vs_full_prefix_logits', 'onnx_prefill_vs_incremental_logits',
    'onnx_prefill_vs_incremental_cache', 'onnx_decode_vs_full_prefix_logits',
    'onnx_decode_vs_incremental_logits', 'onnx_decode_vs_incremental_cache'))


class SanitizedArgumentError(Exception):
    pass


class SanitizedHelpRequested(Exception):
    def __init__(self, text: str):
        self.text = text


class SanitizedArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # Do not let argparse echo raw argument values or filesystem paths.
        raise SanitizedArgumentError()

    def print_help(self, file=None):
        # Do not let argparse write directly to the discarded process stream.
        raise SanitizedHelpRequested(self.format_help())


class ExporterDependenciesUnavailable(Exception):
    pass


class CachedParityFailure(AssertionError):
    """A strict numerical failure with a stable, non-sensitive message."""

    def __init__(self, report_written: bool, report_error_category: str,
                 context: dict, summary: dict | None):
        super().__init__('cached ONNX strict parity assertion failed')
        self.failure_report_written = report_written
        self.failure_report_error_category = report_error_category
        self.failure_context = context
        self.failure_summary = summary


def _safe_failure_context(context: dict) -> dict:
    """Keep only bounded, code-owned labels and integer coordinates."""
    if not isinstance(context, dict):
        return {'phase': 'unknown', 'comparison': 'unknown'}
    phase = context.get('phase')
    comparison = context.get('comparison')
    safe = {
        'phase': phase if isinstance(phase, str) and phase in FAILURE_PHASES else 'unknown',
        'comparison': comparison if isinstance(comparison, str) and comparison in FAILURE_COMPARISONS else 'unknown',
    }
    for key, minimum, maximum in (
            ('source_ordinal', 1, 4096), ('prefix_length', 1, 1 << 31),
            ('decode_step', 1, 1 << 31), ('state_layer', 0, 4096)):
        value = context.get(key)
        if type(value) is int and minimum <= value <= maximum:
            safe[key] = value
    state_name = context.get('state_name')
    if isinstance(state_name, str) and len(state_name) <= 16:
        state_parts = state_name.split('_')
        if (len(state_parts) == 2 and state_parts[0] in ('conv', 'key', 'value')
                and state_parts[1].isdecimal()):
            safe['state_name'] = state_name
    state_kind = context.get('state_kind')
    if (isinstance(state_kind, str)
            and state_kind in ('depthwise-convolution-history', 'attention-key', 'attention-value')):
        safe['state_kind'] = state_kind
    return safe


def _safe_array_summary(np, actual, desired, atol: float, rtol: float) -> dict:
    """Describe a failed pair without retaining values or coordinates.

    The original-dtype ``isclose`` pass matches assert_allclose's elementwise
    predicate for same-shape arrays. Summary arithmetic is descriptive only.
    """
    actual_array = np.asanyarray(actual)
    desired_array = np.asanyarray(desired)

    def shape_of(array):
        shape = tuple(int(dimension) for dimension in array.shape)
        return list(shape) if len(shape) <= 8 else None

    def dtype_of(array):
        dtype = np.dtype(array.dtype)
        if dtype.kind in 'biufc' and dtype.itemsize <= 16:
            return dtype.name
        return 'unsupported'

    actual_shape = shape_of(actual_array)
    desired_shape = shape_of(desired_array)
    summary = {
        'actual_shape': actual_shape,
        'desired_shape': desired_shape,
        'actual_shape_dimensions': int(actual_array.ndim),
        'desired_shape_dimensions': int(desired_array.ndim),
        'actual_dtype': dtype_of(actual_array),
        'desired_dtype': dtype_of(desired_array),
        'element_count': None,
        'failed_element_count': None,
        'failed_element_denominator': None,
        'nonfinite_element_count': None,
        'zero_reference_count': None,
        'max_absolute_error': None,
        'max_relative_error_nonzero_reference': None,
        'aggregate_status': 'unavailable',
        'aggregate_reason': None,
    }
    if actual_array.shape != desired_array.shape:
        summary['aggregate_reason'] = 'shape_mismatch_element_counts_unavailable'
        return summary
    summary['element_count'] = int(actual_array.size)
    summary['failed_element_denominator'] = int(actual_array.size)
    if actual_array.dtype.kind not in 'biufc' or desired_array.dtype.kind not in 'biufc':
        summary['aggregate_reason'] = 'unsupported_dtype'
        return summary

    try:
        # Keep these exact operands in their source dtype. Do not use the
        # descriptive float64 copies below to determine strict failures.
        with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
            close = np.isclose(actual_array, desired_array, atol=atol,
                               rtol=rtol, equal_nan=True)
            finite = np.isfinite(actual_array) & np.isfinite(desired_array)
        summary['failed_element_count'] = int(actual_array.size - np.count_nonzero(close))
        summary['nonfinite_element_count'] = int(actual_array.size - np.count_nonzero(finite))
        if desired_array.dtype.kind == 'c':
            desired_complex = desired_array.astype(np.complex128, copy=False)
            desired_real = desired_complex.real
            desired_imag = desired_complex.imag
            desired_zero = (desired_real == 0) & (desired_imag == 0)
            with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
                desired_magnitude = np.hypot(desired_real, desired_imag)
        else:
            desired_real = desired_array.astype(np.float64, copy=False)
            desired_imag = np.zeros_like(desired_real)
            desired_zero = desired_real == 0
            desired_magnitude = np.abs(desired_real)
        summary['zero_reference_count'] = int(np.count_nonzero(finite & desired_zero))

        if actual_array.dtype.kind in 'fc' and desired_array.dtype.kind in 'fc':
            use_complex = actual_array.dtype.kind == 'c' or desired_array.dtype.kind == 'c'
            summary_dtype = np.complex128 if use_complex else np.float64
            with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
                actual_summary = actual_array.astype(summary_dtype, copy=False)
                desired_summary = desired_array.astype(summary_dtype, copy=False)
                differences = np.abs(actual_summary - desired_summary)
                finite_differences = finite & np.isfinite(differences)
                unrepresentable = bool(np.any(finite & ~np.isfinite(differences)))
                if np.any(finite_differences) and not unrepresentable:
                    summary['max_absolute_error'] = float(np.max(differences[finite_differences]))
                nonzero_reference = finite & (desired_magnitude > 0) & np.isfinite(differences)
                relative = differences[nonzero_reference] / desired_magnitude[nonzero_reference]
                relative_unrepresentable = bool(np.any(~np.isfinite(relative)))
                if relative.size and not unrepresentable and not relative_unrepresentable:
                    summary['max_relative_error_nonzero_reference'] = float(np.max(relative))
            if use_complex:
                actual_complex = actual_array.astype(np.complex128, copy=False)
                actual_real, actual_imag = actual_complex.real, actual_complex.imag
                with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
                    scale = np.maximum(np.abs(desired_real), np.abs(desired_imag))
                    relative_mask = finite & ~desired_zero
                    scaled_desired_real = np.zeros_like(scale)
                    scaled_desired_imag = np.zeros_like(scale)
                    scaled_actual_real = np.zeros_like(scale)
                    scaled_actual_imag = np.zeros_like(scale)
                    np.divide(desired_real, scale, out=scaled_desired_real, where=relative_mask)
                    np.divide(desired_imag, scale, out=scaled_desired_imag, where=relative_mask)
                    np.divide(actual_real, scale, out=scaled_actual_real, where=relative_mask)
                    np.divide(actual_imag, scale, out=scaled_actual_imag, where=relative_mask)
                    desired_norm = np.hypot(scaled_desired_real, scaled_desired_imag)
                    relative_numerator = np.hypot(
                        scaled_actual_real - scaled_desired_real,
                        scaled_actual_imag - scaled_desired_imag)
                    relative = relative_numerator[relative_mask] / desired_norm[relative_mask]
                relative_unrepresentable = bool(np.any(~np.isfinite(relative)))
                if relative.size and not relative_unrepresentable:
                    summary['max_relative_error_nonzero_reference'] = float(np.max(relative))
                elif relative_unrepresentable:
                    summary['max_relative_error_nonzero_reference'] = None
            if unrepresentable:
                summary['aggregate_reason'] = 'finite_difference_unrepresentable'
            elif relative_unrepresentable:
                summary['aggregate_reason'] = 'relative_error_unrepresentable'
        else:
            summary['aggregate_reason'] = 'integer_difference_not_summarized'
        summary['aggregate_status'] = 'measured'
    except Exception:
        # Diagnostics are best-effort; the already-failed NumPy assertion is
        # authoritative and must never be replaced by a reporting exception.
        summary['aggregate_reason'] = 'safe_aggregation_unavailable'
    return summary


def _output_still_matches(output: Path, identity: tuple[int, int]) -> bool:
    try:
        current = os.stat(output, follow_symlinks=False)
        return (stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == identity)
    except OSError:
        return False


def _write_failure_report(output: Path, report: dict,
                          output_identity: tuple[int, int] | None = None) -> tuple[bool, str]:
    """Exclusively create/write through a descriptor bound to the verified output inode."""
    if (os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_DIRECTORY')
            or not FAILURE_REPORT_DIRFD_SUPPORTED or not FAILURE_REPORT_STAT_NOFOLLOW_SUPPORTED):
        return False, 'private_output_controls_unavailable'
    if (not isinstance(output_identity, tuple) or len(output_identity) != 2
            or any(type(value) is not int or value < 0 for value in output_identity)):
        return False, 'verified_output_identity_unavailable'
    try:
        directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return False, 'unsafe_private_directory'
    try:
        directory = os.fstat(directory_fd)
        if (not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.geteuid()
                or stat.S_IMODE(directory.st_mode) & 0o077
                or (directory.st_dev, directory.st_ino) != output_identity):
            os.close(directory_fd)
            return False, 'unsafe_private_directory'
        if not _output_still_matches(output, output_identity):
            os.close(directory_fd)
            return False, 'output_directory_changed'
    except OSError:
        os.close(directory_fd)
        return False, 'unsafe_private_directory'
    try:
        data = (json.dumps(report, sort_keys=True, separators=(',', ':'),
                           allow_nan=False) + '\n').encode('utf-8')
    except (TypeError, ValueError, OverflowError):
        os.close(directory_fd)
        return False, 'serialization_failed'
    if len(data) > MAX_FAILURE_REPORT_BYTES:
        os.close(directory_fd)
        return False, 'report_size_limit'

    descriptor = None
    created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(FAILURE_REPORT_NAME, flags, 0o600, dir_fd=directory_fd)
        created = True
        os.fchmod(descriptor, 0o600)
        file_info = os.fstat(descriptor)
        if (not stat.S_ISREG(file_info.st_mode) or file_info.st_uid != os.geteuid()
                or stat.S_IMODE(file_info.st_mode) & 0o077):
            return False, 'unsafe_report_file'
        report_identity = (file_info.st_dev, file_info.st_ino)
        written = 0
        while written < len(data):
            count = os.write(descriptor, data[written:])
            if count <= 0:
                raise OSError('short report write')
            written += count
        os.fsync(descriptor)
        if not _output_still_matches(output, output_identity):
            return False, 'output_directory_changed'
        if not _report_entry_still_matches(directory_fd, report_identity):
            return False, 'report_replaced'
        os.close(descriptor)
        descriptor = None
        os.fsync(directory_fd)
        if not _output_still_matches(output, output_identity):
            return False, 'output_directory_changed'
        if not _report_entry_still_matches(directory_fd, report_identity):
            return False, 'report_replaced'
        return True, 'written'
    except FileExistsError:
        return False, 'report_already_exists'
    except Exception:
        return False, 'report_durability_uncertain' if created else 'report_write_failed'
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        os.close(directory_fd)


def _report_entry_still_matches(directory_fd: int, identity: tuple[int, int]) -> bool:
    try:
        current = os.stat(FAILURE_REPORT_NAME, dir_fd=directory_fd, follow_symlinks=False)
        return (stat.S_ISREG(current.st_mode) and current.st_uid == os.geteuid()
                and not (stat.S_IMODE(current.st_mode) & 0o077)
                and (current.st_dev, current.st_ino) == identity)
    except OSError:
        return False


def _assert_allclose_with_evidence(np, actual, desired, *, rtol: float, atol: float,
                                   context: dict, output: Path,
                                   report_enabled: bool,
                                   output_identity: tuple[int, int] | None = None) -> None:
    """Use the original NumPy assertion, adding safe evidence only on failure."""
    failed = False
    try:
        # Keep the original call and defaults. In particular, do not cast the
        # arrays, change tolerances, or alter assert_allclose's scalar rules.
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
    except AssertionError:
        failed = True
    if not failed:
        return

    try:
        safe_context = _safe_failure_context(context)
    except Exception:
        safe_context = {'phase': 'unknown', 'comparison': 'unknown'}
    try:
        summary = _safe_array_summary(np, actual, desired, atol, rtol)
    except Exception:
        summary = {
            'aggregate_status': 'unavailable',
            'aggregate_reason': 'safe_aggregation_unavailable',
            'failed_element_count': None,
            'failed_element_denominator': None,
            'max_absolute_error': None,
            'max_relative_error_nonzero_reference': None,
        }
    report_written = False
    report_error = 'disabled'
    if report_enabled:
        report = {
            'schema_version': 'lfg.owned-cached-export-failure.v1',
            'failure_kind': 'numpy_assert_allclose',
            'result': 'strict_parity_failed',
            'first_failure_only': True,
            'later_checks_run': False,
            'success_manifest_written': False,
            'measurement_written': False,
            'promotion_eligible': False,
            'context': safe_context,
            'tolerance': {'absolute': atol, 'relative': rtol, 'equal_nan': True},
            'comparison': summary,
        }
        try:
            report_written, report_error = _write_failure_report(output, report, output_identity)
        except Exception:
            # Reporting is secondary; an unexpected report-path failure must
            # never replace the already-established strict numerical failure.
            report_written, report_error = False, 'report_write_failed'
    # Raise after leaving the handler so the original NumPy exception text and
    # tensor values are not retained as displayed exception context.
    raise CachedParityFailure(report_written, report_error, safe_context, summary)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_bounded(path: Path, maximum: int) -> bytes:
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('input exceeds configured limit')
    return raw


def requested_ort_graph_optimization_level(config: dict) -> str:
    """Validate the optional ORT graph level without changing omitted defaults."""
    if 'graph_optimization_level' not in config:
        return 'unspecified'
    requested = config['graph_optimization_level']
    if not isinstance(requested, str) or requested not in ORT_GRAPH_OPTIMIZATION_LEVELS:
        raise ValueError('invalid ONNX Runtime graph optimization level')
    return requested


def create_ort_session_options(ort, cpu_threads: int, config: dict):
    """Create one cached-session options object, leaving omitted ORT defaults alone."""
    requested = requested_ort_graph_optimization_level(config)
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


def _reported_graph_optimization_level(ort, value) -> str:
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


def ort_session_provenance(ort, sessions: dict, requested: str,
                           cpu_threads: int, scope: str) -> dict:
    """Report read-back settings, keeping unavailable values explicitly unknown."""
    reported = []
    for name, session in sessions.items():
        effective_level = 'unknown'
        intra_threads = 'unknown'
        inter_threads = 'unknown'
        providers = 'unknown'
        try:
            options = session.get_session_options()
            effective_level = _reported_graph_optimization_level(
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
    return {'backend': 'onnxruntime', 'scope': scope,
            'onnxruntime_version': str(getattr(ort, '__version__', 'unknown')),
            'requested_graph_optimization_level': requested,
            'cpu_threads_requested': cpu_threads,
            'inter_op_num_threads_requested': 1,
            'sessions': reported}


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


def create_private_output(directory: Path) -> tuple[int, int]:
    """Create a new private output directory or refuse before exporting data.

    POSIX permission checks assume current-user and root processes are trusted.
    Windows requires a verified ACL; POSIX mode bits are not sufficient there.
    """
    if os.name != 'posix':
        raise ValueError('cached export output is disabled: verified private ACL creation is unavailable on this platform')
    if not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_DIRECTORY') or os.mkdir not in os.supports_dir_fd:
        raise ValueError('cached export output requires no-follow directory operations')
    absolute = directory if directory.is_absolute() else Path.cwd() / directory
    if '..' in absolute.parts or not absolute.name:
        raise ValueError('output path must not contain parent traversal')
    parent = absolute.parent
    effective_uid = os.geteuid()
    for ancestor in (parent, *parent.parents):
        info = os.lstat(ancestor)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid not in (effective_uid, 0):
            raise ValueError('output ancestors must be non-symlink directories owned by the user or root')
        if info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
            raise ValueError('output path has an unsafe writable ancestor')
    info = os.lstat(parent)
    if info.st_uid != effective_uid or info.st_mode & 0o077:
        raise ValueError('output parent must be owned by the current user and private')
    private_output(absolute)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(parent, flags)
    try:
        opened = os.fstat(parent_fd)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError('output parent changed during verification')
        os.mkdir(absolute.name, mode=0o700, dir_fd=parent_fd)
        child_fd = os.open(absolute.name, flags, dir_fd=parent_fd)
        try:
            child = os.fstat(child_fd)
            if child.st_uid != effective_uid or stat.S_IMODE(child.st_mode) & 0o077:
                raise ValueError('created output directory is not private')
            identity = (child.st_dev, child.st_ino)
        finally:
            os.close(child_fd)
    finally:
        os.close(parent_fd)
    return identity


def matching_greedy_byte(actual, reference) -> int:
    """Require identical byte choice even when near-tied logits are close."""
    import numpy as np
    actual_byte = int(np.argmax(actual[0]))
    reference_byte = int(np.argmax(reference[0]))
    if actual_byte != reference_byte:
        raise ValueError('cached ONNX greedy output differs from independent full-prefix output')
    return actual_byte


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


class PrefillGraph(_TorchModule):
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


class DecodeGraph(_TorchModule):
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
    if _DEPENDENCY_IMPORT_FAILED:
        raise ExporterDependenciesUnavailable()
    import numpy as np
    import onnx
    os.environ['ORT_DISABLE_TELEMETRY'] = '1'  # Set before import, or model names and hashes go to Microsoft.
    import onnxruntime as ort

    config_raw = read_bounded(config_path, MAX_CONFIG_BYTES)
    config = json.loads(config_raw)
    integer_fields = ('opset', 'cpu_threads', 'max_model_bytes', 'max_checkpoint_bytes', 'max_manifest_bytes')
    if (not isinstance(config, dict) or config.get('schema_version') != 'lfg.owned-byte-onnx-export.v1'
            or ('failure_evidence' in config and type(config['failure_evidence']) is not bool)
            or any(type(config.get(key)) is not int or config[key] < 1 for key in integer_fields)
            or not isinstance(config.get('parity_lengths'), list) or not config['parity_lengths']
            or any(type(length) is not int or length < 1 for length in config['parity_lengths'])
            or len(set(config['parity_lengths'])) != len(config['parity_lengths'])
            or any(type(config.get(key)) not in (float, int) or not math.isfinite(config[key]) or config[key] < 0
                   for key in ('absolute_tolerance', 'relative_tolerance'))
            or type(decode_steps) is not int or decode_steps < 1
            or type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1):
        raise ValueError('invalid cached export configuration')
    failure_evidence_enabled = config.get('failure_evidence', False)
    # Validate recognized settings and instantiate the options before private
    # output creation. When omitted, ORT retains its own SessionOptions default.
    prefill_session_options, requested_ort_level = create_ort_session_options(
        ort, config['cpu_threads'], config)
    decode_session_options, _ = create_ort_session_options(
        ort, config['cpu_threads'], config)
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
    if model.context < 2 or max(config['parity_lengths']) >= model.context:
        raise ValueError('cached export requires parity lengths below context and room for a decode step')
    sources = [source, *(additional_sources or [])]
    source_raws = [read_bounded(path, model.context) for path in sources]
    if any(not raw for raw in source_raws):
        raise ValueError('development inputs must be nonempty')
    if any(len(raw) >= model.context for raw in source_raws):
        raise ValueError('cached export requires every development input to leave room for a decode step')
    if max(config['parity_lengths']) > len(source_raws[0]):
        raise ValueError('first development input is shorter than configured parity lengths')
    if output.exists():
        raise ValueError('output directory must be new')
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
    output_identity = create_private_output(output)
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
    prefill_session = ort.InferenceSession(
        prefill_raw, sess_options=prefill_session_options, providers=['CPUExecutionProvider'])
    decode_session = ort.InferenceSession(
        decode_raw, sess_options=decode_session_options, providers=['CPUExecutionProvider'])
    onnxruntime_validation = ort_session_provenance(
        ort, {'prefill': prefill_session, 'decode': decode_session},
        requested_ort_level, config['cpu_threads'], 'export-validation-and-benchmark')
    max_logit_diff = 0.0
    max_cache_diff = 0.0
    exact_greedy = True
    source_measurements = []

    def assert_parity(actual, desired, *, phase, comparison, source_ordinal,
                      checked_prefix, decode_step=None, state=None):
        context = {
            'phase': phase,
            'comparison': comparison,
            'source_ordinal': source_ordinal,
            'prefix_length': checked_prefix,
        }
        if decode_step is not None:
            context['decode_step'] = decode_step
        if state is not None:
            context.update({'state_name': state['name'], 'state_kind': state['kind'],
                            'state_layer': state['layer']})
        _assert_allclose_with_evidence(
            np, actual, desired, rtol=config['relative_tolerance'],
            atol=config['absolute_tolerance'], context=context, output=output,
            report_enabled=failure_evidence_enabled, output_identity=output_identity)

    for source_index, raw in enumerate(source_raws):
        source_tokens = torch.tensor([list(raw)], dtype=torch.long)
        prefix_length = len(raw)
        if source_index == 0:
            for parity_length in config['parity_lengths']:
                parity_tokens = source_tokens[:, :parity_length]
                parity_outputs = prefill_session.run(None, {'tokens': parity_tokens.numpy()})
                with torch.inference_mode():
                    parity_expected = model(parity_tokens)[:, -1, :].detach().numpy()
                assert_parity(parity_outputs[0], parity_expected,
                              phase='configured_length_prefill',
                              comparison='onnx_prefill_vs_full_prefix_logits',
                              source_ordinal=source_index + 1, checked_prefix=parity_length)
                parity_cache = DecodeSession(model, 1)
                with torch.inference_mode():
                    parity_python_logits = None
                    for token_index in range(parity_length):
                        parity_python_logits = parity_cache.step(parity_tokens[:, token_index])
                assert_parity(parity_outputs[0], parity_python_logits.detach().cpu().numpy(),
                              phase='configured_length_prefill',
                              comparison='onnx_prefill_vs_incremental_logits',
                              source_ordinal=source_index + 1, checked_prefix=parity_length)
                for entry, state_value in zip(layout, parity_outputs[1:]):
                    reference = parity_cache._states[entry['layer']]
                    kind = entry['kind']
                    reference_key = 'conv' if kind == 'depthwise-convolution-history' else ('keys' if kind == 'attention-key' else 'values')
                    expected_state = reference[reference_key].detach().cpu().numpy()
                    if state_value.shape != expected_state.shape or not np.isfinite(state_value).all():
                        raise ValueError('invalid parity cache state')
                    assert_parity(state_value, expected_state,
                                  phase='configured_length_prefill',
                                  comparison='onnx_prefill_vs_incremental_cache',
                                  source_ordinal=source_index + 1,
                                  checked_prefix=parity_length, state=entry)
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
        assert_parity(actual, expected, phase='full_source_prefill',
                      comparison='onnx_prefill_vs_full_prefix_logits',
                      source_ordinal=source_index + 1, checked_prefix=prefix_length)
        max_logit_diff = max(max_logit_diff, float(np.max(np.abs(actual - expected))))
        python_cache = DecodeSession(model, 1)
        with torch.inference_mode():
            python_logits = torch.stack([python_cache.step(source_tokens[:, index]) for index in range(prefix_length)], dim=1)[:, -1, :].numpy()
        assert_parity(actual, python_logits, phase='full_source_prefill',
                      comparison='onnx_prefill_vs_incremental_logits',
                      source_ordinal=source_index + 1, checked_prefix=prefix_length)
        for entry, state_value in zip(layout, onnx_state[1:]):
            reference = python_cache._states[entry['layer']]
            kind = entry['kind']
            reference_key = 'conv' if kind == 'depthwise-convolution-history' else ('keys' if kind == 'attention-key' else 'values')
            expected_state = reference[reference_key].detach().cpu().numpy()
            if state_value.shape != expected_state.shape or not np.isfinite(state_value).all():
                raise ValueError('invalid cache state output')
            assert_parity(state_value, expected_state, phase='full_source_prefill',
                          comparison='onnx_prefill_vs_incremental_cache',
                          source_ordinal=source_index + 1,
                          checked_prefix=prefix_length, state=entry)
            max_cache_diff = max(max_cache_diff, float(np.max(np.abs(state_value - expected_state))) if state_value.size else 0.0)
        bytes_out = []
        reference_bytes = []
        decode_samples = []
        decode_cpu_samples = []
        state_values = list(onnx_state[1:])
        logits = actual
        reference_logits = expected
        measured_steps = min(decode_steps, model.context - prefix_length)
        if measured_steps < 1:
            raise ValueError('development input leaves no context for decode measurement')
        for step_index in range(measured_steps):
            next_byte = matching_greedy_byte(logits, reference_logits)
            bytes_out.append(next_byte)
            reference_bytes.append(int(np.argmax(reference_logits[0])))
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
                full_tokens = torch.tensor([list(raw) + reference_bytes], dtype=torch.long)
                expected_next = model(full_tokens)[:, -1, :].detach().numpy()
                cached_reference = python_cache.step(torch.tensor([next_byte], dtype=torch.long)).detach().cpu().numpy()
            if logits.shape != expected_next.shape or not np.isfinite(logits).all():
                raise ValueError('invalid cache decode logits')
            assert_parity(logits, expected_next, phase='autoregressive_decode',
                          comparison='onnx_decode_vs_full_prefix_logits',
                          source_ordinal=source_index + 1, checked_prefix=prefix_length,
                          decode_step=step_index + 1)
            assert_parity(logits, cached_reference, phase='autoregressive_decode',
                          comparison='onnx_decode_vs_incremental_logits',
                          source_ordinal=source_index + 1, checked_prefix=prefix_length,
                          decode_step=step_index + 1)
            max_logit_diff = max(max_logit_diff, float(np.max(np.abs(logits - expected_next))))
            for entry, state_value in zip(layout, state_values):
                reference = python_cache._states[entry['layer']]
                kind = entry['kind']
                reference_key = 'conv' if kind == 'depthwise-convolution-history' else ('keys' if kind == 'attention-key' else 'values')
                expected_state = reference[reference_key].detach().cpu().numpy()
                if state_value.shape != expected_state.shape or not np.isfinite(state_value).all():
                    raise ValueError('invalid cache decode state')
                assert_parity(state_value, expected_state, phase='autoregressive_decode',
                              comparison='onnx_decode_vs_incremental_cache',
                              source_ordinal=source_index + 1, checked_prefix=prefix_length,
                              decode_step=step_index + 1, state=entry)
                max_cache_diff = max(max_cache_diff, float(np.max(np.abs(state_value - expected_state))) if state_value.size else 0.0)
            reference_logits = expected_next
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
        'onnxruntime_validation': onnxruntime_validation,
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
        'onnxruntime_validation': onnxruntime_validation,
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


def main(argv=None, *, sanitize_process_output: bool = False) -> int:
    # Fail before parsing so even early argparse exits cannot bypass the
    # standalone process-lifetime output boundary.
    if _STARTUP_OUTPUT_SANITIZATION_FAILED:
        _emit_status({'status': 'output_sanitization_unavailable',
                      'failure_detail_suppressed': True}, to_stderr=True)
        return 6

    parser = SanitizedArgumentParser(
        prog='lfg-export-owned-cached-onnx',
        description='Export and strictly validate owned cached ONNX graphs.')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Directory containing the owned checkpoint weights.')
    parser.add_argument('--config', type=Path, required=True,
                        help='Export configuration JSON file.')
    parser.add_argument('--source', type=Path, required=True,
                        help='Primary authorized source file.')
    parser.add_argument('--output', type=Path, required=True,
                        help='New private output directory.')
    parser.add_argument('--additional-source', type=Path, action='append', default=[],
                        help='Additional authorized source file; may be repeated.')
    parser.add_argument('--decode-steps', type=int, default=32,
                        help='Number of incremental decode steps to validate (default: 32).')
    parser.add_argument('--warmup', type=int, default=1,
                        help='Warmup runs for measurement (default: 1).')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Measurement repeats (default: 3).')
    try:
        arguments = parser.parse_args(argv)
    except SanitizedHelpRequested as help_request:
        try:
            help_bytes = help_request.text.encode('utf-8')
        except UnicodeEncodeError:
            return 7
        return 0 if _emit_payload(help_bytes) else 7
    except SanitizedArgumentError:
        _emit_status({'status': 'invalid_arguments', 'failure_detail_suppressed': True},
                     to_stderr=True)
        return 2
    try:
        result = export(arguments.checkpoint, arguments.config, arguments.source, arguments.output,
                        arguments.additional_source, arguments.decode_steps,
                        arguments.warmup, arguments.repeats)
    except CachedParityFailure as failure:
        result = {
            'status': 'strict_parity_failed',
            'failure_report_written': failure.failure_report_written,
            'failure_report_error_category': failure.failure_report_error_category,
        }
        _emit_status(result, to_stderr=True)
        return 1
    except ExporterDependenciesUnavailable:
        result = {'status': 'dependency_unavailable', 'failure_detail_suppressed': True}
        _emit_status(result, to_stderr=True)
        return 5
    except ImportError:
        result = {'status': 'dependency_unavailable', 'failure_detail_suppressed': True}
        _emit_status(result, to_stderr=True)
        return 5
    except OutputSanitizationUnavailable:
        result = {'status': 'output_sanitization_unavailable', 'failure_detail_suppressed': True}
        _emit_status(result, to_stderr=True)
        return 6
    except ValueError:
        result = {'status': 'invalid_or_unsafe_export', 'failure_detail_suppressed': True}
        _emit_status(result, to_stderr=True)
        return 2
    except OSError:
        result = {'status': 'filesystem_or_runtime_failure', 'failure_detail_suppressed': True}
        _emit_status(result, to_stderr=True)
        return 3
    except Exception:
        result = {'status': 'unexpected_export_failure', 'failure_detail_suppressed': True}
        _emit_status(result, to_stderr=True)
        return 4
    # Do not print private paths, inputs, model output, or artifact digests.
    return 0 if _emit_status(result) else 7


if __name__ == '__main__':
    raise SystemExit(main(sanitize_process_output=True))
