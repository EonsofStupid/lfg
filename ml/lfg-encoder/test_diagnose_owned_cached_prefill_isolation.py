"""Synthetic tests for the three-path prefill isolation diagnostic."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


def strict_json(raw):
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


CORE_PATH = Path(__file__).with_name('diagnose_owned_cached_parity.py')
# These transient test pins stand in for the external launcher inputs only in
# synthetic fixture runs. They are deliberately not accepted runtime defaults.
_test_core_digest = hashlib.sha256(CORE_PATH.read_bytes()).hexdigest()
TOOL_PATH = Path(__file__).with_name('diagnose_owned_cached_prefill_isolation.py')
_test_tool_digest = hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest()
os.environ['LFG_PREFILL_CORE_EXPECTED_SHA256'] = _test_core_digest
SYNTHETIC_CORE_BYTES = b"FIXTURE_MARKER = 'pinned-core-fixture-v1'\n"
SYNTHETIC_CORE_SHA256 = 'b1459bd91e8b46c6a22581bab7252f1233696daf179cdbb070cfbe451822799e'


REQUIRES_STACK = unittest.skipUnless(all(importlib.util.find_spec(name) for name in (
    'numpy', 'torch', 'onnx', 'onnxruntime', 'safetensors')),
    'requires numpy, torch, onnx, onnxruntime, and safetensors')
POSIX_ONLY = unittest.skipUnless(os.name == 'posix', 'requires POSIX private output checks')


@REQUIRES_STACK
class IsolationMathTests(unittest.TestCase):
    def test_git_lf_and_windows_crlf_raw_bytes_require_matching_external_pins(self):
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ['git', 'cat-file', 'blob',
             'HEAD:ml/lfg-encoder/diagnose_owned_cached_parity.py'],
            cwd=repo_root, check=True, capture_output=True)
        lf_raw = result.stdout
        crlf_raw = lf_raw.replace(b'\n', b'\r\n')
        # Expected values are calculated independently for these transient raw
        # test artifacts; production requires launcher-supplied pins.
        lf_pin = hashlib.sha256(lf_raw).hexdigest()
        crlf_pin = hashlib.sha256(crlf_raw).hexdigest()
        self.assertNotEqual(lf_pin, crlf_pin)
        import diagnose_owned_cached_prefill_isolation as subject
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, raw, pin in (
                    ('lf.py', lf_raw, lf_pin),
                    ('crlf.py', crlf_raw, crlf_pin)):
                path = root / name
                path.write_bytes(raw)
                module, captured, digest = subject._execute_pinned_core(path, pin)
                self.assertEqual(captured, raw)
                self.assertEqual(digest, pin)
                self.assertTrue(hasattr(module, 'ORIGINAL_ATOL'))
                with self.assertRaises(subject.CorePinError):
                    subject._execute_pinned_core(path,
                        crlf_pin if pin == lf_pin else lf_pin)
            synthetic = root / 'synthetic-core.py'
            synthetic.write_bytes(SYNTHETIC_CORE_BYTES)
            fixture_module, captured, digest = subject._execute_pinned_core(
                synthetic, SYNTHETIC_CORE_SHA256)
            self.assertEqual(captured, SYNTHETIC_CORE_BYTES)
            self.assertEqual(digest, SYNTHETIC_CORE_SHA256)
            self.assertEqual(fixture_module.FIXTURE_MARKER, 'pinned-core-fixture-v1')
            with self.assertRaises(subject.CorePinError):
                subject._execute_pinned_core(synthetic, '0' * 64)
            with self.assertRaisesRegex(subject.CorePinError, 'core_digest_pin_missing_or_invalid'):
                subject._execute_pinned_core(root / 'lf.py', None)
            with self.assertRaisesRegex(subject.CorePinError, 'core_digest_pin_missing_or_invalid'):
                subject._execute_pinned_core(root / 'lf.py', 'NOT-A-DIGEST')
            with self.assertRaisesRegex(subject.CorePinError, 'core_file_unavailable'):
                subject._execute_pinned_core(root / 'missing.py', lf_pin)

    def test_three_way_perturbations_distinguish_onnx_and_full_reference(self):
        import numpy as np
        import diagnose_owned_cached_prefill_isolation as subject

        base = np.array([[0.25, -0.5]], dtype=np.float32)
        onnx = base.copy()
        full = base.copy()
        incremental = base.copy()
        onnx[0, 0] += 1e-3
        result = subject.three_way_comparisons([onnx], [full], [incremental],
                                                ['logits'], 1e-5, 1e-5)
        self.assertEqual(result['onnx_vs_torch_full_prefill']['logits']['failed_original_count'], 1)
        self.assertEqual(result['torch_full_prefill_vs_torch_incremental']['logits']['failed_original_count'], 0)
        self.assertEqual(result['onnx_vs_torch_incremental']['logits']['failed_original_count'], 1)

        onnx = base.copy()
        full = base.copy()
        incremental = base.copy()
        full[0, 1] -= 1e-3
        result = subject.three_way_comparisons([onnx], [full], [incremental],
                                                ['logits'], 1e-5, 1e-5)
        self.assertEqual(result['onnx_vs_torch_full_prefill']['logits']['failed_original_count'], 1)
        self.assertEqual(result['torch_full_prefill_vs_torch_incremental']['logits']['failed_original_count'], 1)
        self.assertEqual(result['onnx_vs_torch_incremental']['logits']['failed_original_count'], 0)

    def test_trace_perturbation_names_earliest_layer_and_operation(self):
        import torch
        import diagnose_owned_cached_prefill_isolation as subject

        specs = [(0, 'block_input_norm', None, None),
                 (1, 'input_gate_content', None, None)]
        full0 = torch.zeros((1, 2, 2), dtype=torch.float32)
        inc0 = full0.clone()
        full1 = torch.zeros((1, 2, 2), dtype=torch.float32)
        inc1 = full1.clone()
        full1[0, 1, 0] = 0.25
        traces = {'full': {'layer_0.block_input_norm': [full0],
                           'layer_1.input_gate_content': [full1]},
                  'incremental': {'layer_0.block_input_norm': [inc0[:, :1], inc0[:, 1:]],
                                  'layer_1.input_gate_content': [inc1[:, :1], inc1[:, 1:]]}}
        result = subject._trace_summary(traces, specs, 2, 1e-5, 1e-5)
        self.assertEqual(result['earliest_numerical_difference'], {
            'layer': 1, 'operation': 'input_gate_content', 'status': 'measured',
            'failed_original_count': 1})
        self.assertEqual(result['earliest_tolerance_divergence'], {
            'layer': 1, 'operation': 'input_gate_content', 'status': 'measured',
            'failed_original_count': 1})
        self.assertNotIn('full_values', result)
        self.assertNotIn('incremental_values', result)

    def test_comparisons_reject_shape_dtype_and_nonfinite_results(self):
        import numpy as np
        import diagnose_owned_cached_prefill_isolation as subject

        result = subject.three_way_comparisons(
            [np.zeros((1, 2), dtype=np.float64)], [np.zeros((1, 2), dtype=np.float32)],
            [np.zeros((1, 2), dtype=np.float32)], ['logits'], 1e-5, 1e-5)
        self.assertEqual(result['onnx_vs_torch_full_prefill']['status'], 'invalid')
        result = subject.three_way_comparisons(
            [np.array([[np.nan, 0]], dtype=np.float32)],
            [np.zeros((1, 2), dtype=np.float32)],
            [np.zeros((1, 2), dtype=np.float32)], ['logits'], 1e-5, 1e-5)
        self.assertEqual(result['onnx_vs_torch_full_prefill']['logits']['status'], 'nonfinite')


@REQUIRES_STACK
@POSIX_ONLY
class IsolationRunnerTests(unittest.TestCase):
    sentinel = 'NESTED-SOURCE-PATH-AND-METADATA-SENTINEL'

    def make_fixture(self, root: Path, source_bytes=None):
        import hashlib
        import test_diagnose_owned_cached_parity as existing
        import diagnose_owned_cached_parity as core

        factory = existing.DiagnosticRunnerTests('runTest')
        fixture = factory.make_fixture(root)
        if source_bytes is None:
            source_bytes = b'\x03\x07\x09\x0b\x0f\x12\x16\x18\x19'
        fixture['source'].write_bytes(source_bytes)
        screen = json.loads(fixture['screen'].read_text())
        screen['source_sha256'] = hashlib.sha256(source_bytes).hexdigest()
        screen['nested_extension'] = {'private_path': self.sentinel, 'nested': [self.sentinel]}
        fixture['screen'].write_text(json.dumps(screen))
        selection = json.loads(fixture['selection'].read_text())
        entry = selection['entries'][0]
        entry['source_sha256'] = screen['source_sha256']
        entry['screening_manifest_sha256'] = hashlib.sha256(
            fixture['screen'].read_bytes()).hexdigest()
        selection['nested_extension'] = {'source': {'sentinel': self.sentinel}}
        fixture['selection'].write_text(json.dumps(selection))
        policy = json.loads(fixture['policy'].read_text())
        policy['pins']['selection_manifest_sha256'] = hashlib.sha256(
            fixture['selection'].read_bytes()).hexdigest()
        policy['nested_extension'] = {'source': self.sentinel}
        fixture['policy'].write_text(json.dumps(policy))
        return fixture

    def run_fixture(self, fixture, **overrides):
        import hashlib
        import diagnose_owned_cached_prefill_isolation as subject

        args = {'checkpoint': fixture['checkpoint'], 'export_config': fixture['config'],
                'graphs': fixture['graphs'], 'policy_path': fixture['policy'],
                'selection_manifest_path': fixture['selection'],
                'source_paths': {fixture['source_id']: fixture['source']},
                'screening_manifest_paths': {fixture['source_id']: fixture['screen']},
                'output': fixture['output'], 'lengths': [1, 2, 3, 4, 8],
                'cpu_threads': 2, 'max_bytes': 1 << 20, 'max_work_items': 8,
                'max_prefix_tokens': 100000, 'max_intermediate_bytes': 1 << 20,
                'max_seconds': 60,
                'optimization_variants': ('current_default', 'disable_all')}
        args.update(overrides)
        digest = hashlib.sha256(Path(subject.__file__).read_bytes()).hexdigest()
        pins = {'LFG_PREFILL_ISOLATION_EXPECTED_SHA256': digest,
                'LFG_PREFILL_CORE_EXPECTED_SHA256': _test_core_digest}
        with mock.patch.dict(os.environ, pins):
            return subject.diagnose(**args)

    def test_three_paths_lengths_kernel_boundary_context_and_redaction(self):
        import diagnose_owned_cached_prefill_isolation as subject
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            result = self.run_fixture(fixture, lengths=[1, 2, 3, 4, 7, 8])
            self.assertEqual(result['status'], 'diagnostic_complete', result)
            report_raw = (fixture['output'] / 'diagnostic.json').read_bytes()
            report = strict_json(report_raw)
            self.assertEqual(report['requested_work_items'], 6)
            self.assertEqual(report['completed_work_items'], 6)
            self.assertFalse(report['assessment_quality_evaluated'])
            self.assertFalse(report['promotion_eligible'])
            self.assertEqual(report['original_failure']['absolute_tolerance'], 1e-5)
            self.assertEqual(report['original_failure']['relative_tolerance'], 1e-5)
            self.assertEqual(report['shared_core_sha256'], _test_core_digest)
            self.assertEqual(report['shared_core_expected_sha256'], _test_core_digest)
            self.assertTrue(report['launcher_digest_match'])
            self.assertEqual([row['name'] for row in report['session_variants']],
                             ['current_default', 'disable_all'])
            for sample in report['samples']:
                self.assertEqual(sample['status'], 'measured')
                self.assertEqual(set(sample['torch_full_vs_torch_incremental']),
                                 {'status', 'logits', 'cache_tensors'})
                self.assertEqual(len(sample['ort_variants']), 2)
                for variant in sample['ort_variants']:
                    self.assertEqual(variant['provider'], 'CPUExecutionProvider')
                    self.assertEqual(variant['intra_op_num_threads'], 2)
                    self.assertEqual(variant['inter_op_num_threads'], 1)
                    self.assertEqual(set(variant['ort_vs_torch_full_prefill']),
                                     {'status', 'logits', 'cache_tensors'})
                    self.assertEqual(set(variant['ort_vs_torch_incremental']),
                                     {'status', 'logits', 'cache_tensors'})
            self.assertNotIn(self.sentinel.encode(), report_raw)
            self.assertNotIn(str(fixture['source']).encode(), report_raw)
            self.assertEqual(json.loads(fixture['config'].read_text())['absolute_tolerance'], 1e-5)
            self.assertEqual(json.loads(fixture['config'].read_text())['relative_tolerance'], 1e-5)

    def test_invalid_context_length_is_reported_and_other_work_is_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            result = self.run_fixture(fixture, lengths=[8, 9], max_work_items=2)
            self.assertEqual(result['status'], 'incomplete')
            report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
            self.assertEqual([sample['status'] for sample in report['samples']], ['measured', 'invalid'])
            self.assertEqual(report['completed_work_items'], 1)
            self.assertEqual(report['invalid_work_items'], 1)
            self.assertEqual(report['not_tested_work_items'], 0)
            self.assertEqual(report['samples'][1]['failure_code'], 'prefix_out_of_bounds')

    def test_original_tolerance_failure_is_measured_not_mislabeled_incomplete(self):
        import numpy as np
        import onnxruntime as ort
        original_session = ort.InferenceSession

        class PerturbedSession:
            def __init__(self, *args, **kwargs):
                self.inner = original_session(*args, **kwargs)
            def get_inputs(self):
                return self.inner.get_inputs()
            def get_outputs(self):
                return self.inner.get_outputs()
            def run(self, names, feeds):
                outputs = list(self.inner.run(names, feeds))
                outputs[1] = outputs[1].copy()
                outputs[1].flat[0] += np.float32(1e-2)
                return outputs

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            with mock.patch('onnxruntime.InferenceSession', PerturbedSession):
                result = self.run_fixture(fixture, lengths=[2], max_work_items=1)
            self.assertEqual(result['status'], 'diagnostic_complete', result)
            report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
            sample = report['samples'][0]
            self.assertEqual(sample['status'], 'measured')
            self.assertEqual(sample['original_tolerance_status'], 'failed')
            self.assertGreater(sample['original_tolerance_failure_count'], 0)

    def test_invalid_onnx_and_torch_outputs_never_claim_original_tolerance_success(self):
        import numpy as np
        import onnxruntime as ort
        import diagnose_owned_cached_prefill_isolation as subject
        original_session = ort.InferenceSession
        original_tensor_values = subject._tensor_values

        for mode in ('shape', 'dtype', 'paired_nan', 'torch_dtype'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                class InvalidSession:
                    def __init__(self, *args, **kwargs):
                        self.inner = original_session(*args, **kwargs)
                    def get_inputs(self):
                        return self.inner.get_inputs()
                    def get_outputs(self):
                        return self.inner.get_outputs()
                    def run(self, names, feeds):
                        outputs = list(self.inner.run(names, feeds))
                        if mode == 'shape':
                            outputs[0] = outputs[0].reshape(-1)
                        elif mode == 'dtype':
                            outputs[0] = outputs[0].astype(np.float64)
                        elif mode == 'paired_nan':
                            outputs[0] = np.full_like(outputs[0], np.nan)
                        return outputs

                def torch_values(torch, values):
                    arrays = original_tensor_values(torch, values)
                    if mode == 'torch_dtype':
                        arrays[0] = arrays[0].astype(np.float64)
                    elif mode == 'paired_nan':
                        arrays[0] = np.full_like(arrays[0], np.nan)
                    return arrays

                fixture = self.make_fixture(Path(temporary))
                with mock.patch('onnxruntime.InferenceSession', InvalidSession), \
                     mock.patch.object(subject, '_tensor_values', side_effect=torch_values):
                    result = self.run_fixture(fixture, lengths=[2], max_work_items=1)
                self.assertEqual(result['status'], 'incomplete', result)
                report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
                sample = report['samples'][0]
                self.assertEqual(sample['status'], 'invalid')
                self.assertNotEqual(sample['original_tolerance_status'], 'within_tolerance')

    def test_budget_marks_layer_trace_not_tested_while_cache_checks_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            result = self.run_fixture(fixture, lengths=[2], max_work_items=1,
                                      max_intermediate_bytes=1)
            self.assertEqual(result['status'], 'diagnostic_complete', result)
            report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
            self.assertEqual(report['samples'][0]['layer_trace']['status'], 'not_tested')
            self.assertEqual(report['samples'][0]['layer_trace']['reason'],
                             'intermediate_capture_budget_exceeded')

    def test_time_limit_leaves_work_explicitly_not_tested(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            result = self.run_fixture(fixture, lengths=[2], max_work_items=1,
                                      max_seconds=1e-12)
            self.assertEqual(result['status'], 'incomplete')
            report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
            self.assertEqual(report['samples'][0]['status'], 'not_tested')
            self.assertEqual(report['not_tested_work_items'], 1)
            self.assertEqual(report['incomplete_work_items'], 0)
            self.assertEqual(report['failure_code'], 'cooperative_time_limit')

    def test_crash_after_execution_is_not_counted_as_not_tested(self):
        import diagnose_owned_cached_prefill_isolation as subject
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            with mock.patch.object(subject, '_incremental_values', side_effect=RuntimeError()):
                result = self.run_fixture(fixture, lengths=[2], max_work_items=1)
            self.assertEqual(result['status'], 'incomplete')
            report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
            self.assertEqual(report['crashed_work_items'], 1)
            self.assertEqual(report['not_tested_work_items'], 0)

    def test_partial_timeout_is_incomplete_and_later_rows_remain_not_tested(self):
        import diagnose_owned_cached_prefill_isolation as subject
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            calls = {'count': 0}
            def timed_out(_started, _limit):
                calls['count'] += 1
                return calls['count'] >= 3
            with mock.patch.object(subject.core, '_timed_out', side_effect=timed_out):
                result = self.run_fixture(fixture, lengths=[2, 3], max_work_items=2)
            self.assertEqual(result['status'], 'incomplete')
            report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
            self.assertEqual([x['status'] for x in report['samples']], ['incomplete', 'not_tested'])
            self.assertEqual(report['incomplete_work_items'], 1)
            self.assertEqual(report['not_tested_work_items'], 1)
            self.assertEqual(report['requested_work_items'], 2)
            self.assertEqual(report['requested_work_items'], sum(report[name] for name in (
                'completed_work_items', 'invalid_work_items', 'incomplete_work_items',
                'crashed_work_items', 'in_progress_work_items', 'not_tested_work_items')))

    def test_invalid_core_bytes_rejected_and_module_alias_is_ignored(self):
        import types
        import diagnose_owned_cached_prefill_isolation as subject
        import diagnose_owned_cached_parity as existing_core
        with tempfile.TemporaryDirectory() as temporary:
            bad = Path(temporary) / 'core.py'
            bad.write_bytes(SYNTHETIC_CORE_BYTES)
            with self.assertRaisesRegex(subject.CorePinError, 'core_digest_mismatch'):
                subject._execute_pinned_core(bad, '0' * 64)
        alias = types.ModuleType('diagnose_owned_cached_parity')
        alias.marker = 'must not be used'
        with mock.patch.dict('sys.modules', {'diagnose_owned_cached_parity': alias}):
            loaded, raw, digest = subject._execute_pinned_core(
                CORE_PATH, _test_core_digest)
            self.assertIsNot(loaded, alias)
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            self.assertEqual(existing_core.ORIGINAL_ATOL, loaded.ORIGINAL_ATOL)

    def test_final_report_false_or_exception_never_returns_success(self):
        import diagnose_owned_cached_prefill_isolation as subject
        core = subject.core
        for final_behavior in ('false', 'raise'):
            with self.subTest(final_behavior=final_behavior), tempfile.TemporaryDirectory() as temporary:
                fixture = self.make_fixture(Path(temporary))
                original = core._atomic_report
                count = {'value': 0}
                def write(directory, report):
                    count['value'] += 1
                    if count['value'] == 4:
                        if final_behavior == 'raise':
                            raise OSError('private details must not leak')
                        return False
                    return original(directory, report)
                with mock.patch.object(core, '_atomic_report', side_effect=write):
                    result = self.run_fixture(fixture, lengths=[2], max_work_items=1)
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['report_written'])
                self.assertEqual(result['failure_code'], 'final_report_write_failed')
                persisted = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
                self.assertEqual(persisted['status'], 'running')

    def test_tool_or_core_change_during_run_is_detected(self):
        import diagnose_owned_cached_prefill_isolation as subject
        core = subject.core
        original = subject._captured_bytes_unchanged
        targets = (Path(subject.__file__), subject._CORE_PATH)
        for target in targets:
            with self.subTest(target=target.name), tempfile.TemporaryDirectory() as temporary:
                fixture = self.make_fixture(Path(temporary))
                def changed(path, raw, limit):
                    return False if Path(path) == target else original(path, raw, limit)
                with mock.patch.object(subject, '_captured_bytes_unchanged', side_effect=changed):
                    result = self.run_fixture(fixture, lengths=[2], max_work_items=1)
                self.assertEqual(result['status'], 'incomplete', result)
                report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
                self.assertTrue(report['input_mutation_detected'])
                self.assertEqual(report['failure_code'], 'input_mutation_detected')

    def test_main_requires_external_digest_bootstrap_and_returns_nonzero(self):
        import hashlib
        import diagnose_owned_cached_prefill_isolation as subject
        env = os.environ.copy()
        env.pop('LFG_PREFILL_ISOLATION_EXPECTED_SHA256', None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(subject.main([]), 2)
        expected = _test_tool_digest
        with mock.patch.dict(os.environ, {'LFG_PREFILL_ISOLATION_EXPECTED_SHA256': expected}, clear=False), \
             mock.patch.object(subject, 'diagnose', return_value={
                 'status': 'diagnostic_complete', 'report_written': False,
                 'completed_work_items': 1, 'not_tested_work_items': 0,
                 'failure_stage': None, 'failure_code': None, 'original_export_status': 'failed'}):
            result = subject.main([
                '--checkpoint', 'C:/not-used', '--export-config', 'C:/not-used',
                '--graphs', 'C:/not-used', '--policy', 'C:/not-used',
                '--selection-manifest', 'C:/not-used', '--output', 'C:/not-used',
                '--length', '1', '--threads', '1', '--max-bytes', '100',
                '--max-work-items', '1', '--max-prefix-tokens', '10',
                '--max-intermediate-bytes', '100', '--max-seconds', '1',
                '--optimization-variant', 'current_default', '--optimization-variant', 'disable_all'])
            self.assertEqual(result, 2)


    @REQUIRES_STACK
    def test_explicit_optimization_and_thread_settings_reach_sessions(self):
        import onnxruntime as ort
        original_session = ort.InferenceSession
        observed = []

        def capture_options(*args, **kwargs):
            options = kwargs['sess_options']
            observed.append((options.graph_optimization_level,
                             options.intra_op_num_threads, options.inter_op_num_threads,
                             tuple(kwargs['providers'])))
            return original_session(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            with mock.patch.object(ort, 'InferenceSession', side_effect=capture_options):
                result = self.run_fixture(fixture, lengths=[2], max_work_items=1,
                                          cpu_threads=3)
            self.assertEqual(result['status'], 'diagnostic_complete', result)
            self.assertEqual(observed, [
                (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 3, 1, ('CPUExecutionProvider',)),
                (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 3, 1, ('CPUExecutionProvider',)),
                (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 3, 1, ('CPUExecutionProvider',)),
            ])


class SanitizedStartupTests(unittest.TestCase):
    def test_core_pin_startup_failures_are_sanitized_json(self):
        tool = Path(__file__).with_name('diagnose_owned_cached_prefill_isolation.py')
        core = Path(__file__).with_name('diagnose_owned_cached_parity.py')
        tool_raw = tool.read_bytes()
        core_raw = core.read_bytes()
        tool_pin = hashlib.sha256(tool_raw).hexdigest()
        actual_core_pin = hashlib.sha256(core_raw).hexdigest()
        cases = (
            ('missing_pin', None, True, 'core_digest_pin_missing_or_invalid'),
            ('malformed_pin', 'bad-pin', True, 'core_digest_pin_missing_or_invalid'),
            ('mismatched_pin', '0' * 64, True, 'core_digest_mismatch'),
            ('missing_file', actual_core_pin, False, 'core_file_unavailable'),
        )
        for label, pin, include_core, expected_code in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(prefix='startup-sentinel-') as temporary:
                root = Path(temporary)
                tool_path = root / 'diagnose_owned_cached_prefill_isolation.py'
                tool_path.write_bytes(tool_raw)
                if include_core:
                    (root / 'diagnose_owned_cached_parity.py').write_bytes(core_raw)
                env = os.environ.copy()
                env['LFG_PREFILL_ISOLATION_EXPECTED_SHA256'] = tool_pin
                env.pop('LFG_PREFILL_CORE_EXPECTED_SHA256', None)
                if pin is not None:
                    env['LFG_PREFILL_CORE_EXPECTED_SHA256'] = pin
                result = subprocess.run([sys.executable, str(tool_path)], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stderr, '')
                self.assertNotIn(str(root), result.stdout + result.stderr)
                parsed = strict_json(result.stdout.encode())
                self.assertEqual(parsed['failure_code'], expected_code)
                self.assertNotIn('Traceback', result.stdout + result.stderr)

    def test_subprocess_ignores_preexisting_import_alias(self):
        repo_root = Path(__file__).resolve().parents[2]
        expected_core = hashlib.sha256(CORE_PATH.read_bytes()).hexdigest()
        code = (
            'import sys,types; alias=types.ModuleType("diagnose_owned_cached_parity"); '
            'alias.ORIGINAL_ATOL=999; sys.modules["diagnose_owned_cached_parity"]=alias; '
            'import diagnose_owned_cached_prefill_isolation as s; '
            'm=s._load_pinned_core(); '
            'print("ok" if m is not alias and m.ORIGINAL_ATOL != 999 else "bad")')
        env = os.environ.copy()
        env['LFG_PREFILL_CORE_EXPECTED_SHA256'] = expected_core
        env['PYTHONPATH'] = str(repo_root / 'ml/lfg-encoder')
        result = subprocess.run([sys.executable, '-c', code], cwd=repo_root / 'ml/lfg-encoder',
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'ok')

    def test_subprocess_detects_tool_and_core_changes_after_capture(self):
        source_tool = Path(__file__).with_name('diagnose_owned_cached_prefill_isolation.py')
        source_core = Path(__file__).with_name('diagnose_owned_cached_parity.py')
        code = (
            'import hashlib,json,os,sys; from pathlib import Path; '
            'import diagnose_owned_cached_prefill_isolation as s; '
            'm=s._load_pinned_core(); p=s._CORE_PATH; c=s._PINNED_CORE_RAW; '
            't=Path(s.__file__); tr=t.read_bytes(); p.write_bytes(b"changed"); '
            't.write_bytes(b"changed"); '
            'print(json.dumps([s._captured_bytes_unchanged(p,c,1<<20), '
            's._captured_bytes_unchanged(t,tr,1<<20)]))')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(source_tool, root / source_tool.name)
            shutil.copyfile(source_core, root / source_core.name)
            env = os.environ.copy()
            env['LFG_PREFILL_CORE_EXPECTED_SHA256'] = hashlib.sha256(
                source_core.read_bytes()).hexdigest()
            env['PYTHONPATH'] = str(root)
            result = subprocess.run([sys.executable, '-c', code], cwd=root,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(strict_json(result.stdout.encode()), [False, False])

if __name__ == '__main__':
    unittest.main()
