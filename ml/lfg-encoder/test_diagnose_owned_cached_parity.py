"""Synthetic parity diagnostic tests; no owner-private data is used."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest import mock


def strict_json(raw):
    def reject(value):
        raise ValueError(value)
    return json.loads(raw, parse_constant=reject)


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'numpy required')
class DiagnosticMathTests(unittest.TestCase):
    def test_float32_original_dtype_boundary_matches_numpy_exactly(self):
        import numpy as np
        import diagnose_owned_cached_parity as subject

        rng = np.random.default_rng(10)
        reference = rng.uniform(-10, 10, 100000).astype(np.float32)
        limit = np.float32(1e-5) + np.float32(1e-5) * np.abs(reference)
        factor = rng.uniform(.9998, 1.0002, len(reference)).astype(np.float32)
        actual = reference + limit * factor
        numpy_close = np.isclose(actual, reference, atol=1e-5, rtol=1e-5, equal_nan=True)
        promoted = (np.abs(actual.astype(np.float64) - reference.astype(np.float64))
                    <= 1e-5 + 1e-5 * np.abs(reference.astype(np.float64)))
        self.assertEqual(int(np.count_nonzero(numpy_close != promoted)), 1)
        summary = subject.compare_arrays(actual, reference, 1e-5, 1e-5)
        self.assertEqual(summary['failed_original_count'], int(len(actual) - numpy_close.sum()))
        self.assertEqual(summary['failed_original_denominator'], len(actual))

    def test_near_zero_nonfinite_empty_shape_and_dtype_semantics(self):
        import numpy as np
        import diagnose_owned_cached_parity as subject

        actual = np.array([1e-5, 1.00002], dtype=np.float32)
        reference = np.array([0.0, 1.0], dtype=np.float32)
        expected = len(actual) - int(np.isclose(actual, reference, atol=1e-5, rtol=1e-5, equal_nan=True).sum())
        summary = subject.compare_arrays(actual, reference, 1e-5, 1e-5)
        self.assertEqual(summary['failed_original_count'], expected)
        self.assertEqual(summary['zero_reference_count'], 1)
        nonfinite = subject.compare_arrays(np.array([np.nan, np.inf], dtype=np.float32),
                                           np.array([np.nan, np.inf], dtype=np.float32), 1e-5, 1e-5)
        self.assertEqual(nonfinite['status'], 'nonfinite')
        self.assertEqual(nonfinite['failed_original_count'], 0)  # assert_allclose equal_nan=True
        self.assertEqual(nonfinite['nonfinite_count'], 2)
        empty = subject.compare_arrays(np.array([], dtype=np.float32), np.array([], dtype=np.float32),
                                      1e-5, 1e-5)
        self.assertEqual(empty['status'], 'not_tested')
        shape = subject.compare_arrays(np.zeros((1,), np.float32), np.zeros((2,), np.float32), 1e-5, 1e-5)
        self.assertEqual(shape['status'], 'invalid')
        dtype = subject.compare_arrays(np.zeros((1,), np.float32), np.zeros((1,), np.float64), 1e-5, 1e-5)
        self.assertEqual(dtype['status'], 'invalid')

    def test_per_tensor_error_aggregation_and_no_tensor_values(self):
        import numpy as np
        import diagnose_owned_cached_parity as subject

        expected_logits = np.array([[1.0, 2.0]], dtype=np.float32)
        actual_logits = expected_logits.copy()
        expected_key = np.zeros((1, 2), dtype=np.float32)
        actual_key = np.array([[0.0, .001]], dtype=np.float32)
        result = {}
        subject._record_tensors(result, [actual_logits, actual_key],
            [expected_logits, expected_key], ['logits', 'key_0'], 1e-5, 1e-5)
        self.assertEqual(result['logits']['failed_original_count'], 0)
        self.assertEqual((result['key_0']['failed_original_count'],
                          result['key_0']['failed_original_denominator']), (1, 2))
        self.assertNotIn('actual', result['key_0'])
        self.assertNotIn('reference', result['key_0'])

    def test_opposite_sign_max_finite_float32_has_finite_float64_summary(self):
        import numpy as np
        import diagnose_owned_cached_parity as subject

        maximum = np.finfo(np.float32).max
        actual = np.array([maximum, -maximum], dtype=np.float32)
        reference = np.array([-maximum, maximum], dtype=np.float32)
        summary = subject.compare_arrays(actual, reference, 1e-5, 1e-5)
        self.assertEqual(summary['failed_original_count'], 2)
        self.assertTrue(np.isfinite(summary['max_absolute_error']))
        json.dumps(summary, allow_nan=False)

    def test_unrepresentable_float64_difference_is_reported_without_nonstandard_json(self):
        import numpy as np
        import diagnose_owned_cached_parity as subject

        maximum = np.finfo(np.float64).max
        summary = subject.compare_arrays(np.array([maximum]), np.array([-maximum]), 1e-5, 1e-5)
        self.assertEqual(summary['descriptive_status'], 'unrepresentable_finite_difference')
        self.assertIsNone(summary['max_absolute_error'])
        self.assertIsNone(summary['max_relative_error_nonzero_reference'])
        json.loads(json.dumps(summary, allow_nan=False),
                   parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    def test_same_history_stability_is_separate_from_independent_trajectory(self):
        import numpy as np
        import diagnose_owned_cached_parity as subject

        torch_a = np.array([[0.0, 2.0, 1.0]], dtype=np.float32)
        onnx_b = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
        full_onnx_b = onnx_b.copy()
        row = subject.decision_stability(1, torch_a, onnx_b, torch_a, torch_a, full_onnx_b)
        self.assertFalse(row['python_full_vs_onnx_independent_trajectory']['agreement'])
        self.assertTrue(row['onnx_vs_full_same_onnx_history']['agreement'])
        first = subject._first_divergence(None, 0, True)
        first = subject._first_divergence(first, 1, row['python_full_vs_onnx_independent_trajectory']['agreement'])
        first = subject._first_divergence(first, 2, True)  # paths later reconverge
        self.assertEqual(first, 1)


class PolicyTests(unittest.TestCase):
    def test_policy_requires_exact_original_failure_and_hash_only_pins(self):
        import diagnose_owned_cached_parity as subject

        digest = 'a' * 64
        policy = {'schema_version': 'lfg.owned-cached-parity-diagnostic-policy.v2',
                  'pins': {key: digest for key in subject.PIN_NAMES},
                  'original_failure': {'status': 'failed', 'absolute_tolerance': 1e-5,
                                       'relative_tolerance': 1e-5, 'extension': 'SENTINEL'}}
        subject._validate_policy(policy)  # extension data is accepted but inert
        self.assertNotIn('SENTINEL', json.dumps(subject._failure_summary(policy)))
        policy['original_failure']['absolute_tolerance'] = 2e-5
        with self.assertRaises(subject.DiagnosticAbort):
            subject._validate_policy(policy)

    def test_ort_default_is_only_selected_by_omitting_level(self):
        import diagnose_owned_cached_parity as subject

        class SessionOptions:
            def __init__(self):
                self.graph_optimization_level = 'library-default'

        fake_ort = SimpleNamespace(SessionOptions=SessionOptions)
        options, requested = subject._create_ort_session_options(fake_ort, 1, {})
        self.assertEqual(requested, 'unspecified')
        self.assertEqual(options.graph_optimization_level, 'library-default')
        with self.assertRaisesRegex(ValueError, 'invalid ONNX Runtime graph optimization level'):
            subject._create_ort_session_options(
                fake_ort, 1, {'graph_optimization_level': 'unspecified'})


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                         ('torch', 'onnx', 'onnxruntime', 'safetensors', 'numpy')),
                     'requires torch, onnx, onnxruntime, safetensors, numpy')
@unittest.skipUnless(os.name == 'posix', 'requires POSIX verified private output')
class DiagnosticRunnerTests(unittest.TestCase):
    sentinel = 'SENTINEL-PRIVATE-PATH-OR-EXCEPTION'

    def make_fixture(self, root: Path, graph_optimization_level='ORT_DISABLE_ALL'):
        import hashlib
        import owned_student
        from safetensors.torch import save_file
        import diagnose_owned_cached_parity as subject
        import export_owned_cached_onnx as exporter

        private_parent = root / 'private'
        private_parent.mkdir(mode=0o700)
        checkpoint = root / 'checkpoint'
        checkpoint.mkdir(mode=0o700)
        architecture = {'width': 8, 'layers': 2, 'heads': 2, 'conv_kernel': 3,
                        'context': 8, 'ff_multiplier': 2, 'attention_every': 2}
        model = owned_student.OwnedByteStudent(architecture).eval()
        weights = checkpoint / 'weights.safetensors'
        save_file(model.state_dict(), str(weights))
        model_code = Path(owned_student.__file__).read_bytes()
        manifest = {'architecture': architecture,
                    'weights_sha256': subject.sha256(weights.read_bytes()),
                    'code_sha256': {'owned_student.py': subject.sha256(model_code)}}
        manifest_path = checkpoint / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        config = json.loads(Path(exporter.__file__).with_name(
            'owned-byte-cached-onnx-export.example.json').read_text())
        if graph_optimization_level is None:
            config.pop('graph_optimization_level', None)
        else:
            config['graph_optimization_level'] = graph_optimization_level
        config['future_extension'] = self.sentinel
        config['parity_lengths'] = [1]
        config_path = root / 'export-config.json'
        config_path.write_text(json.dumps(config))
        source_id = subject.sha256(b'nonprivate-synthetic-source-id')
        source_path = root / (self.sentinel + '.bin')
        source_path.write_bytes(bytes([3, 7, 9, 11, 15, 18, 22]))
        screen_path = root / 'screening.json'
        screen = {'schema_version': 'lfg.train-source-screening-claim.v1',
                  'source_id_sha256': source_id,
                  'source_sha256': subject.sha256(source_path.read_bytes()),
                  'claim_status': 'screened_train_only', 'ignored_extension': self.sentinel}
        screen_path.write_text(json.dumps(screen))
        selection = {'schema_version': 'lfg.screened-train-selection.v1',
                     'entries': [{'source_id_sha256': source_id,
                                  'source_sha256': subject.sha256(source_path.read_bytes()),
                                  'screening_manifest_sha256': subject.sha256(screen_path.read_bytes()),
                                  'screening_status': 'screened_train_only'}],
                     'ignored_extension': self.sentinel}
        selection_path = root / 'selection.json'
        selection_path.write_text(json.dumps(selection))
        export_parent = private_parent / 'export-run'
        exporter.create_private_output(export_parent)
        artifacts = export_parent / 'graphs'
        exporter.export(checkpoint, config_path, source_path, artifacts,
                        decode_steps=1, warmup=0, repeats=1)
        graph_dir = private_parent / 'graphs'
        exporter.create_private_output(graph_dir)
        for name in ('prefill.onnx', 'decode.onnx'):
            (graph_dir / name).write_bytes((artifacts / name).read_bytes())

        pins = {
            'checkpoint_manifest_sha256': subject.sha256(manifest_path.read_bytes()),
            'weights_sha256': subject.sha256(weights.read_bytes()),
            'model_code_sha256': subject.sha256(model_code),
            'cached_reference_code_sha256': subject.sha256(Path(exporter.__file__).with_name(
                'owned_cached_decode.py').read_bytes()),
            'exporter_code_sha256': subject.sha256(Path(exporter.__file__).read_bytes()),
            'export_config_sha256': subject.sha256(config_path.read_bytes()),
            'prefill_onnx_sha256': subject.sha256((graph_dir / 'prefill.onnx').read_bytes()),
            'decode_onnx_sha256': subject.sha256((graph_dir / 'decode.onnx').read_bytes()),
            'selection_manifest_sha256': subject.sha256(selection_path.read_bytes())}
        policy = {'schema_version': 'lfg.owned-cached-parity-diagnostic-policy.v2',
                  'pins': pins,
                  'original_failure': {'status': 'failed', 'absolute_tolerance': 1e-5,
                                       'relative_tolerance': 1e-5, 'ignored_extension': self.sentinel},
                  'ignored_extension': self.sentinel}
        policy_path = root / 'policy.json'
        policy_path.write_text(json.dumps(policy))
        return {'root': root, 'private_parent': private_parent, 'checkpoint': checkpoint,
            'manifest': manifest_path, 'weights': weights, 'config': config_path,
            'graphs': graph_dir, 'source': source_path, 'source_id': source_id,
            'screen': screen_path, 'selection': selection_path, 'policy': policy_path,
            'output': private_parent / 'diagnostic', 'pins': pins}

    def run_fixture(self, fixture, **overrides):
        import diagnose_owned_cached_parity as subject
        args = {'checkpoint': fixture['checkpoint'], 'export_config': fixture['config'],
            'graphs': fixture['graphs'], 'policy_path': fixture['policy'],
            'selection_manifest_path': fixture['selection'],
            'source_paths': {fixture['source_id']: fixture['source']},
            'screening_manifest_paths': {fixture['source_id']: fixture['screen']},
            'output': fixture['output'], 'lengths': [2], 'decode_steps': 1,
            'cpu_threads': 1, 'max_bytes': 1 << 20, 'max_work_items': 1,
            'max_prefix_tokens': 1000, 'max_seconds': 60}
        args.update(overrides)
        return subject.diagnose(**args)

    def test_public_runner_end_to_end_on_synthetic_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            before = {path: path.read_bytes() for path in
                (fixture['manifest'], fixture['weights'], fixture['config'], fixture['source'],
                 fixture['selection'], fixture['screen'], fixture['graphs'] / 'prefill.onnx',
                 fixture['graphs'] / 'decode.onnx')}
            result = self.run_fixture(fixture)
            self.assertEqual(result['status'], 'diagnostic_complete', result)
            report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
            self.assertEqual(report['status'], 'diagnostic_complete')
            self.assertEqual(report['samples'][0]['source_id_sha256'], fixture['source_id'])
            self.assertEqual(report['screening_claim_status'], 'operator_claim_not_independently_verified')
            self.assertFalse(report['assessment_quality_evaluated'])
            self.assertFalse(report['promotion_eligible'])
            provenance = report['onnxruntime_validation']
            self.assertEqual(provenance['backend'], 'onnxruntime')
            self.assertEqual(provenance['scope'], 'cached-parity-diagnostic-validation-and-decode')
            self.assertEqual(provenance['requested_graph_optimization_level'], 'ORT_DISABLE_ALL')
            self.assertEqual([row['effective_graph_optimization_level_reported']
                              for row in provenance['sessions']], ['ORT_DISABLE_ALL', 'ORT_DISABLE_ALL'])
            self.assertTrue(all(row['providers_reported'] == ['CPUExecutionProvider']
                                for row in provenance['sessions']))
            self.assertFalse(report['execution_control']['real_input_guard_enforced_by_runner'])
            self.assertFalse(report['execution_control']['host_supervisor_verified_by_runner'])
            self.assertEqual({path: path.read_bytes() for path in before}, before)
            self.assertNotIn(self.sentinel, (fixture['output'] / 'diagnostic.json').read_text())
            self.assertNotIn(str(fixture['source']), (fixture['output'] / 'diagnostic.json').read_text())
            self.assertNotIn(self.sentinel, (fixture['output'] / 'diagnostic.json').read_text())

    def test_omitted_graph_level_preserves_ort_default_and_records_readback(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary), graph_optimization_level=None)
            result = self.run_fixture(fixture)
            self.assertEqual(result['status'], 'diagnostic_complete', result)
            report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
            provenance = report['onnxruntime_validation']
            self.assertEqual(provenance['requested_graph_optimization_level'], 'unspecified')
            self.assertTrue(all(row['effective_graph_optimization_level_reported'] == 'ORT_ENABLE_ALL'
                                for row in provenance['sessions']))

    def test_legacy_policy_pinned_exporter_supports_both_ort_settings(self):
        import base64
        import diagnose_owned_cached_parity as subject

        fixture_path = Path(__file__).resolve().parent / 'fixtures' / 'legacy_owned_cached_exporter.b64'
        legacy_exporter = base64.b64decode(fixture_path.read_bytes().strip(), validate=True)
        self.assertEqual(subject.sha256(legacy_exporter),
                         'e8c3ef56f26abaadeef341f7f68201754b4aeb42561b2f64641d25cd16d2ea2f')
        self.assertNotIn(b'create_ort_session_options', legacy_exporter)
        expected = [('ORT_DISABLE_ALL', 'ORT_DISABLE_ALL'), (None, 'ORT_ENABLE_ALL')]

        for requested_level, effective_level in expected:
            with self.subTest(requested_level=requested_level), tempfile.TemporaryDirectory() as temporary:
                fixture = self.make_fixture(Path(temporary), graph_optimization_level=requested_level)
                legacy_path = fixture['root'] / 'legacy-exporter.py'
                legacy_path.write_bytes(legacy_exporter)
                policy = json.loads(fixture['policy'].read_bytes())
                policy['pins']['exporter_code_sha256'] = subject.sha256(legacy_exporter)
                fixture['policy'].write_text(json.dumps(policy))

                original_with_name = Path.with_name
                diagnostic_file = Path(subject.__file__)

                def route_legacy_exporter(path, name):
                    if path == diagnostic_file and name == 'export_owned_cached_onnx.py':
                        return legacy_path
                    return original_with_name(path, name)

                with mock.patch.object(Path, 'with_name', route_legacy_exporter):
                    result = self.run_fixture(fixture)
                self.assertEqual(result['status'], 'diagnostic_complete', result)
                report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
                provenance = report['onnxruntime_validation']
                self.assertEqual(provenance['requested_graph_optimization_level'],
                                 requested_level or 'unspecified')
                self.assertEqual([row['effective_graph_optimization_level_reported']
                                  for row in provenance['sessions']],
                                 [effective_level, effective_level])
                self.assertTrue(all(row['providers_reported'] == ['CPUExecutionProvider']
                                    for row in provenance['sessions']))

    def test_invalid_graph_level_fails_before_private_output_creation(self):
        import diagnose_owned_cached_parity as subject

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            config = json.loads(fixture['config'].read_bytes())
            config['graph_optimization_level'] = 'ORT_ENABLE_UNKNOWN'
            fixture['config'].write_text(json.dumps(config))
            policy = json.loads(fixture['policy'].read_bytes())
            policy['pins']['export_config_sha256'] = subject.sha256(fixture['config'].read_bytes())
            fixture['policy'].write_text(json.dumps(policy))
            result = self.run_fixture(fixture)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['failure_stage'], 'preflight')
            self.assertEqual(result['failure_code'], 'ort_graph_optimization_level_invalid_or_unavailable')
            self.assertFalse(fixture['output'].exists())

    def test_swapped_path_after_pin_load_uses_pinned_weight_bytes_and_reports_mutation(self):
        from safetensors.torch import load as original_load, save_file
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            alternate = fixture['root'] / 'different-weights.safetensors'
            import owned_student
            architecture = json.loads(fixture['manifest'].read_bytes())['architecture']
            save_file(owned_student.OwnedByteStudent(architecture).state_dict(), str(alternate))
            used_hashes = []
            def swap_then_load(raw):
                used_hashes.append(__import__('hashlib').sha256(raw).hexdigest())
                fixture['weights'].write_bytes(alternate.read_bytes())
                return original_load(raw)
            with mock.patch('safetensors.torch.load', side_effect=swap_then_load):
                result = self.run_fixture(fixture)
            self.assertEqual(used_hashes, [fixture['pins']['weights_sha256']])
            self.assertEqual(result['status'], 'incomplete')
            report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
            self.assertTrue(report['input_mutation_detected'])
            self.assertEqual(report['pins']['weights_sha256'], fixture['pins']['weights_sha256'])

    def test_stale_modules_are_shadowed_by_pinned_code_then_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            touched = []
            def stale_getattr(module_name):
                def fail(attribute):
                    touched.append((module_name, attribute))
                    raise RuntimeError(self.sentinel)
                return fail
            stale_student = ModuleType('owned_student')
            stale_student.__getattr__ = stale_getattr('owned_student')
            stale_cache = ModuleType('owned_cached_decode')
            stale_cache.__getattr__ = stale_getattr('owned_cached_decode')
            with mock.patch.dict(sys.modules, {'owned_student': stale_student,
                                                'owned_cached_decode': stale_cache}):
                result = self.run_fixture(fixture)
                self.assertIs(sys.modules['owned_student'], stale_student)
                self.assertIs(sys.modules['owned_cached_decode'], stale_cache)
            self.assertEqual(result['status'], 'diagnostic_complete')
            self.assertEqual(touched, [])
            report_bytes = (fixture['output'] / 'diagnostic.json').read_bytes()
            self.assertNotIn(self.sentinel.encode(), report_bytes)

    def test_swapped_model_code_path_executes_captured_bytes_and_reports_mutation(self):
        import diagnose_owned_cached_parity as subject
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            module_path = fixture['root'] / 'captured-owned-student.py'
            original_code = Path('ml/lfg-encoder/owned_student.py').read_bytes()
            module_path.write_bytes(original_code)
            original_with_name = Path.with_name
            diagnostic_file = Path(subject.__file__)
            original_loader = subject._load_pinned_module
            def route_model_code(path, name):
                if path == diagnostic_file and name == 'owned_student.py':
                    return module_path
                return original_with_name(path, name)
            def swap_after_capture(stack, name, raw, filename):
                if name == 'owned_student':
                    module_path.write_text("raise RuntimeError('" + self.sentinel + "')\n")
                return original_loader(stack, name, raw, filename)
            with mock.patch.object(Path, 'with_name', route_model_code), \
                 mock.patch.object(subject, '_load_pinned_module', side_effect=swap_after_capture):
                result = self.run_fixture(fixture)
            self.assertEqual(result['status'], 'incomplete')
            report_bytes = (fixture['output'] / 'diagnostic.json').read_bytes()
            report = strict_json(report_bytes)
            self.assertTrue(report['input_mutation_detected'])
            self.assertNotIn(self.sentinel.encode(), report_bytes)
            self.assertEqual(report['samples'][0]['status'], 'measured')

    def test_invalid_or_nonfinite_prefill_and_step_logits_never_feed_argmax(self):
        import numpy as np
        os.environ['ORT_DISABLE_TELEMETRY'] = '1'  # Set before import, or model names and hashes go to Microsoft.
        import onnxruntime as ort
        original_session = ort.InferenceSession
        original_argmax = np.argmax

        class AlteredSession:
            instances = []
            def __init__(self, *args, **kwargs):
                self.inner = original_session(*args, **kwargs)
                self.kind = 'prefill' if not self.instances else 'decode'
                self.instances.append(self)
                self.calls = 0
            def get_inputs(self):
                return self.inner.get_inputs()
            def get_outputs(self):
                return self.inner.get_outputs()
            def run(self, names, feeds):
                self.calls += 1
                outputs = list(self.inner.run(names, feeds))
                mode = self.mode
                should_corrupt = self.kind == mode and (mode != 'decode' or self.calls == 1)
                if mode == 'free_decode':
                    should_corrupt = self.kind == 'decode' and self.calls == 2
                if should_corrupt:
                    if self.corruption == 'nonfinite':
                        outputs[0] = np.full_like(outputs[0], np.nan)
                    elif self.corruption == 'invalid_shape':
                        outputs[0] = np.zeros((1, 255), dtype=np.float32)
                    elif self.corruption == 'invalid_dtype':
                        outputs[0] = outputs[0].astype(np.float64)
                return outputs

        for mode, corruption, failure_code in (
                ('prefill', 'nonfinite', 'nonfinite_prefill'),
                ('prefill', 'invalid_shape', 'invalid_prefill'),
                ('decode', 'nonfinite', 'nonfinite_teacher_forced_step'),
                ('decode', 'invalid_shape', 'invalid_teacher_forced_step'),
                ('decode', 'invalid_dtype', 'invalid_teacher_forced_step'),
                ('free_decode', 'nonfinite', 'nonfinite_step_decision'),
                ('free_decode', 'invalid_shape', 'invalid_step_decision'),
                ('free_decode', 'invalid_dtype', 'invalid_step_decision')):
            with self.subTest(mode=mode, corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                fixture = self.make_fixture(Path(temporary))
                AlteredSession.instances = []
                AlteredSession.mode = mode
                AlteredSession.corruption = corruption
                argmax_inputs = []
                def guarded_argmax(values, *args, **kwargs):
                    values = np.asarray(values)
                    self.assertTrue(np.isfinite(values).all())
                    argmax_inputs.append(values.shape)
                    return original_argmax(values, *args, **kwargs)
                with mock.patch('onnxruntime.InferenceSession', AlteredSession), \
                     mock.patch('numpy.argmax', side_effect=guarded_argmax):
                    result = self.run_fixture(fixture, decode_steps=3)
                self.assertEqual(result['status'], 'incomplete')
                report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
                sample = report['samples'][0]
                self.assertEqual(sample['failure_code'], failure_code)
                self.assertEqual(sample['status'], 'incomplete')
                self.assertGreater(len(argmax_inputs), 0)
                if mode == 'prefill':
                    self.assertEqual(AlteredSession.instances[1].calls, 0)
                    self.assertTrue(all(row['status'] == 'not_tested'
                                        for row in sample['teacher_forced_decode']))
                    self.assertEqual([row['decision_index'] for row in
                                      sample['decision_stability']], [0, 1, 2, 3])
                    self.assertIn(sample['decision_stability'][0]['status'],
                                  ('nonfinite', 'invalid'))
                    self.assertEqual(sample['teacher_forced_transitions_completed'], 0)
                    self.assertEqual(sample['free_trajectory_transitions_attempted'], 0)
                    self.assertEqual(sample['free_trajectory_transitions_completed'], 0)
                elif mode == 'free_decode':
                    self.assertEqual(AlteredSession.instances[1].calls, 2)
                    self.assertEqual(sample['teacher_forced_decode'][0]['status'], 'measured')
                    self.assertTrue(all(row['status'] == 'not_tested'
                                        for row in sample['teacher_forced_decode'][1:]))
                    self.assertEqual(sample['teacher_forced_transitions_attempted'], 1)
                    self.assertEqual(sample['teacher_forced_transitions_completed'], 1)
                    self.assertEqual(sample['free_trajectory_transitions_attempted'], 1)
                    self.assertEqual(sample['free_trajectory_transitions_completed'], 0)
                    self.assertEqual(sample['decode_transitions_completed'], 0)
                    self.assertEqual([row['decision_index'] for row in
                                      sample['decision_stability']], [0, 1, 2, 3])
                    failed_decision = sample['decision_stability'][1]
                    self.assertEqual(failed_decision['status'],
                        'nonfinite' if corruption == 'nonfinite' else 'invalid')
                    self.assertTrue(all(row['status'] == 'not_tested'
                                        for row in sample['decision_stability'][2:]))
                else:
                    self.assertEqual(AlteredSession.instances[1].calls, 1)
                    self.assertEqual(sample['free_trajectory_transitions_completed'], 0)
                    self.assertEqual(sample['free_trajectory_transitions_attempted'], 0)
                    self.assertEqual(sample['teacher_forced_transitions_attempted'], 1)
                    self.assertEqual(sample['teacher_forced_transitions_completed'], 0)
                    self.assertEqual(sample['decode_transitions_completed'], 0)
                    self.assertEqual(sample['teacher_forced_decode'][0]['status'],
                        'nonfinite' if corruption == 'nonfinite' else 'invalid')
                    self.assertEqual([row['decision_index'] for row in
                                      sample['decision_stability']], [0, 1, 2, 3])
                self.assertNotIn(b'\x03\x07\x09', (fixture['output'] / 'diagnostic.json').read_bytes())

    def test_invalid_prefix_reports_all_unattempted_decisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            result = self.run_fixture(fixture, lengths=[8], decode_steps=4)
            self.assertEqual(result['status'], 'incomplete')
            report_bytes = (fixture['output'] / 'diagnostic.json').read_bytes()
            report = strict_json(report_bytes)
            sample = report['samples'][0]
            decisions = sample['decision_stability']
            self.assertEqual([row['decision_index'] for row in decisions], list(range(5)))
            self.assertTrue(all(row['status'] == 'not_tested' for row in decisions))
            self.assertTrue(all(row['reason'] == 'prefix_out_of_bounds' for row in decisions))
            self.assertNotIn(self.sentinel.encode(), report_bytes)

    def test_sanitized_crash_partial_and_untrusted_extensions(self):
        os.environ['ORT_DISABLE_TELEMETRY'] = '1'  # Set before import, or model names and hashes go to Microsoft.
        import onnxruntime as ort
        original_session = ort.InferenceSession
        sentinel = self.sentinel
        class FailingSession:
            def __init__(self, *args, **kwargs):
                self.inner = original_session(*args, **kwargs)
            def get_inputs(self):
                return self.inner.get_inputs()
            def get_outputs(self):
                return self.inner.get_outputs()
            def run(self, *args, **kwargs):
                raise RuntimeError(sentinel)
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            stdout, stderr = io.StringIO(), io.StringIO()
            with mock.patch('onnxruntime.InferenceSession', FailingSession), \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = self.run_fixture(fixture)
            self.assertEqual(result['status'], 'incomplete')
            report_bytes = (fixture['output'] / 'diagnostic.json').read_bytes()
            self.assertNotIn(sentinel.encode(), report_bytes)
            self.assertNotIn(sentinel, stdout.getvalue())
            self.assertNotIn(sentinel, stderr.getvalue())
            self.assertNotIn(str(fixture['source']), report_bytes.decode())
            report = strict_json(report_bytes)
            self.assertEqual(report['samples'][0]['status'], 'crash')
            self.assertEqual(report['samples'][0]['failure_code'], 'sanitized_execution_exception')

    def test_context_truncation_marks_remaining_steps_not_tested(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            result = self.run_fixture(fixture, lengths=[7], decode_steps=4)
            self.assertEqual(result['status'], 'incomplete')
            report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
            sample = report['samples'][0]
            self.assertEqual(sample['decode_transitions_planned'], 1)
            self.assertEqual(sample['decode_transitions_completed'], 1, sample)
            self.assertEqual(sample['status'], 'incomplete')
            self.assertTrue(all(row['status'] == 'not_tested'
                                for row in sample['teacher_forced_decode'][1:]))

    def test_attention_position_axes_and_convolution_cache_contract(self):
        os.environ['ORT_DISABLE_TELEMETRY'] = '1'  # Set before import, or model names and hashes go to Microsoft.
        import onnxruntime as ort
        import owned_student
        import diagnose_owned_cached_parity as subject
        import export_owned_cached_onnx as exporter

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            prefill = ort.InferenceSession(str(fixture['graphs'] / 'prefill.onnx'),
                                           providers=['CPUExecutionProvider'])
            decode = ort.InferenceSession(str(fixture['graphs'] / 'decode.onnx'),
                                          providers=['CPUExecutionProvider'])
            architecture = json.loads(fixture['manifest'].read_bytes())['architecture']
            layout = exporter.cache_layout(architecture)
            subject._validate_graph_contract(prefill, decode, layout, architecture)
            prefill_outputs = {item.name: item for item in prefill.get_outputs()}
            decode_inputs = {item.name: item for item in decode.get_inputs()}
            decode_outputs = {item.name: item for item in decode.get_outputs()}
            self.assertEqual(prefill_outputs['key_1'].shape[2], 'sequence')
            self.assertEqual(decode_inputs['key_1'].shape[2], 'past_length')
            self.assertEqual(decode_outputs['key_1_out'].shape[2], 'present_length')
            self.assertEqual(decode_inputs['conv_0'].shape, ['batch', 8, 2])

            class WrongPositionAxis:
                def get_inputs(self):
                    values = []
                    for item in decode.get_inputs():
                        shape = list(item.shape)
                        if item.name == 'key_1':
                            shape[2] = 'wrong_axis'
                        values.append(SimpleNamespace(name=item.name, type=item.type, shape=shape))
                    return values
                def get_outputs(self):
                    return decode.get_outputs()
            with self.assertRaises(subject.DiagnosticAbort) as error:
                subject._validate_graph_contract(prefill, WrongPositionAxis(), layout, architecture)
            self.assertEqual(error.exception.code, 'decode_input_contract')

    def test_cooperative_time_limit_leaves_explicit_not_tested_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            result = self.run_fixture(fixture, decode_steps=4, max_seconds=1e-9)
            self.assertEqual(result['status'], 'incomplete')
            report = strict_json((fixture['output'] / 'diagnostic.json').read_bytes())
            self.assertEqual(report['execution_control']['timer_kind'], 'cooperative_in_process')
            self.assertFalse(report['execution_control']['hard_wall_time_enforced'])
            self.assertFalse(report['execution_control']['real_input_guard_enforced_by_runner'])
            self.assertEqual(report['samples'][0]['status'], 'not_tested')
            self.assertEqual(report['samples'][0]['teacher_forced_decode'][0]['status'], 'not_tested')
            indices = [row['decision_index'] for row in
                       report['samples'][0]['decision_stability']]
            self.assertEqual(indices, list(range(5)))
            self.assertEqual(len(indices), len(set(indices)))


if __name__ == '__main__':
    unittest.main()
