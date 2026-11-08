"""Synthetic parity fixtures for the experimental cached ONNX export."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest


@unittest.skipUnless(os.name == 'posix', 'cached export requires verified private POSIX output')
@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'onnx', 'onnxruntime', 'safetensors')),
                     'torch, onnx, onnxruntime and safetensors required')
class ExportOwnedCachedTests(unittest.TestCase):
    def test_prefill_decode_parity_and_artifact_contract(self):
        import owned_student
        from safetensors.torch import save_file
        import export_owned_cached_onnx as subject

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            architecture = {'width': 8, 'layers': 2, 'heads': 2, 'conv_kernel': 3,
                            'context': 8, 'ff_multiplier': 2, 'attention_every': 2}
            model = owned_student.OwnedByteStudent(architecture).eval()
            weights = checkpoint / 'weights.safetensors'
            save_file(model.state_dict(), str(weights))
            sha = lambda raw: hashlib.sha256(raw).hexdigest()
            manifest = {'architecture': architecture, 'weights_sha256': sha(weights.read_bytes()),
                        'code_sha256': {'owned_student.py': sha(Path(owned_student.__file__).read_bytes())}}
            (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
            config = json.loads(Path(subject.__file__).with_name('owned-byte-cached-onnx-export.example.json').read_text())
            config['graph_optimization_level'] = 'ORT_DISABLE_ALL'
            config['failure_evidence'] = True
            config['future_extension'] = 'accepted-but-inert'
            config['parity_lengths'] = [1, 4]
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            source = root / 'source.bin'
            source.write_bytes(b'abcd')
            extra = root / 'other.bin'
            extra.write_bytes(b'x')
            output = root / 'artifacts'

            result = subject.export(checkpoint, config_path, source, output, [extra], decode_steps=3,
                                    warmup=0, repeats=1)

            self.assertEqual(result['status'], 'verified')
            self.assertTrue(result['greedy_bytes_exact'])
            self.assertFalse((output / 'failure-report.json').exists())
            artifact = json.loads((output / 'manifest.json').read_bytes())
            measurement = json.loads((output / 'measurement.json').read_bytes())
            self.assertEqual(artifact['schema_version'], 'lfg.owned-byte-cached-onnx.v1')
            self.assertEqual(artifact['cache']['overflow'], 'reject')
            self.assertEqual(artifact['cache']['eviction'], 'none')
            self.assertEqual(artifact['prefill']['sequence_max'], 8)
            provenance = artifact['onnxruntime_validation']
            self.assertEqual(provenance['backend'], 'onnxruntime')
            self.assertEqual(provenance['scope'], 'export-validation-and-benchmark')
            self.assertEqual(provenance['requested_graph_optimization_level'], 'ORT_DISABLE_ALL')
            self.assertEqual([row['name'] for row in provenance['sessions']], ['prefill', 'decode'])
            self.assertTrue(all(row['effective_graph_optimization_level_reported'] == 'ORT_DISABLE_ALL'
                                for row in provenance['sessions']))
            self.assertTrue(all(row['providers_reported'] == ['CPUExecutionProvider']
                                for row in provenance['sessions']))
            self.assertEqual(measurement['onnxruntime_validation'], provenance)
            self.assertEqual(measurement['export_config_sha256'], sha(config_path.read_bytes()))
            manifest_canonical = (json.dumps(artifact, sort_keys=True, separators=(',', ':')) + '\n').encode()
            self.assertEqual(measurement['artifact_manifest_sha256'], sha(manifest_canonical))
            self.assertEqual(len(measurement['samples']), 2)
            self.assertEqual([sample['source_bytes'] for sample in measurement['samples']], [4, 1])
            self.assertTrue(all(sample['full_prefix_cached_parity_checked'] for sample in measurement['samples']))

    def test_kernel_one_and_last_decode_position(self):
        import owned_student
        from safetensors.torch import save_file
        import export_owned_cached_onnx as subject

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            architecture = {'width': 8, 'layers': 2, 'heads': 2, 'conv_kernel': 1,
                            'context': 4, 'ff_multiplier': 2, 'attention_every': 2}
            model = owned_student.OwnedByteStudent(architecture).eval()
            weights = checkpoint / 'weights.safetensors'
            save_file(model.state_dict(), str(weights))
            sha = lambda raw: hashlib.sha256(raw).hexdigest()
            manifest = {'architecture': architecture, 'weights_sha256': sha(weights.read_bytes()),
                        'code_sha256': {'owned_student.py': sha(Path(owned_student.__file__).read_bytes())}}
            (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
            config = json.loads(Path(subject.__file__).with_name('owned-byte-cached-onnx-export.example.json').read_text())
            config.pop('graph_optimization_level')
            config['parity_lengths'] = [1]
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            source = root / 'source.bin'
            source.write_bytes(b'abc')
            result = subject.export(checkpoint, config_path, source, root / 'out', decode_steps=1,
                                    warmup=0, repeats=1)
            self.assertEqual(result['status'], 'verified')
            measurement = json.loads((root / 'out' / 'measurement.json').read_bytes())
            self.assertEqual(measurement['samples'][0]['decode_steps'], 1)
            self.assertEqual(measurement['onnxruntime_validation']['requested_graph_optimization_level'],
                             'unspecified')
            self.assertEqual(measurement['onnxruntime_validation']['sessions'][0][
                'effective_graph_optimization_level_reported'], 'ORT_ENABLE_ALL')

    def test_full_context_parity_length_fails_before_writing_output(self):
        import owned_student
        from safetensors.torch import save_file
        import export_owned_cached_onnx as subject

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            architecture = {'width': 8, 'layers': 2, 'heads': 2, 'conv_kernel': 3,
                            'context': 4, 'ff_multiplier': 2, 'attention_every': 2}
            weights = checkpoint / 'weights.safetensors'
            save_file(owned_student.OwnedByteStudent(architecture).state_dict(), str(weights))
            sha = lambda raw: hashlib.sha256(raw).hexdigest()
            manifest = {'architecture': architecture, 'weights_sha256': sha(weights.read_bytes()),
                        'code_sha256': {'owned_student.py': sha(Path(owned_student.__file__).read_bytes())}}
            (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
            config = json.loads(Path(subject.__file__).with_name('owned-byte-cached-onnx-export.example.json').read_text())
            config['parity_lengths'] = [1, 4]
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            source = root / 'source.bin'
            source.write_bytes(b'abcd')
            output = root / 'out'
            with self.assertRaisesRegex(ValueError, 'parity lengths below context'):
                subject.export(checkpoint, config_path, source, output, decode_steps=1, warmup=0, repeats=1)
            self.assertFalse(output.exists())
            config['parity_lengths'] = [1]
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'leave room for a decode step'):
                subject.export(checkpoint, config_path, source, output, decode_steps=1, warmup=0, repeats=1)
            self.assertFalse(output.exists())

    def test_invalid_recognized_graph_level_fails_before_output_creation(self):
        import export_owned_cached_onnx as subject

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(Path(subject.__file__).with_name(
                'owned-byte-cached-onnx-export.example.json').read_text())
            config['graph_optimization_level'] = 'ORT_ENABLE_SOMETHING_NEW'
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            output = root / 'must-not-be-created'
            with self.assertRaisesRegex(ValueError, 'invalid ONNX Runtime graph optimization level'):
                subject.export(root / 'missing-checkpoint', config_path, root / 'missing-source', output)
            self.assertFalse(output.exists())

    def test_invalid_failure_evidence_type_fails_before_output_creation(self):
        import export_owned_cached_onnx as subject

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(Path(subject.__file__).with_name(
                'owned-byte-cached-onnx-export.example.json').read_text())
            config['failure_evidence'] = 'enabled'
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            output = root / 'must-not-be-created'
            with self.assertRaisesRegex(ValueError, 'invalid cached export configuration'):
                subject.export(root / 'missing-checkpoint', config_path, root / 'missing-source', output)
            self.assertFalse(output.exists())


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'safetensors')),
                     'torch and safetensors required to import exporter')
class ORTConfigurationTests(unittest.TestCase):
    def test_each_supported_graph_optimization_level_and_omitted_default(self):
        import export_owned_cached_onnx as subject

        names = subject.ORT_GRAPH_OPTIMIZATION_LEVELS
        levels = {name: object() for name in names}

        class Options:
            def __init__(self):
                object.__setattr__(self, 'graph_optimization_level', 'library-default')
                object.__setattr__(self, 'assigned', [])

            def __setattr__(self, name, value):
                if name == 'graph_optimization_level':
                    self.assigned.append(value)
                object.__setattr__(self, name, value)

        fake_ort = SimpleNamespace(SessionOptions=Options,
                                   GraphOptimizationLevel=SimpleNamespace(**levels),
                                   __version__='synthetic-ort')
        default, requested = subject.create_ort_session_options(fake_ort, 3, {})
        self.assertEqual(requested, 'unspecified')
        self.assertEqual(default.graph_optimization_level, 'library-default')
        self.assertEqual(default.assigned, [])
        self.assertEqual((default.intra_op_num_threads, default.inter_op_num_threads), (3, 1))
        default_provenance = subject.ort_session_provenance(
            fake_ort, {'prefill': type('S', (), {'get_session_options': lambda self: default,
                                                 'get_providers': lambda self: ['CPUExecutionProvider']})(),
                       'decode': type('S', (), {'get_session_options': lambda self: default,
                                                'get_providers': lambda self: ['CPUExecutionProvider']})()},
            requested, 3, 'synthetic-default')
        self.assertTrue(all(row['effective_graph_optimization_level_reported'] == 'unknown'
                            for row in default_provenance['sessions']))

        for name, expected in levels.items():
            with self.subTest(level=name):
                options, requested = subject.create_ort_session_options(
                    fake_ort, 2, {'graph_optimization_level': name})
                self.assertEqual(requested, name)
                self.assertIs(options.graph_optimization_level, expected)

        class Session:
            def __init__(self, options, providers):
                self.options = options
                self.providers = providers

            def get_session_options(self):
                return self.options

            def get_providers(self):
                return self.providers

        selected_prefill, selected_request = subject.create_ort_session_options(
            fake_ort, 2, {'graph_optimization_level': 'ORT_ENABLE_BASIC'})
        selected_decode, decode_request = subject.create_ort_session_options(
            fake_ort, 2, {'graph_optimization_level': 'ORT_ENABLE_BASIC'})
        self.assertIsNot(selected_prefill, selected_decode)
        self.assertEqual(selected_prefill.graph_optimization_level, levels['ORT_ENABLE_BASIC'])
        self.assertEqual(selected_decode.graph_optimization_level, levels['ORT_ENABLE_BASIC'])
        self.assertEqual(selected_request, decode_request)
        provenance = subject.ort_session_provenance(
            fake_ort, {'prefill': Session(selected_prefill, ['CPUExecutionProvider']),
                       'decode': Session(selected_decode, ['CPUExecutionProvider'])},
            selected_request, 2, 'synthetic-export')
        self.assertEqual(provenance['onnxruntime_version'], 'synthetic-ort')
        self.assertEqual([row['effective_graph_optimization_level_reported']
                          for row in provenance['sessions']], ['ORT_ENABLE_BASIC', 'ORT_ENABLE_BASIC'])
        self.assertTrue(all(row['providers_reported'] == ['CPUExecutionProvider']
                            for row in provenance['sessions']))

    def test_invalid_recognized_level_is_rejected(self):
        import export_owned_cached_onnx as subject

        fake_ort = SimpleNamespace(SessionOptions=lambda: object())
        with self.assertRaisesRegex(ValueError, 'invalid ONNX Runtime graph optimization level'):
            subject.create_ort_session_options(
                fake_ort, 1, {'graph_optimization_level': 'ORT_ENABLE_SOMETHING_NEW'})


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'safetensors', 'numpy')),
                     'torch, safetensors and numpy required')
class CachedExportSafetyTests(unittest.TestCase):
    def test_near_tied_first_byte_must_match_independent_reference(self):
        import numpy as np
        import export_owned_cached_onnx as subject

        actual = np.array([[0.0, 1.0, 1.000001]], dtype=np.float32)
        reference = np.array([[0.0, 1.000001, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5)
        with self.assertRaisesRegex(ValueError, 'greedy output differs'):
            subject.matching_greedy_byte(actual, reference)
        self.assertEqual(subject.matching_greedy_byte(reference, reference), 1)

    def test_output_refusal_writes_no_private_bytes(self):
        import export_owned_cached_onnx as subject

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'export'
            if os.name == 'posix':
                root.chmod(0o755)
                with self.assertRaisesRegex(ValueError, 'parent must be owned.*private'):
                    subject.create_private_output(output)
            else:
                with self.assertRaisesRegex(ValueError, 'verified private ACL'):
                    subject.create_private_output(output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.iterdir()), [])

    @unittest.skipUnless(os.name == 'posix', 'POSIX permission and symlink test')
    def test_private_output_mode_and_symlink_ancestor_refusal(self):
        import export_owned_cached_onnx as subject

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'export'
            subject.create_private_output(output)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            link = root / 'linked'
            link.symlink_to(output, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'non-symlink directories'):
                subject.create_private_output(link / 'nested')
            self.assertFalse((output / 'nested').exists())


@unittest.skipUnless(os.name == 'posix', 'failure reports require verified private POSIX output')
@unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                         ('torch', 'safetensors', 'numpy', 'onnx', 'onnxruntime')),
                     'torch, safetensors, numpy, onnx and onnxruntime required')
class CachedExportFailureEvidenceTests(unittest.TestCase):
    @staticmethod
    def _identity(path):
        info = path.stat()
        return info.st_dev, info.st_ino

    def _prepare_synthetic_export(self, root, source_bytes=b'ABCD'):
        import owned_student
        from safetensors.torch import save_file
        import export_owned_cached_onnx as subject

        checkpoint = root / 'checkpoint'
        checkpoint.mkdir(mode=0o700)
        architecture = {'width': 8, 'layers': 2, 'heads': 2, 'conv_kernel': 3,
                        'context': 32, 'ff_multiplier': 2, 'attention_every': 2}
        model = owned_student.OwnedByteStudent(architecture).eval()
        weights = checkpoint / 'weights.safetensors'
        save_file(model.state_dict(), str(weights))
        sha = lambda raw: hashlib.sha256(raw).hexdigest()
        manifest = {'architecture': architecture, 'weights_sha256': sha(weights.read_bytes()),
                    'code_sha256': {'owned_student.py': sha(Path(owned_student.__file__).read_bytes())}}
        (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
        config = json.loads(Path(subject.__file__).with_name(
            'owned-byte-cached-onnx-export.example.json').read_text())
        config.update(failure_evidence=True, parity_lengths=[1])
        config_path = root / 'config.json'
        config_path.write_text(json.dumps(config))
        source = root / 'source.bin'
        source.write_bytes(source_bytes)
        return checkpoint, config_path, source

    def _write_noisy_sitecustomize(self, directory):
        (directory / 'sitecustomize.py').write_text(
            "import os\n"
            "import atexit\n"
            "import ctypes\n"
            "import torch\n"
            "import onnxruntime as ort\n"
            "_libc = ctypes.CDLL(None)\n"
            "_libc.printf.argtypes = [ctypes.c_char_p]\n"
            "_libc.printf.restype = ctypes.c_int\n"
            "_libc.fprintf.argtypes = [ctypes.c_void_p, ctypes.c_char_p]\n"
            "_libc.fprintf.restype = ctypes.c_int\n"
            "_libc.fflush.argtypes = [ctypes.c_void_p]\n"
            "_libc.fflush.restype = ctypes.c_int\n"
            "_libc.setvbuf.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]\n"
            "_libc.setvbuf.restype = ctypes.c_int\n"
            "_native_stderr = ctypes.c_void_p.in_dll(_libc, 'stderr')\n"
            "_native_stderr_buffer = ctypes.create_string_buffer(4096)\n"
            "assert _libc.setvbuf(_native_stderr, _native_stderr_buffer, 0, len(_native_stderr_buffer)) == 0\n"
            "def _flush_native_buffers_at_exit():\n"
            "    _libc.fflush(None)\n"
            "atexit.register(_flush_native_buffers_at_exit)\n"
            "_real_export = torch.onnx.export\n"
            "def _noisy_export(*args, **kwargs):\n"
            "    print('PYTHON_STREAM_SENTINEL')\n"
            "    print('TENSOR_VALUE_SENTINEL')\n"
            "    os.write(2, b'FD_STREAM_SENTINEL ARRAY_VALUE_SENTINEL\\n')\n"
            "    _libc.printf(b'C_STDOUT_BUFFERED_SENTINEL')\n"
            "    _libc.fprintf(_native_stderr, b'C_STDERR_BUFFERED_SENTINEL')\n"
            "    return _real_export(*args, **kwargs)\n"
            "torch.onnx.export = _noisy_export\n"
            "_real_session = ort.InferenceSession\n"
            "_session_count = [0]\n"
            "class _SessionProxy:\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        self.kind = 'prefill' if _session_count[0] == 0 else 'decode'\n"
            "        _session_count[0] += 1\n"
            "        self.inner = _real_session(*args, **kwargs)\n"
            "    def __getattr__(self, name):\n"
            "        return getattr(self.inner, name)\n"
            "    def run(self, *args, **kwargs):\n"
            "        outputs = self.inner.run(*args, **kwargs)\n"
            "        if os.environ.get('LFG_TEST_FAIL_PREFILL') == '1' and self.kind == 'prefill':\n"
            "            outputs[0] = outputs[0] + 1.0\n"
            "        return outputs\n"
            "ort.InferenceSession = _SessionProxy\n",
            encoding='utf-8')

    def _run_actual_cli(self, script, checkpoint, config, source, output, site_directory, fail=False):
        environment = os.environ.copy()
        prior_pythonpath = environment.get('PYTHONPATH')
        environment['PYTHONPATH'] = str(site_directory) + (
            os.pathsep + prior_pythonpath if prior_pythonpath else '')
        if fail:
            environment['LFG_TEST_FAIL_PREFILL'] = '1'
        else:
            environment.pop('LFG_TEST_FAIL_PREFILL', None)
        return subprocess.run(
            [sys.executable, '-W', 'always', str(script), '--checkpoint', str(checkpoint),
             '--config', str(config), '--source', str(source), '--output', str(output),
             '--decode-steps', '1', '--warmup', '0', '--repeats', '1'],
            capture_output=True, text=True, env=environment, timeout=120)

    def test_allclose_failure_is_first_only_and_writes_private_safe_context(self):
        import contextlib
        import io
        from unittest import mock
        import numpy as np
        import owned_student
        from safetensors.torch import save_file
        import export_owned_cached_onnx as subject

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir(mode=0o700)
            architecture = {'width': 8, 'layers': 2, 'heads': 2, 'conv_kernel': 3,
                            'context': 32, 'ff_multiplier': 2, 'attention_every': 2}
            model = owned_student.OwnedByteStudent(architecture).eval()
            weights = checkpoint / 'weights.safetensors'
            save_file(model.state_dict(), str(weights))
            sha = lambda raw: hashlib.sha256(raw).hexdigest()
            manifest = {'architecture': architecture, 'weights_sha256': sha(weights.read_bytes()),
                        'code_sha256': {'owned_student.py': sha(Path(owned_student.__file__).read_bytes())}}
            (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
            config = json.loads(Path(subject.__file__).with_name(
                'owned-byte-cached-onnx-export.example.json').read_text())
            config.update(failure_evidence=True, parity_lengths=[1])
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            source = root / 'source.bin'
            source.write_bytes(b'RAW_SENTINEL')
            output = root / 'out'

            original_prefill = subject.PrefillGraph

            class MismatchingPrefill(original_prefill):
                def forward(self, tokens):
                    outputs = super().forward(tokens)
                    return (outputs[0] + 1.0, *outputs[1:])

            stdout, stderr = io.StringIO(), io.StringIO()
            with mock.patch.object(subject, 'PrefillGraph', MismatchingPrefill):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(subject.CachedParityFailure) as caught:
                        subject.export(checkpoint, config_path, source, output,
                                       decode_steps=1, warmup=0, repeats=1)

            failure = caught.exception
            self.assertIsInstance(failure, AssertionError)
            self.assertTrue(failure.failure_report_written)
            self.assertEqual(failure.failure_report_error_category, 'written')
            self.assertEqual(str(failure), 'cached ONNX strict parity assertion failed')
            self.assertIsNone(failure.__context__)
            self.assertEqual(stdout.getvalue(), '')
            # Torch's legacy ONNX exporter may emit deprecation and constant
            # folding warnings; only assert that private inputs and failure
            # details are absent from that diagnostic stream.
            self.assertNotIn('RAW_SENTINEL', stderr.getvalue())
            self.assertNotIn(str(source), stderr.getvalue())
            self.assertNotIn('strict parity assertion failed', stderr.getvalue())
            self.assertTrue(output.is_dir())
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            report_path = output / 'failure-report.json'
            self.assertTrue(report_path.is_file())
            self.assertEqual(stat.S_IMODE(report_path.stat().st_mode), 0o600)
            raw_report = report_path.read_bytes()
            self.assertLessEqual(len(raw_report), subject.MAX_FAILURE_REPORT_BYTES)
            report = json.loads(raw_report, parse_constant=lambda _value: self.fail('non-strict JSON'))
            self.assertEqual(report['schema_version'], 'lfg.owned-cached-export-failure.v1')
            self.assertEqual(report['failure_kind'], 'numpy_assert_allclose')
            self.assertTrue(report['first_failure_only'])
            self.assertFalse(report['later_checks_run'])
            self.assertFalse(report['success_manifest_written'])
            self.assertFalse(report['measurement_written'])
            self.assertFalse(report['promotion_eligible'])
            self.assertEqual(report['context']['phase'], 'configured_length_prefill')
            self.assertEqual(report['context']['comparison'], 'onnx_prefill_vs_full_prefix_logits')
            self.assertEqual(report['context']['source_ordinal'], 1)
            self.assertEqual(report['context']['prefix_length'], 1)
            self.assertEqual(report['tolerance'], {'absolute': 1e-5, 'relative': 1e-5, 'equal_nan': True})
            self.assertEqual(report['comparison']['actual_dtype'], 'float32')
            self.assertEqual(report['comparison']['desired_dtype'], 'float32')
            self.assertGreater(report['comparison']['failed_element_count'], 0)
            self.assertFalse((output / 'manifest.json').exists())
            self.assertFalse((output / 'measurement.json').exists())
            encoded = raw_report.decode('utf-8')
            self.assertNotIn('RAW_SENTINEL', encoded)
            self.assertNotIn(str(source), encoded)
            self.assertNotIn('1.0,', encoded)

    def test_failure_contexts_cover_prefill_decode_logits_and_cache(self):
        import numpy as np
        import export_owned_cached_onnx as subject

        contexts = (
            {'phase': 'configured_length_prefill', 'comparison': 'onnx_prefill_vs_full_prefix_logits',
             'source_ordinal': 1, 'prefix_length': 1},
            {'phase': 'full_source_prefill', 'comparison': 'onnx_prefill_vs_incremental_logits',
             'source_ordinal': 2, 'prefix_length': 7},
            {'phase': 'autoregressive_decode', 'comparison': 'onnx_decode_vs_full_prefix_logits',
             'source_ordinal': 1, 'prefix_length': 7, 'decode_step': 2},
            {'phase': 'autoregressive_decode', 'comparison': 'onnx_decode_vs_incremental_cache',
             'source_ordinal': 1, 'prefix_length': 7, 'decode_step': 2,
             'state_name': 'key_1', 'state_kind': 'attention-key', 'state_layer': 1},
        )
        for index, context in enumerate(contexts):
            with self.subTest(context=context), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'private-output'
                output.mkdir(mode=0o700)
                actual = np.array([[2.0, 4.0]], dtype=np.float32)
                desired = np.array([[2.0, 3.0]], dtype=np.float32)
                with self.assertRaises(subject.CachedParityFailure) as caught:
                    subject._assert_allclose_with_evidence(
                        np, actual, desired, atol=1e-5, rtol=1e-5,
                        context=context, output=output, report_enabled=True,
                        output_identity=self._identity(output))
                self.assertTrue(caught.exception.failure_report_written)
                report = json.loads((output / 'failure-report.json').read_bytes())
                self.assertEqual(report['context']['phase'], context['phase'])
                self.assertEqual(report['context']['comparison'], context['comparison'])
                self.assertEqual(report['context'].get('decode_step'), context.get('decode_step'))
                self.assertEqual(report['context'].get('state_name'), context.get('state_name'))
                self.assertEqual(report['comparison']['failed_element_count'], 1)

    def test_actual_export_call_sites_report_full_source_and_decode_failures(self):
        import onnxruntime as ort
        from unittest import mock
        import export_owned_cached_onnx as subject

        for injection in ('full_source', 'decode_logits', 'decode_cache'):
            with self.subTest(injection=injection), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint, config_path, source = self._prepare_synthetic_export(root)
                output = root / 'output'
                additional_sources = []
                if injection == 'decode_logits':
                    second_source = root / 'source-second.bin'
                    second_source.write_bytes(b'GHIJKL')
                    additional_sources.append(second_source)
                real_session = ort.InferenceSession
                session_count = 0

                class InjectedSession:
                    def __init__(self, *args, **kwargs):
                        nonlocal session_count
                        self.kind = 'prefill' if session_count == 0 else 'decode'
                        session_count += 1
                        self.inner = real_session(*args, **kwargs)

                    def __getattr__(self, name):
                        return getattr(self.inner, name)

                    def run(self, *args, **kwargs):
                        feeds = kwargs.get('input_feed', args[1] if len(args) > 1 else {})
                        outputs = self.inner.run(*args, **kwargs)
                        if (injection == 'full_source' and self.kind == 'prefill'
                                and feeds['tokens'].shape[1] == 4):
                            outputs[0] = outputs[0] + 1.0
                        elif (injection == 'decode_logits' and self.kind == 'decode'
                              and int(feeds['position']) == 6):
                            outputs[0] = outputs[0] + 1.0
                        elif injection == 'decode_cache' and self.kind == 'decode':
                            outputs[1] = outputs[1] + 1.0
                        return outputs

                with mock.patch.object(ort, 'InferenceSession', InjectedSession):
                    with self.assertRaises(subject.CachedParityFailure) as caught:
                        subject.export(checkpoint, config_path, source, output,
                                       additional_sources=additional_sources,
                                       decode_steps=1, warmup=0, repeats=1)

                self.assertTrue(caught.exception.failure_report_written)
                report = json.loads((output / 'failure-report.json').read_bytes())
                self.assertTrue(report['first_failure_only'])
                self.assertFalse(report['later_checks_run'])
                self.assertFalse(report['success_manifest_written'])
                self.assertFalse(report['measurement_written'])
                self.assertFalse((output / 'manifest.json').exists())
                self.assertFalse((output / 'measurement.json').exists())
                context = report['context']
                expected_source_ordinal = 2 if injection == 'decode_logits' else 1
                self.assertEqual(context['source_ordinal'], expected_source_ordinal)
                if injection == 'full_source':
                    self.assertEqual(context['phase'], 'full_source_prefill')
                    self.assertEqual(context['comparison'], 'onnx_prefill_vs_full_prefix_logits')
                    self.assertEqual(context['prefix_length'], 4)
                    self.assertNotIn('decode_step', context)
                    self.assertNotIn('state_name', context)
                elif injection == 'decode_logits':
                    self.assertEqual(context['phase'], 'autoregressive_decode')
                    self.assertEqual(context['comparison'], 'onnx_decode_vs_full_prefix_logits')
                    self.assertEqual(context['prefix_length'], 6)
                    self.assertEqual(context['decode_step'], 1)
                    self.assertNotIn('state_name', context)
                else:
                    self.assertEqual(context['phase'], 'autoregressive_decode')
                    self.assertEqual(context['comparison'], 'onnx_decode_vs_incremental_cache')
                    self.assertEqual(context['prefix_length'], 4)
                    self.assertEqual(context['decode_step'], 1)
                    self.assertEqual(context['state_name'], 'conv_0')
                    self.assertEqual(context['state_kind'], 'depthwise-convolution-history')
                    self.assertEqual(context['state_layer'], 0)

    def test_nonfinite_or_shape_guard_failure_is_not_misreported_as_assertion(self):
        import onnxruntime as ort
        import numpy as np
        from unittest import mock
        import export_owned_cached_onnx as subject

        for invalid_kind in ('shape', 'nonfinite'):
            with self.subTest(invalid_kind=invalid_kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint, config_path, source = self._prepare_synthetic_export(root)
                output = root / 'output'
                real_session = ort.InferenceSession

                class InvalidStateSession:
                    def __init__(self, *args, **kwargs):
                        self.inner = real_session(*args, **kwargs)

                    def __getattr__(self, name):
                        return getattr(self.inner, name)

                    def run(self, *args, **kwargs):
                        outputs = self.inner.run(*args, **kwargs)
                        feeds = kwargs.get('input_feed', args[1] if len(args) > 1 else {})
                        if 'tokens' in feeds and feeds['tokens'].shape[1] == 1:
                            if invalid_kind == 'shape':
                                outputs[1] = outputs[1][..., :1]
                            else:
                                outputs[1] = np.full_like(outputs[1], np.nan)
                        return outputs

                with mock.patch.object(ort, 'InferenceSession', InvalidStateSession):
                    with self.assertRaisesRegex(ValueError, 'invalid parity cache state'):
                        subject.export(checkpoint, config_path, source, output,
                                       decode_steps=1, warmup=0, repeats=1)
                self.assertFalse((output / 'failure-report.json').exists())
                self.assertFalse((output / 'manifest.json').exists())
                self.assertFalse((output / 'measurement.json').exists())

    def test_summary_preserves_original_dtype_nan_shape_scalar_and_extreme_semantics(self):
        import numpy as np
        import export_owned_cached_onnx as subject

        actual = np.array([np.nan, np.inf, 0.0, 1.0], dtype=np.float32)
        desired = np.array([np.nan, np.inf, 1.0, 1.01], dtype=np.float32)
        summary = subject._safe_array_summary(np, actual, desired, 1e-5, 1e-5)
        expected_close = np.isclose(actual, desired, atol=1e-5, rtol=1e-5, equal_nan=True)
        self.assertEqual(summary['failed_element_count'], int(actual.size - np.count_nonzero(expected_close)))
        self.assertEqual(summary['nonfinite_element_count'], 2)
        self.assertEqual(summary['zero_reference_count'], 0)
        self.assertEqual(summary['actual_dtype'], 'float32')

        zero_reference = subject._safe_array_summary(
            np, np.array([1e-12, 0.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64), 0.0, 0.0)
        self.assertEqual(zero_reference['zero_reference_count'], 2)
        self.assertEqual(zero_reference['failed_element_count'], 1)
        self.assertIsNone(zero_reference['max_relative_error_nonzero_reference'])

        empty_summary = subject._safe_array_summary(
            np, np.array([], dtype=np.float32), np.array([], dtype=np.float32), 1e-5, 1e-5)
        self.assertEqual(empty_summary['element_count'], 0)
        self.assertEqual(empty_summary['failed_element_count'], 0)
        self.assertIsNone(empty_summary['max_absolute_error'])

        shape_summary = subject._safe_array_summary(
            np, np.array([1.0, 2.0]), np.array([[1.0, 2.0]]), 1e-5, 1e-5)
        self.assertIsNone(shape_summary['failed_element_count'])
        self.assertEqual(shape_summary['aggregate_reason'], 'shape_mismatch_element_counts_unavailable')

        scalar_summary = subject._safe_array_summary(np, np.array(1.0), np.array(2.0), 1e-5, 1e-5)
        self.assertEqual(scalar_summary['actual_shape'], [])
        self.assertEqual(scalar_summary['element_count'], 1)
        self.assertEqual(scalar_summary['failed_element_count'], 1)

        maximum = np.finfo(np.float64).max
        extreme_summary = subject._safe_array_summary(
            np, np.array([maximum]), np.array([-maximum]), 0.0, 0.0)
        self.assertEqual(extreme_summary['failed_element_count'], 1)
        self.assertIsNone(extreme_summary['max_absolute_error'])
        self.assertEqual(extreme_summary['aggregate_reason'], 'finite_difference_unrepresentable')

        complex_max = np.finfo(np.float64).max
        complex_summary = subject._safe_array_summary(
            np,
            np.array([complex(complex_max * 0.75, complex_max * 0.75)], dtype=np.complex128),
            np.array([complex(complex_max, complex_max)], dtype=np.complex128),
            0.0, 0.0)
        self.assertEqual(complex_summary['actual_dtype'], 'complex128')
        self.assertAlmostEqual(complex_summary['max_relative_error_nonzero_reference'], 0.25)
        self.assertNotEqual(complex_summary['max_relative_error_nonzero_reference'], 0.0)
        self.assertNotIn(complex_summary['aggregate_reason'],
                         ('relative_error_unrepresentable', 'safe_aggregation_unavailable'))

        complex_normal = subject._safe_array_summary(
            np, np.array([1 + 2j], dtype=np.complex128),
            np.array([1 + 1j], dtype=np.complex128), 0.0, 0.0)
        self.assertAlmostEqual(complex_normal['max_relative_error_nonzero_reference'], 1 / 2**0.5)

    def test_original_allclose_operands_and_tolerances_are_unchanged(self):
        import numpy as np
        import export_owned_cached_onnx as subject
        from unittest import mock

        actual = np.array([1.0], dtype=np.float32)
        desired = np.array([1.0001], dtype=np.float32)
        original = np.testing.assert_allclose
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with mock.patch.object(np.testing, 'assert_allclose', wraps=original) as assertion:
                with self.assertRaises(subject.CachedParityFailure):
                    subject._assert_allclose_with_evidence(
                        np, actual, desired, atol=1e-5, rtol=1e-5,
                        context={'phase': 'configured_length_prefill',
                                 'comparison': 'onnx_prefill_vs_full_prefix_logits',
                                 'source_ordinal': 1, 'prefix_length': 1},
                        output=output, report_enabled=False,
                        output_identity=self._identity(output))
            args, kwargs = assertion.call_args
            self.assertIs(args[0], actual)
            self.assertIs(args[1], desired)
            self.assertEqual(kwargs, {'rtol': 1e-5, 'atol': 1e-5})

            with mock.patch.object(np.testing, 'assert_allclose',
                                   side_effect=AssertionError('EXCEPTION_TEXT_SENTINEL')):
                with self.assertRaises(subject.CachedParityFailure) as caught:
                    subject._assert_allclose_with_evidence(
                        np, actual, desired, atol=1e-5, rtol=1e-5,
                        context={'phase': 'configured_length_prefill',
                                 'comparison': 'onnx_prefill_vs_full_prefix_logits',
                                 'source_ordinal': 1, 'prefix_length': 1,
                                 'state_name': 'PRIVATE_PATH_SENTINEL'},
                        output=output, report_enabled=True,
                        output_identity=self._identity(output))
            self.assertNotIn('EXCEPTION_TEXT_SENTINEL', str(caught.exception))
            self.assertNotIn('PRIVATE_PATH_SENTINEL', json.dumps(caught.exception.failure_context))
            report_text = (output / 'failure-report.json').read_text()
            self.assertNotIn('EXCEPTION_TEXT_SENTINEL', report_text)
            self.assertNotIn('PRIVATE_PATH_SENTINEL', report_text)

            with mock.patch.object(subject, '_write_failure_report',
                                   side_effect=RuntimeError('REPORT_PATH_SENTINEL')):
                with self.assertRaises(subject.CachedParityFailure) as caught:
                    subject._assert_allclose_with_evidence(
                        np, actual, desired, atol=1e-5, rtol=1e-5,
                        context={'phase': 'configured_length_prefill',
                                 'comparison': 'onnx_prefill_vs_full_prefix_logits',
                                 'source_ordinal': 1, 'prefix_length': 1},
                        output=output, report_enabled=True,
                        output_identity=self._identity(output))
            self.assertFalse(caught.exception.failure_report_written)
            self.assertEqual(caught.exception.failure_report_error_category, 'report_write_failed')
            self.assertNotIn('REPORT_PATH_SENTINEL', str(caught.exception))

    def test_disabled_existing_sidecar_symlink_and_serialization_fail_closed(self):
        import numpy as np
        from unittest import mock
        import export_owned_cached_onnx as subject

        actual = np.array([2.0], dtype=np.float32)
        desired = np.array([1.0], dtype=np.float32)
        context = {'phase': 'full_source_prefill',
                   'comparison': 'onnx_prefill_vs_full_prefix_logits',
                   'source_ordinal': 1, 'prefix_length': 1,
                   'private_path': 'PRIVATE_PATH_SENTINEL'}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'disabled'
            output.mkdir(mode=0o700)
            with self.assertRaises(subject.CachedParityFailure) as caught:
                subject._assert_allclose_with_evidence(
                    np, actual, desired, atol=1e-5, rtol=1e-5, context=context,
                    output=output, report_enabled=False,
                    output_identity=self._identity(output))
            self.assertFalse(caught.exception.failure_report_written)
            self.assertEqual(caught.exception.failure_report_error_category, 'disabled')
            self.assertFalse((output / 'failure-report.json').exists())
            safe = subject._safe_failure_context(context)
            self.assertNotIn('private_path', safe)
            self.assertNotIn('PRIVATE_PATH_SENTINEL', json.dumps(safe))

            existing = Path(directory) / 'existing'
            existing.mkdir(mode=0o700)
            report = existing / 'failure-report.json'
            report.write_bytes(b'earlier-evidence')
            written, category = subject._write_failure_report(
                existing, {'schema_version': 'test'}, self._identity(existing))
            self.assertFalse(written)
            self.assertEqual(category, 'report_already_exists')
            self.assertEqual(report.read_bytes(), b'earlier-evidence')
            with self.assertRaises(subject.CachedParityFailure) as caught:
                subject._assert_allclose_with_evidence(
                    np, actual, desired, atol=1e-5, rtol=1e-5, context=context,
                    output=existing, report_enabled=True,
                    output_identity=self._identity(existing))
            self.assertFalse(caught.exception.failure_report_written)
            self.assertEqual(caught.exception.failure_report_error_category, 'report_already_exists')
            self.assertEqual(report.read_bytes(), b'earlier-evidence')

            linked = Path(directory) / 'linked'
            linked.mkdir(mode=0o700)
            target = Path(directory) / 'target'
            target.write_bytes(b'target-content')
            (linked / 'failure-report.json').symlink_to(target)
            written, category = subject._write_failure_report(
                linked, {'schema_version': 'test'}, self._identity(linked))
            self.assertFalse(written)
            self.assertEqual(category, 'report_already_exists')
            self.assertEqual(target.read_bytes(), b'target-content')
            with self.assertRaises(subject.CachedParityFailure) as caught:
                subject._assert_allclose_with_evidence(
                    np, actual, desired, atol=1e-5, rtol=1e-5, context=context,
                    output=linked, report_enabled=True,
                    output_identity=self._identity(linked))
            self.assertFalse(caught.exception.failure_report_written)
            self.assertEqual(caught.exception.failure_report_error_category, 'report_already_exists')
            self.assertEqual(target.read_bytes(), b'target-content')

            self.assertEqual(subject._write_failure_report(
                existing, {'bad': float('nan')}, self._identity(existing)),
                             (False, 'serialization_failed'))
            oversized = {'value': 'x' * (subject.MAX_FAILURE_REPORT_BYTES + 1)}
            self.assertEqual(subject._write_failure_report(
                Path(directory), oversized, self._identity(Path(directory))),
                             (False, 'report_size_limit'))

            write_failure = Path(directory) / 'write-failure'
            write_failure.mkdir(mode=0o700)
            with mock.patch.object(subject.os, 'write', side_effect=OSError('WRITE_SENTINEL')):
                self.assertEqual(subject._write_failure_report(
                    write_failure, {'schema_version': 'synthetic'}, self._identity(write_failure)),
                    (False, 'report_durability_uncertain'))
            partial = write_failure / 'failure-report.json'
            self.assertTrue(partial.is_file())
            self.assertEqual(partial.read_bytes(), b'')
            self.assertEqual(stat.S_IMODE(partial.stat().st_mode), 0o600)

    def test_output_replacement_before_open_never_writes_elsewhere(self):
        import export_owned_cached_onnx as subject

        for replacement_kind in ('symlink', 'private_directory'):
            with self.subTest(replacement_kind=replacement_kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / 'output'
                output.mkdir(mode=0o700)
                (output / 'keep.txt').write_bytes(b'original evidence')
                identity = self._identity(output)
                original = root / 'original-output'
                output.rename(original)
                external = root / 'external'
                external.mkdir(mode=0o700)
                if replacement_kind == 'symlink':
                    output.symlink_to(external, target_is_directory=True)
                else:
                    output.mkdir(mode=0o700)
                written, category = subject._write_failure_report(
                    output, {'schema_version': 'synthetic'}, identity)
                self.assertFalse(written)
                self.assertIn(category, ('unsafe_private_directory', 'output_directory_changed'))
                self.assertFalse((external / 'failure-report.json').exists())
                self.assertFalse((original / 'failure-report.json').exists())
                self.assertEqual((original / 'keep.txt').read_bytes(), b'original evidence')
                self.assertEqual([path.name for path in original.iterdir()], ['keep.txt'])
                if replacement_kind == 'private_directory':
                    self.assertFalse((output / 'failure-report.json').exists())

    def test_competing_report_replacement_is_never_unlinked_by_name(self):
        import numpy as np
        from unittest import mock
        import export_owned_cached_onnx as subject

        for phase in ('report_only', 'file_fsync', 'directory_fsync'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / 'output'
                output.mkdir(mode=0o700)
                (output / 'keep.txt').write_bytes(b'original evidence')
                identity = self._identity(output)
                original = root / 'original-output'
                external = root / 'external'
                external.mkdir(mode=0o700)

                def substitute_path():
                    output.rename(original)
                    if phase == 'file_fsync':
                        output.mkdir(mode=0o700)
                    elif phase == 'directory_fsync':
                        output.symlink_to(external, target_is_directory=True)

                def replace_report_with_competing_sentinel(directory_fd):
                    os.unlink('failure-report.json', dir_fd=directory_fd)
                    descriptor = os.open(
                        'failure-report.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600, dir_fd=directory_fd)
                    try:
                        os.write(descriptor, b'COMPETING_REPORT_SENTINEL')
                    finally:
                        os.close(descriptor)

                real_fsync = os.fsync
                original_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                changed = False

                def swapping_fsync(descriptor):
                    nonlocal changed
                    is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
                    should_replace = (
                        phase == 'report_only' and not is_directory
                        or phase == 'file_fsync' and not is_directory
                        or phase == 'directory_fsync' and is_directory)
                    if should_replace and not changed:
                        if phase != 'report_only':
                            substitute_path()
                        replace_report_with_competing_sentinel(
                            descriptor if is_directory else original_fd)
                        changed = True
                    return real_fsync(descriptor)

                patcher = mock.patch.object(subject.os, 'fsync', side_effect=swapping_fsync)
                with patcher:
                    with self.assertRaises(subject.CachedParityFailure) as caught:
                        subject._assert_allclose_with_evidence(
                            np, np.array([2.0], dtype=np.float32),
                            np.array([1.0], dtype=np.float32), atol=0.0, rtol=0.0,
                            context={'phase': 'configured_length_prefill',
                                     'comparison': 'onnx_prefill_vs_full_prefix_logits',
                                     'source_ordinal': 1, 'prefix_length': 1},
                            output=output, report_enabled=True, output_identity=identity)
                os.close(original_fd)
                report_directory = output if phase == 'report_only' else original
                self.assertIsInstance(caught.exception, AssertionError)
                self.assertFalse(caught.exception.failure_report_written)
                self.assertEqual(caught.exception.failure_report_error_category,
                                 'report_replaced' if phase == 'report_only'
                                 else 'output_directory_changed')
                self.assertEqual(str(caught.exception),
                                 'cached ONNX strict parity assertion failed')
                self.assertFalse((external / 'failure-report.json').exists())
                self.assertEqual((report_directory / 'failure-report.json').read_bytes(),
                                 b'COMPETING_REPORT_SENTINEL')
                self.assertEqual((report_directory / 'keep.txt').read_bytes(), b'original evidence')
                self.assertEqual(set(path.name for path in report_directory.iterdir()),
                                 {'keep.txt', 'failure-report.json'})
                self.assertFalse((report_directory / 'manifest.json').exists())
                self.assertFalse((report_directory / 'measurement.json').exists())
                if phase == 'file_fsync':
                    self.assertEqual(list(output.iterdir()), [])

    def test_cli_failure_output_is_sanitized(self):
        import contextlib
        import io
        from unittest import mock
        import export_owned_cached_onnx as subject

        failure = subject.CachedParityFailure(False, 'disabled',
                                              {'phase': 'configured_length_prefill'}, None)
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(subject, 'export', side_effect=failure):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = subject.main(['--checkpoint', 'PRIVATE_PATH_SENTINEL', '--config', 'c',
                                       '--source', 's', '--output', 'o'])
        self.assertEqual(result, 1)
        self.assertEqual(stdout.getvalue(), '')
        self.assertNotIn('PRIVATE_PATH_SENTINEL', stderr.getvalue())
        self.assertNotIn('assert_allclose', stderr.getvalue())
        self.assertEqual(json.loads(stderr.getvalue()), {
            'status': 'strict_parity_failed', 'failure_report_written': False,
            'failure_report_error_category': 'disabled'})

    def test_greedy_mismatch_remains_rejected_without_assertion_sidecar_claim(self):
        import contextlib
        import io
        from unittest import mock
        import export_owned_cached_onnx as subject

        if importlib.util.find_spec('numpy'):
            import numpy as np
            with self.assertRaisesRegex(
                    ValueError,
                    'cached ONNX greedy output differs from independent full-prefix output'):
                subject.matching_greedy_byte(
                    np.array([[0.0, 2.0]], dtype=np.float32),
                    np.array([[3.0, 0.0]], dtype=np.float32))

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(subject, 'export', side_effect=ValueError('GREEDY_SENTINEL')):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = subject.main(['--checkpoint', 'synthetic-checkpoint', '--config', 'c',
                                       '--source', 's', '--output', 'synthetic-output'])
        self.assertEqual(result, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertNotIn('GREEDY_SENTINEL', stderr.getvalue())
        self.assertNotIn('failure_report_written', stderr.getvalue())
        self.assertEqual(json.loads(stderr.getvalue()), {
            'status': 'invalid_or_unsafe_export', 'failure_detail_suppressed': True})

    def test_real_cli_discards_python_and_native_warning_output_on_success_and_failure(self):
        import export_owned_cached_onnx as subject

        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory(
                    prefix='PRIVATE_PATH_SENTINEL_') as directory:
                root = Path(directory)
                checkpoint, config_path, source = self._prepare_synthetic_export(
                    root, source_bytes=b'TOKEN_SENTINEL')
                output = root / 'output'
                site_directory = root / 'site'
                site_directory.mkdir(mode=0o700)
                self._write_noisy_sitecustomize(site_directory)
                result = self._run_actual_cli(
                    subject.__file__, checkpoint, config_path, source, output,
                    site_directory, fail=fail)
                combined = result.stdout + result.stderr
                for sentinel in ('PRIVATE_PATH_SENTINEL', 'TOKEN_SENTINEL',
                                 'PYTHON_STREAM_SENTINEL', 'FD_STREAM_SENTINEL',
                                 'C_STDOUT_BUFFERED_SENTINEL',
                                 'C_STDERR_BUFFERED_SENTINEL',
                                 'TENSOR_VALUE_SENTINEL', 'ARRAY_VALUE_SENTINEL',
                                 'Traceback', str(root),
                                 str(Path(subject.__file__).resolve().parent)):
                    self.assertNotIn(sentinel, combined)
                if fail:
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, '')
                    status = json.loads(result.stderr)
                    self.assertEqual(status['status'], 'strict_parity_failed')
                    self.assertTrue(status['failure_report_written'])
                    report = (output / 'failure-report.json').read_text(encoding='utf-8')
                    self.assertNotIn('PRIVATE_PATH_SENTINEL', report)
                    self.assertNotIn('TOKEN_SENTINEL', report)
                    self.assertFalse((output / 'manifest.json').exists())
                    self.assertFalse((output / 'measurement.json').exists())
                else:
                    self.assertEqual(result.returncode, 0)
                    status = json.loads(result.stdout)
                    self.assertEqual(status['status'], 'verified')
                    self.assertEqual(result.stderr, '')

    def test_library_main_preserves_caller_streams_and_fds(self):
        import contextlib
        import io
        import export_owned_cached_onnx as subject
        from unittest import mock

        original_stdout, original_stderr = sys.stdout, sys.stderr
        try:
            before = tuple((os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in (1, 2))
        except OSError:
            self.skipTest('stdout/stderr file descriptors are unavailable')
        stdout, stderr = io.StringIO(), io.StringIO()
        arguments = ['--checkpoint', 'checkpoint', '--config', 'config', '--source',
                     'source', '--output', 'output']
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                mock.patch.object(subject, 'export', return_value={'status': 'verified'}):
            self.assertEqual(subject.main(arguments, sanitize_process_output=True), 0)
        self.assertIs(sys.stdout, original_stdout)
        self.assertIs(sys.stderr, original_stderr)
        self.assertEqual(json.loads(stdout.getvalue()), {'status': 'verified'})
        self.assertEqual(stderr.getvalue(), '')
        after = tuple((os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in (1, 2))
        self.assertEqual(after, before)


class SanitizedExporterStartupTests(unittest.TestCase):
    def _run_cli(self, script, arguments, pythonpath=None):
        environment = os.environ.copy()
        if pythonpath is not None:
            previous = environment.get('PYTHONPATH')
            environment['PYTHONPATH'] = str(pythonpath) + (os.pathsep + previous if previous else '')
        return subprocess.run([sys.executable, str(script), *arguments],
                              capture_output=True, text=True, env=environment, timeout=30)

    def test_argument_errors_are_fixed_json_without_usage_values(self):
        import export_owned_cached_onnx as subject

        sentinel = 'ARGUMENT_PATH_SENTINEL'
        cases = (
            ['--unknown', sentinel],
            ['--checkpoint', sentinel, '--config', 'c', '--source', 's', '--output', 'o',
             '--decode-steps', sentinel],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self._run_cli(subject.__file__, arguments)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn(sentinel, result.stdout + result.stderr)
                self.assertNotIn('Traceback', result.stdout + result.stderr)
                self.assertEqual(json.loads(result.stderr), {
                    'status': 'invalid_arguments', 'failure_detail_suppressed': True})

    def test_help_is_fixed_useful_and_independent_of_private_script_path_or_dependencies(self):
        import export_owned_cached_onnx as subject

        with tempfile.TemporaryDirectory(prefix='PRIVATE_SCRIPT_PATH_SENTINEL_') as directory:
            root = Path(directory)
            script = root / 'PRIVATE_SCRIPT_PATH_SENTINEL.py'
            script.write_bytes(Path(subject.__file__).read_bytes())
            blocked_dependencies = root / 'blocked'
            blocked_dependencies.mkdir()
            (blocked_dependencies / 'torch.py').write_text(
                "raise ModuleNotFoundError('PRIVATE_DEPENDENCY_PATH_SENTINEL')\n",
                encoding='utf-8')
            result = self._run_cli(script, ['--help'], pythonpath=blocked_dependencies)
            self.assertEqual(result.returncode, 0)
            self.assertIn('usage: lfg-export-owned-cached-onnx', result.stdout)
            for option in ('--checkpoint', '--config', '--source', '--output',
                           '--additional-source', '--decode-steps', '--warmup', '--repeats'):
                self.assertIn(option, result.stdout)
            self.assertEqual(result.stderr, '')
            for sentinel in ('PRIVATE_SCRIPT_PATH_SENTINEL',
                             'PRIVATE_DEPENDENCY_PATH_SENTINEL', str(root), 'Traceback'):
                self.assertNotIn(sentinel, result.stdout + result.stderr)

    @unittest.skipUnless(os.name == 'posix', 'closing a pipe reader is POSIX-specific')
    def test_help_returns_nonzero_when_caller_closes_status_reader(self):
        import export_owned_cached_onnx as subject

        process = subprocess.Popen(
            [sys.executable, str(subject.__file__), '--help'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process.stdout.close()
        stderr = process.stderr.read()
        process.stderr.close()
        returncode = process.wait(timeout=30)
        self.assertEqual(returncode, 7)
        self.assertEqual(stderr, '')

    def test_boundary_setup_failure_precedes_help_and_is_nonzero(self):
        import export_owned_cached_onnx as subject

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sitecustomize.py').write_text(
                "import os\n"
                "def _fail_dup2(*args, **kwargs):\n"
                "    raise OSError('PRIVATE_BOUNDARY_SENTINEL')\n"
                "os.dup2 = _fail_dup2\n",
                encoding='utf-8')
            result = self._run_cli(subject.__file__, ['--help'], pythonpath=root)
            self.assertEqual(result.returncode, 6)
            self.assertEqual(result.stdout, '')
            self.assertEqual(json.loads(result.stderr), {
                'status': 'output_sanitization_unavailable',
                'failure_detail_suppressed': True})
            self.assertNotIn('PRIVATE_BOUNDARY_SENTINEL', result.stderr)

    def test_status_writer_handles_short_interrupted_zero_and_broken_writes(self):
        import export_owned_cached_onnx as subject
        from unittest import mock

        expected = (json.dumps({'status': 'verified'}, separators=(',', ':'),
                               allow_nan=False) + '\n').encode('utf-8')
        chunks = []
        attempts = [0]

        def short_write(descriptor, payload):
            attempts[0] += 1
            if attempts[0] == 1:
                raise InterruptedError()
            chunk = payload[:2]
            chunks.append(chunk)
            return len(chunk)

        with mock.patch.object(subject, '_CLI_MODE', True), \
                mock.patch.object(subject, '_CLI_STATUS_STDOUT_FD', 91), \
                mock.patch.object(subject.os, 'write', side_effect=short_write):
            self.assertTrue(subject._emit_status({'status': 'verified'}))
        self.assertEqual(b''.join(chunks), expected)

        for write_failure in (lambda _fd, _data: 0,
                              lambda _fd, _data: (_ for _ in ()).throw(BrokenPipeError())):
            with self.subTest(write_failure=write_failure), \
                    mock.patch.object(subject, '_CLI_MODE', True), \
                    mock.patch.object(subject, '_CLI_STATUS_STDOUT_FD', 91), \
                    mock.patch.object(subject.os, 'write', side_effect=write_failure):
                self.assertFalse(subject._emit_status({'status': 'verified'}))

    def test_error_status_falls_back_to_remaining_saved_channel(self):
        import export_owned_cached_onnx as subject
        from unittest import mock

        writes = []
        with mock.patch.object(subject, '_CLI_MODE', True), \
                mock.patch.object(subject, '_CLI_STATUS_STDOUT_FD', 91), \
                mock.patch.object(subject, '_CLI_STATUS_STDERR_FD', None), \
                mock.patch.object(subject.os, 'write',
                                  side_effect=lambda fd, data: (writes.append((fd, data)) or len(data))):
            self.assertTrue(subject._emit_status({'status': 'failure'}, to_stderr=True))
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][0], 91)
        self.assertEqual(json.loads(writes[0][1]), {'status': 'failure'})

    def test_main_fails_nonzero_when_success_or_help_status_cannot_be_delivered(self):
        import export_owned_cached_onnx as subject
        from unittest import mock

        arguments = ['--checkpoint', 'checkpoint', '--config', 'config', '--source',
                     'source', '--output', 'output']
        with mock.patch.object(subject, '_CLI_MODE', True), \
                mock.patch.object(subject, '_CLI_STATUS_STDOUT_FD', 91), \
                mock.patch.object(subject.os, 'write', side_effect=BrokenPipeError()), \
                mock.patch.object(subject, 'export', return_value={'status': 'verified'}):
            self.assertEqual(subject.main(arguments), 7)
            self.assertEqual(subject.main(['--help']), 7)

    def test_failure_exit_is_preserved_when_error_status_cannot_be_delivered(self):
        import export_owned_cached_onnx as subject
        from unittest import mock

        arguments = ['--checkpoint', 'checkpoint', '--config', 'config', '--source',
                     'source', '--output', 'output']
        parity_failure = subject.CachedParityFailure(False, 'report_write_failed', {}, None)
        with mock.patch.object(subject, '_CLI_MODE', True), \
                mock.patch.object(subject, '_CLI_STATUS_STDERR_FD', 92), \
                mock.patch.object(subject.os, 'write', side_effect=BrokenPipeError()), \
                mock.patch.object(subject, 'export', side_effect=parity_failure):
            self.assertEqual(subject.main(arguments), 1)

    def test_missing_or_raising_dependency_imports_are_sanitized(self):
        import export_owned_cached_onnx as subject

        for import_failure in ('ModuleNotFoundError', 'RuntimeError'):
            with self.subTest(import_failure=import_failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'torch.py').write_text(
                    f"raise {import_failure}('PRIVATE_IMPORT_PATH_SENTINEL')\n", encoding='utf-8')
                result = self._run_cli(
                    subject.__file__,
                    ['--checkpoint', 'checkpoint', '--config', 'config', '--source', 'source',
                     '--output', 'output'],
                    pythonpath=root)
                combined = result.stdout + result.stderr
                self.assertEqual(result.returncode, 5)
                self.assertNotIn('PRIVATE_IMPORT_PATH_SENTINEL', combined)
                self.assertNotIn(str(root), combined)
                self.assertNotIn(str(Path(subject.__file__).resolve().parent), combined)
                self.assertNotIn('Traceback', combined)
                self.assertEqual(json.loads(result.stderr), {
                    'status': 'dependency_unavailable', 'failure_detail_suppressed': True})


if __name__ == '__main__':
    unittest.main()
