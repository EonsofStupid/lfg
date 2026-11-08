"""Synthetic parity fixtures for the experimental cached ONNX export."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
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


if __name__ == '__main__':
    unittest.main()
