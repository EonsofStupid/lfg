"""Synthetic parity fixtures for the experimental cached ONNX export."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
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
            config = json.loads(Path(subject.__file__).with_name('owned-byte-onnx-export.example.json').read_text())
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
            config = json.loads(Path(subject.__file__).with_name('owned-byte-onnx-export.example.json').read_text())
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
            config = json.loads(Path(subject.__file__).with_name('owned-byte-onnx-export.example.json').read_text())
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
