"""Synthetic parity fixtures for the experimental cached ONNX export."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


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

    def test_kernel_one_and_exact_context_boundary(self):
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


if __name__ == '__main__':
    unittest.main()
