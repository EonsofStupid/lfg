"""Small software fixtures for export mechanics, not model-quality data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'onnx', 'onnxruntime', 'safetensors')),
                     'torch, onnx, onnxruntime and safetensors required')
class ExportOwnedByteTests(unittest.TestCase):
    def test_export_checks_checkpoint_and_onnx_parity(self):
        import owned_student
        from safetensors.torch import save_file
        import export_owned_byte_onnx as subject
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            model = owned_student.OwnedByteStudent({'width': 8, 'layers': 2, 'heads': 2, 'conv_kernel': 3,
                        'context': 8, 'ff_multiplier': 2, 'attention_every': 2}).eval()
            weights = checkpoint / 'weights.safetensors'
            save_file(model.state_dict(), str(weights))
            hash_bytes = lambda raw: hashlib.sha256(raw).hexdigest()
            manifest = {'architecture': {'width': 8, 'layers': 2, 'heads': 2, 'conv_kernel': 3,
                        'context': 8, 'ff_multiplier': 2, 'attention_every': 2},
                        'weights_sha256': hash_bytes(weights.read_bytes()),
                        'code_sha256': {'owned_student.py': hash_bytes(Path(owned_student.__file__).read_bytes())}}
            (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
            config = json.loads(Path(subject.__file__).with_name('owned-byte-onnx-export.example.json').read_text())
            config['parity_lengths'] = [1, 4, 8]
            config_path, source = root / 'config.json', root / 'source.txt'
            config_path.write_text(json.dumps(config))
            source.write_bytes(b'fixture!')
            output = root / 'output'
            result = subject.export(checkpoint, config_path, source, output)
            exported = json.loads((output / 'manifest.json').read_bytes())
            self.assertEqual(result['model_sha256'], hash_bytes((output / 'model.onnx').read_bytes()))
            self.assertEqual(result['manifest_sha256'], hash_bytes((output / 'manifest.json').read_bytes()))
            self.assertEqual(exported['parity_lengths'], [1, 4, 8])
            self.assertFalse(exported['promotion_eligible'])
            self.assertFalse(exported['native_runtime_checked'])
            manifest['weights_sha256'] = '0' * 64
            (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'checkpoint does not match'):
                subject.export(checkpoint, config_path, source, root / 'bad')
            self.assertFalse((root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
