"""Software fixtures for incremental decoding; not evidence of model quality."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch required')
class CachedDecodeTests(unittest.TestCase):
    def test_cached_logits_match_full_model_with_different_backbones(self):
        import torch
        from owned_student import OwnedByteStudent
        from owned_cached_decode import DecodeSession
        torch.set_num_threads(1)
        for kernel, attention_every in ((1, 1), (3, 2), (4, 9)):
            with self.subTest(kernel=kernel, attention_every=attention_every):
                torch.manual_seed(19)
                model = OwnedByteStudent({'width': 12, 'layers': 3, 'heads': 3, 'conv_kernel': kernel,
                                          'context': 9, 'ff_multiplier': 2, 'attention_every': attention_every}).eval()
                tokens = torch.randint(0, 256, (2, 9))
                with torch.inference_mode():
                    expected = model(tokens)
                    session = DecodeSession(model, 2)
                    actual = torch.stack([session.step(tokens[:, i]) for i in range(tokens.shape[1])], dim=1)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                with self.assertRaisesRegex(ValueError, 'context exhausted'):
                    session.step(tokens[:, 0])

    def test_sessions_are_isolated_and_bad_input_does_not_advance(self):
        import torch
        from owned_student import OwnedByteStudent
        from owned_cached_decode import DecodeSession
        model = OwnedByteStudent({'width': 8, 'layers': 2, 'heads': 2, 'conv_kernel': 3,
                                  'context': 4, 'ff_multiplier': 2, 'attention_every': 1}).eval()
        first, second = DecodeSession(model, 1), DecodeSession(model, 1)
        token = torch.tensor([10])
        torch.testing.assert_close(first.step(token), second.step(token))
        with self.assertRaisesRegex(ValueError, 'one byte'):
            first.step(torch.tensor([256]))
        self.assertEqual(first.position, 1)
        first.step(torch.tensor([11]))
        torch.testing.assert_close(second.step(torch.tensor([12])), model(torch.tensor([[10, 12]]))[:, -1],
                                   rtol=1e-5, atol=1e-6)
        model.train()
        with self.assertRaisesRegex(ValueError, 'evaluation mode'):
            first.step(token)
        with self.assertRaisesRegex(ValueError, 'evaluation mode'):
            DecodeSession(model, 1)

    @unittest.skipUnless(importlib.util.find_spec('safetensors'), 'safetensors required')
    def test_benchmark_uses_and_checks_the_selected_checkpoint(self):
        import hashlib
        import json
        from pathlib import Path
        import tempfile
        from safetensors.torch import save_file
        import benchmark_owned_cached_decode as benchmark
        import owned_student
        architecture = {'width': 8, 'layers': 2, 'heads': 2, 'conv_kernel': 3,
                        'context': 4, 'ff_multiplier': 2, 'attention_every': 2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, source = root / 'weights.safetensors', root / 'source.txt'
            save_file(owned_student.OwnedByteStudent(architecture).state_dict(), str(weights))
            source.write_bytes(b'test')
            manifest = {'architecture': architecture, 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
                        'code_sha256': {'owned_student.py': hashlib.sha256(Path(owned_student.__file__).read_bytes()).hexdigest()}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            report = benchmark.run(root, source, length=4, threads=1, warmup=0, repeats=2)
            self.assertEqual(report['length_bytes'], 4)
            self.assertFalse(report['native_runtime_measured'])
            self.assertFalse(report['assessment_quality_measured'])
            self.assertLess(report['max_absolute_logit_difference'], 1e-5)
            manifest['weights_sha256'] = '0' * 64
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'do not match'):
                benchmark.run(root, source, length=4, threads=1, warmup=0, repeats=2)


if __name__ == '__main__':
    unittest.main()
