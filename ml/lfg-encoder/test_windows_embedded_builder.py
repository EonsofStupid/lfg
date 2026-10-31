import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == 'nt', 'requires Windows PowerShell')
class WindowsEmbeddedBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = Path(__file__).resolve().parents[2]
        self.script = self.repo / 'scripts/build-windows-baseline.ps1'
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.repo, text=True).strip()
        self.output = self.root / 'new output'
        self.model = self.root / 'model artifact'
        self.model.mkdir()
        hashes = {}
        for name in ('tokenizer.json', 'labels.json', 'prediction-config.json', 'taxonomy.json', 'model.onnx', 'vocab.txt'):
            payload = ('test-only-' + name).encode()
            (self.model / name).write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        self.manifest = self.model / 'manifest.json'
        self.manifest.write_text(json.dumps({'schema_version': 'lfg.encoder-onnx-artifact.v2',
                                             'files_sha256': hashes}), encoding='utf-8')
        self.manifest_sha = hashlib.sha256(self.manifest.read_bytes()).hexdigest()

    def run_builder(self, *args):
        return subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-File', str(self.script),
                               '-Commit', self.commit, '-OutputDir', str(self.output), *args],
                              cwd=self.repo, text=True, capture_output=True, timeout=30)

    def test_unpaired_invalid_and_mismatched_artifacts_are_rejected_before_output(self):
        cases = [(['-ModelArtifactDir', str(self.model)], 'must be supplied together'),
                 (['-ModelArtifactDir', '', '-ModelManifestSha256', ''], 'must be supplied together'),
                 (['-ModelArtifactDir', str(self.model), '-ModelManifestSha256', 'bad'], 'lowercase sha-256'),
                 (['-ModelArtifactDir', str(self.model), '-ModelManifestSha256', '0' * 64], 'manifest digest mismatch')]
        for args, message in cases:
            with self.subTest(message=message):
                result = self.run_builder(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, (result.stdout + result.stderr).lower())
                self.assertFalse(self.output.exists())

    def test_changed_weight_file_is_rejected_before_output(self):
        (self.model / 'model.onnx').write_bytes(b'changed fixture')
        result = self.run_builder('-ModelArtifactDir', str(self.model), '-ModelManifestSha256', self.manifest_sha)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('model artifact digest mismatch', (result.stdout + result.stderr).lower())
        self.assertFalse(self.output.exists())

    def test_changed_manifested_nonembedded_file_is_also_rejected(self):
        (self.model / 'vocab.txt').write_bytes(b'changed fixture')
        result = self.run_builder('-ModelArtifactDir', str(self.model), '-ModelManifestSha256', self.manifest_sha)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('model artifact digest mismatch', (result.stdout + result.stderr).lower())
        self.assertFalse(self.output.exists())

    def test_unsafe_manifest_names_are_rejected_before_output(self):
        original = json.loads(self.manifest.read_bytes())
        for name in ('../escape', 'nested/file', 'nested\\file', 'manifest.json'):
            value = json.loads(json.dumps(original))
            value['files_sha256'][name] = '0' * 64
            self.manifest.write_text(json.dumps(value), encoding='utf-8')
            digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
            with self.subTest(name=name):
                result = self.run_builder('-ModelArtifactDir', str(self.model), '-ModelManifestSha256', digest)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('invalid model artifact file entry', (result.stdout + result.stderr).lower())
                self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / 'existing'
        marker.write_text('keep', encoding='utf-8')
        result = self.run_builder('-ModelArtifactDir', str(self.model), '-ModelManifestSha256', self.manifest_sha)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')


if __name__ == '__main__':
    unittest.main()
