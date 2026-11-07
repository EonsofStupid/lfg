import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fetch_model_snapshot as fetch
import training_inputs


class FetchSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.revision = 'a' * 40
        self.blobs = {'config.json': b'fixture config', 'model.safetensors': b'fixture weights'}
        for name, content in self.blobs.items():
            (self.root / name).write_bytes(content)
        self.metadata = [SimpleNamespace(rfilename='config.json', size=len(self.blobs['config.json']),
                         blob_id=hashlib.sha1(b'blob 14\0' + self.blobs['config.json']).hexdigest(), lfs=None),
                         SimpleNamespace(rfilename='model.safetensors', size=len(self.blobs['model.safetensors']),
                         blob_id='b' * 40, lfs=SimpleNamespace(size=len(self.blobs['model.safetensors']),
                         sha256=hashlib.sha256(self.blobs['model.safetensors']).hexdigest()))]
        self.info = SimpleNamespace(sha=self.revision, private=False, gated=False, siblings=self.metadata)
        self.module = SimpleNamespace(HfApi=lambda **kwargs: SimpleNamespace(model_info=lambda *a, **k: self.info),
                                      hf_hub_download=lambda **k: str(self.root / k['filename']))

    def run_fetch(self, files=None):
        with patch.dict(sys.modules, {'huggingface_hub': self.module}):
            return fetch.fetch('owner/model', self.revision, files or list(self.blobs), self.root / 'cache',
                               self.root / 'snapshot', self.root / 'manifest.json', self.root / 'provenance.json')

    def test_fetch_binds_git_and_lfs_content_to_snapshot(self):
        result = self.run_fetch()
        self.assertEqual(result['files'], 2)
        evidence = training_inputs.verify_model(self.root / 'snapshot', self.root / 'manifest.json', self.revision)
        self.assertEqual(evidence['manifest_sha256'], result['manifest_sha256'])
        provenance = json.loads((self.root / 'provenance.json').read_text())
        self.assertEqual(provenance['endpoint'], 'https://huggingface.co')
        self.assertEqual(provenance['files']['model.safetensors']['source_lfs_sha256'],
                         hashlib.sha256(self.blobs['model.safetensors']).hexdigest())
        self.assertFalse((self.root / 'snapshot' / 'model.safetensors').is_symlink())

    def test_rejects_bad_content_revision_and_paths(self):
        self.info.sha = 'b' * 40
        with self.assertRaisesRegex(ValueError, 'revision'):
            self.run_fetch()
        self.info.sha = self.revision
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            self.run_fetch(['../escape'])
        self.metadata[0].blob_id = '0' * 40
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.run_fetch()
        self.assertFalse((self.root / 'manifest.json').exists())

    def test_rejects_lfs_tampering_and_gated_source(self):
        self.info.gated = True
        with self.assertRaisesRegex(ValueError, 'public, ungated'):
            self.run_fetch()
        self.info.gated = False
        (self.root / 'model.safetensors').write_bytes(b'changed weights')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.run_fetch()
        self.assertFalse((self.root / 'manifest.json').exists())

    def test_refuses_existing_or_nested_output_paths(self):
        (self.root / 'snapshot').mkdir()
        with self.assertRaisesRegex(ValueError, 'must be new'):
            self.run_fetch()
        with patch.dict(sys.modules, {'huggingface_hub': self.module}), \
                self.assertRaisesRegex(ValueError, 'must be new'):
            fetch.fetch('owner/model', self.revision, ['config.json'], self.root / 'inside' / 'cache',
                        self.root / 'inside', self.root / 'manifest.json', self.root / 'provenance.json')

    def test_rejects_untrusted_endpoint_forms(self):
        for endpoint in ('http://example.com', 'https://user:password@example.com', 'https://example.com/?token=secret'):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(ValueError, 'endpoint'):
                fetch.fetch('owner/model', self.revision, ['config.json'], self.root / 'cache',
                            self.root / 'snapshot', self.root / 'manifest.json', self.root / 'provenance.json', endpoint)


if __name__ == '__main__':
    unittest.main()
