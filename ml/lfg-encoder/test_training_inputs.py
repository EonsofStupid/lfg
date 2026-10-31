import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lock_wheelhouse
import training_inputs as inputs


class TrainingInputTests(unittest.TestCase):
    def test_model_manifest_binds_all_files_revision_and_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_bytes(b'fixture')
            (model / 'nested').mkdir()
            (model / 'nested' / 'weights').write_bytes(b'weights')
            revision = 'a' * 40
            manifest = root / 'manifest.json'
            manifest.write_bytes(inputs.model_manifest(model, 'test-model', revision))
            proof = inputs.verify_model(model, manifest, revision)
            self.assertEqual(proof['files'], 2)
            self.assertEqual(proof['manifest_sha256'], hashlib.sha256(manifest.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, 'revision'):
                inputs.verify_model(model, manifest, 'b' * 40)
            (model / 'config.json').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                inputs.verify_model(model, manifest, revision)
            (model / 'config.json').write_bytes(b'fixture')
            (model / 'new-directory').mkdir()
            with self.assertRaisesRegex(ValueError, 'does not match'):
                inputs.verify_model(model, manifest, revision)

    def test_environment_requires_exact_wheels_versions_and_explicit_extras(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheels = root / 'wheels'
            wheels.mkdir()
            wheel = wheels / 'fixture.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('Example-1.2.dist-info/METADATA', 'Metadata-Version: 2.1\nName: Example\nVersion: 1.2\n')
            lock = root / 'lock.txt'
            lock.write_bytes(lock_wheelhouse.lock(wheels))
            packages = [SimpleNamespace(metadata={'Name': 'example'}, version='1.2'),
                        SimpleNamespace(metadata={'Name': 'pip'}, version='24.0')]
            with patch.object(inputs.importlib.metadata, 'distributions', return_value=packages):
                evidence = inputs.verify_environment(lock, wheels, ['pip'])
                self.assertFalse(evidence['installed_files_verified'])
                with self.assertRaisesRegex(ValueError, 'unlisted'):
                    inputs.verify_environment(lock, wheels, [])
                packages[0].version = '1.3'
                with self.assertRaisesRegex(ValueError, 'versions'):
                    inputs.verify_environment(lock, wheels, ['pip'])
                packages[0].version = '1.2'
                with wheel.open('ab') as stream:
                    stream.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'wheelhouse'):
                    inputs.verify_environment(lock, wheels, ['pip'])

    def test_environment_rejects_duplicate_distribution_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheels = root / 'wheels'
            wheels.mkdir()
            with zipfile.ZipFile(wheels / 'fixture.whl', 'w') as archive:
                archive.writestr('Example-1.2.dist-info/METADATA', 'Metadata-Version: 2.1\nName: Example\nVersion: 1.2\n')
            lock = root / 'lock.txt'
            lock.write_bytes(lock_wheelhouse.lock(wheels))
            packages = [SimpleNamespace(metadata={'Name': 'example'}, version='1.2')] * 2
            with patch.object(inputs.importlib.metadata, 'distributions', return_value=packages):
                with self.assertRaisesRegex(ValueError, 'multiple installed'):
                    inputs.verify_environment(lock, wheels, [])


if __name__ == '__main__':
    unittest.main()
