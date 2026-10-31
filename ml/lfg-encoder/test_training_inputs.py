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
    def test_payload_verification_checks_locked_files_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheels = root / 'wheels'; wheels.mkdir()
            site = root / 'site'; site.mkdir()
            data = root / 'data'; data.mkdir()
            dist = site / 'Example-1.2.dist-info'; dist.mkdir()
            package = site / 'example'; package.mkdir()
            meta = b'Metadata-Version: 2.1\nName: Example\nVersion: 1.2\n'
            wheel_meta = b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n'
            (dist / 'METADATA').write_bytes(meta)
            (dist / 'WHEEL').write_bytes(wheel_meta)
            (package / '__init__.py').write_bytes(b'VALUE = 1\n')
            (data / 'asset.txt').write_bytes(b'asset')
            wheel = wheels / 'fixture.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('Example-1.2.dist-info/METADATA', meta)
                archive.writestr('Example-1.2.dist-info/WHEEL', wheel_meta)
                archive.writestr('Example-1.2.dist-info/RECORD', 'installer replaces this')
                archive.writestr('example/__init__.py', b'VALUE = 1\n')
                archive.writestr('Example-1.2.data/data/asset.txt', b'asset')
                archive.writestr('Example-1.2.data/scripts/command', '#!python\n')
            lock = root / 'lock.txt'; lock.write_bytes(lock_wheelhouse.lock(wheels))
            packages = [SimpleNamespace(metadata={'Name': 'example'}, version='1.2')]
            paths = {'purelib': str(site), 'platlib': str(site), 'data': str(data)}
            with patch.object(inputs.importlib.metadata, 'distributions', return_value=packages), \
                    patch.object(inputs.sysconfig, 'get_paths', return_value=paths):
                evidence = inputs.verify_environment(lock, wheels, [], verify_payload=True)
                self.assertFalse(evidence['installed_files_verified'])
                self.assertEqual(evidence['locked_wheel_payload']['files'], 4)
                self.assertEqual(evidence['locked_wheel_payload']['excluded_wheel_records'], 1)
                self.assertEqual(evidence['locked_wheel_payload']['excluded_wheel_scripts'], 1)
                (package / '__init__.py').write_bytes(b'VALUE = 2\n')
                with self.assertRaisesRegex(ValueError, 'payload differs'):
                    inputs.verify_environment(lock, wheels, [], verify_payload=True)
                (package / '__init__.py').unlink()
                with self.assertRaises(FileNotFoundError):
                    inputs.verify_environment(lock, wheels, [], verify_payload=True)

    def test_payload_rejects_unsafe_wheel_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); wheels = root / 'wheels'; wheels.mkdir()
            for name in ('site', 'data'):
                (root / name).mkdir()
            (root / 'site' / 'Example-1.2.dist-info').mkdir()
            (root / 'site' / 'Example-1.2.dist-info' / 'WHEEL').write_bytes(b'Root-Is-Purelib: true\n')
            with zipfile.ZipFile(wheels / 'fixture.whl', 'w') as archive:
                archive.writestr('Example-1.2.dist-info/WHEEL', 'Root-Is-Purelib: true\n')
                archive.writestr('../escape.py', 'bad')
            paths = {'purelib': str(root / 'site'), 'platlib': str(root / 'site'), 'data': str(root / 'data')}
            with patch.object(inputs.sysconfig, 'get_paths', return_value=paths), \
                    self.assertRaisesRegex(ValueError, 'unsafe file path'):
                inputs.verify_wheel_payload(wheels)

    def test_payload_rejects_collisions_and_unsupported_scheme(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); wheels = root / 'wheels'; wheels.mkdir()
            site = root / 'site'; site.mkdir()
            data = root / 'data'; data.mkdir()
            dist = site / 'Example-1.2.dist-info'; dist.mkdir()
            wheel_meta = b'Root-Is-Purelib: true\n'
            (dist / 'WHEEL').write_bytes(wheel_meta)
            (site / 'same.py').write_bytes(b'same')
            path = wheels / 'fixture.whl'
            def write_wheel(extra):
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr('Example-1.2.dist-info/WHEEL', wheel_meta)
                    archive.writestr('same.py', b'same')
                    archive.writestr(extra, b'same')
            paths = {'purelib': str(site), 'platlib': str(site), 'data': str(data)}
            with patch.object(inputs.sysconfig, 'get_paths', return_value=paths):
                write_wheel('Example-1.2.data/purelib/same.py')
                with self.assertRaisesRegex(ValueError, 'same installed file'):
                    inputs.verify_wheel_payload(wheels)
                write_wheel('Example-1.2.data/headers/same.py')
                with self.assertRaisesRegex(ValueError, 'unsupported data scheme'):
                    inputs.verify_wheel_payload(wheels)

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
