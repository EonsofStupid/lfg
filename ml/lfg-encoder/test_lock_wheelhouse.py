import hashlib
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import lock_wheelhouse


class WheelhouseLockTests(unittest.TestCase):
    def make_wheel(self, directory, filename, name, version):
        path = directory / filename
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(f'{name}-{version}.dist-info/METADATA', f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
        return path

    def test_lock_binds_package_version_and_wheel_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.make_wheel(root, 'example.whl', 'Example_Pkg', '1.2.3')
            line = lock_wheelhouse.lock(root).decode().splitlines()[1]
            self.assertEqual(line, f'example-pkg==1.2.3 --hash=sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}')
            with zipfile.ZipFile(path, 'a') as archive:
                archive.writestr('extra', 'changed')
            self.assertNotEqual(line, lock_wheelhouse.lock(root).decode().splitlines()[1])

    def test_duplicate_packages_and_non_wheel_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_wheel(root, 'first.whl', 'Example_Pkg', '1.0')
            second = self.make_wheel(root, 'second.whl', 'example-pkg', '1.1')
            with self.assertRaisesRegex(ValueError, 'multiple wheels'):
                lock_wheelhouse.lock(root)
            second.unlink()
            (root / 'other.txt').write_text('invalid')
            with self.assertRaisesRegex(ValueError, 'only regular wheel'):
                lock_wheelhouse.lock(root)

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheels = root / 'wheels'
            wheels.mkdir()
            self.make_wheel(wheels, 'one.whl', 'pkg', '1.0')
            output = root / 'lock.txt'
            output.write_text('keep', encoding='utf-8')
            script = Path(lock_wheelhouse.__file__)
            result = subprocess.run([sys.executable, str(script), '--wheelhouse', str(wheels), '--output', str(output)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(encoding='utf-8'), 'keep')


if __name__ == '__main__':
    unittest.main()
