"""A non-hosted runner must never run hosted SDK cleanup."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest


class HostedCleanupTests(unittest.TestCase):
    @unittest.skipIf(os.name == 'nt', 'hosted cleanup runs on Linux')
    @unittest.skipUnless(shutil.which('bash'), 'bash is not installed')
    def test_non_hosted_runner_skips_cleanup(self):
        script = Path(__file__).resolve().parents[2] / 'ci' / 'reclaim-hosted-sdk-space.sh'
        result = subprocess.run([shutil.which('bash'), str(script)], env={**os.environ, 'RUNNER_ENVIRONMENT': 'self-hosted'},
                                check=True, capture_output=True, text=True)
        self.assertIn('Skipping hosted-image cleanup', result.stdout)


if __name__ == '__main__':
    unittest.main()
