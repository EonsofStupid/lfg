import subprocess
import sys
import unittest
from pathlib import Path


class ModelRuntimeProbeTests(unittest.TestCase):
    def test_invalid_capacity_settings_fail_before_model_access(self):
        script = Path(__file__).with_name('probe_model_runtime.py')
        base = [sys.executable, str(script), '--model', 'missing-model', '--model-manifest', 'missing-manifest',
                '--revision', 'unused', '--taxonomy', 'missing-taxonomy', '--device', 'cpu',
                '--threads', '1', '--tokens', '1', '--seed', '1']
        for extra, message in ((['--batch-size', '0'], 'positive'),
                               (['--cuda-memory-fraction', '0.1'], 'CUDA memory fraction'),
                               (['--optimizer-learning-rate', '0.001'], 'optimizer step requires'),
                               (['--optimizer-learning-rate', 'nan', '--optimizer-weight-decay', '0'], 'optimizer step requires'),
                               (['--optimizer-learning-rate', '0.001', '--optimizer-weight-decay', '-1'], 'optimizer step requires')):
            with self.subTest(extra=extra):
                result = subprocess.run(base + extra, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

        indexed = subprocess.run(base + ['--device', 'cuda:0', '--cuda-memory-fraction', '0.1'],
                                 capture_output=True, text=True)
        self.assertNotEqual(indexed.returncode, 0)
        self.assertNotIn('CUDA memory fraction', indexed.stderr)


if __name__ == '__main__':
    unittest.main()
