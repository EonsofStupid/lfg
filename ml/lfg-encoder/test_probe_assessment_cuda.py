"""CLI guards for a fixture-only GPU probe; no project data are used."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name('probe_assessment_cuda.py')


class AssessmentCudaProbeTests(unittest.TestCase):
    def args(self, root: Path, fraction: str) -> list[str]:
        return [sys.executable, str(SCRIPT), '--model', str(root / 'missing-model'),
                '--model-manifest', str(root / 'missing-manifest'), '--revision', 'missing',
                '--config', str(root / 'missing-config'), '--environment-lock', str(root / 'missing-lock'),
                '--wheelhouse', str(root / 'missing-wheels'), '--device-index', '0',
                '--cuda-memory-fraction', fraction, '--output', str(root / 'output.json')]

    def test_invalid_budget_fails_before_accessing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run(self.args(root, '0'), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('invalid CUDA device or allocator fraction', result.stderr)
            self.assertFalse((root / 'output.json').exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output.json'
            output.write_text('preserve', encoding='utf-8')
            result = subprocess.run(self.args(root, '0.01'), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('probe output must be new', result.stderr)
            self.assertEqual(output.read_text(encoding='utf-8'), 'preserve')


if __name__ == '__main__':
    unittest.main()
