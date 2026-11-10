"""Every ONNX Runtime import must turn its telemetry off first, so private model names stay on this machine."""
import pathlib
import re
import unittest

HERE = pathlib.Path(__file__).parent
IMPORT = re.compile(r'^\s*(?:import onnxruntime\b|from onnxruntime\b)')
SWITCH = re.compile(r"os\.environ\['ORT_DISABLE_TELEMETRY'\] = '1'")


class OrtTelemetryTest(unittest.TestCase):
    def test_the_switch_comes_before_every_first_import_in_a_function(self):
        checked = 0
        for path in sorted(HERE.glob('*.py')):
            if path.name == pathlib.Path(__file__).name:
                continue
            lines = path.read_text(encoding='utf-8').splitlines()
            for number, line in enumerate(lines):
                if not IMPORT.match(line):
                    continue
                checked += 1
                # A later import in the same block may follow one that already set the switch.
                nearby = lines[max(0, number - 3):number]
                self.assertTrue(any(SWITCH.search(item) for item in nearby), f'{path.name}:{number + 1} imports onnxruntime without the telemetry switch')
        self.assertGreater(checked, 0)


if __name__ == '__main__':
    unittest.main()
