from pathlib import Path
import tempfile
import unittest

from local_teacher_service import command


class TeacherServiceTests(unittest.TestCase):
    def test_restart_adds_resume_only_for_an_existing_output(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [Path(root) / name for name in ('python', 'collector', 'selection', 'config', 'policy', 'output')]
            self.assertNotIn('--resume', command(*paths))
            paths[-1].mkdir()
            self.assertEqual(command(*paths)[-1], '--resume')


if __name__ == '__main__':
    unittest.main()
