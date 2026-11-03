import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import teacher_collection_status as subject


class TeacherCollectionStatusTests(unittest.TestCase):
    def test_only_counts_regular_final_records_and_exposes_no_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, output = root / 'selection', root / 'output'
            selection.mkdir(); output.mkdir()
            (selection / 'manifest.json').write_text(json.dumps({
                'schema_version': 'lfg.teacher-request-selection-result.v1', 'requests': 4,
                'private_text': 'do not expose'}))
            (output / 'records').mkdir(); (output / 'failures').mkdir()
            (output / 'records' / '000000.json').write_text('private response')
            (output / 'records' / '.writing-temp').write_text('private partial')
            (output / 'failures' / '000001.json').write_text('private failure')
            with patch.object(subject.subprocess, 'run') as run:
                run.return_value.stdout = 'active\n'
                result = subject.status(selection, output, 'fixture.service')
            self.assertEqual(result, {'state': 'active', 'completed': 1, 'failed': 1, 'requested': 4})
            self.assertNotIn('private', json.dumps(result))


if __name__ == '__main__':
    unittest.main()
