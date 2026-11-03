import json
from pathlib import Path
import tempfile
import time
import unittest

import owned_training_dashboard as dashboard


class DashboardStatusTests(unittest.TestCase):
    def test_only_public_progress_fields_are_exposed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            path.write_text(json.dumps({'schema_version': 'lfg.owned-training-status.v1',
                'state': 'training', 'step': 10, 'total_steps': 100, 'updated_unix': time.time(),
                'private_path': '/secret/corpus', 'private_text': 'not public'}))
            result = dashboard.status(path)
            self.assertEqual(set(result), {'state', 'step', 'total_steps', 'updated_unix'})
            self.assertEqual(result['step'], 10)
            path.write_text('{')
            self.assertEqual(dashboard.status(path)['state'], 'unavailable')

    def test_teacher_url_is_escaped_when_embedded_in_page(self):
        page = dashboard.render_page('</script><script>bad</script>')
        self.assertNotIn(b'</script><script>bad</script>', page)
        self.assertIn(b'\\u003c/script>', page)


if __name__ == '__main__':
    unittest.main()
