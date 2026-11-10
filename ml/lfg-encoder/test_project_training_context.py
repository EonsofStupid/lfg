import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import project_training_context as context


class ProjectTrainingContextTests(unittest.TestCase):
    def test_render_has_stable_utf8_bytes_and_requires_digest(self):
        prompt = 'project: α\nstatus: active'
        digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        rendered = context.render(prompt, 'fix the route', max_context_bytes=100,
                                  expected_sha256=digest)
        self.assertEqual(rendered.encode('utf-8'), 'project: α\nstatus: active\n\nfix the route'.encode('utf-8'))
        self.assertEqual(context.render(None, 'legacy', max_context_bytes=100), 'legacy')
        for bad_digest in (None, '0' * 64):
            with self.assertRaises(ValueError):
                context.render(prompt, 'fix', max_context_bytes=100, expected_sha256=bad_digest)
        with self.assertRaises(ValueError):
            context.render(prompt, 'fix', max_context_bytes=2, expected_sha256=digest)

    def test_load_checks_context_digest_schema_and_size(self):
        text = 'known project context'
        entry = {'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                 'extensions': {'custom': 'retained'}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'context.json'
            path.write_text(json.dumps({'schema_version': context.SCHEMA, 'top_extension': 'retained', 'projects': {'demo': {
                'text': text, 'sha256': entry['sha256'], 'custom': 'retained'}}}))
            loaded, file_sha, extensions = context.load(path, max_file_bytes=1000, max_context_bytes=100)
            self.assertEqual(loaded['demo'], entry)
            self.assertEqual(file_sha, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(extensions, {'top_extension': 'retained'})
            path.write_text(json.dumps({'schema_version': context.SCHEMA, 'projects': {
                'demo': {'text': text, 'sha256': '0' * 64}}}))
            with self.assertRaisesRegex(ValueError, 'entry'):
                context.load(path, max_file_bytes=1000, max_context_bytes=100)
            with self.assertRaisesRegex(ValueError, 'limit'):
                context.load(path, max_file_bytes=5, max_context_bytes=100)


if __name__ == '__main__':
    unittest.main()
