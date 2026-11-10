import base64
import hashlib
from http.server import HTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import teacher_dataset_receiver_server as subject


class ReceiverServerTests(unittest.TestCase):
    def test_authenticated_submission_is_bounded_and_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); token = root / 'token'; token.write_bytes(b'x' * 40)
            server = HTTPServer(('127.0.0.1', 0), subject.handler(root / 'inbox', token, 10000, 5000, 3))
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            self.addCleanup(thread.join, 5); self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
            train = b'{"id":"fixture"}\n'
            manifest = json.dumps({'schema_version': 'lfg.assessment-distillation-dataset.v1',
                'test_assessed': False, 'files': {'train.jsonl': hashlib.sha256(train).hexdigest()},
                'sources': [{'split': 'train', 'origin': 'owner_chat'}]}).encode()
            payload = json.dumps({'schema_version': 'lfg.teacher-dataset-transfer.v1',
                'manifest_base64': base64.b64encode(manifest).decode(),
                'train_base64': base64.b64encode(train).decode()}).encode()
            url = f'http://127.0.0.1:{server.server_port}/api/datasets'
            def request(credential):
                return urllib.request.Request(url, data=payload, headers={'Authorization': 'Bearer ' + credential})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request('wrong'), timeout=5)
            self.assertEqual(error.exception.code, 401)
            with urllib.request.urlopen(request('x' * 40), timeout=5) as response:
                self.assertFalse(json.load(response)['already_present'])
            with urllib.request.urlopen(request('x' * 40), timeout=5) as response:
                self.assertTrue(json.load(response)['already_present'])


if __name__ == '__main__':
    unittest.main()
