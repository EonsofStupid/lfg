import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import receive_teacher_dataset as subject


class ReceiveTeacherDatasetTests(unittest.TestCase):
    def payload(self, train=b'{"id":"fixture"}\n', **changes):
        manifest = {'schema_version': 'lfg.assessment-distillation-dataset.v1', 'test_assessed': False,
                    'files': {'train.jsonl': hashlib.sha256(train).hexdigest()},
                    'sources': [{'split': 'train', 'origin': 'owner_chat'}]}
        manifest.update(changes)
        return json.dumps({'schema_version': 'lfg.teacher-dataset-transfer.v1',
                           'manifest_base64': base64.b64encode(json.dumps(manifest).encode()).decode(),
                           'train_base64': base64.b64encode(train).decode()}).encode()

    def test_atomic_private_delivery_and_idempotence(self):
        with tempfile.TemporaryDirectory() as folder:
            inbox = Path(folder) / 'inbox'; payload = self.payload()
            first = subject.receive(payload, inbox, 10000, 10000)
            self.assertFalse(first['already_present'])
            self.assertEqual(first['records'], 1)
            second = subject.receive(payload, inbox, 10000, 10000)
            self.assertTrue(second['already_present'])
            self.assertEqual(first['dataset_id'], second['dataset_id'])
            (inbox / first['dataset_id'] / 'train.jsonl').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                subject.receive(payload, inbox, 10000, 10000)

    def test_rejects_test_data_duplicates_and_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            inbox = Path(folder) / 'inbox'
            with self.assertRaisesRegex(ValueError, 'train-only'):
                subject.receive(self.payload(test_assessed=True), inbox, 10000, 10000)
            with self.assertRaisesRegex(ValueError, 'train-only'):
                subject.receive(self.payload(sources=[{'split': 'test', 'origin': 'owner_chat'}]), inbox, 10000, 10000)
            with self.assertRaisesRegex(ValueError, 'identity'):
                subject.receive(self.payload(b'{"id":"x"}\n{"id":"x"}\n'), inbox, 10000, 10000)
            with self.assertRaisesRegex(ValueError, 'size'):
                subject.receive(self.payload(), inbox, 3, 10000)


if __name__ == '__main__':
    unittest.main()
