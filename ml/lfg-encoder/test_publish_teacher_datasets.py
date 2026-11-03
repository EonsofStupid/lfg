import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import publish_teacher_datasets as subject


class PublishTeacherDatasetsTests(unittest.TestCase):
    def test_delivery_cache_resends_when_credential_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); dataset = root / 'dataset'; dataset.mkdir()
            (dataset / 'manifest.json').write_bytes(b'manifest')
            (dataset / 'train.jsonl').write_bytes(b'training')
            token = root / 'token'; token.write_bytes(b'x' * 40)
            value = {'staging_root': str(root / 'staging'), 'max_manifest_bytes': 10000,
                     'max_dataset_bytes': 10000, 'resend_interval_seconds': 3600,
                     'delivery': {'token_file': str(token)}}
            with patch.object(subject, 'config_file', return_value=(value, b'config')), \
                 patch.object(subject, 'prepare', return_value=dataset), \
                 patch.object(subject, 'deliver', return_value={'records': 1}) as deliver:
                subject.run(root / 'config.json')
                self.assertTrue(subject.run(root / 'config.json')['recently_delivered'])
                self.assertEqual(deliver.call_count, 1)
                token.write_bytes(b'y' * 40)
                subject.run(root / 'config.json')
                self.assertEqual(deliver.call_count, 2)

    def test_prepares_verified_collections_and_delivers_only_train_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); collections = root / 'collections'; batch = collections / 'batch'; batch.mkdir(parents=True)
            teacher_config = root / 'teacher.json'
            teacher_config.write_bytes(subject.encoded({'schema_version': 'lfg.local-teacher-collection.v1',
                'system_prompt': 'Assess the request', 'source_origin': 'owner_chat'}))
            trace = {'id': 'fixture-id', 'request': 'Real request placeholder', 'response': '{"intent":"test"}',
                     'teacher': 'fixture', 'review_status': 'unreviewed', 'outcome': 'unknown',
                     'source_ref': 'fixture', 'source_sha256': 'a' * 64}
            trace_raw = subject.encoded(trace); (batch / 'trace.jsonl').write_bytes(trace_raw)
            manifest = {'schema_version': 'lfg.local-teacher-collection-result.v1',
                'config_sha256': subject.sha(teacher_config.read_bytes()), 'test_opened': False,
                'validation_opened': False, 'review_status': 'unreviewed', 'outcome': 'unknown',
                'files': [{'path': 'trace.jsonl', 'sha256': subject.sha(trace_raw), 'project': 'fixture',
                           'project_group': 'fixture', 'split': 'train', 'origin': 'owner_chat'}]}
            (batch / 'manifest.json').write_bytes(subject.encoded(manifest))
            value = {'schema_version': 'lfg.teacher-dataset-publisher.v1', 'collection_root': str(collections),
                'staging_root': str(root / 'staging'), 'collector_configs': [str(teacher_config)],
                'max_manifest_bytes': 100000, 'max_dataset_bytes': 100000, 'max_transfer_bytes': 200000,
                'plan': {'schema_version': 'lfg.assessment-distillation-plan.v1', 'revision': 'fixture',
                    'task': 'assessment', 'system_prompt': 'Assess the request', 'authorization_attestation': 'fixture',
                    'redaction_attestation': 'fixture', 'forbidden_literals': [], 'include_outcomes': ['unknown'],
                    'max_source_bytes': 100000, 'max_total_bytes': 100000, 'max_examples': 10},
                'delivery': {'endpoint': 'https://spark.example.test/api/datasets',
                             'allowed_hosts': ['spark.example.test'], 'token_file': str(root / 'token'),
                             'timeout_seconds': 10}}
            (root / 'token').write_bytes(b'x' * 40)
            raw_config = subject.encoded(value)
            dataset = subject.prepare(value, raw_config)
            self.assertTrue((dataset / 'train.jsonl').is_file())
            self.assertEqual(subject.prepare(value, raw_config), dataset)
            man = (dataset / 'manifest.json').read_bytes(); data = (dataset / 'train.jsonl').read_bytes()
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self, limit):
                    return subject.encoded({'dataset_id': hashlib.sha256(man + b'\0' + data).hexdigest(),
                                            'records': 1, 'already_present': False})
            with patch.object(subject.urllib.request, 'build_opener') as opener:
                opener.return_value.open.return_value = Response()
                self.assertEqual(subject.deliver(dataset, value)['records'], 1)
                request = opener.return_value.open.call_args.args[0]
                self.assertEqual(request.full_url, value['delivery']['endpoint'])
                self.assertNotIn(b'x' * 40, request.data)
            value['delivery']['endpoint'] = 'https://unapproved.example.test/api/datasets'
            with self.assertRaisesRegex(ValueError, 'approved HTTPS'):
                subject.deliver(dataset, value)
            manifest['test_opened'] = True
            (batch / 'manifest.json').write_bytes(subject.encoded(manifest))
            with self.assertRaisesRegex(ValueError, 'identity or policy'):
                subject.prepare(value, raw_config)


if __name__ == '__main__':
    unittest.main()
