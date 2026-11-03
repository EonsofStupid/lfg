import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import owned_dataset_queue as queue
import receive_teacher_dataset as receiver


def dataset(inbox, rows):
    raw = b''.join((json.dumps(row, sort_keys=True) + '\n').encode() for row in rows)
    manifest = json.dumps({'schema_version': 'lfg.assessment-distillation-dataset.v1',
        'test_assessed': False, 'files': {'train.jsonl': hashlib.sha256(raw).hexdigest()},
        'sources': [{'split': 'train', 'origin': 'owner_chat'}]}).encode()
    transfer = json.dumps({'schema_version': 'lfg.teacher-dataset-transfer.v1',
        'manifest_base64': base64.b64encode(manifest).decode(), 'train_base64': base64.b64encode(raw).decode()}).encode()
    return inbox / receiver.receive(transfer, inbox, 100000, 100000)['dataset_id']


class DatasetQueueTests(unittest.TestCase):
    def test_context_sizing_fits_full_messages_and_respects_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); data = root / 'data'; data.mkdir(); checkpoint = root / 'checkpoint'; checkpoint.mkdir()
            (data / 'train.jsonl').write_text(json.dumps({'messages': [{'content': 'sys'},
                {'content': 'u' * 100}, {'content': 'a' * 20}]}) + '\n')
            (checkpoint / 'manifest.json').write_text(json.dumps({'architecture': {'context': 64,
                'heads': 2, 'layers': 2, 'attention_every': 2}}))
            config = {'context': 64, 'batch_size': 1, 'max_attention_elements': 100000,
                      'framing': {'system_prefix': '<s>', 'system_suffix': '', 'user_prefix': '<u>',
                                  'user_suffix': '', 'assistant_prefix': '<a>', 'assistant_suffix': ''}}
            self.assertEqual(queue.fit_sft_config(config, data, checkpoint, 256, 64)['context'], 192)
            with self.assertRaisesRegex(ValueError, 'resource limits'):
                queue.fit_sft_config(config, data, checkpoint, 128, 64)

    def test_detects_new_supersets_and_rejects_conflicting_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); inbox = root / 'inbox'
            first = {'id': 'first', 'value': 1}
            base = dataset(root / 'baseline', [first])
            previous = queue.dataset_records(base, 100000, 100000)
            same = dataset(inbox, [first])
            self.assertIsNone(queue.choose_dataset(inbox, previous, 100000, 100000, 10, False))
            next_path = dataset(inbox, [first, {'id': 'second', 'value': 2}])
            self.assertEqual(queue.choose_dataset(inbox, previous, 100000, 100000, 10, False)[0], next_path)
            dataset(inbox, [first, {'id': 'different', 'value': 3}])
            with self.assertRaisesRegex(ValueError, 'competing'):
                queue.choose_dataset(inbox, previous, 100000, 100000, 10, False)
            self.assertTrue(same.exists())

    def test_rejects_unapproved_revision_and_tampered_inbox(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); inbox = root / 'inbox'
            baseline = dataset(root / 'baseline', [{'id': 'same', 'value': 1}])
            previous = queue.dataset_records(baseline, 100000, 100000)
            changed = dataset(inbox, [{'id': 'same', 'value': 2}])
            with self.assertRaisesRegex(ValueError, 'revises'):
                queue.choose_dataset(inbox, previous, 100000, 100000, 10, False)
            self.assertEqual(queue.choose_dataset(inbox, previous, 100000, 100000, 10, True)[0], changed)
            (changed / 'train.jsonl').write_bytes(b'{}\n')
            with self.assertRaisesRegex(ValueError, 'train-only'):
                queue.choose_dataset(inbox, previous, 100000, 100000, 10, True)

    @unittest.skipUnless(importlib.util.find_spec('fcntl'), 'Linux queue lock required')
    def test_queue_runs_one_update_and_recovers_without_repeating(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); inbox = root / 'inbox'; baseline = dataset(root / 'baseline', [{'id': 'first'}])
            dataset(inbox, [{'id': 'first'}, {'id': 'second'}])
            initial = root / 'initial'; checkpoint = initial / 'segments' / 'result'; checkpoint.mkdir(parents=True)
            weights = b'initial model'; weight_sha = hashlib.sha256(weights).hexdigest()
            (checkpoint / 'weights.safetensors').write_bytes(weights)
            (checkpoint / 'manifest.json').write_text(json.dumps({'schema_version': 'lfg.owned-assessment-sft-result.v1',
                'completed': True, 'steps': 1, 'weights_sha256': weight_sha,
                'training_file_sha256': hashlib.sha256((baseline / 'train.jsonl').read_bytes()).hexdigest()}))
            (initial / 'complete.json').write_text(json.dumps({'result_path': str(checkpoint),
                'completed_steps': 1, 'weights_sha256': weight_sha}))
            fake = root / 'supervisor.py'
            fake.write_text('''import hashlib,json,pathlib,sys
c=json.load(open(sys.argv[sys.argv.index('--config')+1])); state=pathlib.Path(c['state_dir']); out=state/'segments'/'result'; out.mkdir(parents=True,exist_ok=True)
w=b'updated model'; h=hashlib.sha256(w).hexdigest(); (out/'weights.safetensors').write_bytes(w)
t=hashlib.sha256((pathlib.Path(c['dataset'])/'train.jsonl').read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps({'schema_version':'lfg.owned-assessment-sft-result.v1','completed':True,'steps':1,'weights_sha256':h,'training_file_sha256':t}))
(state/'complete.json').write_text(json.dumps({'result_path':str(out),'completed_steps':1,'weights_sha256':h}))
''')
            trainer = root / 'trainer.py'; trainer.write_text('# fixture')
            sft = root / 'sft.json'; sft.write_text('{}')
            state = root / 'queue'
            config = {'schema_version': 'lfg.owned-dataset-queue.v1', 'python': sys.executable,
                'supervisor': str(fake), 'trainer': str(trainer), 'sft_config': str(sft), 'initial_state': str(initial),
                'baseline_dataset': str(baseline), 'inbox': str(inbox), 'queue_state': str(state),
                'segment_steps': 2, 'poll_seconds': 1, 'max_manifest_bytes': 100000, 'max_dataset_bytes': 100000,
                'max_checkpoint_bytes': 100000, 'max_datasets': 10, 'allow_record_revisions': False}
            config_path = root / 'queue.json'; config_path.write_text(json.dumps(config))
            def run_once():
                process = subprocess.Popen([sys.executable, str(Path(queue.__file__)), '--config', str(config_path)],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        if (state / 'state.json').exists() and json.loads((state / 'state.json').read_text()).get('completed_updates') == 1:
                            time.sleep(1.2)
                            return
                        if process.poll() is not None:
                            self.fail(process.stderr.read().decode())
                        time.sleep(.1)
                    self.fail('queue did not finish fixture update')
                finally:
                    process.terminate(); process.communicate(timeout=5)
            run_once(); run_once()
            self.assertEqual(json.loads((state / 'state.json').read_text())['completed_updates'], 1)
            self.assertEqual(len(list((state / 'jobs').iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
