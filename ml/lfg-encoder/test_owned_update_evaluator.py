import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import owned_update_evaluator as subject


class UpdateEvaluatorTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('fcntl'), 'Linux evaluator lock required')
    def test_evaluates_a_completed_job_once_and_recovers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); jobs = root / 'jobs'; job = jobs / ('a' * 64)
            state = job / 'state'; checkpoint = state / 'segments' / 'result'; checkpoint.mkdir(parents=True)
            weights = b'fixture weights'; digest = hashlib.sha256(weights).hexdigest()
            (checkpoint / 'weights.safetensors').write_bytes(weights)
            (checkpoint / 'manifest.json').write_text(json.dumps({'schema_version': 'lfg.owned-assessment-sft-result.v1',
                'completed': True, 'steps': 2, 'weights_sha256': digest}))
            (state / 'complete.json').write_text(json.dumps({'result_path': str(checkpoint),
                'completed_steps': 2, 'weights_sha256': digest}))
            data = root / 'validation'; data.mkdir(); (data / 'validation.jsonl').write_text('fixture')
            policy = root / 'policy.json'; policy.write_text('{}')
            eval_config = root / 'evaluation.json'; eval_config.write_text('{}')
            counter = root / 'calls.txt'; fake = root / 'evaluator.py'
            fake.write_text('''import hashlib,json,pathlib,sys
a=sys.argv; get=lambda n:pathlib.Path(a[a.index(n)+1]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=get('--output');out.mkdir();pathlib.Path(%r).open('a').write('run\\n')
v={'schema_version':'lfg.owned-torch-format-evaluation.v1','weights_sha256':sha(get('--checkpoint')/'weights.safetensors'),
'config_sha256':sha(get('--config')),'validation_source':{'source_sha256':sha(get('--dataset')/'validation.jsonl'),'policy_sha256':sha(get('--policy'))},
'format_counts':{'invalid_json':1}}
(out/'report.json').write_text(json.dumps(v))
''' % str(counter))
            config = {'schema_version': 'lfg.owned-update-evaluator.v1', 'python': sys.executable,
                'evaluator': str(fake), 'jobs_root': str(jobs), 'dataset': str(data), 'policy': str(policy),
                'evaluation_config': str(eval_config), 'report_root': str(root / 'reports'),
                'state_root': str(root / 'evaluator-state'), 'threads': 1, 'poll_seconds': 1,
                'max_checkpoint_bytes': 100000, 'max_jobs': 10, 'device': 'cpu'}
            path = root / 'config.json'; path.write_text(json.dumps(config))
            def run_once():
                process = subprocess.Popen([sys.executable, str(Path(subject.__file__)), '--config', str(path)],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        if (root / 'evaluator-state' / 'completed.json').exists():
                            time.sleep(1.2); return
                        if process.poll() is not None:
                            self.fail(process.stderr.read().decode())
                        time.sleep(.1)
                    self.fail('evaluator did not finish fixture job')
                finally:
                    process.terminate(); process.communicate(timeout=5)
            run_once(); run_once()
            self.assertEqual(counter.read_text().splitlines(), ['run'])
            status = json.loads((root / 'evaluator-state' / 'status.json').read_text())
            self.assertEqual((status['required_fields_present'], status['total']), (0, 1))


if __name__ == '__main__':
    unittest.main()
