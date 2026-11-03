import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import owned_training_chain as chain


class TrainingChainTests(unittest.TestCase):
    def test_waits_then_accepts_only_verified_source_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            checkpoint = source / 'segments' / 'segment-1'
            checkpoint.mkdir(parents=True)
            config = {'schema_version': 'lfg.owned-training-chain.v1', 'segment_steps': 10, 'poll_seconds': 2,
                      'max_checkpoint_bytes': 1000}
            for key in ('python', 'supervisor', 'trainer', 'target_state', 'dataset', 'sft_config'):
                config[key] = str(root / key)
            config['source_state'] = str(source)
            path = root / 'config.json'; path.write_text(json.dumps(config))
            config, _ = chain.load_config(path)
            self.assertIsNone(chain.target_config(config))
            weights = b'fixture owned model'
            digest = hashlib.sha256(weights).hexdigest()
            (checkpoint / 'weights.safetensors').write_bytes(weights)
            (checkpoint / 'manifest.json').write_text(json.dumps({
                'schema_version': 'lfg.owned-distillation-result.v1', 'completed': True,
                'steps': 100, 'weights_sha256': digest}))
            (source / 'complete.json').write_text(json.dumps({'result_path': str(checkpoint),
                'completed_steps': 100, 'weights_sha256': digest}))
            result = chain.target_config(config)
            self.assertEqual(result['checkpoint'], str(checkpoint.resolve()))
            self.assertEqual(result['trainer_kind'], 'assessment_sft')
            (checkpoint / 'weights.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                chain.target_config(config)

    def test_rejects_checkpoint_outside_source_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'source'; source.mkdir()
            other = root / 'other'; other.mkdir()
            (source / 'complete.json').write_text(json.dumps({'result_path': str(other)}))
            with self.assertRaisesRegex(ValueError, 'outside'):
                chain.target_config({'source_state': str(source), 'max_checkpoint_bytes': 1000})

    def test_completed_source_starts_configured_supervisor(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'source'; checkpoint = source / 'segments' / 'result'
            checkpoint.mkdir(parents=True)
            weights = b'fixture weights'; digest = hashlib.sha256(weights).hexdigest()
            (checkpoint / 'weights.safetensors').write_bytes(weights)
            (checkpoint / 'manifest.json').write_text(json.dumps({'schema_version': 'lfg.owned-distillation-result.v1',
                'completed': True, 'steps': 10, 'weights_sha256': digest}))
            (source / 'complete.json').write_text(json.dumps({'result_path': str(checkpoint),
                'completed_steps': 10, 'weights_sha256': digest}))
            fake = root / 'supervisor.py'
            fake.write_text('import json,sys\nprint(json.load(open(sys.argv[2]))["trainer_kind"])\n')
            config = {'schema_version': 'lfg.owned-training-chain.v1', 'python': sys.executable,
                      'supervisor': str(fake), 'trainer': str(root / 'trainer'), 'source_state': str(source),
                      'target_state': str(root / 'target'), 'dataset': str(root / 'data'),
                      'sft_config': str(root / 'config'), 'segment_steps': 2, 'poll_seconds': 1,
                      'max_checkpoint_bytes': 1000}
            path = root / 'chain.json'; path.write_text(json.dumps(config))
            result = subprocess.run([sys.executable, str(Path(chain.__file__)), '--config', str(path)],
                                    capture_output=True, text=True, timeout=10, check=True)
            self.assertEqual(result.stdout.strip(), 'assessment_sft')
            generated = json.loads((root / 'target' / 'generated-service-config.json').read_text())
            self.assertEqual(generated['checkpoint'], str(checkpoint.resolve()))


if __name__ == '__main__':
    unittest.main()
