import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import owned_evaluation_chain as chain


class EvaluationChainTests(unittest.TestCase):
    def test_finished_assessment_starts_configured_evaluator_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'source'; checkpoint = source / 'segments' / 'result'
            checkpoint.mkdir(parents=True)
            weights = b'fixture weights'; digest = hashlib.sha256(weights).hexdigest()
            (checkpoint / 'weights.safetensors').write_bytes(weights)
            (checkpoint / 'manifest.json').write_text(json.dumps({'schema_version': 'lfg.owned-assessment-sft-result.v1',
                'completed': True, 'steps': 10, 'weights_sha256': digest}))
            (source / 'complete.json').write_text(json.dumps({'result_path': str(checkpoint),
                'completed_steps': 10, 'weights_sha256': digest}))
            fake = root / 'evaluator.py'
            fake.write_text('import sys\nprint(sys.argv[sys.argv.index("--checkpoint")+1])\n')
            config = {'schema_version': 'lfg.owned-evaluation-chain.v1', 'python': sys.executable,
                      'evaluator': str(fake), 'source_state': str(source), 'dataset': str(root / 'data'),
                      'policy': str(root / 'policy'), 'evaluation_config': str(root / 'config'),
                      'output': str(root / 'output'), 'device': 'cpu', 'threads': 1, 'poll_seconds': 1,
                      'max_checkpoint_bytes': 1000}
            path = root / 'chain.json'; path.write_text(json.dumps(config))
            result = subprocess.run([sys.executable, str(Path(chain.__file__)), '--config', str(path)],
                                    capture_output=True, text=True, timeout=10, check=True)
            self.assertEqual(result.stdout.strip(), str(checkpoint.resolve()))


if __name__ == '__main__':
    unittest.main()
