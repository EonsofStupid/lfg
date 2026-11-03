import json
from pathlib import Path
import tempfile
import unittest

import owned_training_metrics as subject


class TrainingMetricsTests(unittest.TestCase):
    def test_uses_latest_completed_checkpoint_and_omits_private_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'segment' / 'checkpoints' / 'step-0000000000000003'
            path.mkdir(parents=True)
            (path / 'state.json').write_text(json.dumps({'schema_version': 'lfg.owned-distillation-checkpoint.v1',
                'progress': {'step': 3, 'losses': [3.0, 2.0, 1.0], 'ce_losses': [2.0, 1.0, .5],
                             'supervised_tokens': 768}, 'private_path': '/not/exposed'}))
            result = subject.metrics(root, 10000, 2)
            self.assertEqual(result['mean_training_loss'], 1.5)
            self.assertEqual(result['mean_cross_entropy'], .75)
            self.assertEqual(result['byte_positions'], 768)
            self.assertNotIn('private', json.dumps(result))
            self.assertEqual(subject.metrics(root, 5, 2)['state'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
