"""Negative tests for experimental native encoder tooling; no training data."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import compare_node_encoder
import export_node_encoder
import node_review


class EncoderToolBoundaryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_export_rejects_oversize_queue_before_creating_output(self):
        args = argparse.Namespace(config=self.root / 'config', export_config=self.root / 'export',
            queue=self.root / 'queue', snapshot=self.root / 'snapshot', ledger=self.root / 'ledger',
            max_parity_items=1, output=self.root / 'output', model=self.root / 'model')
        with mock.patch.object(export_node_encoder.node_exemplars, 'read_config', return_value=({}, 'a')), \
             mock.patch.object(export_node_encoder.export_onnx, 'read_config', return_value=({'cpu_threads': 1}, 'b')), \
             mock.patch.object(export_node_encoder.node_review, 'load_ledger', return_value=([{}, {}], {}, b'')):
            with self.assertRaisesRegex(ValueError, 'item limit'):
                export_node_encoder.run(args)
        self.assertFalse(args.output.exists())

    def test_compare_rejects_unmatched_model_before_starting_binary(self):
        artifact = self.root / 'artifact'
        artifact.mkdir()
        raw = json.dumps({'schemaVersion': 'lfg.node-onnx-encoder.v1', 'promotionEligible': False,
                          'sourceModelManifestSha256': 'wrong'}).encode()
        (artifact / 'manifest.json').write_bytes(raw)
        args = argparse.Namespace(config=self.root / 'config', export_config=self.root / 'export',
            queue=self.root / 'queue', snapshot=self.root / 'snapshot', ledger=self.root / 'ledger',
            max_items=2, max_input_bytes=100, timeout_seconds=2.0, model=self.root / 'model',
            model_manifest=self.root / 'manifest', revision='fixture', artifact=artifact,
            manifest_sha256=node_review.digest(raw))
        with mock.patch.object(compare_node_encoder.node_exemplars, 'read_config', return_value=({}, 'a')), \
             mock.patch.object(compare_node_encoder.export_onnx, 'read_config', return_value=({}, 'b')), \
             mock.patch.object(compare_node_encoder.node_review, 'load_ledger', return_value=([{}], {}, b'')), \
             mock.patch.object(compare_node_encoder.training_inputs, 'verify_model', return_value={'manifest_sha256': 'expected'}), \
             mock.patch.object(compare_node_encoder.subprocess, 'run') as process:
            with self.assertRaisesRegex(ValueError, 'does not match'):
                compare_node_encoder.run(args)
            process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
