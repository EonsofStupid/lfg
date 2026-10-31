import json
import tempfile
import unittest
from pathlib import Path

import predict_model


class PredictionConfigurationTests(unittest.TestCase):
    def test_thresholds_are_configured_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            value = {"schema_version": "lfg.encoder-prediction-config.v1", "default_threshold": 0.5,
                     "label_thresholds": {"example": 0.25}, "batch_size": 2, "cpu_threads": 1,
                     "max_input_bytes": 100}
            path.write_text(json.dumps(value), encoding="utf-8")
            config, digest = predict_model.prediction_config(path)
            self.assertEqual(config["label_thresholds"]["example"], 0.25)
            self.assertEqual(len(digest), 64)
            value["label_thresholds"]["example"] = float("nan")
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "thresholds"):
                predict_model.prediction_config(path)

    def test_artifact_digest_changes_when_model_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"first")
            first = predict_model.artifact_digest(root)
            (root / "model.safetensors").write_bytes(b"second")
            self.assertNotEqual(first, predict_model.artifact_digest(root))


if __name__ == "__main__":
    unittest.main()
