import json
import tempfile
import unittest
from pathlib import Path

import benchmark_process


class ProcessBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config, _ = benchmark_process.read_config(Path(benchmark_process.__file__).with_name("benchmark-process.v1.json"))

    def test_builds_bounded_unique_batch_without_copying_labels(self):
        raw = (json.dumps({"id": "private-id", "text": "example text", "risks": ["private-label"]}) + "\n").encode()
        payload, rows, count = benchmark_process.prepare_batch(raw, {**self.config, "repetitions": 2})
        values = [json.loads(line) for line in payload.splitlines()]
        self.assertEqual((rows, count), (1, 2))
        self.assertEqual([value["id"] for value in values], ["0:0", "1:0"])
        self.assertNotIn(b"private-id", payload)
        self.assertNotIn(b"private-label", payload)

    def test_rejects_duplicates_and_batch_overflow(self):
        row = json.dumps({"id": "same", "text": "request"}).encode() + b"\n"
        with self.assertRaisesRegex(ValueError, "unique"):
            benchmark_process.prepare_batch(row + row, self.config)
        with self.assertRaisesRegex(ValueError, "batch exceeds"):
            benchmark_process.prepare_batch(row, {**self.config, "max_batch_bytes": 1})

    def test_config_rejects_invalid_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({**self.config, "measured_runs": 0}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "measured_runs"):
                benchmark_process.read_config(path)


if __name__ == "__main__":
    unittest.main()
