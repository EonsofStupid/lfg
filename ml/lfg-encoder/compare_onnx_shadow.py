#!/usr/bin/env python3
"""Compare local native shadow predictions with their source Python model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import predict_model
import train


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare(native: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, int]:
    def index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result = {}
        for row in rows:
            if not isinstance(row, dict) or row.get("schema_version") != "lfg.evaluation-prediction.v1":
                raise ValueError("unsupported prediction record")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier.strip() or identifier in result:
                raise ValueError("prediction IDs must be nonempty and unique")
            for field in ("intents", "capabilities", "risks"):
                values = row.get(field)
                if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values) or len(values) != len(set(values)):
                    raise ValueError("prediction labels must be unique nonempty strings")
            result[identifier] = row
        return result
    left, right = index(native), index(reference)
    if not left or left.keys() != right.keys():
        raise ValueError("native and reference prediction IDs differ")
    mismatches = 0
    for identifier, first in left.items():
        second = right[identifier]
        if first.get("input_sha256") != second.get("input_sha256"):
            raise ValueError("native and reference input digests differ")
        if any(set(first[field]) != set(second[field]) for field in ("intents", "capabilities", "risks")):
            mismatches += 1
    return {"examples": len(left), "decision_mismatches": mismatches}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_examples < 1 or args.timeout_seconds < 1:
        raise ValueError("example and time limits must be positive")
    records, dataset_hash = train.load_dataset(args.dataset)
    if not records or len(records) > args.max_examples:
        raise ValueError("dataset record count exceeds configured limits")
    raw = args.dataset.read_bytes()
    if digest(raw) != dataset_hash:
        raise ValueError("dataset changed during loading")
    manifest_path = args.artifact / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    source_hash = predict_model.artifact_digest(args.model)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "lfg.encoder-onnx-artifact.v2" or manifest.get("source_artifact_sha256") != source_hash:
        raise ValueError("ONNX artifact does not identify the supplied source model")
    manifest_hash = digest(manifest_raw)
    binary_hash = digest(args.binary.read_bytes())
    runtime_config_hash = digest(args.runtime_config.read_bytes())
    operations_path = Path(os.environ["LFG_OPERATIONS_CONFIG"]) if "LFG_OPERATIONS_CONFIG" in os.environ else None
    operations_hash = digest(operations_path.read_bytes()) if operations_path else None
    predictor_hash = digest(Path(predict_model.__file__).read_bytes())
    commands = [
        [str(args.binary.resolve()), "onnx-predict", "--artifact", str(args.artifact.resolve()),
         "--manifest-sha256", manifest_hash, "--runtime-config", str(args.runtime_config.resolve())],
        [sys.executable, str(Path(predict_model.__file__).resolve()), "--model", str(args.model.resolve()),
         "--dataset", str(args.dataset.resolve()), "--taxonomy", str((args.artifact / "taxonomy.json").resolve()),
         "--config", str((args.artifact / "prediction-config.json").resolve())],
    ]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    results = []
    for index, command in enumerate(commands):
        process = subprocess.run(command, input=raw if index == 0 else None, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, env=env, timeout=args.timeout_seconds)
        if process.returncode:
            raise ValueError(f"{'native' if index == 0 else 'reference'} prediction process failed")
        results.append([json.loads(line) for line in process.stdout.decode("utf-8").splitlines() if line.strip()])
    counts = compare(*results)
    expected_inputs = {row["id"]: digest(row["text"].encode("utf-8")) for row in records}
    if {row["id"]: row.get("input_sha256") for row in results[0]} != expected_inputs:
        raise ValueError("predictions do not match the supplied dataset")
    if digest(args.dataset.read_bytes()) != dataset_hash or digest(manifest_path.read_bytes()) != manifest_hash or \
            predict_model.artifact_digest(args.model) != source_hash or digest(args.binary.read_bytes()) != binary_hash or \
            digest(args.runtime_config.read_bytes()) != runtime_config_hash or \
            (digest(operations_path.read_bytes()) if operations_path else None) != operations_hash or \
            digest(Path(predict_model.__file__).read_bytes()) != predictor_hash:
        raise ValueError("comparison inputs changed during execution")
    return {"schema_version": "lfg.onnx-shadow-comparison.v1", **counts,
            "unverified_examples": sum(not row["verified"] for row in records),
            "dataset_sha256": dataset_hash, "source_artifact_sha256": source_hash,
            "onnx_manifest_sha256": manifest_hash, "binary_sha256": binary_hash,
            "runtime_config_sha256": runtime_config_hash,
            "operations_config_sha256": operations_hash, "reference_predictor_sha256": predictor_hash,
            "python_version": sys.version, "reference_packages": train.package_versions(("torch", "transformers", "tokenizers", "numpy", "safetensors")),
            "comparator_sha256": digest(Path(__file__).read_bytes()), "promotion_eligible": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "artifact", "model", "dataset", "runtime-config", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--max-examples", required=True, type=int)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    args = parser.parse_args()
    try:
        result = run(args)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
        print(json.dumps(result, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"comparison rejected: {error}") from error


if __name__ == "__main__":
    main()
