#!/usr/bin/env python3
"""Run an experimental saved classifier locally on CPU and emit label-only JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import evaluate
import train


def prediction_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != "lfg.encoder-prediction-config.v1":
        raise ValueError("unsupported prediction configuration")
    for field in ("batch_size", "cpu_threads", "max_input_bytes"):
        if type(config.get(field)) is not int or config[field] < 1:
            raise ValueError(f"invalid {field}")
    thresholds = config.get("label_thresholds")
    if not isinstance(thresholds, dict) or any(not isinstance(k, str) or not k.strip() for k in thresholds):
        raise ValueError("invalid per-label thresholds")
    for value in [config.get("default_threshold"), *thresholds.values()]:
        if type(value) not in (float, int) or not 0 <= value <= 1:
            raise ValueError("thresholds must be finite numbers from 0 to 1")
    return config, hashlib.sha256(raw).hexdigest()


def artifact_digest(directory: Path) -> str:
    """Bind every top-level model, tokenizer and run file, without reading checkpoints."""
    entries = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise ValueError("model artifact directory may not contain symlinks")
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries.append([path.name, digest.hexdigest()])
    if not entries:
        raise ValueError("model artifact directory is empty")
    return hashlib.sha256(json.dumps(entries, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def predict(args: argparse.Namespace) -> None:
    config, config_hash = prediction_config(args.config)
    taxonomy, taxonomy_hash = evaluate.load_taxonomy(args.taxonomy)
    records, _ = train.load_dataset(args.dataset)
    if not records:
        raise ValueError("no examples supplied")
    if any(not isinstance(row["id"], str) or not row["id"].strip() or
           len(row["text"].encode("utf-8")) > config["max_input_bytes"] for row in records):
        raise ValueError("example ID or input length is invalid")
    manifest = json.loads((args.model / "run-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "lfg.encoder-run.v1":
        raise ValueError("unsupported model run manifest")
    max_length = manifest.get("max_length")
    if type(max_length) is not int or max_length < 1:
        raise ValueError("invalid model sequence length")
    class_names = json.loads((args.model / "labels.json").read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or any(not isinstance(name, str) or not name.strip() for name in class_names):
        raise ValueError("invalid model labels")
    if not class_names or len(class_names) != len(set(class_names)):
        raise ValueError("model labels must be nonempty and unique")
    all_labels = set.union(*taxonomy.values())
    if set(class_names) - all_labels or set(config["label_thresholds"]) - all_labels:
        raise ValueError("model or threshold label is outside the taxonomy")
    artifact_hash = artifact_digest(args.model)
    runtime = {"python": sys.version, "platform": platform.platform(),
               "packages": train.package_versions(("torch", "transformers", "tokenizers", "safetensors", "numpy"))}
    runtime_hash = hashlib.sha256(json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity_input = {"artifact": artifact_hash, "config": config_hash, "taxonomy": taxonomy_hash, "runtime": runtime_hash,
                      "predictor": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    identity = hashlib.sha256(json.dumps(identity_input, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(config["cpu_threads"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False, use_safetensors=True).to("cpu").eval()
    if model.config.num_labels != len(class_names):
        raise ValueError("model output count does not match labels")
    results = []
    with torch.inference_mode():
        for start in range(0, len(records), config["batch_size"]):
            batch = records[start:start + config["batch_size"]]
            inputs = tokenizer([row["text"] for row in batch], truncation=True, padding=True,
                               max_length=max_length, return_tensors="pt")
            logits = model(**inputs).logits
            if not torch.isfinite(logits).all():
                raise ValueError("model returned non-finite output")
            probabilities = torch.sigmoid(logits)
            for row, values in zip(batch, probabilities.tolist(), strict=True):
                chosen = {name for name, value in zip(class_names, values, strict=True)
                          if value >= config["label_thresholds"].get(name, config["default_threshold"])}
                results.append({"schema_version": evaluate.PREDICTION_SCHEMA, "id": row["id"],
                                "input_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                                "producer_sha256": identity_input["predictor"],
                                "model": {"id": "local-experimental-classifier", "version": "1", "sha256": identity,
                                          **identity_input},
                                **{field: sorted(chosen & taxonomy[field]) for field in evaluate.FIELDS}})
    if artifact_digest(args.model) != artifact_hash:
        raise ValueError("model artifact changed during inference")
    for result in results:
        print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        predict(args)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        raise SystemExit(f"prediction rejected: {error}") from error


if __name__ == "__main__":
    main()
