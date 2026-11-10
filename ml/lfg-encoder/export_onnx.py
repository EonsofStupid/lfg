#!/usr/bin/env python3
"""Export an experimental classifier and check CPU parity; never promote it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import evaluate
import predict_model
import train


def read_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != "lfg.encoder-onnx-export-config.v1":
        raise ValueError("unsupported ONNX export configuration")
    for field in ("opset_version", "cpu_threads", "max_model_bytes"):
        if type(config.get(field)) is not int or config[field] < 1:
            raise ValueError(f"invalid {field}")
    sizes = config.get("parity_batch_sizes")
    if not isinstance(sizes, list) or not sizes or any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("parity_batch_sizes must contain positive integers")
    if len(sizes) != len(set(sizes)):
        raise ValueError("parity_batch_sizes must be unique")
    for field in ("absolute_tolerance", "relative_tolerance"):
        value = config.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {field}")
    return config, hashlib.sha256(raw).hexdigest()


def reject_external_tensors(message: Any, onnx_module: Any) -> None:
    """Reject external tensors anywhere in the protobuf, including nested graphs."""
    if message.DESCRIPTOR.full_name == "onnx.TensorProto":
        if message.data_location == onnx_module.TensorProto.EXTERNAL or message.external_data:
            raise ValueError("ONNX model references external tensor data")
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            children = value if field.is_repeated else [value]
            for child in children:
                reject_external_tensors(child, onnx_module)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(args: argparse.Namespace) -> dict[str, Any]:
    config, config_hash = read_config(args.export_config)
    decision, decision_hash = predict_model.prediction_config(args.prediction_config)
    taxonomy, taxonomy_hash = evaluate.load_taxonomy(args.taxonomy)
    rows, dataset_hash = train.load_dataset(args.parity_dataset)
    if not rows:
        raise ValueError("parity dataset must be nonempty")
    if max(config["parity_batch_sizes"]) > len(rows):
        raise ValueError("parity dataset is too small for the configured batch sizes")
    if any(not isinstance(row["text"], str) or len(row["text"].encode("utf-8")) > decision["max_input_bytes"] for row in rows):
        raise ValueError("parity input exceeds configured bounds")
    manifest = json.loads((args.model / "run-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "lfg.encoder-run.v1" or type(manifest.get("max_length")) is not int or manifest["max_length"] < 1:
        raise ValueError("invalid model run manifest")
    labels_raw = (args.model / "labels.json").read_bytes()
    labels = json.loads(labels_raw)
    if not isinstance(labels, list) or not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("invalid model labels")
    if len(labels) != len(set(labels)) or set(labels) - set.union(*taxonomy.values()):
        raise ValueError("model labels must be unique and in the taxonomy")
    if set(decision["label_thresholds"]) - set.union(*taxonomy.values()):
        raise ValueError("decision threshold label is outside the taxonomy")
    artifact_hash = predict_model.artifact_digest(args.model)
    prediction_raw = args.prediction_config.read_bytes()
    taxonomy_raw = args.taxonomy.read_bytes()
    if hashlib.sha256(prediction_raw).hexdigest() != decision_hash or hashlib.sha256(taxonomy_raw).hexdigest() != taxonomy_hash:
        raise ValueError("configuration changed during export")

    import numpy as np
    import onnx
    os.environ['ORT_DISABLE_TELEMETRY'] = '1'  # Set before import, or model names and hashes go to Microsoft.
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(config["cpu_threads"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False, use_safetensors=True).to("cpu").eval()
    if model.config.num_labels != len(labels):
        raise ValueError("model output count does not match labels")
    example = tokenizer([rows[0]["text"]], truncation=True, padding=True,
                        max_length=manifest["max_length"], return_tensors="pt")
    names = list(example.keys())
    if (not names or "input_ids" not in names or set(names) - {"input_ids", "token_type_ids", "attention_mask"}
            or any(value.ndim != 2 for value in example.values())):
        raise ValueError("unsupported tokenizer input shape")

    class Logits(torch.nn.Module):
        def __init__(self, inner: Any):
            super().__init__()
            self.inner = inner

        def forward(self, *values: Any) -> Any:
            return self.inner(**dict(zip(names, values, strict=True))).logits

    args.output.mkdir(parents=True, exist_ok=False)
    model_path = args.output / "model.onnx"
    dynamic_axes = {name: {0: "batch", 1: "tokens"} for name in names}
    dynamic_axes["logits"] = {0: "batch"}
    wrapper = Logits(model).eval()
    torch.onnx.export(wrapper, tuple(example[name] for name in names), str(model_path),
                      input_names=names, output_names=["logits"], dynamic_axes=dynamic_axes,
                      opset_version=config["opset_version"], dynamo=False, external_data=False)
    model.eval()
    if model_path.stat().st_size > config["max_model_bytes"]:
        raise ValueError("exported model exceeds configured size limit")
    graph = onnx.load(str(model_path), load_external_data=False)
    reject_external_tensors(graph, onnx)
    onnx.checker.check_model(graph, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = config["cpu_threads"]
    options.inter_op_num_threads = config["cpu_threads"]
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    if [item.name for item in session.get_inputs()] != names or [item.name for item in session.get_outputs()] != ["logits"]:
        raise ValueError("exported input or output contract differs from the tokenizer")
    worst = 0.0
    cases = 0
    shapes: set[tuple[int, int]] = set()
    thresholds = np.array([decision["label_thresholds"].get(label, decision["default_threshold"]) for label in labels])
    with torch.inference_mode():
        for batch_size in config["parity_batch_sizes"]:
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                inputs = tokenizer([row["text"] for row in batch], truncation=True, padding=True,
                                   max_length=manifest["max_length"], return_tensors="pt")
                if list(inputs.keys()) != names:
                    raise ValueError("tokenizer input names changed across batches")
                reference = model(**inputs).logits.numpy()
                actual = session.run(["logits"], {name: inputs[name].numpy() for name in names})[0]
                if reference.shape != actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
                    raise ValueError("ONNX output is non-finite or has an unexpected shape")
                if not np.allclose(reference, actual, rtol=config["relative_tolerance"], atol=config["absolute_tolerance"]):
                    raise ValueError("ONNX logits differ beyond configured tolerance")
                expected_decisions = torch.sigmoid(torch.from_numpy(reference)).numpy() >= thresholds
                actual_decisions = torch.sigmoid(torch.from_numpy(actual)).numpy() >= thresholds
                if not np.array_equal(expected_decisions, actual_decisions):
                    raise ValueError("ONNX classification differs at the configured thresholds")
                worst = max(worst, float(np.max(np.abs(reference - actual))))
                shapes.add(tuple(inputs[names[0]].shape))
                cases += len(batch)
    if predict_model.artifact_digest(args.model) != artifact_hash:
        raise ValueError("source model changed during export")
    tokenizer.save_pretrained(args.output)
    tokenizer_path = args.output / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise ValueError("native shadow requires a serialized fast tokenizer")
    tokenizer_data = json.loads(tokenizer_path.read_bytes())
    truncation = tokenizer_data.get("truncation") or {}
    padding = tokenizer_data.get("padding") or {}
    if (truncation.get("max_length") != manifest["max_length"]
            or str(truncation.get("direction", "")).lower() != tokenizer.truncation_side
            or truncation.get("strategy") != "LongestFirst" or truncation.get("stride") != 0
            or padding.get("strategy") != "BatchLongest" or padding.get("pad_to_multiple_of") is not None
            or str(padding.get("direction", "")).lower() != tokenizer.padding_side):
        raise ValueError("tokenizer settings are unsupported by native shadow")
    (args.output / "labels.json").write_bytes(labels_raw)
    (args.output / "prediction-config.json").write_bytes(prediction_raw)
    (args.output / "taxonomy.json").write_bytes(taxonomy_raw)
    files = {}
    for path in sorted(args.output.iterdir()):
        if not path.is_file() or path.is_symlink():
            raise ValueError("unexpected non-file output")
        files[path.name] = file_digest(path)
    result = {"schema_version": "lfg.encoder-onnx-artifact.v2", "promotion_eligible": False,
              "source_artifact_sha256": artifact_hash, "export_config_sha256": config_hash,
              "prediction_config_sha256": decision_hash, "taxonomy_sha256": taxonomy_hash,
              "parity_dataset_sha256": dataset_hash, "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "inputs": names, "outputs": ["logits"], "opset_version": config["opset_version"],
              "max_length": manifest["max_length"], "padding_side": tokenizer.padding_side,
              "truncation_side": tokenizer.truncation_side,
              "cases": cases, "observed_batch_token_shapes": sorted([list(shape) for shape in shapes]),
              "max_absolute_logit_error": worst, "absolute_tolerance": config["absolute_tolerance"],
              "relative_tolerance": config["relative_tolerance"], "decision_mismatches": 0,
              "python": sys.version, "platform": platform.platform(),
              "packages": train.package_versions(("torch", "transformers", "numpy", "onnx", "onnxruntime", "tokenizers", "safetensors")),
              "files_sha256": files}
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--parity-dataset", required=True, type=Path)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--prediction-config", required=True, type=Path)
    parser.add_argument("--export-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = export(args)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, ImportError) as error:
        raise SystemExit(f"ONNX export rejected: {error}") from error
    print(json.dumps({"output": str(args.output), "cases": result["cases"],
                      "max_absolute_logit_error": result["max_absolute_logit_error"],
                      "decision_mismatches": result["decision_mismatches"], "promotion_eligible": False}))


if __name__ == "__main__":
    main()
