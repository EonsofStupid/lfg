#!/usr/bin/env python3
"""Validate or train the experimental multi-label LFG encoder."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "synthetic.v1.jsonl"
POLICY = ROOT / "promotion-policy.json"
TRAINING_CONFIG = ROOT / "training-config.v1.json"


def load_training_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_training_config(config)
    return config


def validate_training_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "lfg.encoder-training-config.v1":
        raise ValueError("unsupported training configuration")
    if not isinstance(config.get("base_model"), str) or not config["base_model"].strip():
        raise ValueError("base_model must be non-empty")
    for field in ("seed", "max_length", "batch_size", "checkpoint_steps"):
        if type(config.get(field)) is not int or config[field] < (0 if field == "seed" else 1):
            raise ValueError(f"invalid {field}")
    if type(config.get("epochs")) not in (int, float) or not 0 < config["epochs"] < float("inf"):
        raise ValueError("invalid epochs")
    if type(config.get("allow_mismatched_sizes")) is not bool:
        raise ValueError("invalid allow_mismatched_sizes")


def prepare_run(output: Path, manifest: dict[str, Any], resume: bool) -> None:
    """Do not mix unrelated experiments or overwrite an existing run."""
    manifest_path = output / "run-manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise ValueError("resume requires an existing run manifest")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("resume refused: run configuration, code, dataset, or environment differs")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output directory is not empty; choose a new directory or resume")
        output.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_checkpoint(path: Path) -> None:
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    missing = [name for name in required if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if not any((path / name).is_file() and (path / name).stat().st_size > 0 for name in
               ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")):
        missing.append("model weights")
    if missing:
        raise ValueError(f"incomplete checkpoint, missing {', '.join(missing)}")
    try:
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        if not isinstance(state, dict) or type(state.get("global_step")) is not int:
            raise ValueError("trainer state has no global step")
    except (OSError, ValueError) as error:
        raise ValueError(f"incomplete checkpoint: invalid trainer state: {error}") from error


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@contextlib.contextmanager
def run_lock(output: Path):
    """Refuse concurrent writers; a crash leaves a lock requiring operator review."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError(f"run lock exists: {lock}; verify no trainer is active before removing it") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def load_records(path: Path) -> list[dict[str, Any]]:
    return load_dataset(path)[0]


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    required = {"id", "text", "intents", "capabilities", "risks", "verified", "family"}
    seen: set[str] = set()
    for index, record in enumerate(records, 1):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"record {index} missing {sorted(missing)}")
        if record["id"] in seen:
            raise ValueError(f"duplicate id {record['id']}")
        if not isinstance(record["verified"], bool):
            raise ValueError(f"record {record['id']} verified must be a boolean")
        if not isinstance(record["family"], str) or not record["family"].strip():
            raise ValueError(f"record {record['id']} family must be a non-empty string")
        if not record["text"].strip() or not record["intents"]:
            raise ValueError(f"record {record['id']} has empty text or intents")
        seen.add(record["id"])
    return records, hashlib.sha256(raw).hexdigest()


def validate(args: argparse.Namespace) -> None:
    records, dataset_digest = load_dataset(args.dataset)
    labels = sorted({label for record in records for key in ("intents", "capabilities", "risks") for label in record[key]})
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    print(json.dumps({
        "schema_version": "lfg.encoder-dataset-report.v1",
        "records": len(records),
        "verified": sum(bool(record["verified"]) for record in records),
        "families": sorted({record["family"] for record in records}),
        "labels": labels,
        "sha256": dataset_digest,
        "sample_gate_passed": sum(record["verified"] for record in records) >= policy["minimum_human_verified_examples"],
    }, indent=2))


def grouped_split(records: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep each explicitly assigned family entirely in one development split."""
    families = sorted({record["family"] for record in records})
    if len(families) < 2:
        raise ValueError("training requires at least two distinct families for a grouped split")
    random.Random(seed).shuffle(families)
    validation_families = set(families[:max(1, len(families) // 5)])
    return ([record for record in records if record["family"] not in validation_families],
            [record for record in records if record["family"] in validation_families])


def select_training_records(records: list[dict[str, Any]], minimum: int, development: bool) -> list[dict[str, Any]]:
    verified = [record for record in records if record["verified"]]
    if not development and len(verified) < minimum:
        raise SystemExit(f"training gate blocked: {len(verified)} verified examples; {minimum} required")
    return records if development else verified


def train(args: argparse.Namespace) -> None:
    records, dataset_digest = load_dataset(args.dataset)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    records = select_training_records(records, policy["minimum_human_verified_examples"], args.allow_below_gate)
    corpus_audit = None
    if not args.allow_below_gate:
        if args.audited_plan is None:
            raise SystemExit("training gate blocked: --audited-plan is required for non-development training")
        try:
            import corpus_audit as audit_module
            corpus_audit = audit_module.validate_training_corpus(records, args.audited_plan)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise SystemExit(f"training corpus gate blocked: {error}") from error
    try:
        train_rows, validation_rows = grouped_split(records, args.seed)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision or ""):
        raise SystemExit("training requires --model-revision as a full 40-character lowercase commit hash")
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from sklearn.preprocessing import MultiLabelBinarizer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as error:
        raise SystemExit("Install requirements.txt before training") from error

    combined = [sorted(set(row["intents"] + row["capabilities"] + row["risks"])) for row in records]
    binarizer = MultiLabelBinarizer().fit(combined)
    split = {
        "schema_version": "lfg.encoder-development-split.v1",
        "dataset_sha256": dataset_digest,
        "seed": args.seed,
        "development_override": args.allow_below_gate,
        "corpus_audit": corpus_audit,
        "train_ids": [row["id"] for row in train_rows],
        "validation_ids": [row["id"] for row in validation_rows],
        "train_families": sorted({row["family"] for row in train_rows}),
        "validation_families": sorted({row["family"] for row in validation_rows}),
    }
    run_manifest = {
        "schema_version": "lfg.encoder-run.v1",
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "dataset_sha256": split["dataset_sha256"],
        "policy_sha256": hashlib.sha256(POLICY.read_bytes()).hexdigest(),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_config_sha256": hashlib.sha256(args.training_config.read_bytes()).hexdigest(),
        "seed": args.seed, "epochs": args.epochs, "max_length": args.max_length,
        "batch_size": args.batch_size, "checkpoint_steps": args.checkpoint_steps,
        "allow_mismatched_sizes": args.allow_mismatched_sizes,
        "development_override": args.allow_below_gate,
        "split": split,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(("torch", "transformers", "datasets", "accelerate", "numpy",
                                      "scikit-learn", "scipy", "tokenizers", "huggingface-hub", "safetensors", "pyarrow")),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    try:
        prepare_run(args.output, run_manifest, args.resume)
        checkpoint = get_last_checkpoint(str(args.output)) if args.resume else None
        if args.resume:
            if checkpoint is None:
                raise ValueError("resume requested but no checkpoint exists")
            validate_checkpoint(Path(checkpoint))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not args.resume:
        (args.output / "split-manifest.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.model_revision)
    def make_dataset(rows: list[dict[str, Any]]) -> Any:
        labels = [sorted(set(row["intents"] + row["capabilities"] + row["risks"])) for row in rows]
        return Dataset.from_dict({"text": [row["text"] for row in rows], "labels": binarizer.transform(labels).astype(np.float32)})

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)

    train_data = make_dataset(train_rows).map(tokenize, batched=True)
    validation_data = make_dataset(validation_rows).map(tokenize, batched=True)
    def model_init() -> Any:
        return AutoModelForSequenceClassification.from_pretrained(
            args.base_model, revision=args.model_revision, num_labels=len(binarizer.classes_),
            problem_type="multi_label_classification", ignore_mismatched_sizes=args.allow_mismatched_sizes)
    trainer = Trainer(model_init=model_init, args=TrainingArguments(
        output_dir=str(args.output), seed=args.seed, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps", eval_steps=args.checkpoint_steps, save_strategy="steps",
        save_steps=args.checkpoint_steps, load_best_model_at_end=True, save_only_model=False,
        report_to="none"), train_dataset=train_data, eval_dataset=validation_data)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "labels.json").write_text(json.dumps(binarizer.classes_.tolist(), indent=2), encoding="utf-8")
    print(f"trained experimental model at {args.output}; ONNX export and promotion remain separate gates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate", "train"])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--training-config", type=Path, default=TRAINING_CONFIG)
    parser.add_argument("--base-model")
    parser.add_argument("--model-revision", help="full immutable Hugging Face commit hash")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "experimental")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint-steps", type=int)
    parser.add_argument("--allow-mismatched-sizes", action=argparse.BooleanOptionalAction, default=None,
                        help="permit reinitialization of mismatched weights (review model loading warnings)")
    parser.add_argument("--resume", action="store_true", help="resume the latest complete checkpoint of the same run")
    parser.add_argument("--allow-below-gate", action="store_true", help="development only; output remains ineligible")
    parser.add_argument("--audited-plan", type=Path, help="reviewed and project-separated corpus plan")
    args = parser.parse_args()
    try:
        config = load_training_config(args.training_config)
        fields = ("base_model", "seed", "epochs", "max_length", "batch_size", "checkpoint_steps", "allow_mismatched_sizes")
        for field in fields:
            if getattr(args, field) is None:
                setattr(args, field, config[field])
        effective_config = {**config, **{field: getattr(args, field) for field in fields}}
        # Validate command-line overrides by the same rules as the configuration.
        validate_training_config(effective_config)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"training configuration: {error}") from error
    if args.command == "validate":
        validate(args)
    else:
        try:
            with run_lock(args.output):
                train(args)
        except ValueError as error:
            raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
