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
SUPPORT_POLICY = ROOT / "training-support-policy.v1.json"


def load_promotion_policy(path: Path) -> dict[str, Any]:
    return parse_promotion_policy(path.read_bytes())


def parse_promotion_policy(raw: bytes) -> dict[str, Any]:
    policy = json.loads(raw)
    if not isinstance(policy, dict) or policy.get("schema_version") != "lfg.encoder-promotion.v1":
        raise ValueError("unsupported promotion policy")
    for field in ("minimum_human_verified_examples", "maximum_p95_micros_512_tokens"):
        if type(policy.get(field)) is not int or policy[field] < 1:
            raise ValueError(f"invalid {field}")
    for field in ("minimum_intent_capability_macro_f1", "minimum_security_recall",
                  "maximum_expected_calibration_error", "maximum_quantization_quality_loss"):
        value = policy.get(field)
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError(f"invalid {field}")
    return policy


def load_training_config(path: Path) -> dict[str, Any]:
    return parse_training_config(path.read_bytes())


def parse_training_config(raw: bytes) -> dict[str, Any]:
    config = json.loads(raw)
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


def load_dataset(path: Path, *, allow_unlabeled_review: bool = False) -> tuple[list[dict[str, Any]], str]:
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
        pending = record.get("labels_pending", False)
        if type(pending) is not bool:
            raise ValueError(f"record {record['id']} has invalid labels_pending")
        if pending and (not allow_unlabeled_review or record["verified"] or
                        any(record[field] != [] for field in ("intents", "capabilities", "risks"))):
            raise ValueError(f"record {record['id']} has pending labels and is not usable here")
        if not record["text"].strip() or (not record["intents"] and not pending):
            raise ValueError(f"record {record['id']} has empty text or intents")
        seen.add(record["id"])
    return records, hashlib.sha256(raw).hexdigest()


def validate(args: argparse.Namespace) -> None:
    records, dataset_digest = load_dataset(args.dataset)
    labels = sorted({label for record in records for key in ("intents", "capabilities", "risks") for label in record[key]})
    policy_path = getattr(args, "promotion_policy", POLICY)
    policy = load_promotion_policy(policy_path)
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


def validate_validation_labels(training: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    def labels(rows: list[dict[str, Any]]) -> set[str]:
        return {label for row in rows for field in ("intents", "capabilities", "risks") for label in row[field]}
    unseen = labels(validation) - labels(training)
    if unseen:
        raise ValueError(f"validation contains labels absent from the training partition: {sorted(unseen)}")


def training_support(training: list[dict[str, Any]], validation: list[dict[str, Any]],
                     taxonomy: dict[str, set[str]], raw_policy: bytes) -> dict[str, Any]:
    """Count reviewed positive and negative labels; never inspect the test partition."""
    policy = json.loads(raw_policy)
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "defaults", "overrides"} or \
            policy.get("schema_version") != "lfg.encoder-training-support.v1":
        raise ValueError("unsupported training support policy")
    defaults, overrides = policy["defaults"], policy["overrides"]
    splits = ("train", "validation")
    fields = ("minimum_positive", "minimum_negative")
    if not isinstance(defaults, dict) or set(defaults) != set(splits) or not isinstance(overrides, dict):
        raise ValueError("invalid training support defaults or overrides")
    all_labels = set.union(*taxonomy.values())
    if set(overrides) - all_labels:
        raise ValueError("training support override names an unknown label")
    for split in splits:
        rule = defaults[split]
        if not isinstance(rule, dict) or set(rule) != set(fields) or \
                any(type(rule[name]) is not int or rule[name] < 1 for name in fields):
            raise ValueError(f"invalid {split} training support defaults")
    for label, rules in overrides.items():
        if not isinstance(rules, dict) or not rules or set(rules) - set(splits):
            raise ValueError(f"invalid training support override for {label}")
        for split, rule in rules.items():
            if not isinstance(rule, dict) or not rule or set(rule) - set(fields) or \
                    any(type(value) is not int or value < 1 for value in rule.values()):
                raise ValueError(f"invalid {split} training support override for {label}")
    result: dict[str, Any] = {}
    failures = []
    for split, rows in (("train", training), ("validation", validation)):
        result[split] = {}
        for group, labels in taxonomy.items():
            for label in sorted(labels):
                rule = {**defaults[split], **overrides.get(label, {}).get(split, {})}
                positives = sum(label in row[group] for row in rows)
                negatives = len(rows) - positives
                result[split][label] = {"positive": positives, "negative": negatives, **rule}
                if positives < rule["minimum_positive"] or negatives < rule["minimum_negative"]:
                    failures.append(f"{split}:{label} has {positives} positive/{negatives} negative; "
                                    f"requires {rule['minimum_positive']}/{rule['minimum_negative']}")
    if failures:
        raise ValueError("training support gate blocked: " + "; ".join(failures))
    return {"schema_version": "lfg.encoder-training-support-result.v1",
            "policy_sha256": hashlib.sha256(raw_policy).hexdigest(),
            "counts": result}


def train(args: argparse.Namespace) -> None:
    input_records, dataset_digest = load_dataset(args.dataset)
    policy_path = getattr(args, "promotion_policy", POLICY)
    policy_bytes = policy_path.read_bytes()
    policy = parse_promotion_policy(policy_bytes)
    records = select_training_records(input_records, policy["minimum_human_verified_examples"], args.allow_below_gate)
    corpus_audit = None
    support = None
    taxonomy = None
    if not args.allow_below_gate:
        if len(records) != len(input_records):
            raise SystemExit("training corpus gate blocked: remove unverified examples from the training input")
        if args.audited_plan is None:
            raise SystemExit("training gate blocked: --audited-plan is required for non-development training")
        try:
            import corpus_audit as audit_module
            audit_report, validation_rows = audit_module.load_reviewed_validation(records, args.audited_plan)
            train_rows = records
            plan_raw = args.audited_plan.read_bytes()
            if hashlib.sha256(plan_raw).hexdigest() != audit_report["plan_sha256"]:
                raise ValueError("plan changed during support validation")
            import evaluate as evaluation_module
            taxonomy, taxonomy_hash = evaluation_module.load_taxonomy(
                audit_module.resolve(args.audited_plan.parent, json.loads(plan_raw)["taxonomy"]))
            if taxonomy_hash != audit_report["taxonomy_sha256"]:
                raise ValueError("taxonomy changed during support validation")
            support = training_support(train_rows, validation_rows, taxonomy, args.support_policy.read_bytes())
            corpus_audit = {key: audit_report[key] for key in (
                "schema_version", "plan_sha256", "taxonomy_sha256", "review_ledger_sha256",
                "auditor_sha256", "taxonomy_loader_sha256", "sources", "counts", "duplicate_within_split")}
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise SystemExit(f"training corpus gate blocked: {error}") from error
    try:
        if args.allow_below_gate:
            train_rows, validation_rows = grouped_split(records, args.seed)
        validate_validation_labels(train_rows, validation_rows)
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
    # Reviewed training always exposes the complete audited taxonomy rather
    # than silently dropping labels that happen to be absent from the data.
    classes = sorted(set.union(*taxonomy.values())) if taxonomy else None
    binarizer = MultiLabelBinarizer(classes=classes).fit(combined)
    split = {
        "schema_version": "lfg.encoder-development-split.v1",
        "dataset_sha256": dataset_digest,
        "seed": args.seed,
        "development_override": args.allow_below_gate,
        "training_support": support,
        "validation_source": "grouped-development-split" if args.allow_below_gate else "audited-validation-partition",
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
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_config_sha256": args.training_config_sha256,
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
    parser.add_argument("--promotion-policy", type=Path, default=POLICY)
    parser.add_argument("--support-policy", type=Path, default=SUPPORT_POLICY,
                        help="reviewed train/validation positive and negative support requirements")
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
        config_bytes = args.training_config.read_bytes()
        config = parse_training_config(config_bytes)
        args.training_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
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
