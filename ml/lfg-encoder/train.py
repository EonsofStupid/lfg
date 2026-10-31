#!/usr/bin/env python3
"""Validate, plan, or GPU-train the experimental LFG prompt encoder."""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import stat
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
    config.setdefault("device", "auto")
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
    if config.get("device", "auto") not in ("auto", "cuda"):
        raise ValueError("device must be auto or cuda; training is GPU-only")


def verify_training_device(requested: str, selected: str, cuda_available: bool) -> None:
    """Fail closed when an explicit training target is unavailable or ignored."""
    if requested not in ("auto", "cuda"):
        raise ValueError("invalid requested training device")
    if not cuda_available:
        raise ValueError("CUDA is unavailable; GPU-only training cannot fall back to CPU")
    if selected != "cuda":
        raise ValueError(f"training device mismatch: requested {requested}, selected {selected}")


def make_training_arguments(args: argparse.Namespace, torch: Any, arguments_class: Any) -> tuple[Any, bool]:
    cuda_available = torch.cuda.is_available()
    verify_training_device(args.device, "cuda", cuda_available)
    result = arguments_class(
        output_dir=str(args.output), seed=args.seed, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps", eval_steps=args.checkpoint_steps, save_strategy="steps",
        save_steps=args.checkpoint_steps, load_best_model_at_end=True, save_only_model=False,
        use_cpu=False, report_to="none")
    verify_training_device(args.device, result.device.type, cuda_available)
    return result, cuda_available


def make_final_evaluation_callback(callback_base: Any) -> Any:
    class FinalEvaluationCallback(callback_base):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if state.max_steps > 0 and state.global_step >= state.max_steps:
                control.should_evaluate = True
                control.should_save = True
            return control
    return FinalEvaluationCallback()


def make_trainer(model_init: Any, arguments: Any, train_data: Any, validation_data: Any,
                 tokenizer: Any, trainer_class: Any, collator_class: Any, callback_base: Any) -> Any:
    """Pad both datasets and evaluate the final checkpoint even between intervals."""
    return trainer_class(model_init=model_init, args=arguments, train_dataset=train_data,
                         eval_dataset=validation_data, data_collator=collator_class(tokenizer=tokenizer),
                         callbacks=[make_final_evaluation_callback(callback_base)])


def validate_selected_checkpoint(state: Any) -> None:
    if (state.best_model_checkpoint is None or not Path(state.best_model_checkpoint).is_dir()
            or state.best_metric is None or not math.isfinite(state.best_metric)):
        raise ValueError("training ended without a checkpoint selected by a finite validation loss")


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


RUN_LOCK_MAGIC = b"lfg.training-lock.v2\n"


@contextlib.contextmanager
def run_lock(output: Path):
    """Preserve the legacy exclusive-create lock used by review workspaces."""
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


@contextlib.contextmanager
def training_run_lock(output: Path):
    """Hold an OS lock on a persistent marker; do not unlink the shared inode."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".lock")
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(lock, flags | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        descriptor = os.open(lock, flags)
        created = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"run lock is not a regular file: {lock}")
        if created:
            if os.write(descriptor, RUN_LOCK_MAGIC) != len(RUN_LOCK_MAGIC):
                raise OSError("could not write complete training lock marker")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            elif os.name == "posix":
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                raise ValueError("training locks are unsupported on this platform")
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK) or getattr(error, "winerror", None) in (33, 36):
                raise ValueError(f"run lock exists and is held by another trainer: {lock}") from error
            raise
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, len(RUN_LOCK_MAGIC) + 1) != RUN_LOCK_MAGIC:
            raise ValueError(f"unrecognized or legacy run lock: {lock}; verify no trainer is active before removing it")
        yield
    finally:
        os.close(descriptor)


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
    verified = sum(record["verified"] for record in records)
    minimum = policy["minimum_human_verified_examples"]
    print(json.dumps({
        "schema_version": "lfg.encoder-dataset-report.v1",
        "records": len(records),
        "verified": verified,
        "unverified": len(records) - verified,
        "required_verified": minimum,
        "families": sorted({str(record["family"]) for record in records}),
        "labels": labels,
        "sha256": dataset_digest,
        "sample_gate_passed": verified >= minimum,
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


def validate_input_options(development: bool, environment_lock: Path | None, wheelhouse: Path | None,
                           model_manifest: Path | None, allowed_extra: list[str]) -> None:
    if bool(environment_lock) != bool(wheelhouse) or (allowed_extra and not environment_lock):
        raise ValueError("environment lock and wheelhouse must be supplied together")
    if not development and not (environment_lock and wheelhouse and model_manifest):
        raise ValueError("reviewed training requires an environment lock, wheelhouse, and local model manifest")


def effective_token_limit(tokenizer: Any, maximum: int) -> int:
    if type(maximum) is not int or maximum < 1:
        raise ValueError("token coverage requires a positive configured limit")
    advertised = getattr(tokenizer, "model_max_length", None)
    if advertised is None:
        return maximum
    if type(advertised) is not int or advertised < 1:
        raise ValueError("invalid tokenizer model_max_length")
    return min(maximum, advertised)


def token_coverage(rows: list[dict[str, Any]], tokenizer: Any, maximum: int, split: str) -> dict[str, int]:
    """Measure complete inputs, including special tokens, without recording their text."""
    limit = effective_token_limit(tokenizer, maximum)
    if not rows:
        raise ValueError("token coverage requires a nonempty split")
    lengths = []
    for row in rows:
        encoded = tokenizer(row["text"], add_special_tokens=True, truncation=False, padding=False)
        ids = encoded.get("input_ids")
        if not isinstance(ids, list) or not ids or any(type(token) is not int for token in ids):
            raise ValueError(f"invalid tokenizer output for {split}")
        lengths.append(len(ids))
    over = sum(length > limit for length in lengths)
    result = {"examples": len(lengths), "maximum_tokens": max(lengths), "over_limit": over,
              "effective_limit": limit}
    if over:
        raise ValueError(f"{split} token coverage gate blocked: {over} of {len(lengths)} examples "
                         f"exceed effective maximum {limit} (configured maximum {maximum}); "
                         f"longest has {max(lengths)} tokens")
    return result


def tokenize_complete_batch(batch: dict[str, Any], tokenizer: Any, maximum: int) -> dict[str, Any]:
    limit = effective_token_limit(tokenizer, maximum)
    encoded = tokenizer(batch["text"], add_special_tokens=True, truncation=False, padding=False)
    ids = encoded.get("input_ids")
    if (not isinstance(ids, list) or len(ids) != len(batch["text"])
            or any(not isinstance(tokens, list) or not tokens or len(tokens) > limit
                   or any(type(token) is not int for token in tokens) for tokens in ids)):
        raise ValueError("tokenizer batch is invalid or exceeds the effective maximum")
    return encoded


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
            audit_report, validation_rows = audit_module.load_reviewed_validation(
                records, args.audited_plan, development_only=getattr(args, "development_plan_only", False))
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
            if audit_report.get("scope") == "development":
                corpus_audit.update(scope="development", holdout_assessment="not_performed")
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
    environment_lock = getattr(args, "environment_lock", None)
    wheelhouse = getattr(args, "wheelhouse", None)
    model_manifest = getattr(args, "model_manifest", None)
    allowed_extra = getattr(args, "environment_allowed_extra", [])
    validate_input_options(args.allow_below_gate, environment_lock, wheelhouse, model_manifest, allowed_extra)
    environment_evidence = model_evidence = None
    verify_payload = not args.allow_below_gate or getattr(args, "verify_installed_payload", False)
    if verify_payload and not environment_lock:
        raise ValueError("installed payload verification requires an environment lock and wheelhouse")
    if environment_lock or model_manifest:
        import training_inputs
        if environment_lock:
            environment_evidence = training_inputs.verify_environment(environment_lock, wheelhouse, allowed_extra,
                                                                      verify_payload=verify_payload)
        if model_manifest:
            model_evidence = training_inputs.verify_model(Path(args.base_model), model_manifest, args.model_revision)
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from sklearn.metrics import f1_score, recall_score
        from sklearn.preprocessing import MultiLabelBinarizer
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                                  Trainer, TrainerCallback, TrainingArguments)
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as error:
        raise SystemExit(
            "Install the locked LFG training dependencies before training."
        ) from error

    try:
        training_arguments, cuda_available = make_training_arguments(args, torch, TrainingArguments)
    except ValueError as error:
        raise SystemExit(f"training device rejected: {error}") from error
    selected_device = training_arguments.device

    combined = [sorted(set(row["intents"] + row["capabilities"] + row["risks"])) for row in records]
    # Reviewed training always exposes the complete audited taxonomy rather
    # than silently dropping labels that happen to be absent from the data.
    classes = sorted(set.union(*taxonomy.values())) if taxonomy else None
    binarizer = MultiLabelBinarizer(classes=classes).fit(combined)
    def check_model_snapshot() -> None:
        if model_manifest and training_inputs.verify_model(Path(args.base_model), model_manifest, args.model_revision) != model_evidence:
            raise ValueError("model snapshot evidence changed")

    check_model_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.model_revision,
                                               local_files_only=bool(model_manifest))
    try:
        coverage = {"train": token_coverage(train_rows, tokenizer, args.max_length, "train"),
                    "validation": token_coverage(validation_rows, tokenizer, args.max_length, "validation")}
    except ValueError as error:
        raise SystemExit(str(error)) from error
    check_model_snapshot()
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
        "environment_evidence": environment_evidence,
        "model_snapshot_evidence": model_evidence,
        "training_inputs_sha256": hashlib.sha256(Path(training_inputs.__file__).read_bytes()).hexdigest() if environment_lock or model_manifest else None,
        "wheelhouse_verifier_sha256": hashlib.sha256(Path(training_inputs.lock_wheelhouse.__file__).read_bytes()).hexdigest() if environment_lock else None,
        "seed": args.seed, "epochs": args.epochs, "max_length": args.max_length,
        "token_coverage": coverage,
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
        "requested_device": args.device,
        "selected_device": str(selected_device),
        "cuda_available": cuda_available,
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

    check_model_snapshot()
    def make_dataset(rows: list[dict[str, Any]]) -> Any:
        labels = [sorted(set(row["intents"] + row["capabilities"] + row["risks"])) for row in rows]
        return Dataset.from_dict({"text": [row["text"] for row in rows], "labels": binarizer.transform(labels).astype(np.float32)})

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenize_complete_batch(batch, tokenizer, args.max_length)

    train_data = make_dataset(train_rows).map(tokenize, batched=True)
    validation_data = make_dataset(validation_rows).map(tokenize, batched=True)
    def model_init() -> Any:
        check_model_snapshot()
        return AutoModelForSequenceClassification.from_pretrained(
            args.base_model, revision=args.model_revision, num_labels=len(binarizer.classes_),
            problem_type="multi_label_classification", ignore_mismatched_sizes=args.allow_mismatched_sizes,
            local_files_only=bool(model_manifest))
    trainer = make_trainer(model_init, training_arguments, train_data, validation_data,
                           tokenizer, Trainer, DataCollatorWithPadding, TrainerCallback)
    verify_training_device(args.device, trainer.args.device.type, torch.cuda.is_available())
    verify_training_device(args.device, next(trainer.model.parameters()).device.type, torch.cuda.is_available())
    trainer.train(resume_from_checkpoint=checkpoint)
    validate_selected_checkpoint(trainer.state)
    check_model_snapshot()
    if verify_payload and training_inputs.verify_environment(environment_lock, wheelhouse, allowed_extra,
                                                              verify_payload=True) != environment_evidence:
        raise ValueError("installed wheel payload evidence changed during training")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "labels.json").write_text(json.dumps(binarizer.classes_.tolist(), indent=2), encoding="utf-8")
    trainer.save_state()
    print(f"trained experimental model at {args.output}; ONNX export and promotion remain separate gates")
    # The separately authored GPU prototype below has not been integrated with
    # this reviewed training path and must not run after it.
    return

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        logits, expected = eval_prediction
        if isinstance(logits, tuple):
            logits = logits[0]
        probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
        expected_array = np.asarray(expected)
        predicted = probabilities >= args.threshold
        macro_f1 = f1_score(
            expected_array,
            predicted,
            average="macro",
            zero_division=0,
        )
        security_positives = (
            int(expected_array[:, risk_indices].sum()) if risk_indices else 0
        )
        security_recall = (
            recall_score(
                expected_array[:, risk_indices],
                predicted[:, risk_indices],
                average="macro",
                zero_division=0,
            )
            if security_positives
            else 0.0
        )
        return {
            "macro_f1": float(macro_f1),
            "security_recall": float(security_recall),
            "security_positive_labels": float(security_positives),
            "expected_calibration_error": _expected_calibration_error(
                probabilities,
                expected_array,
            ),
        }

    bf16 = bool(torch.cuda.is_bf16_supported())
    run_manifest = {
        **plan_payload,
        "schema_version": "lfg.encoder-training-run.v1",
        "started_at": datetime.now(UTC).isoformat(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "labels": labels,
        "risk_labels": risk_labels,
        "train_examples": len(tokenized["train"]),
        "evaluation_examples": len(tokenized["test"]),
    }
    updates_per_epoch = math.ceil(
        len(tokenized["train"])
        / (args.batch_size * args.gradient_accumulation_steps)
    )
    total_updates = max(1, math.ceil(updates_per_epoch * args.epochs))
    warmup_steps = round(total_updates * args.warmup_ratio)
    run_manifest["warmup_steps"] = warmup_steps
    _write_json(args.output / "run-manifest.json", run_manifest)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            output_dir=str(args.output / "checkpoints"),
            seed=args.seed,
            data_seed=args.seed,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=warmup_steps,
            eval_strategy="epoch",
            save_strategy="no" if args.development_run else "epoch",
            save_total_limit=2,
            load_best_model_at_end=not args.development_run,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=1,
            report_to="none",
            fp16=not bf16,
            bf16=bf16,
            tf32=True,
            use_cpu=False,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    evaluation = {
        key: float(value) if isinstance(value, (int, float)) else value
        for key, value in trainer.evaluate().items()
    }

    final_dir = args.output / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    _write_json(final_dir / "labels.json", {"labels": labels})
    _write_json(args.output / "evaluation.json", evaluation)

    policy = load_policy()
    quality_gate = (
        evaluation.get("eval_macro_f1", 0.0)
        >= policy["minimum_intent_capability_macro_f1"]
        and evaluation.get("eval_security_recall", 0.0)
        >= policy["minimum_security_recall"]
        and evaluation.get("eval_expected_calibration_error", 1.0)
        <= policy["maximum_expected_calibration_error"]
    )
    candidate = {
        "schema_version": "lfg.encoder-candidate.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "development_run": args.development_run,
        "sample_gate_passed": plan_payload["dataset_report"]["promotion_ready"],
        "quality_gate_passed": quality_gate,
        "promotion_eligible": False,
        "remaining_gates": plan_payload["remaining_gates"][1:],
        "evaluation": evaluation,
        "artifact": str(final_dir.resolve()),
    }
    _write_json(args.output / "candidate.json", candidate)
    print(
        f"trained candidate at {final_dir}; ONNX export and Rust CPU proof remain required"
    )


def export_onnx(args: argparse.Namespace) -> None:
    checkpoint = (args.checkpoint or (args.output / "final")).resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"trained checkpoint not found: {checkpoint}")
    onnx_output = (
        args.onnx_output or (args.output / "onnx" / "model.fp32.onnx")
    ).resolve()
    quantized_output = onnx_output.with_name("model.int8.onnx")
    existing = [path for path in (onnx_output, quantized_output) if path.exists()]
    if existing and not args.force_export:
        raise SystemExit(
            "export output already exists: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --force-export to replace these exact artifacts."
        )
    onnx_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install requirements.txt before ONNX export.") from error

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            return self.wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

    sample_texts = [
        "Research the current routing design and cite primary sources.",
        "Implement the component, run its tests, and do not expose secrets.",
    ]
    encoded = tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_names = ["input_ids", "attention_mask"]
    sample_inputs = tuple(encoded[name] for name in input_names)
    wrapper = LogitsOnly(model).eval()
    with torch.inference_mode():
        reference = wrapper(*sample_inputs).detach().cpu().numpy()

    batch_dimension = torch.export.Dim("batch", min=1)
    sequence_dimension = torch.export.Dim(
        "sequence",
        min=1,
        max=args.max_length,
    )
    torch.onnx.export(
        wrapper,
        sample_inputs,
        str(onnx_output),
        input_names=input_names,
        output_names=["logits"],
        dynamic_shapes=(
            {0: batch_dimension, 1: sequence_dimension},
            {0: batch_dimension, 1: sequence_dimension},
        ),
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    onnx.checker.check_model(onnx.load(onnx_output))

    ort_inputs = {
        name: encoded[name].detach().cpu().numpy().astype(np.int64)
        for name in input_names
    }
    fp32_session = ort.InferenceSession(
        str(onnx_output),
        providers=["CPUExecutionProvider"],
    )
    fp32_logits = fp32_session.run(["logits"], ort_inputs)[0]
    fp32_max_difference = float(np.max(np.abs(reference - fp32_logits)))
    export_parity_passed = fp32_max_difference <= args.export_tolerance

    quantized: dict[str, Any] | None = None
    if not args.skip_quantization:
        # ORT's dynamic quantizer converts the two classifier Gemm nodes to
        # MatMul and reruns ONNX shape inference. Dynamo's intermediate
        # value_info currently conflicts with that conversion (768 hidden
        # features versus 13 labels), so let ORT infer fresh intermediates.
        quantization_source = onnx.load(onnx_output)
        del quantization_source.graph.value_info[:]
        quantize_dynamic(
            model_input=quantization_source,
            model_output=str(quantized_output),
            per_channel=True,
            weight_type=QuantType.QInt8,
        )
        int8_session = ort.InferenceSession(
            str(quantized_output),
            providers=["CPUExecutionProvider"],
        )
        int8_logits = int8_session.run(["logits"], ort_inputs)[0]
        quantized = {
            "path": str(quantized_output),
            "bytes": quantized_output.stat().st_size,
            "sha256": _file_sha256(quantized_output),
            "maximum_logit_difference_from_pytorch": float(
                np.max(np.abs(reference - int8_logits))
            ),
            "smoke_output_finite": bool(np.isfinite(int8_logits).all()),
            "quality_gate_passed": False,
            "quality_gate_note": (
                "Logit smoke parity is not the policy's held-out quantization quality gate."
            ),
        }

    manifest = {
        "schema_version": "lfg.encoder-onnx-export.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint),
        "opset": 18,
        "maximum_length": args.max_length,
        "sample_count": len(sample_texts),
        "fp32": {
            "path": str(onnx_output),
            "bytes": onnx_output.stat().st_size,
            "sha256": _file_sha256(onnx_output),
            "maximum_logit_difference_from_pytorch": fp32_max_difference,
            "tolerance": args.export_tolerance,
            "parity_passed": export_parity_passed,
        },
        "int8": quantized,
        "promotion_eligible": False,
        "remaining_gates": [
            "held-out quantization quality loss",
            "Rust target-CPU p95 latency",
        ],
    }
    manifest_path = args.output / "onnx-manifest.json"
    _write_json(manifest_path, manifest)

    candidate_path = args.output / "candidate.json"
    if candidate_path.is_file():
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate["onnx_export"] = manifest
        if export_parity_passed:
            candidate["remaining_gates"] = [
                gate
                for gate in candidate.get("remaining_gates", [])
                if gate != "ONNX export parity"
            ]
        candidate["promotion_eligible"] = False
        _write_json(candidate_path, candidate)

    if not export_parity_passed:
        raise SystemExit(
            f"ONNX parity failed: maximum logit difference {fp32_max_difference:.6g} "
            f"exceeds {args.export_tolerance:.6g}"
        )
    print(f"ONNX export proven: {manifest_path}")


def benchmark_onnx(args: argparse.Namespace) -> None:
    import platform
    import time

    try:
        import numpy as np
        import onnxruntime as ort
        import psutil
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install requirements.txt before the CPU preflight.") from error

    checkpoint = (args.checkpoint or (args.output / "final")).resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"trained checkpoint not found: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    text = (
        "Research the request, identify the required capabilities and risks, "
        "implement the work, verify the result, and preserve the evidence. "
    ) * 80
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        return_tensors="np",
    )
    ort_inputs = {
        name: np.asarray(encoded[name], dtype=np.int64)
        for name in ("input_ids", "attention_mask")
    }
    policy = load_policy()
    process = psutil.Process()

    def summarize(samples: list[float]) -> dict[str, float]:
        return {
            "p50_micros": float(np.percentile(samples, 50)),
            "p95_micros": float(np.percentile(samples, 95)),
            "minimum_micros": float(min(samples)),
            "maximum_micros": float(max(samples)),
            "mean_micros": float(np.mean(samples)),
        }

    def measure(path: Path) -> dict[str, Any]:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if args.cpu_threads:
            options.intra_op_num_threads = args.cpu_threads
            options.inter_op_num_threads = 1
        rss_before = process.memory_info().rss
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(args.benchmark_warmup):
            session.run(["logits"], ort_inputs)
        rss_after_warmup = process.memory_info().rss

        model_samples: list[float] = []
        end_to_end_samples: list[float] = []
        for _ in range(args.benchmark_iterations):
            started = time.perf_counter_ns()
            session.run(["logits"], ort_inputs)
            model_samples.append((time.perf_counter_ns() - started) / 1000.0)

            started = time.perf_counter_ns()
            current = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
                return_tensors="np",
            )
            session.run(
                ["logits"],
                {
                    name: np.asarray(current[name], dtype=np.int64)
                    for name in ("input_ids", "attention_mask")
                },
            )
            end_to_end_samples.append(
                (time.perf_counter_ns() - started) / 1000.0
            )

        model_latency = summarize(model_samples)
        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "model_only": model_latency,
            "tokenization_and_model": summarize(end_to_end_samples),
            "rss_before_session_bytes": rss_before,
            "rss_after_warmup_bytes": rss_after_warmup,
            "rss_growth_bytes": max(0, rss_after_warmup - rss_before),
            "meets_python_preflight_latency": (
                model_latency["p95_micros"]
                <= policy["maximum_p95_micros_512_tokens"]
            ),
            "authoritative_rust_gate_passed": False,
        }

    onnx_dir = args.output / "onnx"
    artifacts = [onnx_dir / "model.fp32.onnx", onnx_dir / "model.int8.onnx"]
    existing = [path for path in artifacts if path.is_file()]
    if not existing:
        raise SystemExit(f"no ONNX artifacts found under {onnx_dir.resolve()}")
    results = [measure(path) for path in existing]
    report = {
        "schema_version": "lfg.encoder-cpu-preflight.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "authoritative": False,
        "why_not_authoritative": (
            "The promotion policy requires the Rust runtime on the pinned target CPU."
        ),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "onnxruntime": ort.__version__,
        "maximum_length": args.max_length,
        "iterations": args.benchmark_iterations,
        "warmup_iterations": args.benchmark_warmup,
        "intra_op_threads": args.cpu_threads or "onnxruntime-default",
        "policy_p95_micros": policy["maximum_p95_micros_512_tokens"],
        "artifacts": results,
        "promotion_eligible": False,
    }
    report_path = args.output / "cpu-preflight.json"
    _write_json(report_path, report)

    candidate_path = args.output / "candidate.json"
    if candidate_path.is_file():
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate["cpu_preflight"] = report
        candidate["promotion_eligible"] = False
        _write_json(candidate_path, candidate)
    print(f"CPU preflight recorded: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["validate", "plan", "train", "export", "benchmark"],
    )
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--training-config", type=Path, default=TRAINING_CONFIG)
    parser.add_argument("--promotion-policy", type=Path, default=POLICY)
    parser.add_argument("--support-policy", type=Path, default=SUPPORT_POLICY,
                        help="reviewed train/validation positive and negative support requirements")
    parser.add_argument("--base-model")
    parser.add_argument("--model-revision", help="full immutable Hugging Face commit hash")
    parser.add_argument("--model-manifest", type=Path, help="manifest for a local, link-free base-model directory")
    parser.add_argument("--environment-lock", type=Path, help="hash-locked wheelhouse requirements")
    parser.add_argument("--wheelhouse", type=Path, help="exact wheelhouse used to build the environment")
    parser.add_argument("--environment-allowed-extra", action="append", default=[], metavar="DISTRIBUTION",
                        help="explicit installed bootstrap distribution not present in the wheel lock")
    parser.add_argument("--verify-installed-payload", action="store_true",
                        help="also verify installed locked-wheel payloads for below-gate development runs")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "experimental")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint-steps", type=int)
    parser.add_argument("--device", choices=("auto", "cuda"),
                        help="GPU training target; neither option permits CPU fallback")
    parser.add_argument("--allow-mismatched-sizes", action=argparse.BooleanOptionalAction, default=None,
                        help="permit reinitialization of mismatched weights (review model loading warnings)")
    parser.add_argument("--resume", action="store_true", help="resume the latest complete checkpoint of the same run")
    parser.add_argument("--allow-below-gate", action="store_true", help="development only; output remains ineligible")
    parser.add_argument("--audited-plan", type=Path, help="reviewed and project-separated corpus plan")
    parser.add_argument("--development-plan-only", action="store_true",
                        help="reject complete or test-containing plans before opening corpus sources")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command not in ("validate", "train"):
        raise SystemExit(f"{args.command} is not yet integrated with the reviewed training pipeline")
    try:
        config_bytes = args.training_config.read_bytes()
        config = parse_training_config(config_bytes)
        args.training_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        fields = ("base_model", "seed", "epochs", "max_length", "batch_size", "checkpoint_steps", "allow_mismatched_sizes", "device")
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
            with training_run_lock(args.output):
                train(args)
        except (OSError, ValueError, UnicodeError) as error:
            raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
