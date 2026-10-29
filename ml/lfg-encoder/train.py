#!/usr/bin/env python3
"""Validate or train the experimental multi-label LFG encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "synthetic.v1.jsonl"
POLICY = ROOT / "promotion-policy.json"


def load_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
    return records


def validate(args: argparse.Namespace) -> None:
    records = load_records(args.dataset)
    labels = sorted({label for record in records for key in ("intents", "capabilities", "risks") for label in record[key]})
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    print(json.dumps({
        "schema_version": "lfg.encoder-dataset-report.v1",
        "records": len(records),
        "verified": sum(bool(record["verified"]) for record in records),
        "families": sorted({record["family"] for record in records}),
        "labels": labels,
        "sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
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
    records = load_records(args.dataset)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    records = select_training_records(records, policy["minimum_human_verified_examples"], args.allow_below_gate)
    try:
        train_rows, validation_rows = grouped_split(records, args.seed)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        import numpy as np
        from datasets import Dataset
        from sklearn.preprocessing import MultiLabelBinarizer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as error:
        raise SystemExit("Install requirements.txt before training") from error

    combined = [sorted(set(row["intents"] + row["capabilities"] + row["risks"])) for row in records]
    binarizer = MultiLabelBinarizer().fit(combined)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    def make_dataset(rows: list[dict[str, Any]]) -> Any:
        labels = [sorted(set(row["intents"] + row["capabilities"] + row["risks"])) for row in rows]
        return Dataset.from_dict({"text": [row["text"] for row in rows], "labels": binarizer.transform(labels).astype(np.float32)})

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "lfg.encoder-development-split.v1",
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "seed": args.seed,
        "development_override": args.allow_below_gate,
        "train_ids": [row["id"] for row in train_rows],
        "validation_ids": [row["id"] for row in validation_rows],
        "train_families": sorted({row["family"] for row in train_rows}),
        "validation_families": sorted({row["family"] for row in validation_rows}),
    }
    (args.output / "split-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)

    train_data = make_dataset(train_rows).map(tokenize, batched=True)
    validation_data = make_dataset(validation_rows).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=len(binarizer.classes_), problem_type="multi_label_classification")
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(args.output), seed=args.seed, num_train_epochs=args.epochs, per_device_train_batch_size=8, per_device_eval_batch_size=8, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True), train_dataset=train_data, eval_dataset=validation_data)
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "labels.json").write_text(json.dumps(binarizer.classes_.tolist(), indent=2), encoding="utf-8")
    print(f"trained experimental model at {args.output}; ONNX export and promotion remain separate gates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate", "train"])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "experimental")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--allow-below-gate", action="store_true", help="development only; output remains ineligible")
    args = parser.parse_args()
    validate(args) if args.command == "validate" else train(args)


if __name__ == "__main__":
    main()
