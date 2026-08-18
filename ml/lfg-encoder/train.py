#!/usr/bin/env python3
"""Validate or train the experimental multi-label LFG encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
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
        "promotion_ready": len(records) >= policy["minimum_human_verified_examples"],
    }, indent=2))


def train(args: argparse.Namespace) -> None:
    records = load_records(args.dataset)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if len(records) < policy["minimum_human_verified_examples"] and not args.allow_below_gate:
        raise SystemExit(f"training gate blocked: {len(records)} verified examples; {policy['minimum_human_verified_examples']} required")
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
    dataset = Dataset.from_dict({"text": [row["text"] for row in records], "labels": binarizer.transform(combined).astype(np.float32)})
    dataset = dataset.train_test_split(test_size=max(1, len(records) // 5), seed=args.seed)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=len(binarizer.classes_), problem_type="multi_label_classification")
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(args.output), seed=args.seed, num_train_epochs=args.epochs, per_device_train_batch_size=8, per_device_eval_batch_size=8, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True), train_dataset=tokenized["train"], eval_dataset=tokenized["test"])
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
