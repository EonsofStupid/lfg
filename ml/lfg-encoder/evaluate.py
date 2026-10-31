#!/usr/bin/env python3
"""Compare metadata-only predictions with labeled examples; never promote a model."""

from __future__ import annotations

import argparse
import assessment_targets
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import train

FIELDS = ("intents", "capabilities", "risks")
PREDICTION_SCHEMA = "lfg.evaluation-prediction.v1"


def labels(record: dict[str, Any], field: str) -> set[str]:
    values = record.get(field)
    if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field} must be a list of nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicate labels")
    return set(values)


def read_predictions(path: Path) -> tuple[dict[str, dict[str, Any]], str, dict[str, Any], str]:
    raw = path.read_bytes()
    rows: dict[str, dict[str, Any]] = {}
    model = None
    producer = None
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("schema_version") != PREDICTION_SCHEMA:
            raise ValueError("unsupported prediction record")
        identifier = row.get("id")
        identity = row.get("model")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in rows:
            raise ValueError("prediction IDs must be nonempty and unique")
        if not isinstance(row.get("input_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["input_sha256"]):
            raise ValueError("prediction input digest is missing or invalid")
        digest = row.get("producer_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("prediction producer digest is missing or invalid")
        if producer is not None and producer != digest:
            raise ValueError("prediction records contain multiple producers")
        producer = digest
        if not isinstance(identity, dict) or any(not isinstance(identity.get(key), str) or not identity[key].strip()
                                                 for key in ("id", "version", "sha256")) or not re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]):
            raise ValueError("prediction model identity is missing or invalid")
        optional_digests = {"artifact", "config", "taxonomy", "predictor", "runtime"}
        if set(identity) - ({"id", "version", "sha256"} | optional_digests):
            raise ValueError("prediction model identity contains unsupported fields")
        if any(not isinstance(identity[key], str) or not re.fullmatch(r"[0-9a-f]{64}", identity[key])
               for key in optional_digests & identity.keys()):
            raise ValueError("prediction model component digest is invalid")
        if model is not None and identity != model:
            raise ValueError("prediction records contain multiple model identities")
        for field in FIELDS:
            labels(row, field)
        rows[identifier] = row
        model = identity
    if not rows:
        raise ValueError("no predictions supplied")
    return rows, hashlib.sha256(raw).hexdigest(), model, producer


def checked_separation(records: list[dict[str, Any]], path: Path | None) -> str:
    if path is None:
        return "not_checked"
    split = json.loads(path.read_text(encoding="utf-8"))
    if split.get("schema_version") != "lfg.encoder-development-split.v1":
        raise ValueError("unsupported training split manifest")
    used_ids: set[str] = set()
    used_families: set[str] = set()
    for field, target in (("train_ids", used_ids), ("validation_ids", used_ids),
                          ("train_families", used_families), ("validation_families", used_families)):
        values = split.get(field)
        if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"invalid split manifest field {field}")
        target.update(values)
    if not used_ids or not used_families:
        raise ValueError("split manifest is empty")
    if any(row["id"] in used_ids or row["family"] in used_families for row in records):
        raise ValueError("evaluation IDs or families overlap training or validation")
    return "declared_ids_and_families_disjoint"


def load_taxonomy(path: Path) -> tuple[dict[str, set[str]], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema_version") != "lfg.encoder-evaluation-taxonomy.v1":
        raise ValueError("unsupported evaluation taxonomy")
    taxonomy = {field: labels(value, field) for field in FIELDS}
    if any(not group for group in taxonomy.values()):
        raise ValueError("each taxonomy group must contain labels")
    if sum(map(len, taxonomy.values())) != len(set.union(*taxonomy.values())):
        raise ValueError("taxonomy labels must be unique across groups")
    return taxonomy, hashlib.sha256(raw).hexdigest()


def score(records: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], taxonomy: dict[str, set[str]]) -> dict[str, Any]:
    if any(row.get("schema_version") == "lfg.evaluation-abstention.v1" for row in predictions.values()):
        raise ValueError("abstentions cannot be scored as predictions")
    ids = {row["id"] for row in records}
    if len(ids) != len(records) or ids != predictions.keys():
        raise ValueError("prediction IDs must exactly match the dataset")
    if any(predictions[row["id"]]["input_sha256"] != hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
           for row in records):
        raise ValueError("prediction input text does not match the dataset")
    output: dict[str, Any] = {}
    for field in FIELDS:
        counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in taxonomy[field]}
        for row in records:
            actual = labels(row, field)
            predicted = labels(predictions[row["id"]], field)
            if (actual | predicted) - taxonomy[field]:
                raise ValueError(f"unknown {field} label in dataset or predictions")
            for label in actual | predicted:
                count = counts[label]
                if label in actual and label in predicted:
                    count["tp"] += 1
                elif label in predicted:
                    count["fp"] += 1
                else:
                    count["fn"] += 1
        by_label = {}
        f1s = []
        for label, count in sorted(counts.items()):
            tp, fp, fn = count["tp"], count["fp"], count["fn"]
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            denominator = 2 * tp + fp + fn
            f1 = 2 * tp / denominator if denominator else None
            by_label[label] = {**count, "support": tp + fn, "precision": precision, "recall": recall, "f1": f1}
            if tp + fn:
                f1s.append(f1)
        totals = {key: sum(count[key] for count in counts.values()) for key in ("tp", "fp", "fn")}
        denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
        output[field] = {"labels": by_label, "macro_f1": sum(f1s) / len(f1s) if f1s else None,
                         "macro_label_count": len(f1s), "micro_f1": 2 * totals["tp"] / denominator if denominator else None}
    return output


def evaluate(dataset: Path, prediction_file: Path, taxonomy_file: Path, split_file: Path | None, allow_unverified: bool) -> dict[str, Any]:
    validator_hash = hashlib.sha256(Path(assessment_targets.__file__).read_bytes()).hexdigest()
    records, dataset_hash = train.load_dataset(dataset)
    if not records:
        raise ValueError("no evaluation examples supplied")
    for row in records:
        if not isinstance(row["id"], str) or not row["id"].strip():
            raise ValueError("evaluation IDs must be nonempty strings")
        for field in FIELDS:
            labels(row, field)
    unverified = sum(not row["verified"] for row in records)
    if unverified and not allow_unverified:
        raise ValueError("unverified examples require --allow-unverified")
    separation = checked_separation(records, split_file)
    predictions, prediction_hash, model, producer = read_predictions(prediction_file)
    taxonomy, taxonomy_hash = load_taxonomy(taxonomy_file)
    if "taxonomy" in model and model["taxonomy"] != taxonomy_hash:
        raise ValueError("prediction taxonomy differs from evaluation taxonomy")
    if hashlib.sha256(Path(assessment_targets.__file__).read_bytes()).hexdigest() != validator_hash:
        raise ValueError("assessment target validator changed during evaluation")
    return {"schema_version": "lfg.encoder-evaluation.v1", "dataset_sha256": dataset_hash,
            "predictions_sha256": prediction_hash, "taxonomy_sha256": taxonomy_hash, "model": model,
            "producer_sha256": producer, "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dataset_loader_sha256": hashlib.sha256(Path(train.__file__).read_bytes()).hexdigest(), "examples": len(records),
            "assessment_targets_validator_sha256": validator_hash,
            "unverified_examples": unverified, "separation": separation,
            "promotion_eligible": False, "groups": score(records, predictions, taxonomy)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--training-split", type=Path)
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args()
    try:
        report = evaluate(args.dataset, args.predictions, args.taxonomy, args.training_split, args.allow_unverified)
    except (ValueError, OSError, UnicodeError, TypeError, KeyError) as error:
        raise SystemExit(f"evaluation rejected: {error}") from error
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
