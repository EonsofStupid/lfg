#!/usr/bin/env python3
"""Audit project-separated example files and prepare a private review queue."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import evaluate
import train

PLAN_SCHEMA = "lfg.encoder-corpus-plan.v1"
REVIEW_SCHEMA = "lfg.encoder-review.v1"
SPLITS = ("train", "validation", "test")


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def resolve(base: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("plan paths must be nonempty strings")
    return (base / value).resolve()


def annotation_digest(record: dict[str, Any], project: str, split: str, taxonomy_digest: str) -> str:
    # Verification is a separate review assertion, not part of the text/label attestation.
    content = {key: value for key, value in record.items() if key != "verified"}
    payload = {"record": content, "project": project, "split": split, "taxonomy_sha256": taxonomy_digest}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_key(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read_ledger(path: Path | None) -> tuple[dict[str, dict[str, Any]], str | None]:
    if path is None:
        return {}, None
    raw = path.read_bytes()
    items = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or item.get("schema_version") != REVIEW_SCHEMA:
            raise ValueError("unsupported review record")
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in items:
            raise ValueError("review IDs must be nonempty and unique")
        if item.get("decision") not in ("pending", "approved", "rejected"):
            raise ValueError("invalid review decision")
        items[identifier] = item
    return items, hashlib.sha256(raw).hexdigest()


def approved(item: dict[str, Any] | None, digest: str) -> bool:
    if not item or item.get("decision") != "approved" or item.get("annotation_sha256") != digest:
        return False
    if not isinstance(item.get("reviewer"), str) or not item["reviewer"].strip():
        return False
    try:
        reviewed_at = dt.datetime.fromisoformat(item["reviewed_at"].replace("Z", "+00:00"))
        return reviewed_at.utcoffset() is not None
    except (KeyError, AttributeError, TypeError, ValueError):
        return False


def audit(plan_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_plan = plan_path.read_bytes()
    plan = json.loads(raw_plan)
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported corpus plan")
    sources = plan.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("corpus plan requires sources")
    taxonomy, taxonomy_hash = evaluate.load_taxonomy(resolve(plan_path.parent, plan.get("taxonomy")))
    ledger_path = None if plan.get("review_ledger") is None else resolve(plan_path.parent, plan["review_ledger"])
    ledger, ledger_hash = read_ledger(ledger_path)
    partitions: dict[str, str] = {}
    ids: set[str] = set()
    families: dict[str, str] = {}
    texts: dict[str, str] = {}
    duplicate_within_split = 0
    queue: list[dict[str, Any]] = []
    counts = {split: Counter() for split in SPLITS}
    coverage = {split: {field: Counter() for field in evaluate.FIELDS} for split in SPLITS}
    source_digests = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("invalid corpus source")
        project, split = source.get("project"), source.get("split")
        if not isinstance(project, str) or not project.strip() or split not in SPLITS:
            raise ValueError("source requires a project and valid split")
        if project in partitions and partitions[project] != split:
            raise ValueError("a project appears in more than one split")
        partitions[project] = split
        rows, digest = train.load_dataset(resolve(plan_path.parent, source.get("path")))
        if not rows:
            raise ValueError("source dataset is empty")
        source_digests.append({"project": project, "split": split, "sha256": digest, "records": len(rows)})
        for row in rows:
            identifier = row["id"]
            if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
                raise ValueError("example IDs must be nonempty and globally unique")
            ids.add(identifier)
            provenance = row.get("source")
            if not isinstance(provenance, dict) or provenance.get("project") != project:
                raise ValueError("example source project does not match the plan")
            family = row["family"]
            if family in families and families[family] != split:
                raise ValueError("a family appears in more than one split")
            families[family] = split
            key = text_key(row["text"])
            if key in texts and texts[key] != split:
                raise ValueError("normalized text appears in more than one split")
            if key in texts:
                duplicate_within_split += 1
            texts[key] = split
            for field in evaluate.FIELDS:
                labels = evaluate.labels(row, field)
                if labels - taxonomy[field]:
                    raise ValueError("example has a label outside the configured taxonomy")
                coverage[split][field].update(labels)
            digest = annotation_digest(row, project, split, taxonomy_hash)
            item = ledger.get(identifier)
            is_approved = approved(item, digest)
            counts[split]["examples"] += 1
            counts[split]["verified_flag"] += int(row["verified"])
            counts[split]["approved_attestation"] += int(is_approved)
            counts[split]["ready_for_reviewed_subset"] += int(is_approved and row["verified"])
            queue.append({"schema_version": REVIEW_SCHEMA, "id": identifier, "project": project, "split": split,
                          "annotation_sha256": digest, "decision": "pending", "reviewer": None, "reviewed_at": None})
    if set(ledger) - ids:
        raise ValueError("review ledger contains IDs absent from this plan")
    report = {"schema_version": "lfg.encoder-corpus-audit.v1", "plan_sha256": hashlib.sha256(raw_plan).hexdigest(),
              "taxonomy_sha256": taxonomy_hash, "review_ledger_sha256": ledger_hash,
              "sources": source_digests, "counts": {name: dict(value) for name, value in counts.items()},
              "label_support": {split: {field: dict(sorted(values.items())) for field, values in fields.items()}
                                for split, fields in coverage.items()},
              "duplicate_within_split": duplicate_within_split,
              "project_separation": "declared_projects_disjoint", "approval_is_attestation": True,
              "promotion_eligible": False}
    return report, queue


def validate_training_corpus(records: list[dict[str, Any]], plan_path: Path) -> dict[str, Any]:
    """Require the input to be exactly the reviewed train partition of this plan."""
    report, queue = audit(plan_path)
    if not report["review_ledger_sha256"]:
        raise ValueError("audited training requires a review ledger")
    for split in SPLITS:
        counts = report["counts"][split]
        if counts.get("examples", 0) == 0:
            raise ValueError(f"audited training requires a nonempty {split} partition")
        if counts.get("ready_for_reviewed_subset", 0) != counts["examples"]:
            raise ValueError(f"every {split} example requires a current approval and verified flag")
    expected = {item["id"]: item["annotation_sha256"] for item in queue if item["split"] == "train"}
    actual: dict[str, str] = {}
    for row in records:
        if not isinstance(row.get("source"), dict) or not isinstance(row["source"].get("project"), str):
            raise ValueError("training example lacks project provenance")
        identifier = row.get("id")
        if not isinstance(identifier, str) or identifier in actual or not row.get("verified"):
            raise ValueError("training examples must have unique IDs and verified flags")
        actual[identifier] = annotation_digest(row, row["source"]["project"], "train", report["taxonomy_sha256"])
    if actual != expected:
        raise ValueError("training input does not exactly match the audited training partition")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--review-queue", type=Path, help="create a new private, pending-only JSONL review queue")
    args = parser.parse_args()
    try:
        report, queue = audit(args.plan)
        if args.review_queue is not None:
            descriptor = os.open(args.review_queue, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as out:
                for item in queue:
                    out.write(json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, UnicodeError, TypeError, KeyError) as error:
        raise SystemExit(f"corpus audit rejected: {error}") from error


if __name__ == "__main__":
    main()
