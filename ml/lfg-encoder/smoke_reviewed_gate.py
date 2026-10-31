#!/usr/bin/env python3
"""Run the reviewed-corpus gate on simulated fixtures; never produce a release model."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-model", required=True)
parser.add_argument("--model-revision", required=True)
parser.add_argument("--output", required=True, type=Path, help="new fixture-only directory")
parser.add_argument("--epochs", required=True)
parser.add_argument("--max-length", required=True)
parser.add_argument("--batch-size", required=True)
parser.add_argument("--checkpoint-steps", required=True)
parser.add_argument("--allow-mismatched-sizes", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "ml/lfg-encoder"))
import corpus_audit

work = args.output.resolve()
work.mkdir()
taxonomy = root / "ml/lfg-encoder/evaluation-taxonomy.v1.json"
fixture_labels = json.loads(taxonomy.read_text(encoding="utf-8"))
fixture_a = {"intents": [label for label in fixture_labels["intents"] if label != "build.software"],
             "capabilities": fixture_labels["capabilities"], "risks": fixture_labels["risks"]}
fixture_b = {"intents": ["build.software"], "capabilities": [], "risks": []}
rows = [
    {"id": "fixture-1", "text": "Explain the fixture architecture", "family": "fixture-a", "verified": False,
     **fixture_a, "source": {"project": "fixture-train"}},
    {"id": "fixture-2", "text": "Implement the fixture component", "family": "fixture-b", "verified": False,
     **fixture_b, "source": {"project": "fixture-train"}},
    {"id": "fixture-3", "text": "Explain the validation fixture", "family": "fixture-c", "verified": False,
     **fixture_a, "source": {"project": "fixture-validation"}},
    {"id": "fixture-4", "text": "Implement the validation fixture", "family": "fixture-d", "verified": False,
     **fixture_b, "source": {"project": "fixture-validation"}},
    {"id": "fixture-5", "text": "Implement the test fixture", "family": "fixture-e", "verified": False,
     **fixture_b, "source": {"project": "fixture-test"}},
]
files = [("train.jsonl", rows[:2]), ("validation.jsonl", rows[2:4]), ("test.jsonl", rows[4:])]
def save_rows():
    for name, group in files:
        (work / name).write_text("".join(json.dumps(row) + "\n" for row in group), encoding="utf-8")
save_rows()
plan = {"schema_version": corpus_audit.PLAN_SCHEMA, "taxonomy": str(taxonomy), "review_ledger": None,
        "sources": [{"path": "train.jsonl", "project": "fixture-train", "split": "train"},
                    {"path": "validation.jsonl", "project": "fixture-validation", "split": "validation"},
                    {"path": "test.jsonl", "project": "fixture-test", "split": "test"}]}
plan_path = work / "plan.json"
plan_path.write_text(json.dumps(plan), encoding="utf-8")
_, queue = corpus_audit.audit(plan_path)
reviews = [{**row, "decision": "approved", "reviewer": "simulated-fixture-only", "reviewed_at": "2026-10-30T12:00:00Z"} for row in queue]
(work / "reviews.jsonl").write_text("".join(json.dumps(row) + "\n" for row in reviews), encoding="utf-8")
plan["review_ledger"] = "reviews.jsonl"
plan_path.write_text(json.dumps(plan), encoding="utf-8")
for row in rows:
    row["verified"] = True
save_rows()
payload, _ = corpus_audit.assemble_training(plan_path)
(work / "training.jsonl").write_bytes(payload)
policy = json.loads((root / "ml/lfg-encoder/promotion-policy.json").read_text(encoding="utf-8"))
policy["minimum_human_verified_examples"] = len(rows[:2])
(work / "fixture-policy.json").write_text(json.dumps(policy), encoding="utf-8")
cmd = [sys.executable, "ml/lfg-encoder/train.py", "train", "--dataset", str(work / "training.jsonl"),
       "--promotion-policy", str(work / "fixture-policy.json"),
       "--audited-plan", str(plan_path),
       "--base-model", args.base_model, "--model-revision", args.model_revision, "--output", str(work / "model"),
       "--epochs", args.epochs, "--max-length", args.max_length, "--batch-size", args.batch_size,
       "--checkpoint-steps", args.checkpoint_steps]
if args.allow_mismatched_sizes:
    cmd.append("--allow-mismatched-sizes")
env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
result = subprocess.run(cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if result.returncode:
    print(result.stderr.decode(errors="replace"))
    raise SystemExit(result.returncode)
manifest = json.loads((work / "model/run-manifest.json").read_text())
assert manifest["split"]["corpus_audit"]["review_ledger_sha256"]
assert manifest["split"]["validation_source"] == "audited-validation-partition"
assert manifest["split"]["training_support"]["counts"]["train"]["sensitive-data"]["positive"] == 1
assert set(manifest["split"]["train_ids"]) == {"fixture-1", "fixture-2"}
assert manifest["split"]["validation_ids"] == ["fixture-3", "fixture-4"]
assert "fixture-5" not in manifest["split"]["train_ids"] + manifest["split"]["validation_ids"]
assert (work / "model/model.safetensors").is_file()
classes = json.loads((work / "model/labels.json").read_text(encoding="utf-8"))
assert classes == sorted(label for group in ("intents", "capabilities", "risks") for label in fixture_labels[group])
print(json.dumps({"fixture_only": True, "training_rows": len(rows[:2]), "audit_bound": True,
                  "model_sha256": hashlib.sha256((work / "model/model.safetensors").read_bytes()).hexdigest()}))
