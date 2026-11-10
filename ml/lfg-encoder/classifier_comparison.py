#!/usr/bin/env python3
"""Run local classical multilabel baselines on a screened development corpus.

The report is descriptive exploratory evidence. It does not promote or qualify
LFG, and it never writes prompt text or per-example predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import evaluate
import screen_private_corpus

SCHEMA = "lfg.classifier-comparison-config.v1"
REPORT_SCHEMA = "lfg.classifier-comparison-report.v1"
FIELDS = ("intents", "capabilities", "risks")
HEX = re.compile(r"^[0-9a-f]{64}$")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(raw: bytes, name: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate {name} key")
            result[key] = value
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid constant")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name}") from error


def load_config(path: Path, max_bytes: int) -> tuple[dict[str, Any], str]:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max config bytes must be a positive integer")
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("config exceeds configured byte limit")
    config = _json_bytes(raw, "config")
    required = {"schema_version", "revision", "screened_corpus", "label_plan", "screen_policy",
                "review_ledger",
                "target_field", "selected_labels",
                "seed", "max_rows", "max_text_bytes", "max_source_bytes", "max_manifest_bytes",
                "max_total_source_bytes", "max_screened_file_bytes", "max_features", "knn_neighbors", "tree_max_depth",
                "tfidf_ngram_range", "logistic_c", "logistic_max_iter", "knn_weights", "knn_p",
                "tree_criterion", "near_duplicate_similarity", "max_near_duplicate_pairs",
                "max_near_duplicate_work", "max_policy_bytes", "constant_target_behavior",
                "validation_group_use_claim", "max_ledger_bytes", "rejected_label_handling"}
    if (not isinstance(config, dict) or not required <= set(config) or config.get("schema_version") != SCHEMA
            or not isinstance(config.get("revision"), str) or not config["revision"].strip()
            or not isinstance(config.get("screened_corpus"), str) or not config["screened_corpus"].strip()
            or not isinstance(config.get("label_plan"), str) or not config["label_plan"].strip()
            or not isinstance(config.get("screen_policy"), str) or not config["screen_policy"].strip()
            or not isinstance(config.get("review_ledger"), str) or not config["review_ledger"].strip()
            or config.get("target_field") not in FIELDS or not isinstance(config.get("selected_labels"), list)
            or not config["selected_labels"] or any(not isinstance(x, str) or not x.strip() for x in config["selected_labels"])):
        raise ValueError("invalid comparison config")
    for key in ("seed", "max_rows", "max_text_bytes", "max_source_bytes", "max_manifest_bytes",
                "max_total_source_bytes", "max_screened_file_bytes",
                "max_features", "knn_neighbors", "max_near_duplicate_pairs",
                "max_near_duplicate_work", "max_policy_bytes", "max_ledger_bytes"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid {key}")
    if config["tree_max_depth"] is not None and (type(config["tree_max_depth"]) is not int or config["tree_max_depth"] < 1):
        raise ValueError("invalid tree_max_depth")
    ngrams = config["tfidf_ngram_range"]
    if (not isinstance(ngrams, list) or len(ngrams) != 2
            or any(type(value) is not int or value < 1 for value in ngrams) or ngrams[1] < ngrams[0]):
        raise ValueError("invalid tfidf_ngram_range")
    if type(config["logistic_c"]) not in (int, float) or not math.isfinite(config["logistic_c"]) or config["logistic_c"] <= 0:
        raise ValueError("invalid logistic_c")
    if type(config["logistic_max_iter"]) is not int or config["logistic_max_iter"] < 1:
        raise ValueError("invalid logistic_max_iter")
    if config["knn_weights"] not in ("uniform", "distance") or config["knn_p"] not in (1, 2):
        raise ValueError("invalid KNN configuration")
    if config["tree_criterion"] not in ("gini", "entropy", "log_loss"):
        raise ValueError("invalid tree criterion")
    similarity = config["near_duplicate_similarity"]
    if type(similarity) not in (int, float) or not math.isfinite(similarity) or not 0 <= similarity <= 1:
        raise ValueError("invalid near_duplicate_similarity")
    if config["constant_target_behavior"] not in ("abstain", "majority_class"):
        raise ValueError("invalid constant_target_behavior")
    if config["validation_group_use_claim"] not in ("previously_used", "reported_unused", "unknown"):
        raise ValueError("invalid validation_group_use_claim")
    if config["rejected_label_handling"] not in ("exclude", "include_exploratory"):
        raise ValueError("invalid rejected_label_handling")
    return config, sha(raw)


def _read_bounded(path: Path, limit: int, name: str) -> bytes:
    if type(limit) is not int or limit < 1:
        raise ValueError(f"invalid {name} byte limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"{name} exceeds configured byte limit")
    return raw


def _sensitive_metadata(value: str, rules) -> bool:
    return any(pattern.search(value) is not None for _, pattern, _ in rules)


def _require_outside_git_worktrees(path: Path) -> Path:
    """Resolve the path and reject every Git worktree, including linked worktrees."""
    if path.is_symlink():
        raise ValueError("report path must not be a symlink")
    resolved = path.resolve(strict=False)
    directory = resolved if resolved.is_dir() else resolved.parent
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    if directory.exists() and not directory.is_dir():
        directory = directory.parent
    try:
        result = subprocess.run(["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                                timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0:
        raise ValueError("private report must be outside every Git worktree")
    if result is None or result.returncode != 0:
        if any((ancestor / ".git").exists() for ancestor in (directory, *directory.parents)):
            raise ValueError("cannot verify that private report is outside Git worktrees")
    return resolved


def _secure_owner_acl(path: Path) -> None:
    """Create an owner-only ACL boundary for the report directory."""
    if os.name == "nt":
        script = r"""
$p = [Console]::In.ReadToEnd().Trim()
try {
  $acl = Get-Acl -LiteralPath $p -ErrorAction Stop
  $acl.SetAccessRuleProtection($true, $false)
  foreach ($rule in @($acl.Access)) { [void]$acl.RemoveAccessRuleSpecific($rule) }
  $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User
  $principals = @($sid, [Security.Principal.SecurityIdentifier]::new('S-1-5-18'),
                  [Security.Principal.SecurityIdentifier]::new('S-1-5-32-544'))
  foreach ($principal in $principals) {
    $rule = [Security.AccessControl.FileSystemAccessRule]::new(
      $principal, [Security.AccessControl.FileSystemRights]::FullControl,
      [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [Security.AccessControl.InheritanceFlags]::ObjectInherit,
      [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Allow)
    [void]$acl.AddAccessRule($rule)
  }
  Set-Acl -LiteralPath $p -AclObject $acl -ErrorAction Stop
  '{"ok":true}'
} catch { '{"ok":false}' }
"""
        try:
            result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                                    input=str(path).encode("utf-8"), stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, check=False, timeout=10)
            evidence = json.loads(result.stdout.decode("utf-8", errors="strict"))
        except (OSError, subprocess.TimeoutExpired, UnicodeError, json.JSONDecodeError):
            raise ValueError("cannot secure owner-private report directory") from None
        if result.returncode != 0 or not isinstance(evidence, dict) or evidence.get("ok") is not True:
            raise ValueError("cannot secure owner-private report directory")
    else:
        os.chmod(path, 0o700)
        if path.stat().st_mode & 0o077:
            raise ValueError("report directory permissions are not owner-only")


def _require_owner_acl(path: Path) -> None:
    """On Windows, require ACL entries only for the current owner, SYSTEM, and admins."""
    if os.name != "nt":
        if path.stat().st_mode & 0o077:
            raise ValueError("report permissions are not owner-only")
        return
    script = r"""
$p = [Console]::In.ReadToEnd().Trim()
try {
  $acl = Get-Acl -LiteralPath $p -ErrorAction Stop
  $current = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
  $owner = $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value
  $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
  $allowed = @($sid, 'S-1-5-18', 'S-1-5-32-544')
  $rules = @($acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]))
  $ok = $true
  foreach ($rule in $rules) {
    if ($rule.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow -or
        $allowed -notcontains $rule.IdentityReference.Value) { $ok = $false }
  }
  @{ ok = ($ok -and $owner -eq $current); rules = $rules.Count } | ConvertTo-Json -Compress
} catch { '{"ok":false,"rules":0}' }
"""
    try:
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                                input=str(path).encode("utf-8"), stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, check=False, timeout=10)
        evidence = json.loads(result.stdout.decode("utf-8", errors="strict"))
    except (OSError, subprocess.TimeoutExpired, UnicodeError, json.JSONDecodeError):
        raise ValueError("cannot verify owner-private report ACL") from None
    if result.returncode != 0 or not isinstance(evidence, dict) or evidence.get("ok") is not True:
        raise ValueError("report ACL permits access beyond owner, SYSTEM, or administrators")


def require_private_report_path(path: Path, *, require_existing: bool = False) -> Path:
    """Guard the actual report path, independently of screening output names."""
    resolved = _require_outside_git_worktrees(path)
    if require_existing and not resolved.is_file():
        raise ValueError("private report file is unavailable")
    if not require_existing and resolved.exists():
        raise ValueError("private report path already exists")
    if not resolved.parent.is_dir():
        raise ValueError("report parent directory must exist")
    if resolved.exists():
        _require_owner_acl(resolved)
    return resolved


def load_screened_corpus(directory: Path, plan_path: Path, max_rows: int,
                         max_text_bytes: int, max_source_bytes: int,
                         max_total_source_bytes: int, max_manifest_bytes: int, max_screened_file_bytes: int,
                         near_duplicate_similarity: float, max_near_duplicate_pairs: int,
                         max_near_duplicate_work: int, screen_policy_path: Path,
                         max_policy_bytes: int) -> tuple[list[dict], list[dict], dict, dict, dict]:
    """Join screened text to explicit labels, reading only declared train/validation sources."""
    directory = directory.resolve(strict=True)
    manifest_raw = _read_bounded(directory / "manifest.json", max_manifest_bytes, "screen manifest")
    manifest = _json_bytes(manifest_raw, "screen manifest")
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != "lfg.private-corpus-screen-manifest.v1"
            or not isinstance(manifest.get("outputs"), dict) or not isinstance(manifest.get("counts"), dict)
            or not isinstance(manifest.get("claims"), dict)):
        raise ValueError("unsupported screened corpus manifest")
    policy, screen_rules, policy_digest = screen_private_corpus.load_policy(screen_policy_path, max_policy_bytes)
    current_screen_digest = sha(Path(screen_private_corpus.__file__).read_bytes())
    helper_hashes = manifest.get("helper_sha256")
    expected_helpers = (screen_private_corpus.assessment_distill, screen_private_corpus.corpus_audit,
                        screen_private_corpus.review_corpus)
    helpers_match = isinstance(helper_hashes, dict) and all(
        helper_hashes.get(Path(module.__file__).name) == sha(Path(module.__file__).read_bytes())
        for module in expected_helpers)
    if (manifest.get("policy_sha256") != policy_digest
            or manifest.get("screen_code_sha256") != current_screen_digest or not helpers_match):
        raise ValueError("screening manifest does not match the trusted local policy and scanner code")
    plan_raw = _read_bounded(plan_path, max_manifest_bytes, "label plan")
    if sha(plan_raw) != manifest.get("plan_sha256"):
        raise ValueError("screened manifest does not bind the selected label plan")
    plan = _json_bytes(plan_raw, "label plan")
    sources = plan.get("sources") if isinstance(plan, dict) else None
    if (not isinstance(plan, dict) or plan.get("schema_version") != screen_private_corpus.corpus_audit.DEVELOPMENT_PLAN_SCHEMA
            or plan.get("review_ledger") is not None or not isinstance(sources, list) or not sources):
        raise ValueError("label plan requires a source list")
    # Refuse every heldout declaration before resolving or reading any dataset path.
    if any(not isinstance(source, dict) or source.get("split") not in ("train", "validation")
           for source in sources):
        raise ValueError("label plan declares a non-development source")
    projects, groups = {}, {}
    for source in sources:
        project, split = source.get("project"), source.get("split")
        group = source.get("project_group", project)
        if any(not isinstance(value, str) or not value.strip() for value in (project, group)):
            raise ValueError("source project/group is invalid")
        if any(_sensitive_metadata(value, screen_rules) for value in (project, group)):
            raise ValueError("sensitive source metadata detected")
        if project in projects and projects[project] != (group, split):
            raise ValueError("project is assigned to multiple groups or splits")
        if group in groups and groups[group] != split:
            raise ValueError("project group crosses train/validation")
        projects[project], groups[group] = (group, split), split
    taxonomy_path = screen_private_corpus.corpus_audit.resolve(plan_path.parent, plan.get("taxonomy"))
    taxonomy_raw = _read_bounded(taxonomy_path, max_manifest_bytes, "taxonomy")
    taxonomy, taxonomy_digest = evaluate.load_taxonomy(taxonomy_path)
    outputs = manifest["outputs"]
    if set(outputs) - {"train", "validation"}:
        raise ValueError("screen manifest includes a non-development split")
    if not {"train", "validation"} <= set(outputs):
        raise ValueError("screened corpus requires train and validation outputs")
    screened_by_split = {}
    screened_by_id = {}
    for split in ("train", "validation"):
        digest = outputs[split]
        if not isinstance(digest, str) or not HEX.fullmatch(digest):
            raise ValueError("invalid screened output digest")
        raw = _read_bounded(directory / f"{split}.jsonl", max_screened_file_bytes, f"{split} data")
        if sha(raw) != digest:
            raise ValueError(f"{split} output digest mismatch")
        rows, index = [], {}
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                raise ValueError("blank screened data line")
            row = _json_bytes(line.encode("utf-8"), "screened row")
            if not isinstance(row, dict) or row.get("schema_version") != screen_private_corpus.OUTPUT_VERSION:
                raise ValueError("unsupported screened row")
            metadata_values = [row.get(key) for key in ("id", "family", "project", "project_group")]
            if any(not isinstance(value, str) or not value.strip() for value in metadata_values):
                raise ValueError("screened row metadata is invalid")
            if any(_sensitive_metadata(value, screen_rules) for value in metadata_values):
                raise ValueError("sensitive screened metadata detected")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > max_text_bytes:
                raise ValueError("invalid or oversized screened text")
            if (row.get("split") != split or row.get("redaction_complete") is not False
                    or row.get("review_status") != "unreviewed"
                    or not isinstance(row.get("screen_policy_sha256"), str)
                    or not HEX.fullmatch(row["screen_policy_sha256"])):
                raise ValueError("screened row provenance mismatch")
            labels = row.get("labels")
            if labels is not None:
                raise ValueError("unexpected labels in screened row")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier.strip() or identifier in index:
                raise ValueError("screened row IDs must be nonempty and unique")
            index[identifier] = row
            rows.append(row)
            if len(rows) > max_rows:
                raise ValueError("screened corpus exceeds configured row limit")
        if len(rows) != manifest["counts"].get(split):
            raise ValueError(f"{split} count mismatch")
        screened_by_split[split] = rows
        screened_by_id[split] = index
    if len(screened_by_split["train"]) + len(screened_by_split["validation"]) > max_rows:
        raise ValueError("screened corpus exceeds configured total row limit")
    source_rows_by_split = {"train": [], "validation": []}
    matched_ids = {"train": set(), "validation": set()}
    source_evidence = manifest.get("source_snapshots")
    if not isinstance(source_evidence, list) or len(source_evidence) != len(sources):
        raise ValueError("screened manifest source evidence mismatch")
    total_source_bytes = 0
    recomputed_matched_records = 0
    source_artifacts = []
    for source, evidence in zip(sources, source_evidence, strict=True):
        split = source["split"]
        source_path = screen_private_corpus.corpus_audit.resolve(plan_path.parent, source.get("path"))
        raw = _read_bounded(source_path, max_source_bytes, "label source")
        total_source_bytes += len(raw)
        if total_source_bytes > max_total_source_bytes:
            raise ValueError("label sources exceed configured total byte limit")
        if sha(raw) != evidence.get("sha256"):
            raise ValueError("screened manifest source digest mismatch")
        decoded = raw.decode("utf-8")
        lines = decoded.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if any(not line.strip() for line in lines):
            raise ValueError("blank label source line")
        if len(lines) != evidence.get("records"):
            raise ValueError("screened manifest source count mismatch")
        source_artifacts.append({"project": source.get("project"),
                                 "project_group": source.get("project_group", source.get("project")),
                                 "split": split, "sha256": evidence.get("sha256"),
                                 "records": evidence.get("records")})
        for line in lines:
            source_record = _json_bytes(line.encode("utf-8"), "label source record")
            identifier = source_record.get("id") if isinstance(source_record, dict) else None
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("label source record lacks ID")
            record_source = source_record.get("source")
            metadata_values = [identifier, source_record.get("family"),
                               record_source.get("project") if isinstance(record_source, dict) else None,
                               source.get("project"), source.get("project_group", source.get("project"))]
            if any(not isinstance(value, str) or not value.strip() for value in metadata_values):
                raise ValueError("label source metadata is invalid")
            if any(_sensitive_metadata(value, screen_rules) for value in metadata_values):
                raise ValueError("sensitive label source metadata detected")
            record_hash = sha(line.encode("utf-8"))
            screen_row = screened_by_id[split].get(identifier)
            if (screen_row is None or identifier in matched_ids[split]
                    or screen_row.get("source_record_sha256") != record_hash):
                raise ValueError("screened and labeled records do not match")
            if (source_record.get("family") != screen_row.get("family")
                    or not isinstance(source_record.get("source"), dict)
                    or source_record["source"].get("project") != screen_row.get("project")
                    or screen_row.get("project") != source.get("project")
                    or screen_row.get("project_group") != source.get("project_group", source.get("project"))
                    or screen_row.get("candidate_file_sha256") != evidence.get("sha256")
                    or screen_row.get("screen_policy_sha256") != manifest.get("policy_sha256")
                    or screen_row.get("screened_text_sha256") != sha(screen_row["text"].encode("utf-8"))):
                raise ValueError("screened and labeled provenance mismatch")
            original_text = source_record.get("text")
            if not isinstance(original_text, str) or not original_text.strip():
                raise ValueError("label source text is missing")
            verified_screen_text, match_counts = screen_private_corpus._screen(
                original_text, screen_rules, policy["limits"])
            if (verified_screen_text != screen_row["text"]
                    or dict(sorted(match_counts.items())) != screen_row.get("redaction_counts")):
                raise ValueError("screened text does not match recomputed policy output")
            recomputed_matched_records += int(bool(match_counts))
            if screen_row.get("source_text_sha256") != sha(original_text.encode("utf-8")):
                raise ValueError("source text digest does not match screening provenance")
            source_record["text"] = screen_row["text"]
            source_record["project"] = screen_row["project"]
            source_record["project_group"] = screen_row["project_group"]
            source_record["split"] = screen_row["split"]
            source_rows_by_split[split].append(source_record)
            matched_ids[split].add(identifier)
    for split in ("train", "validation"):
        if matched_ids[split] != set(screened_by_id[split]):
            raise ValueError("screened and labeled row sets do not match exactly")
        if len(source_rows_by_split[split]) != manifest["counts"].get(split):
            raise ValueError("selected source count does not match screened count")
    if manifest.get("matched_records") != recomputed_matched_records:
        raise ValueError("screening manifest redaction count does not match recomputation")
    all_rows = source_rows_by_split["train"] + source_rows_by_split["validation"]
    ids, family_splits, text_splits = set(), {}, {}
    project_partitions = {}
    for row in all_rows:
        identifier = row.get("id")
        family = row.get("family")
        project = row.get("project")
        group = row.get("project_group")
        if any(not isinstance(value, str) or not value.strip() for value in (identifier, family, project, group)):
            raise ValueError("screened row lacks provenance")
        if identifier in ids:
            raise ValueError("duplicate row ID")
        ids.add(identifier)
        project_claim = (group, row["split"])
        if project in project_partitions and project_partitions[project] != project_claim:
            raise ValueError("project crosses train/validation or changes declared group")
        project_partitions[project] = project_claim
        if family in family_splits and family_splits[family] != row["split"]:
            raise ValueError("family crosses train/validation")
        family_splits[family] = row["split"]
        if group in family_splits and family_splits[group] != row["split"]:
            raise ValueError("project group crosses train/validation")
        family_splits[group] = row["split"]
        key = " ".join(row["text"].casefold().split())
        if key in text_splits and text_splits[key] != row["split"]:
            raise ValueError("exact normalized text crosses train/validation")
        text_splits[key] = row["split"]
    if not source_rows_by_split["train"] or not source_rows_by_split["validation"]:
        raise ValueError("empty train or validation partition")
    # Bound both pair count and worst-case character comparison work.
    import difflib
    near_duplicate_pairs = near_duplicate_work = 0
    for train_row in source_rows_by_split["train"]:
        left = " ".join(train_row["text"].casefold().split())
        for validation_row in source_rows_by_split["validation"]:
            right = " ".join(validation_row["text"].casefold().split())
            near_duplicate_pairs += 1
            near_duplicate_work += len(left) * len(right)
            if near_duplicate_pairs > max_near_duplicate_pairs or near_duplicate_work > max_near_duplicate_work:
                raise ValueError("near-duplicate check exceeds configured work budget")
            if difflib.SequenceMatcher(None, left, right, autojunk=False).ratio() >= near_duplicate_similarity:
                raise ValueError("near-duplicate text crosses train/validation")
    return source_rows_by_split["train"], source_rows_by_split["validation"], {
        "manifest_sha256": sha(manifest_raw), "train_sha256": outputs["train"],
        "validation_sha256": outputs["validation"], "manifest_claims": manifest["claims"],
        "policy_sha256": manifest.get("policy_sha256"), "screen_code_sha256": manifest.get("screen_code_sha256"),
        "screening_matched_records": recomputed_matched_records,
        "source_artifacts": source_artifacts,
        "screening_recomputed_with_local_code": True,
        "near_duplicate_checks": {"pairs": near_duplicate_pairs, "character_product_work": near_duplicate_work},
        "taxonomy_sha256": taxonomy_digest,
        "counts": {split: len(source_rows_by_split[split]) for split in source_rows_by_split}}, taxonomy, {
            "plan_sha256": sha(plan_raw), "taxonomy_sha256": sha(taxonomy_raw)}


def _matrix(rows: list[dict], field: str, selected: list[str], taxonomy: dict[str, set[str]]):
    from sklearn.preprocessing import MultiLabelBinarizer
    supported = taxonomy[field]
    if len(set(selected)) != len(selected) or set(selected) - supported:
        raise ValueError("selected labels are duplicate or outside the configured taxonomy")
    labels = sorted(selected)
    targets, observed = [], []
    for row in rows:
        values = row.get(field)
        if values is None:
            values = []
            has_array = False
        else:
            has_array = True
        if not isinstance(values, list) or any(not isinstance(x, str) or not x.strip() for x in values):
            raise ValueError("target arrays must contain nonempty label strings")
        if len(values) != len(set(values)) or set(values) - supported:
            raise ValueError("invalid or out-of-taxonomy target array")
        if type(row.get("verified")) is not bool:
            raise ValueError("target row must preserve its verified status")
        complete = row.get("target_completeness", {})
        if not isinstance(complete, dict) or any(type(value) is not bool for value in complete.values()):
            raise ValueError("invalid target completeness attestation")
        pending = row.get("labels_pending")
        if pending is not None and type(pending) is not bool:
            raise ValueError("invalid labels_pending flag")
        complete_value = complete.get(field)
        selected_values = set(values) & set(labels)
        if complete_value is False and has_array and values and pending is not True:
            # Explicit positives remain usable; omitted labels stay unknown.
            row_known = [label in selected_values for label in labels]
        elif complete_value is True and has_array and pending is not True:
            # The source explicitly attests that an empty list is complete.
            row_known = [True] * len(labels)
        elif complete_value is None and has_array and values and pending is not True:
            # Legacy review rows use a nonempty array as the complete label set.
            # This is a source/schema claim, not independent verification.
            row_known = [True] * len(labels)
        else:
            row_known = [False] * len(labels)
        targets.append([label for label in values if label in set(labels)])
        observed.append(row_known)
    encoder = MultiLabelBinarizer(classes=labels)
    import numpy as np
    return encoder, encoder.fit_transform(targets), np.asarray(observed, dtype=bool)


def _target_partition(y_train, known_train):
    variable, constant, unsupported = [], [], []
    for i in range(y_train.shape[1]):
        values = y_train[known_train[:, i], i]
        if not len(values):
            unsupported.append(i)
        elif 0 < int(values.sum()) < len(values):
            variable.append(i)
        else:
            constant.append(i)
    return variable, constant, unsupported


def _constant_fallbacks(y_train, known_train, labels: list[str], row_count: int, behavior: str):
    import numpy as np
    predictions = np.full((row_count, len(labels)), -1, dtype=int)
    if behavior == "majority_class":
        for index in range(len(labels)):
            known = known_train[:, index]
            if not known.any():
                continue
            positives = int(y_train[known, index].sum())
            negatives = int(known.sum()) - positives
            # Explicit tie rule: choose negative.
            predictions[:, index] = int(positives > negatives)
    return predictions


def _estimators(config: dict[str, Any]):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    return {
        "logistic_regression": (TfidfVectorizer(ngram_range=tuple(config["tfidf_ngram_range"]),
                                                 max_features=config["max_features"], dtype=np.float64), None),
        "knn": (TfidfVectorizer(ngram_range=tuple(config["tfidf_ngram_range"]),
                                max_features=config["max_features"], dtype=np.float64), None),
        "decision_tree": (TfidfVectorizer(ngram_range=tuple(config["tfidf_ngram_range"]),
                                           max_features=config["max_features"], dtype=np.float64), None)
    }


def _estimator(algorithm: str, config: dict[str, Any], sample_count: int):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    if algorithm == "logistic_regression":
        return LogisticRegression(random_state=config["seed"], max_iter=config["logistic_max_iter"],
                                  C=config["logistic_c"])
    if algorithm == "knn":
        return KNeighborsClassifier(n_neighbors=min(config["knn_neighbors"], sample_count),
                                    weights=config["knn_weights"], p=config["knn_p"])
    return DecisionTreeClassifier(random_state=config["seed"], max_depth=config["tree_max_depth"],
                                  criterion=config["tree_criterion"])


def _metrics(y_true, y_pred, target_known, labels: list[str]) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import accuracy_score
    by_label = {}
    aggregate = Counter()
    standard_values = {key: [] for key in ("precision", "recall", "f1")}
    supported_values = {key: [] for key in ("precision", "recall", "f1")}
    for index, label in enumerate(labels):
        target_mask = target_known[:, index]
        known = target_mask & (y_pred[:, index] >= 0)
        actual, predicted = y_true[known, index], y_pred[known, index]
        tp = int(((actual == 1) & (predicted == 1)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())
        unknown_target_count = int((~target_mask).sum())
        unknown_prediction_count = int((target_mask & (y_pred[:, index] < 0)).sum())
        unknown_positive_count = int((target_mask & (y_true[:, index] == 1) & (y_pred[:, index] < 0)).sum())
        precision_denom, recall_denom, f1_denom = tp + fp, tp + fn, 2 * tp + fp + fn
        precision = tp / precision_denom if precision_denom else None
        recall = tp / recall_denom if recall_denom else None
        f1 = 2 * tp / f1_denom if f1_denom else None
        support = int((target_mask & (y_true[:, index] == 1)).sum())
        negative_support = int((target_mask & (y_true[:, index] == 0)).sum())
        by_label[label] = {"support": support,
                           "negative_support": negative_support,
                           "scored_support": int(known.sum()), "predicted_positive": int((predicted == 1).sum()),
                           "unknown_predictions": unknown_prediction_count,
                           "unknown_targets": unknown_target_count,
                           "unknown_positive_targets": unknown_positive_count,
                           "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                           "precision": precision, "recall": recall, "f1": f1}
        for name, value in (("precision", precision), ("recall", recall), ("f1", f1)):
            if value is not None:
                standard_values[name].append(value)
            if support > 0 and value is not None:
                supported_values[name].append(value)
        aggregate.update(tp=tp, fp=fp, fn=fn, tn=tn,
                         unknown_predictions=unknown_prediction_count,
                         unknown_targets=unknown_target_count, unknown_positive=unknown_positive_count)
    total_predictions = y_true.shape[0] * y_true.shape[1]
    known_targets = int(target_known.sum())
    known_count = int((target_known & (y_pred >= 0)).sum())
    micro_p_denom = aggregate["tp"] + aggregate["fp"]
    micro_r_denom = aggregate["tp"] + aggregate["fn"]
    micro_f_denom = 2 * aggregate["tp"] + aggregate["fp"] + aggregate["fn"]
    fully_scored = np.all(target_known & (y_pred >= 0), axis=1)
    subset = float(accuracy_score(y_true[fully_scored], y_pred[fully_scored])) if fully_scored.any() else None
    macro = {name: (sum(values) / len(values) if values else None)
             for name, values in standard_values.items()}
    macro_supported = {name: (sum(values) / len(values) if values else None)
                       for name, values in supported_values.items()}
    return {"examples": int(y_true.shape[0]), "subset_accuracy_on_fully_scored": subset,
            "scored_label_predictions": known_count, "total_label_predictions": total_predictions,
            "target_coverage": known_targets / total_predictions if total_predictions else None,
            "prediction_coverage_on_known_targets": known_count / known_targets if known_targets else None,
            "overall_coverage": known_count / total_predictions if total_predictions else None,
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "macro_f1": macro["f1"],
            "macro_precision_supported_labels": macro_supported["precision"],
            "macro_recall_supported_labels": macro_supported["recall"],
            "macro_f1_supported_labels": macro_supported["f1"],
            "macro_metric_label_coverage": {
                name: {"defined_labels": len(standard_values[name]), "total_labels": len(labels)}
                for name in standard_values},
            "micro_precision_on_covered": aggregate["tp"] / micro_p_denom if micro_p_denom else None,
            "micro_recall_on_covered": aggregate["tp"] / micro_r_denom if micro_r_denom else None,
            "micro_f1_on_covered": 2 * aggregate["tp"] / micro_f_denom if micro_f_denom else None,
            "labels": by_label, "unknown_or_abstained": aggregate["unknown_predictions"],
            "unknown_target_cells": aggregate["unknown_targets"],
            "unknown_positive_targets": aggregate["unknown_positive"]}


def _label_review_state(rows: list[dict], field: str) -> dict[str, Any]:
    return {"rows": len(rows),
            "target_array_nonempty": sum(isinstance(row.get(field), list) and bool(row[field]) for row in rows),
            "target_array_empty": sum(isinstance(row.get(field), list) and not row[field] for row in rows),
            "target_array_missing_or_null": sum(field not in row or row.get(field) is None for row in rows),
            "verified_true": sum(row.get("verified") is True for row in rows),
            "verified_false": sum(row.get("verified") is False for row in rows),
            "labels_pending_true": sum(row.get("labels_pending") is True for row in rows),
            "labels_pending_false": sum(row.get("labels_pending") is False for row in rows),
            "labels_pending_absent": sum("labels_pending" not in row for row in rows),
            "target_completeness_true": sum(isinstance(row.get("target_completeness"), dict)
                                              and row["target_completeness"].get(field) is True
                                              for row in rows),
            "target_completeness_false": sum(isinstance(row.get("target_completeness"), dict)
                                               and row["target_completeness"].get(field) is False
                                               for row in rows),
            "target_completeness_absent": sum(not isinstance(row.get("target_completeness"), dict)
                                                or field not in row["target_completeness"]
                                                for row in rows)}


def _load_review_ledger(path: Path, max_bytes: int) -> tuple[dict[str, str], str]:
    raw = _read_bounded(path, max_bytes, "review ledger")
    ledger = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = _json_bytes(line.encode("utf-8"), "review ledger row")
        if (not isinstance(row, dict) or row.get("schema_version") != "lfg.encoder-review.v1"
                or not isinstance(row.get("id"), str) or not row["id"].strip()
                or row["decision"] not in ("pending", "approved", "rejected")
                or row["id"] in ledger):
            raise ValueError("invalid or duplicate review ledger entry")
        ledger[row["id"]] = row["decision"]
    if not ledger:
        raise ValueError("review ledger is empty")
    return ledger, sha(raw)


def run(config_path: Path, output: Path, config_limit: int = 65_536) -> dict[str, Any]:
    config, config_digest = load_config(config_path, config_limit)
    directory = Path(config["screened_corpus"])
    output = require_private_report_path(output)
    train_rows, validation_rows, corpus, taxonomy, source_identity = load_screened_corpus(
        directory, Path(config["label_plan"]), config["max_rows"], config["max_text_bytes"],
        config["max_source_bytes"], config["max_total_source_bytes"], config["max_manifest_bytes"],
        config["max_screened_file_bytes"],
        config["near_duplicate_similarity"], config["max_near_duplicate_pairs"],
        config["max_near_duplicate_work"], Path(config["screen_policy"]), config["max_policy_bytes"])
    review_ledger, review_ledger_digest = _load_review_ledger(Path(config["review_ledger"]),
                                                               config["max_ledger_bytes"])
    selected_ids = {row["id"] for row in train_rows + validation_rows}
    if set(review_ledger) != selected_ids:
        raise ValueError("review ledger must cover exactly the selected train and validation rows")
    review_decisions = {decision: sum(value == decision for value in review_ledger.values())
                        for decision in ("pending", "approved", "rejected")}
    taxonomy_digest = corpus["taxonomy_sha256"]
    field, selected = config["target_field"], config["selected_labels"]
    labels = sorted(selected)
    _, y_train, train_known = _matrix(train_rows, field, selected, taxonomy)
    _, y_validation, validation_known = _matrix(validation_rows, field, selected, taxonomy)
    if config["rejected_label_handling"] == "exclude":
        train_rejected = [review_ledger[row["id"]] == "rejected" for row in train_rows]
        validation_rejected = [review_ledger[row["id"]] == "rejected" for row in validation_rows]
        train_known[train_rejected, :] = False
        validation_known[validation_rejected, :] = False
    else:
        train_rejected = [False] * len(train_rows)
        validation_rejected = [False] * len(validation_rows)
    if not labels:
        raise ValueError("no selected targets")
    train_row_mask = train_known.any(axis=1)
    train_rows_with_labels = [row for row, known in zip(train_rows, train_row_mask, strict=True) if known]
    if not train_rows_with_labels:
        raise ValueError("no train rows have any known selected target")
    variable_targets, constant_targets, unsupported_targets = _target_partition(y_train, train_known)
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn"):
        try:
            from importlib.metadata import version
            versions[package] = version(package)
        except Exception:
            versions[package] = None
    results = {}
    train_support = {}
    validation_support = {}
    for i, label in enumerate(labels):
        known_train_values = y_train[train_known[:, i], i]
        known_validation_values = y_validation[validation_known[:, i], i]
        train_support[label] = {"positive": int(known_train_values.sum()),
                                "negative": int(len(known_train_values) - known_train_values.sum()),
                                "unknown_rows": int((~train_known[:, i]).sum())}
        validation_support[label] = {"positive": int(known_validation_values.sum()),
                                     "negative": int(len(known_validation_values) - known_validation_values.sum()),
                                     "unknown_rows": int((~validation_known[:, i]).sum())}
    text_train = [row["text"] for row, known in zip(train_rows, train_known, strict=True) if known.any()]
    y_fit, known_fit = y_train[train_row_mask], train_known[train_row_mask]
    text_val = [row["text"] for row in validation_rows]
    import time
    for algorithm, (vectorizer, _) in _estimators(config).items():
        start = time.perf_counter_ns()
        try:
            import numpy as np
            y_pred = _constant_fallbacks(y_fit, known_fit, labels, len(validation_rows),
                                         config["constant_target_behavior"])
            x_train = vectorizer.fit_transform(text_train)
            x_validation = vectorizer.transform(text_val)
            fitted_targets, fit_rows_by_target = [], {}
            knn_index_bytes = knn_target_bytes = knn_rows = 0
            knn_indexes = 0
            for index, label in enumerate(labels):
                target_mask = known_fit[:, index]
                target_values = y_fit[target_mask, index]
                if not len(target_values) or len(set(target_values.tolist())) < 2:
                    continue
                estimator = _estimator(algorithm, config, int(target_mask.sum()))
                estimator.fit(x_train[target_mask], target_values)
                predictions = estimator.predict(x_validation)
                y_pred[:, index] = predictions
                fitted_targets.append(label)
                fit_rows_by_target[label] = int(target_mask.sum())
                if algorithm == "knn":
                    retained = estimator._fit_X
                    retained_targets = np.asarray(estimator._y)
                    knn_index_bytes += int(retained.data.nbytes + retained.indices.nbytes + retained.indptr.nbytes)
                    knn_target_bytes += int(retained_targets.nbytes)
                    knn_rows += int(retained.shape[0])
                    knn_indexes += 1
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            results[algorithm] = {"status": "completed" if fitted_targets else
                                           "constant_target_fallback_only" if constant_targets else "no_supported_targets",
                                  "metrics": _metrics(y_validation, y_pred, validation_known, labels),
                                  "fit_vectorize_predict_elapsed_ms": elapsed,
                                  "feature_count": int(x_train.shape[1]),
                                  "fit_rows": len(train_rows_with_labels), "prediction_rows": len(validation_rows),
                                  "training_label_support": train_support,
                                  "fitted_targets": fitted_targets,
                                  "fit_rows_by_target": fit_rows_by_target,
                                  "constant_train_targets": [labels[i] for i in constant_targets],
                                  "unsupported_train_targets": [labels[i] for i in unsupported_targets],
                                  "unknown_validation_target_rows": int((~validation_known.any(axis=1)).sum()),
                                  "unknown_validation_target_cells": int((~validation_known).sum())}
            if algorithm == "knn":
                if knn_indexes:
                    results[algorithm]["retained_training_features"] = {
                        "target_indexes": knn_indexes, "rows_across_indexes": knn_rows,
                        "sparse_storage_bytes_across_indexes": knn_index_bytes,
                        "target_storage_bytes_across_indexes": knn_target_bytes,
                        "note": "measured retained sparse feature/target arrays only; excludes Python object and runtime overhead"}
                else:
                    results[algorithm]["retained_training_features"] = {"status": "no_fitted_knn_index"}
            else:
                results[algorithm]["retained_training_features"] = {"status": "not_applicable_to_prediction_index"}
        except Exception as error:
            results[algorithm] = {"status": "failed", "error_type": type(error).__name__,
                                  "fit_vectorize_predict_elapsed_ms": (time.perf_counter_ns() - start) / 1_000_000,
                                  "fit_rows": len(train_rows_with_labels), "prediction_rows": 0}
        results[algorithm]["persisted_model_artifact_bytes"] = 0
        results[algorithm]["parameters"] = {
            "tfidf_analyzer": "word", "tfidf_ngram_range": config["tfidf_ngram_range"],
            "tfidf_max_features": config["max_features"], "seed": config["seed"],
            "logistic_c": config["logistic_c"] if algorithm == "logistic_regression" else None,
            "logistic_max_iter": config["logistic_max_iter"] if algorithm == "logistic_regression" else None,
            "knn_neighbors": config["knn_neighbors"] if algorithm == "knn" else None,
            "knn_neighbors_by_target": {label: min(config["knn_neighbors"], int(train_known[:, i].sum()))
                                         for i, label in enumerate(labels)} if algorithm == "knn" else None,
            "knn_weights": config["knn_weights"] if algorithm == "knn" else None,
            "knn_p": config["knn_p"] if algorithm == "knn" else None,
            "tree_criterion": config["tree_criterion"] if algorithm == "decision_tree" else None,
            "tree_max_depth": config["tree_max_depth"] if algorithm == "decision_tree" else None}
    id_hashes = lambda rows: [sha(row["id"].encode("utf-8")) for row in sorted(rows, key=lambda item: item["id"])]
    report = {"schema_version": REPORT_SCHEMA, "experiment": "exploratory_classical_multilabel_comparison",
              "promotion_eligible": False, "qualification_claim": False,
              "data": {**corpus, **source_identity,
                       "train_group_claims": sorted({row["project_group"] for row in train_rows}),
                       "validation_group_claims": sorted({row["project_group"] for row in validation_rows}),
                       "train_rows": len(train_rows), "validation_rows": len(validation_rows),
                       "train_record_id_sha256": id_hashes(train_rows),
                       "validation_record_id_sha256": id_hashes(validation_rows),
                       "near_duplicate_similarity_limit": config["near_duplicate_similarity"],
                       "label_review_state": {"train": _label_review_state(train_rows, field),
                                              "validation": _label_review_state(validation_rows, field)},
                       "review_ledger_sha256": review_ledger_digest,
                       "review_ledger_decisions": review_decisions,
                       "rejected_label_handling": config["rejected_label_handling"],
                       "rejected_rows_excluded": {"train": sum(train_rejected),
                                                  "validation": sum(validation_rejected)},
                       "validation_group_use_claim": config["validation_group_use_claim"],
                       "validation_group_use_claim_verified": False,
                       "screening_redaction_complete": False,
                       "screening_matched_records": corpus["screening_matched_records"],
                       "independence_claim": "project/group/family and screened-text separation checked against the selected plan; source independence remains unverified"},
              "task": {"field": field, "labels": labels,
                       "semantics": "indicator matrix over configured labels; explicit completeness=false keeps omitted labels unknown",
                       "default_completeness_claim": "nonempty legacy arrays are treated as complete under the source schema; this claim is not independently verified",
                       "taxonomy_sha256": taxonomy_digest, "training_support": train_support,
                       "validation_support": validation_support},
              "config_sha256": config_digest, "code_sha256": sha(Path(__file__).read_bytes()),
              "runtime": {"python": platform.python_version(), "platform": platform.platform(), "packages": versions},
              "hardware_and_memory": {"cpu_identity": "unknown", "memory_usage": "unknown"},
              "algorithms": results,
              "external_student_comparisons": {"lfg": "unavailable_not_invoked", "owned_model": "unavailable_not_invoked"},
              "unavailable_tasks": {"complexity": "no selected explicit target", "agent_count": "no selected explicit target",
                                    "model": "no selected explicit target", "effort": "no selected explicit target",
                                    "unselected_taxonomy_fields": [x for x in FIELDS if x != field]},
              "limits": ["source review/verification status is reported per partition; labels are not independently verified",
                         "screening rules do not prove redaction completeness", "no binary model artifact was retained",
                         "fit_vectorize_predict_elapsed_ms covers TF-IDF fit/transform and estimator fit/predict; excludes loading, imports, metrics, and report writing"],
              "metric_policy": {"coverage": "reports unknown targets and abstentions separately; metrics use covered predictions",
                               "empty_or_missing_target": "unknown unless a present empty array has target_completeness[field] true",
                               "partial_target": "explicit positive labels remain known; omitted labels remain unknown",
                               "legacy_nonempty_array": "treated as complete under source schema when no field-level completeness flag exists; not independently verified",
                               "macro_precision": "standard macro includes predicted-only labels with false positives; zero-denominator metrics are null and excluded from macro means",
                               "macro_metric_coverage": "defined label counts and total labels are reported for each macro metric",
                               "constant_training_target": config["constant_target_behavior"]}}
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output.mkdir(mode=0o700)
    _require_outside_git_worktrees(output)
    _secure_owner_acl(output)
    _require_owner_acl(output)
    report_path = require_private_report_path(output / "comparison-report.json")
    fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    require_private_report_path(report_path, require_existing=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config-limit", type=int, default=65_536)
    args = parser.parse_args()
    try:
        report = run(args.config, args.output, args.config_limit)
    except (OSError, ValueError, TypeError, KeyError, ImportError, OverflowError) as error:
        raise SystemExit("comparison failed; inspect local configuration and inputs without sharing private files") from None
    print(json.dumps({"schema_version": report["schema_version"], "report_sha256": sha(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"),
        "promotion_eligible": False}, sort_keys=True))


if __name__ == "__main__":
    main()
