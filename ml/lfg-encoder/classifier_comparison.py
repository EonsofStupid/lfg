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
from collections import Counter
from pathlib import Path
from typing import Any

import evaluate
import screen_private_corpus

SCHEMA = "lfg.classifier-comparison-config.v1"
REPORT_SCHEMA = "lfg.classifier-comparison-report.v1"
FIELDS = ("intents", "capabilities", "risks")
ALGORITHMS = {"logistic_regression", "knn", "decision_tree"}
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
    required = {"schema_version", "revision", "screened_corpus", "label_plan", "target_field", "selected_labels",
                "seed", "max_rows", "max_text_bytes", "max_source_bytes", "max_manifest_bytes",
                "max_total_source_bytes", "max_screened_file_bytes", "max_features", "knn_neighbors", "tree_max_depth",
                "tfidf_ngram_range", "logistic_c", "logistic_max_iter", "knn_weights", "knn_p",
                "tree_criterion", "near_duplicate_similarity"}
    if (not isinstance(config, dict) or not required <= set(config) or config.get("schema_version") != SCHEMA
            or not isinstance(config.get("revision"), str) or not config["revision"].strip()
            or not isinstance(config.get("screened_corpus"), str) or not config["screened_corpus"].strip()
            or not isinstance(config.get("label_plan"), str) or not config["label_plan"].strip()
            or config.get("target_field") not in FIELDS or not isinstance(config.get("selected_labels"), list)
            or not config["selected_labels"] or any(not isinstance(x, str) or not x.strip() for x in config["selected_labels"])):
        raise ValueError("invalid comparison config")
    for key in ("seed", "max_rows", "max_text_bytes", "max_source_bytes", "max_manifest_bytes",
                "max_total_source_bytes", "max_screened_file_bytes",
                "max_features", "knn_neighbors"):
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
    return config, sha(raw)


def _read_bounded(path: Path, limit: int, name: str) -> bytes:
    if type(limit) is not int or limit < 1:
        raise ValueError(f"invalid {name} byte limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"{name} exceeds configured byte limit")
    return raw


def load_screened_corpus(directory: Path, plan_path: Path, max_rows: int,
                         max_text_bytes: int, max_source_bytes: int,
                         max_total_source_bytes: int, max_manifest_bytes: int, max_screened_file_bytes: int,
                         near_duplicate_similarity: float) -> tuple[list[dict], list[dict], dict, dict, dict]:
    """Join screened text to explicit labels, reading only declared train/validation sources."""
    directory = directory.resolve(strict=True)
    manifest_raw = _read_bounded(directory / "manifest.json", max_manifest_bytes, "screen manifest")
    manifest = _json_bytes(manifest_raw, "screen manifest")
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != "lfg.private-corpus-screen-manifest.v1"
            or not isinstance(manifest.get("outputs"), dict) or not isinstance(manifest.get("counts"), dict)
            or not isinstance(manifest.get("claims"), dict)):
        raise ValueError("unsupported screened corpus manifest")
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
    taxonomy_path = screen_private_corpus.corpus_audit.resolve(plan_path.parent, plan.get("taxonomy"))
    taxonomy_raw = _read_bounded(taxonomy_path, max_manifest_bytes, "taxonomy")
    taxonomy, taxonomy_digest = evaluate.load_taxonomy(taxonomy_path)
    outputs = manifest["outputs"]
    if set(outputs) - {"train", "validation"}:
        raise ValueError("screen manifest includes a non-development split")
    if not {"train", "validation"} <= set(outputs):
        raise ValueError("screened corpus requires train and validation outputs")
    screened_by_split = {}
    for split in ("train", "validation"):
        digest = outputs[split]
        if not isinstance(digest, str) or not HEX.fullmatch(digest):
            raise ValueError("invalid screened output digest")
        raw = _read_bounded(directory / f"{split}.jsonl", max_screened_file_bytes, f"{split} data")
        if sha(raw) != digest:
            raise ValueError(f"{split} output digest mismatch")
        rows = []
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                raise ValueError("blank screened data line")
            row = _json_bytes(line.encode("utf-8"), "screened row")
            if not isinstance(row, dict) or row.get("schema_version") != screen_private_corpus.OUTPUT_VERSION:
                raise ValueError("unsupported screened row")
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
            rows.append(row)
            if len(rows) > max_rows:
                raise ValueError("screened corpus exceeds configured row limit")
        if len(rows) != manifest["counts"].get(split):
            raise ValueError(f"{split} count mismatch")
        screened_by_split[split] = rows
    source_rows_by_split = {"train": [], "validation": []}
    source_evidence = manifest.get("source_snapshots")
    if not isinstance(source_evidence, list) or len(source_evidence) != len(sources):
        raise ValueError("screened manifest source evidence mismatch")
    total_source_bytes = 0
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
            record_hash = sha(line.encode("utf-8"))
            screen_row = next((row for row in screened_by_split[split] if row.get("id") == identifier), None)
            if screen_row is None or screen_row.get("source_record_sha256") != record_hash:
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
            source_record["text"] = screen_row["text"]
            source_record["project"] = screen_row["project"]
            source_record["project_group"] = screen_row["project_group"]
            source_record["split"] = screen_row["split"]
            source_rows_by_split[split].append(source_record)
    all_rows = source_rows_by_split["train"] + source_rows_by_split["validation"]
    ids, family_splits, text_splits = set(), {}, {}
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
    # Pairwise near-duplicate checking is bounded by max_rows; reject suspicious pairs.
    # This is a conservative screen, not a semantic independence claim.
    import difflib
    for train_row in source_rows_by_split["train"]:
        left = " ".join(train_row["text"].casefold().split())
        for validation_row in source_rows_by_split["validation"]:
            right = " ".join(validation_row["text"].casefold().split())
            if difflib.SequenceMatcher(None, left, right, autojunk=False).ratio() >= near_duplicate_similarity:
                raise ValueError("near-duplicate text crosses train/validation")
    return source_rows_by_split["train"], source_rows_by_split["validation"], {
        "manifest_sha256": sha(manifest_raw), "train_sha256": outputs["train"],
        "validation_sha256": outputs["validation"], "manifest_claims": manifest["claims"],
        "policy_sha256": manifest.get("policy_sha256"), "screen_code_sha256": manifest.get("screen_code_sha256"),
        "screening_matched_records": manifest.get("matched_records"),
        "source_artifacts": source_artifacts,
        "taxonomy_sha256": taxonomy_digest,
        "counts": {split: len(source_rows_by_split[split]) for split in source_rows_by_split}}, taxonomy, {
            "plan_sha256": sha(plan_raw), "taxonomy_sha256": sha(taxonomy_raw)}


def _matrix(rows: list[dict], field: str, selected: list[str], taxonomy: dict[str, set[str]]):
    from sklearn.preprocessing import MultiLabelBinarizer
    supported = taxonomy[field]
    if len(set(selected)) != len(selected) or set(selected) - supported:
        raise ValueError("selected labels are duplicate or outside the configured taxonomy")
    labels = sorted(selected)
    targets = []
    for row in rows:
        values = row.get(field)
        if not isinstance(values, list) or any(not isinstance(x, str) or not x.strip() for x in values):
            raise ValueError("target arrays must be present and contain nonempty labels")
        if len(values) != len(set(values)) or set(values) - supported:
            raise ValueError("invalid or out-of-taxonomy target array")
        if row.get("labels_pending") is True or row.get("verified") is not False:
            raise ValueError("screened labels must remain explicitly unreviewed, non-pending source labels")
        # An explicit complete array is used as a multilabel target. Empty arrays
        # stay empty; a missing array is rejected above, never converted to negatives.
        targets.append([label for label in values if label in set(labels)])
    encoder = MultiLabelBinarizer(classes=labels)
    return encoder, encoder.fit_transform(targets)


class _ConstantBinary:
    def __init__(self, value: int): self.value = value
    def fit(self, x, y): return self
    def predict(self, x):
        import numpy as np
        return np.full(x.shape[0], self.value, dtype=int)


def _estimators(config: dict[str, Any], n_train: int):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    k = min(config["knn_neighbors"], n_train)
    return {
        "logistic_regression": (TfidfVectorizer(ngram_range=tuple(config["tfidf_ngram_range"]),
                                                 max_features=config["max_features"], dtype=np.float64),
                                OneVsRestClassifier(LogisticRegression(random_state=config["seed"],
                                                                       max_iter=config["logistic_max_iter"],
                                                                       C=config["logistic_c"]))),
        "knn": (TfidfVectorizer(ngram_range=tuple(config["tfidf_ngram_range"]),
                                max_features=config["max_features"], dtype=np.float64),
                KNeighborsClassifier(n_neighbors=k, weights=config["knn_weights"], p=config["knn_p"])),
        "decision_tree": (TfidfVectorizer(ngram_range=tuple(config["tfidf_ngram_range"]),
                                           max_features=config["max_features"], dtype=np.float64),
                          DecisionTreeClassifier(random_state=config["seed"], max_depth=config["tree_max_depth"],
                                                 criterion=config["tree_criterion"]))
    }


def _metrics(y_true, y_pred, labels: list[str]) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
    supports = y_true.sum(axis=0).astype(int)
    predicted = y_pred.sum(axis=0).astype(int)
    by_label = {}
    for index, label in enumerate(labels):
        tn, fp, fn, tp = (int(x) for x in multilabel_confusion_matrix(y_true[:, index], y_pred[:, index], labels=[0, 1])[1].ravel())
        precision_denom, recall_denom, f1_denom = tp + fp, tp + fn, 2 * tp + fp + fn
        by_label[label] = {"support": int(supports[index]), "predicted_positive": int(predicted[index]),
                           "negative_support": int(tn + fp),
                           "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                           "precision": tp / precision_denom if precision_denom else None,
                           "recall": tp / recall_denom if recall_denom else None,
                           "f1": 2 * tp / f1_denom if f1_denom else None}
    supported_f1 = [value["f1"] for value in by_label.values() if value["support"] > 0]
    return {"examples": int(y_true.shape[0]), "subset_accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1_supported_labels": (sum(supported_f1) / len(supported_f1) if supported_f1 else None),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "labels": by_label, "unknown_or_abstained": 0}


def run(config_path: Path, output: Path, config_limit: int = 65_536) -> dict[str, Any]:
    config, config_digest = load_config(config_path, config_limit)
    directory = Path(config["screened_corpus"])
    screen_private_corpus.require_private_outputs(output)
    if output.exists():
        raise ValueError("output directory already exists")
    train_rows, validation_rows, corpus, taxonomy, source_identity = load_screened_corpus(
        directory, Path(config["label_plan"]), config["max_rows"], config["max_text_bytes"],
        config["max_source_bytes"], config["max_total_source_bytes"], config["max_manifest_bytes"],
        config["max_screened_file_bytes"],
        config["near_duplicate_similarity"])
    taxonomy_digest = corpus["taxonomy_sha256"]
    field, selected = config["target_field"], config["selected_labels"]
    labels = sorted(selected)
    train_encoder, y_train = _matrix(train_rows, field, selected, taxonomy)
    val_encoder, y_validation = _matrix(validation_rows, field, selected, taxonomy)
    if not labels:
        raise ValueError("no selected targets")
    label_source = Counter(label for row in train_rows for label in row[field] if label in set(labels))
    if not any(label_source.values()):
        raise ValueError("no selected target has positive training support")
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn"):
        try:
            from importlib.metadata import version
            versions[package] = version(package)
        except Exception:
            versions[package] = None
    results = {}
    train_support = {label: {"positive": int(y_train[:, i].sum()),
                             "negative": int(y_train.shape[0] - y_train[:, i].sum())}
                     for i, label in enumerate(labels)}
    validation_support = {label: {"positive": int(y_validation[:, i].sum()),
                                  "negative": int(y_validation.shape[0] - y_validation[:, i].sum())}
                          for i, label in enumerate(labels)}
    text_train = [row["text"] for row in train_rows]
    text_val = [row["text"] for row in validation_rows]
    import time
    for algorithm, (vectorizer, estimator) in _estimators(config, len(train_rows)).items():
        start = time.perf_counter_ns()
        try:
            x_train = vectorizer.fit_transform(text_train)
            x_validation = vectorizer.transform(text_val)
            if algorithm == "logistic_regression":
                from sklearn.multiclass import OneVsRestClassifier
                from sklearn.linear_model import LogisticRegression
                # A label with one observed value has no fitted decision boundary;
                # report its output as always absent without inventing a model fit.
                active = [i for i in range(y_train.shape[1]) if len(set(y_train[:, i].tolist())) > 1]
                predictions = [[0] * y_train.shape[1] for _ in range(y_validation.shape[0])]
                if active:
                    fitted = OneVsRestClassifier(LogisticRegression(random_state=config["seed"],
                                                                    max_iter=config["logistic_max_iter"],
                                                                    C=config["logistic_c"]))
                    fitted.fit(x_train, y_train[:, active])
                    active_predictions = fitted.predict(x_validation)
                    if active_predictions.ndim == 1:
                        active_predictions = active_predictions.reshape(-1, 1)
                    for row_idx in range(len(predictions)):
                        for col_idx, original_idx in enumerate(active):
                            predictions[row_idx][original_idx] = int(active_predictions[row_idx][col_idx])
                import numpy as np
                y_pred = np.asarray(predictions, dtype=int)
            else:
                estimator.fit(x_train, y_train[:, 0] if y_train.shape[1] == 1 and algorithm == "knn" else y_train)
                y_pred = estimator.predict(x_validation)
                if getattr(y_pred, "ndim", 1) == 1:
                    y_pred = y_pred.reshape(-1, 1)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            results[algorithm] = {"status": "completed", "metrics": _metrics(y_validation, y_pred, labels),
                                  "elapsed_ms": elapsed, "feature_count": int(x_train.shape[1]),
                                  "fit_rows": len(train_rows), "prediction_rows": len(validation_rows),
                                  "training_label_support": train_support,
                                  "zero_variance_train_targets": [label for i, label in enumerate(labels)
                                                                   if len(set(y_train[:, i].tolist())) < 2]}
            if algorithm == "knn":
                retained = estimator._fit_X
                results[algorithm]["retained_training_features"] = {
                    "rows": int(retained.shape[0]), "columns": int(retained.shape[1]),
                    "sparse_storage_bytes": int(retained.data.nbytes + retained.indices.nbytes + retained.indptr.nbytes),
                    "target_matrix_bytes": int(y_train.nbytes),
                    "note": "measured retained feature/target arrays only; excludes Python object and runtime overhead"}
            else:
                results[algorithm]["retained_training_features"] = {"status": "not_applicable_to_prediction_index"}
        except Exception as error:
            results[algorithm] = {"status": "failed", "error_type": type(error).__name__,
                                  "elapsed_ms": (time.perf_counter_ns() - start) / 1_000_000,
                                  "fit_rows": len(train_rows), "prediction_rows": 0}
        results[algorithm]["persisted_model_artifact_bytes"] = 0
        results[algorithm]["parameters"] = {
            "tfidf_analyzer": "word", "tfidf_ngram_range": config["tfidf_ngram_range"],
            "tfidf_max_features": config["max_features"], "seed": config["seed"],
            "logistic_c": config["logistic_c"] if algorithm == "logistic_regression" else None,
            "logistic_max_iter": config["logistic_max_iter"] if algorithm == "logistic_regression" else None,
            "knn_neighbors": min(config["knn_neighbors"], len(train_rows)) if algorithm == "knn" else None,
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
                       "all_labels_unreviewed": True, "screening_redaction_complete": False,
                       "screening_matched_records": corpus["screening_matched_records"],
                       "independence_claim": "operator_declared_development_split; not independently verified"},
              "task": {"field": field, "labels": labels,
                       "semantics": "indicator matrix over configured labels from complete explicit multilabel arrays",
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
              "limits": ["all source labels are unreviewed proposals", "validation group was previously used",
                         "screening rules do not prove redaction completeness", "no binary model artifact was retained",
                         "elapsed time covers TF-IDF fit/transform and estimator fit/predict; excludes loading, imports, metrics, and report writing"]}
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output.mkdir(parents=True, mode=0o700)
    fd = os.open(output / "comparison-report.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
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
