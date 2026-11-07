#!/usr/bin/env python3
"""Create private, source-bound local-teacher proposals from train-only masked exports."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import platform
import queue
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any, Callable

import compare_local_12b as comparator
masked_supervision = None

SCHEMA = "lfg.train-teacher-proposals-config.v1"
REPORT_SCHEMA = "lfg.train-teacher-proposals.v1"
PAYLOAD_SCHEMA = "lfg.train-teacher-worker-payload.v1"
WORKER_RESULT_SCHEMA = "lfg.train-teacher-worker-result.v1"
HEX = re.compile(r"^[0-9a-f]{64}$")
CONFIG_LIMIT = 65_536
WORKER_CONFIG_LIMIT = comparator.MAX_CONFIG_BYTES
MASKED_ROW_SCHEMA = "lfg.masked-supervision-row.v1"
MAX_TRAIN_PAYLOAD_ROWS = 100_000


class ProposalError(ValueError):
    pass


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _masked() -> Any:
    global masked_supervision
    if masked_supervision is None:
        import importlib
        masked_supervision = importlib.import_module("masked_supervision")
    return masked_supervision


def _read(path: Path, limit: int, what: str) -> bytes:
    try:
        return _masked()._read_bounded(path, limit, what)
    except (OSError, ValueError):
        raise ProposalError(f"unavailable, unsafe, or oversize {what} file") from None


def _json(raw: bytes, what: str) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (UnicodeError, ValueError, json.JSONDecodeError):
        raise ProposalError(f"invalid {what} JSON") from None


def _private_file(path: Path) -> bool:
    try:
        _masked()._verify_owner_private_access(path, is_directory=False)
        return True
    except (OSError, ValueError, AttributeError):
        return False


def _private_dir(path: Path) -> bool:
    try:
        _masked()._verify_owner_private_access(path, is_directory=True)
        return True
    except (OSError, ValueError, AttributeError):
        return False


def _outside_git(path: Path) -> bool:
    path = path.resolve()
    ancestor = path if path.exists() else path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    result = subprocess.run(["git", "-C", str(ancestor), "rev-parse", "--show-toplevel"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, text=True)
    if result.returncode == 0:
        root = Path(result.stdout.rstrip("\r\n")).resolve()
        return not path.is_relative_to(root)
    return not any((p / ".git").exists() for p in (ancestor, *ancestor.parents))


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink():
        raise ProposalError("configuration cannot be a symlink")
    path = path.resolve(strict=True)
    if not _private_file(path):
        raise ProposalError("configuration must be an owner-private regular file")
    raw = _read(path, CONFIG_LIMIT, "configuration")
    cfg = _json(raw, "configuration")
    required = {"schema_version", "revision", "export_dir", "export_manifest_sha256",
                "comparison_config", "output_dir", "selection", "max_rows",
                "assume_exhaustive_labels", "max_output_bytes", "max_protocol_bytes", "max_prompt_bytes", "max_export_bytes", "export_config_sha256",
                "export_train_sha256", "taxonomy_sha256", "selected_columns",
                "screening_policy_sha256", "screening_code_sha256"}
    if not isinstance(cfg, dict) or not required <= set(cfg) or cfg["schema_version"] != SCHEMA:
        raise ProposalError("unsupported proposal configuration")
    if not isinstance(cfg["revision"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", cfg["revision"]):
        raise ProposalError("invalid configuration revision")
    for field in ("export_manifest_sha256", "export_config_sha256", "export_train_sha256", "taxonomy_sha256",
                  "screening_policy_sha256", "screening_code_sha256"):
        if not isinstance(cfg[field], str) or not HEX.fullmatch(cfg[field]):
            raise ProposalError(f"invalid expected {field}")
    cols = cfg["selected_columns"]
    if (not isinstance(cols, list) or not cols
            or any(not isinstance(c, dict) or set(c) != {"field", "label"}
                   or not isinstance(c["field"], str) or not c["field"]
                   or not isinstance(c["label"], str) or not c["label"] for c in cols)
            or len({(c["field"], c["label"]) for c in cols}) != len(cols)):
        raise ProposalError("invalid pinned taxonomy columns")
    if type(cfg["max_rows"]) is not int or cfg["max_rows"] < 1:
        raise ProposalError("max_rows must be a positive integer")
    if (type(cfg["max_output_bytes"]) is not int or cfg["max_output_bytes"] < 1 or
            type(cfg["max_prompt_bytes"]) is not int or cfg["max_prompt_bytes"] < 1 or
            type(cfg["max_export_bytes"]) is not int or cfg["max_export_bytes"] < 1 or
            type(cfg["max_protocol_bytes"]) is not int or cfg["max_protocol_bytes"] < 1):
        raise ProposalError("invalid output or prompt bound")
    if type(cfg["assume_exhaustive_labels"]) is not bool:
        raise ProposalError("assume_exhaustive_labels must be boolean")
    if not isinstance(cfg["selection"], dict) or set(cfg["selection"]) - {"row_ids", "max_rows"}:
        raise ProposalError("selection must contain only row_ids and max_rows")
    row_ids = cfg["selection"].get("row_ids", [])
    if not isinstance(row_ids, list) or any(not isinstance(x, str) or not HEX.fullmatch(x) for x in row_ids) or len(row_ids) != len(set(row_ids)):
        raise ProposalError("selection row_ids must be unique digests")
    for key in ("export_dir", "comparison_config", "output_dir"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ProposalError(f"invalid {key}")
    if set(cfg) - required:
        # Contract extensions are retained by callers but have no behavior here.
        pass
    return cfg, _sha(raw)


def _private_temp_directory(prefix: str) -> Path:
    base = Path(tempfile.gettempdir())
    path = base / f"{prefix}-{uuid.uuid4().hex}"
    try:
        return comparator._private_report_directory(path)
    except (OSError, ValueError, subprocess.SubprocessError):
        raise ProposalError("cannot create owner-private temporary directory") from None


def _bound_export_bytes(cfg: dict[str, Any], max_manifest_bytes: int) -> tuple[Path, dict, bytes, bytes, bytes, dict]:
    export_input = Path(cfg["export_dir"])
    if export_input.is_symlink():
        raise ProposalError("masked export directory cannot be a symlink")
    export = export_input.resolve(strict=True)
    if not _private_dir(export):
        raise ProposalError("masked export directory is not owner-private")

    # Pin manifest bytes before considering any split payload; forbidden split names abort here.
    manifest_path = export / "manifest.json"
    manifest_raw = _read(manifest_path, max_manifest_bytes, "export manifest")
    if not _private_file(manifest_path) or _sha(manifest_raw) != cfg["export_manifest_sha256"]:
        raise ProposalError("export manifest identity or permissions changed")
    manifest = _json(manifest_raw, "export manifest")
    files = manifest.get("output_files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or set(files) != {"train.jsonl"}:
        raise ProposalError("export must contain exactly one train split")
    details = files.get("train.jsonl")
    if (not isinstance(details, dict) or set(details) != {"bytes", "sha256"}
            or type(details.get("bytes")) is not int or details["bytes"] < 1
            or not isinstance(details.get("sha256"), str) or not HEX.fullmatch(details["sha256"])):
        raise ProposalError("train file manifest entry is malformed")
    if details["bytes"] > cfg["max_export_bytes"]:
        raise ProposalError("train export exceeds configured byte limit")

    config_path = export / "config.json"
    export_config_raw = _read(config_path, 2 * 1024 * 1024, "export configuration")
    if (not _private_file(config_path) or _sha(export_config_raw) != manifest.get("config_sha256")
            or _sha(export_config_raw) != cfg["export_config_sha256"]):
        raise ProposalError("export configuration digest or permissions changed")
    train_path = export / "train.jsonl"
    train_raw = _read(train_path, details["bytes"], "train split")
    if (not _private_file(train_path) or len(train_raw) != details["bytes"]
            or _sha(train_raw) != details["sha256"] or details["sha256"] != cfg["export_train_sha256"]
            or manifest.get("taxonomy_sha256") != cfg["taxonomy_sha256"]
            or manifest.get("screening_policy_sha256") != cfg["screening_policy_sha256"]
            or manifest.get("screening_code_sha256") != cfg["screening_code_sha256"]
            or manifest.get("selected_columns") != cfg["selected_columns"]):
        raise ProposalError("train split identity or permissions changed")
    binding = {"manifest_sha256": _sha(manifest_raw), "config_sha256": _sha(export_config_raw),
               "train_sha256": _sha(train_raw), "train_bytes": len(train_raw)}
    return export, manifest, manifest_raw, export_config_raw, train_raw, binding


def _preflight_export(cfg: dict[str, Any], max_manifest_bytes: int = 16 * 1024 * 1024) -> tuple[dict, list[dict], dict]:
    export, manifest, manifest_raw, config_raw, train_raw, binding = _bound_export_bytes(cfg, max_manifest_bytes)
    snapshot = _private_temp_directory("lfg-train-export-bound")
    try:
        for name, raw in (("config.json", config_raw), ("manifest.json", manifest_raw), ("train.jsonl", train_raw)):
            _write_new(snapshot / name, raw)
            _masked()._verify_owner_private_access(snapshot / name, is_directory=False)
        _masked()._verify_owner_private_access(snapshot, is_directory=True)
        loaded_manifest, splits = _masked().load_export(snapshot, train_only=True)
        if loaded_manifest != manifest or set(splits) != {"train"}:
            raise ProposalError("bound export loader returned a different train-only inventory")
        rows = splits["train"]
        selected = set(cfg["selection"].get("row_ids", []))
        available = {row["id"] for row in rows}
        if selected and not selected <= available:
            raise ProposalError("selected train row identity is absent")
        return manifest, rows, binding
    finally:
        # This exact task-owned directory is private and contains only pinned export bytes.
        shutil.rmtree(snapshot)


def _reverify_export(cfg: dict[str, Any], original_binding: dict, max_manifest_bytes: int) -> None:
    _, _, _, _, _, current = _bound_export_bytes(cfg, max_manifest_bytes)
    if current != original_binding:
        raise ProposalError("pinned train export changed before transfer or publication")


def select_rows(rows: list[dict], selection: dict, max_rows: int) -> list[dict]:
    wanted = set(selection.get("row_ids", []))
    chosen = [row for row in rows if not wanted or row["id"] in wanted]
    limit = selection.get("max_rows", max_rows)
    if type(limit) is not int or limit < 1 or limit > max_rows:
        raise ProposalError("selection max_rows exceeds configured bound")
    if limit:
        chosen = chosen[:limit]
    if not chosen:
        raise ProposalError("selection produced no train rows")
    return chosen


def _payload_row(row: dict) -> dict:
    """Return only the train-row fields required by the isolated teacher worker."""
    fields = ("schema_version", "id", "split", "family", "text", "text_sha256",
              "original_text_sha256", "screening", "source_identity", "columns", "targets",
              "known_mask", "weights", "target_origins", "target_reviewer_claims",
              "verified_source_claim")
    if not isinstance(row, dict) or any(key not in row for key in fields):
        raise ProposalError("train row lacks required worker fields")
    return {key: row[key] for key in fields}


def _check_result_bounds(rows: list[dict], columns: list[dict], max_output_bytes: int,
                         max_protocol_bytes: int) -> None:
    labels = [f"{c['field']}:{c['label']}" for c in columns]
    output_bound = 0
    protocol_bound = 0
    for index, row in enumerate(rows):
        item = {"row_id": row["id"], "family_sha256": row["family"],
                "text_sha256": row["text_sha256"], "original_text_sha256": row["original_text_sha256"],
                "screening": row["screening"], "verified_source_claim": row["verified_source_claim"],
                "source_identity": row["source_identity"], "columns": columns,
                "source_targets": row["targets"], "source_known_mask": row["known_mask"],
                "source_target_origins": row["target_origins"], "source_reviewer_claims": row["target_reviewer_claims"],
                "status": "not_attempted_after_abort", "elapsed_ms": None,
                "proposal_states": {label: "proposed_negative" for label in labels},
                "conflict_labels": labels, "provenance": "unreviewed_local_teacher_claim"}
        encoded = (json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        event = {"phase": "progress", "index": index, "status": item["status"],
                 "elapsed_ms": None, "result": item}
        line = (json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(line) > 1_048_576:
            raise ProposalError("one worker result line exceeds protocol limit")
        output_bound += len(encoded)
        protocol_bound += len(line)
    if output_bound > max_output_bytes or protocol_bound > max_protocol_bytes:
        raise ProposalError("selected train rows exceed configured proposal or worker-protocol bounds")


def _request(config: dict, labels: list[str], text: str, max_bytes: int) -> bytes:
    schema = comparator.output_schema(labels)
    body = comparator._schema_request(config, schema, text)
    if len(body) > max_bytes:
        raise ProposalError("bounded model request exceeds configured limit")
    return body


def _create_private_output(path: Path) -> Path:
    path = path.resolve()
    if not _outside_git(path) or path.exists() or path.is_symlink():
        raise ProposalError("proposal output must be a new directory outside Git")
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(path, 0o700)
    if not _private_dir(path):
        raise ProposalError("proposal output directory is not owner-private")
    return path


def _write_new(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _validate_worker_payload(payload: Any, expected: dict) -> tuple[list[dict], list[dict]]:
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "binding", "records", "zero_row_probe"}:
        raise ProposalError("worker payload schema is invalid")
    binding = payload.get("binding")
    expected_payload_binding = expected["payload_binding"]
    if (payload.get("schema_version") != PAYLOAD_SCHEMA or not isinstance(binding, dict)
            or set(binding) != set(expected_payload_binding) or binding != expected_payload_binding
            or payload["zero_row_probe"] != expected["zero_row_probe"]
            or binding.get("manifest_sha256") != expected["export_manifest_sha256"]
            or binding.get("train_sha256") != expected["train_sha256"]
            or binding.get("taxonomy_sha256") != expected["taxonomy_sha256"]
            or binding.get("config_sha256") != expected["export_config_sha256"]):
        raise ProposalError("worker payload source binding is invalid")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) > expected["max_rows"]:
        raise ProposalError("worker payload row count is invalid")
    if type(payload.get("zero_row_probe")) is not bool:
        raise ProposalError("worker probe marker is invalid")
    if payload["zero_row_probe"] and records:
        raise ProposalError("zero-row probe cannot contain training records")
    if not payload["zero_row_probe"] and not records:
        raise ProposalError("training payload is empty")
    columns = expected["selected_columns"]
    labels = [f"{c['field']}:{c['label']}" for c in columns]
    if not labels or len(labels) != len(set(labels)):
        raise ProposalError("worker taxonomy is invalid")
    seen = set()
    for row in records:
        if not isinstance(row, dict) or set(row) != {"schema_version", "id", "split", "family", "text", "text_sha256", "original_text_sha256",
            "screening", "source_identity", "columns", "targets", "known_mask", "weights", "target_origins",
            "target_reviewer_claims", "verified_source_claim"}:
            raise ProposalError("worker row schema is invalid")
        if (row["schema_version"] != MASKED_ROW_SCHEMA or row["split"] != "train"
                or not isinstance(row["family"], str) or not HEX.fullmatch(row["family"])
                or not isinstance(row["id"], str) or not HEX.fullmatch(row["id"]) or row["id"] in seen
                or not isinstance(row["text"], str) or not row["text"].strip()
                or row["text_sha256"] != _sha(row["text"].encode("utf-8"))
                or not isinstance(row["original_text_sha256"], str) or not HEX.fullmatch(row["original_text_sha256"])
                or row["columns"] != columns or type(row["verified_source_claim"]) is not bool):
            raise ProposalError("worker row identity, split or taxonomy mismatch")
        screening = row["screening"]
        if (not isinstance(screening, dict) or set(screening) != {"policy_sha256", "redaction_counts", "text_changed"}
                or screening["policy_sha256"] != expected["screening_policy_sha256"]
                or not isinstance(screening["redaction_counts"], dict)
                or any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in screening["redaction_counts"].items())
                or type(screening["text_changed"]) is not bool):
            raise ProposalError("worker screening provenance is invalid")
        seen.add(row["id"])
        source = row["source_identity"]
        required_source = {"artifact_path", "artifact_sha256", "project_claim_sha256", "project_group_claim_sha256",
                           "record_id_sha256", "source_claim_sha256"}
        if (not isinstance(source, dict) or set(source) != required_source
                or not isinstance(source["artifact_path"], str) or Path(source["artifact_path"]).is_absolute()
                or ".." in Path(source["artifact_path"]).parts
                or any(not isinstance(source[k], str) or not HEX.fullmatch(source[k]) for k in required_source - {"artifact_path"})
                or source["record_id_sha256"] != row["id"]):
            raise ProposalError("worker source identity is invalid")
        count = len(columns)
        for key in ("targets", "known_mask", "weights", "target_origins", "target_reviewer_claims"):
            if not isinstance(row[key], list) or len(row[key]) != count:
                raise ProposalError("worker target arrays differ from taxonomy")
        for target, known, weight, origin, claim in zip(row["targets"], row["known_mask"], row["weights"],
                                                        row["target_origins"], row["target_reviewer_claims"], strict=True):
            if type(known) is not bool or not isinstance(origin, str) or not origin:
                raise ProposalError("worker target mask or origin is invalid")
            if claim is not None and (not isinstance(claim, str) or not HEX.fullmatch(claim)):
                raise ProposalError("worker reviewer claim digest is invalid")
            if known:
                if type(target) is not int or target not in (0, 1) or type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
                    raise ProposalError("worker known target or weight is invalid")
            elif target is not None or weight != 0:
                raise ProposalError("worker unknown target must stay null with zero weight")
    return records, [{"field": c["field"], "label": c["label"]} for c in columns]


def _worker_loop(records: list[dict], columns: list[dict], config: dict,
                 infer: Callable[[bytes, set[str]], tuple[str, list[str] | None, float]],
                 exhaustive: bool, max_prompt_bytes: int) -> tuple[list[dict], list[dict], bool, str | None]:
    labels = [f"{c['field']}:{c['label']}" for c in columns]
    allowed = set(labels)
    progress, result_rows = [], []
    aborted = False
    abort_reason = None
    for index, row in enumerate(records):
        if aborted:
            status, elapsed, positives = "not_attempted_after_abort", None, None
        else:
            try:
                request = _request(config, labels, row["text"], max_prompt_bytes)
                try:
                    status, positives, elapsed = infer(request, allowed)
                except comparator.RequestLimitError:
                    status, positives, elapsed = "request_too_large_not_started", None, None
                except TimeoutError:
                    status, positives, elapsed = "timeout", None, None
                    aborted, abort_reason = True, "timeout_or_worker_lifecycle_ambiguous"
                except (OSError, http.client.HTTPException):
                    status, positives, elapsed = "transport_error", None, None
                    aborted, abort_reason = True, "transport_or_worker_lifecycle_ambiguous"
                except ValueError:
                    status, positives, elapsed = "protocol_error", None, None
                    aborted, abort_reason = True, "protocol_or_worker_lifecycle_ambiguous"
            except ProposalError:
                status, elapsed, positives = "request_too_large_not_started", None, None
        if status not in {"answered", "abstain", "malformed", "invalid_schema", "refusal", "truncated",
                          "incomplete", "identity_mismatch", "http_error", "timeout", "transport_error",
                          "protocol_error", "request_too_large_not_started", "not_attempted_after_abort"}:
            status, positives, elapsed = "protocol_error", None, None
            aborted, abort_reason = True, "unknown_worker_status"
        if status in {"identity_mismatch", "http_error", "incomplete"}:
            aborted, abort_reason = True, "model_identity_or_response_lifecycle_ambiguous"
        if status == "answered" and positives is not None:
            states = {label: ("proposed_positive" if label in positives else
                              "proposed_negative" if exhaustive else "unknown") for label in labels}
        else:
            states = {label: "unknown" for label in labels}
        conflicts = []
        if status != "not_attempted_after_abort":
            for i, label in enumerate(labels):
                target, known = row["targets"][i], row["known_mask"][i]
                if known and ((target == 0 and states[label] == "proposed_positive") or
                              (target == 1 and states[label] == "proposed_negative")):
                    conflicts.append(label)
        item = {"row_id": row["id"], "family_sha256": row["family"],
                "text_sha256": row["text_sha256"], "original_text_sha256": row["original_text_sha256"],
                "screening": row["screening"], "verified_source_claim": row["verified_source_claim"],
                "source_identity": row["source_identity"], "columns": columns,
                "source_targets": row["targets"], "source_known_mask": row["known_mask"],
                "source_target_origins": row["target_origins"],
                "source_reviewer_claims": row["target_reviewer_claims"],
                "status": status, "elapsed_ms": elapsed, "proposal_states": states,
                "conflict_labels": conflicts, "provenance": "unreviewed_local_teacher_claim"}
        result_rows.append(item)
        progress.append({"phase": "progress", "index": index, "status": status,
                         "elapsed_ms": elapsed, "result": item})
    return progress, result_rows, aborted, abort_reason


def _worker(config_path: Path, expected_config_sha256: str) -> None:
    cfg, config_sha = comparator.load_config(config_path)
    if config_sha != expected_config_sha256 or cfg.get("worker_code_sha256") != _sha(Path(__file__).read_bytes()):
        raise ProposalError("worker code or configuration digest mismatch")
    receipt, receipt_sha = comparator.verify_acquisition(cfg)
    if (not sys.platform.startswith("linux") or not comparator._route_is_private()
            or os.geteuid() != cfg["remote_uid"]):
        raise ProposalError("worker is outside its configured private network namespace")
    resources_before = comparator.resource_snapshot(cfg, enforce=True)
    artifact = comparator.verify_installed_artifact(cfg, receipt)
    address, port, _ = comparator._loopback_endpoint(cfg["endpoint_url"])
    if address != "127.0.0.1":
        raise ProposalError("worker must use IPv4 loopback")
    server = subprocess.Popen([cfg["server_binary_path"], "serve"], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False, start_new_session=True,
        env={**os.environ, "OLLAMA_NO_CLOUD": "1", "OLLAMA_MODELS": cfg["model_store"],
             "OLLAMA_HOST": f"127.0.0.1:{port}"})
    response = None
    cleanup_verified = False
    try:
        deadline = time.monotonic() + cfg["worker_start_timeout_seconds"]
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise ProposalError("task-owned server exited before readiness")
            try:
                status, raw = comparator._http_call(cfg, None, method="GET", path_override="/api/version")
                if status == 200:
                    version = comparator._json(raw, "runtime version")
                    if not isinstance(version, dict) or version.get("version") != cfg["server_version"]:
                        raise ProposalError("runtime version identity mismatch")
                    break
            except (OSError, http.client.HTTPException, TimeoutError):
                pass
            time.sleep(0.25)
        else:
            raise TimeoutError("task-owned server did not become ready")
        runtime_cfg = {**cfg, "server_pid": server.pid}
        boundary = comparator.verify_runtime_boundary(runtime_cfg)
        control_nonce = uuid.uuid4().hex
        control_ready = {"phase": "control_ready", "control_nonce": control_nonce,
                 "network_namespace_inode": boundary["network_namespace_inode"],
                 "server_pid": server.pid, "port": port,
                 "receipt_sha256": receipt_sha,
                 "server_binary_sha256": boundary["server_binary_sha256"],
                 "server_version": boundary["server_version"],
                 "worker_code_sha256": cfg["worker_code_sha256"],
                 "worker_config_sha256": config_sha, "runtime_boundary": boundary,
                 "resources_before": resources_before}
        print(json.dumps(control_ready, sort_keys=True, separators=(",", ":")), flush=True)
        raw_ack = sys.stdin.buffer.readline(4097)
        if not raw_ack or len(raw_ack) > 4096 or not raw_ack.endswith(b"\n"):
            raise ProposalError("outside-boundary warmup acknowledgement is absent or invalid")
        ack = _json(raw_ack, "warmup acknowledgement")
        if (not isinstance(ack, dict) or set(ack) != {"phase", "control_nonce", "network_namespace_inode", "port"}
                or ack.get("phase") != "warmup_ack" or ack.get("control_nonce") != control_nonce
                or ack.get("network_namespace_inode") != boundary["network_namespace_inode"]
                or ack.get("port") != port):
            raise ProposalError("outside-boundary warmup acknowledgement does not match worker control state")
        warmup = comparator._warmup(cfg)
        artifact_identity = comparator.verify_loaded_model(cfg, artifact)
        boundary_after_warmup = comparator.verify_runtime_boundary(runtime_cfg)
        if (boundary_after_warmup.get("network_namespace_inode") != boundary.get("network_namespace_inode")
                or boundary_after_warmup.get("server_pid") != boundary.get("server_pid")
                or boundary_after_warmup.get("server_binary_sha256") != boundary.get("server_binary_sha256")
                or boundary_after_warmup.get("server_version") != boundary.get("server_version")):
            raise ProposalError("worker runtime boundary changed during isolated warmup")
        ready = {"phase": "ready_for_data", "network_namespace_inode": boundary_after_warmup["network_namespace_inode"],
                 "server_pid": server.pid, "port": port,
                 "model_manifest_sha256": artifact_identity["reported_manifest_sha256"],
                 "model_layer_bytes": artifact["model_layer_bytes"],
                 "artifact_layer_bytes": artifact["model_layer_bytes"], "receipt_sha256": receipt_sha,
                 "server_binary_sha256": boundary_after_warmup["server_binary_sha256"],
                 "server_version": boundary_after_warmup["server_version"],
                 "worker_code_sha256": cfg["worker_code_sha256"],
                 "worker_config_sha256": config_sha, "runtime_boundary": boundary_after_warmup,
                 "model_identity": artifact_identity, "warmup": warmup,
                 "resources_before": resources_before}
        print(json.dumps(ready, sort_keys=True, separators=(",", ":")), flush=True)
        raw_payload = sys.stdin.buffer.readline(cfg["max_payload_bytes"] + 1)
        if not raw_payload or len(raw_payload) > cfg["max_payload_bytes"] or not raw_payload.endswith(b"\n"):
            raise ProposalError("bounded worker payload is absent, oversized or incomplete")
        payload = _json(raw_payload, "worker payload")
        if payload.get("zero_row_probe") is True:
            records, columns = _validate_worker_payload(payload, cfg["teacher_binding"])
            if records:
                raise ProposalError("zero-row probe unexpectedly contains records")
            progress, outputs, aborted, abort_reason = [], [], False, None
        else:
            records, columns = _validate_worker_payload(payload, cfg["teacher_binding"])
        if not payload["zero_row_probe"]:
            progress, outputs, aborted, abort_reason = _worker_loop(records, columns, cfg,
                lambda body, labels: comparator._prediction_http(cfg, body, labels),
                cfg["teacher_binding"]["assume_exhaustive_labels"], cfg["teacher_binding"]["max_prompt_bytes"])
        encoded_progress = [(json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for event in progress]
        if (sum(map(len, encoded_progress)) > cfg["teacher_binding"]["max_protocol_bytes"]
                or any(len(line) > 1_048_576 for line in encoded_progress)):
            raise ProposalError("bounded worker progress protocol exceeded its configured limit")
        for line in encoded_progress:
            print(line.decode("utf-8").rstrip("\n"), flush=True)
        resources_after = comparator.resource_snapshot(cfg, enforce=False)
        response = {"phase": "result", "schema_version": WORKER_RESULT_SCHEMA,
                    "payload_sha256": _sha(raw_payload), "rows_received": len(outputs),
                    "statuses": [row["status"] for row in outputs],
                    "progress_sha256": _sha(b"".join(encoded_progress)), "aborted": aborted,
                    "abort_reason": abort_reason if aborted else None,
                    "resources_after": resources_after, "warmup": warmup, "artifact": artifact,
                    "model_identity": artifact_identity, "model_layer_bytes": artifact["model_layer_bytes"],
                    "receipt_sha256": receipt_sha,
                    "runtime_boundary": boundary, "worker_config_sha256": config_sha}
    finally:
        children = comparator._descendant_pids(server.pid)
        try:
            os.killpg(server.pid, 15)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server.pid, 9)
            except ProcessLookupError:
                pass
            server.wait(timeout=5)
        lingering = [pid for pid in children if Path(f"/proc/{pid}").exists()]
        if not lingering and server.poll() is not None:
            cleanup_verified = True
        if response is None:
            response = {"phase": "result", "schema_version": WORKER_RESULT_SCHEMA,
                        "worker_failed": True, "payload_sha256": None, "rows_received": 0,
                        "statuses": [], "rows": [], "aborted": True,
                        "abort_reason": "worker_failed_before_result", "cleanup_verified": cleanup_verified}
        response["cleanup_verified"] = cleanup_verified
        response["server_exit_code"] = server.returncode
    if not cleanup_verified:
        raise ProposalError("task-owned worker server cleanup failed")
    encoded = json.dumps(response, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > cfg["max_response_bytes"]:
        raise ProposalError("bounded worker result exceeds configured response limit")
    print(encoded, flush=True)
    if response.get("worker_failed"):
        raise SystemExit(2)


def _worker_command(config: dict, adapter_path: str, config_path: str, config_sha: str, unit: str) -> str:
    required = ("remote_sudo_path", "remote_systemd_run_path", "remote_unshare_path", "remote_bash_path",
                "remote_ip_path", "remote_setpriv_path", "remote_python_path")
    if any(not comparator._safe_absolute_posix_path(config[key]) for key in required):
        raise ProposalError("unsafe executable path in transient worker command")
    if (not comparator._valid_ssh_user(config["remote_user"]) or not comparator._valid_ssh_host(config["remote_host"])
            or not comparator._within_remote_home(adapter_path, config["remote_home"])
            or not comparator._within_remote_home(config_path, config["remote_home"])
            or not re.fullmatch(r"lfg12b-[0-9a-f]{20}\.scope", unit)):
        raise ProposalError("unsafe worker identity or remote path")
    inner = (shlex.join([config["remote_ip_path"], "link", "set", "lo", "up"])
             + " && exec " + shlex.join([config["remote_setpriv_path"], "--reuid=" + str(config["remote_uid"]),
                "--regid=" + str(config["remote_uid"]), "--init-groups", config["remote_python_path"], "-I",
                "-c", _isolated_worker_bootstrap(), adapter_path, config_path, config_sha]))
    argv = [config["remote_sudo_path"], "-n", config["remote_systemd_run_path"], "--scope", "--collect", "--quiet",
            f"--unit={unit}", "-p", f"MemoryMax={config['resource_memory_max_bytes']}",
            "-p", f"CPUQuota={config['resource_cpu_quota_percent']}%", "-p",
            f"TasksMax={config['remote_scope_tasks_max']}", "--", config["remote_unshare_path"], "--net", "--fork",
            "--mount-proc", config["remote_bash_path"], "-c", inner]
    return shlex.join(argv)


def _read_worker_line(process: subprocess.Popen, lines: queue.Queue, timeout: float) -> dict:
    try:
        line = lines.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError("isolated worker protocol deadline expired") from None
    if line is None or not isinstance(line, str) or len(line) > 1_048_576:
        raise ProposalError("isolated worker protocol ended or exceeded its bound")
    value = _json(line.encode("utf-8"), "worker protocol")
    if not isinstance(value, dict):
        raise ProposalError("isolated worker protocol must be an object")
    return value


def _worker_warmup_handshake(process: subprocess.Popen, lines: queue.Queue, comp: dict,
                             host_before: dict, adapter_sha: str, worker_config_sha: str,
                             receipt_sha: str) -> dict:
    control = _read_worker_line(process, lines,
        comp["worker_start_timeout_seconds"] + comp["worker_ready_wait_margin_seconds"])
    boundary = control.get("runtime_boundary") if isinstance(control, dict) else None
    nonce = control.get("control_nonce") if isinstance(control, dict) else None
    if (control.get("phase") != "control_ready" or not isinstance(boundary, dict)
            or not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{32}", nonce)
            or control.get("worker_code_sha256") != adapter_sha
            or control.get("worker_config_sha256") != worker_config_sha
            or control.get("receipt_sha256") != receipt_sha
            or control.get("server_binary_sha256") != comp["server_binary_sha256"]
            or control.get("server_version") != comp["server_version"]
            or control.get("port") != comp["runtime_port"]
            or type(control.get("server_pid")) is not int or control["server_pid"] <= 0
            or type(control.get("network_namespace_inode")) is not int
            or control["network_namespace_inode"] == host_before["netns_inode"]
            or boundary.get("network_namespace_inode") != control.get("network_namespace_inode")
            or boundary.get("server_pid") != control.get("server_pid")
            or boundary.get("boundary") != "same_uid_private_network_namespace_without_nonloopback_routes"
            or boundary.get("server_binary_sha256") != control.get("server_binary_sha256")
            or boundary.get("server_version") != control.get("server_version")):
        raise ProposalError("transient worker control boundary does not match pinned host/runtime identity")
    if not comparator._remote_port_closed(comp):
        raise ProposalError("host namespace can reach the task worker loopback listener")
    ack = {"phase": "warmup_ack", "control_nonce": nonce,
           "network_namespace_inode": control["network_namespace_inode"], "port": control["port"]}
    assert process.stdin is not None
    try:
        process.stdin.write((json.dumps(ack, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        process.stdin.flush()
    except OSError:
        raise ProposalError("empty warmup acknowledgement transport failed") from None
    ready = _read_worker_line(process, lines,
        comp["worker_start_timeout_seconds"] + comp["timeout_seconds"]
        + comp["worker_ready_wait_margin_seconds"])
    ready_boundary = ready.get("runtime_boundary") if isinstance(ready, dict) else None
    model_identity = ready.get("model_identity") if isinstance(ready, dict) else None
    layer_bytes = ready.get("artifact_layer_bytes") if isinstance(ready, dict) else None
    if (ready.get("phase") != "ready_for_data" or not isinstance(ready_boundary, dict)
            or not isinstance(model_identity, dict)
            or ready.get("worker_code_sha256") != adapter_sha
            or ready.get("worker_config_sha256") != worker_config_sha
            or ready.get("receipt_sha256") != receipt_sha
            or ready.get("network_namespace_inode") != control.get("network_namespace_inode")
            or ready.get("server_pid") != control.get("server_pid")
            or ready.get("port") != control.get("port")
            or ready.get("server_binary_sha256") != control.get("server_binary_sha256")
            or ready.get("server_version") != control.get("server_version")
            or ready_boundary.get("network_namespace_inode") != control.get("network_namespace_inode")
            or ready_boundary.get("server_pid") != control.get("server_pid")
            or ready_boundary.get("boundary") != "same_uid_private_network_namespace_without_nonloopback_routes"
            or ready_boundary.get("server_binary_sha256") != control.get("server_binary_sha256")
            or ready_boundary.get("server_version") != control.get("server_version")
            or type(layer_bytes) is not int or layer_bytes <= 0
            or type(ready.get("model_layer_bytes")) is not int
            or ready.get("model_layer_bytes") != layer_bytes
            or not isinstance(ready.get("model_manifest_sha256"), str)
            or not HEX.fullmatch(ready["model_manifest_sha256"])
            or model_identity.get("reported_manifest_sha256") != ready.get("model_manifest_sha256")
            or ready.get("warmup", {}).get("status") != "completed"):
        raise ProposalError("transient worker is not verified and ready for data")
    return ready


def _validate_progress_result(event: dict, index: int, row: dict, columns: list[dict],
                              exhaustive: bool) -> dict:
    item = event.get("result")
    expected_keys = {"row_id", "family_sha256", "text_sha256", "original_text_sha256", "screening",
        "verified_source_claim", "source_identity", "columns", "source_targets", "source_known_mask",
        "source_target_origins", "source_reviewer_claims", "status", "elapsed_ms", "proposal_states",
        "conflict_labels", "provenance"}
    status = event.get("status")
    if (event.get("phase") != "progress" or event.get("index") != index
            or not isinstance(item, dict) or set(item) != expected_keys
            or item.get("row_id") != row["id"]
            or item.get("family_sha256") != row["family"]
            or item.get("text_sha256") != row["text_sha256"]
            or item.get("original_text_sha256") != row["original_text_sha256"]
            or item.get("screening") != row["screening"]
            or item.get("verified_source_claim") != row["verified_source_claim"]
            or item.get("source_identity") != row["source_identity"]
            or item.get("columns") != columns
            or item.get("source_targets") != row["targets"]
            or item.get("source_known_mask") != row["known_mask"]
            or item.get("source_target_origins") != row["target_origins"]
            or item.get("source_reviewer_claims") != row["target_reviewer_claims"]
            or item.get("status") != status
            or item.get("elapsed_ms") != event.get("elapsed_ms")
            or item.get("provenance") != "unreviewed_local_teacher_claim"):
        raise ProposalError("worker progress result does not match its source-bound input row")
    ordered_labels = [f"{column['field']}:{column['label']}" for column in columns]
    labels = set(ordered_labels)
    states = item.get("proposal_states")
    if (not isinstance(states, dict) or set(states) != labels
            or any(value not in {"proposed_positive", "proposed_negative", "unknown"} for value in states.values())):
        raise ProposalError("worker proposal states differ from the pinned taxonomy")
    if status != "answered" and any(value != "unknown" for value in states.values()):
        raise ProposalError("non-answer worker status must leave every proposal unknown")
    if status == "answered" and exhaustive and any(value == "unknown" for value in states.values()):
        raise ProposalError("exhaustive answered rows must explicitly mark every label")
    if status == "answered" and not exhaustive and any(value == "proposed_negative" for value in states.values()):
        raise ProposalError("omitted labels cannot become negative without the explicit exhaustive policy")
    expected_conflicts = []
    if status != "not_attempted_after_abort":
        for index, label in enumerate(ordered_labels):
            target, known = row["targets"][index], row["known_mask"][index]
            if known and ((target == 0 and states[label] == "proposed_positive")
                          or (target == 1 and states[label] == "proposed_negative")):
                expected_conflicts.append(label)
    if item.get("conflict_labels") != expected_conflicts:
        raise ProposalError("worker conflicts do not match source masks and proposal states")
    return item


def _validate_progress_event(event: dict, index: int, row: dict, columns: list[dict],
                             exhaustive: bool) -> dict:
    statuses = {"answered", "abstain", "malformed", "invalid_schema", "refusal", "truncated",
                "incomplete", "identity_mismatch", "http_error", "timeout", "transport_error",
                "protocol_error", "request_too_large_not_started", "not_attempted_after_abort"}
    status = event.get("status") if isinstance(event, dict) else None
    elapsed = event.get("elapsed_ms") if isinstance(event, dict) else None
    unknown_elapsed = {"timeout", "transport_error", "protocol_error",
                       "request_too_large_not_started", "not_attempted_after_abort"}
    if (not isinstance(event, dict) or event.get("phase") != "progress" or event.get("index") != index
            or status not in statuses
            or (status in unknown_elapsed and elapsed is not None)
            or (status not in unknown_elapsed and
                (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0))):
        raise ProposalError("worker progress status or timing protocol is invalid")
    return _validate_progress_result(event, index, row, columns, exhaustive)


def _atomic_private_publish(final_path: Path, proposals: list[dict], manifest: dict) -> tuple[Path, str]:
    if final_path.is_symlink():
        raise ProposalError("final proposal directory cannot be a symlink")
    final_path = final_path.resolve()
    if final_path.exists() or final_path.is_symlink() or not _outside_git(final_path):
        raise ProposalError("final proposal directory must be new and outside Git")
    parent = final_path.parent
    if not parent.exists() or not _private_dir(parent):
        raise ProposalError("proposal output parent must already be owner-private")
    publish_status = manifest.get("status")
    if publish_status not in {"complete", "partial_aborted_ambiguous"}:
        raise ProposalError("proposal publication status is not a recognized terminal state")
    partial = parent / f".{final_path.name}.partial-{uuid.uuid4().hex}"
    partial = comparator._private_report_directory(partial)
    incomplete = {"schema_version": REPORT_SCHEMA, "status": "incomplete", "promotion_eligible": False,
                  "run_id": partial.name}
    try:
        _write_new(partial / "manifest.json", (json.dumps(incomplete, sort_keys=True) + "\n").encode())
    except Exception:
        try:
            shutil.rmtree(partial)
        except OSError:
            pass
        raise
    try:
        payload = ("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                              for row in proposals) + "\n").encode("utf-8")
        _write_new(partial / "proposals.jsonl", payload)
        manifest = {**manifest, "status": publish_status, "proposals_sha256": _sha(payload),
                    "proposals_bytes": len(payload), "rows_written": len(proposals)}
        raw_manifest = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
        temp_manifest = partial / ".manifest-complete.tmp"
        _write_new(temp_manifest, raw_manifest)
        if final_path.exists():
            raise ProposalError("final proposal directory appeared during publication")
        os.rename(partial, final_path)
        temp_manifest = final_path / temp_manifest.name
        # The destination remains explicitly incomplete until the last atomic file replace.
        os.replace(temp_manifest, final_path / "manifest.json")
        return final_path, _sha(raw_manifest)
    except Exception:
        # Keep the exact private partial directory and its incomplete marker for operator cleanup.
        try:
            if 'temp_manifest' in locals() and temp_manifest.exists():
                temp_manifest.unlink()
        except OSError:
            pass
        raise



def _remote_stage_paths(comp: dict, workdir: str, unit_id: str) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{32}", unit_id):
        raise ProposalError("invalid task identifier")
    paths = {"workdir": workdir,
             "teacher": f"{workdir}/train_teacher_proposals.py",
             "comparator": f"{workdir}/compare_local_12b.py",
             "config": f"{workdir}/worker-config.json",
             "model_store": f"{workdir}/models"}
    if not comparator._within_remote_home(workdir, comp["remote_home"]):
        raise ProposalError("task staging path is outside the configured remote home")
    if any(not comparator._within_remote_home(paths[k], comp["remote_home"])
           for k in ("teacher", "comparator", "config", "model_store")):
        raise ProposalError("task staging path escaped the configured remote home")
    return paths


def _remote_create_workdir(comp: dict, workdir: str) -> None:
    home, base, target = comp["remote_home"], comp["remote_workdir"], workdir
    script = "\n".join((
        "import os,pathlib,stat,sys",
        f"home=pathlib.Path({home!r}).resolve(strict=True); base=pathlib.Path({base!r}); target=pathlib.Path({target!r}); uid=os.geteuid()",
        "if home.stat().st_uid!=uid or not base.is_absolute() or home not in base.parents: sys.exit(31)",
        "cur=home",
        "for part in base.relative_to(home).parts:",
        " cur=cur/part",
        " if cur.exists() or cur.is_symlink():",
        "  st=cur.lstat()",
        "  if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode) or st.st_uid!=uid: sys.exit(32)",
        " else:",
        "  cur.mkdir(mode=0o700); os.chmod(cur,0o700)",
        "st=base.lstat()",
        "if (st.st_mode&0o077)!=0: sys.exit(33)",
        "target.mkdir(mode=0o700,exist_ok=False); os.chmod(target,0o700)",
        "st=target.lstat()",
        "sys.exit(0 if stat.S_ISDIR(st.st_mode) and st.st_uid==uid and (st.st_mode&0o777)==0o700 else 34)",
    ))
    result = comparator._ssh_run(comp, script, timeout=30)
    if result.returncode != 0:
        raise ProposalError("owner-private remote task directory creation failed")


def _remote_verify_files(comp: dict, paths: dict, expected: dict[str, tuple[str, int]]) -> None:
    files = {path: digest for path, (digest, _) in expected.items()}
    sizes = {path: size for path, (_, size) in expected.items()}
    script = "\n".join((
        "import hashlib,json,os,pathlib,stat,sys",
        f"files={files!r}; sizes={sizes!r}; uid={comp['remote_uid']}; root=pathlib.Path({paths['workdir']!r})",
        "rst=root.lstat(); ok=(stat.S_ISDIR(rst.st_mode) and rst.st_uid==uid and (rst.st_mode&0o777)==0o700)",
        "ok=ok and all(pathlib.Path(name).is_relative_to(root) for name in files)",
        "for name,digest in files.items():",
        " p=pathlib.Path(name); st=p.lstat()",
        " if not stat.S_ISREG(st.st_mode) or st.st_uid!=uid: ok=False; continue",
        " if st.st_mode&0o077: os.chmod(p,0o600)",
        " st=p.lstat(); raw=p.read_bytes()",
        " ok=ok and st.st_uid==uid and (st.st_mode&0o077)==0 and st.st_size==sizes[name] and hashlib.sha256(raw).hexdigest()==digest",
        "print(json.dumps({'verified':bool(ok),'files':len(files)})); sys.exit(0 if ok else 35)",
    ))
    result = comparator._ssh_run(comp, script, timeout=30)
    if result.returncode != 0:
        raise ProposalError("remote staged code or config identity/permissions failed")
    value = comparator._json(result.stdout.strip(), "remote staging receipt")
    if not isinstance(value, dict) or value.get("verified") is not True or value.get("files") != len(expected):
        raise ProposalError("remote staging receipt did not verify all files")


def _remote_cleanup_worker(comp: dict, workdir: str, unit: str) -> None:
    root = comparator.PurePosixPath(comp["remote_workdir"])
    path = comparator.PurePosixPath(workdir)
    if (not comparator._within_remote_home(workdir, comp["remote_home"])
            or path.parent != root or not re.fullmatch(r"run-[0-9a-f]{32}", path.name)
            or not re.fullmatch(r"lfg12b-[0-9a-f]{20}\.scope", unit)):
        raise ProposalError("refusing cleanup outside this transient worker")
    script = "\n".join((
        "import os,pathlib,shutil,stat,subprocess,sys",
        f"uid={comp['remote_uid']}; target=pathlib.Path({workdir!r}); unit={unit!r}; sudo={comp['remote_sudo_path']!r}; systemctl={comp['remote_systemctl_path']!r}",
        "st=target.lstat()",
        "if not stat.S_ISDIR(st.st_mode) or st.st_uid!=uid or (st.st_mode&0o777)!=0o700: sys.exit(41)",
        "subprocess.run([sudo,'-n',systemctl,'stop',unit],capture_output=True,timeout=20)",
        "state=subprocess.run([sudo,'-n',systemctl,'is-active','--quiet',unit],capture_output=True,timeout=10)",
        "if state.returncode==0: sys.exit(42)",
        "if state.returncode not in (3,4): sys.exit(43)",
        "shutil.rmtree(target)",
        "sys.exit(0)",
    ))
    result = comparator._ssh_run(comp, script, timeout=35)
    if result.returncode != 0:
        raise ProposalError("transient worker cleanup could not verify stopped scope and remove its private directory")


def _worker_config(comp: dict, paths: dict, teacher_binding: dict, teacher_code_sha: str) -> dict:
    return {**comp, "acquisition_receipt": comp["remote_acquisition_receipt"],
            "model_store": paths["model_store"], "remote_model_store": paths["model_store"],
            "remote_adapter_path": paths["teacher"],
            "endpoint_url": f"http://127.0.0.1:{comp['runtime_port']}/api/chat",
            "worker_code_sha256": teacher_code_sha,
            "comparator_code_sha256": teacher_binding["comparator_code_sha256"],
            "teacher_binding": teacher_binding}


def _isolated_worker_bootstrap() -> str:
    """Explicitly admit only the staged, digest-pinned sibling comparator under Python -I."""
    return ("import hashlib,json,pathlib,runpy,sys; "
            "a,c,h=sys.argv[1:4]; p=pathlib.Path(c); raw=p.read_bytes(); "
            "assert hashlib.sha256(raw).hexdigest()==h; cfg=json.loads(raw); "
            "d=pathlib.Path(a).parent; q=d/'compare_local_12b.py'; "
            "assert hashlib.sha256(pathlib.Path(a).read_bytes()).hexdigest()==cfg['worker_code_sha256']; "
            "assert hashlib.sha256(q.read_bytes()).hexdigest()==cfg['comparator_code_sha256']; "
            "sys.path.insert(0,str(d)); sys.argv=[a,'--worker','--config',c,'--config-sha256',h]; "
            "runpy.run_path(a,run_name='__main__')")


def _remote_identity_snapshot(comp: dict) -> dict:
    snapshot = comparator._remote_snapshot(comp)
    if (snapshot.get("hostname") != comp["remote_expected_hostname"]
            or snapshot.get("architecture") != comp["remote_expected_architecture"]):
        raise ProposalError("remote host identity or architecture differs from the pinned configuration")
    return snapshot


def _validate_zero_probe_match(probe: dict, ready: dict, host_before: dict, cfg_sha: str,
                               comp_sha: str, adapter_sha: str, comparator_sha: str,
                               receipt_sha: str) -> None:
    identity = probe.get("model_identity") if isinstance(probe, dict) else None
    probe_host = probe.get("host_before", {}) if isinstance(probe, dict) else None
    boundary = probe.get("runtime_boundary", {}) if isinstance(probe, dict) else None
    artifact = probe.get("artifact") if isinstance(probe, dict) else None
    if (not isinstance(probe, dict) or probe.get("zero_row_probe") is not True
            or probe.get("cleanup_verified") is not True
            or probe.get("outside_listener_denied") is not True
            or probe.get("persistent_service_unchanged") is not True
            or probe.get("proposal_config_sha256") != cfg_sha
            or probe.get("comparison_config_sha256") != comp_sha
            or probe.get("adapter_code_sha256") != adapter_sha
            or probe.get("comparator_code_sha256") != comparator_sha
            or probe.get("receipt_sha256") != receipt_sha
            or not isinstance(probe_host, dict)
            or probe_host.get("hostname") != host_before.get("hostname")
            or probe_host.get("architecture") != host_before.get("architecture")
            or not isinstance(identity, dict)
            or identity.get("reported_manifest_sha256") != ready.get("model_manifest_sha256")
            or probe.get("model_layer_bytes") != ready.get("model_layer_bytes")
            or not isinstance(artifact, dict)
            or artifact.get("model_layer_bytes") != ready.get("artifact_layer_bytes")
            or not isinstance(boundary, dict)
            or boundary.get("server_binary_sha256") != ready.get("server_binary_sha256")
            or boundary.get("server_version") != ready.get("server_version")):
        raise ProposalError("zero-row probe identity does not match the actual worker")


def _run_transient_worker(cfg: dict, cfg_sha: str, comp: dict, comp_sha: str,
                          local_receipt: dict, local_receipt_sha: str,
                          manifest: dict | None, rows: list[dict] | None, binding: dict | None,
                          zero_rows: bool, expected_probe: dict | None = None) -> dict:
    host_before = _remote_identity_snapshot(comp)
    if (host_before["memory_available_bytes"] < comp["min_memory_available_bytes"]
            or host_before["home_disk_free_bytes"] < comp["min_disk_free_bytes"]
            or type(host_before.get("loaded_model_count")) is not int):
        raise ProposalError("remote host identity, capacity or service coexistence preflight failed")
    if not comparator._remote_port_closed(comp):
        raise ProposalError("configured outside-host loopback port is occupied")
    run_id = uuid.uuid4().hex
    unit = f"lfg12b-{run_id[:20]}.scope"
    probe_unit = f"probe-{run_id[:12]}.scope"
    base = comp["remote_workdir"].rstrip("/")
    workdir = f"{base}/run-{run_id}"
    paths = _remote_stage_paths(comp, workdir, run_id)
    preflight_argv = [comp["remote_sudo_path"], "-n", comp["remote_systemd_run_path"],
        "--scope", "--collect", "--quiet", f"--unit={probe_unit}",
        "-p", f"MemoryMax={comp['resource_memory_max_bytes']}",
        "-p", f"CPUQuota={comp['resource_cpu_quota_percent']}%",
        "-p", f"TasksMax={comp['remote_scope_tasks_max']}", "--", comp["remote_true_path"]]
    preflight_script = f"import subprocess,sys; r=subprocess.run({preflight_argv!r},capture_output=True,timeout=30); sys.exit(r.returncode)"
    if comparator._ssh_run(comp, preflight_script, timeout=40).returncode != 0:
        raise ProposalError("transient cgroup resource preflight failed")

    adapter_path = Path(__file__).resolve(strict=True)
    comparator_path = Path(comparator.__file__).resolve(strict=True)
    adapter_raw, comparator_raw = adapter_path.read_bytes(), comparator_path.read_bytes()
    adapter_sha, comparator_sha = _sha(adapter_raw), _sha(comparator_raw)
    teacher_binding = {"export_manifest_sha256": cfg["export_manifest_sha256"],
        "train_sha256": binding["train_sha256"] if binding else cfg["export_train_sha256"],
        "taxonomy_sha256": cfg["taxonomy_sha256"],
        "export_config_sha256": binding["config_sha256"] if binding else cfg["export_config_sha256"],
        "selected_columns": cfg["selected_columns"], "max_rows": cfg["max_rows"],
        "screening_policy_sha256": cfg["screening_policy_sha256"],
        "screening_code_sha256": cfg["screening_code_sha256"],
        "max_prompt_bytes": cfg["max_prompt_bytes"], "max_output_bytes": cfg["max_output_bytes"],
        "max_protocol_bytes": cfg["max_protocol_bytes"],
        "assume_exhaustive_labels": cfg["assume_exhaustive_labels"],
        "proposal_config_sha256": cfg_sha, "comparison_config_sha256": comp_sha,
        "adapter_code_sha256": adapter_sha, "comparator_code_sha256": comparator_sha,
        "zero_row_probe": bool(zero_rows)}
    payload_binding = {"manifest_sha256": teacher_binding["export_manifest_sha256"],
        "train_sha256": teacher_binding["train_sha256"], "taxonomy_sha256": teacher_binding["taxonomy_sha256"],
        "config_sha256": teacher_binding["export_config_sha256"],
        "proposal_config_sha256": cfg_sha, "comparison_config_sha256": comp_sha,
        "adapter_code_sha256": adapter_sha, "comparator_code_sha256": comparator_sha}
    teacher_binding["payload_binding"] = payload_binding
    worker_config = _worker_config(comp, paths, teacher_binding, adapter_sha)
    worker_config_raw = (json.dumps(worker_config, sort_keys=True, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n").encode("utf-8")
    if len(worker_config_raw) > WORKER_CONFIG_LIMIT:
        raise ProposalError("private worker configuration exceeds bound")
    worker_config_sha = _sha(worker_config_raw)
    install_config = {**worker_config, "remote_adapter_path": paths["comparator"]}
    work_created = False
    process = None
    progress_rows: list[dict] = []
    sent_payload_sha = None
    result = None
    installed = None
    cleanup_error = None
    try:
        _remote_create_workdir(comp, workdir)
        work_created = True
        local_config_dir = _private_temp_directory("lfg-teacher-worker-config")
        worker_config_path_local = local_config_dir / "worker-config.json"
        _write_new(worker_config_path_local, worker_config_raw)
        _masked()._verify_owner_private_access(worker_config_path_local, is_directory=False)
        for local_path, remote_path, raw in ((adapter_path, paths["teacher"], adapter_raw),
                                              (comparator_path, paths["comparator"], comparator_raw),
                                              (worker_config_path_local, paths["config"], worker_config_raw)):
            argv = ["scp", "-oBatchMode=yes", "-oConnectTimeout=10", "-P", str(comp["ssh_port"]),
                    str(local_path), f"{comp['remote_user']}@{comp['remote_host']}:{remote_path}"]
            copied = subprocess.run(argv, check=False, capture_output=True, timeout=120, shell=False)
            if copied.returncode != 0:
                raise ProposalError("bounded private worker file transfer failed")
        _remote_verify_files(comp, paths, {paths["teacher"]: (adapter_sha, len(adapter_raw)),
            paths["comparator"]: (comparator_sha, len(comparator_raw)),
            paths["config"]: (worker_config_sha, len(worker_config_raw))})
        install_config["remote_adapter_path"] = paths["comparator"]
        installed = comparator._remote_install(install_config, paths["config"])
        if installed.get("receipt_sha256") != local_receipt_sha:
            raise ProposalError("remote model receipt differs from locally pinned receipt")
        if not comparator._remote_port_closed(comp):
            raise ProposalError("outside-host loopback port became occupied before worker start")
        command = _worker_command(comp, paths["teacher"], paths["config"], worker_config_sha, unit)
        process = subprocess.Popen(["ssh", "-T", "-oBatchMode=yes", "-oConnectTimeout=10", "-p", str(comp["ssh_port"]),
            f"{comp['remote_user']}@{comp['remote_host']}", command], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=False)
        lines = comparator._queue_protocol(process)
        ready = _worker_warmup_handshake(process, lines, comp, host_before, adapter_sha,
                                         worker_config_sha, local_receipt_sha)
        if ready.get("model_manifest_sha256") != comp["expected_manifest_sha256"]:
            raise ProposalError("warmed model identity differs from the pinned expected manifest")
        if not zero_rows:
            _validate_zero_probe_match(expected_probe, ready, host_before, cfg_sha, comp_sha,
                                       adapter_sha, comparator_sha, local_receipt_sha)
        if zero_rows:
            payload = {"schema_version": PAYLOAD_SCHEMA, "binding": payload_binding,
                       "records": [], "zero_row_probe": True}
            selected_ids = []
        else:
            assert manifest is not None and rows is not None and binding is not None
            _reverify_export(cfg, binding, comp["max_manifest_bytes"])
            payload = {"schema_version": PAYLOAD_SCHEMA, "binding": payload_binding,
                       "records": [_payload_row(row) for row in rows], "zero_row_probe": False}
            selected_ids = [row["id"] for row in rows]
        payload_raw = (json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        if len(payload_raw) > comp["max_payload_bytes"]:
            raise ProposalError("source-bound train payload exceeds the configured transfer bound")
        sent_payload_sha = _sha(payload_raw)
        assert process.stdin is not None
        try:
            process.stdin.write(payload_raw)
            process.stdin.flush()
            process.stdin.close()
        except OSError:
            raise ProposalError("worker input transport failed after readiness") from None
        count = 0 if zero_rows else len(rows or [])
        batch_timeout = float(comp["timeout_seconds"]) * (count + 2) + comp["batch_overhead_margin_seconds"]
        if not math.isfinite(batch_timeout) or batch_timeout > sys.maxsize / 1_000_000_000:
            raise ProposalError("batch timeout exceeds safe platform arithmetic")
        deadline = time.monotonic() + batch_timeout
        expected_rows = rows or []
        for index in range(count):
            event = _read_worker_line(process, lines, max(0.001, deadline - time.monotonic()))
            row = expected_rows[index]
            item = _validate_progress_event(event, index, row, cfg["selected_columns"],
                                            cfg["assume_exhaustive_labels"])
            progress_rows.append(item)
        progress_digest = _sha(b"".join(
            (json.dumps({"phase": "progress", "index": i, "status": row["status"],
                         "elapsed_ms": row["elapsed_ms"], "result": row}, sort_keys=True,
                        ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            for i, row in enumerate(progress_rows)))
        result = _read_worker_line(process, lines, max(0.001, deadline - time.monotonic()))
        return_code = process.wait(timeout=30)
        reported_statuses = [row["status"] for row in progress_rows]
        requires_abort = any(status in {"timeout", "transport_error", "protocol_error", "identity_mismatch",
                                        "http_error", "incomplete", "not_attempted_after_abort"}
                             for status in reported_statuses)
        if (return_code != 0 or result.get("phase") != "result"
                or result.get("schema_version") != WORKER_RESULT_SCHEMA
                or result.get("payload_sha256") != sent_payload_sha
                or result.get("rows_received") != count
                or result.get("statuses") != reported_statuses
                or type(result.get("aborted")) is not bool
                or result.get("aborted") != requires_abort
                or (requires_abort and not isinstance(result.get("abort_reason"), str))
                or result.get("progress_sha256") != progress_digest
                or result.get("cleanup_verified") is not True
                or result.get("model_layer_bytes") != ready.get("model_layer_bytes")
                or result.get("artifact", {}).get("model_layer_bytes") != ready.get("artifact_layer_bytes")
                or result.get("model_identity", {}).get("reported_manifest_sha256") != ready.get("model_manifest_sha256")
                or result.get("receipt_sha256") != local_receipt_sha
                or result.get("runtime_boundary", {}).get("network_namespace_inode") != ready.get("network_namespace_inode")
                or result.get("worker_config_sha256") != worker_config_sha
                or result.get("worker_failed") is True):
            raise ProposalError("worker result, cleanup, or progress accounting failed")
        if zero_rows and (result.get("rows_received") != 0 or result.get("warmup", {}).get("status") != "completed"):
            raise ProposalError("zero-row probe did not complete isolated empty-input warmup")
        host_after = comparator._remote_snapshot(comp)
        if (host_after["ollama_service_pid"] != host_before["ollama_service_pid"]
                or host_after["ollama_version"] != host_before["ollama_version"]
                or host_after["loaded_model_count"] != host_before["loaded_model_count"]):
            raise ProposalError("persistent host Ollama state changed during task-owned run")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        if work_created:
            try:
                _remote_cleanup_worker(comp, workdir, unit)
            except (OSError, ValueError, subprocess.SubprocessError):
                cleanup_error = "task-owned remote cleanup failed"
        if 'local_config_dir' in locals():
            try:
                shutil.rmtree(local_config_dir)
            except OSError:
                cleanup_error = "private local worker-config cleanup failed"
    if cleanup_error:
        raise ProposalError(cleanup_error)
    if result is None:
        raise ProposalError("transient worker ended without a final result")
    if zero_rows:
        return {"zero_row_probe": True, "rows_sent": 0, "train_payload_opened": False,
                "worker_ready": True, "outside_listener_denied": True,
                "runtime_boundary": result["runtime_boundary"], "model_identity": result["model_identity"],
                "artifact": result["artifact"], "receipt_sha256": result["receipt_sha256"],
                "model_layer_bytes": ready["model_layer_bytes"],
                "resources_before": ready["resources_before"], "resources_after": result["resources_after"],
                "warmup": result["warmup"], "cleanup_verified": result["cleanup_verified"],
                "persistent_service_unchanged": True, "host_before": host_before, "host_after": host_after,
                "proposal_config_sha256": cfg_sha, "comparison_config_sha256": comp_sha,
                "adapter_code_sha256": adapter_sha, "comparator_code_sha256": comparator_sha}
    assert manifest is not None and rows is not None and binding is not None
    _reverify_export(cfg, binding, comp["max_manifest_bytes"])
    if len(progress_rows) != len(rows):
        raise ProposalError("worker returned incomplete row status coverage")
    status_counts = dict(sorted(Counter(row["status"] for row in progress_rows).items()))
    label_coverage = Counter()
    for row in progress_rows:
        for label, state in row["proposal_states"].items():
            label_coverage[f"{label}:{state}"] += 1
    artifacts = manifest.get("source_artifacts", [])
    artifact_hashes = {item["path"]: item["sha256"] for item in artifacts
                       if isinstance(item, dict) and isinstance(item.get("path"), str)
                       and isinstance(item.get("sha256"), str)}
    source_copy_claim = manifest.get("source_copy_claim", {})
    source_lineage = {"source_artifacts": artifacts, "source_artifacts_sha256": artifact_hashes,
        "source_artifact_inventory_sha256": _sha(json.dumps(artifact_hashes, sort_keys=True, separators=(",", ":")).encode()),
        "screening_policy_sha256": manifest.get("screening_policy_sha256"),
        "screening_code_sha256": manifest.get("screening_code_sha256"),
        "taxonomy_sha256": manifest.get("taxonomy_sha256"),
        "taxonomy_loader_sha256": manifest.get("taxonomy_loader_sha256"),
        "export_config_sha256": manifest.get("config_sha256"),
        "workspace_plan_sha256": manifest.get("workspace_plan_sha256") or source_copy_claim.get("plan_sha256"),
        "review_ledger_sha256": manifest.get("review_ledger_sha256") or source_copy_claim.get("ledger_sha256"),
        "partial_sidecar_sha256": manifest.get("partial_sidecar_sha256") or source_copy_claim.get("sidecar_sha256"),
        "label_guide_sha256": manifest.get("label_guide_sha256"),
        "target_policy": manifest.get("target_policy"), "selected_columns": manifest.get("selected_columns"),
        "counts": manifest.get("counts"), "promotion_eligible": False}
    model = result["model_identity"]
    output_manifest = {"schema_version": REPORT_SCHEMA, "created_at_epoch_seconds": int(time.time()),
        "status": "complete" if not result.get("aborted") else "partial_aborted_ambiguous",
        "promotion_eligible": False, "claims": {"teacher_suggestions_are_unreviewed": True,
            "heldout_accessed": False, "labels_independently_verified": False,
            "screening_independently_verified": False, "promotion_eligible": False},
        "provenance": {"proposal_config_sha256": cfg_sha, "comparison_config_sha256": comp_sha,
            "adapter_code_sha256": adapter_sha, "comparator_code_sha256": comparator_sha,
            "worker_config_sha256": worker_config_sha, "payload_sha256": sent_payload_sha,
            "export_manifest_sha256": binding["manifest_sha256"], "export_config_sha256": binding["config_sha256"],
            "train_file_sha256": binding["train_sha256"], "train_file_bytes": binding["train_bytes"],
            "taxonomy_sha256": cfg["taxonomy_sha256"], "source_lineage": source_lineage,
            "selected_row_count": len(rows), "selected_row_ids_sha256": _sha(json.dumps(selected_ids, separators=(",", ":")).encode()),
            "selected_text_digests_sha256": _sha(json.dumps([r["text_sha256"] for r in rows], separators=(",", ":")).encode()),
            "target_mask_sha256": _sha(json.dumps([r["known_mask"] for r in rows], separators=(",", ":")).encode()),
            "proposal_schema_sha256": _sha(json.dumps(comparator.output_schema([f"{c['field']}:{c['label']}" for c in cfg["selected_columns"]]),
                sort_keys=True, separators=(",", ":")).encode()),
            "prompt_contract_sha256": _sha((comp["system_prompt"] + "\0" + comp["prompt_revision"]).encode()),
            "model_name": comp["model_name"], "model_manifest_sha256": model["reported_manifest_sha256"],
            "model_layer_bytes": result["artifact"]["model_layer_bytes"], "receipt_sha256": result["receipt_sha256"],
            "server_binary_sha256": comp["server_binary_sha256"], "server_version": comp["server_version"],
            "runtime_boundary": result["runtime_boundary"],
            "request_policy": {k: comp[k] for k in ("timeout_seconds", "max_request_bytes", "max_response_bytes",
                "num_ctx", "num_predict", "temperature", "top_k", "top_p", "repeat_penalty")},
            "resource_policy": {k: comp[k] for k in ("resource_memory_max_bytes", "resource_cpu_quota_percent",
                "remote_scope_tasks_max", "min_memory_available_bytes", "min_disk_free_bytes", "min_gpu_free_bytes")}},
        "counts": {"selected_rows": len(rows), "statuses": status_counts,
            "answered_rows": status_counts.get("answered", 0), "abstentions": status_counts.get("abstain", 0),
            "failures": sum(n for status, n in status_counts.items() if status not in ("answered", "abstain")),
            "not_attempted_after_abort": status_counts.get("not_attempted_after_abort", 0),
            "per_label_and_state": dict(sorted(label_coverage.items()))},
        "pre_data_zero_row_probe": expected_probe,
        "abort": {"aborted": result.get("aborted", False), "reason": result.get("abort_reason")},
        "resources": {"before": ready["resources_before"], "after": result["resources_after"]},
        "uncertainty": {"source_and_review_claims_independently_verified": False,
            "teacher_labels_are_claims": True, "omitted_labels_are_negative": cfg["assume_exhaustive_labels"],
            "selection_claim": "configured train-only subset; no heldout data opened"}}
    final_path, manifest_sha = _atomic_private_publish(Path(cfg["output_dir"]), progress_rows, output_manifest)
    return {"rows_selected": len(rows), "rows_completed": len(progress_rows),
            "manifest_sha256": manifest_sha, "output_directory": str(final_path),
            "status_counts": status_counts, "promotion_eligible": False}


def run(config_path: Path, zero_rows: bool = False) -> dict:
    cfg, cfg_sha = load_config(config_path)
    comp_input = Path(cfg["comparison_config"])
    if comp_input.is_symlink():
        raise ProposalError("comparison configuration cannot be a symlink")
    comp_path = comp_input.resolve(strict=True)
    if not _private_file(comp_path):
        raise ProposalError("comparator configuration must be owner-private")
    comp, comp_sha = comparator.load_config(comp_path)
    local_receipt, local_receipt_sha = comparator.verify_acquisition(comp)
    # Reject a stale or unintended destination before opening any train payload.
    _remote_identity_snapshot(comp)
    manifest = rows = binding = None
    if not zero_rows:
        manifest, all_rows, binding = _preflight_export(cfg, comp["max_manifest_bytes"])
        rows = select_rows(all_rows, cfg["selection"], cfg["max_rows"])
        if len(rows) > MAX_TRAIN_PAYLOAD_ROWS:
            raise ProposalError("selected row count exceeds hard protocol bound")
        _check_result_bounds(rows, manifest["selected_columns"], cfg["max_output_bytes"], cfg["max_protocol_bytes"])
    if zero_rows:
        return _run_transient_worker(cfg, cfg_sha, comp, comp_sha, local_receipt, local_receipt_sha,
                                     None, None, None, True)
    probe = _run_transient_worker(cfg, cfg_sha, comp, comp_sha, local_receipt, local_receipt_sha,
                                   None, None, None, True)
    return _run_transient_worker(cfg, cfg_sha, comp, comp_sha, local_receipt, local_receipt_sha,
                                 manifest, rows, binding, False, expected_probe=probe)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--zero-row-probe", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--config-sha256")
    args = parser.parse_args()
    try:
        if args.worker:
            if not args.config_sha256 or not HEX.fullmatch(args.config_sha256):
                raise ProposalError("worker config digest is required")
            _worker(args.config, args.config_sha256)
            return
        if args.config_sha256 is not None:
            raise ProposalError("worker config digest is only valid in worker mode")
        result = run(args.config, zero_rows=args.zero_row_probe)
        print(json.dumps(result, sort_keys=True))
    except (ProposalError, OSError, ValueError, TypeError, KeyError, AttributeError, subprocess.SubprocessError):
        raise SystemExit("Teacher proposal task failed; check private configuration, source identity and bounds.") from None


if __name__ == "__main__":
    main()
