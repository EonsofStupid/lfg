#!/usr/bin/env python3
"""Create private, source-bound local-teacher proposals from train-only masked exports."""
from __future__ import annotations

import argparse
import contextlib
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
import teacher_explicit_labels as explicit_labels
masked_supervision = None

SCHEMA = "lfg.train-teacher-proposals-config.v1"
REPORT_SCHEMA = "lfg.train-teacher-proposals.v1"
PAYLOAD_SCHEMA = "lfg.train-teacher-worker-payload.v2"
WORKER_RESULT_SCHEMA = "lfg.train-teacher-worker-result.v1"
PROGRESS_ACK_SCHEMA = "lfg.teacher-progress-ack.v1"
PROGRESS_ACK_MAX_BYTES = 1024
HEX = re.compile(r"^[0-9a-f]{64}$")
CONFIG_LIMIT = 65_536
WORKER_CONFIG_LIMIT = comparator.MAX_CONFIG_BYTES
MASKED_ROW_SCHEMA = "lfg.masked-supervision-row.v1"
MAX_TRAIN_PAYLOAD_ROWS = 100_000
PROGRESS_STORE_SCHEMA = "lfg.teacher-progress-store.v1"
PROGRESS_EVENT_SCHEMA = "lfg.teacher-progress-event.v1"
PROGRESS_TERMINAL_SCHEMA = "lfg.teacher-progress-terminal.v1"
PROGRESS_REVISION_SCHEMA = "lfg.teacher-progress-revision.v1"
PROGRESS_PROBE_RECEIPT_SCHEMA = "lfg.teacher-progress-probe-receipt.v1"
PROGRESS_PUBLICATION_SCHEMA = "lfg.teacher-progress-publication.v1"
PROGRESS_ATTEMPT_SCHEMA = "lfg.teacher-progress-attempt.v1"
PRIVATE_EXTENSION_SCHEMA = "lfg.teacher-private-extensions.v1"
PROGRESS_STORE_HARD_MAX_BYTES = 1_073_741_824
PROGRESS_STORE_DEFAULT_BYTES = 67_108_864
PROGRESS_CAPACITY_SCHEMA = "lfg.teacher-progress-capacity.v1"
PROGRESS_HEADER_MAX_BYTES = 16 * 1024 * 1024
PROGRESS_TERMINAL_MAX_BYTES = 32 * 1024 * 1024
PROGRESS_EVENT_MAX_BYTES = 1_048_576
PUBLICATION_TEMP_EVIDENCE_MAX = 32

# These diagnostics are deliberately independent of exception text and config values.
DIAGNOSTIC_SCHEMA = "lfg.teacher-lifecycle-diagnostic.v1"
DIAGNOSTIC_PHASES = frozenset({
    "config_validation", "receipt_validation", "remote_snapshot", "port_check",
    "transient_preflight", "task_directory", "remote_task_directory",
    "local_temp_directory", "local_config_file", "file_staging", "file_verification",
    "model_install", "worker_startup", "control_handshake", "outside_port_denial",
    "warmup_readiness", "payload_result", "host_recheck", "cleanup", "unknown",
    "worker_config_validation", "worker_receipt_validation", "worker_boundary_check",
    "worker_resource_snapshot", "worker_artifact_check", "worker_server_startup",
    "worker_warmup", "worker_payload", "worker_result", "worker_cleanup",
    "source_recheck", "publication",
})
_DIAGNOSTIC_PHASE = "unknown"
_DIAGNOSTIC_CLEANUP = "not_started"


def _diagnostic_phase(value: str) -> None:
    global _DIAGNOSTIC_PHASE
    _DIAGNOSTIC_PHASE = value if value in DIAGNOSTIC_PHASES else "unknown"


def _diagnostic_category(exc: BaseException) -> str:
    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
        return "timeout"
    if isinstance(exc, (OSError, subprocess.SubprocessError)):
        return "transport_or_io_failure"
    if isinstance(exc, ProposalError):
        return "validation_or_protocol_failure"
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError)):
        return "validation_or_protocol_failure"
    return "internal_failure"


def _diagnostic_record(exc: BaseException) -> dict[str, str]:
    record = {"schema_version": DIAGNOSTIC_SCHEMA,
              "phase": _DIAGNOSTIC_PHASE if _DIAGNOSTIC_PHASE in DIAGNOSTIC_PHASES else "unknown",
              "category": _diagnostic_category(exc), "cleanup": _DIAGNOSTIC_CLEANUP}
    if isinstance(exc, WorkerDiagnosticError):
        record.update({"worker_phase": exc.worker_phase, "worker_category": exc.worker_category,
                       "worker_cleanup": exc.worker_cleanup})
    return record


class ProposalError(ValueError):
    pass


class WorkerDiagnosticError(ProposalError):
    def __init__(self, phase: str, category: str, cleanup: str):
        super().__init__("worker reported a bounded lifecycle failure")
        self.worker_phase = phase
        self.worker_category = category
        self.worker_cleanup = cleanup


class PrivateTeacherExtensions:
    """Deliberately retrieved, owner-private inert extension values."""
    __slots__ = ("__metadata", "__extensions", "__present")

    def __init__(self, metadata: dict, extensions: dict | None, present: bool):
        self.__metadata = json.loads(json.dumps(metadata, ensure_ascii=False))
        self.__extensions = json.loads(json.dumps(extensions, ensure_ascii=False)) if present else None
        self.__present = present

    def metadata(self) -> dict:
        return json.loads(json.dumps(self.__metadata, ensure_ascii=False))

    def extensions_present(self) -> bool:
        return self.__present

    def private_extensions(self) -> dict | None:
        """Return a detached copy; callers must treat it as sensitive input."""
        return json.loads(json.dumps(self.__extensions, ensure_ascii=False)) if self.__present else None

    def __repr__(self) -> str:
        return "PrivateTeacherExtensions(<private values withheld>)"

    __str__ = __repr__

    def __reduce_ex__(self, _protocol: int) -> Any:
        raise TypeError("private teacher extension serialization is disabled")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identity_token(field: str, label: str) -> str:
    """Collision-free serialization used only as a JSON object key."""
    return json.dumps([field, label], ensure_ascii=True, separators=(",", ":"))


def _teacher_wire_identity(config: dict, module_sha256: str | None = None) -> dict:
    if module_sha256 is None:
        module_path = Path(explicit_labels.__file__).resolve(strict=True)
        module_sha256 = _sha(module_path.read_bytes())
    if not isinstance(module_sha256, str) or not HEX.fullmatch(module_sha256):
        raise ProposalError("explicit-label parser module identity is invalid")
    if config.get("teacher_response_mode") != explicit_labels.SCHEMA_VERSION:
        return {"mode": "legacy_positive", "schema_sha256": None,
                "prompt_sha256": None, "module_sha256": module_sha256}
    selected = config["selected_columns"]
    schema = explicit_labels.build_response_schema(selected, config["taxonomy_sha256"])
    prompt_sha = _sha(config["teacher_prompt_contract"].encode("utf-8"))
    schema_sha = _sha(_canonical_json(schema).rstrip(b"\n"))
    return {"mode": explicit_labels.SCHEMA_VERSION, "schema_sha256": schema_sha,
            "prompt_sha256": prompt_sha, "module_sha256": module_sha256}


def _proposal_schema_digest(config: dict) -> str:
    identity = _teacher_wire_identity(config)
    if identity["schema_sha256"] is not None:
        return identity["schema_sha256"]
    labels = [f"{c['field']}:{c['label']}" for c in config["selected_columns"]]
    return _sha(_canonical_json(comparator.output_schema(labels)).rstrip(b"\n"))


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
    requested_capacity = cfg.get("max_progress_store_bytes")
    if requested_capacity is None:
        requested_capacity = min(PROGRESS_STORE_HARD_MAX_BYTES,
                                  max(PROGRESS_STORE_DEFAULT_BYTES, 4 * cfg["max_protocol_bytes"] +
                                      PROGRESS_TERMINAL_MAX_BYTES))
    if type(requested_capacity) is not int or not 1 <= requested_capacity <= PROGRESS_STORE_HARD_MAX_BYTES:
        raise ProposalError("max_progress_store_bytes must be within the supported progress-store capacity")
    cfg["max_progress_store_bytes"] = requested_capacity
    if type(cfg["assume_exhaustive_labels"]) is not bool:
        raise ProposalError("assume_exhaustive_labels must be boolean")
    mode = cfg.get("teacher_response_mode", "legacy_positive")
    if not isinstance(mode, str) or mode not in {"legacy_positive", explicit_labels.SCHEMA_VERSION}:
        raise ProposalError("unsupported teacher response mode")
    cfg["teacher_response_mode"] = mode
    extension_limit = cfg.get("max_teacher_extension_bytes", explicit_labels.MAX_EXTENSION_BYTES)
    if (type(extension_limit) is not int or extension_limit < 0
            or extension_limit > explicit_labels.MAX_EXTENSION_BYTES):
        raise ProposalError("max_teacher_extension_bytes exceeds the supported parser bound")
    cfg["max_teacher_extension_bytes"] = extension_limit
    if mode == explicit_labels.SCHEMA_VERSION:
        prompt_contract = cfg.get("teacher_prompt_contract")
        try:
            prompt_size = len(prompt_contract.encode("utf-8")) if isinstance(prompt_contract, str) else 0
            explicit_labels.build_response_schema(cols, cfg["taxonomy_sha256"])
        except (UnicodeError, explicit_labels.ContractError):
            raise ProposalError("explicit tri-state taxonomy or prompt contract is invalid") from None
        if (cfg["assume_exhaustive_labels"] is not False
                or not isinstance(prompt_contract, str) or not prompt_contract.strip()
                or prompt_size > 16_384
                or "{identities}" not in prompt_contract or "{text}" not in prompt_contract):
            raise ProposalError("explicit tri-state mode requires non-exhaustive labels and a bounded prompt contract with identity and text slots")
    else:
        # The legacy positive-only wire format must not silently alias
        # field/label pairs joined by a colon.
        joined = [f"{c['field']}:{c['label']}" for c in cols]
        if len(joined) != len(set(joined)):
            raise ProposalError("legacy label identities are ambiguous")
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
    created = False
    try:
        # Create exclusively, then apply the same owner-only ACL/mode used by
        # masked export inputs before the directory can contain private bytes.
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
        created = True
        masked = _masked()
        masked._secure_private_path(path, is_directory=True)
        masked._verify_owner_private_access(path, is_directory=True)
        return path
    except (OSError, ValueError, subprocess.SubprocessError):
        # Remove only the empty directory this call created. Never recurse from
        # a failed ACL setup where another process could have added entries.
        if created and path.is_dir() and not path.is_symlink():
            try:
                path.rmdir()
            except OSError:
                pass
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
                         max_protocol_bytes: int, teacher_identity: dict | None = None,
                         max_extension_bytes: int = explicit_labels.MAX_EXTENSION_BYTES,
                         model_name: str | None = None) -> dict[str, int]:
    labels = [f"{c['field']}:{c['label']}" for c in columns]
    explicit_mode = (teacher_identity or {}).get("mode") == explicit_labels.SCHEMA_VERSION
    if type(max_extension_bytes) is not int or not 0 <= max_extension_bytes <= explicit_labels.MAX_EXTENSION_BYTES:
        raise ProposalError("configured teacher extension bound is invalid")
    if explicit_mode and (not isinstance(model_name, str) or not model_name):
        raise ProposalError("explicit teacher result sizing requires the pinned model identity")
    state_keys = [_identity_token(c["field"], c["label"]) for c in columns] if explicit_mode else labels
    output_bound = 0
    protocol_bound = 0
    max_record_bytes = 0
    max_ack_bytes = 0
    statuses = {"answered", "abstain", "malformed", "invalid_schema", "refusal", "truncated",
                "incomplete", "identity_mismatch", "http_error", "timeout", "transport_error",
                "protocol_error", "request_too_large_not_started", "not_attempted_after_abort"}
    unknown_elapsed = {"timeout", "transport_error", "protocol_error",
                       "request_too_large_not_started", "not_attempted_after_abort"}
    for index, row in enumerate(rows):
        common = {"row_id": row["id"], "family_sha256": row["family"],
                "text_sha256": row["text_sha256"], "original_text_sha256": row["original_text_sha256"],
                "screening": row["screening"], "verified_source_claim": row["verified_source_claim"],
                "source_identity": row["source_identity"], "columns": columns,
                "source_targets": row["targets"], "source_known_mask": row["known_mask"],
                "source_target_origins": row["target_origins"], "source_reviewer_claims": row["target_reviewer_claims"],
                # Negative is the longest legal state encoding; using it keeps
                # this byte bound conservative for explicit and exhaustive modes.
                "proposal_states": {label: "proposed_negative" for label in state_keys},
                "conflict_labels": ([{"field": c["field"], "label": c["label"]} for c in columns]
                                    if explicit_mode else labels),
                "provenance": "unreviewed_local_teacher_claim"}
        if explicit_mode:
            common["teacher_identity"] = teacher_identity
        extension_present = explicit_mode and max_extension_bytes >= 2
        extension_value = {}
        if extension_present and max_extension_bytes == 6:
            extension_value = {"": 0}
        elif extension_present and max_extension_bytes >= 7:
            # A canonical extension object may use every byte of the budget.
            # ASCII string bytes are a conservative wire-size witness because
            # JSON escaping controls and Unicode never expands beyond their
            # ensure_ascii canonical representation in the surrounding event.
            extension_value = {"": "x" * (max_extension_bytes - 7)}
        extension_bytes = explicit_labels._canonical_bytes(extension_value) if explicit_mode else b""
        extension_digest = _sha(extension_bytes) if extension_present else None
        max_row_output = 0
        max_event_protocol = 0
        for status in statuses:
            elapsed = None if status in unknown_elapsed else sys.float_info.max
            states = (common["proposal_states"] if status == "answered"
                      else {label: "unknown" for label in state_keys})
            conflicts = common["conflict_labels"] if status == "answered" else []
            item = {**common, "status": status, "elapsed_ms": elapsed,
                    "proposal_states": states, "conflict_labels": conflicts}
            encoded = (json.dumps(item, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            event = {"phase": "progress", "index": index, "status": status,
                     "elapsed_ms": elapsed, "result": item}
            if explicit_mode:
                answered = status in {"answered", "abstain"} and extension_present
                event.update({"private_extensions_present": answered,
                    "private_extensions_sha256": extension_digest if answered else None,
                    "private_extensions": extension_value if answered else {}})
            line = (json.dumps(event, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            if len(line) > PROGRESS_EVENT_MAX_BYTES:
                raise ProposalError("one worker result line exceeds protocol limit")
            ack_bytes = len(_canonical_json(_progress_ack("f" * 32, index, event)))
            if ack_bytes > PROGRESS_ACK_MAX_BYTES:
                raise ProposalError("serialized progress acknowledgment exceeds protocol limit")
            stored_event = {k: v for k, v in event.items() if k != "private_extensions"}
            stored_event["index"] = index
            record_base = {"schema_version": PROGRESS_EVENT_SCHEMA,
                           "attempt_id": "f" * 32, "index": index,
                           "event": stored_event}
            if explicit_mode:
                answered = status in {"answered", "abstain"} and extension_present
                private_base = {"schema_version": PRIVATE_EXTENSION_SCHEMA,
                    "attempt_id": "f" * 32, "worker_index": index, "original_index": index,
                    "source_row_id": row["id"],
                    "source_identity_sha256": _sha(_canonical_json(row["source_identity"]).rstrip(b"\n")),
                    "source_text_sha256": row["text_sha256"],
                    "checkpoint_binding_sha256": "f" * 64, "teacher_identity": teacher_identity,
                    "model_name": model_name, "model_manifest_sha256": "f" * 64,
                    "taxonomy_sha256": "f" * 64,
                    "extensions_present": answered,
                    "extensions_sha256": extension_digest if answered else None,
                    "extensions": extension_value if answered else None,
                    "worker_event_sha256": "f" * 64}
                record_base["private_extension_record"] = {**private_base, "record_sha256": "f" * 64}
            record = {**record_base,
                      "record_sha256": _sha(_canonical_json(record_base).rstrip(b"\n"))}
            record_raw = _canonical_json(record)
            if len(record_raw) > PROGRESS_EVENT_MAX_BYTES:
                raise ProposalError("serialized checkpoint event exceeds stored-record limit")
            max_row_output = max(max_row_output, len(encoded))
            max_event_protocol = max(max_event_protocol, len(line))
            max_record_bytes = max(max_record_bytes, len(record_raw))
            max_ack_bytes = max(max_ack_bytes, ack_bytes)
        output_bound += max_row_output
        protocol_bound += max_event_protocol
    if output_bound > max_output_bytes or protocol_bound > max_protocol_bytes:
        raise ProposalError("selected train rows exceed configured proposal or worker-protocol bounds")
    # The persisted envelope adds fixed JSON framing per row, a per-attempt
    # record, and reserved room for the terminal publication record.
    progress_store_bound = (sum(max_record_bytes for _ in rows) + 512 * len(rows)
                            + PROGRESS_TERMINAL_MAX_BYTES + 4096)
    return {"max_output_bytes": output_bound, "max_protocol_bytes": protocol_bound,
            "max_progress_record_bytes": max_record_bytes if rows else 0,
            "max_ack_bytes": max_ack_bytes if rows else 0,
            "progress_store_bytes": progress_store_bound}


def _render_teacher_prompt(template: str, identities: str, text: str) -> str:
    """Substitute only placeholders present in the configured template."""
    replacements = {"identities": identities, "text": text}
    return re.sub(r"\{(identities|text)\}", lambda match: replacements[match.group(1)], template)


def _request(config: dict, labels: list[str], text: str, max_bytes: int) -> bytes:
    if config.get("teacher_response_mode") == explicit_labels.SCHEMA_VERSION:
        selected = config["selected_columns"]
        schema = explicit_labels.build_response_schema(selected, config["taxonomy_sha256"])
        identities = _canonical_json(selected).decode("utf-8").strip()
        prompt = _render_teacher_prompt(config["teacher_prompt_contract"], identities, text)
        body = comparator._explicit_schema_request(config, schema, prompt)
        if len(body) > max_bytes:
            raise ProposalError("bounded model request exceeds configured limit")
        return body
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
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL |
             getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    fd = os.open(path, flags, 0o600)
    try:
        # New files may inherit broad Windows ACLs from their parent. Replace
        # those grants and validate the resulting path before writing bytes.
        _masked()._secure_private_path(path, is_directory=False)
        _masked()._verify_owner_private_access(path, is_directory=False)
        with os.fdopen(fd, "wb") as out:
            fd = -1
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
    except Exception:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
            fd = -1
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        if fd >= 0:
            os.close(fd)


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                        separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _private_extension_sha256(value: dict) -> str:
    return _sha(explicit_labels._canonical_bytes(value))


def _sync_directory(path: Path) -> bool:
    """Flush directory metadata where Python exposes a supported operation."""
    if os.name == "nt":
        # Windows file fsync uses the CRT commit operation. Python does not
        # provide a portable directory FlushFileBuffers handle here.
        return False
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def _progress_store_path(final_path: Path, allow_final: bool = False) -> Path:
    if final_path.is_symlink() or (final_path.exists() and not allow_final):
        raise ProposalError("proposal output destination must be new for a fresh run and not a symlink")
    final_path = final_path.resolve()
    return final_path.parent / f".{final_path.name}.teacher-progress"


@contextlib.contextmanager
def _output_lease(final_path: Path):
    """Kernel-released, cross-process exclusive lease; keep the lock inode forever."""
    parent = final_path.parent.resolve(strict=True)
    if not _private_dir(parent):
        raise ProposalError("progress lease parent is not owner-private")
    lock_path = parent / f".{final_path.name}.teacher-progress.lock"
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    created = False
    try:
        fd = os.open(lock_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        if lock_path.is_symlink():
            raise ProposalError("progress lease is unsafe") from None
        fd = os.open(lock_path, flags)
    acquired = False
    try:
        info = os.fstat(fd)
        path_info = lock_path.stat(follow_symlinks=False)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino)):
            raise ProposalError("progress lease is unsafe")
        if created:
            # A fresh Windows file inherits ACEs. Secure it before writing even
            # the sentinel byte; never rewrite an existing lease's ACL.
            _masked()._secure_private_path(lock_path, is_directory=False)
        if not _private_file(lock_path):
            raise ProposalError("progress lease permissions are not owner-private")
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except (OSError, BlockingIOError) as exc:
            raise ProposalError("another teacher run holds the progress lease") from exc
        # Read or initialize the byte only after acquiring the lock; on Windows
        # another holder's byte-range lock can prohibit even a sentinel read.
        if os.fstat(fd).st_size == 0:
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, b"\0")
            os.fsync(fd)
        elif os.fstat(fd).st_size != 1:
            raise ProposalError("progress lease has invalid contents")
        else:
            os.lseek(fd, 0, os.SEEK_SET)
            sentinel = os.read(fd, 1)
            os.lseek(fd, 0, os.SEEK_SET)
            if sentinel != b"\0":
                raise ProposalError("progress lease has invalid contents")
        yield
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def _proposal_config_identity_sha(cfg: dict) -> str:
    # Storage capacity is a mutable operational ceiling; it does not alter the
    # source, teacher, prompt, or inference semantics bound to proposal rows.
    identity = {key: value for key, value in cfg.items() if key != "max_progress_store_bytes"}
    return _sha(_canonical_json(identity).rstrip(b"\n"))


def _stable_runtime_identity(boundary: Any) -> dict:
    if not isinstance(boundary, dict):
        raise ProposalError("runtime boundary evidence is invalid")
    stable = {key: boundary.get(key) for key in
              ("boundary", "server_binary_sha256", "server_version")}
    if (stable["boundary"] != "same_uid_private_network_namespace_without_nonloopback_routes"
            or not isinstance(stable["server_binary_sha256"], str)
            or not HEX.fullmatch(stable["server_binary_sha256"])
            or not isinstance(stable["server_version"], str) or not stable["server_version"]):
        raise ProposalError("runtime boundary stable identity is invalid")
    return stable


def _validate_runtime_observation(boundary: Any) -> dict:
    stable = _stable_runtime_identity(boundary)
    if (type(boundary.get("server_pid")) is not int or boundary["server_pid"] <= 0
            or type(boundary.get("network_namespace_inode")) is not int
            or boundary["network_namespace_inode"] <= 0
            or type(boundary.get("process_tree_rss_bytes")) is not int
            or boundary["process_tree_rss_bytes"] < 0
            or boundary.get("process_tree_rss_scope") !=
                "server and observed descendants; summed RSS can double-count shared pages"):
        raise ProposalError("runtime boundary observation is incomplete or invalid")
    return stable


def _progress_binding(cfg: dict, cfg_sha: str, comp: dict, comp_sha: str,
                      rows: list[dict], source_binding: dict, adapter_sha: str,
                      comparator_sha: str, ready: dict) -> dict:
    selected_ids = [row["id"] for row in rows]
    if len(selected_ids) != len(set(selected_ids)):
        raise ProposalError("selected train rows contain duplicate identities")
    model_identity = ready.get("model_identity", {})
    runtime = ready.get("runtime_boundary", {})
    teacher_identity = _teacher_wire_identity(cfg)
    return {
        "schema_version": PROGRESS_STORE_SCHEMA,
        "proposal_config_identity_sha256": _proposal_config_identity_sha(cfg),
        "comparison_config_sha256": comp_sha,
        "export_manifest_sha256": source_binding["manifest_sha256"],
        "export_config_sha256": source_binding["config_sha256"],
        "train_sha256": source_binding["train_sha256"],
        "taxonomy_sha256": cfg["taxonomy_sha256"],
        "selected_columns_sha256": _sha(_canonical_json(cfg["selected_columns"])),
        "selected_row_ids": selected_ids,
        "selected_row_order_sha256": _sha(_canonical_json(selected_ids)),
        "selected_source_identities_sha256": _sha(_canonical_json([
            {"row_id": row["id"], "source_identity": row["source_identity"],
             "text_sha256": row["text_sha256"], "original_text_sha256": row["original_text_sha256"]}
            for row in rows])),
        "model_name": comp["model_name"],
        "model_manifest_sha256": model_identity.get("reported_manifest_sha256"),
        "server_binary_sha256": comp["server_binary_sha256"],
        "server_version": comp["server_version"],
        "prompt_contract_sha256": _sha((comp["system_prompt"] + "\0" + comp["prompt_revision"]).encode()),
        "teacher_response_mode": cfg.get("teacher_response_mode", "legacy_positive"),
        "max_teacher_extension_bytes": cfg.get("max_teacher_extension_bytes", explicit_labels.MAX_EXTENSION_BYTES),
        "teacher_schema_sha256": teacher_identity["schema_sha256"],
        "teacher_prompt_sha256": teacher_identity["prompt_sha256"],
        "explicit_labels_module_sha256": teacher_identity["module_sha256"],
        "adapter_code_sha256": adapter_sha,
        "comparator_code_sha256": comparator_sha,
        "runtime_boundary": runtime.get("boundary"),
        "request_policy_sha256": _sha(_canonical_json({k: comp[k] for k in (
            "timeout_seconds", "max_request_bytes", "max_response_bytes", "num_ctx", "num_predict",
            "temperature", "top_k", "top_p", "repeat_penalty")})),
        "resource_policy_sha256": _sha(_canonical_json({k: comp[k] for k in (
            "resource_memory_max_bytes", "resource_cpu_quota_percent", "remote_scope_tasks_max",
            "min_memory_available_bytes", "min_disk_free_bytes", "min_gpu_free_bytes")})),
    }


def _progress_store_usage(store: Path) -> int:
    total = 0
    for path in store.iterdir():
        if path.is_symlink() or not path.is_file() or not _private_file(path):
            raise ProposalError("progress store contains unsafe or unverified storage")
        total += path.stat(follow_symlinks=False).st_size
        if total > PROGRESS_STORE_HARD_MAX_BYTES:
            raise ProposalError("progress store exceeds its hard safety limit")
    return total


def _progress_capacity_records(store: Path, requested: int, create: bool = False,
                               update_increase: bool = True,
                               config_sha256: str | None = None,
                               original_config_sha256: str | None = None) -> int:
    path = store / "capacity.jsonl"
    if create:
        history = []
    else:
        if path.is_symlink() or not path.is_file() or not _private_file(path):
            raise ProposalError("progress capacity history is unavailable or unsafe")
        raw = _read(path, 65_536, "progress capacity history")
        if not raw.endswith(b"\n"):
            raise ProposalError("progress capacity history is truncated")
        history = []
        for line in raw.splitlines():
            record = _json(line, "progress capacity record")
            required = {"schema_version", "sequence", "capacity_bytes", "record_sha256"}
            if (not isinstance(record, dict)
                    or set(record) not in (required, required | {"config_sha256"})):
                raise ProposalError("progress capacity record schema is invalid")
            base = {key: value for key, value in record.items() if key != "record_sha256"}
            if (record["schema_version"] != PROGRESS_CAPACITY_SCHEMA
                    or type(record["sequence"]) is not int
                    or type(record["capacity_bytes"]) is not int
                    or not 1 <= record["capacity_bytes"] <= PROGRESS_STORE_HARD_MAX_BYTES
                    or record["record_sha256"] != _sha(_canonical_json(base).rstrip(b"\n"))
                    or record["sequence"] != len(history)
                    or (history and record["capacity_bytes"] <= history[-1]["capacity_bytes"])):
                raise ProposalError("progress capacity history integrity or monotonicity failed")
            recorded_config = record.get("config_sha256")
            if recorded_config is not None and (not isinstance(recorded_config, str)
                    or not HEX.fullmatch(recorded_config)):
                raise ProposalError("progress capacity configuration provenance is invalid")
            history.append({"capacity_bytes": record["capacity_bytes"],
                            "config_sha256": recorded_config})
        if not history:
            raise ProposalError("progress capacity history is empty")
        if (original_config_sha256 is not None
                and history[0]["config_sha256"] not in (None, original_config_sha256)):
            raise ProposalError("progress capacity history does not preserve original config provenance")
    current_record = history[-1] if history else {"capacity_bytes": 0, "config_sha256": None}
    current = current_record["capacity_bytes"]
    if not create and requested < current:
        raise ProposalError("progress store capacity cannot be decreased")
    if (not create and requested == current and config_sha256 is not None
            and current_record["config_sha256"] is not None
            and current_record["config_sha256"] != config_sha256):
        raise ProposalError("proposal config changed without a capacity increase")
    if requested > current and update_increase:
        base = {"schema_version": PROGRESS_CAPACITY_SCHEMA, "sequence": len(history),
                "capacity_bytes": requested, "config_sha256": config_sha256}
        record = {**base, "record_sha256": _sha(_canonical_json(base).rstrip(b"\n"))}
        line = _canonical_json(record)
        if create:
            _write_new(path, line)
        else:
            existing_usage = _progress_store_usage(store)
            if existing_usage > current:
                raise ProposalError("stored progress already exceeds its recorded capacity")
            if existing_usage + len(line) > requested:
                raise ProposalError("requested capacity cannot cover its durable history record")
            flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
            fd = os.open(path, flags)
            try:
                view = memoryview(line)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short capacity-history write")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            _sync_directory(store)
    capacity = requested if requested > current and update_increase else current
    if _progress_store_usage(store) > capacity:
        raise ProposalError("progress store exceeds configured capacity")
    return capacity


def _published_report_package(path: Path) -> tuple[dict, bytes, str]:
    if (path.is_symlink() or not path.is_dir()
            or not comparator._verify_private_report_directory(path)
            or not _outside_git(path)):
        raise ProposalError("prior report revision is unsafe")
    names = {item.name for item in path.iterdir()}
    if not {"manifest.json", "proposals.jsonl"}.issubset(names):
        raise ProposalError("prior report revision lacks its published files")
    manifest_path, proposals_path = path / "manifest.json", path / "proposals.jsonl"
    if (manifest_path.is_symlink() or proposals_path.is_symlink()
            or not _private_file(manifest_path) or not _private_file(proposals_path)):
        raise ProposalError("prior report revision files are unsafe")
    raw = _read(manifest_path, PROGRESS_TERMINAL_MAX_BYTES, "prior report manifest")
    manifest = _json(raw, "prior report manifest")
    proposal_raw = _read(proposals_path, PROGRESS_STORE_HARD_MAX_BYTES, "prior proposal rows")
    if (not isinstance(manifest, dict) or manifest.get("status") not in {"complete", "partial_aborted_ambiguous"}
            or manifest.get("promotion_eligible") is not False
            or manifest.get("proposals_sha256") != _sha(proposal_raw)
            or manifest.get("proposals_bytes") != len(proposal_raw)
            or manifest.get("rows_written") != len(proposal_raw.splitlines())):
        raise ProposalError("prior report revision contents do not match its manifest")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ProposalError("prior report revision provenance is invalid")
    event_sha = provenance.get("progress_store_events_sha256")
    expected_publication_id = _sha(_canonical_json({"proposal_sha256": _sha(proposal_raw),
        "progress_events_sha256": event_sha, "status": manifest["status"]}).rstrip(b"\n"))
    if manifest.get("publication_id") != expected_publication_id:
        raise ProposalError("prior report revision publication identity is invalid")
    allowed = {"manifest.json", "proposals.jsonl", "revision-reservation.json"}
    if "revision-reservation.json" in names:
        reservation_path = path / "revision-reservation.json"
        if reservation_path.is_symlink() or not _private_file(reservation_path):
            raise ProposalError("prior report revision reservation is unsafe")
        reservation = _json(_read(reservation_path, 65_536, "report revision reservation"),
                            "report revision reservation")
        revision = manifest.get("provenance", {}).get("publication_revision")
        reference = {key: reservation.get(key) for key in ("revision_id", "sequence",
            "parent_output_name", "parent_manifest_sha256", "parent_proposals_sha256")}
        if (not isinstance(reservation, dict)
                or reservation.get("schema_version") != PROGRESS_REVISION_SCHEMA
                or not isinstance(revision, dict) or revision != reference
                or reservation.get("record_sha256") != _sha(_canonical_json({k: v for k, v in
                    reservation.items() if k != "record_sha256"}).rstrip(b"\n"))):
            raise ProposalError("published report revision reservation does not match its manifest")
    evidence_re = re.compile(r"^\.(?:manifest-incomplete|manifest-complete|proposals)-"
                             + re.escape(expected_publication_id) + r"-[0-9a-f]{32}\.tmp$")
    if any(name not in allowed and not evidence_re.fullmatch(name) for name in names):
        raise ProposalError("prior report revision contains unknown entries")
    for name in names - allowed:
        item = path / name
        if item.is_symlink() or not item.is_file() or not _private_file(item):
            raise ProposalError("prior report revision staging evidence is unsafe")
    return manifest, raw, _sha(proposal_raw)


def _published_matches_expected(path: Path, proposals: list[dict], manifest: dict) -> bool:
    if not path.exists():
        return False
    saved, _raw, _proposal_sha = _published_report_package(path)
    payload = ("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                           for row in proposals) + "\n").encode("utf-8")
    event_sha = manifest.get("provenance", {}).get("progress_store_events_sha256")
    publication_id = _sha(_canonical_json({"proposal_sha256": _sha(payload),
        "progress_events_sha256": event_sha, "status": manifest["status"]}).rstrip(b"\n"))
    expected = {**manifest, "publication_id": publication_id, "proposals_sha256": _sha(payload),
        "proposals_bytes": len(payload), "rows_written": len(proposals)}
    return saved == expected


def _progress_attempt_indices(store: Path, attempt_id: str, binding: dict,
                             selected_ids: list[str]) -> list[int]:
    path = store / f"attempt-{attempt_id}.json"
    if path.is_symlink() or not path.is_file() or not _private_file(path):
        raise ProposalError("private progress attempt evidence is unavailable")
    record = _json(_read(path, 1_048_576, "progress attempt record"), "progress attempt record")
    keys = {"schema_version", "attempt_id", "binding_sha256", "indices",
        "row_ids_sha256", "state", "record_sha256"}
    base = ({key: record[key] for key in keys - {"record_sha256"}}
            if isinstance(record, dict) else {})
    indices = record.get("indices") if isinstance(record, dict) else None
    if (not isinstance(record, dict) or set(record) != keys
            or record.get("schema_version") != PROGRESS_ATTEMPT_SCHEMA
            or record.get("attempt_id") != attempt_id
            or record.get("binding_sha256") != _sha(_canonical_json(binding).rstrip(b"\n"))
            or not isinstance(indices, list) or not indices
            or any(type(i) is not int or not 0 <= i < len(selected_ids) for i in indices)
            or len(indices) != len(set(indices)) or indices != sorted(indices)
            or record.get("row_ids_sha256") != _sha(_canonical_json([selected_ids[i] for i in indices]).rstrip(b"\n"))
            or record.get("state") != "prepared_or_sent_outcome_unknown"
            or record.get("record_sha256") != _sha(_canonical_json(base).rstrip(b"\n"))):
        raise ProposalError("private progress attempt binding or integrity failed")
    return indices


def _progress_event_snapshot(store: Path, binding: dict, events_sha256: str) -> tuple[bytes, dict, set[str]]:
    path = store / "events.jsonl"
    if path.is_symlink() or not _private_file(path):
        raise ProposalError("historical progress event snapshot is unavailable or unsafe")
    current = _read(path, PROGRESS_STORE_HARD_MAX_BYTES, "historical progress event log")
    candidates = [b""] if _sha(b"") == events_sha256 else []
    digest = hashlib.sha256()
    prefix = bytearray()
    if current and not current.endswith(b"\n"):
        raise ProposalError("historical progress event log is truncated")
    for line in current.splitlines(keepends=True):
        prefix.extend(line)
        digest.update(line)
        if digest.hexdigest() == events_sha256:
            candidates.append(bytes(prefix))
    if len(candidates) != 1:
        raise ProposalError("historical report event digest is not a unique durable prefix")
    snapshot = candidates[0]
    latest: dict[int, dict] = {}
    attempts: set[str] = set()
    explicit_mode = binding.get("teacher_response_mode") == explicit_labels.SCHEMA_VERSION
    teacher_identity = {"mode": binding.get("teacher_response_mode"),
        "schema_sha256": binding.get("teacher_schema_sha256"),
        "prompt_sha256": binding.get("teacher_prompt_sha256"),
        "module_sha256": binding.get("explicit_labels_module_sha256")}
    if explicit_mode and any(not isinstance(teacher_identity[key], str)
                             or not HEX.fullmatch(teacher_identity[key])
                             for key in ("schema_sha256", "prompt_sha256", "module_sha256")):
        raise ProposalError("historical explicit teacher identity is invalid")
    selected_count = len(binding.get("selected_row_ids", []))
    selected_ids = binding.get("selected_row_ids", [])
    attempt_orders: dict[str, list[int]] = {}
    seen: set[tuple[str, int]] = set()
    for line in snapshot.splitlines():
        record = _json(line, "historical progress event")
        expected_record_keys = {"schema_version", "attempt_id", "index", "event", "record_sha256"}
        if explicit_mode:
            expected_record_keys.add("private_extension_record")
        if (not isinstance(record, dict) or set(record) != expected_record_keys):
            raise ProposalError("historical progress event schema is invalid")
        base = {key: record[key] for key in expected_record_keys - {"record_sha256"}}
        if (record["schema_version"] != PROGRESS_EVENT_SCHEMA
                or not isinstance(record["attempt_id"], str)
                or not re.fullmatch(r"[0-9a-f]{32}", record["attempt_id"])
                or type(record["index"]) is not int
                or not 0 <= record["index"] < selected_count
                or not isinstance(record["event"], dict)
                or record["record_sha256"] != _sha(_canonical_json(base).rstrip(b"\n"))):
            raise ProposalError("historical progress event integrity is invalid")
        attempt_id = record["attempt_id"]
        key = (attempt_id, record["index"])
        if key in seen:
            raise ProposalError("historical progress contains duplicate attempt rows")
        seen.add(key)
        if attempt_id not in attempt_orders:
            attempt_orders[attempt_id] = _progress_attempt_indices(store, attempt_id, binding, selected_ids)
        if record["index"] not in attempt_orders[attempt_id]:
            raise ProposalError("historical event index is not a member of its attempt")
        attempts.add(record["attempt_id"])
        event = record["event"]
        if explicit_mode:
            item = event.get("result") if isinstance(event, dict) else None
            if not isinstance(item, dict):
                raise ProposalError("historical explicit progress event has no proposal result")
            row = {"id": item.get("row_id"), "source_identity": item.get("source_identity"),
                   "text_sha256": item.get("text_sha256")}
            restored = _restore_private_extension_event(event, record["private_extension_record"],
                record["attempt_id"], record["index"], row, binding, teacher_identity,
                binding.get("max_teacher_extension_bytes", explicit_labels.MAX_EXTENSION_BYTES))
            worker_index = record["private_extension_record"].get("worker_index")
            if worker_index != attempt_orders[attempt_id].index(record["index"]):
                raise ProposalError("historical event worker ordinal does not map to its attempt row")
            event = {**restored, "index": record["index"]}
        latest[record["index"]] = event.get("result")
    if set(latest) != set(range(selected_count)) or any(not isinstance(value, dict) for value in latest.values()):
        raise ProposalError("published parent does not have a complete historical row snapshot")
    return snapshot, latest, attempts


def _progress_publication_receipts_validate(store: Path, binding: dict) -> None:
    binding_sha = _sha(_canonical_json(binding).rstrip(b"\n"))
    for path in store.iterdir():
        if not path.name.startswith("publication-"):
            continue
        if (path.is_symlink() or not path.is_file() or not _private_file(path)
                or not re.fullmatch(r"publication-[0-9a-f]{64}\.json", path.name)):
            raise ProposalError("private progress publication receipt is unsafe")
        record = _json(_read(path, 65_536, "progress publication receipt"),
                       "progress publication receipt")
        required = {"schema_version", "publication_id", "binding_sha256", "events_sha256",
                    "report_manifest_sha256", "proposal_sha256", "proposal_bytes", "rows_written",
                    "status", "record_sha256"}
        base = {key: value for key, value in record.items() if key != "record_sha256"} if isinstance(record, dict) else {}
        if (not isinstance(record, dict) or set(record) != required
                or record["schema_version"] != PROGRESS_PUBLICATION_SCHEMA
                or record["publication_id"] != path.stem.removeprefix("publication-")
                or record["binding_sha256"] != binding_sha
                or not isinstance(record["events_sha256"], str) or not HEX.fullmatch(record["events_sha256"])
                or not isinstance(record["report_manifest_sha256"], str) or not HEX.fullmatch(record["report_manifest_sha256"])
                or not isinstance(record["proposal_sha256"], str) or not HEX.fullmatch(record["proposal_sha256"])
                or type(record["proposal_bytes"]) is not int or record["proposal_bytes"] < 0
                or type(record["rows_written"]) is not int or record["rows_written"] < 1
                or record["status"] != "partial_aborted_ambiguous"
                or record["record_sha256"] != _sha(_canonical_json(base).rstrip(b"\n"))):
            raise ProposalError("progress publication receipt identity or integrity is invalid")
        _snapshot, latest, _attempts = _progress_event_snapshot(store, binding, record["events_sha256"])
        expected_payload = ("\n".join(json.dumps(latest[index], sort_keys=True,
            ensure_ascii=False, separators=(",", ":")) for index in range(len(binding["selected_row_ids"])))
            + "\n").encode("utf-8")
        expected_publication_id = _sha(_canonical_json({"proposal_sha256": _sha(expected_payload),
            "progress_events_sha256": record["events_sha256"], "status": record["status"]}).rstrip(b"\n"))
        if (record["proposal_sha256"] != _sha(expected_payload)
                or record["proposal_bytes"] != len(expected_payload)
                or record["rows_written"] != len(latest)
                or record["publication_id"] != expected_publication_id):
            raise ProposalError("progress publication receipt does not match its durable event snapshot")


def _progress_store_write_publication_receipt(store: Path, binding: dict, proposals: list[dict],
                                               manifest: dict, capacity: int) -> None:
    payload = ("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                         for row in proposals) + "\n").encode("utf-8")
    provenance = manifest.get("provenance", {})
    events_sha256 = provenance.get("progress_store_events_sha256")
    if (manifest.get("status") != "partial_aborted_ambiguous"
            or provenance.get("terminal_worker_result_verified") is not False
            or provenance.get("progress_store_id_sha256") != _sha(_canonical_json(binding).rstrip(b"\n"))
            or not isinstance(events_sha256, str) or not HEX.fullmatch(events_sha256)):
        raise ProposalError("unverified report cannot be bound to progress history")
    publication_id = _sha(_canonical_json({"proposal_sha256": _sha(payload),
        "progress_events_sha256": events_sha256, "status": manifest["status"]}).rstrip(b"\n"))
    base = {"schema_version": PROGRESS_PUBLICATION_SCHEMA,
        "publication_id": publication_id,
        "binding_sha256": _sha(_canonical_json(binding).rstrip(b"\n")),
        "events_sha256": events_sha256,
        "report_manifest_sha256": _sha(_canonical_json(manifest).rstrip(b"\n")),
        "proposal_sha256": _sha(payload), "proposal_bytes": len(payload),
        "rows_written": len(proposals), "status": manifest["status"]}
    record = {**base, "record_sha256": _sha(_canonical_json(base).rstrip(b"\n"))}
    raw = _canonical_json(record)
    path = store / f"publication-{publication_id}.json"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not _private_file(path) or _read(path, 65_536,
                "progress publication receipt") != raw:
            raise ProposalError("existing progress publication receipt differs from this report")
        return
    if len(raw) > 65_536 or _progress_store_usage(store) + len(raw) > capacity:
        raise ProposalError("progress store capacity cannot cover the publication receipt")
    _write_new(path, raw)
    _sync_directory(store)


def _verify_historical_parent_report(store: Path, path: Path, manifest: dict,
                                     binding: dict) -> None:
    package_fields = {"publication_id", "proposals_sha256", "proposals_bytes", "rows_written"}
    if not package_fields.issubset(manifest):
        raise ProposalError("historical parent manifest lacks immutable publication fields")
    provenance = manifest.get("provenance")
    if (not isinstance(provenance, dict)
            or provenance.get("progress_store_id_sha256") != _sha(_canonical_json(binding).rstrip(b"\n"))):
        raise ProposalError("historical parent report is not bound to this progress store")
    events_sha256 = provenance.get("progress_store_events_sha256")
    if not isinstance(events_sha256, str) or not HEX.fullmatch(events_sha256):
        raise ProposalError("historical parent report lacks a valid event snapshot")
    _snapshot, latest, attempts = _progress_event_snapshot(store, binding, events_sha256)
    proposals_path = path / "proposals.jsonl"
    proposal_raw = _read(proposals_path, PROGRESS_STORE_HARD_MAX_BYTES, "historical parent proposals")
    expected_proposals = ("\n".join(json.dumps(latest[index], sort_keys=True, ensure_ascii=False,
        separators=(",", ":")) for index in range(len(binding["selected_row_ids"]))) + "\n").encode("utf-8")
    if proposal_raw != expected_proposals:
        raise ProposalError("historical parent proposals do not match its durable event snapshot")
    original_manifest = {key: value for key, value in manifest.items() if key not in package_fields}
    if provenance.get("terminal_worker_result_verified") is True:
        terminal = _progress_store_matching_terminal(store, binding, events_sha256)
        if (terminal is None or terminal["attempt_id"] not in attempts
                or terminal["output_manifest"] != original_manifest):
            raise ProposalError("historical parent manifest does not match its terminal event snapshot")
        return
    if provenance.get("terminal_worker_result_verified") is not False:
        raise ProposalError("historical parent verification state is invalid")
    receipt_path = store / f"publication-{manifest['publication_id']}.json"
    if receipt_path.is_symlink() or not _private_file(receipt_path):
        raise ProposalError("unverified historical parent lacks a durable publication receipt")
    receipt = _json(_read(receipt_path, 65_536, "progress publication receipt"),
                    "progress publication receipt")
    if not isinstance(receipt, dict):
        raise ProposalError("unverified historical parent publication receipt is invalid")
    receipt_base = {key: value for key, value in receipt.items() if key != "record_sha256"}
    expected_publication_id = _sha(_canonical_json({"proposal_sha256": _sha(proposal_raw),
        "progress_events_sha256": events_sha256, "status": manifest["status"]}).rstrip(b"\n"))
    if (receipt.get("events_sha256") != events_sha256
            or receipt.get("schema_version") != PROGRESS_PUBLICATION_SCHEMA
            or receipt.get("publication_id") != manifest.get("publication_id")
            or receipt.get("binding_sha256") != _sha(_canonical_json(binding).rstrip(b"\n"))
            or receipt.get("record_sha256") != _sha(_canonical_json(receipt_base).rstrip(b"\n"))
            or receipt.get("report_manifest_sha256") != _sha(_canonical_json(original_manifest).rstrip(b"\n"))
            or receipt.get("proposal_sha256") != _sha(proposal_raw)
            or receipt.get("proposal_bytes") != len(proposal_raw)
            or receipt.get("rows_written") != len(proposal_raw.splitlines())
            or receipt.get("status") != manifest.get("status")
            or receipt.get("publication_id") != expected_publication_id):
        raise ProposalError("unverified historical parent differs from its durable publication receipt")


def _progress_revision_state(store: Path, base_output: Path,
                             binding: dict) -> dict | None:
    paths = []
    for path in store.iterdir():
        if path.name.startswith("revision-"):
            if (path.is_symlink() or not path.is_file() or not _private_file(path)
                    or not re.fullmatch(r"revision-[0-9]{4}-[0-9a-f]{32}\.json", path.name)):
                raise ProposalError("private report revision record is unsafe")
            paths.append(path)
    records = []
    previous_name = base_output.name
    previous_manifest_sha = previous_proposals_sha = None
    for path in sorted(paths):
        record = _json(_read(path, 65_536, "report revision record"), "report revision record")
        required = {"schema_version", "sequence", "revision_id", "binding_sha256",
                    "parent_output_name", "parent_manifest_sha256", "parent_proposals_sha256",
                    "target_output_name", "record_sha256"}
        if not isinstance(record, dict) or set(record) != required:
            raise ProposalError("report revision record schema is invalid")
        base = {key: value for key, value in record.items() if key != "record_sha256"}
        sequence = len(records)
        if (record["schema_version"] != PROGRESS_REVISION_SCHEMA
                or record["sequence"] != sequence
                or record["revision_id"] != path.stem.removeprefix("revision-").split("-")[-1]
                or record["binding_sha256"] != _sha(_canonical_json(binding).rstrip(b"\n"))
                or record["parent_output_name"] != previous_name
                or record["record_sha256"] != _sha(_canonical_json(base).rstrip(b"\n"))
                or Path(record["target_output_name"]).name != record["target_output_name"]
                or not record["target_output_name"].startswith(base_output.name + ".revision-")):
            raise ProposalError("report revision record binding or chain is invalid")
        parent_path = base_output.parent / record["parent_output_name"]
        parent_manifest, parent_raw, parent_proposals_sha = _published_report_package(parent_path)
        if (record["parent_manifest_sha256"] != _sha(parent_raw)
                or record["parent_proposals_sha256"] != parent_proposals_sha):
            raise ProposalError("report revision parent evidence changed")
        _verify_historical_parent_report(store, parent_path, parent_manifest, binding)
        target = base_output.parent / record["target_output_name"]
        target_published = False
        if target.exists() or target.is_symlink():
            if (target.is_symlink() or not target.is_dir()
                    or not comparator._verify_private_report_directory(target)):
                raise ProposalError("reserved report revision target is unsafe")
            reservation_path = target / "revision-reservation.json"
            if reservation_path.exists() or reservation_path.is_symlink():
                if (reservation_path.is_symlink() or not reservation_path.is_file()
                        or not _private_file(reservation_path)
                        or _read(reservation_path, 65_536, "report revision reservation") !=
                            _canonical_json(record)):
                    raise ProposalError("reserved report revision path does not match its private reservation")
            elif any(target.iterdir()):
                raise ProposalError("unmarked report revision directory contains foreign data; preserve it and inspect the private revision record")
            marker = target / "manifest.json"
            if marker.exists():
                marker_value = _json(_read(marker, PROGRESS_TERMINAL_MAX_BYTES,
                    "reserved report revision manifest"), "reserved report revision manifest")
                if marker_value.get("status") != "incomplete":
                    child_manifest, child_raw, child_proposals_sha = _published_report_package(target)
                    revision = child_manifest.get("provenance", {}).get("publication_revision")
                    if (revision != {key: record[key] for key in ("revision_id", "sequence",
                        "parent_output_name", "parent_manifest_sha256", "parent_proposals_sha256")}):
                        raise ProposalError("published report revision does not link to its reservation")
                    previous_manifest_sha, previous_proposals_sha = _sha(child_raw), child_proposals_sha
                    target_published = True
                else:
                    previous_manifest_sha = previous_proposals_sha = None
            else:
                previous_manifest_sha = previous_proposals_sha = None
        else:
            previous_manifest_sha = previous_proposals_sha = None
        previous_name = record["target_output_name"]
        records.append(record)
    if not records:
        return None
    latest = records[-1]
    return {"record": latest, "target": base_output.parent / latest["target_output_name"],
            "published": target_published,
            "records": records,
            "reference": {key: latest[key] for key in ("revision_id", "sequence",
                "parent_output_name", "parent_manifest_sha256", "parent_proposals_sha256")}}


def _progress_revision_reserve(store: Path, base_output: Path, binding: dict,
                               capacity: int) -> dict | None:
    state = _progress_revision_state(store, base_output, binding)
    if state is None:
        if not base_output.exists():
            return None
        parent_path = base_output
        sequence = 0
    elif not state["published"]:
        _ensure_revision_target(state)
        return state
    else:
        parent_path = state["target"]
        sequence = len(state["records"])
    parent_manifest, parent_raw, parent_proposals_sha = _published_report_package(parent_path)
    _verify_historical_parent_report(store, parent_path, parent_manifest, binding)
    if parent_manifest.get("status") != "partial_aborted_ambiguous":
        raise ProposalError("a completed report cannot be revised from retryable progress")
    revision_id = uuid.uuid4().hex
    target_name = f"{base_output.name}.revision-{sequence:04d}-{revision_id}"
    target = base_output.parent / target_name
    if target.exists() or target.is_symlink():
        raise ProposalError("reserved report revision destination already exists")
    if not _private_dir(base_output.parent):
        raise ProposalError("report revision parent is not owner-private")
    base = {"schema_version": PROGRESS_REVISION_SCHEMA, "sequence": sequence,
        "revision_id": revision_id,
        "binding_sha256": _sha(_canonical_json(binding).rstrip(b"\n")),
        "parent_output_name": parent_path.name,
        "parent_manifest_sha256": _sha(parent_raw),
        "parent_proposals_sha256": parent_proposals_sha,
        "target_output_name": target_name}
    record = {**base, "record_sha256": _sha(_canonical_json(base).rstrip(b"\n"))}
    record_path = store / f"revision-{sequence:04d}-{revision_id}.json"
    raw = _canonical_json(record)
    if _progress_store_usage(store) + len(raw) > capacity:
        raise ProposalError("progress store capacity cannot cover the report revision reservation")
    _write_new(record_path, raw)
    _sync_directory(store)
    state = {"record": record, "target": target, "published": False,
            "records": ([] if state is None else state["records"]) + [record],
            "reference": {key: record[key] for key in ("revision_id", "sequence",
                "parent_output_name", "parent_manifest_sha256", "parent_proposals_sha256")}}
    _ensure_revision_target(state)
    return state


def _ensure_revision_target(state: dict) -> None:
    target, record = state["target"], state["record"]
    if not target.exists() and not target.is_symlink():
        created = comparator._private_report_directory(target)
        if created != target or not comparator._verify_private_report_directory(target):
            raise ProposalError("reserved report revision directory is not owner-private")
        _sync_directory(target.parent)
    elif (target.is_symlink() or not target.is_dir()
            or not comparator._verify_private_report_directory(target)):
        raise ProposalError("reserved report revision target is unsafe")
    reservation_path = target / "revision-reservation.json"
    if reservation_path.exists() or reservation_path.is_symlink():
        if (reservation_path.is_symlink() or not _private_file(reservation_path)
                or _read(reservation_path, 65_536, "report revision reservation") != _canonical_json(record)):
            raise ProposalError("reserved report revision marker differs from its durable record")
    else:
        if any(target.iterdir()):
            raise ProposalError("unreserved data exists at the report revision destination")
        _write_new(reservation_path, _canonical_json(record))
        _sync_directory(target)


def _progress_probe_receipts_validate(store: Path, binding: dict) -> None:
    expected_binding_sha = _sha(_canonical_json(binding).rstrip(b"\n"))
    for path in store.iterdir():
        if not path.name.startswith("recovery-probe-"):
            continue
        if (path.is_symlink() or not path.is_file() or not _private_file(path)
                or not re.fullmatch(r"recovery-probe-[0-9a-f]{32}\.json", path.name)):
            raise ProposalError("private recovery probe receipt is unsafe")
        record = _json(_read(path, PROGRESS_TERMINAL_MAX_BYTES, "recovery probe receipt"),
                       "recovery probe receipt")
        required = {"schema_version", "receipt_id", "binding_sha256",
                    "proposal_config_sha256", "probe", "record_sha256"}
        if not isinstance(record, dict) or set(record) != required:
            raise ProposalError("recovery probe receipt schema is invalid")
        base = {key: value for key, value in record.items() if key != "record_sha256"}
        probe = record["probe"]
        if (record["schema_version"] != PROGRESS_PROBE_RECEIPT_SCHEMA
                or record["receipt_id"] != path.stem.removeprefix("recovery-probe-")
                or record["binding_sha256"] != expected_binding_sha
                or not isinstance(record["proposal_config_sha256"], str)
                or not HEX.fullmatch(record["proposal_config_sha256"])
                or not isinstance(probe, dict) or probe.get("zero_row_probe") is not True
                or probe.get("proposal_config_identity_sha256") != binding["proposal_config_identity_sha256"]
                or probe.get("model_identity", {}).get("reported_manifest_sha256") != binding["model_manifest_sha256"]
                or probe.get("runtime_boundary", {}).get("server_binary_sha256") != binding["server_binary_sha256"]
                or probe.get("runtime_boundary", {}).get("server_version") != binding["server_version"]
                or record["record_sha256"] != _sha(_canonical_json(base).rstrip(b"\n"))):
            raise ProposalError("recovery probe receipt identity or integrity is invalid")
        _validate_runtime_observation(probe.get("runtime_boundary"))


def _progress_probe_receipt_write(store: Path, binding: dict, probe: dict,
                                  config_sha256: str, capacity: int) -> None:
    receipt_id = uuid.uuid4().hex
    base = {"schema_version": PROGRESS_PROBE_RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        "binding_sha256": _sha(_canonical_json(binding).rstrip(b"\n")),
        "proposal_config_sha256": config_sha256, "probe": probe}
    record = {**base, "record_sha256": _sha(_canonical_json(base).rstrip(b"\n"))}
    raw = _canonical_json(record)
    if len(raw) > PROGRESS_TERMINAL_MAX_BYTES or _progress_store_usage(store) + len(raw) > capacity:
        raise ProposalError("progress store capacity cannot cover the recovery probe receipt")
    _write_new(store / f"recovery-probe-{receipt_id}.json", raw)
    _sync_directory(store)


def _progress_store_read(store: Path, expected_binding: dict, rows: list[dict], cfg: dict,
                         config_sha256: str | None = None,
                         return_metadata: bool = False,
                         update_capacity: bool = True) -> tuple[list[dict], dict[int, dict]] | tuple[list[dict], dict[int, dict], dict]:
    if (store.is_symlink() or not store.is_dir() or not _outside_git(store)
            or not _private_dir(store.parent) or not _private_dir(store)):
        raise ProposalError("private progress store is unavailable or unsafe")
    header_path, events_path = store / "checkpoint.json", store / "events.jsonl"
    if (header_path.is_symlink() or events_path.is_symlink()
            or not _private_file(header_path) or not _private_file(events_path)):
        raise ProposalError("private progress store permissions changed")
    header_raw = _read(header_path, PROGRESS_HEADER_MAX_BYTES, "progress checkpoint")
    header = _json(header_raw, "progress checkpoint")
    if (not isinstance(header, dict)
            or set(header) != {"schema_version", "binding", "proposal_config_sha256"}
            or header.get("schema_version") != PROGRESS_STORE_SCHEMA
            or header.get("binding") != expected_binding
            or (header.get("proposal_config_sha256") is not None
                and (not isinstance(header["proposal_config_sha256"], str)
                     or not HEX.fullmatch(header["proposal_config_sha256"])) )):
        raise ProposalError("progress checkpoint identity or selected row order changed")
    historical_config_sha256 = header["proposal_config_sha256"]
    capacity = _progress_capacity_records(store, cfg["max_progress_store_bytes"],
                                          update_increase=False,
                                          config_sha256=config_sha256,
                                          original_config_sha256=historical_config_sha256)
    events_raw = _read(events_path, capacity, "progress event log")
    attempt_rows = {}
    attempt_orders = {}
    for path in store.iterdir():
        if path.name.startswith("attempt-"):
            if (path.is_symlink() or not path.is_file() or not _private_file(path)
                    or not re.fullmatch(r"attempt-[0-9a-f]{32}\.json", path.name)):
                raise ProposalError("private progress attempt record is unsafe")
            record = _json(_read(path, 1_048_576, "progress attempt record"), "progress attempt record")
            expected_keys = {"schema_version", "attempt_id", "binding_sha256", "indices",
                             "row_ids_sha256", "state", "record_sha256"}
            if not isinstance(record, dict) or set(record) != expected_keys:
                raise ProposalError("progress attempt record schema is invalid")
            base = {key: record[key] for key in expected_keys - {"record_sha256"}}
            attempt_id = path.stem.removeprefix("attempt-")
            indices = record["indices"]
            if (record["schema_version"] != PROGRESS_ATTEMPT_SCHEMA or record["attempt_id"] != attempt_id
                    or record["binding_sha256"] != _sha(_canonical_json(expected_binding).rstrip(b"\n"))
                    or not isinstance(indices, list) or not indices
                    or any(type(index) is not int or index < 0 or index >= len(rows) for index in indices)
                    or len(indices) != len(set(indices)) or indices != sorted(indices)
                    or record["row_ids_sha256"] != _sha(_canonical_json([expected_binding["selected_row_ids"][i]
                                                                           for i in indices]).rstrip(b"\n"))
                    or record["state"] != "prepared_or_sent_outcome_unknown"
                    or record["record_sha256"] != _sha(_canonical_json(base).rstrip(b"\n"))):
                raise ProposalError("progress attempt identity or integrity is invalid")
            attempt_rows[attempt_id] = set(indices)
            attempt_orders[attempt_id] = indices
        elif (path.name not in {"checkpoint.json", "events.jsonl", "capacity.jsonl"}
              and not path.name.startswith(("terminal-", "revision-", "recovery-probe-", "publication-"))):
            raise ProposalError("progress store contains an unknown entry")
    if events_raw and not events_raw.endswith(b"\n"):
        raise ProposalError("progress event log is truncated")
    seen = set()
    records = []
    latest: dict[int, dict] = {}
    expected_teacher_identity = _teacher_wire_identity(cfg)
    explicit_mode = expected_teacher_identity["mode"] == explicit_labels.SCHEMA_VERSION
    for line in events_raw.splitlines():
        if not line or len(line) > PROGRESS_EVENT_MAX_BYTES:
            raise ProposalError("progress event log contains an invalid record")
        record = _json(line, "progress event")
        expected_record_keys = {"schema_version", "attempt_id", "index", "event", "record_sha256"}
        if explicit_mode:
            expected_record_keys.add("private_extension_record")
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise ProposalError("progress event schema is invalid")
        base = {k: record[k] for k in expected_record_keys - {"record_sha256"}}
        if (record["schema_version"] != PROGRESS_EVENT_SCHEMA
                or not isinstance(record["attempt_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", record["attempt_id"])
                or type(record["index"]) is not int or not 0 <= record["index"] < len(rows)
                or record["record_sha256"] != _sha(_canonical_json(base).rstrip(b"\n"))):
            raise ProposalError("progress event integrity or index is invalid")
        key = (record["attempt_id"], record["index"])
        if key in seen or record["attempt_id"] not in attempt_rows or record["index"] not in attempt_rows[record["attempt_id"]]:
            raise ProposalError("progress event contains duplicate attempt indices")
        seen.add(key)
        event = record["event"]
        if not isinstance(event, dict) or event.get("index") != record["index"]:
            raise ProposalError("progress event does not preserve original row index")
        if explicit_mode:
            private_record = record["private_extension_record"]
            if (not isinstance(private_record, dict)
                    or private_record.get("worker_index") != attempt_orders[record["attempt_id"]].index(record["index"])):
                raise ProposalError("private extension worker ordinal does not map to its attempt row")
            event = _restore_private_extension_event(event, record["private_extension_record"],
                record["attempt_id"], record["index"], rows[record["index"]],
                expected_binding, expected_teacher_identity,
                cfg.get("max_teacher_extension_bytes", explicit_labels.MAX_EXTENSION_BYTES))
            event = {**event, "index": record["index"]}
        _validate_progress_event(event, record["index"], rows[record["index"]],
                                 cfg["selected_columns"], cfg["assume_exhaustive_labels"],
                                 expected_teacher_identity,
                                 cfg.get("max_teacher_extension_bytes", explicit_labels.MAX_EXTENSION_BYTES))
        public_record = {k: record[k] for k in ("schema_version", "attempt_id", "index", "event", "record_sha256")}
        records.append(public_record)
        latest[record["index"]] = event["result"]
    terminal = _progress_store_matching_terminal(store, expected_binding, _sha(events_raw))
    if terminal is not None and terminal["attempt_id"] not in attempt_rows:
        raise ProposalError("progress terminal attempt is absent from attempt records")
    _progress_probe_receipts_validate(store, expected_binding)
    _progress_publication_receipts_validate(store, expected_binding)
    revision_state = _progress_revision_state(store, Path(cfg["output_dir"]), expected_binding)
    # An increase becomes durable only after all identity, event and terminal
    # evidence has passed validation; a rejected resume leaves history untouched.
    if update_capacity:
        _progress_capacity_records(store, cfg["max_progress_store_bytes"], update_increase=True,
                                   config_sha256=config_sha256,
                                   original_config_sha256=historical_config_sha256)
    if return_metadata:
        capacity_raw = _read(store / "capacity.jsonl", 65_536, "progress capacity history")
        capacity_config_hashes = [_json(line, "progress capacity record").get("config_sha256")
                                  for line in capacity_raw.splitlines()]
        return records, latest, {"proposal_config_sha256": historical_config_sha256,
                                 "capacity_config_sha256s": capacity_config_hashes,
                                 "capacity_bytes": cfg["max_progress_store_bytes"],
                                 "revision_state": revision_state}
    return records, latest


def _progress_store_create(store: Path, expected_binding: dict,
                           capacity: int = PROGRESS_STORE_DEFAULT_BYTES,
                           proposal_config_sha256: str | None = None) -> None:
    if (store.exists() or store.is_symlink() or not _outside_git(store)
            or not store.parent.exists() or not _private_dir(store.parent)):
        raise ProposalError("progress store already exists or is inside Git; use explicit resume")
    if proposal_config_sha256 is not None and not HEX.fullmatch(proposal_config_sha256):
        raise ProposalError("original proposal config digest is invalid")
    header_raw = _canonical_json({"schema_version": PROGRESS_STORE_SCHEMA,
                                  "binding": expected_binding,
                                  "proposal_config_sha256": proposal_config_sha256})
    capacity_base = {"schema_version": PROGRESS_CAPACITY_SCHEMA,
                     "sequence": 0, "capacity_bytes": capacity,
                     "config_sha256": proposal_config_sha256}
    capacity_raw = _canonical_json({**capacity_base,
        "record_sha256": _sha(_canonical_json(capacity_base).rstrip(b"\n"))})
    if len(header_raw) + len(capacity_raw) > capacity:
        raise ProposalError("progress store capacity cannot cover its initial identity records")
    created = False
    try:
        store.mkdir(mode=0o700, parents=False, exist_ok=False)
        created = True
        os.chmod(store, 0o700)
        _masked()._secure_private_path(store, is_directory=True)
        _masked()._verify_owner_private_access(store, is_directory=True)
        _write_new(store / "checkpoint.json", header_raw)
        _write_new(store / "events.jsonl", b"")
        _progress_capacity_records(store, capacity, create=True,
                                   config_sha256=proposal_config_sha256)
        _sync_directory(store)
        _sync_directory(store.parent)
    except Exception:
        # Remove only this call's empty directory; keep it intact once it has
        # acquired checkpoint or event bytes so interrupted work remains evidence.
        try:
            if created and store.is_dir() and not store.is_symlink() and not any(store.iterdir()):
                store.rmdir()
        except OSError:
            pass
        raise


def _progress_store_append(store: Path, attempt_id: str, index: int, event: dict,
                           capacity: int,
                           private_extension_record: dict | None = None) -> None:
    path = store / "events.jsonl"
    if (not _private_dir(store.parent) or not _private_dir(store)
            or path.is_symlink() or not path.is_file() or not _private_file(path)):
        raise ProposalError("private progress event file is unavailable or unsafe")
    base = {"schema_version": PROGRESS_EVENT_SCHEMA, "attempt_id": attempt_id,
            "index": index, "event": event}
    if private_extension_record is not None:
        base["private_extension_record"] = private_extension_record
    record = {**base, "record_sha256": _sha(_canonical_json(base).rstrip(b"\n"))}
    raw = _canonical_json(record)
    if len(raw) > PROGRESS_EVENT_MAX_BYTES or _progress_store_usage(store) + len(raw) > capacity:
        raise ProposalError("private progress event log exceeds configured bound")
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        path_info = path.stat(follow_symlinks=False)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino)):
            raise ProposalError("private progress event file changed type")
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short progress checkpoint write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _progress_store_record_attempt(store: Path, attempt_id: str, indices: list[int],
                                   binding: dict, capacity: int = PROGRESS_STORE_DEFAULT_BYTES) -> None:
    base = {"schema_version": PROGRESS_ATTEMPT_SCHEMA, "attempt_id": attempt_id,
            "binding_sha256": _sha(_canonical_json(binding).rstrip(b"\n")),
            "indices": indices,
            "row_ids_sha256": _sha(_canonical_json([binding["selected_row_ids"][i]
                                                     for i in indices]).rstrip(b"\n")),
            "state": "prepared_or_sent_outcome_unknown"}
    record = {**base, "record_sha256": _sha(_canonical_json(base).rstrip(b"\n"))}
    raw = _canonical_json(record)
    if _progress_store_usage(store) + len(raw) > capacity:
        raise ProposalError("progress store capacity cannot cover the next attempt record")
    _write_new(store / f"attempt-{attempt_id}.json", raw)
    _sync_directory(store)


def _progress_store_write_terminal(store: Path, attempt_id: str, binding: dict,
                                   events_sha256: str, output_manifest: dict,
                                   capacity: int = PROGRESS_STORE_DEFAULT_BYTES) -> None:
    base = {"schema_version": PROGRESS_TERMINAL_SCHEMA, "attempt_id": attempt_id,
            "binding": binding, "events_sha256": events_sha256,
            "output_manifest": output_manifest}
    record = {**base, "record_sha256": _sha(_canonical_json(base).rstrip(b"\n"))}
    raw = _canonical_json(record)
    if len(raw) > PROGRESS_TERMINAL_MAX_BYTES or _progress_store_usage(store) + len(raw) > capacity:
        raise ProposalError("progress store capacity cannot cover the terminal record")
    _write_new(store / f"terminal-{attempt_id}.json", raw)
    _sync_directory(store)


def _progress_store_matching_terminal(store: Path, binding: dict,
                                      events_sha256: str) -> dict | None:
    matches = []
    for path in store.iterdir():
        if not path.name.startswith("terminal-"):
            continue
        if (path.is_symlink() or not path.is_file() or not _private_file(path)
                or len(path.name) != len("terminal-") + 32 + len(".json")
                or not re.fullmatch(r"terminal-[0-9a-f]{32}\.json", path.name)):
            raise ProposalError("private progress terminal record is unsafe")
        record = _json(_read(path, PROGRESS_TERMINAL_MAX_BYTES, "progress terminal record"),
                       "progress terminal record")
        expected_keys = {"schema_version", "attempt_id", "binding", "events_sha256",
                         "output_manifest", "record_sha256"}
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise ProposalError("progress terminal record schema is invalid")
        base = {key: record[key] for key in expected_keys - {"record_sha256"}}
        if (record["schema_version"] != PROGRESS_TERMINAL_SCHEMA
                or record["attempt_id"] != path.stem.removeprefix("terminal-")
                or not isinstance(record["events_sha256"], str) or not HEX.fullmatch(record["events_sha256"])
                or record["binding"] != binding
                or record["record_sha256"] != _sha(_canonical_json(base).rstrip(b"\n"))
                or not isinstance(record["output_manifest"], dict)
                or record["output_manifest"].get("schema_version") != REPORT_SCHEMA
                or record["output_manifest"].get("provenance", {}).get("progress_store_events_sha256") != record["events_sha256"]
                or record["output_manifest"].get("provenance", {}).get("terminal_worker_result_verified") is not True):
            raise ProposalError("progress terminal record identity or integrity is invalid")
        if record["events_sha256"] == events_sha256:
            matches.append(record)
    if len(matches) > 1:
        raise ProposalError("progress store has ambiguous matching terminal records")
    return matches[0] if matches else None


def _progress_pending_indices(rows: list[dict], stored_progress: dict[int, dict]) -> list[int]:
    retryable = {"timeout", "transport_error", "protocol_error", "identity_mismatch",
                 "http_error", "incomplete", "request_too_large_not_started",
                 "not_attempted_after_abort"}
    return [index for index in range(len(rows))
            if index not in stored_progress or stored_progress[index]["status"] in retryable]


def _validate_terminal_report(report: Any, cfg: dict, cfg_sha: str, comp: dict, comp_sha: str,
                              source_manifest: dict, source_binding: dict, rows: list[dict],
                              latest: dict[int, dict], checkpoint_binding: dict,
                              events_sha: str, probe: dict, adapter_sha: str,
                              comparator_sha: str, config_history: dict,
                              expected_revision: dict | None) -> dict:
    if not isinstance(report, dict):
        raise ProposalError("durable terminal report is not an object")
    top_keys = {"schema_version", "created_at_epoch_seconds", "status", "promotion_eligible", "claims",
                "provenance", "counts", "pre_data_zero_row_probe", "abort", "resources", "uncertainty"}
    if set(report) != top_keys or report.get("schema_version") != REPORT_SCHEMA:
        raise ProposalError("durable terminal report schema is invalid")
    status_counts = dict(sorted(Counter(item["status"] for item in latest.values()).items()))
    label_coverage = Counter(f"{label}:{state}" for item in latest.values()
        for label, state in item["proposal_states"].items())
    expected_counts = {"selected_rows": len(rows), "statuses": status_counts,
        "answered_rows": status_counts.get("answered", 0),
        "abstentions": status_counts.get("abstain", 0),
        "failures": sum(n for status, n in status_counts.items() if status not in ("answered", "abstain")),
        "not_attempted_after_abort": status_counts.get("not_attempted_after_abort", 0),
        "per_label_and_state": dict(sorted(label_coverage.items()))}
    abort_map = {"timeout": "timeout_or_worker_lifecycle_ambiguous",
        "transport_error": "transport_or_worker_lifecycle_ambiguous",
        "protocol_error": "protocol_or_worker_lifecycle_ambiguous",
        "identity_mismatch": "model_identity_or_response_lifecycle_ambiguous",
        "http_error": "model_identity_or_response_lifecycle_ambiguous",
        "incomplete": "model_identity_or_response_lifecycle_ambiguous"}
    abort_status = next((latest[i]["status"] for i in range(len(rows))
                         if latest[i]["status"] in abort_map), None)
    expected_abort = {"aborted": abort_status is not None,
                      "reason": abort_map.get(abort_status)}
    expected_claims = {"teacher_suggestions_are_unreviewed": True, "heldout_accessed": False,
        "labels_independently_verified": False, "screening_independently_verified": False,
        "promotion_eligible": False}
    expected_uncertainty = {"source_and_review_claims_independently_verified": False,
        "teacher_labels_are_claims": True, "omitted_labels_are_negative": cfg["assume_exhaustive_labels"],
        "selection_claim": "configured train-only subset; no heldout data opened"}
    expected_status = "partial_aborted_ambiguous" if abort_status is not None else "complete"
    if (report.get("status") != expected_status or report.get("promotion_eligible") is not False
            or report.get("claims") != expected_claims or report.get("counts") != expected_counts
            or report.get("abort") != expected_abort or report.get("uncertainty") != expected_uncertainty
            or type(report.get("created_at_epoch_seconds")) is not int
            or report["created_at_epoch_seconds"] < 0):
        raise ProposalError("durable terminal claims, counts, status or uncertainty do not match row evidence")
    probe_record = report.get("pre_data_zero_row_probe")
    if not isinstance(probe_record, dict):
        raise ProposalError("durable terminal probe evidence is invalid")
    stable_config_sha = _proposal_config_identity_sha(cfg)
    original_config_sha = config_history.get("proposal_config_sha256") or cfg_sha
    allowed_raw_configs = {value for value in config_history.get("capacity_config_sha256s", [])
                           if isinstance(value, str) and HEX.fullmatch(value)}
    allowed_raw_configs.add(original_config_sha)
    if (probe_record.get("zero_row_probe") is not True
            or probe_record.get("worker_ready") is not True
            or probe_record.get("cleanup_verified") is not True
            or probe_record.get("outside_listener_denied") is not True
            or probe_record.get("persistent_service_unchanged") is not True
            or probe_record.get("proposal_config_sha256") not in allowed_raw_configs
            or probe_record.get("proposal_config_identity_sha256") != stable_config_sha
            or probe_record.get("comparison_config_sha256") != comp_sha
            or probe_record.get("adapter_code_sha256") != adapter_sha
            or probe_record.get("comparator_code_sha256") != comparator_sha
            or probe_record.get("teacher_identity") != _teacher_wire_identity(cfg)
            or probe_record.get("receipt_sha256") != probe.get("receipt_sha256")
            or probe_record.get("model_identity", {}).get("reported_manifest_sha256") !=
                probe.get("model_identity", {}).get("reported_manifest_sha256")
            or probe_record.get("model_layer_bytes") != probe.get("model_layer_bytes")):
        raise ProposalError("durable original probe evidence differs from stable task identity")
    original_host = probe_record.get("host_before")
    current_host = probe.get("host_before")
    if (not isinstance(original_host, dict) or not isinstance(current_host, dict)
            or original_host.get("hostname") != current_host.get("hostname")
            or original_host.get("architecture") != current_host.get("architecture")):
        raise ProposalError("durable original probe host identity differs from current task")
    original_probe_runtime = _validate_runtime_observation(probe_record.get("runtime_boundary"))
    fresh_probe_runtime = _validate_runtime_observation(probe.get("runtime_boundary"))
    if original_probe_runtime != fresh_probe_runtime:
        raise ProposalError("fresh probe changed stable runtime identity")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ProposalError("durable terminal provenance is invalid")
    artifacts = source_manifest.get("source_artifacts", [])
    artifact_hashes = {item["path"]: item["sha256"] for item in artifacts
        if isinstance(item, dict) and isinstance(item.get("path"), str)
        and isinstance(item.get("sha256"), str)}
    copy_claim = source_manifest.get("source_copy_claim", {})
    source_lineage = {"source_artifacts": artifacts, "source_artifacts_sha256": artifact_hashes,
        "source_artifact_inventory_sha256": _sha(json.dumps(artifact_hashes, sort_keys=True,
            separators=(",", ":")).encode()),
        "screening_policy_sha256": source_manifest.get("screening_policy_sha256"),
        "screening_code_sha256": source_manifest.get("screening_code_sha256"),
        "taxonomy_sha256": source_manifest.get("taxonomy_sha256"),
        "taxonomy_loader_sha256": source_manifest.get("taxonomy_loader_sha256"),
        "export_config_sha256": source_manifest.get("config_sha256"),
        "workspace_plan_sha256": source_manifest.get("workspace_plan_sha256") or copy_claim.get("plan_sha256"),
        "review_ledger_sha256": source_manifest.get("review_ledger_sha256") or copy_claim.get("ledger_sha256"),
        "partial_sidecar_sha256": source_manifest.get("partial_sidecar_sha256") or copy_claim.get("sidecar_sha256"),
        "label_guide_sha256": source_manifest.get("label_guide_sha256"),
        "target_policy": source_manifest.get("target_policy"),
        "selected_columns": source_manifest.get("selected_columns"),
        "counts": source_manifest.get("counts"), "promotion_eligible": False}
    selected_ids = [row["id"] for row in rows]
    labels = [f"{column['field']}:{column['label']}" for column in cfg["selected_columns"]]
    expected_provenance = {"comparison_config_sha256": comp_sha,
        "adapter_code_sha256": adapter_sha, "comparator_code_sha256": comparator_sha,
        "export_manifest_sha256": source_binding["manifest_sha256"],
        "export_config_sha256": source_binding["config_sha256"],
        "train_file_sha256": source_binding["train_sha256"], "train_file_bytes": source_binding["train_bytes"],
        "taxonomy_sha256": cfg["taxonomy_sha256"], "source_lineage": source_lineage,
        "teacher_identity": _teacher_wire_identity(cfg),
        "selected_row_count": len(rows),
        "selected_row_ids_sha256": _sha(json.dumps(selected_ids, separators=(",", ":")).encode()),
        "selected_text_digests_sha256": _sha(json.dumps([r["text_sha256"] for r in rows], separators=(",", ":")).encode()),
        "target_mask_sha256": _sha(json.dumps([r["known_mask"] for r in rows], separators=(",", ":")).encode()),
        "proposal_schema_sha256": _proposal_schema_digest(cfg),
        "prompt_contract_sha256": checkpoint_binding["prompt_contract_sha256"],
        "progress_store_schema": PROGRESS_STORE_SCHEMA,
        "progress_store_id_sha256": _sha(_canonical_json(checkpoint_binding).rstrip(b"\n")),
        "progress_store_events_sha256": events_sha, "terminal_worker_result_verified": True,
        "model_name": comp["model_name"], "model_manifest_sha256": checkpoint_binding["model_manifest_sha256"],
        "model_layer_bytes": probe.get("model_layer_bytes"), "receipt_sha256": probe.get("receipt_sha256"),
        "server_binary_sha256": comp["server_binary_sha256"], "server_version": comp["server_version"],
        "request_policy": {k: comp[k] for k in ("timeout_seconds", "max_request_bytes", "max_response_bytes",
            "num_ctx", "num_predict", "temperature", "top_k", "top_p", "repeat_penalty")},
        "resource_policy": {k: comp[k] for k in ("resource_memory_max_bytes", "resource_cpu_quota_percent",
            "remote_scope_tasks_max", "min_memory_available_bytes", "min_disk_free_bytes", "min_gpu_free_bytes")}}
    allowed_provenance = set(expected_provenance) | {"proposal_config_sha256",
        "proposal_config_origin_sha256", "proposal_config_identity_sha256",
        "worker_config_sha256", "payload_sha256", "progress_resumed", "runtime_boundary",
        "publication_revision"}
    if (set(provenance) != allowed_provenance
            or any(provenance.get(key) != value for key, value in expected_provenance.items())
            or provenance.get("proposal_config_sha256") not in allowed_raw_configs
            or provenance.get("proposal_config_origin_sha256") != original_config_sha
            or provenance.get("proposal_config_identity_sha256") != stable_config_sha
            or provenance.get("publication_revision") != expected_revision
            or type(provenance.get("progress_resumed")) is not bool
            or not isinstance(provenance.get("worker_config_sha256"), str)
            or not HEX.fullmatch(provenance["worker_config_sha256"])
            or not isinstance(provenance.get("payload_sha256"), str)
            or not HEX.fullmatch(provenance["payload_sha256"])):
        raise ProposalError("durable terminal provenance does not match source/model/task identity")
    actual_runtime = _validate_runtime_observation(provenance.get("runtime_boundary"))
    if actual_runtime != original_probe_runtime:
        raise ProposalError("actual worker runtime differs from original probe stable identity")
    resources = report.get("resources")
    if (not isinstance(resources, dict) or set(resources) != {"before", "after"}
            or not isinstance(resources["before"], dict) or not isinstance(resources["after"], dict)):
        raise ProposalError("durable terminal resource accounting is invalid")
    return report


def _checkpoint_worker_event(store: Path, attempt_id: str, worker_index: int,
                             original_index: int, row: dict, event: dict,
                             columns: list[dict], exhaustive: bool,
                             max_bytes: int, teacher_identity: dict | None = None,
                             max_extension_bytes: int = explicit_labels.MAX_EXTENSION_BYTES,
                             checkpoint_binding: dict | None = None) -> dict:
    item = _validate_progress_event(event, worker_index, row, columns, exhaustive,
                                    teacher_identity, max_extension_bytes)
    stored_event = {key: value for key, value in event.items() if key != "private_extensions"}
    stored_event["index"] = original_index
    extension_record = None
    if teacher_identity is not None and teacher_identity.get("mode") == explicit_labels.SCHEMA_VERSION:
        if not isinstance(checkpoint_binding, dict):
            raise ProposalError("explicit private-extension checkpoint binding is absent")
        extension_record = _private_extension_record(event, attempt_id, worker_index,
            original_index, row, teacher_identity, checkpoint_binding, max_extension_bytes)
    _progress_store_append(store, attempt_id, original_index, stored_event, max_bytes,
                           private_extension_record=extension_record)
    return item


def _progress_event_digest(event: dict) -> str:
    return _sha(_canonical_json(event).rstrip(b"\n"))


def _progress_ack(attempt_id: str, index: int, event: dict) -> dict:
    return {"schema_version": PROGRESS_ACK_SCHEMA, "phase": "ack",
            "attempt_id": attempt_id, "index": index,
            "event_sha256": _progress_event_digest(event)}


def _private_extension_record(event: dict, attempt_id: str, worker_index: int,
                             original_index: int, row: dict, teacher_identity: dict,
                             checkpoint_binding: dict, max_extension_bytes: int) -> dict:
    present = event["private_extensions_present"]
    extensions = event["private_extensions"]
    extension_digest = event["private_extensions_sha256"]
    if present:
        value = explicit_labels._canonical_bytes(extensions)
        if len(value) > max_extension_bytes or _sha(value) != extension_digest:
            raise ProposalError("private extension bytes exceed the bound or fail their digest")
        persisted_value = extensions
    else:
        persisted_value = None
    base = {"schema_version": PRIVATE_EXTENSION_SCHEMA, "attempt_id": attempt_id,
        "worker_index": worker_index, "original_index": original_index,
        "source_row_id": row["id"],
        "source_identity_sha256": _sha(_canonical_json(row["source_identity"]).rstrip(b"\n")),
        "source_text_sha256": row["text_sha256"],
        "checkpoint_binding_sha256": _sha(_canonical_json(checkpoint_binding).rstrip(b"\n")),
        "teacher_identity": teacher_identity,
        "model_name": checkpoint_binding["model_name"],
        "model_manifest_sha256": checkpoint_binding["model_manifest_sha256"],
        "taxonomy_sha256": checkpoint_binding["taxonomy_sha256"],
        "extensions_present": present, "extensions_sha256": extension_digest,
        "extensions": persisted_value, "worker_event_sha256": _progress_event_digest(event)}
    return {**base, "record_sha256": _sha(_canonical_json(base).rstrip(b"\n"))}


def _restore_private_extension_event(event: dict, private_record: Any, attempt_id: str,
                                     original_index: int, row: dict, expected_binding: dict,
                                     teacher_identity: dict, max_extension_bytes: int) -> dict:
    keys = {"schema_version", "attempt_id", "worker_index", "original_index", "source_row_id",
        "source_identity_sha256", "source_text_sha256", "checkpoint_binding_sha256",
        "teacher_identity", "model_name", "model_manifest_sha256", "taxonomy_sha256",
        "extensions_present", "extensions_sha256", "extensions", "worker_event_sha256", "record_sha256"}
    if not isinstance(private_record, dict) or set(private_record) != keys:
        raise ProposalError("private extension checkpoint record schema is invalid")
    base = {k: private_record[k] for k in keys - {"record_sha256"}}
    present, extensions = private_record["extensions_present"], private_record["extensions"]
    digest = private_record["extensions_sha256"]
    if (private_record["schema_version"] != PRIVATE_EXTENSION_SCHEMA
            or private_record["attempt_id"] != attempt_id
            or type(private_record["worker_index"]) is not int or private_record["worker_index"] < 0
            or private_record["original_index"] != original_index
            or private_record["source_row_id"] != row["id"]
            or private_record["source_identity_sha256"] != _sha(_canonical_json(row["source_identity"]).rstrip(b"\n"))
            or private_record["source_text_sha256"] != row["text_sha256"]
            or private_record["checkpoint_binding_sha256"] != _sha(_canonical_json(expected_binding).rstrip(b"\n"))
            or private_record["teacher_identity"] != teacher_identity
            or private_record["model_name"] != expected_binding["model_name"]
            or private_record["model_manifest_sha256"] != expected_binding["model_manifest_sha256"]
            or private_record["taxonomy_sha256"] != expected_binding["taxonomy_sha256"]
            or private_record["record_sha256"] != _sha(_canonical_json(base).rstrip(b"\n"))
            or type(present) is not bool):
        raise ProposalError("private extension checkpoint identity or integrity is invalid")
    if present:
        if not isinstance(extensions, dict) or not isinstance(digest, str) or not HEX.fullmatch(digest):
            raise ProposalError("private extension checkpoint value is invalid")
        extension_bytes = explicit_labels._canonical_bytes(extensions)
        if len(extension_bytes) > max_extension_bytes or _sha(extension_bytes) != digest:
            raise ProposalError("private extension checkpoint value failed its bound or digest")
    elif extensions is not None or digest is not None:
        raise ProposalError("missing private extension checkpoint value was changed")
    if (not isinstance(private_record["worker_event_sha256"], str)
            or not HEX.fullmatch(private_record["worker_event_sha256"])):
        raise ProposalError("private extension original event digest is invalid")
    full_event = {**event, "index": private_record["worker_index"],
        "private_extensions_present": present,
        "private_extensions_sha256": digest,
        "private_extensions": extensions if present else {}}
    if _progress_event_digest(full_event) != private_record["worker_event_sha256"]:
        raise ProposalError("private extension checkpoint does not match its original worker event")
    return full_event


def read_private_teacher_extensions(store: Path, row_id: str,
                                    expected_binding: dict) -> list[PrivateTeacherExtensions]:
    """Deliberately retrieve bounded extension values from a verified private checkpoint."""
    if (not isinstance(row_id, str) or not HEX.fullmatch(row_id)
            or not isinstance(expected_binding, dict)
            or expected_binding.get("teacher_response_mode") != explicit_labels.SCHEMA_VERSION):
        raise ProposalError("private extension retrieval identity is invalid")
    if (store.is_symlink() or not store.is_dir() or not _outside_git(store)
            or not _private_dir(store.parent) or not _private_dir(store)):
        raise ProposalError("private extension progress store is unavailable or unsafe")
    header_path, events_path = store / "checkpoint.json", store / "events.jsonl"
    if any(path.is_symlink() or not path.is_file() or not _private_file(path)
           for path in (header_path, events_path)):
        raise ProposalError("private extension progress files are unavailable or unsafe")
    header = _json(_read(header_path, PROGRESS_HEADER_MAX_BYTES, "progress checkpoint"),
                   "progress checkpoint")
    selected_ids = expected_binding.get("selected_row_ids")
    if (not isinstance(header, dict) or header.get("schema_version") != PROGRESS_STORE_SCHEMA
            or header.get("binding") != expected_binding
            or not isinstance(selected_ids, list) or row_id not in selected_ids):
        raise ProposalError("private extension checkpoint binding changed")
    teacher_identity = {"mode": expected_binding.get("teacher_response_mode"),
        "schema_sha256": expected_binding.get("teacher_schema_sha256"),
        "prompt_sha256": expected_binding.get("teacher_prompt_sha256"),
        "module_sha256": expected_binding.get("explicit_labels_module_sha256")}
    if any(not isinstance(teacher_identity[key], str) or not HEX.fullmatch(teacher_identity[key])
           for key in ("schema_sha256", "prompt_sha256", "module_sha256")):
        raise ProposalError("private extension schema or code identity is invalid")
    events_raw = _read(events_path, PROGRESS_STORE_HARD_MAX_BYTES, "progress event log")
    if events_raw and not events_raw.endswith(b"\n"):
        raise ProposalError("private extension progress log is truncated")
    binding_sha = _sha(_canonical_json(expected_binding).rstrip(b"\n"))
    attempts: dict[str, dict] = {}
    seen: set[tuple[str, int]] = set()
    found: list[PrivateTeacherExtensions] = []
    for line in events_raw.splitlines():
        record = _json(line, "progress event")
        required = {"schema_version", "attempt_id", "index", "event", "record_sha256", "private_extension_record"}
        if (not isinstance(record, dict) or set(record) != required
                or record.get("schema_version") != PROGRESS_EVENT_SCHEMA
                or not isinstance(record.get("attempt_id"), str)
                or not re.fullmatch(r"[0-9a-f]{32}", record["attempt_id"])
                or type(record.get("index")) is not int
                or not 0 <= record["index"] < len(selected_ids)):
            raise ProposalError("private extension progress record schema is invalid")
        base = {key: record[key] for key in required - {"record_sha256"}}
        if record["record_sha256"] != _sha(_canonical_json(base).rstrip(b"\n")):
            raise ProposalError("private extension progress record integrity failed")
        public_event = record["event"]
        item = public_event.get("result") if isinstance(public_event, dict) else None
        index = record["index"]
        if (not isinstance(item, dict) or item.get("row_id") != selected_ids[index]
                or public_event.get("index") != index):
            raise ProposalError("private extension event source row changed")
        attempt_id = record["attempt_id"]
        key = (attempt_id, index)
        if key in seen:
            raise ProposalError("private extension progress contains duplicate attempt events")
        seen.add(key)
        if attempt_id not in attempts:
            attempt_path = store / f"attempt-{attempt_id}.json"
            if attempt_path.is_symlink() or not _private_file(attempt_path):
                raise ProposalError("private extension attempt evidence is unavailable")
            attempt = _json(_read(attempt_path, 1_048_576, "progress attempt record"),
                            "progress attempt record")
            attempt_keys = {"schema_version", "attempt_id", "binding_sha256", "indices",
                "row_ids_sha256", "state", "record_sha256"}
            attempt_base = ({k: attempt[k] for k in attempt_keys - {"record_sha256"}}
                            if isinstance(attempt, dict) else {})
            indices = attempt.get("indices") if isinstance(attempt, dict) else None
            if (not isinstance(attempt, dict) or set(attempt) != attempt_keys
                    or attempt.get("schema_version") != PROGRESS_ATTEMPT_SCHEMA
                    or attempt.get("attempt_id") != attempt_id
                    or attempt.get("binding_sha256") != binding_sha
                    or not isinstance(indices, list) or not indices
                    or any(type(i) is not int or not 0 <= i < len(selected_ids) for i in indices)
                    or len(indices) != len(set(indices)) or indices != sorted(indices)
                    or index not in indices
                    or attempt.get("row_ids_sha256") != _sha(_canonical_json([selected_ids[i] for i in indices]).rstrip(b"\n"))
                    or attempt.get("record_sha256") != _sha(_canonical_json(attempt_base).rstrip(b"\n"))):
                raise ProposalError("private extension attempt binding or integrity failed")
            attempts[attempt_id] = attempt
        attempt_indices = attempts[attempt_id]["indices"]
        private_record = record["private_extension_record"]
        worker_index = private_record.get("worker_index") if isinstance(private_record, dict) else None
        if (index not in attempt_indices
                or worker_index != attempt_indices.index(index)):
            raise ProposalError("private extension event ordinal is not bound to its attempt row")
        source_row = {"id": item["row_id"], "source_identity": item["source_identity"],
                      "text_sha256": item["text_sha256"]}
        full_event = _restore_private_extension_event(public_event, record["private_extension_record"],
            attempt_id, index, source_row, expected_binding, teacher_identity,
            expected_binding.get("max_teacher_extension_bytes", explicit_labels.MAX_EXTENSION_BYTES))
        validation_row = {"id": item["row_id"], "family": item["family_sha256"],
            "text_sha256": item["text_sha256"], "original_text_sha256": item["original_text_sha256"],
            "screening": item["screening"], "verified_source_claim": item["verified_source_claim"],
            "source_identity": item["source_identity"], "columns": item["columns"],
            "targets": item["source_targets"], "known_mask": item["source_known_mask"],
            "target_origins": item["source_target_origins"],
            "target_reviewer_claims": item["source_reviewer_claims"]}
        _validate_progress_event({**full_event, "index": index}, index, validation_row,
            item["columns"], False, teacher_identity,
            expected_binding.get("max_teacher_extension_bytes", explicit_labels.MAX_EXTENSION_BYTES))
        if row_id == item["row_id"]:
            private_record = record["private_extension_record"]
            metadata = {k: private_record[k] for k in (
                "attempt_id", "worker_index", "original_index", "source_row_id",
                "source_identity_sha256", "source_text_sha256", "teacher_identity",
                "model_name", "model_manifest_sha256", "taxonomy_sha256",
                "extensions_present", "extensions_sha256", "worker_event_sha256")}
            found.append(PrivateTeacherExtensions(metadata,
                private_record["extensions"] if private_record["extensions_present"] else None,
                private_record["extensions_present"]))
    return found


def _validate_progress_ack(value: Any, attempt_id: str, index: int, event: dict) -> None:
    if (not isinstance(value, dict)
            or set(value) != {"schema_version", "phase", "attempt_id", "index", "event_sha256"}
            or value.get("schema_version") != PROGRESS_ACK_SCHEMA
            or value.get("phase") != "ack"
            or value.get("attempt_id") != attempt_id
            or value.get("index") != index
            or value.get("event_sha256") != _progress_event_digest(event)):
        raise ProposalError("worker progress acknowledgment is missing or mismatched")


def _validate_worker_payload(payload: Any, expected: dict) -> tuple[list[dict], list[dict]]:
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "binding", "records", "zero_row_probe", "attempt_id"}:
        raise ProposalError("worker payload schema is invalid")
    binding = payload.get("binding")
    expected_payload_binding = expected["payload_binding"]
    if (payload.get("schema_version") != PAYLOAD_SCHEMA or not isinstance(binding, dict)
            or not isinstance(payload.get("attempt_id"), str)
            or not re.fullmatch(r"[0-9a-f]{32}", payload["attempt_id"])
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
    explicit_mode = expected.get("teacher_response_mode") == explicit_labels.SCHEMA_VERSION
    identity_keys = [_identity_token(c["field"], c["label"]) for c in columns]
    if (not labels or (not explicit_mode and len(labels) != len(set(labels)))
            or len(identity_keys) != len(set(identity_keys))):
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
                 exhaustive: bool, max_prompt_bytes: int,
                 progress_emit: Callable[[dict], None] | None = None) -> tuple[list[dict], list[dict], bool, str | None]:
    explicit_mode = config.get("teacher_response_mode") == explicit_labels.SCHEMA_VERSION
    labels = [f"{c['field']}:{c['label']}" for c in columns]
    allowed = set(labels)
    progress, result_rows = [], []
    aborted = False
    abort_reason = None
    for index, row in enumerate(records):
        extensions_present = False
        private_extensions: dict[str, Any] = {}
        if aborted:
            status, elapsed, positives = "not_attempted_after_abort", None, None
        else:
            try:
                request = _request(config, labels, row["text"], max_prompt_bytes)
                try:
                    inference = infer(request, allowed)
                    if explicit_mode and isinstance(inference, tuple) and len(inference) == 5:
                        status, positives, elapsed, extensions_present, private_extensions = inference
                    else:
                        status, positives, elapsed = inference
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
        if status == "answered" and positives is not None and explicit_mode:
            states = {_identity_token(c["field"], c["label"]): {
                "positive": "proposed_positive", "negative": "proposed_negative", "unknown": "unknown"
            }[positives.get((c["field"], c["label"]), "unknown")] for c in columns}
        elif status == "answered" and positives is not None:
            states = {label: ("proposed_positive" if label in positives else
                              "proposed_negative" if exhaustive else "unknown") for label in labels}
        elif explicit_mode:
            states = {_identity_token(c["field"], c["label"]): "unknown" for c in columns}
        else:
            states = {label: "unknown" for label in labels}
        conflicts = []
        if status != "not_attempted_after_abort":
            for i, label in enumerate(labels):
                target, known = row["targets"][i], row["known_mask"][i]
                key = _identity_token(columns[i]["field"], columns[i]["label"]) if explicit_mode else label
                if known and ((target == 0 and states[key] == "proposed_positive") or
                              (target == 1 and states[key] == "proposed_negative")):
                    conflicts.append({"field": columns[i]["field"], "label": columns[i]["label"]}
                                     if explicit_mode else label)
        item = {"row_id": row["id"], "family_sha256": row["family"],
                "text_sha256": row["text_sha256"], "original_text_sha256": row["original_text_sha256"],
                "screening": row["screening"], "verified_source_claim": row["verified_source_claim"],
                "source_identity": row["source_identity"], "columns": columns,
                "source_targets": row["targets"], "source_known_mask": row["known_mask"],
                "source_target_origins": row["target_origins"],
                "source_reviewer_claims": row["target_reviewer_claims"],
                "status": status, "elapsed_ms": elapsed, "proposal_states": states,
                "conflict_labels": conflicts, "provenance": "unreviewed_local_teacher_claim"}
        if explicit_mode:
            item["teacher_identity"] = _teacher_wire_identity(config)
        result_rows.append(item)
        event = {"phase": "progress", "index": index, "status": status,
                 "elapsed_ms": elapsed, "result": item}
        if explicit_mode:
            if not isinstance(private_extensions, dict):
                status, item["status"], event["status"] = "protocol_error", "protocol_error", "protocol_error"
                item["proposal_states"] = {_identity_token(c["field"], c["label"]): "unknown" for c in columns}
                item["conflict_labels"] = []
                item["elapsed_ms"] = None
                event["elapsed_ms"] = None
                extensions_present, private_extensions = False, {}
                aborted, abort_reason = True, "protocol_or_worker_lifecycle_ambiguous"
            extension_digest = (_private_extension_sha256(private_extensions) if extensions_present else None)
            event.update({"private_extensions_present": extensions_present,
                          "private_extensions_sha256": extension_digest,
                          "private_extensions": private_extensions})
        progress.append(event)
        if progress_emit is not None:
            progress_emit(event)
    return progress, result_rows, aborted, abort_reason


def _worker(config_path: Path, expected_config_sha256: str) -> None:
    _diagnostic_phase("worker_config_validation")
    cfg, config_sha = comparator.load_config(config_path)
    if config_sha != expected_config_sha256 or cfg.get("worker_code_sha256") != _sha(Path(__file__).read_bytes()):
        raise ProposalError("worker code or configuration digest mismatch")
    _diagnostic_phase("worker_receipt_validation")
    receipt, receipt_sha = comparator.verify_acquisition(cfg)
    _diagnostic_phase("worker_boundary_check")
    if (not sys.platform.startswith("linux") or not comparator._route_is_private()
            or os.geteuid() != cfg["remote_uid"]):
        raise ProposalError("worker is outside its configured private network namespace")
    _diagnostic_phase("worker_resource_snapshot")
    resources_before = comparator.resource_snapshot(cfg, enforce=True)
    _diagnostic_phase("worker_artifact_check")
    artifact = comparator.verify_installed_artifact(cfg, receipt)
    address, port, _ = comparator._loopback_endpoint(cfg["endpoint_url"])
    if address != "127.0.0.1":
        raise ProposalError("worker must use IPv4 loopback")
    _diagnostic_phase("worker_server_startup")
    server = subprocess.Popen([cfg["server_binary_path"], "serve"], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False, start_new_session=True,
        env={**os.environ, "OLLAMA_NO_CLOUD": "1", "OLLAMA_MODELS": cfg["model_store"],
             "OLLAMA_HOST": f"127.0.0.1:{port}"})
    response = None
    cleanup_verified = False
    worker_primary_error = None
    worker_primary_phase = None
    try:
        _diagnostic_phase("worker_warmup")
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
        _diagnostic_phase("worker_warmup")
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
                 "resources_before": resources_before,
                 "teacher_identity": _teacher_wire_identity(cfg["teacher_binding"])}
        print(json.dumps(ready, sort_keys=True, separators=(",", ":")), flush=True)
        _diagnostic_phase("worker_payload")
        raw_payload = sys.stdin.buffer.readline(cfg["max_payload_bytes"] + 1)
        if not raw_payload or len(raw_payload) > cfg["max_payload_bytes"] or not raw_payload.endswith(b"\n"):
            raise ProposalError("bounded worker payload is absent, oversized or incomplete")
        payload = _json(raw_payload, "worker payload")
        encoded_progress = []
        encoded_progress_bytes = 0
        def emit_progress(event: dict) -> None:
            nonlocal encoded_progress_bytes
            line = (json.dumps(event, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":")) + "\n").encode("utf-8")
            encoded_progress_bytes += len(line)
            if (len(line) > PROGRESS_EVENT_MAX_BYTES
                    or encoded_progress_bytes > cfg["teacher_binding"]["max_protocol_bytes"]):
                raise ProposalError("bounded worker progress protocol exceeded its configured limit")
            encoded_progress.append(line)
            print(line.decode("utf-8").rstrip("\n"), flush=True)
            ack_raw = sys.stdin.buffer.readline(PROGRESS_ACK_MAX_BYTES + 1)
            if (not ack_raw or len(ack_raw) > PROGRESS_ACK_MAX_BYTES
                    or not ack_raw.endswith(b"\n")):
                raise ProposalError("bounded worker progress acknowledgment is absent or incomplete")
            _validate_progress_ack(_json(ack_raw, "progress acknowledgment"),
                                   payload["attempt_id"], event["index"], event)
        if payload.get("zero_row_probe") is True:
            records, columns = _validate_worker_payload(payload, cfg["teacher_binding"])
            if records:
                raise ProposalError("zero-row probe unexpectedly contains records")
            progress, outputs, aborted, abort_reason = [], [], False, None
        else:
            records, columns = _validate_worker_payload(payload, cfg["teacher_binding"])
        if not payload["zero_row_probe"]:
            teacher_binding = cfg["teacher_binding"]
            loop_config = {**cfg, **teacher_binding}
            if teacher_binding.get("teacher_response_mode") == explicit_labels.SCHEMA_VERSION:
                infer = lambda body, _labels: comparator._prediction_http_explicit(
                    cfg, body, teacher_binding["selected_columns"], teacher_binding["taxonomy_sha256"],
                    explicit_labels, teacher_binding["max_teacher_extension_bytes"])
            else:
                infer = lambda body, labels: comparator._prediction_http(cfg, body, labels)
            progress, outputs, aborted, abort_reason = _worker_loop(records, columns, loop_config,
                infer,
                cfg["teacher_binding"]["assume_exhaustive_labels"],
                cfg["teacher_binding"]["max_prompt_bytes"], progress_emit=emit_progress)
        _diagnostic_phase("worker_result")
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
    except BaseException as exc:
        worker_primary_error = exc
        worker_primary_phase = _DIAGNOSTIC_PHASE
    finally:
        _diagnostic_phase("worker_cleanup")
        cleanup_failure = False
        try:
            children = comparator._descendant_pids(server.pid)
        except Exception:
            children = []
            cleanup_failure = True
        try:
            os.killpg(server.pid, 15)
        except ProcessLookupError:
            pass
        except Exception:
            cleanup_failure = True
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server.pid, 9)
            except ProcessLookupError:
                pass
            except Exception:
                cleanup_failure = True
            try:
                server.wait(timeout=5)
            except Exception:
                cleanup_failure = True
        except Exception:
            cleanup_failure = True
        try:
            lingering = [pid for pid in children if Path(f"/proc/{pid}").exists()]
            if not cleanup_failure and not lingering and server.poll() is not None:
                cleanup_verified = True
        except Exception:
            cleanup_failure = True
        if response is None:
            response = {"phase": "result", "schema_version": WORKER_RESULT_SCHEMA,
                        "worker_failed": True, "payload_sha256": None, "rows_received": 0,
                        "statuses": [], "rows": [], "aborted": True,
                        "abort_reason": "worker_failed_before_result", "cleanup_verified": cleanup_verified}
        response["cleanup_verified"] = cleanup_verified
        response["server_exit_code"] = server.returncode
        global _DIAGNOSTIC_CLEANUP
        _DIAGNOSTIC_CLEANUP = "succeeded" if cleanup_verified else "failed"
    if not cleanup_verified:
        if worker_primary_error is not None:
            _diagnostic_phase(worker_primary_phase or "unknown")
            raise worker_primary_error.with_traceback(worker_primary_error.__traceback__)
        raise ProposalError("task-owned worker server cleanup failed")
    if worker_primary_error is not None:
        _diagnostic_phase(worker_primary_phase or "unknown")
        raise worker_primary_error.with_traceback(worker_primary_error.__traceback__)
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
    if value.get("phase") == "worker_diagnostic":
        valid_categories = {"timeout", "transport_or_io_failure", "validation_or_protocol_failure", "internal_failure"}
        if (set(value) != {"phase", "schema_version", "worker_phase", "category", "cleanup"}
                or value.get("schema_version") != DIAGNOSTIC_SCHEMA
                or value.get("worker_phase") not in DIAGNOSTIC_PHASES
                or not value.get("worker_phase", "").startswith("worker_")
                or value.get("category") not in valid_categories
                or value.get("cleanup") not in {"not_started", "succeeded", "failed"}):
            raise ProposalError("isolated worker diagnostic protocol is invalid")
        raise WorkerDiagnosticError(value["worker_phase"], value["category"], value["cleanup"])
    return value


def _worker_warmup_handshake(process: subprocess.Popen, lines: queue.Queue, comp: dict,
                             host_before: dict, adapter_sha: str, worker_config_sha: str,
                             receipt_sha: str, teacher_identity: dict | None = None) -> dict:
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
    _diagnostic_phase("outside_port_denial")
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
    _diagnostic_phase("warmup_readiness")
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
            or (teacher_identity is not None and ready.get("teacher_identity") != teacher_identity)
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
                              exhaustive: bool, teacher_identity: dict | None = None) -> dict:
    item = event.get("result")
    expected_keys = {"row_id", "family_sha256", "text_sha256", "original_text_sha256", "screening",
        "verified_source_claim", "source_identity", "columns", "source_targets", "source_known_mask",
        "source_target_origins", "source_reviewer_claims", "status", "elapsed_ms", "proposal_states",
        "conflict_labels", "provenance"}
    explicit_mode = (teacher_identity or {}).get("mode") == explicit_labels.SCHEMA_VERSION
    if explicit_mode:
        expected_keys.add("teacher_identity")
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
    if explicit_mode and item.get("teacher_identity") != teacher_identity:
        raise ProposalError("worker progress explicit-label identity differs from the pinned contract")
    ordered_labels = [f"{column['field']}:{column['label']}" for column in columns]
    labels = set(ordered_labels)
    states = item.get("proposal_states")
    if explicit_mode:
        expected_state_keys = {_identity_token(c["field"], c["label"]) for c in columns}
    else:
        expected_state_keys = labels
    if (not isinstance(states, dict) or set(states) != expected_state_keys
            or any(value not in {"proposed_positive", "proposed_negative", "unknown"} for value in states.values())):
        raise ProposalError("worker proposal states differ from the pinned taxonomy")
    if status != "answered" and any(value != "unknown" for value in states.values()):
        raise ProposalError("non-answer worker status must leave every proposal unknown")
    if status == "answered" and exhaustive and any(value == "unknown" for value in states.values()):
        raise ProposalError("exhaustive answered rows must explicitly mark every label")
    if status == "answered" and not exhaustive and not explicit_mode and any(value == "proposed_negative" for value in states.values()):
        raise ProposalError("omitted labels cannot become negative without the explicit exhaustive policy")
    expected_conflicts = []
    if status != "not_attempted_after_abort":
        for index, label in enumerate(ordered_labels):
            target, known = row["targets"][index], row["known_mask"][index]
            key = _identity_token(columns[index]["field"], columns[index]["label"]) if explicit_mode else label
            if known and ((target == 0 and states[key] == "proposed_positive")
                          or (target == 1 and states[key] == "proposed_negative")):
                expected_conflicts.append({"field": columns[index]["field"], "label": columns[index]["label"]}
                                          if explicit_mode else label)
    if item.get("conflict_labels") != expected_conflicts:
        raise ProposalError("worker conflicts do not match source masks and proposal states")
    return item


def _validate_progress_event(event: dict, index: int, row: dict, columns: list[dict],
                             exhaustive: bool, teacher_identity: dict | None = None,
                             max_extension_bytes: int = explicit_labels.MAX_EXTENSION_BYTES) -> dict:
    statuses = {"answered", "abstain", "malformed", "invalid_schema", "refusal", "truncated",
                "incomplete", "identity_mismatch", "http_error", "timeout", "transport_error",
                "protocol_error", "request_too_large_not_started", "not_attempted_after_abort"}
    status = event.get("status") if isinstance(event, dict) else None
    elapsed = event.get("elapsed_ms") if isinstance(event, dict) else None
    unknown_elapsed = {"timeout", "transport_error", "protocol_error",
                       "request_too_large_not_started", "not_attempted_after_abort"}
    explicit_mode = (teacher_identity or {}).get("mode") == explicit_labels.SCHEMA_VERSION
    expected_event_keys = {"phase", "index", "status", "elapsed_ms", "result"}
    if explicit_mode:
        expected_event_keys |= {"private_extensions_present", "private_extensions_sha256", "private_extensions"}
    if (not isinstance(event, dict) or set(event) != expected_event_keys
            or event.get("phase") != "progress" or event.get("index") != index
            or status not in statuses
            or (status in unknown_elapsed and elapsed is not None)
            or (status not in unknown_elapsed and
                (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0))):
        raise ProposalError("worker progress status or timing protocol is invalid")
    if explicit_mode:
        present = event.get("private_extensions_present")
        extensions = event.get("private_extensions")
        digest = event.get("private_extensions_sha256")
        if type(present) is not bool or not isinstance(extensions, dict):
            raise ProposalError("worker private extension envelope is invalid")
        if present:
            if (status not in {"answered", "abstain"}
                    or not isinstance(digest, str) or not HEX.fullmatch(digest)):
                raise ProposalError("private extensions are not bound to an answered or abstained response")
            try:
                extension_bytes = explicit_labels._canonical_bytes(extensions)
            except explicit_labels.ContractError:
                raise ProposalError("private extension data is invalid") from None
            if len(extension_bytes) > max_extension_bytes or _sha(extension_bytes) != digest:
                raise ProposalError("private extension digest or size differs from the response")
        elif extensions or digest is not None:
            raise ProposalError("absent extensions must remain distinguishable from present values")
    return _validate_progress_result(event, index, row, columns, exhaustive, teacher_identity)


def _atomic_private_publish(final_path: Path, proposals: list[dict], manifest: dict,
                            max_output_bytes: int | None = None) -> tuple[Path, str]:
    if final_path.is_symlink():
        raise ProposalError("final proposal directory cannot be a symlink")
    final_path = final_path.resolve()
    if final_path.is_symlink() or not _outside_git(final_path):
        raise ProposalError("final proposal directory must be outside Git and not a symlink")
    parent = final_path.parent
    if not parent.exists() or not _private_dir(parent):
        raise ProposalError("proposal output parent must already be owner-private")
    publish_status = manifest.get("status")
    if publish_status not in {"complete", "partial_aborted_ambiguous"}:
        raise ProposalError("proposal publication status is not a recognized terminal state")
    payload = ("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                          for row in proposals) + "\n").encode("utf-8")
    if max_output_bytes is not None and len(payload) > max_output_bytes:
        raise ProposalError("encoded proposals exceed configured output-byte bound")
    event_sha = manifest.get("provenance", {}).get("progress_store_events_sha256")
    publication_id = _sha(_canonical_json({"proposal_sha256": _sha(payload),
        "progress_events_sha256": event_sha, "status": publish_status}).rstrip(b"\n"))
    manifest = {**manifest, "publication_id": publication_id}
    complete_manifest = {**manifest, "status": publish_status, "proposals_sha256": _sha(payload),
                         "proposals_bytes": len(payload), "rows_written": len(proposals)}
    desired_raw = (json.dumps(complete_manifest, indent=2, sort_keys=True,
                              ensure_ascii=False, allow_nan=False) + "\n").encode()

    def safe_dir(path: Path) -> bool:
        return (not path.is_symlink() and path.is_dir()
                and comparator._verify_private_report_directory(path)
                and _outside_git(path))

    def file_bytes(path: Path, limit: int, label: str) -> bytes:
        try:
            info = path.stat(follow_symlinks=False)
        except OSError:
            raise ProposalError(f"{label} is unsafe") from None
        if (path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or not _private_file(path)):
            raise ProposalError(f"{label} is unsafe")
        return _read(path, limit, label)

    def validate_revision_reservation(directory: Path) -> bool:
        reservation_path = directory / "revision-reservation.json"
        if not reservation_path.exists() and not reservation_path.is_symlink():
            return False
        raw = file_bytes(reservation_path, 65_536, "report revision reservation")
        record = _json(raw, "report revision reservation")
        required = {"schema_version", "sequence", "revision_id", "binding_sha256",
                    "parent_output_name", "parent_manifest_sha256", "parent_proposals_sha256",
                    "target_output_name", "record_sha256"}
        if not isinstance(record, dict) or set(record) != required:
            raise ProposalError("report revision reservation schema is invalid")
        base = {key: value for key, value in record.items() if key != "record_sha256"}
        reference = {key: record[key] for key in ("revision_id", "sequence", "parent_output_name",
            "parent_manifest_sha256", "parent_proposals_sha256")}
        if (record["schema_version"] != PROGRESS_REVISION_SCHEMA
                or record["record_sha256"] != _sha(_canonical_json(base).rstrip(b"\n"))
                or manifest.get("provenance", {}).get("publication_revision") != reference):
            raise ProposalError("report revision reservation does not match publication input")
        return True

    def validate_complete(raw: bytes) -> dict:
        value = _json(raw, "completed proposal manifest")
        if not isinstance(value, dict):
            raise ProposalError("persisted publication manifest is invalid")
        provenance = value.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ProposalError("persisted publication source identity is invalid")
        if value != complete_manifest:
            raise ProposalError("persisted full publication manifest does not match trusted task evidence")
        return value

    def validate_proposal_file(directory: Path) -> None:
        if file_bytes(directory / "proposals.jsonl", max_output_bytes or PROGRESS_STORE_HARD_MAX_BYTES,
                      "published proposal bytes") != payload:
            raise ProposalError("persisted proposal bytes do not match durable task evidence")

    def atomic_new(directory: Path, target_name: str, prefix: str, raw: bytes) -> Path:
        temp = directory / f".{prefix}-{uuid.uuid4().hex}.tmp"
        _write_new(temp, raw)
        os.replace(temp, directory / target_name)
        _sync_directory(directory)
        return directory / target_name

    marker_evidence_prefix = f".manifest-incomplete-{publication_id}-"
    proposal_evidence_prefix = f".proposals-{publication_id}-"
    complete_evidence_prefix = f".manifest-complete-{publication_id}-"

    def validate_staging_evidence(directory: Path, names: set[str]) -> set[str]:
        evidence = [name for name in names if name.startswith((marker_evidence_prefix,
            proposal_evidence_prefix, complete_evidence_prefix)) and name.endswith(".tmp")]
        if len(evidence) > PUBLICATION_TEMP_EVIDENCE_MAX:
            raise ProposalError("publication contains too many retained staging records")
        for name in evidence:
            path = directory / name
            if name.startswith(marker_evidence_prefix):
                raw = file_bytes(path, 1_048_576, "retained incomplete marker")
                if not raw or not marker_raw_expected.startswith(raw):
                    raise ProposalError("retained incomplete marker evidence is altered")
            elif name.startswith(proposal_evidence_prefix):
                raw = file_bytes(path, max_output_bytes or PROGRESS_STORE_HARD_MAX_BYTES,
                                 "retained proposal staging bytes")
                if not raw or not payload.startswith(raw):
                    raise ProposalError("retained proposal staging evidence is altered")
            else:
                raw = file_bytes(path, PROGRESS_TERMINAL_MAX_BYTES,
                                 "retained complete-manifest staging bytes")
                try:
                    _json(raw, "retained complete-manifest staging bytes")
                except ProposalError:
                    if not raw or not desired_raw.startswith(raw):
                        raise ProposalError("retained complete-manifest staging evidence is altered") from None
                else:
                    validate_complete(raw)
        return set(evidence)

    marker_raw_expected = _canonical_json({"schema_version": REPORT_SCHEMA, "status": "incomplete",
        "promotion_eligible": False, "publication_id": publication_id})

    reserved_revision_dir = False
    if final_path.exists():
        if safe_dir(final_path):
            initial_names = {p.name for p in final_path.iterdir()}
            if ("revision-reservation.json" in initial_names
                    and "manifest.json" not in initial_names
                    and "proposals.jsonl" not in initial_names):
                validate_revision_reservation(final_path)
                validate_staging_evidence(final_path, initial_names)
                if not initial_names.issubset({"revision-reservation.json"} |
                        validate_staging_evidence(final_path, initial_names)):
                    raise ProposalError("reserved report revision contains foreign entries")
                reserved_revision_dir = True
    if final_path.exists() and not reserved_revision_dir:
        if not safe_dir(final_path):
            raise ProposalError("existing proposal directory is unsafe")
        names = {p.name for p in final_path.iterdir()}
        has_revision_reservation = validate_revision_reservation(final_path)
        marker_path = final_path / "manifest.json"
        marker_raw = file_bytes(marker_path, PROGRESS_TERMINAL_MAX_BYTES, "proposal manifest")
        marker_value = _json(marker_raw, "proposal manifest")
        if not isinstance(marker_value, dict):
            raise ProposalError("proposal manifest is invalid")
        retained = validate_staging_evidence(final_path, names)
        if not names.issubset({"manifest.json", "proposals.jsonl", "revision-reservation.json"} | retained):
            raise ProposalError("existing proposal directory has foreign or unknown entries")
        if marker_value.get("status") != "incomplete":
            if not {"manifest.json", "proposals.jsonl"}.issubset(names):
                raise ProposalError("completed proposal directory is missing required files")
            validate_proposal_file(final_path)
            validate_complete(marker_raw)
            return final_path, _sha(marker_raw)
        marker = _json(marker_raw, "incomplete proposal marker")
        if marker != {"schema_version": REPORT_SCHEMA, "status": "incomplete",
                      "promotion_eligible": False, "publication_id": publication_id}:
            raise ProposalError("existing proposal marker does not match recoverable publication")
        if "proposals.jsonl" in names:
            validate_proposal_file(final_path)
        else:
            atomic_new(final_path, "proposals.jsonl", "proposals", payload)
        manifests = []
        for name in sorted(names):
            if name.startswith(complete_evidence_prefix) and name.endswith(".tmp"):
                raw = file_bytes(final_path / name, PROGRESS_TERMINAL_MAX_BYTES,
                                 "pending publication manifest")
                try:
                    parsed = _json(raw, "pending publication manifest")
                except ProposalError:
                    # A process may have died while writing this staging file.
                    # Keep it as evidence and rebuild from the durable terminal.
                    continue
                validate_complete(raw)
                manifests.append((name, raw))
        if manifests and len({raw for _name, raw in manifests}) != 1:
            raise ProposalError("multiple different pending publication manifests are ambiguous")
        raw_manifest = manifests[0][1] if manifests else desired_raw
        marker_evidence = final_path / f"{marker_evidence_prefix}{uuid.uuid4().hex}.tmp"
        _write_new(marker_evidence, marker_raw)
        _sync_directory(final_path)
        temp_path = final_path / f"{complete_evidence_prefix}{uuid.uuid4().hex}.tmp"
        _write_new(temp_path, raw_manifest)
        _sync_directory(final_path)
        os.replace(temp_path, marker_path)
        _sync_directory(final_path)
        return final_path, _sha(raw_manifest)

    already_final = reserved_revision_dir
    partial = final_path if reserved_revision_dir else parent / f".{final_path.name}.partial-{publication_id}"
    if already_final:
        pass
    elif partial.exists():
        if not safe_dir(partial):
            raise ProposalError("matching partial publication directory is unsafe")
    else:
        partial = comparator._private_report_directory(partial)
        if not safe_dir(partial):
            raise ProposalError("new partial publication directory is not owner-private")
        _sync_directory(parent)
    names = {p.name for p in partial.iterdir()}
    retained = validate_staging_evidence(partial, names)
    allowed = {"manifest.json", "proposals.jsonl"} | retained
    if already_final:
        allowed.add("revision-reservation.json")
    if not names.issubset(allowed):
        raise ProposalError("partial publication contains unexpected entries")
    marker_path = partial / "manifest.json"
    if marker_path.exists():
        marker = _json(file_bytes(marker_path, 1_048_576, "partial publication marker"),
                       "partial publication marker")
        if marker != {"schema_version": REPORT_SCHEMA, "status": "incomplete",
                      "promotion_eligible": False, "publication_id": publication_id}:
            raise ProposalError("partial publication marker identity does not match")
    else:
        atomic_new(partial, "manifest.json", f"manifest-incomplete-{publication_id}",
                   marker_raw_expected)
    if (partial / "proposals.jsonl").exists():
        validate_proposal_file(partial)
    else:
        # Never consume a retained staged file: it records what survived the
        # earlier interruption. Publish from the durable row evidence again.
        atomic_new(partial, "proposals.jsonl", f"proposals-{publication_id}", payload)
    manifest_temps = []
    for path in partial.iterdir():
        if path.name.startswith(".manifest-complete-") and path.name.endswith(".tmp"):
            raw = file_bytes(path, PROGRESS_TERMINAL_MAX_BYTES, "pending publication manifest")
            try:
                _json(raw, "pending publication manifest")
            except ProposalError:
                # Retain interrupted staging bytes; a later valid temp can be
                # written without discarding this evidence.
                continue
            validate_complete(raw)
            manifest_temps.append((path, raw))
    if manifest_temps and len({raw for _path, raw in manifest_temps}) != 1:
        raise ProposalError("multiple different pending publication manifests are ambiguous")
    raw_manifest = manifest_temps[0][1] if manifest_temps else desired_raw
    # Keep any interrupted complete-manifest staging files as evidence. A
    # fresh publication temp is the one atomically promoted to the marker.
    temp_manifest = partial / f"{complete_evidence_prefix}{uuid.uuid4().hex}.tmp"
    _write_new(temp_manifest, raw_manifest)
    _sync_directory(partial)
    if final_path.exists() and not already_final:
        raise ProposalError("final proposal directory appeared during publication")
    if not already_final:
        os.rename(partial, final_path)
        _sync_directory(parent)
    os.replace(final_path / temp_manifest.name, final_path / "manifest.json")
    _sync_directory(final_path)
    return final_path, _sha(raw_manifest)



def _remote_stage_paths(comp: dict, workdir: str, unit_id: str) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{32}", unit_id):
        raise ProposalError("invalid task identifier")
    paths = {"workdir": workdir,
             "teacher": f"{workdir}/train_teacher_proposals.py",
             "comparator": f"{workdir}/compare_local_12b.py",
             "explicit_labels": f"{workdir}/teacher_explicit_labels.py",
             "config": f"{workdir}/worker-config.json",
             "model_store": f"{workdir}/models"}
    if not comparator._within_remote_home(workdir, comp["remote_home"]):
        raise ProposalError("task staging path is outside the configured remote home")
    if any(not comparator._within_remote_home(paths[k], comp["remote_home"])
           for k in ("teacher", "comparator", "explicit_labels", "config", "model_store")):
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
            "remote_explicit_labels_path": paths["explicit_labels"],
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
            "e=d/'teacher_explicit_labels.py'; "
            "assert hashlib.sha256(pathlib.Path(a).read_bytes()).hexdigest()==cfg['worker_code_sha256']; "
            "assert hashlib.sha256(q.read_bytes()).hexdigest()==cfg['comparator_code_sha256']; "
            "assert hashlib.sha256(e.read_bytes()).hexdigest()==cfg['teacher_binding']['explicit_labels_module_sha256']; "
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
                               receipt_sha: str, config_identity_sha256: str | None = None) -> None:
    identity = probe.get("model_identity") if isinstance(probe, dict) else None
    probe_host = probe.get("host_before", {}) if isinstance(probe, dict) else None
    boundary = probe.get("runtime_boundary", {}) if isinstance(probe, dict) else None
    artifact = probe.get("artifact") if isinstance(probe, dict) else None
    if (not isinstance(probe, dict) or probe.get("zero_row_probe") is not True
            or probe.get("cleanup_verified") is not True
            or probe.get("outside_listener_denied") is not True
            or probe.get("persistent_service_unchanged") is not True
            or probe.get("proposal_config_sha256") != cfg_sha
            or (config_identity_sha256 is not None
                and probe.get("proposal_config_identity_sha256") != config_identity_sha256)
            or probe.get("comparison_config_sha256") != comp_sha
            or probe.get("adapter_code_sha256") != adapter_sha
            or probe.get("comparator_code_sha256") != comparator_sha
            or probe.get("teacher_identity") != ready.get("teacher_identity")
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
                          zero_rows: bool, expected_probe: dict | None = None,
                          resume: bool = False, publication_path: Path | None = None,
                          publication_revision: dict | None = None) -> dict:
    global _DIAGNOSTIC_CLEANUP
    _DIAGNOSTIC_CLEANUP = "not_started"
    _diagnostic_phase("remote_snapshot")
    host_before = _remote_identity_snapshot(comp)
    if (host_before["memory_available_bytes"] < comp["min_memory_available_bytes"]
            or host_before["home_disk_free_bytes"] < comp["min_disk_free_bytes"]
            or type(host_before.get("loaded_model_count")) is not int):
        raise ProposalError("remote host identity, capacity or service coexistence preflight failed")
    _diagnostic_phase("port_check")
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
    _diagnostic_phase("transient_preflight")
    if comparator._ssh_run(comp, preflight_script, timeout=40).returncode != 0:
        raise ProposalError("transient cgroup resource preflight failed")

    adapter_path = Path(__file__).resolve(strict=True)
    comparator_path = Path(comparator.__file__).resolve(strict=True)
    explicit_path = Path(explicit_labels.__file__).resolve(strict=True)
    adapter_raw, comparator_raw, explicit_raw = adapter_path.read_bytes(), comparator_path.read_bytes(), explicit_path.read_bytes()
    adapter_sha, comparator_sha, explicit_sha = _sha(adapter_raw), _sha(comparator_raw), _sha(explicit_raw)
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
        "teacher_response_mode": cfg.get("teacher_response_mode", "legacy_positive"),
        "teacher_prompt_contract": cfg.get("teacher_prompt_contract"),
        "max_teacher_extension_bytes": cfg.get("max_teacher_extension_bytes", explicit_labels.MAX_EXTENSION_BYTES),
        "explicit_labels_module_sha256": explicit_sha,
        "proposal_config_sha256": cfg_sha, "comparison_config_sha256": comp_sha,
        "adapter_code_sha256": adapter_sha, "comparator_code_sha256": comparator_sha,
        "zero_row_probe": bool(zero_rows)}
    teacher_identity = _teacher_wire_identity(cfg, explicit_sha)
    teacher_binding["teacher_identity"] = teacher_identity
    payload_binding = {"manifest_sha256": teacher_binding["export_manifest_sha256"],
        "train_sha256": teacher_binding["train_sha256"], "taxonomy_sha256": teacher_binding["taxonomy_sha256"],
        "config_sha256": teacher_binding["export_config_sha256"],
        "proposal_config_sha256": cfg_sha, "comparison_config_sha256": comp_sha,
        "adapter_code_sha256": adapter_sha, "comparator_code_sha256": comparator_sha,
        "teacher_identity": teacher_identity}
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
    worker_progress_events: list[dict] = []
    stored_progress: dict[int, dict] = {}
    progress_store: Path | None = None
    progress_attempt_id = run_id
    checkpoint_binding = None
    sent_payload_sha = None
    result = None
    installed = None
    cleanup_error = None
    primary_error = None
    primary_phase = None
    try:
        _diagnostic_phase("remote_task_directory")
        _remote_create_workdir(comp, workdir)
        work_created = True
        _diagnostic_phase("local_temp_directory")
        local_config_dir = _private_temp_directory("lfg-teacher-worker-config")
        worker_config_path_local = local_config_dir / "worker-config.json"
        _diagnostic_phase("local_config_file")
        _write_new(worker_config_path_local, worker_config_raw)
        _masked()._verify_owner_private_access(worker_config_path_local, is_directory=False)
        _diagnostic_phase("file_staging")
        for local_path, remote_path, raw in ((adapter_path, paths["teacher"], adapter_raw),
                                              (comparator_path, paths["comparator"], comparator_raw),
                                              (explicit_path, paths["explicit_labels"], explicit_raw),
                                              (worker_config_path_local, paths["config"], worker_config_raw)):
            argv = ["scp", "-oBatchMode=yes", "-oConnectTimeout=10", "-P", str(comp["ssh_port"]),
                    str(local_path), f"{comp['remote_user']}@{comp['remote_host']}:{remote_path}"]
            copied = subprocess.run(argv, check=False, capture_output=True, timeout=120, shell=False)
            if copied.returncode != 0:
                raise ProposalError("bounded private worker file transfer failed")
        _diagnostic_phase("file_verification")
        _remote_verify_files(comp, paths, {paths["teacher"]: (adapter_sha, len(adapter_raw)),
            paths["comparator"]: (comparator_sha, len(comparator_raw)),
            paths["explicit_labels"]: (explicit_sha, len(explicit_raw)),
            paths["config"]: (worker_config_sha, len(worker_config_raw))})
        install_config["remote_adapter_path"] = paths["comparator"]
        _diagnostic_phase("model_install")
        installed = comparator._remote_install(install_config, paths["config"])
        if installed.get("receipt_sha256") != local_receipt_sha:
            raise ProposalError("remote model receipt differs from locally pinned receipt")
        if not comparator._remote_port_closed(comp):
            raise ProposalError("outside-host loopback port became occupied before worker start")
        _diagnostic_phase("worker_startup")
        command = _worker_command(comp, paths["teacher"], paths["config"], worker_config_sha, unit)
        process = subprocess.Popen(["ssh", "-T", "-oBatchMode=yes", "-oConnectTimeout=10", "-p", str(comp["ssh_port"]),
            f"{comp['remote_user']}@{comp['remote_host']}", command], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=False)
        lines = comparator._queue_protocol(process)
        _diagnostic_phase("control_handshake")
        ready = _worker_warmup_handshake(process, lines, comp, host_before, adapter_sha,
                                         worker_config_sha, local_receipt_sha, teacher_identity)
        _diagnostic_phase("warmup_readiness")
        if ready.get("model_manifest_sha256") != comp["expected_manifest_sha256"]:
            raise ProposalError("warmed model identity differs from the pinned expected manifest")
        if not zero_rows:
            _validate_zero_probe_match(expected_probe, ready, host_before, cfg_sha, comp_sha,
                                       adapter_sha, comparator_sha, local_receipt_sha,
                                       _proposal_config_identity_sha(cfg))
        _diagnostic_phase("payload_result")
        if zero_rows:
            payload = {"schema_version": PAYLOAD_SCHEMA, "binding": payload_binding,
                       "records": [], "zero_row_probe": True, "attempt_id": progress_attempt_id}
            selected_ids = []
        else:
            assert manifest is not None and rows is not None and binding is not None
            _reverify_export(cfg, binding, comp["max_manifest_bytes"])
            payload = {"schema_version": PAYLOAD_SCHEMA, "binding": payload_binding,
                       "records": [_payload_row(row) for row in rows], "zero_row_probe": False}
            selected_ids = [row["id"] for row in rows]
            checkpoint_binding = _progress_binding(cfg, cfg_sha, comp, comp_sha, rows, binding,
                                                   adapter_sha, comparator_sha, ready)
            progress_store = _progress_store_path(Path(cfg["output_dir"]), allow_final=resume)
            if resume:
                _records, stored_progress, progress_metadata = _progress_store_read(
                    progress_store, checkpoint_binding, rows, cfg,
                    config_sha256=cfg_sha, return_metadata=True)
                original_config_sha256 = progress_metadata["proposal_config_sha256"] or cfg_sha
            else:
                _progress_store_create(progress_store, checkpoint_binding,
                                        cfg["max_progress_store_bytes"], cfg_sha)
                original_config_sha256 = cfg_sha
            pending_indices = _progress_pending_indices(rows, stored_progress)
            if not pending_indices:
                raise ProposalError("durable progress is complete; publication recovery is required")
            pending_rows = [rows[i] for i in pending_indices]
            capacity_bound = _check_result_bounds(pending_rows, cfg["selected_columns"],
                cfg["max_output_bytes"], cfg["max_protocol_bytes"], teacher_identity,
                cfg.get("max_teacher_extension_bytes", explicit_labels.MAX_EXTENSION_BYTES),
                comp["model_name"])
            if (_progress_store_usage(progress_store) + capacity_bound["progress_store_bytes"]
                    > cfg["max_progress_store_bytes"]):
                raise ProposalError("progress store capacity cannot cover a full attempted batch")
            payload = {"schema_version": PAYLOAD_SCHEMA, "binding": payload_binding,
                       "records": [_payload_row(row) for row in pending_rows], "zero_row_probe": False,
                       "attempt_id": progress_attempt_id}
            selected_ids = [row["id"] for row in rows]
        payload_raw = (json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        if len(payload_raw) > comp["max_payload_bytes"]:
            raise ProposalError("source-bound train payload exceeds the configured transfer bound")
        sent_payload_sha = _sha(payload_raw)
        assert process.stdin is not None
        try:
            if not zero_rows:
                assert progress_store is not None and checkpoint_binding is not None
                _progress_store_record_attempt(progress_store, progress_attempt_id,
                                               pending_indices, checkpoint_binding,
                                               cfg["max_progress_store_bytes"])
            process.stdin.write(payload_raw)
            process.stdin.flush()
            if zero_rows:
                process.stdin.close()
        except OSError:
            raise ProposalError("worker input transport failed after readiness") from None
        count = 0 if zero_rows else len(pending_rows)
        batch_timeout = float(comp["timeout_seconds"]) * (count + 2) + comp["batch_overhead_margin_seconds"]
        if not math.isfinite(batch_timeout) or batch_timeout > sys.maxsize / 1_000_000_000:
            raise ProposalError("batch timeout exceeds safe platform arithmetic")
        deadline = time.monotonic() + batch_timeout
        expected_rows = pending_rows if not zero_rows else []
        for worker_index in range(count):
            event = _read_worker_line(process, lines, max(0.001, deadline - time.monotonic()))
            row = expected_rows[worker_index]
            original_index = pending_indices[worker_index]
            assert progress_store is not None
            item = _checkpoint_worker_event(progress_store, progress_attempt_id, worker_index,
                original_index, row, event, cfg["selected_columns"], cfg["assume_exhaustive_labels"],
                cfg["max_progress_store_bytes"], teacher_identity,
                cfg.get("max_teacher_extension_bytes", explicit_labels.MAX_EXTENSION_BYTES),
                checkpoint_binding)
            ack_raw = (_canonical_json(_progress_ack(progress_attempt_id, worker_index, event)))
            if len(ack_raw) > PROGRESS_ACK_MAX_BYTES:
                raise ProposalError("progress acknowledgment exceeds protocol limit")
            try:
                process.stdin.write(ack_raw)
                process.stdin.flush()
            except OSError:
                raise ProposalError("progress acknowledgment transport failed") from None
            worker_progress_events.append(event)
            progress_rows.append(item)
        if not zero_rows:
            assert process.stdin is not None
            process.stdin.close()
        progress_digest = _sha(b"".join(
            (json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            for event in worker_progress_events))
        result = _read_worker_line(process, lines, max(0.001, deadline - time.monotonic()))
        return_code = process.wait(timeout=30)
        try:
            trailing = lines.get(timeout=5)
        except queue.Empty:
            raise ProposalError("worker protocol did not terminate cleanly") from None
        if trailing is not None:
            raise ProposalError("worker emitted protocol data after its terminal result")
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
        _diagnostic_phase("host_recheck")
        host_after = comparator._remote_snapshot(comp)
        if (host_after["ollama_service_pid"] != host_before["ollama_service_pid"]
                or host_after["ollama_version"] != host_before["ollama_version"]
                or host_after["loaded_model_count"] != host_before["loaded_model_count"]):
            raise ProposalError("persistent host Ollama state changed during task-owned run")
    except BaseException as exc:
        primary_error = exc
        primary_phase = _DIAGNOSTIC_PHASE
    finally:
        _diagnostic_phase("cleanup")
        _DIAGNOSTIC_CLEANUP = "succeeded"
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=15)
            except Exception:
                cleanup_error = "task-owned worker process cleanup failed"
                _DIAGNOSTIC_CLEANUP = "failed"
        if work_created:
            try:
                _remote_cleanup_worker(comp, workdir, unit)
            except Exception:
                cleanup_error = "task-owned remote cleanup failed"
                _DIAGNOSTIC_CLEANUP = "failed"
        if 'local_config_dir' in locals():
            try:
                shutil.rmtree(local_config_dir)
            except Exception:
                cleanup_error = "private local worker-config cleanup failed"
                _DIAGNOSTIC_CLEANUP = "failed"
    if primary_error is not None:
        _diagnostic_phase(primary_phase or "unknown")
        raise primary_error.with_traceback(primary_error.__traceback__)
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
                "proposal_config_sha256": cfg_sha,
                "proposal_config_identity_sha256": _proposal_config_identity_sha(cfg),
                "comparison_config_sha256": comp_sha,
                "adapter_code_sha256": adapter_sha, "comparator_code_sha256": comparator_sha,
                "teacher_identity": ready.get("teacher_identity")}
    assert manifest is not None and rows is not None and binding is not None
    assert progress_store is not None
    for original_index, item in zip(pending_indices, progress_rows):
        stored_progress[original_index] = item
    progress_rows = [stored_progress[index] for index in range(len(rows))]
    _diagnostic_phase("source_recheck")
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
    progress_events_sha256 = _sha(_read(
        progress_store / "events.jsonl",
        cfg["max_progress_store_bytes"],
        "progress event log"))
    output_manifest = {"schema_version": REPORT_SCHEMA, "created_at_epoch_seconds": int(time.time()),
        "status": "complete" if not result.get("aborted") else "partial_aborted_ambiguous",
        "promotion_eligible": False, "claims": {"teacher_suggestions_are_unreviewed": True,
            "heldout_accessed": False, "labels_independently_verified": False,
            "screening_independently_verified": False, "promotion_eligible": False},
        "provenance": {"proposal_config_sha256": cfg_sha,
            "proposal_config_origin_sha256": original_config_sha256,
            "proposal_config_identity_sha256": _proposal_config_identity_sha(cfg),
            "publication_revision": publication_revision,
            "comparison_config_sha256": comp_sha,
            "adapter_code_sha256": adapter_sha, "comparator_code_sha256": comparator_sha,
            "worker_config_sha256": worker_config_sha, "payload_sha256": sent_payload_sha,
            "export_manifest_sha256": binding["manifest_sha256"], "export_config_sha256": binding["config_sha256"],
            "train_file_sha256": binding["train_sha256"], "train_file_bytes": binding["train_bytes"],
            "taxonomy_sha256": cfg["taxonomy_sha256"], "source_lineage": source_lineage,
            "selected_row_count": len(rows), "selected_row_ids_sha256": _sha(json.dumps(selected_ids, separators=(",", ":")).encode()),
            "selected_text_digests_sha256": _sha(json.dumps([r["text_sha256"] for r in rows], separators=(",", ":")).encode()),
            "target_mask_sha256": _sha(json.dumps([r["known_mask"] for r in rows], separators=(",", ":")).encode()),
            "proposal_schema_sha256": _proposal_schema_digest(cfg),
            "teacher_identity": _teacher_wire_identity(cfg),
            "prompt_contract_sha256": _sha((comp["system_prompt"] + "\0" + comp["prompt_revision"]).encode()),
            "progress_store_schema": PROGRESS_STORE_SCHEMA,
            "progress_store_id_sha256": _sha(_canonical_json(checkpoint_binding).rstrip(b"\n")),
            "progress_store_events_sha256": progress_events_sha256,
            "progress_resumed": resume,
            "terminal_worker_result_verified": True,
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
    _diagnostic_phase("publication")
    _progress_store_write_terminal(progress_store, progress_attempt_id, checkpoint_binding,
                                   progress_events_sha256, output_manifest,
                                   cfg["max_progress_store_bytes"])
    final_path, manifest_sha = _atomic_private_publish(publication_path or Path(cfg["output_dir"]), progress_rows,
                                                        output_manifest, cfg["max_output_bytes"])
    return {"rows_selected": len(rows), "rows_completed": len(progress_rows),
            "manifest_sha256": manifest_sha, "output_directory": str(final_path),
            "status_counts": status_counts, "promotion_eligible": False}


def _resume_published_progress(cfg: dict, cfg_sha: str, comp: dict, comp_sha: str,
                               manifest: dict, rows: list[dict], source_binding: dict,
                               probe: dict) -> dict | None:
    adapter_sha = _sha(Path(__file__).resolve(strict=True).read_bytes())
    comparator_sha = _sha(Path(comparator.__file__).resolve(strict=True).read_bytes())
    checkpoint_binding = _progress_binding(cfg, cfg_sha, comp, comp_sha, rows,
                                           source_binding, adapter_sha, comparator_sha, probe)
    store = _progress_store_path(Path(cfg["output_dir"]), allow_final=True)
    _records, latest, config_history = _progress_store_read(store, checkpoint_binding, rows, cfg,
        config_sha256=cfg_sha, return_metadata=True, update_capacity=False)
    pending = bool(_progress_pending_indices(rows, latest))
    events_sha = _sha(_read(store / "events.jsonl",
                            cfg["max_progress_store_bytes"],
                            "progress event log"))
    terminal = _progress_store_matching_terminal(store, checkpoint_binding, events_sha)
    status_counts = dict(sorted(Counter(item["status"] for item in latest.values()).items()))
    revision_state = config_history["revision_state"]
    expected_revision = revision_state["reference"] if revision_state is not None else None
    publication_path = revision_state["target"] if revision_state is not None else Path(cfg["output_dir"])
    if terminal is not None:
        output_manifest = _validate_terminal_report(terminal["output_manifest"], cfg, cfg_sha,
            comp, comp_sha, manifest, source_binding, rows, latest, checkpoint_binding,
            events_sha, probe, adapter_sha, comparator_sha, config_history, expected_revision)
        if pending:
            proposals = [latest[index] for index in range(len(rows))]
            _reverify_export(cfg, source_binding, comp["max_manifest_bytes"])
            _progress_capacity_records(store, cfg["max_progress_store_bytes"], update_increase=True,
                config_sha256=cfg_sha, original_config_sha256=config_history["proposal_config_sha256"])
            _progress_probe_receipt_write(store, checkpoint_binding, probe, cfg_sha,
                                          cfg["max_progress_store_bytes"])
            _atomic_private_publish(publication_path, proposals, output_manifest, cfg["max_output_bytes"])
            return None
    elif pending:
        _progress_capacity_records(store, cfg["max_progress_store_bytes"], update_increase=True,
            config_sha256=cfg_sha, original_config_sha256=config_history["proposal_config_sha256"])
        _progress_probe_receipt_write(store, checkpoint_binding, probe, cfg_sha,
                                      cfg["max_progress_store_bytes"])
        return None
    else:
        # Per-row events remain usable when the process stopped after a durable
        # event but before a verified terminal result. Publish them explicitly as
        # an ambiguous partial report, never as a complete worker run.
        artifacts = manifest.get("source_artifacts", [])
        artifact_hashes = {item["path"]: item["sha256"] for item in artifacts
                           if isinstance(item, dict) and isinstance(item.get("path"), str)
                           and isinstance(item.get("sha256"), str)}
        source_copy_claim = manifest.get("source_copy_claim", {})
        source_lineage = {"source_artifacts": artifacts,
            "source_artifacts_sha256": artifact_hashes,
            "source_artifact_inventory_sha256": _sha(_canonical_json(artifact_hashes).rstrip(b"\n")),
            "screening_policy_sha256": manifest.get("screening_policy_sha256"),
            "screening_code_sha256": manifest.get("screening_code_sha256"),
            "taxonomy_sha256": manifest.get("taxonomy_sha256"),
            "taxonomy_loader_sha256": manifest.get("taxonomy_loader_sha256"),
            "export_config_sha256": manifest.get("config_sha256"),
            "workspace_plan_sha256": manifest.get("workspace_plan_sha256") or source_copy_claim.get("plan_sha256"),
            "review_ledger_sha256": manifest.get("review_ledger_sha256") or source_copy_claim.get("ledger_sha256"),
            "partial_sidecar_sha256": manifest.get("partial_sidecar_sha256") or source_copy_claim.get("sidecar_sha256"),
            "label_guide_sha256": manifest.get("label_guide_sha256"),
            "target_policy": manifest.get("target_policy"),
            "selected_columns": manifest.get("selected_columns"),
            "counts": manifest.get("counts"), "promotion_eligible": False}
        selected_ids = [row["id"] for row in rows]
        coverage = Counter(f"{label}:{state}" for row in latest.values()
                           for label, state in row["proposal_states"].items())
        # With no trusted terminal event, keep the recovery report entirely
        # derived from durable row evidence and stable task bindings. A new
        # timestamp or resource snapshot on each resume would make identical
        # ambiguous partial recovery appear to be a different publication.
        output_manifest = {"schema_version": REPORT_SCHEMA,
            "status": "partial_aborted_ambiguous", "promotion_eligible": False,
            "claims": {"teacher_suggestions_are_unreviewed": True, "heldout_accessed": False,
                       "labels_independently_verified": False, "screening_independently_verified": False,
                       "promotion_eligible": False},
            "provenance": {"proposal_config_sha256": config_history["proposal_config_sha256"] or cfg_sha,
                "proposal_config_origin_sha256": config_history["proposal_config_sha256"] or cfg_sha,
                "proposal_config_identity_sha256": _proposal_config_identity_sha(cfg),
                "publication_revision": expected_revision,
                "comparison_config_sha256": comp_sha,
                "adapter_code_sha256": adapter_sha, "comparator_code_sha256": comparator_sha,
                "worker_config_sha256": None, "payload_sha256": None,
                "export_manifest_sha256": source_binding["manifest_sha256"],
                "export_config_sha256": source_binding["config_sha256"],
                "train_file_sha256": source_binding["train_sha256"],
                "train_file_bytes": source_binding["train_bytes"],
                "taxonomy_sha256": cfg["taxonomy_sha256"], "source_lineage": source_lineage,
                "selected_row_count": len(rows),
                "selected_row_ids_sha256": _sha(_canonical_json(selected_ids).rstrip(b"\n")),
                "selected_text_digests_sha256": _sha(_canonical_json([r["text_sha256"] for r in rows]).rstrip(b"\n")),
                "target_mask_sha256": _sha(_canonical_json([r["known_mask"] for r in rows]).rstrip(b"\n")),
                "proposal_schema_sha256": _proposal_schema_digest(cfg),
                "teacher_identity": _teacher_wire_identity(cfg),
                "prompt_contract_sha256": checkpoint_binding["prompt_contract_sha256"],
                "progress_store_schema": PROGRESS_STORE_SCHEMA,
                "progress_store_id_sha256": _sha(_canonical_json(checkpoint_binding).rstrip(b"\n")),
                "progress_store_events_sha256": events_sha, "progress_resumed": True,
                "publication_recovered": True, "terminal_worker_result_verified": False,
                "individual_row_events_validated": True,
                "model_name": comp["model_name"],
                "model_manifest_sha256": checkpoint_binding["model_manifest_sha256"],
                "model_layer_bytes": probe.get("model_layer_bytes"),
                "receipt_sha256": probe.get("receipt_sha256"),
                "server_binary_sha256": comp["server_binary_sha256"],
                "server_version": comp["server_version"],
                "runtime_boundary": _stable_runtime_identity(probe.get("runtime_boundary")),
                "request_policy": {k: comp[k] for k in ("timeout_seconds", "max_request_bytes", "max_response_bytes",
                    "num_ctx", "num_predict", "temperature", "top_k", "top_p", "repeat_penalty")},
                "resource_policy": {k: comp[k] for k in ("resource_memory_max_bytes", "resource_cpu_quota_percent",
                    "remote_scope_tasks_max", "min_memory_available_bytes", "min_disk_free_bytes", "min_gpu_free_bytes")}},
            "counts": {"selected_rows": len(rows), "statuses": status_counts,
                "answered_rows": status_counts.get("answered", 0),
                "abstentions": status_counts.get("abstain", 0),
                "failures": sum(n for status, n in status_counts.items() if status not in ("answered", "abstain")),
                "not_attempted_after_abort": status_counts.get("not_attempted_after_abort", 0),
                "per_label_and_state": dict(sorted(coverage.items()))},
            "abort": {"aborted": True, "reason": "verified_terminal_result_unavailable"},
            "uncertainty": {"source_and_review_claims_independently_verified": False,
                "teacher_labels_are_claims": True,
                "omitted_labels_are_negative": cfg["assume_exhaustive_labels"],
                "selection_claim": "configured train-only subset; no heldout data opened"}}
        partial_rows = [latest[index] for index in range(len(rows))]
        if (terminal is None and revision_state is None
                and Path(cfg["output_dir"]).exists()
                and not _published_matches_expected(Path(cfg["output_dir"]), partial_rows, output_manifest)):
            revision_state = _progress_revision_reserve(store, Path(cfg["output_dir"]), checkpoint_binding,
                                                        cfg["max_progress_store_bytes"])
            if revision_state is None:
                raise ProposalError("changed partial report could not reserve a safe immutable revision")
            expected_revision = revision_state["reference"]
            publication_path = revision_state["target"]
            output_manifest["provenance"]["publication_revision"] = expected_revision
    _diagnostic_phase("source_recheck")
    _reverify_export(cfg, source_binding, comp["max_manifest_bytes"])
    # A terminal manifest is immutable publication input. Resume uses the
    # original fsynced bytes; changing probe/progress provenance here would
    # make a persisted temporary manifest appear mismatched after a crash.
    proposals = [latest[index] for index in range(len(rows))]
    _diagnostic_phase("publication")
    if terminal is None:
        _progress_capacity_records(store, cfg["max_progress_store_bytes"], update_increase=True,
            config_sha256=cfg_sha, original_config_sha256=config_history["proposal_config_sha256"])
        _progress_store_write_publication_receipt(store, checkpoint_binding, proposals,
            output_manifest, cfg["max_progress_store_bytes"])
    final_path, manifest_sha = _atomic_private_publish(publication_path, proposals,
                                                        output_manifest, cfg["max_output_bytes"])
    _progress_capacity_records(store, cfg["max_progress_store_bytes"], update_increase=True,
        config_sha256=cfg_sha, original_config_sha256=config_history["proposal_config_sha256"])
    _progress_probe_receipt_write(store, checkpoint_binding, probe, cfg_sha,
                                  cfg["max_progress_store_bytes"])
    return {"rows_selected": len(rows), "rows_completed": len(proposals),
            "manifest_sha256": manifest_sha, "output_directory": str(final_path),
            "status_counts": status_counts, "promotion_eligible": False}


def _run_unleased(config_path: Path, zero_rows: bool = False, resume: bool = False) -> dict:
    global _DIAGNOSTIC_CLEANUP
    _DIAGNOSTIC_CLEANUP = "not_started"
    _diagnostic_phase("config_validation")
    cfg, cfg_sha = load_config(config_path)
    comp_input = Path(cfg["comparison_config"])
    if comp_input.is_symlink():
        raise ProposalError("comparison configuration cannot be a symlink")
    comp_path = comp_input.resolve(strict=True)
    if not _private_file(comp_path):
        raise ProposalError("comparator configuration must be owner-private")
    _diagnostic_phase("config_validation")
    comp, comp_sha = comparator.load_config(comp_path)
    _diagnostic_phase("receipt_validation")
    local_receipt, local_receipt_sha = comparator.verify_acquisition(comp)
    # Reject a stale or unintended destination before opening any train payload.
    _diagnostic_phase("remote_snapshot")
    _remote_identity_snapshot(comp)
    manifest = rows = binding = None
    if not zero_rows:
        manifest, all_rows, binding = _preflight_export(cfg, comp["max_manifest_bytes"])
        rows = select_rows(all_rows, cfg["selection"], cfg["max_rows"])
        if len(rows) > MAX_TRAIN_PAYLOAD_ROWS:
            raise ProposalError("selected row count exceeds hard protocol bound")
        _check_result_bounds(rows, manifest["selected_columns"], cfg["max_output_bytes"],
                             cfg["max_protocol_bytes"], _teacher_wire_identity(cfg),
                             cfg.get("max_teacher_extension_bytes", explicit_labels.MAX_EXTENSION_BYTES),
                             comp["model_name"])
    if zero_rows:
        if resume:
            raise ProposalError("resume cannot be combined with a zero-row probe")
        return _run_transient_worker(cfg, cfg_sha, comp, comp_sha, local_receipt, local_receipt_sha,
                                     None, None, None, True)
    probe = _run_transient_worker(cfg, cfg_sha, comp, comp_sha, local_receipt, local_receipt_sha,
                                   None, None, None, True)
    if resume:
        recovered = _resume_published_progress(cfg, cfg_sha, comp, comp_sha,
                                               manifest, rows, binding, probe)
        if recovered is not None:
            return recovered
        adapter_sha = _sha(Path(__file__).resolve(strict=True).read_bytes())
        comparator_sha = _sha(Path(comparator.__file__).resolve(strict=True).read_bytes())
        checkpoint_binding = _progress_binding(cfg, cfg_sha, comp, comp_sha, rows, binding,
            adapter_sha, comparator_sha, probe)
        progress_store = _progress_store_path(Path(cfg["output_dir"]), allow_final=True)
        revision_state = _progress_revision_reserve(progress_store, Path(cfg["output_dir"]),
                                                    checkpoint_binding, cfg["max_progress_store_bytes"])
        publication_path = revision_state["target"] if revision_state is not None else Path(cfg["output_dir"])
        publication_revision = revision_state["reference"] if revision_state is not None else None
    else:
        publication_path = Path(cfg["output_dir"])
        publication_revision = None
    return _run_transient_worker(cfg, cfg_sha, comp, comp_sha, local_receipt, local_receipt_sha,
                                 manifest, rows, binding, False, expected_probe=probe, resume=resume,
                                 publication_path=publication_path,
                                 publication_revision=publication_revision)


def run(config_path: Path, zero_rows: bool = False, resume: bool = False) -> dict:
    cfg, _ = load_config(config_path)
    with _output_lease(Path(cfg["output_dir"])):
        return _run_unleased(config_path, zero_rows=zero_rows, resume=resume)


def main() -> None:
    global _DIAGNOSTIC_CLEANUP
    _diagnostic_phase("config_validation")
    _DIAGNOSTIC_CLEANUP = "not_started"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--zero-row-probe", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="explicitly resume matching owner-private durable progress")
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
        result = run(args.config, zero_rows=args.zero_row_probe, resume=args.resume)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        record = _diagnostic_record(exc)
        if args.worker and record["phase"].startswith("worker_"):
            worker_record = {"phase": "worker_diagnostic", "schema_version": DIAGNOSTIC_SCHEMA,
                             "worker_phase": record["phase"], "category": record["category"],
                             "cleanup": record["cleanup"]}
            print(json.dumps(worker_record, sort_keys=True, separators=(",", ":")), flush=True)
        print(json.dumps(record, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        raise SystemExit("Teacher proposal task failed; check private configuration, source identity and bounds.") from None


if __name__ == "__main__":
    main()
