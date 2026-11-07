#!/usr/bin/env python3
"""Matched exploratory evaluation of the owned assessment API.

This tool binds the exact development rows and target masks used by the local
classical and 12B reports. Prompts are sent once to a short lived, loopback-only
Spark service. The report contains aggregate statuses and metrics only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import uuid
import inspect
from pathlib import Path, PurePosixPath
from typing import Any

import compare_local_12b as comparison_12b
import classifier_comparison as classical_comparison

SCHEMA = "lfg.owned-matched-validation-config.v1"
REPORT_SCHEMA = "lfg.owned-matched-validation-report.v1"
HEX = re.compile(r"^[0-9a-f]{64}$")
HOST_PART = re.compile(r"^[A-Za-z0-9_.-]{1,253}$")
CURRENT_STAGE = "configuration"
DEFAULT_OUTPUT_TEXT_BYTE_BUCKET_BOUNDARIES = (256, 1024, 4096, 16384)
MAX_OUTPUT_TEXT_BYTE_BUCKET_BOUNDARIES = 16
MAX_OUTPUT_TEXT_BYTE_BOUNDARY = (1 << 63) - 1


def _validated_output_text_byte_bucket_boundaries(value: Any = None) -> list[int]:
    """Return bounded public byte-bucket boundaries, rejecting ambiguous values."""
    if value is None:
        return list(DEFAULT_OUTPUT_TEXT_BYTE_BUCKET_BOUNDARIES)
    if (not isinstance(value, list) or not 1 <= len(value) <= MAX_OUTPUT_TEXT_BYTE_BUCKET_BOUNDARIES
            or any(type(boundary) is not int or boundary < 1 or boundary > MAX_OUTPUT_TEXT_BYTE_BOUNDARY
                   for boundary in value)
            or any(left >= right for left, right in zip(value, value[1:]))):
        raise ValueError("invalid output_text_utf8_byte_bucket_boundaries")
    return list(value)


def _new_owned_response_diagnostics(boundaries: list[int]) -> dict[str, Any]:
    return {
        "accepted_responses": 0,
        "stop_reason_counts": {key: 0 for key in
                               ("stop_sequence", "byte_limit", "context_limit", "other", "unavailable")},
        "output_text_shape_counts": {key: 0 for key in
                                     ("valid_json_object", "valid_json_non_object", "invalid_json",
                                      "unavailable_or_invalid_type")},
        "output_text_utf8_byte_bucket_counts": [0] * (len(boundaries) + 1),
        "output_text_utf8_bytes_unavailable_or_invalid": 0,
    }


def _record_owned_response_diagnostics(stop_reason: Any, output_text: Any, measurement: Any,
                                       boundaries: list[int], counts: dict[str, Any]) -> None:
    """Add one response to fixed aggregates; never retain response content or parse errors."""
    counts["accepted_responses"] += 1
    stop_category = stop_reason if isinstance(stop_reason, str) and stop_reason in (
        "stop_sequence", "byte_limit", "context_limit") else (
            "unavailable" if stop_reason is None or not isinstance(stop_reason, str) else "other")
    counts["stop_reason_counts"][stop_category] += 1

    if not isinstance(output_text, str):
        shape = "unavailable_or_invalid_type"
    else:
        try:
            parsed = json.loads(output_text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (ValueError, TypeError, json.JSONDecodeError, RecursionError):
            shape = "invalid_json"
        else:
            shape = "valid_json_object" if isinstance(parsed, dict) else "valid_json_non_object"
    counts["output_text_shape_counts"][shape] += 1

    byte_count = measurement.get("outputTextUtf8Bytes") if isinstance(measurement, dict) else None
    if type(byte_count) is not int or byte_count < 0:
        counts["output_text_utf8_bytes_unavailable_or_invalid"] += 1
    else:
        index = 0
        while index < len(boundaries) and byte_count >= boundaries[index]:
            index += 1
        counts["output_text_utf8_byte_bucket_counts"][index] += 1


def _payload_output_text_byte_bucket_boundaries(payload: dict[str, Any]) -> list[int]:
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != {"output_text_utf8_byte_bucket_boundaries"}:
        raise ValueError("invalid payload diagnostics configuration")
    return _validated_output_text_byte_bucket_boundaries(
        diagnostics["output_text_utf8_byte_bucket_boundaries"])


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _ssh_remote_command(remote_python_path: str, script: str) -> str:
    if (not isinstance(remote_python_path, str) or not PurePosixPath(remote_python_path).is_absolute()
            or ".." in PurePosixPath(remote_python_path).parts
            or re.fullmatch(r"/[A-Za-z0-9_./ -]+", remote_python_path) is None):
        raise ValueError("remote Python executable must be a safe absolute POSIX path")
    return shlex.join([remote_python_path, "-I", "-c", script])


def _ssh_destination(user: str, host: str) -> str:
    if (not isinstance(user, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", user) is None
            or not isinstance(host, str)
            or re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", host) is None):
        raise ValueError("invalid SSH destination")
    return f"{user}@{host}"


def _remote_cleanup_script(unit: str, remote_home: str) -> str:
    if re.fullmatch(r"lfg-owned-matched-[0-9a-f]{18}\.scope", unit) is None:
        raise ValueError("refusing cleanup outside the generated matched-evaluation unit")
    if (not PurePosixPath(remote_home).is_absolute() or ".." in PurePosixPath(remote_home).parts
            or re.fullmatch(r"/[A-Za-z0-9_./ -]+", remote_home) is None):
        raise ValueError("refusing cleanup with an invalid remote home")
    return f'''import json,os,pathlib,subprocess
unit={unit!r}
env={{'PATH':'/usr/bin:/bin','HOME':{remote_home!r},'XDG_RUNTIME_DIR':os.environ.get('XDG_RUNTIME_DIR','/run/user/'+str(os.getuid()))}}
try: stop=subprocess.run(['/usr/bin/systemctl','--user','stop',unit],capture_output=True,timeout=20,env=env)
except Exception: stop=None
try: state=subprocess.run(['/usr/bin/systemctl','--user','show',unit,'--property=ActiveState,MainPID,ControlGroup'],capture_output=True,text=True,timeout=10,env=env)
except Exception: state=None
props=dict(x.split('=',1) for x in state.stdout.splitlines() if '=' in x) if state is not None and state.returncode==0 else {{}}
inactive=(props.get('ActiveState') in ('inactive','failed') and props.get('MainPID','0') in ('0','')) if state is not None and state.returncode==0 else None
cg=pathlib.Path('/sys/fs/cgroup')/props['ControlGroup'].lstrip('/') if props.get('ControlGroup','').startswith('/') else None
files=list(cg.rglob('cgroup.procs')) if cg is not None and cg.is_dir() else []
observed=bool(files)
tasks=sum(len([line for line in f.read_text().splitlines() if line.strip()]) for f in files) if observed else None
no_tasks=(tasks==0) if observed else None
# A timeout cleanup request has no trusted prior active-cgroup observation.
# Stopping and observing an empty or missing unit now cannot prove that a late
# launch never raced this query, so it remains unknown by contract.
print(json.dumps({{'verified':False,'unit_presence':'unknown_without_prior_active_observation','unit_inactive':inactive,'cgroup_observed_after_stop':observed,'cgroup_task_count_after_stop':tasks,'no_cgroup_tasks_after_stop':no_tasks,'private_namespace_listener_absent':None,'stop_exit_code':None if stop is None else stop.returncode}},sort_keys=True,separators=(',',':')))
'''


def _json(raw: bytes, name: str) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate {name} key")
            value[key] = item
        return value
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid constant")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name}") from error


def _read_bounded(path: Path, limit: int, name: str) -> bytes:
    if type(limit) is not int or limit < 1:
        raise ValueError(f"invalid {name} byte limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"{name} exceeds configured byte limit")
    return raw


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = _read_bounded(path, 65_536, "configuration")
    config = _json(raw, "configuration")
    required = {
        "schema_version", "revision", "classical_report", "classical_report_sha256",
        "classical_config", "classical_config_sha256", "comparison_12b_report",
        "comparison_12b_report_sha256", "comparison_12b_config", "comparison_12b_config_sha256",
        "remote_host", "remote_user", "remote_home", "ssh_port", "remote_python_path", "systemd_run_path",
        "private_source_root", "service_binary", "service_binary_sha256", "service_config_template",
        "service_config_template_sha256",
        "metadata_directory", "expected_model_sha256", "expected_framing_sha256",
        "expected_cached_manifest_sha256", "expected_runtime_config_sha256",
        "expected_training_manifest_sha256", "coexisting_service_unit", "coexisting_listener_port",
        "resource_cpu_quota_percent", "resource_memory_max_bytes", "min_memory_available_bytes",
        "min_disk_free_bytes", "max_source_bytes", "max_total_source_bytes", "max_manifest_bytes",
        "max_screened_file_bytes", "max_policy_bytes", "max_ledger_bytes", "max_config_bytes",
        "max_payload_bytes", "max_response_bytes", "max_report_bytes", "per_request_deadline_seconds",
        "total_runtime_deadline_seconds", "report_max_bytes", "host_snapshot_deadline_seconds",
        "file_preflight_deadline_seconds", "file_transfer_deadline_seconds", "file_cleanup_deadline_seconds",
        "file_max_binary_bytes", "file_max_metadata_bytes",
        "file_min_gpu_free_bytes",
        "expected_remote_architecture",
    }
    if (not isinstance(config, dict) or not required <= set(config) or config.get("schema_version") != SCHEMA
            or not isinstance(config.get("revision"), str) or not config["revision"].strip()):
        raise ValueError("invalid matched evaluation configuration")
    if "output_text_utf8_byte_bucket_boundaries" in config and config["output_text_utf8_byte_bucket_boundaries"] is None:
        raise ValueError("invalid output_text_utf8_byte_bucket_boundaries")
    _validated_output_text_byte_bucket_boundaries(
        config["output_text_utf8_byte_bucket_boundaries"]
        if "output_text_utf8_byte_bucket_boundaries" in config else None)
    for name in ("classical_report_sha256", "classical_config_sha256", "comparison_12b_report_sha256",
                 "comparison_12b_config_sha256", "service_binary_sha256", "expected_model_sha256",
                 "service_config_template_sha256",
                 "expected_framing_sha256", "expected_cached_manifest_sha256", "expected_runtime_config_sha256",
                 "expected_training_manifest_sha256"):
        if not isinstance(config[name], str) or not HEX.fullmatch(config[name]):
            raise ValueError(f"invalid {name}")
    for name in ("classical_report", "classical_config", "comparison_12b_report", "comparison_12b_config",
                 "remote_home", "private_source_root", "service_binary", "service_config_template", "metadata_directory",
                 "coexisting_service_unit", "remote_python_path", "systemd_run_path"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"invalid {name}")
    if (not isinstance(config["remote_host"], str) or not HOST_PART.fullmatch(config["remote_host"])
            or not isinstance(config["remote_user"], str) or not HOST_PART.fullmatch(config["remote_user"])):
        raise ValueError("invalid remote SSH host or user")
    if not PurePosixPath(config["remote_home"]).is_absolute():
        raise ValueError("remote_home must be an absolute POSIX path")
    for name in ("private_source_root", "service_binary", "service_config_template", "metadata_directory"):
        if not PurePosixPath(config[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute POSIX path")
    positive_ints = ("ssh_port", "coexisting_listener_port", "resource_cpu_quota_percent", "resource_memory_max_bytes",
                     "min_memory_available_bytes", "min_disk_free_bytes", "max_source_bytes",
                     "max_total_source_bytes", "max_manifest_bytes", "max_screened_file_bytes",
                     "max_policy_bytes", "max_ledger_bytes", "max_config_bytes", "max_payload_bytes",
                     "max_response_bytes", "max_report_bytes", "per_request_deadline_seconds", "total_runtime_deadline_seconds",
                     "report_max_bytes", "host_snapshot_deadline_seconds")
    positive_ints += ("file_preflight_deadline_seconds", "file_transfer_deadline_seconds", "file_cleanup_deadline_seconds",
                      "file_max_binary_bytes", "file_max_metadata_bytes")
    for name in positive_ints:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"invalid {name}")
    if type(config["file_min_gpu_free_bytes"]) is not int or config["file_min_gpu_free_bytes"] < 0:
        raise ValueError("invalid file_min_gpu_free_bytes")
    if config["ssh_port"] > 65535 or config["coexisting_listener_port"] > 65535 or config["resource_cpu_quota_percent"] > 2000:
        raise ValueError("configured SSH port or CPU quota is outside the supported range")
    if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*\.service", config["coexisting_service_unit"]) is None:
        raise ValueError("invalid coexisting service unit")
    if (not isinstance(config["expected_remote_architecture"], str)
            or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", config["expected_remote_architecture"]) is None):
        raise ValueError("invalid expected remote architecture")
    if config["per_request_deadline_seconds"] <= 0 or config["total_runtime_deadline_seconds"] < config["per_request_deadline_seconds"]:
        raise ValueError("invalid bounded request/runtime deadlines")
    for name in ("classical_report", "classical_config", "comparison_12b_report", "comparison_12b_config"):
        if not Path(config[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute local path")
    return config, sha(raw)


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    return sha(raw)


def _validate_12b_payload_binding(payload: dict[str, Any], report: dict[str, Any],
                                  validation_id_hashes: list[str]) -> None:
    """Reconstruct the exact newline-terminated bytes used by the 12B run."""
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False) + "\n").encode("utf-8")
    inference = report.get("inference")
    identity = sha(json.dumps(sorted(validation_id_hashes), separators=(",", ":")).encode())
    if (not isinstance(inference, dict) or inference.get("payload_sha256") != sha(raw)
            or inference.get("response_record_id_set_sha256") != identity):
        raise ValueError("12B input bytes or response row identity differ from the current validation")


def _validate_process_isolation(status: dict[str, str], uid: int, gid: int,
                                groups: list[int]) -> None:
    """Require all real/effective/saved/fs IDs and Linux capability sets."""
    try:
        user_ids = [int(x) for x in status["Uid"].split()]
        group_ids = [int(x) for x in status["Gid"].split()]
        actual_groups = sorted(int(x) for x in status["Groups"].split())
        caps = [int(status[name], 16) for name in
                ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")]
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("incomplete process identity") from error
    if (type(uid) is not int or uid <= 0 or type(gid) is not int or gid < 0
            or user_ids != [uid] * 4 or group_ids != [gid] * 4
            or actual_groups != sorted(groups) or any(caps)
            or status.get("NoNewPrivs", "").strip() != "1"):
        raise ValueError("process privilege drop is incomplete")


def _validate_network_isolation(links: Any, ipv4_routes: Any, ipv6_routes: Any) -> None:
    if (not isinstance(links, list) or len(links) != 1 or not isinstance(links[0], dict)
            or links[0].get("ifname") != "lo" or "UP" not in links[0].get("flags", [])
            or ipv4_routes != [] or ipv6_routes != []):
        raise ValueError("network namespace has unexpected interfaces or routes")


def _validate_unit_cgroup(path: Any, unit: str) -> None:
    if (not isinstance(unit, str) or re.fullmatch(r"lfg-owned-matched-[0-9a-f]{18}\.service", unit) is None
            or not isinstance(path, str) or not path.startswith("/")
            or "//" in path or any(x in (".", "..", "") for x in path[1:].split("/"))
            or path.split("/")[-1] != unit):
        raise ValueError("invalid transient service cgroup")


def _validate_host_preflight(marker: Any, properties: dict[str, str], main_cgroup: str,
                             worker_status: dict[str, str], server_status: dict[str, str],
                             worker_cgroup: str, server_cgroup: str, worker_netns: int,
                             server_netns: int, host_netns: int, nonce: str,
                             uid: int, gid: int, groups: list[int], host_port_open: bool) -> None:
    unit = "lfg-owned-matched-" + nonce[:18] + ".service"
    if (not isinstance(marker, dict) or marker.get("schema_version") != "lfg.file-preflight.v1"
            or marker.get("nonce") != nonce or marker.get("unit") != unit
            or marker.get("uid") != uid or marker.get("gid") != gid or marker.get("groups") != groups
            or properties.get("ActiveState") != "active" or properties.get("KillMode") != "control-group"
            or not properties.get("MainPID", "").isdigit() or int(properties["MainPID"]) <= 0
            or host_port_open is not False):
        raise ValueError("host preflight identity or unit state mismatch")
    cgroup = properties.get("ControlGroup")
    _validate_unit_cgroup(cgroup, unit)
    if (cgroup != main_cgroup or cgroup != worker_cgroup or cgroup != server_cgroup
            or marker.get("cgroup") != cgroup or worker_netns != server_netns
            or worker_netns == host_netns or marker.get("netns_inode") != worker_netns
            or marker.get("outer_netns_inode") != host_netns
            or type(marker.get("port")) is not int or not 1024 <= marker["port"] <= 65535
            or type(marker.get("worker_pid")) is not int or marker["worker_pid"] <= 0
            or type(marker.get("server_pid")) is not int or marker["server_pid"] <= 0):
        raise ValueError("host preflight process namespace or cgroup mismatch")
    _validate_process_isolation(worker_status, uid, gid, groups)
    _validate_process_isolation(server_status, uid, gid, groups)


def _verified_stop(properties: dict[str, str], stop_exit: int, launch_complete: bool,
                   observed_cgroup: str, observed_ns: int,
                   no_tasks: bool | None, namespace_empty: bool | None) -> bool:
    missing = properties.get("LoadState") == "not-found"
    inactive = missing or (properties.get("ActiveState") in ("inactive", "failed")
                           and properties.get("MainPID") in ("0", ""))
    return bool(launch_complete and observed_cgroup and observed_ns and inactive
                and (stop_exit == 0 or missing) and no_tasks is True and namespace_empty is True)


def _read_private_file(path: Path, limit: int) -> bytes:
    """Read one owner-private regular file without following a final symlink."""
    import stat
    if type(limit) is not int or limit < 1 or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("secure file read is unavailable")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid()
                or stat.S_IMODE(st.st_mode) != 0o600 or st.st_nlink != 1 or st.st_size > limit):
            raise ValueError("private file owner, mode, link count, or size is invalid")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("private file exceeds bound")
        return raw
    finally:
        os.close(fd)


def _restrict_private_file(path: Path) -> None:
    import stat
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("secure file operation unavailable")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_nlink != 1:
            raise ValueError("transferred file identity is invalid")
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


def _hash_private_source(path: Path, limit: int) -> str:
    import stat
    if type(limit) is not int or limit < 1 or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("secure source hash unavailable")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid()
                or stat.S_IMODE(st.st_mode) & 0o077 or st.st_nlink != 1 or st.st_size > limit):
            raise ValueError("source owner, mode, link count, or size invalid")
        digest = hashlib.sha256(); total = 0
        with os.fdopen(fd, "rb", closefd=False) as stream:
            while True:
                chunk = stream.read(min(1048576, limit - total + 1))
                if not chunk: break
                total += len(chunk)
                if total > limit: raise ValueError("source exceeds bound")
                digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)


def _copy_pinned_source(source: Path, target: Path, limit: int, expected: str) -> None:
    if _hash_private_source(source, limit) != expected:
        raise ValueError("source identity mismatch")
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except Exception:
        os.close(source_fd)
        raise
    try:
        digest = hashlib.sha256(); total = 0
        with os.fdopen(source_fd, "rb", closefd=False) as src, os.fdopen(target_fd, "wb", closefd=False) as dst:
            while True:
                chunk = src.read(min(1048576, limit - total + 1))
                if not chunk: break
                total += len(chunk)
                if total > limit: raise ValueError("copied source exceeds bound")
                digest.update(chunk); dst.write(chunk)
            dst.flush(); os.fsync(dst.fileno())
        if digest.hexdigest() != expected: raise ValueError("copied source identity mismatch")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_fd); os.close(target_fd)


def _rename_noreplace(source: Path, target: Path) -> None:
    """Use Linux atomic no-replace rename, never a transient second link."""
    import ctypes
    import errno
    if os.name != "posix" or not sys.platform.startswith("linux"):
        raise ValueError("atomic no-replace rename requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise ValueError("atomic no-replace rename is unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, "target already exists", str(target))
        raise OSError(number, "atomic no-replace rename failed", str(target))


def _publish_private_file(path: Path, raw: bytes, limit: int) -> None:
    """Publish a bounded file once; an existing target is never overwritten."""
    if not isinstance(raw, bytes) or type(limit) is not int or len(raw) > limit:
        raise ValueError("invalid private publication")
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.close(fd)
        fd = -1
        _rename_noreplace(temp, path)
    finally:
        if fd >= 0:
            os.close(fd)
        temp.unlink(missing_ok=True)


def _validate_12b_review_provenance(schema: str, provenance: dict[str, Any],
                                   baseline_report: dict[str, Any]) -> dict[str, Any]:
    if schema == "lfg.local-12b-comparison-config.v1":
        if (provenance.get("labels_unverified") is not True
                or provenance.get("validation_independence_verified") is not False):
            raise ValueError("historical 12B report lacks its explicit unverified-label and independence claims")
        return {"legacy_labels_unverified": True,
                "validation_independence_verified": False,
                "validation_history_claim": "legacy_report_does_not_record_current_claim_schema"}
    if schema != comparison_12b.SCHEMA:
        raise ValueError("unsupported 12B config schema")
    review = comparison_12b._baseline_review_provenance(baseline_report)
    fields = ("label_review_state", "all_rows_marked_verified", "validation_group_use_claim",
              "validation_group_use_claim_verified", "validation_independence_claim",
              "validation_independence_verified")
    if any(provenance.get(field) != review[field] for field in fields):
        raise ValueError("current 12B report review/history provenance differs from the pinned baseline claims")
    return review


def _owned_prediction(status: Any, output: Any, mapping_supported: bool,
                      target_label: str) -> tuple[int | None, str]:
    """Only an exact, justified positive is observable; every other value is unknown."""
    if status != "ok":
        return None, status if isinstance(status, str) else "invalid_status"
    if not isinstance(output, dict) or not isinstance(output.get("intent"), str):
        return None, "malformed_contract"
    if not mapping_supported:
        return None, "unknown_unmapped_taxonomy"
    if output["intent"] == target_label:
        return 1, "exact_positive"
    return None, "unknown_non_target_intent"


def _owned_lifecycle_complete(status: Any, measurement: Any) -> tuple[bool, str]:
    """Accept a row as terminal only when the pinned runtime proves worker completion."""
    if (not isinstance(measurement, dict)
            or measurement.get("schemaVersion") != "lfg.owned-run-measurement.v1"
            or measurement.get("status") != status
            or type(measurement.get("workerStarted")) is not bool
            or type(measurement.get("workerCompletedSuccessfully")) is not bool):
        return False, "unknown_lifecycle_measurement"
    if status in ("timeout", "busy"):
        return False, "owned_timeout" if status == "timeout" else "owned_busy"
    if measurement["workerStarted"] is not True:
        return False, "owned_worker_not_started"
    if measurement["workerCompletedSuccessfully"] is not True:
        return False, "owned_worker_incomplete"
    return True, "completed"


def _owned_terminal_prediction(status: Any, output: Any, measurement: Any,
                               mapping_supported: bool, target_label: str,
                               response_status_counts: dict[str, int] | None = None,
                               lifecycle_outcome_counts: dict[str, int] | None = None) -> tuple[int | None, str, bool]:
    """Do not retain a classifier prediction until its worker completion is evidenced."""
    if response_status_counts is not None and isinstance(status, str):
        response_status_counts[status] = response_status_counts.get(status, 0) + 1
    completed, lifecycle = _owned_lifecycle_complete(status, measurement)
    if not completed:
        if lifecycle_outcome_counts is not None:
            lifecycle_outcome_counts[lifecycle] = lifecycle_outcome_counts.get(lifecycle, 0) + 1
        return None, lifecycle, False
    if lifecycle_outcome_counts is not None:
        lifecycle_outcome_counts["completed"] = lifecycle_outcome_counts.get("completed", 0) + 1
    prediction, outcome = _owned_prediction(status, output, mapping_supported, target_label)
    return prediction, outcome, True


def _deterministic_prediction(response: Any, rule_ids: list[str], fallback_id: str, target_label: str,
                              assessment_sha256: str, rules_sha256: str,
                              assessment_revision: str, session_id: str) -> tuple[int | None, str, dict[str, str] | None]:
    """Map only a complete exhaustive rule hit; all invalid/truncated output abstains."""
    if not isinstance(response, dict) or response.get("schemaVersion") != "lfg.assessment.v1":
        return None, "unknown_invalid_response", None
    prompt = response.get("prompt")
    intent = response.get("intent")
    if response.get("configDigest") != assessment_sha256 or response.get("configRevision") != assessment_revision:
        return None, "unknown_config_identity_drift", None
    if (response.get("sessionId") != session_id or response.get("revision") != 1
            or not isinstance(response.get("configRevision"), str) or not response["configRevision"]
            or not isinstance(prompt, dict) or prompt.get("scoringTruncated") is not False
            or not isinstance(intent, dict) or not isinstance(intent.get("all"), list)
            or not isinstance(intent.get("primary"), str) or type(intent.get("fallback")) is not bool):
        return None, "unknown_invalid_or_truncated", None
    hits = []
    for item in intent["all"]:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or item["id"] not in rule_ids or item["id"] in hits):
            return None, "unknown_invalid_rule_list", None
        hits.append(item["id"])
    if not hits:
        if intent["fallback"] is not True or intent["primary"] != fallback_id:
            return None, "unknown_invalid_fallback", None
    elif intent["fallback"] or intent["primary"] not in hits:
        return None, "unknown_invalid_primary", None
    identity = {"config_revision": response["configRevision"],
                "assessment_sha256": response["configDigest"], "rules_sha256": rules_sha256}
    return int(target_label in hits), "rule_hit_positive" if target_label in hits else "rule_hit_negative", identity


def _metrics(targets: list[int], known: list[bool], predictions: list[int | None]) -> dict[str, Any]:
    if not (len(targets) == len(known) == len(predictions)):
        raise ValueError("metric vectors have inconsistent lengths")
    tp = fp = fn = tn = covered = known_cells = 0
    for target, is_known, prediction in zip(targets, known, predictions, strict=True):
        if type(target) is not int or target not in (0, 1) or type(is_known) is not bool:
            raise ValueError("invalid target mask cell")
        if prediction is not None and (type(prediction) is not int or prediction not in (0, 1)):
            raise ValueError("invalid prediction cell")
        if is_known:
            known_cells += 1
        if is_known and prediction is not None:
            covered += 1
            if target and prediction:
                tp += 1
            elif prediction:
                fp += 1
            elif target:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1_denominator = 2 * tp + fp + fn
    f1 = (2 * tp / f1_denominator) if f1_denominator else None
    return {"known_target_cells": known_cells, "covered_known_target_cells": covered,
            "coverage": covered / known_cells if known_cells else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def _http_loopback(port: int, path: str, method: str = "GET", body: bytes | None = None,
                   token: str | None = None, timeout: float = 30,
                   max_bytes: int = 1_048_576) -> tuple[int, dict[str, str], bytes]:
    """Direct bounded HTTP with no proxy or redirect behavior."""
    import http.client
    import signal
    import threading
    if not (1024 <= port <= 65535) or not path.startswith("/") or "\r" in path or "\n" in path:
        raise ValueError("invalid loopback request target")
    if not math.isfinite(timeout) or timeout <= 0 or max_bytes < 1:
        raise ValueError("invalid HTTP response bound")
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    headers = {"Accept": "application/json", "Cache-Control": "no-store"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["X-LFG-Session-Token"] = token
    total_deadline_supported = (hasattr(signal, "setitimer")
                                and threading.current_thread() is threading.main_thread())
    previous_handler = None
    previous_timer = None
    deadline_started = time.monotonic()
    if total_deadline_supported:
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        if previous_timer[0] > 0:
            connection.close()
            raise RuntimeError("an unrelated process alarm prevents a strict HTTP deadline")
        def deadline_alarm(_signum, _frame):
            raise TimeoutError("loopback HTTP total deadline exceeded")
        signal.signal(signal.SIGALRM, deadline_alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        if 300 <= response.status < 400:
            raise ValueError("loopback redirects are forbidden")
        raw = response.read(max_bytes + 1)
        if time.monotonic() - deadline_started > timeout:
            raise TimeoutError("loopback HTTP total deadline exceeded")
        if len(raw) > max_bytes:
            raise ValueError("loopback response exceeds configured limit")
        return response.status, {key.lower(): value for key, value in response.getheaders()}, raw
    finally:
        connection.close()
        if total_deadline_supported:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _descendant_pids(root_pid: int) -> list[int]:
    """Return visible Linux descendants of one process without shelling out."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            parent = int(fields[1])
            children.setdefault(parent, []).append(int(entry.name))
        except (OSError, IndexError, ValueError):
            continue
    descendants, queue = [], list(children.get(root_pid, []))
    while queue:
        current = queue.pop()
        descendants.append(current)
        queue.extend(children.get(current, []))
    return descendants


def _process_alive(pid: int) -> bool:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text()
        state = fields[fields.rfind(")") + 2:].split()[0]
        return state not in ("Z", "X")
    except (OSError, IndexError):
        return False


def _terminate_process_group(child: Any, graceful_seconds: int = 15,
                             forced_seconds: int = 10) -> dict[str, Any]:
    """Stop and reap the complete task-owned worker process group."""
    import signal
    descendants = _descendant_pids(child.pid)
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=graceful_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=forced_seconds)
    deadline = time.monotonic() + forced_seconds
    while time.monotonic() < deadline and any(_process_alive(pid) for pid in descendants):
        time.sleep(0.1)
    lingering = sum(_process_alive(pid) for pid in descendants)
    return {"verified": child.poll() is not None and lingering == 0,
            "descendants_observed": len(descendants), "descendants_lingering": lingering,
            "exit_code": child.returncode}


def _loopback_port_closed(port: int, timeout_seconds: float = 5.0) -> bool:
    import socket
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return True
        time.sleep(0.1)
    return False


def _private_report_directory(path: Path) -> Path:
    if path.is_symlink() or path.exists():
        raise ValueError("private report directory must be new and not a symlink")
    resolved = path.resolve(strict=False)
    for directory in (resolved, *resolved.parents):
        if (directory / ".git").exists():
            raise ValueError("private report output must be outside every Git worktree")
    path.mkdir(mode=0o700, parents=False, exist_ok=False)
    if os.name == "posix":
        if path.stat().st_uid != os.geteuid() or path.stat().st_mode & 0o077:
            raise ValueError("report directory is not owner-private")
    elif os.name == "nt":
        who = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], check=True,
                             capture_output=True, text=True, timeout=10, shell=False).stdout
        row = next(csv.reader([who.strip()]), [])
        if len(row) != 2 or not re.fullmatch(r"S-1-[0-9-]+", row[1]):
            raise ValueError("current Windows user SID could not be verified")
        subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r",
                        f"*{row[1]}:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F"],
                       check=True, capture_output=True, text=True, timeout=10, shell=False)
        acl = subprocess.run(["icacls", str(path)], check=True, capture_output=True, text=True,
                             timeout=10, shell=False).stdout.splitlines()
        first = acl[0].partition(str(path))[2].strip() if acl else ""
        entries = [match.groups() for line in ([first] + [item.strip() for item in acl[1:]])
                   if (match := re.fullmatch(r"(.+?):((?:\([^)]+\))+)", line))]
        allowed = {row[0].casefold(), "builtin\\administrators", "nt authority\\system"}
        if ({name.strip().casefold() for name, _ in entries} != allowed
                or any("F" not in rights or "(I)" in rights for _, rights in entries)):
            raise ValueError("report directory ACL is not restricted to owner, SYSTEM, and Administrators")
    return path


def _bind_inputs(config: dict[str, Any]) -> dict[str, Any]:
    for field, digest_field in (("classical_report", "classical_report_sha256"),
                                ("classical_config", "classical_config_sha256"),
                                ("comparison_12b_report", "comparison_12b_report_sha256"),
                                ("comparison_12b_config", "comparison_12b_config_sha256")):
        if sha(_read_bounded(Path(config[field]), config["max_config_bytes"], field)) != config[digest_field]:
            raise ValueError(f"{field} digest mismatch")
    config12_raw = _read_bounded(Path(config["comparison_12b_config"]), config["max_config_bytes"], "12B config")
    cfg12_digest = sha(config12_raw)
    cfg12 = _json(config12_raw, "12B config")
    if (not isinstance(cfg12, dict) or cfg12.get("schema_version") not in
            {"lfg.local-12b-comparison-config.v1", comparison_12b.SCHEMA}
            or cfg12.get("baseline_report_sha256") != config["classical_report_sha256"]
            or cfg12.get("classical_config_sha256") != config["classical_config_sha256"]):
        raise ValueError("12B config does not bind the selected classical report and config")
    if cfg12.get("schema_version") == comparison_12b.SCHEMA:
        validated_cfg12, validated_digest = comparison_12b.load_config(Path(config["comparison_12b_config"]))
        if validated_digest != cfg12_digest or validated_cfg12 != cfg12:
            raise ValueError("current 12B config does not pass its current schema validator")
    if (Path(cfg12.get("baseline_report", "")).resolve() != Path(config["classical_report"]).resolve()
            or Path(cfg12.get("classical_config", "")).resolve() != Path(config["classical_config"]).resolve()):
        raise ValueError("12B config paths differ from the exact selected baselines")
    classical_config, classical_digest = classical_comparison.load_config(
        Path(config["classical_config"]), config["max_config_bytes"])
    baseline = _json(_read_bounded(Path(config["classical_report"]), config["max_report_bytes"], "classical report"),
                     "classical report")
    if (not isinstance(baseline, dict) or baseline.get("schema_version") != classical_comparison.REPORT_SCHEMA
            or baseline.get("promotion_eligible") is not False or baseline.get("qualification_claim") is not False
            or baseline.get("config_sha256") != classical_digest
            or sha(Path(classical_comparison.__file__).read_bytes()) != baseline.get("code_sha256")):
        raise ValueError("classical report code/config identity is not current and pinned")
    cc = classical_config
    train_rows, validation_rows, corpus, taxonomy, source_identity = classical_comparison.load_screened_corpus(
        Path(cc["screened_corpus"]), Path(cc["label_plan"]), cc["max_rows"], cc["max_text_bytes"],
        cc["max_source_bytes"], cc["max_total_source_bytes"], cc["max_manifest_bytes"],
        cc["max_screened_file_bytes"], cc["near_duplicate_similarity"], cc["max_near_duplicate_pairs"],
        cc["max_near_duplicate_work"], Path(cc["screen_policy"]), cc["max_policy_bytes"])
    ledger, ledger_digest = classical_comparison._load_review_ledger(Path(cc["review_ledger"]), cc["max_ledger_bytes"])
    if set(ledger) != {row["id"] for row in train_rows + validation_rows}:
        raise ValueError("selected review ledger row set drifted")
    _, y_train, train_known = classical_comparison._matrix(train_rows, cc["target_field"],
                                                            sorted(cc["selected_labels"]), taxonomy)
    _, y_validation, target_known = classical_comparison._matrix(validation_rows, cc["target_field"],
                                                                  sorted(cc["selected_labels"]), taxonomy)
    if cc["rejected_label_handling"] == "exclude":
        train_known[[ledger[row["id"]] == "rejected" for row in train_rows], :] = False
        target_known[[ledger[row["id"]] == "rejected" for row in validation_rows], :] = False
    report_data = baseline.get("data", {})
    if (report_data.get("plan_sha256") != source_identity["plan_sha256"]
            or report_data.get("taxonomy_sha256") != corpus["taxonomy_sha256"]
            or report_data.get("manifest_sha256") != corpus["manifest_sha256"]
            or report_data.get("train_sha256") != corpus["train_sha256"]
            or report_data.get("validation_sha256") != corpus["validation_sha256"]
            or report_data.get("policy_sha256") != corpus["policy_sha256"]
            or report_data.get("screen_code_sha256") != corpus["screen_code_sha256"]
            or report_data.get("review_ledger_sha256") != ledger_digest
            or report_data.get("rejected_label_handling") != cc["rejected_label_handling"]):
        raise ValueError("classical report does not bind the current screened data and ledger")
    row_hashes = comparison_12b._baseline_id_hashes(validation_rows)
    train_hashes = comparison_12b._baseline_id_hashes(train_rows)
    if (len(train_rows) != report_data.get("train_rows") or len(validation_rows) != report_data.get("validation_rows")
            or train_hashes != report_data.get("train_record_id_sha256")
            or row_hashes != report_data.get("validation_record_id_sha256")):
        raise ValueError("current selected source row identities differ from the classical report")
    if (baseline.get("task", {}).get("field") != cc["target_field"]
            or baseline.get("task", {}).get("labels") != sorted(cc["selected_labels"])
            or baseline.get("task", {}).get("taxonomy_sha256") != corpus["taxonomy_sha256"]):
        raise ValueError("classical task or taxonomy identity changed")
    decisions = {decision: sum(value == decision for value in ledger.values())
                 for decision in ("pending", "approved", "rejected")}
    excluded = {"train": sum(ledger[row["id"]] == "rejected" for row in train_rows)
                if cc["rejected_label_handling"] == "exclude" else 0,
                "validation": sum(ledger[row["id"]] == "rejected" for row in validation_rows)
                if cc["rejected_label_handling"] == "exclude" else 0}
    if (report_data.get("review_ledger_decisions") != decisions
            or report_data.get("rejected_rows_excluded") != excluded):
        raise ValueError("review decisions or rejected-target mask changed")
    for index, label in enumerate(sorted(cc["selected_labels"])):
        for split, values, mask in (("train", y_train, train_known), ("validation", y_validation, target_known)):
            observed = values[mask[:, index], index]
            expected = baseline["task"][f"{'training' if split == 'train' else 'validation'}_support"][label]
            if (int(observed.sum()) != expected.get("positive")
                    or len(observed) - int(observed.sum()) != expected.get("negative")
                    or len(values) - len(observed) != expected.get("unknown_rows")):
                raise ValueError("per-label target mask differs from the classical report")
    report12 = _json(_read_bounded(Path(config["comparison_12b_report"]),
                                   config["max_report_bytes"], "12B report"), "12B report")
    if (not isinstance(report12, dict) or report12.get("schema_version") != comparison_12b.REPORT_SCHEMA
            or report12.get("promotion_eligible") is not False or report12.get("qualification_claim") is not False):
        raise ValueError("unsupported exploratory 12B report")
    mask_digest = comparison_12b._target_mask_digest(validation_rows, target_known)
    task = report12.get("task", {})
    provenance = report12.get("provenance", {})
    train_mask_digest = comparison_12b._target_mask_digest(train_rows, train_known)
    report_baseline = report12.get("baseline", {})
    if (report12.get("config_sha256") != cfg12_digest
            or report_baseline.get("report_sha256") != config["classical_report_sha256"]
            or report_baseline.get("config_sha256") != classical_digest
            or provenance.get("baseline_report_sha256") != config["classical_report_sha256"]
            or provenance.get("classical_config_sha256") != classical_digest
            or provenance.get("validation_sha256") != baseline["data"]["validation_sha256"]
            or provenance.get("train_sha256") != baseline["data"]["train_sha256"]
            or provenance.get("plan_sha256") != baseline["data"]["plan_sha256"]
            or provenance.get("manifest_sha256") != baseline["data"]["manifest_sha256"]
            or provenance.get("policy_sha256") != baseline["data"]["policy_sha256"]
            or provenance.get("screen_code_sha256") != baseline["data"]["screen_code_sha256"]
            or provenance.get("review_ledger_sha256") != baseline["data"]["review_ledger_sha256"]
            or provenance.get("taxonomy_sha256") != baseline["data"]["taxonomy_sha256"]
            or task.get("validation_record_id_sha256") != row_hashes
            or task.get("target_mask_sha256", {}).get("train") != train_mask_digest
            or task.get("target_mask_sha256", {}).get("validation") != mask_digest
            or task.get("target_observation_support") != baseline.get("task", {}).get("validation_support")
            or task.get("validation_rows") != len(validation_rows)
            or task.get("field") != classical_config["target_field"]
            or task.get("labels") != sorted(classical_config["selected_labels"])
            ):
        raise ValueError("12B report does not bind the same rows, task, targets and review claims")
    review = _validate_12b_review_provenance(cfg12.get("schema_version"), provenance, baseline)
    if report12.get("inference", {}).get("validation_rows") != len(validation_rows):
        raise ValueError("12B inference count does not match the selected validation")
    prior_payload = comparison_12b._make_payload(
        cfg12, baseline, classical_config, train_rows, validation_rows, y_train, train_known,
        y_validation, target_known, corpus, classical_digest)
    _validate_12b_payload_binding(prior_payload, report12, row_hashes)
    labels = sorted(classical_config["selected_labels"])
    if len(labels) != 1 or classical_config["target_field"] != "intents":
        raise ValueError("matched owned evaluation requires one bound intent target")
    target_label = labels[0]
    repo_root = Path(__file__).resolve().parents[2]
    assessment_path = repo_root / "crates/lfg-core/config/assessment.v1.json"
    rules_path = repo_root / "crates/lfg-core/config/preflight.v1.json"
    assessment = _json(assessment_path.read_bytes(), "assessment config")
    rules = _json(rules_path.read_bytes(), "preflight config")
    rule_ids = [rule.get("intent") for rule in rules.get("rules", []) if isinstance(rule, dict)]
    fallback = rules.get("fallback", {}).get("intent")
    if (not isinstance(rule_ids, list) or not rule_ids or any(not isinstance(x, str) for x in rule_ids)
            or len(rule_ids) != len(set(rule_ids)) or target_label not in rule_ids
            or not isinstance(fallback, str) or fallback in rule_ids):
        raise ValueError("local deterministic intent rule list is not exhaustive and valid")
    code_paths = [assessment_path, rules_path]
    return {"classical_report": baseline, "classical_config": classical_config,
            "comparison_12b_report": report12, "comparison_12b_config": cfg12,
            "train_rows": train_rows, "validation_rows": validation_rows,
            "y_validation": y_validation, "target_known": target_known,
            "taxonomy": taxonomy, "corpus": corpus, "classical_config_sha256": classical_digest,
            "comparison_12b_config_sha256": cfg12_digest, "validation_id_hashes": row_hashes,
            "validation_id_set_sha256": sha(json.dumps(sorted(row_hashes), separators=(",", ":")).encode()),
            "target_mask_sha256": mask_digest, "review_provenance": review,
            "deterministic_assessment_sha256": _canonical_digest(assessment),
            "deterministic_rules_sha256": _canonical_digest(rules), "deterministic_rule_ids": rule_ids,
            "deterministic_fallback_id": fallback, "target_label": target_label,
            "deterministic_files_sha256": {p.relative_to(repo_root).as_posix(): sha(p.read_bytes()) for p in code_paths}}


def _make_payload(config: dict[str, Any], bound: dict[str, Any], run_id: str) -> bytes:
    validation = bound["validation_rows"]
    values = bound["y_validation"].tolist()
    known = bound["target_known"].tolist()
    target_label = bound["target_label"]
    label_index = sorted(bound["classical_config"]["selected_labels"]).index(target_label)
    records = []
    for index, row in enumerate(validation):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("screened validation row text is invalid")
        encoded = text.encode("utf-8")
        if len(encoded) > bound["classical_config"]["max_text_bytes"]:
            raise ValueError("screened validation text exceeds its bound")
        records.append({"text": text, "text_sha256": sha(encoded),
                        "record_id_sha256": sha(row["id"].encode("utf-8")),
                        "target_value": int(values[index][label_index]) if known[index][label_index] else None,
                        "target_known": bool(known[index][label_index])})
    payload = {"schema_version": "lfg.owned-matched-worker-input.v1", "run_id": run_id,
               "binding": {"classical_report_sha256": config["classical_report_sha256"],
                           "classical_config_sha256": bound["classical_config_sha256"],
                           "comparison_12b_report_sha256": config["comparison_12b_report_sha256"],
                           "comparison_12b_config_sha256": bound["comparison_12b_config_sha256"],
                           "validation_sha256": bound["classical_report"]["data"]["validation_sha256"],
                           "validation_record_id_set_sha256": bound["validation_id_set_sha256"],
                           "target_mask_sha256": bound["target_mask_sha256"],
                           "target_label": target_label,
                           "rows": len(records), "assessment_sha256": bound["deterministic_assessment_sha256"],
                           "rules_sha256": bound["deterministic_rules_sha256"],
                           "rule_ids": bound["deterministic_rule_ids"],
                           "fallback_id": bound["deterministic_fallback_id"]},
               "remote": {"private_source_root": config["private_source_root"],
                          "service_binary": config["service_binary"],
                          "service_binary_sha256": config["service_binary_sha256"],
                          "service_config_template": config["service_config_template"],
                          "service_config_template_sha256": config["service_config_template_sha256"],
                          "metadata_directory": config["metadata_directory"],
                          "expected_model_sha256": config["expected_model_sha256"],
                          "expected_framing_sha256": config["expected_framing_sha256"],
                          "expected_cached_manifest_sha256": config["expected_cached_manifest_sha256"],
                          "expected_runtime_config_sha256": config["expected_runtime_config_sha256"],
                          "expected_training_manifest_sha256": config["expected_training_manifest_sha256"],
                          "coexisting_service_unit": config["coexisting_service_unit"],
                          "resource_cpu_quota_percent": config["resource_cpu_quota_percent"],
                          "resource_memory_max_bytes": config["resource_memory_max_bytes"],
                          "min_memory_available_bytes": config["min_memory_available_bytes"],
                          "min_disk_free_bytes": config["min_disk_free_bytes"],
                          "per_request_deadline_seconds": config["per_request_deadline_seconds"],
                          "total_runtime_deadline_seconds": config["total_runtime_deadline_seconds"],
                          "max_response_bytes": config["max_response_bytes"]},
               "diagnostics": {"output_text_utf8_byte_bucket_boundaries":
                               _validated_output_text_byte_bucket_boundaries(
                                   config.get("output_text_utf8_byte_bucket_boundaries")
                                   if "output_text_utf8_byte_bucket_boundaries" in config else None)},
               "records": records}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > config["max_payload_bytes"]:
        raise ValueError("matched private worker payload exceeds configured size")
    return raw


def _validate_payload_identity(payload: bytes, remote_digest: Any) -> str:
    expected = sha(payload)
    if not isinstance(remote_digest, str) or not HEX.fullmatch(remote_digest) or remote_digest != expected:
        raise ValueError("remote worker payload identity does not match the transferred canonical payload")
    return expected


def _remote_scope_launcher_body() -> str:
    """Bridge SSH input into a unique scope only after in-scope preflight."""
    return r'''import base64,json,os,pathlib,pwd,select,shlex,shutil,subprocess,sys,time,zlib
def scope_state(unit,home):
 env={'PATH':'/usr/bin:/bin','HOME':home,'XDG_RUNTIME_DIR':os.environ.get('XDG_RUNTIME_DIR','/run/user/'+str(os.getuid()))}
 try: r=subprocess.run(['/usr/bin/systemctl','--user','show',unit,'--property=LoadState,ActiveState,MainPID,ControlGroup'],capture_output=True,text=True,timeout=3,env=env)
 except Exception: return None
 if r.returncode!=0: return None
 p=dict(x.split('=',1) for x in r.stdout.splitlines() if '=' in x); cg=p.get('ControlGroup','')
 path=pathlib.Path('/sys/fs/cgroup')/cg.lstrip('/') if cg.startswith('/') else None
 try:
  resolved=path.resolve(strict=True) if path else None; croot=pathlib.Path('/sys/fs/cgroup').resolve(strict=True)
  if resolved is None or not resolved.is_relative_to(croot): return {'props':p,'path':None,'tasks':None}
  files=list(resolved.rglob('cgroup.procs'))
  if not files: return {'props':p,'path':str(resolved),'tasks':None}
  count=sum(len([x for x in f.read_text().splitlines() if x.strip()]) for f in files)
  return {'props':p,'path':str(resolved),'tasks':count}
 except Exception: return {'props':p,'path':None,'tasks':None}
def stop_verify(unit,home,observed):
 env={'PATH':'/usr/bin:/bin','HOME':home,'XDG_RUNTIME_DIR':os.environ.get('XDG_RUNTIME_DIR','/run/user/'+str(os.getuid()))}
 try: stop=subprocess.run(['/usr/bin/systemctl','--user','stop',unit],capture_output=True,timeout=20,env=env)
 except Exception: stop=None
 after=scope_state(unit,home); prior=isinstance(observed,dict) and observed.get('path') and observed.get('tasks') is not None
 inactive=None; released=None; tasks=None
 if after is not None:
  props=after['props']; inactive=props.get('LoadState')=='not-found' or (props.get('ActiveState') in ('inactive','failed') and props.get('MainPID') in ('0',''))
 if prior:
  previous=pathlib.Path(observed['path'])
  if not previous.exists(): released=True; tasks=0
  else:
   files=list(previous.rglob('cgroup.procs'))
   if files:
    tasks=sum(len([x for x in f.read_text().splitlines() if x.strip()]) for f in files); released=tasks==0
 stop_ok=stop is not None and stop.returncode in (0,5)
 verified=bool(prior and inactive is True and released is True and stop_ok)
 return {'verified':verified,'unit_presence':'observed_then_released' if verified else ('observed' if prior else 'unknown'),
         'stop_exit_code':None if stop is None else stop.returncode,'unit_inactive':inactive,
         'cgroup_observed_prior':bool(prior),'cgroup_task_count_after':tasks,
         'cgroup_released_after_observation':released}
def main():
 req=json.loads(sys.argv[1]); req['supervisor_b64']=base64.b64encode(zlib.decompress(base64.b64decode(req['supervisor_z_b64']))).decode('ascii'); runid=req['run_id']
 if not isinstance(runid,str) or len(runid)!=32 or any(c not in '0123456789abcdef' for c in runid): raise SystemExit(31)
 unit='lfg-owned-matched-'+runid[:18]+'.scope'; user=pwd.getpwuid(os.geteuid())
 if user.pw_name!=req['remote_user'] or user.pw_uid==0: raise SystemExit(32)
 outer=os.stat('/proc/self/ns/net').st_ino; sb64=req['supervisor_b64']
 root_script='set -eu; /usr/sbin/ip link set lo up; exec /usr/bin/setpriv --reuid="$1" --regid="$2" --init-groups --inh-caps=-all --ambient-caps=-all --bounding-set=-all --no-new-privs /usr/bin/python3 -I -c '+shlex.quote("import base64;exec(base64.b64decode('"+sb64+"'))")+' "$3" "$4"'
 argv=[req['systemd_run_path'],'--user','--scope','--quiet','--unit='+unit,
       '--property=KillMode=control-group','--property=CPUQuota='+str(req['cpu_quota'])+'%',
       '--property=MemoryMax='+str(req['memory_max']), '--','/usr/bin/sudo','-n','/usr/bin/unshare','--net','--',
       '/bin/sh','-c',root_script,'lfg-owned-bootstrap',str(user.pw_uid),str(user.pw_gid),str(outer),unit]
 env={'PATH':'/usr/bin:/bin','HOME':user.pw_dir,'XDG_RUNTIME_DIR':os.environ.get('XDG_RUNTIME_DIR','/run/user/'+str(user.pw_uid))}
 child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=env,bufsize=0)
 deadline=time.monotonic()+req['preflight_deadline']; observed=None
 try:
  ready=b''
  while time.monotonic()<deadline and child.poll() is None:
   if select.select([child.stdout],[],[],.1)[0]:
    ch=child.stdout.read(1)
    if not ch: break
    ready+=ch
    if len(ready)>65536: raise RuntimeError('preflight response exceeded limit')
    if ch==b'\n': break
   state=scope_state(unit,user.pw_dir)
   if state and state.get('path') and state.get('tasks') is not None: observed=state
  if not ready.endswith(b'\n'):
   child.wait(timeout=5)
   print(json.dumps({'type':'preflight','ready':False,'reason':'scope_start_or_preflight_failed'}),flush=True); return
  check=json.loads(ready)
  if check.get('type')!='preflight' or check.get('ready') is not True:
   child.stdin.close(); child.wait(timeout=5); print(ready.decode('utf-8'),end='',flush=True); return
  scope_deadline=time.monotonic()+req['preflight_deadline']; expected_cgroup=check.get('scope_cgroup')
  while time.monotonic()<scope_deadline:
   state=scope_state(unit,user.pw_dir)
   if (state and state.get('props',{}).get('ActiveState')=='active' and state.get('path')
       and state.get('tasks') is not None and state.get('props',{}).get('ControlGroup','').lstrip('/')==expected_cgroup):
    observed=state; break
   time.sleep(.1)
  if observed is None:
   child.stdin.close()
   try: child.wait(timeout=5)
   except Exception: pass
   print(json.dumps({'type':'preflight','ready':False,'reason':'scope_cgroup_not_observed_before_transfer'}),flush=True); return
  sys.stdout.buffer.write(ready); sys.stdout.buffer.flush()
  raw=sys.stdin.buffer.read(req['max_payload']+1)
  if len(raw)>req['max_payload']:
   child.stdin.close(); raise RuntimeError('private transfer exceeded configured limit')
  child.stdin.write(raw); child.stdin.close()
  run_deadline=time.monotonic()+req['run_deadline']; captured=bytearray(); oversized=False
  while child.poll() is None and time.monotonic()<run_deadline:
   if select.select([child.stdout],[],[],.1)[0]:
    chunk=os.read(child.stdout.fileno(),min(65536,req['max_response']+1-len(captured)))
    if not chunk: break
    captured.extend(chunk)
    if len(captured)>req['max_response']: oversized=True; break
   state=scope_state(unit,user.pw_dir)
   if state and state.get('path') and state.get('tasks') is not None: observed=state
  if oversized or child.poll() is None:
   try: subprocess.run(['/usr/bin/systemctl','--user','stop',unit],capture_output=True,timeout=20,env=env)
   except Exception: pass
  try: child.wait(timeout=10)
  except Exception:
   child.terminate()
   try: child.wait(timeout=3)
   except Exception: child.kill(); child.wait(timeout=3)
  if not oversized:
   tail=child.stdout.read(req['max_response']+1-len(captured)); captured.extend(tail)
   oversized=len(captured)>req['max_response']
  cleanup=stop_verify(unit,user.pw_dir,observed)
  if oversized or child.returncode!=0 or not captured.endswith(b'\n'):
   value={'rows_total':None,'rows_attempted':None,'aborted':True,'supervisor_failure':True,
          'abort_reason':'supervisor_scope_timeout_or_invalid_output','payload_transfer_state':'unknown_after_scope_failure',
          'scope_cleanup':cleanup,'cleanup_verified':False,'remote_unit':unit}
  else:
   value=json.loads(captured)
   value['scope_cleanup']=cleanup; value['scope_cleanup_verified']=cleanup['verified']; value['remote_unit']=unit
   value['unit_exit_code']=child.returncode
   worker_cleanup=value.get('owned_service_cleanup_verified') is True
   listener_closed=value.get('listener_closed_verified') is True
   full_cleanup=cleanup['verified'] and worker_cleanup and listener_closed
   value['owned_listener_closed_verified']=listener_closed
   value['owned_service_cleanup_verified']=worker_cleanup
   value['cleanup_verified']=full_cleanup
   if full_cleanup:
    try:
     request=json.loads(raw)
     runid=request['run_id']; source=pathlib.Path(request['remote']['private_source_root']).resolve(strict=True)
     workspace=source/('lfg-owned-matched-'+runid[:18])
     if (workspace.parent==source and not workspace.is_symlink() and workspace.is_dir()
         and workspace.stat().st_uid==user.pw_uid and workspace.stat().st_mode&0o077==0):
      shutil.rmtree(workspace); value['remote_workspace_retained']=False; value['remote_workspace_location']=None
     else: full_cleanup=False
    except Exception: full_cleanup=False
   if not full_cleanup:
    value['remote_workspace_retained']=True
    value['remote_workspace_location']=value.get('remote_workspace_location')
    value['cleanup_verified']=False
    value['aborted']=True; value['abort_reason']='scope_cleanup_unverified'; value['supervisor_failure']=True
    for name in ('owned_metrics','deterministic_metrics'):
     if isinstance(value.get(name),dict):
      for key in ('covered_known_target_cells','coverage','tp','fp','fn','tn','precision','recall','f1'): value[name][key]=None
      value[name]['metric_status']='unknown_after_scope_cleanup_failure'
  print(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False),flush=True)
 except Exception:
  try: child.stdin.close()
  except Exception: pass
  try: subprocess.run(['/usr/bin/systemctl','--user','stop',unit],capture_output=True,timeout=20,env=env)
  except Exception: pass
  try: child.wait(timeout=10)
  except Exception:
   try: child.terminate(); child.wait(timeout=3)
   except Exception: child.kill()
  cleanup=stop_verify(unit,user.pw_dir,observed)
  print(json.dumps({'type':'preflight','ready':False,'reason':'scope_launcher_failure','scope_cleanup':cleanup}),flush=True)
if __name__=='__main__': main()
'''


def _validate_scope_preflight(value: Any, remote_user: str, run_id: str) -> dict[str, Any]:
    unit = "lfg-owned-matched-" + run_id[:18] + ".scope"
    if (not isinstance(value, dict) or value.get("type") != "preflight" or value.get("ready") is not True
            or value.get("username") != remote_user or type(value.get("uid")) is not int or value["uid"] <= 0
            or type(value.get("gid")) is not int or value["gid"] < 0
            or value.get("scope_unit") != unit
            or type(value.get("outer_netns_inode")) is not int
            or type(value.get("netns_inode")) is not int
            or value["netns_inode"] == value["outer_netns_inode"]
            or value.get("interfaces") != ["lo"] or value.get("ipv4_routes") != []
            or value.get("ipv6_routes") != [] or value.get("capabilities_empty") is not True
            or value.get("no_new_privs") is not True or value.get("owner_groups_verified") is not True
            or value.get("scope_cgroup_observed") is not True):
        raise ValueError("root-assisted isolated scope preflight is incomplete")
    return value


REMOTE_WORKER = r'''import hashlib,http.client,json,math,os,pathlib,platform,re,secrets,shutil,signal,socket,statistics,subprocess,sys,time,uuid
from typing import Any
def sha(b): return hashlib.sha256(b).hexdigest()
def verify_payload_evidence(p):
 rows=p.get('records'); b=p.get('binding')
 _payload_output_text_byte_bucket_boundaries(p)
 if not isinstance(rows,list) or not isinstance(b,dict) or len(rows)!=b.get('rows'): raise RuntimeError('worker payload row count mismatch')
 ids=[]; masks=[]
 for row in rows:
  if not isinstance(row,dict) or not isinstance(row.get('record_id_sha256'),str) or re.fullmatch('[0-9a-f]{64}',row['record_id_sha256']) is None: raise RuntimeError('worker record identity is invalid')
  if not isinstance(row.get('text'),str) or sha(row['text'].encode())!=row.get('text_sha256'): raise RuntimeError('worker text identity mismatch')
  if type(row.get('target_known')) is not bool or (row['target_known'] and type(row.get('target_value')) is not int) or (row['target_known'] and row['target_value'] not in (0,1)) or (not row['target_known'] and row.get('target_value') is not None): raise RuntimeError('worker target mask is invalid')
  ids.append(row['record_id_sha256']); masks.append([row['record_id_sha256'],[row['target_known']]])
 id_digest=sha(json.dumps(sorted(ids),separators=(',',':')).encode())
 mask_digest=sha(json.dumps(masks,separators=(',',':')).encode())
 if id_digest!=b.get('validation_record_id_set_sha256') or mask_digest!=b.get('target_mask_sha256'): raise RuntimeError('worker target or row identity digest mismatch')
 original={k:v for k,v in p.items() if k not in ('outer_netns_inode','run_directory')}
 payload_digest=sha(json.dumps(original,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
 return {'payload_sha256':payload_digest,'validation_record_id_set_sha256':id_digest,'target_mask_sha256':mask_digest}
def readjson(p): return json.loads(pathlib.Path(p).read_bytes())
def contained(p,r):
 original=pathlib.Path(p)
 if original.is_symlink(): raise RuntimeError('symlink source is not permitted')
 p=original.resolve(strict=True); r=pathlib.Path(r).resolve(strict=True)
 if p==r or r not in p.parents: raise RuntimeError('private path outside configured source root')
 return p
def stats(xs):
 a=sorted(x for x in xs if type(x) in (float,int) and math.isfinite(x))
 return {'count':len(a),'median_ms':statistics.median(a) if a else None,'p95_ms':a[max(0,math.ceil(.95*len(a))-1)] if a else None,'min_ms':min(a) if a else None,'max_ms':max(a) if a else None}
'''


def _remote_worker_source() -> str:
    helpers = "\n".join(inspect.getsource(fn) for fn in
                         (_validated_output_text_byte_bucket_boundaries, _payload_output_text_byte_bucket_boundaries,
                          _new_owned_response_diagnostics,
                          _record_owned_response_diagnostics, _owned_prediction, _owned_lifecycle_complete, _owned_terminal_prediction,
                          _deterministic_prediction, _metrics, _http_loopback,
                          _descendant_pids, _process_alive, _terminate_process_group, _loopback_port_closed))
    constants = (f"DEFAULT_OUTPUT_TEXT_BYTE_BUCKET_BOUNDARIES = {DEFAULT_OUTPUT_TEXT_BYTE_BUCKET_BOUNDARIES!r}\n"
                 f"MAX_OUTPUT_TEXT_BYTE_BUCKET_BOUNDARIES = {MAX_OUTPUT_TEXT_BYTE_BUCKET_BOUNDARIES}\n"
                 f"MAX_OUTPUT_TEXT_BYTE_BOUNDARY = {MAX_OUTPUT_TEXT_BYTE_BOUNDARY}\n")
    return REMOTE_WORKER + "\n" + constants + helpers + "\n" + _remote_worker_body()


FILE_WORKER_HELPERS = r'''
def proc_identity(pid):
 base=pathlib.Path('/proc')/str(pid)
 status=dict(line.split(':',1) for line in (base/'status').read_text().splitlines() if ':' in line)
 lines=[line.split('::',1)[1] for line in (base/'cgroup').read_text().splitlines() if line.startswith('0::')]
 if len(lines)!=1: raise ValueError('missing unified cgroup')
 return status,lines[0],os.stat(base/'ns/net').st_ino
def ip_json(*args):
 raw=_bounded_command(['/usr/bin/ip','-j',*args],5,65536)
 return json.loads(raw)
def check_self(static):
 status,cgroup,netns=proc_identity(os.getpid())
 _validate_process_isolation(status,static['uid'],static['gid'],static['groups'])
 _validate_unit_cgroup(cgroup,static['unit'])
 if netns==static['outer_netns_inode']: raise ValueError('worker namespace is not isolated')
 _validate_network_isolation(ip_json('link','show'),ip_json('-4','route','show'),ip_json('-6','route','show'))
def check_child_before_exec(static):
 status,cgroup,netns=proc_identity(os.getpid())
 _validate_process_isolation(status,static['uid'],static['gid'],static['groups'])
 _validate_unit_cgroup(cgroup,static['unit'])
 if netns==static['outer_netns_inode']: raise ValueError('child namespace is not isolated')
def file_preflight(static,child,port):
 status,cgroup,netns=proc_identity(os.getpid())
 child_status,child_cgroup,child_netns=proc_identity(child.pid)
 for s in (status,child_status): _validate_process_isolation(s,static['uid'],static['gid'],static['groups'])
 _validate_unit_cgroup(cgroup,static['unit'])
 if cgroup!=child_cgroup or netns!=child_netns or netns==static['outer_netns_inode']:
  raise ValueError('server or worker namespace/cgroup mismatch')
 _validate_network_isolation(ip_json('link','show'),ip_json('-4','route','show'),ip_json('-6','route','show'))
 marker={'schema_version':'lfg.file-preflight.v1','nonce':static['nonce'],'unit':static['unit'],
         'worker_pid':os.getpid(),'server_pid':child.pid,'uid':static['uid'],'gid':static['gid'],
         'groups':static['groups'],'cgroup':cgroup,'netns_inode':netns,
         'outer_netns_inode':static['outer_netns_inode'],'port':port}
 _publish_private_file(pathlib.Path(static['run_directory'])/'preflight.json',
   json.dumps(marker,sort_keys=True,separators=(',',':')).encode(),65536)
def receive_payload(static):
 path=pathlib.Path(static['run_directory'])/'payload.ready'
 deadline=time.monotonic()+static['transfer_deadline_seconds']
 while time.monotonic()<deadline:
  if path.exists():
   raw=_read_private_file(path,static['max_payload_bytes'])
   if sha(raw)!=static['payload_sha256']: raise ValueError('transferred payload digest mismatch')
   payload=_json(raw,'payload')
   if (not isinstance(payload,dict) or payload.get('run_id')!=static['nonce']
       or payload.get('binding')!=static['plan']['binding']
       or payload.get('remote')!=static['plan']['remote']): raise ValueError('payload binding mismatch')
   return payload
  time.sleep(.1)
 raise TimeoutError('payload transfer deadline expired')
'''


def _file_worker_source() -> str:
    """Build the isolated worker; its service preflight precedes payload access."""
    source = _remote_worker_source()
    old_start = (" p=json.load(sys.stdin); b=p['binding']; q=p['remote']; rows=p['records']; run=p['run_id']\n"
                 " diag=p.get('diagnostics')\n"
                 " if not isinstance(diag,dict) or set(diag)!={'output_text_utf8_byte_bucket_boundaries'}: raise RuntimeError('worker diagnostics configuration is invalid')\n"
                 " boundaries=_validated_output_text_byte_bucket_boundaries(diag['output_text_utf8_byte_bucket_boundaries']); response_diagnostics=_new_owned_response_diagnostics(boundaries)\n"
                 " payload_evidence=verify_payload_evidence(p)")
    new_start = (" static=_json(_read_private_file(pathlib.Path(sys.argv[1]),65536),'static configuration')\n"
                 " check_self(static)\n"
                 " p=static['plan']; b=p['binding']; q=p['remote']; rows=[]; run=static['nonce']\n"
                 " diag=p.get('diagnostics')\n"
                 " if not isinstance(diag,dict) or set(diag)!={'output_text_utf8_byte_bucket_boundaries'}: raise RuntimeError('worker diagnostics configuration is invalid')\n"
                 " boundaries=_validated_output_text_byte_bucket_boundaries(diag['output_text_utf8_byte_bucket_boundaries']); response_diagnostics=_new_owned_response_diagnostics(boundaries)\n"
                 " p['outer_netns_inode']=static['outer_netns_inode']; p['run_directory']=static['run_directory']")
    old_deadline = "q['total_runtime_deadline_seconds']<len(rows)*q['per_request_deadline_seconds']+60"
    new_deadline = "q['total_runtime_deadline_seconds']<b['rows']*q['per_request_deadline_seconds']+60"
    marker = "  if (p.get('schema_version')!='lfg.owned-matched-worker-input.v1' or len(rows)!=b.get('rows')"
    insertion = ("  file_preflight(static,child,port)\n"
                 "  received=receive_payload(static)\n"
                 "  payload_evidence=verify_payload_evidence(received)\n"
                 "  p=received; p['outer_netns_inode']=static['outer_netns_inode']; p['run_directory']=static['run_directory']\n"
                 "  diag=p.get('diagnostics'); boundaries=_validated_output_text_byte_bucket_boundaries(diag['output_text_utf8_byte_bucket_boundaries']); response_diagnostics=_new_owned_response_diagnostics(boundaries)\n"
                 "  b=p['binding']; q=p['remote']; rows=p['records']; started=time.monotonic()\n" + marker)
    old_entry = "if __name__=='__main__':\n try: print(json.dumps(main(),sort_keys=True,separators=(',',':'),allow_nan=False))\n except Exception: sys.exit(31)"
    new_entry = ("if __name__=='__main__':\n"
                 " try:\n"
                 "  result=main(); static=_json(_read_private_file(pathlib.Path(sys.argv[1]),65536),'static configuration')\n"
                 "  result['file_nonce']=static['nonce']\n"
                 "  raw=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()\n"
                 "  _publish_private_file(pathlib.Path(static['run_directory'])/'result.json',raw,static['max_result_bytes'])\n"
                 " except Exception: sys.exit(31)")
    old_spawn = "env=env,start_new_session=True,close_fds=True)"
    new_spawn = "env=env,start_new_session=True,close_fds=True,preexec_fn=lambda: check_child_before_exec(static))"
    old_copy = "  for name in ('framing.json','runtime.json','training.json'):\n   shutil.copyfile(md/name,metadata/name); os.chmod(metadata/name,0o600)\n  shutil.copyfile(binary,ownerdir/'lfg-server'); os.chmod(ownerdir/'lfg-server',0o700)"
    new_copy = ("  for name,expected in (('framing.json',q['expected_framing_sha256']),('runtime.json',q['expected_runtime_config_sha256']),('training.json',q['expected_training_manifest_sha256'])):\n"
                "   _copy_pinned_source(md/name,metadata/name,static['max_metadata_bytes'],expected)\n"
                "  _copy_pinned_source(binary,ownerdir/'lfg-server',static['max_binary_bytes'],q['service_binary_sha256']); os.chmod(ownerdir/'lfg-server',0o700)")
    for old, new in ((old_start, new_start), (" route_isolated()\n root=", " root="), (old_deadline, new_deadline),
                     (old_spawn, new_spawn), (old_copy, new_copy),
                     ("sha(binary.read_bytes())", "_hash_private_source(binary,static['max_binary_bytes'])"),
                     ("sha((md/'framing.json').read_bytes())", "_hash_private_source(md/'framing.json',static['max_metadata_bytes'])"),
                     ("sha((md/'runtime.json').read_bytes())", "_hash_private_source(md/'runtime.json',static['max_metadata_bytes'])"),
                     ("sha((md/'training.json').read_bytes())", "_hash_private_source(md/'training.json',static['max_metadata_bytes'])"),
                     ("service=readjson(template)", "template_raw=_read_private_file(template,static['max_metadata_bytes'])\n if sha(template_raw)!=q['service_config_template_sha256']: raise RuntimeError('service template digest mismatch')\n service=_json(template_raw,'service template')"),
                     ("framing=readjson(md/'framing.json')", "framing=_json(_read_private_file(md/'framing.json',static['max_metadata_bytes']),'framing')"),
                     ("sha((ownerdir/'lfg-server').read_bytes())", "_hash_private_source(ownerdir/'lfg-server',static['max_binary_bytes'])"),
                     ("sha((metadata/'framing.json').read_bytes())", "_hash_private_source(metadata/'framing.json',static['max_metadata_bytes'])"),
                     ("sha((metadata/'runtime.json').read_bytes())", "_hash_private_source(metadata/'runtime.json',static['max_metadata_bytes'])"),
                     ("sha((metadata/'training.json').read_bytes())", "_hash_private_source(metadata/'training.json',static['max_metadata_bytes'])"),
                     (marker, insertion), (old_entry, new_entry)):
        if source.count(old) != 1:
            raise ValueError("file worker source anchor changed")
        source = source.replace(old, new, 1)
    helpers = "\n".join(inspect.getsource(fn) for fn in
                        (_json, _validate_process_isolation, _validate_network_isolation,
                         _validate_unit_cgroup, _read_private_file, _hash_private_source,
                         _copy_pinned_source, _rename_noreplace, _publish_private_file, _bounded_command))
    # Definitions must precede the worker's main entry point.
    source = source.replace("def canonical(v):", "from pathlib import Path\n" + helpers + "\n" + FILE_WORKER_HELPERS + "\ndef canonical(v):", 1)
    return source


FILE_CONTROL_BODY = r'''
def proc(pid):
 base=pathlib.Path('/proc')/str(pid)
 status=dict(x.split(':',1) for x in (base/'status').read_text().splitlines() if ':' in x)
 lines=[x.split('::',1)[1] for x in (base/'cgroup').read_text().splitlines() if x.startswith('0::')]
 if len(lines)!=1: raise ValueError('missing cgroup')
 return status,lines[0],os.stat(base/'ns/net').st_ino
def systemctl(unit):
 raw=_bounded_command(['/usr/bin/systemctl','--user','show',unit,'--property=LoadState,ActiveState,MainPID,ControlGroup,KillMode'],5,8192,allowed_exit_codes=(0,1,3,4,5))
 return dict(x.split('=',1) for x in raw.decode().splitlines() if '=' in x)
def host_listens(port):
 with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
  s.settimeout(1)
  result=s.connect_ex(('127.0.0.1',port))
  if result==0: return True
  if result in (111,10061): return False
  raise ValueError('host listener state unknown')
def preflight(static,work):
 marker=_json(_read_private_file(work/'preflight.json',65536),'preflight')
 props=systemctl(static['unit'])
 main_pid=int(props.get('MainPID','0'))
 main_status=dict(x.split(':',1) for x in (pathlib.Path('/proc')/str(main_pid)/'status').read_text().splitlines() if ':' in x)
 main_fields=('Uid','Gid','Groups','CapInh','CapPrm','CapEff','CapBnd','CapAmb','NoNewPrivs')
 if any(name not in main_status for name in main_fields): raise ValueError('main process identity unavailable')
 main_lines=[x.split('::',1)[1] for x in (pathlib.Path('/proc')/str(main_pid)/'cgroup').read_text().splitlines() if x.startswith('0::')]
 if len(main_lines)!=1: raise ValueError('missing main process cgroup')
 worker=proc(marker['worker_pid']); server=proc(marker['server_pid'])
 hostns=os.stat('/proc/self/ns/net').st_ino
 _validate_host_preflight(marker,props,main_lines[0],worker[0],server[0],worker[1],server[1],
  worker[2],server[2],hostns,static['nonce'],static['uid'],static['gid'],static['groups'],host_listens(marker['port']))
 cgp=pathlib.Path('/sys/fs/cgroup')/marker['cgroup'].lstrip('/')
 if not cgp.resolve(strict=True).is_relative_to(pathlib.Path('/sys/fs/cgroup').resolve(strict=True)): raise ValueError('cgroup outside hierarchy')
 memory=(cgp/'memory.max').read_text().strip(); cpu=(cgp/'cpu.max').read_text().split()
 if memory=='max' or int(memory)>static['memory_max_bytes'] or len(cpu)!=2 or cpu[0]=='max': raise ValueError('cgroup resource controls unavailable')
 if int(cpu[0])*100>static['cpu_quota_percent']*int(cpu[1]): raise ValueError('cpu quota exceeds configured limit')
 if (work/'payload.ready').exists() or (work/'result.json').exists(): raise ValueError('worker no longer waiting')
 marker['main_process']={'pid':main_pid,**{name:main_status[name].strip() for name in main_fields}}
 return marker
def snapshot(static):
 mem={}
 for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
  key,_,value=line.partition(':')
  if key in ('MemAvailable','MemTotal','SwapFree','SwapTotal'): mem[key]=int(value.strip().split()[0])*1024
 props=systemctl(static['coexisting_service_unit'])
 port=static['coexisting_listener_port']
 free=shutil.disk_usage(static['private_source_root']).free
 try:
  gpu_raw=_bounded_command(['/usr/bin/nvidia-smi','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],10,16384).decode()
  gpu=[{'total_bytes':int(x.split(',')[0].strip())*1048576,'used_bytes':int(x.split(',')[1].strip())*1048576} for x in gpu_raw.splitlines() if x.strip()]
 except Exception: gpu=None
 return {'memory_bytes':mem,'disk_free_bytes':free,'coexisting_service':props,
         'coexisting_listener':host_listens(port),'load_average':list(os.getloadavg()),
         'architecture':platform.machine(),'gpu':gpu}
def load_static(work):
 return _json(_read_private_file(work/'static.json',65536),'static configuration')
def lock_launch(work,timeout):
 import fcntl,stat
 fd=os.open(work/'launch.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
 try:
  st=os.fstat(fd)
  if not stat.S_ISREG(st.st_mode) or st.st_uid!=os.getuid() or stat.S_IMODE(st.st_mode)!=0o600 or st.st_nlink!=1: raise ValueError('invalid launch lock')
  deadline=time.monotonic()+timeout
  while True:
   try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); return fd
   except BlockingIOError:
    if time.monotonic()>=deadline: raise TimeoutError('launch lock deadline')
    time.sleep(.05)
 except Exception:
  os.close(fd); raise
def main():
 action=sys.argv[1]; work=pathlib.Path(sys.argv[2]); nonce=sys.argv[3]
 if re.fullmatch('[0-9a-f]{32}',nonce) is None or work.name!='lfg-owned-matched-'+nonce[:18]: raise ValueError('invalid run identity')
 if not work.is_absolute() or work.resolve(strict=True)!=work or work.stat().st_uid!=os.getuid() or work.stat().st_mode&0o077: raise ValueError('invalid private workspace')
 if len(sys.argv)<5 or sha(_read_private_file(work/'static.json',65536))!=sys.argv[4]: raise ValueError('static config digest mismatch')
 static=load_static(work)
 if static['nonce']!=nonce or static['run_directory']!=str(work): raise ValueError('workspace binding mismatch')
 if action in ('start','stop'):
  cancel=work/'launch-cancel.json'
  if action=='stop':
   try: _publish_private_file(cancel,json.dumps({'nonce':nonce},separators=(',',':')).encode(),1024)
   except FileExistsError:
    if _json(_read_private_file(cancel,1024),'cancel marker')!={'nonce':nonce}: raise ValueError('invalid cancel marker')
  fd=lock_launch(work,static['cleanup_deadline_seconds'])
  try:
   if action=='start':
    if cancel.exists(): raise ValueError('launch was cancelled')
    _publish_private_file(work/'launch-intent.json',json.dumps({'nonce':nonce},separators=(',',':')).encode(),1024)
   dispatch(action,work,nonce,static)
  finally: os.close(fd)
 else: dispatch(action,work,nonce,static)
def dispatch(action,work,nonce,static):
 if action=='start':
  if sha(_read_private_file(work/'worker.py',1048576))!=static['worker_sha256']: raise ValueError('worker code mismatch')
  if static['uid']!=os.getuid() or static['gid']!=os.getgid() or static['groups']!=sorted(os.getgroups()): raise ValueError('owner mismatch')
  before=snapshot(static)
  if (before['architecture']!=static['expected_remote_architecture'] or before['coexisting_service'].get('ActiveState')!='active'
      or not before['coexisting_listener'] or before['memory_bytes'].get('MemAvailable',0)<static['min_memory_available_bytes']
      or before['disk_free_bytes']<static['min_disk_free_bytes']): raise ValueError('coexistence preflight failed')
  if static['min_gpu_free_bytes'] and (not before['gpu'] or not any(x['total_bytes']-x['used_bytes']>=static['min_gpu_free_bytes'] for x in before['gpu'])): raise ValueError('GPU coexistence floor unavailable')
  _publish_private_file(work/'host-before.json',json.dumps(before,sort_keys=True,separators=(',',':')).encode(),65536)
  script='set -eu; /usr/bin/ip link set lo up; exec /usr/bin/setpriv --reuid "$1" --regid "$2" --init-groups --inh-caps=-all --bounding-set=-all --ambient-caps=-all --no-new-privs /usr/bin/python3 -I "$3" "$4"'
  argv=[static['systemd_run_path'],'--user','--unit',static['unit'],
   '--property=KillMode=control-group','--property=StandardOutput=null','--property=StandardError=null',
   '--property=RuntimeMaxSec='+str(static['runtime_deadline_seconds'])+'s',
   '--property=CPUQuota='+str(static['cpu_quota_percent'])+'%',
   '--property=MemoryMax='+str(static['memory_max_bytes']),
   '/usr/bin/sudo','-n','/usr/bin/unshare','--net','/bin/sh','-c',script,'lfg-bootstrap',
   str(static['uid']),str(static['gid']),str(work/'worker.py'),str(work/'static.json')]
  r=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
  if r.returncode: raise ValueError('transient unit launch failed')
  _publish_private_file(work/'launch-complete.json',json.dumps({'nonce':nonce},separators=(',',':')).encode(),1024)
  if (work/'launch-cancel.json').exists():
   subprocess.run(['/usr/bin/systemctl','--user','stop',static['unit']],stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=static['cleanup_deadline_seconds'])
   raise ValueError('launch cancelled while starting')
  print('{"started":true}')
 elif action=='check':
  m=preflight(static,work)
  print(json.dumps({'ready':True,'cgroup':m['cgroup'],'netns_inode':m['netns_inode'],'worker_pid':m['worker_pid'],'server_pid':m['server_pid'],'main_process':m['main_process']},separators=(',',':')))
 elif action=='state':
  p=systemctl(static['unit'])
  print(json.dumps({'active':p.get('ActiveState') in ('active','activating')},separators=(',',':')))
 elif action=='promote':
  preflight(static,work)
  incoming=work/'payload.partial'
  _restrict_private_file(incoming)
  raw=_read_private_file(incoming,static['max_payload_bytes'])
  if len(raw)!=static['payload_bytes'] or sha(raw)!=static['payload_sha256']: raise ValueError('partial payload mismatch')
  p=_json(raw,'payload')
  if p.get('run_id')!=nonce or p.get('binding')!=static['plan']['binding'] or p.get('remote')!=static['plan']['remote']: raise ValueError('payload plan mismatch')
  verify_payload_evidence(p)
  _rename_noreplace(incoming,work/'payload.ready')
  print('{"promoted":true}')
 elif action=='result':
  p=work/'result.json'
  if not p.exists(): print('{"ready":false}'); return
  raw=_read_private_file(p,static['max_result_bytes']); result=_json(raw,'result')
  if result.get('file_nonce')!=nonce: raise ValueError('result nonce mismatch')
  print(json.dumps({'ready':True,'result':result},sort_keys=True,separators=(',',':'),allow_nan=False))
 elif action=='stop':
  observed_cgroup=sys.argv[5] if len(sys.argv)>5 else ''
  observed_ns=int(sys.argv[6]) if len(sys.argv)>6 else 0
  if observed_cgroup: _validate_unit_cgroup(observed_cgroup,static['unit'])
  r=subprocess.run(['/usr/bin/systemctl','--user','stop',static['unit']],stdin=subprocess.DEVNULL,
   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=static['cleanup_deadline_seconds'])
  props=systemctl(static['unit'])
  missing=props.get('LoadState')=='not-found'
  inactive=missing or (props.get('ActiveState') in ('inactive','failed') and props.get('MainPID') in ('0',''))
  cgpath=pathlib.Path('/sys/fs/cgroup')/observed_cgroup.lstrip('/') if observed_cgroup else None
  no_tasks=None
  if cgpath is not None:
   if not cgpath.exists(): no_tasks=True
   else:
    files=list(cgpath.rglob('cgroup.procs'))
    no_tasks=bool(files) and all(not f.read_text().strip() for f in files)
  namespace_empty=None
  if observed_ns:
   namespace_empty=True
   for entry in pathlib.Path('/proc').iterdir():
    if not entry.name.isdigit(): continue
    try:
     st=dict(x.split(':',1) for x in (entry/'status').read_text().splitlines() if ':' in x)
     if int(st['Uid'].split()[0])==static['uid'] and os.stat(entry/'ns/net').st_ino==observed_ns: namespace_empty=False; break
    except FileNotFoundError: continue
    except (PermissionError,KeyError,ValueError): namespace_empty=None; break
  try:
   after=snapshot(static); before=_json(_read_private_file(work/'host-before.json',65536),'host before')
   old=before['coexisting_service']; new=after['coexisting_service']
   coexist=(old.get('ActiveState')==new.get('ActiveState')=='active' and old.get('MainPID')==new.get('MainPID')
            and before['coexisting_listener'] is True and after['coexisting_listener'] is True)
  except Exception: before=None; after=None; coexist=None
  try: launch_completed=_json(_read_private_file(work/'launch-complete.json',1024),'launch receipt')=={'nonce':nonce}
  except Exception: launch_completed=False
  verified=_verified_stop(props,r.returncode,launch_completed,observed_cgroup,observed_ns,no_tasks,namespace_empty)
  result={'verified':verified,'unit_inactive':inactive,'cgroup_empty':no_tasks,'namespace_owner_processes_absent':namespace_empty,
          'stop_exit_code':r.returncode,'unit_load_state':props.get('LoadState'),
          'unit_disappeared_after_observation':bool(missing and observed_cgroup),
          'launch_completed_before_stop':launch_completed,
          'coexisting_service_snapshots_match':coexist,'host_before':before,'host_after':after,
          'coexisting_service_intervening_state':'unknown_between_point_in_time_snapshots'}
  if verified: _publish_private_file(work/'cleanup.json',json.dumps({'nonce':nonce,'verified':True},separators=(',',':')).encode(),1024)
  print(json.dumps(result,sort_keys=True,separators=(',',':')))
 elif action=='clean':
  receipt=_json(_read_private_file(work/'cleanup.json',1024),'cleanup receipt')
  if receipt!={'nonce':nonce,'verified':True}: raise ValueError('cleanup not verified')
  if work.parent.resolve(strict=True)!=pathlib.Path(static['private_source_root']).resolve(strict=True): raise ValueError('workspace parent changed')
  shutil.rmtree(work)
  print('{"removed":true}')
 else: raise ValueError('unknown action')
if __name__=='__main__':
 try: main()
 except Exception: sys.exit(23)
'''


def _file_control_source() -> str:
    helpers = "\n".join(inspect.getsource(fn) for fn in
                        (_json, _validate_process_isolation, _validate_network_isolation,
                         _validate_unit_cgroup, _validate_host_preflight,
                         _verified_stop, _read_private_file, _restrict_private_file,
                         _rename_noreplace, _publish_private_file, _bounded_command))
    return REMOTE_WORKER + "\nfrom pathlib import Path\n" + helpers + "\n" + FILE_CONTROL_BODY


def _remote_worker_body() -> str:
    return r'''
def canonical(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def route_isolated():
 if any(name!='lo' for _,name in socket.if_nameindex()): raise RuntimeError('network namespace has a non-loopback interface')
 for p in ('/proc/net/route','/proc/net/ipv6_route'):
  if pathlib.Path(p).exists():
   lines=pathlib.Path(p).read_text().splitlines()[1:]
   if p.endswith('route') and not p.endswith('ipv6_route'):
    if any(len(x.split())>1 and x.split()[1]=='00000000' for x in lines): raise RuntimeError('network namespace has a default IPv4 route')
   elif any(x.split() and x.split()[0]=='00000000000000000000000000000000' for x in lines): raise RuntimeError('network namespace has a default IPv6 route')
def main():
 p=json.load(sys.stdin); b=p['binding']; q=p['remote']; rows=p['records']; run=p['run_id']
 diag=p.get('diagnostics')
 if not isinstance(diag,dict) or set(diag)!={'output_text_utf8_byte_bucket_boundaries'}: raise RuntimeError('worker diagnostics configuration is invalid')
 boundaries=_validated_output_text_byte_bucket_boundaries(diag['output_text_utf8_byte_bucket_boundaries']); response_diagnostics=_new_owned_response_diagnostics(boundaries)
 payload_evidence=verify_payload_evidence(p)
 netns=os.stat('/proc/self/ns/net').st_ino
 if netns==p['outer_netns_inode']: raise RuntimeError('worker did not enter a distinct network namespace')
 route_isolated()
 root=pathlib.Path(q['private_source_root']).resolve(strict=True)
 binary=contained(q['service_binary'],root); template=contained(q['service_config_template'],root); md=contained(q['metadata_directory'],root)
 if not md.is_dir() or binary.stat().st_uid!=os.geteuid() or binary.stat().st_mode&0o077 or md.stat().st_uid!=os.geteuid() or md.stat().st_mode&0o077: raise RuntimeError('binary source ownership or privacy check failed')
 for name in ('framing.json','runtime.json','training.json'):
  source=md/name
  if source.is_symlink() or not source.is_file() or source.stat().st_uid!=os.geteuid() or source.stat().st_mode&0o077: raise RuntimeError('metadata ownership or privacy check failed')
 if sha(binary.read_bytes())!=q['service_binary_sha256']: raise RuntimeError('owned service binary digest mismatch')
 if sha((md/'framing.json').read_bytes())!=q['expected_framing_sha256']: raise RuntimeError('framing digest mismatch')
 if sha((md/'runtime.json').read_bytes())!=q['expected_runtime_config_sha256']: raise RuntimeError('runtime config digest mismatch')
 if sha((md/'training.json').read_bytes())!=q['expected_training_manifest_sha256']: raise RuntimeError('training manifest digest mismatch')
 service=readjson(template)
 if service.get('schemaVersion')!='lfg.owned-assessment-service.v1' or service.get('enabled') is not True or service.get('backend')!='cached_onnx' or service.get('artifactMode')!='embedded': raise RuntimeError('owned service template is not the expected cached embedded backend')
 for key,expected in (('expectedModelSha256',q['expected_model_sha256']),('expectedFramingSha256',q['expected_framing_sha256']),('expectedCachedManifestSha256',q['expected_cached_manifest_sha256'])):
  if service.get(key)!=expected: raise RuntimeError('owned service template identity mismatch')
 if service.get('maxConcurrent')!=1 or service.get('workerThreads')!=1: raise RuntimeError('service template is not single flight')
 if q['per_request_deadline_seconds']*1000<=service.get('timeoutMs',0) or q['total_runtime_deadline_seconds']<len(rows)*q['per_request_deadline_seconds']+60: raise RuntimeError('configured client and whole-run deadlines are not above the service limits')
 framing=readjson(md/'framing.json'); system=framing.get('system','')
 # The current service configuration enforces only a nonempty string for
 # intent. Text mentioning a label cannot prove an exhaustive, same-meaning
 # taxonomy; the mapping therefore stays disabled for this checkpoint.
 mapping_supported=False
 if not isinstance(service.get('requiredOutputFields'),list) or not any(isinstance(x,dict) and x.get('name')=='intent' and x.get('valueType')=='string' and x.get('nonEmpty') is True for x in service['requiredOutputFields']): raise RuntimeError('intent output contract missing')
 ownerdir=pathlib.Path(p['run_directory']).resolve(strict=True)
 if ownerdir.parent!=root or ownerdir.stat().st_uid!=os.geteuid() or ownerdir.stat().st_mode&0o077: raise RuntimeError('isolated run directory owner or mode check failed')
 child=None; report=None; started=time.monotonic(); token=secrets.token_hex(32); port=24000+(secrets.randbelow(40000))
 try:
  metadata=ownerdir/'metadata'; metadata.mkdir(mode=0o700)
  state_dir=ownerdir/'state'; state_dir.mkdir(mode=0o700)
  for name in ('framing.json','runtime.json','training.json'):
   shutil.copyfile(md/name,metadata/name); os.chmod(metadata/name,0o600)
  shutil.copyfile(binary,ownerdir/'lfg-server'); os.chmod(ownerdir/'lfg-server',0o700)
  service['framingFile']=str(metadata/'framing.json'); service['runtimeConfigFile']=str(metadata/'runtime.json'); service['trainingManifestFile']=str(metadata/'training.json')
  conf=ownerdir/'owned-service.json'; conf.write_text(canonical(service),encoding='utf-8'); os.chmod(conf,0o600)
  env={'PATH':'/usr/bin:/bin','HOME':os.environ.get('HOME',os.path.expanduser('~')),'LFG_STATE_DIR':str(state_dir),'LFG_LEARNING':'off','LFG_OWNED_ASSESSMENT_CONFIG':str(conf),'LFG_OWNED_ASSESSMENT_TOKEN':token,'NO_PROXY':'*','no_proxy':'*','RUST_LOG':'off'}
  child=subprocess.Popen([str(ownerdir/'lfg-server'),'serve','--bind',f'127.0.0.1:{port}'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env,start_new_session=True,close_fds=True)
  deadline=time.monotonic()+min(60,q['per_request_deadline_seconds'])
  status=None
  while time.monotonic()<deadline:
   if child.poll() is not None: raise RuntimeError('isolated owned service exited during startup')
   try:
    code,h,raw=_http_loopback(port,'/api/lfg/assess/owned/status',token=token,timeout=2,max_bytes=q['max_response_bytes'])
    if code==200:
     if 'no-store' not in h.get('cache-control','').lower(): raise RuntimeError('owned status lacks no-store response policy')
     status=json.loads(raw); break
   except (OSError,TimeoutError): pass
   time.sleep(.2)
  if not isinstance(status,dict) or status.get('status')!='ready' or status.get('enabled') is not True or not isinstance(status.get('engine'),dict): raise RuntimeError('isolated owned service did not become ready')
  engine=status['engine']; limits=status.get('limits')
  if (engine.get('modelIdentityKind')!='source_weights_sha256' or engine.get('modelSha256')!=q['expected_model_sha256']
      or engine.get('sourceWeightsSha256')!=q['expected_model_sha256']
      or engine.get('cachedManifestSha256')!=q['expected_cached_manifest_sha256']
      or engine.get('framingSha256')!=q['expected_framing_sha256']
      or engine.get('runtimeConfigSha256')!=q['expected_runtime_config_sha256']
      or engine.get('trainingManifestSha256')!=q['expected_training_manifest_sha256']): raise RuntimeError('loaded engine identity does not match pinned artifacts')
  expected_limits={'backend':'cached_onnx','maxRequestBytes':service['maxRequestBytes'],'maxContextBytes':service['maxContextBytes'],'maxGeneratedBytes':service['maxGeneratedBytes'],'maxCachedGraphBytes':service['maxCachedGraphBytes'],'maxCacheStateBytesPerRequest':service['maxCacheStateBytesPerRequest'],'timeoutMs':service['timeoutMs'],'maxConcurrent':1,'workerThreads':1}
  if not isinstance(limits,dict) or any(limits.get(k)!=v for k,v in expected_limits.items()): raise RuntimeError('isolated service limits differ from pinned single-flight config')
  unauth_code,unauth_headers,_=_http_loopback(port,'/api/lfg/assess/owned/status',timeout=10,max_bytes=q['max_response_bytes'])
  if unauth_code not in (401,403) or 'no-store' not in unauth_headers.get('cache-control','').lower(): raise RuntimeError('unauthenticated owned status was not rejected')
  cap_code,cap_headers,cap_raw=_http_loopback(port,'/api/lfg/observations/capabilities',timeout=15,max_bytes=q['max_response_bytes'])
  caps=json.loads(cap_raw) if cap_code==200 else None
  if (not isinstance(caps,dict) or 'no-store' not in cap_headers.get('cache-control','').lower()
      or caps.get('assessmentSha256')!=b['assessment_sha256'] or caps.get('rulesSha256')!=b['rules_sha256']): raise RuntimeError('deterministic assessment or preflight identity differs from bound source')
  if (p.get('schema_version')!='lfg.owned-matched-worker-input.v1' or len(rows)!=b.get('rows')
      or not isinstance(b.get('target_label'),str) or b['target_label'] not in b['rule_ids']): raise RuntimeError('matched payload shape differs from its bound report evidence')
  fallback_id=b['fallback_id']; deterministic=[]; owned=[None]*len(rows); det_ms=[]; owned_e2e_ms=[]; owned_handler_ms=[]; det_statuses={}; owned_statuses={}; owned_lifecycle_outcomes={}; mapping_outcomes={}; det_cache_headers={}; det_identity=None
  rows_attempted=0; deterministic_requests_started=0; owned_requests_started=0; aborted=False; abort_reason=None
  for ix,row in enumerate(rows):
   if time.monotonic()-started>q['total_runtime_deadline_seconds']:
    aborted=True; abort_reason='whole_run_deadline'; break
   active_endpoint='pre_request'
   try:
    if sha(row['text'].encode())!=row['text_sha256']: raise RuntimeError('transferred validation text digest mismatch')
    req={'sessionId':'owned-matched-'+run[:12]+'-'+str(ix+1),'revision':1,'text':row['text'],'declaredSubscriptions':[]}
    rawbody=json.dumps(req,ensure_ascii=False,separators=(',',':')).encode()
    active_endpoint='deterministic'
    deterministic_requests_started+=1
    rows_attempted=ix+1
    t0=time.monotonic(); code,headers,raw=_http_loopback(port,'/api/lfg/assess','POST',rawbody,timeout=q['per_request_deadline_seconds'],max_bytes=q['max_response_bytes']); det_ms.append((time.monotonic()-t0)*1000)
    if code!=200: raise RuntimeError('deterministic request returned an unexpected HTTP response')
    cache_header=headers.get('cache-control','<absent>'); det_cache_headers[cache_header]=det_cache_headers.get(cache_header,0)+1
    a=json.loads(raw)
    pred,outcome,identity=_deterministic_prediction(a,b['rule_ids'],fallback_id,b['target_label'],b['assessment_sha256'],b['rules_sha256'],caps['assessmentRevision'],req['sessionId'])
    if outcome=='unknown_config_identity_drift': raise RuntimeError('deterministic config identity drift')
    if identity is not None:
     if det_identity is None: det_identity=identity
     elif det_identity!=identity: raise RuntimeError('deterministic config identity drift')
    deterministic.append(pred); det_statuses[outcome]=det_statuses.get(outcome,0)+1
    body=json.dumps({'text':row['text']},ensure_ascii=False,separators=(',',':')).encode()
    active_endpoint='owned'
    owned_requests_started+=1
    t0=time.monotonic(); code,headers,raw=_http_loopback(port,'/api/lfg/assess/owned','POST',body,token=token,timeout=q['per_request_deadline_seconds'],max_bytes=q['max_response_bytes']); elapsed=(time.monotonic()-t0)*1000
    owned_e2e_ms.append(elapsed)
    if code!=200 or 'no-store' not in headers.get('cache-control','').lower(): raise RuntimeError('owned inference returned an unexpected HTTP response')
    out=json.loads(raw)
    if not isinstance(out,dict) or out.get('schemaVersion')!='lfg.owned-assessment.v1' or out.get('engine')!=engine: raise RuntimeError('owned response lifecycle or engine identity changed')
    measurement=out.get('measurement')
    state=out.get('status'); owned_handler_ms.append(out.get('elapsedMs') if type(out.get('elapsedMs')) is int else None)
    if state not in ('ok','truncated','context_limit','invalid_input','malformed_output','incomplete_output','timeout','inference_error','unavailable','busy'):
     raise RuntimeError('owned service returned an unrecognized status')
    # Only a confirmed completed worker is a completed attempt. Any timeout,
    # busy state, missing child, or incomplete child aborts the remaining rows.
    _record_owned_response_diagnostics(out.get('stopReason'),out.get('outputText'),measurement,boundaries,response_diagnostics)
    owned[ix],outcome,completed = _owned_terminal_prediction(state,out.get('output'),measurement,mapping_supported,b['target_label'],owned_statuses,owned_lifecycle_outcomes)
    if not completed:
     aborted=True; abort_reason=outcome
     break
    mapping_outcomes[outcome]=mapping_outcomes.get(outcome,0)+1
   except Exception as error:
    aborted=True
    if isinstance(error,TimeoutError): abort_reason='client_timeout'
    elif active_endpoint=='deterministic' and str(error)=='deterministic config identity drift': abort_reason='deterministic_identity_drift'
    else: abort_reason='request_transport_protocol_or_lifecycle_failure'
    if active_endpoint=='deterministic': det_statuses['unknown_no_terminal_response']=det_statuses.get('unknown_no_terminal_response',0)+1
    elif active_endpoint=='owned': owned_statuses['no_terminal_response']=owned_statuses.get('no_terminal_response',0)+1
    if abort_reason=='deterministic_identity_drift': deterministic=[None]*len(rows); det_identity=None
    break
  if len(deterministic)<len(rows): deterministic.extend([None]*(len(rows)-len(deterministic)))
  targets=[x['target_value'] if x['target_known'] else 0 for x in rows]; known=[x['target_known'] for x in rows]
  try:
   cap_code_after,cap_headers_after,cap_raw_after=_http_loopback(port,'/api/lfg/observations/capabilities',timeout=15,max_bytes=q['max_response_bytes'])
   caps_after=json.loads(cap_raw_after) if cap_code_after==200 else None
   if (not isinstance(caps_after,dict) or caps_after.get('assessmentSha256')!=caps.get('assessmentSha256')
       or caps_after.get('rulesSha256')!=caps.get('rulesSha256') or caps_after.get('assessmentRevision')!=caps.get('assessmentRevision')
       or caps_after.get('rulesRevision')!=caps.get('rulesRevision')):
    raise RuntimeError('deterministic capability identity drift')
  except Exception:
   aborted=True; abort_reason='deterministic_capability_identity_drift_or_transport'; deterministic=[None]*len(rows); det_identity=None
  report={'rows_attempted':rows_attempted,'rows_total':len(rows),'deterministic_requests_started':deterministic_requests_started,'owned_requests_started':owned_requests_started,'aborted':aborted,'abort_reason':abort_reason,'deterministic_rule_status_counts':det_statuses,'deterministic_cache_control_counts':det_cache_headers,'owned_response_status_counts':owned_statuses,'owned_lifecycle_outcome_counts':owned_lifecycle_outcomes,'owned_mapping_outcome_counts':mapping_outcomes,'owned_response_diagnostics':response_diagnostics,
   'deterministic_metrics':_metrics(targets,known,deterministic),'owned_metrics':_metrics(targets,known,owned),
   'deterministic_latency_ms':stats(det_ms),'owned_end_to_end_latency_ms':stats(owned_e2e_ms),'owned_handler_elapsed_ms':stats(owned_handler_ms),
   'owned_mapping':{'supported':mapping_supported,'status':'exact_taxonomy_not_established_by_checkpoint_framing' if not mapping_supported else 'exact_label_contract_seen'},
   'owned_engine_identity':engine,'owned_status_limits':limits,'deterministic_identity':det_identity,
   'namespace':{'private_network':True,'netns_inode':netns,'interfaces':['lo'],'default_routes':False},
   'runtime':{'binary_sha256':sha((ownerdir/'lfg-server').read_bytes()),'framing_sha256':sha((metadata/'framing.json').read_bytes()),'runtime_config_sha256':sha((metadata/'runtime.json').read_bytes()),'training_manifest_sha256':sha((metadata/'training.json').read_bytes()),'service_config_sha256':sha(conf.read_bytes()),'status':'ready'},
   'worker_resource_sample':resource_sample(),'inference_device':'unknown; endpoint does not report actual CPU/GPU placement','cleanup_verified':False,'service_exit_code':None,
   'worker_payload_sha256':payload_evidence['payload_sha256'],'worker_validation_record_id_set_sha256':payload_evidence['validation_record_id_set_sha256'],'worker_target_mask_sha256':payload_evidence['target_mask_sha256']}
 finally:
  if child is not None:
   cleanup=_terminate_process_group(child)
   if report is not None:
    report['listener_closed_verified']=_loopback_port_closed(port)
    report['owned_service_cleanup_verified']=cleanup['verified']
    report['cleanup_verified']=cleanup['verified'] and report['listener_closed_verified']
    report['service_exit_code']=cleanup['exit_code']
    report['worker_process_cleanup']={k:v for k,v in cleanup.items() if k!='exit_code'}
   if not cleanup['verified'] or not _loopback_port_closed(port): raise RuntimeError('isolated service process tree or listener did not terminate')
 if report is None: raise RuntimeError('evaluation produced no result')
 return report
def resource_sample():
 cg=None
 try:
  line=next(x for x in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
  d=pathlib.Path('/sys/fs/cgroup')/line.split('::',1)[1].lstrip('/')
  cpu={x.split()[0]:int(x.split()[1]) for x in (d/'cpu.stat').read_text().splitlines() if len(x.split())==2}
  def val(n):
   p=d/n; return int(p.read_text()) if p.exists() and p.read_text().strip()!='max' else None
  cg={'cpu_usage_usec':cpu.get('usage_usec'),'memory_current_bytes':val('memory.current'),'memory_peak_bytes':val('memory.peak'),'memory_max_bytes':val('memory.max')}
 except Exception: pass
 return {'cgroup':cg}
if __name__=='__main__':
 try: print(json.dumps(main(),sort_keys=True,separators=(',',':'),allow_nan=False))
 except Exception: sys.exit(31)
'''


def _remote_supervisor_body() -> str:
    return r'''import base64,hashlib,json,os,pathlib,re,shutil,subprocess,sys,tempfile,time
def cmd(a,timeout):
 try:
  return subprocess.run(a,check=False,capture_output=True,timeout=timeout,env={'PATH':'/usr/bin:/bin','HOME':os.environ.get('HOME',os.path.expanduser('~')),'XDG_RUNTIME_DIR':os.environ.get('XDG_RUNTIME_DIR','/run/user/'+str(os.getuid())),'NO_PROXY':'*','no_proxy':'*'})
 except Exception: return None
def snap(unit,disk_path,listener_port,deadline):
 mem={}
 try:
  for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
   k,_,v=line.partition(':')
   if k in ('MemAvailable','MemTotal','SwapFree','SwapTotal'): mem[k]=int(v.strip().split()[0])*1024
 except Exception: pass
 try: load=[float(x) for x in pathlib.Path('/proc/loadavg').read_text().split()[:3]]
 except Exception: load=None
 try: disk=shutil.disk_usage(disk_path).free
 except Exception: disk=None
 try:
  r=subprocess.run(['/usr/bin/systemctl','--user','show',unit,'--property=ActiveState,MainPID,MemoryCurrent,CPUUsageNSec'],capture_output=True,text=True,timeout=deadline,env={'PATH':'/usr/bin:/bin','HOME':os.environ.get('HOME',os.path.expanduser('~')),'XDG_RUNTIME_DIR':os.environ.get('XDG_RUNTIME_DIR','/run/user/'+str(os.getuid()))})
  svc={k:v for k,v in (line.split('=',1) for line in r.stdout.splitlines() if '=' in line)}
 except Exception: svc={}
 try:
  r=subprocess.run(['/usr/bin/ss','-ltn'],capture_output=True,text=True,timeout=deadline,env={'PATH':'/usr/bin:/bin'})
  listener=any('127.0.0.1:'+str(listener_port) in line for line in r.stdout.splitlines())
 except Exception: listener=None
 try:
  r=subprocess.run(['/usr/bin/nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=deadline,env={'PATH':'/usr/bin:/bin'})
  gpu=[line[:200] for line in r.stdout.splitlines()] if r.returncode==0 else None
 except Exception: gpu=None
 return {'architecture':__import__('platform').machine(),'cpus':os.cpu_count(),'memory_bytes':mem,'load_average':load,'private_disk_free_bytes':disk,'coexisting_service':svc,'existing_loopback_listener_'+str(listener_port):listener,'gpu':gpu}
def verify_worker_identity(payload,worker_result):
 expected_payload=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 observed={'payload_sha256':worker_result.get('worker_payload_sha256'),'validation_record_id_set_sha256':worker_result.get('worker_validation_record_id_set_sha256'),'target_mask_sha256':worker_result.get('worker_target_mask_sha256')}
 valid=(observed['payload_sha256']==expected_payload and observed['validation_record_id_set_sha256']==payload['binding']['validation_record_id_set_sha256'] and observed['target_mask_sha256']==payload['binding']['target_mask_sha256'])
 return valid,expected_payload,observed
def observe_unit(unit,home):
 env={'PATH':'/usr/bin:/bin','HOME':home,'XDG_RUNTIME_DIR':os.environ.get('XDG_RUNTIME_DIR','/run/user/'+str(os.getuid()))}
 try: state=subprocess.run(['/usr/bin/systemctl','--user','show',unit,'--property=LoadState,ActiveState,MainPID,ControlGroup'],capture_output=True,text=True,timeout=3,env=env)
 except Exception: return None
 if state.returncode!=0: return None
 props={k:v for k,v in (line.split('=',1) for line in state.stdout.splitlines() if '=' in line)}
 cg=props.get('ControlGroup','')
 cgp=pathlib.Path('/sys/fs/cgroup')/cg.lstrip('/') if cg.startswith('/') else None
 try:
  resolved=cgp.resolve(strict=True) if cgp is not None else None
  root=pathlib.Path('/sys/fs/cgroup').resolve(strict=True)
  if resolved is None or not resolved.is_relative_to(root): return {'unit_observed':bool(props.get('LoadState') not in (None,'not-found')),'load_state':props.get('LoadState'),'active_state':props.get('ActiveState'),'main_pid':props.get('MainPID'),'cgroup':cg,'cgroup_path':None,'cgroup_observed':False,'task_count':None}
  files=list(resolved.rglob('cgroup.procs'))
  if not files: return {'unit_observed':True,'load_state':props.get('LoadState'),'active_state':props.get('ActiveState'),'main_pid':props.get('MainPID'),'cgroup':cg,'cgroup_path':str(resolved),'cgroup_observed':False,'task_count':None}
  tasks=sum(len([line for line in path.read_text().splitlines() if line.strip()]) for path in files)
  return {'unit_observed':True,'load_state':props.get('LoadState'),'active_state':props.get('ActiveState'),'main_pid':props.get('MainPID'),'cgroup':cg,'cgroup_path':str(resolved),'cgroup_observed':True,'task_count':tasks}
 except Exception: return {'unit_observed':True,'load_state':props.get('LoadState'),'active_state':props.get('ActiveState'),'main_pid':props.get('MainPID'),'cgroup':cg,'cgroup_path':None,'cgroup_observed':False,'task_count':None}
def run_unit(argv,input_bytes,timeout,unit,home,workdir,max_bytes):
 env={'PATH':'/usr/bin:/bin','HOME':home,'XDG_RUNTIME_DIR':os.environ.get('XDG_RUNTIME_DIR','/run/user/'+str(os.getuid())),'NO_PROXY':'*','no_proxy':'*'}
 deadline=time.monotonic()+timeout; observed=None; timed_out=False
 with tempfile.TemporaryFile(dir=workdir) as out, tempfile.TemporaryFile(dir=workdir) as err:
  child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=out,stderr=err,env=env)
  try:
   child.stdin.write(input_bytes); child.stdin.close()
   while child.poll() is None:
    sample=observe_unit(unit,home)
    if sample is not None and sample.get('unit_observed') and sample.get('cgroup_observed'):
     observed=sample
    if time.monotonic()>=deadline:
     timed_out=True
     child.terminate()
     try: child.wait(timeout=5)
     except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
     break
    time.sleep(.1)
   code=child.wait(timeout=5)
  except Exception:
   if child.poll() is None:
    child.terminate()
    try: child.wait(timeout=5)
    except Exception: child.kill(); child.wait(timeout=5)
   raise
  out.flush(); err.flush(); out.seek(0); err.seek(0)
  stdout=out.read(max_bytes+1); stderr=err.read(4096)
  return {'returncode':code,'stdout':stdout,'stderr_bytes':len(stderr),'stdout_oversized':len(stdout)>max_bytes,'timed_out':timed_out,'observed_unit':observed}
def stop_and_verify(unit,deadline,home,observed):
 env={'PATH':'/usr/bin:/bin','HOME':home,'XDG_RUNTIME_DIR':os.environ.get('XDG_RUNTIME_DIR','/run/user/'+str(os.getuid()))}
 try: stop=subprocess.run(['/usr/bin/systemctl','--user','stop',unit],capture_output=True,timeout=deadline,env=env)
 except Exception: stop=None
 after=observe_unit(unit,home)
 observed_unit=bool(isinstance(observed,dict) and observed.get('unit_observed') and observed.get('cgroup_observed') and observed.get('cgroup_path'))
 unit_inactive=None; released=None; task_count=None; cgroup_observed_prior=observed_unit
 if observed_unit and after is not None:
  unit_inactive=(after.get('load_state')=='not-found' or (after.get('active_state') in ('inactive','failed') and after.get('main_pid') in ('0','')))
  previous=pathlib.Path(observed['cgroup_path'])
  if not previous.exists():
   released=True; task_count=0
  else:
   files=list(previous.rglob('cgroup.procs'))
   if files:
    task_count=sum(len([line for line in path.read_text().splitlines() if line.strip()]) for path in files)
    released=False
   else: released=None
 stop_ok=stop is not None and stop.returncode in (0,5)
 verified=bool(observed_unit and unit_inactive is True and released is True and stop_ok)
 return {'verified':verified,'stop_attempted':True,'stop_exit_code':None if stop is None else stop.returncode,
         'unit_presence':'observed_then_released' if verified else ('observed' if observed_unit else 'unknown'),
         'unit_inactive':unit_inactive,'cgroup_observed_prior':cgroup_observed_prior,
         'cgroup_task_count_after':task_count,'cgroup_released_after_observation':released,
         'private_namespace_listener_absent':None}
def failure_result(payload,reason,cleanup,unit,run_dir,result=None):
 unknown_metrics={'known_target_cells':sum(bool(row['target_known']) for row in payload['records']),
                  'covered_known_target_cells':None,'coverage':None,'tp':None,'fp':None,'fn':None,'tn':None,
                  'precision':None,'recall':None,'f1':None,'metric_status':'unknown_after_supervisor_failure'}
 observed=None if not isinstance(result,dict) else result.get('observed_unit')
 return {'rows_attempted':None,'rows_total':payload['binding']['rows'],'deterministic_requests_started':None,
         'owned_requests_started':None,'aborted':True,'abort_reason':reason,'supervisor_failure':True,
         'cleanup_verified':False,'unit_cleanup_verified':cleanup['verified'],'worker_cleanup':cleanup,
         'remote_workspace_retained':True,'remote_workspace_location':str(run_dir),
         'owned_service_cleanup_verified':None,'owned_listener_closed_verified':None,
         'payload_transfer_state':'received_by_isolated_supervisor',
         'owned_metrics':unknown_metrics,'deterministic_metrics':unknown_metrics,
         'owned_mapping':{'supported':False,'status':'unknown_after_supervisor_failure'},
         'remote_unit':unit,'unit_exit_code':None if not isinstance(result,dict) else result.get('returncode'),
         'unit_stderr_bytes':None if not isinstance(result,dict) else result.get('stderr_bytes'),
         'unit_observed_while_running':None if not isinstance(observed,dict) else {
             'load_state':observed.get('load_state'),'active_state':observed.get('active_state'),
             'cgroup_observed':observed.get('cgroup_observed'),'task_count':observed.get('task_count')},
         'supervisor_payload_sha256':hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'worker_payload_sha256':None}
def main():
 outer=int(sys.argv[1]); unit=sys.argv[2]; uid=os.getuid(); euid=os.geteuid(); gid=os.getgid(); egid=os.getegid()
 user=__import__('pwd').getpwuid(euid)
 groups=sorted(os.getgroups()); expected_groups=sorted(set(os.getgrouplist(user.pw_name,user.pw_gid)))
 status=dict(line.split(':',1) for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line)
 cap_names=('CapInh','CapPrm','CapEff','CapBnd','CapAmb')
 caps_empty=all(status.get(k,'').strip()=='0000000000000000' for k in cap_names)
 nnp=status.get('NoNewPrivs','').strip()=='1'; netns=os.stat('/proc/self/ns/net').st_ino
 if any(name!='lo' for _,name in __import__('socket').if_nameindex()): raise SystemExit(41)
 for path in ('/proc/net/route','/proc/net/ipv6_route'):
  if pathlib.Path(path).exists() and len(pathlib.Path(path).read_text().splitlines())>1: raise SystemExit(42)
 cgroup_line=next((x.split('::',1)[1] for x in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::')),None)
 if not cgroup_line or unit.removesuffix('.scope') not in cgroup_line: raise SystemExit(43)
 env={'PATH':'/usr/bin:/bin','HOME':os.environ.get('HOME',user.pw_dir),'XDG_RUNTIME_DIR':os.environ.get('XDG_RUNTIME_DIR','/run/user/'+str(euid))}
 try: scope=subprocess.run(['/usr/bin/systemctl','--user','show',unit,'--property=ActiveState,ControlGroup'],capture_output=True,text=True,timeout=3,env=env)
 except Exception: raise SystemExit(44)
 props=dict(x.split('=',1) for x in scope.stdout.splitlines() if '=' in x)
 if scope.returncode!=0 or props.get('ActiveState')!='active' or props.get('ControlGroup','').lstrip('/')!=cgroup_line: raise SystemExit(45)
 preflight={'type':'preflight','ready':bool(uid==euid and gid==egid and euid>0 and user.pw_name and groups==expected_groups and caps_empty and nnp and netns!=outer),
  'username':user.pw_name,'uid':euid,'gid':egid,'groups':groups,'owner_groups_verified':groups==expected_groups,
  'capabilities_empty':caps_empty,'no_new_privs':nnp,'outer_netns_inode':outer,'netns_inode':netns,
  'interfaces':['lo'],'ipv4_routes':[],'ipv6_routes':[],'scope_unit':unit,'scope_cgroup_observed':True,
  'scope_cgroup':cgroup_line}
 print(json.dumps(preflight,sort_keys=True,separators=(',',':')),flush=True)
 if not preflight['ready']: raise SystemExit(46)
 wrapper=json.load(sys.stdin); payload=wrapper['payload']; q=wrapper['remote']; runid=wrapper['run_id']
 if (not isinstance(runid,str) or re.fullmatch('[0-9a-f]{32}',runid) is None
     or unit!='lfg-owned-matched-'+runid[:18]+'.scope' or not isinstance(payload,dict)
     or not isinstance(wrapper.get('worker_b64'),str)): raise SystemExit(47)
 root=pathlib.Path(q['private_source_root']).resolve(strict=True)
 if not root.is_dir() or root.stat().st_uid!=euid or root.stat().st_mode&0o077: raise SystemExit(48)
 before=snap(q['coexisting_service_unit'],q['private_source_root'],q['coexisting_listener_port'],q['host_snapshot_deadline_seconds'])
 listener_key='existing_loopback_listener_'+str(q['coexisting_listener_port'])
 if before['architecture']!='aarch64' or not before['coexisting_service'] or before['coexisting_service'].get('ActiveState')!='active' or not before[listener_key] or before['memory_bytes'].get('MemAvailable',0)<q['min_memory_available_bytes'] or (before['private_disk_free_bytes'] or 0)<q['min_disk_free_bytes']:
  print(json.dumps({'rows_total':payload['binding']['rows'],'rows_attempted':None,'aborted':True,'abort_reason':'host_preflight_failed','supervisor_failure':True,'payload_transfer_state':'received_by_isolated_supervisor','supervisor_payload_sha256':hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()},sort_keys=True,separators=(',',':'))); return
 run_dir=root/('lfg-owned-matched-'+runid[:18])
 if run_dir.exists() or run_dir.is_symlink(): raise SystemExit(49)
 run_dir.mkdir(mode=0o700)
 if run_dir.stat().st_uid!=euid or run_dir.stat().st_mode&0o077: raise SystemExit(50)
 enc_payload=json.dumps({**payload,'outer_netns_inode':outer,'run_directory':str(run_dir)},ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()
 worker_code=base64.b64decode(wrapper['worker_b64'],validate=True)
 if len(worker_code)>q['max_response_bytes']*20: raise SystemExit(51)
 deadline=time.monotonic()+q['total_runtime_deadline_seconds']+30
 with tempfile.TemporaryFile(dir=run_dir) as out, tempfile.TemporaryFile(dir=run_dir) as err:
  child=subprocess.Popen(['/usr/bin/python3','-I','-c',worker_code.decode('utf-8')],stdin=subprocess.PIPE,stdout=out,stderr=err,env=env,start_new_session=True,close_fds=True)
  try:
   child.stdin.write(enc_payload); child.stdin.close()
   while child.poll() is None and time.monotonic()<deadline: time.sleep(.1)
   if child.poll() is None:
    child.terminate()
    try: child.wait(timeout=3)
    except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=3)
   code=child.wait(timeout=3); out.flush(); err.flush(); out.seek(0); err.seek(0)
   raw=out.read(q['max_response_bytes']+1); stderr_bytes=len(err.read(4096))
  except Exception:
   if child.poll() is None:
    child.terminate()
    try: child.wait(timeout=3)
    except Exception: child.kill(); child.wait(timeout=3)
   raise
 if code!=0 or len(raw)>q['max_response_bytes']:
  print(json.dumps({'rows_total':payload['binding']['rows'],'rows_attempted':None,'aborted':True,
   'abort_reason':'worker_timeout_or_error','supervisor_failure':True,'payload_transfer_state':'received_by_isolated_supervisor',
   'supervisor_payload_sha256':hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
   'worker_exit_code':code,'worker_stderr_bytes':stderr_bytes,'remote_workspace_location':str(run_dir)},sort_keys=True,separators=(',',':'))); return
 try: worker_result=json.loads(raw)
 except Exception:
  print(json.dumps({'rows_total':payload['binding']['rows'],'rows_attempted':None,'aborted':True,'abort_reason':'worker_invalid_output','supervisor_failure':True,'payload_transfer_state':'received_by_isolated_supervisor','remote_workspace_location':str(run_dir)},sort_keys=True,separators=(',',':'))); return
 identity_ok,expected_payload_sha,worker_identity=verify_worker_identity(payload,worker_result)
 if not identity_ok:
  worker_result.update({'rows_attempted':None,'aborted':True,'abort_reason':'worker_payload_identity_mismatch','supervisor_failure':True})
  worker_result['worker_input_identity_verified']=False
 else:
  after=snap(q['coexisting_service_unit'],q['private_source_root'],q['coexisting_listener_port'],q['host_snapshot_deadline_seconds'])
  svc_before=before.get('coexisting_service') or {}; svc_after=after.get('coexisting_service') or {}
  snapshots_match=(svc_before.get('ActiveState')==svc_after.get('ActiveState')=='active' and svc_before.get('MainPID')==svc_after.get('MainPID') and before[listener_key] is True and after[listener_key] is True)
  worker_result.update({'host_before':before,'host_after':after,'coexisting_service_snapshots_match':snapshots_match,'coexisting_service_intervening_state':'unknown_between_point_in_time_snapshots','worker_exit_code':code,'worker_stderr_bytes':stderr_bytes,'worker_input_identity_verified':True,'supervisor_payload_sha256':expected_payload_sha,'worker_payload_sha256':worker_identity['payload_sha256'],'worker_validation_record_id_set_sha256':worker_identity['validation_record_id_set_sha256'],'worker_target_mask_sha256':worker_identity['target_mask_sha256'],'payload_transfer_state':'received_by_isolated_supervisor','remote_workspace_location':str(run_dir),'scope_cleanup_verified':False,'cleanup_verified':False})
  if not snapshots_match: worker_result.update({'aborted':True,'abort_reason':'coexisting_service_snapshot_changed','supervisor_failure':True})
 print(json.dumps(worker_result,sort_keys=True,separators=(',',':'),allow_nan=False),flush=True)
if __name__=='__main__':
 try: main()
 except SystemExit: raise
 except Exception: sys.exit(27)
'''


def _unknown_execution(payload: dict[str, Any], reason: str, transfer_state: str) -> dict[str, Any]:
    known=sum(bool(row.get('target_known')) for row in payload.get('records', []))
    metrics={'known_target_cells':known,'covered_known_target_cells':None,'coverage':None,
             'tp':None,'fp':None,'fn':None,'tn':None,'precision':None,'recall':None,'f1':None,
             'metric_status':'unknown_after_isolation_or_transport_failure'}
    return {'rows_total':payload.get('binding',{}).get('rows'),'rows_attempted':None,
            'deterministic_requests_started':None,'owned_requests_started':None,'aborted':True,
            'abort_reason':reason,'supervisor_failure':True,'cleanup_verified':False,
            'payload_transfer_state':transfer_state,'owned_metrics':metrics,'deterministic_metrics':metrics,
            'owned_mapping':{'supported':False,'status':'unknown_after_isolation_or_transport_failure'},
            'owned_response_diagnostics':None,
            'worker_payload_sha256':None,'supervisor_payload_sha256':None,
            'remote_workspace_retained':None,'remote_workspace_status':'unknown'}


def _bounded_command(argv: list[str], timeout: int, limit: int,
                     allowed_exit_codes: tuple[int, ...] = (0,)) -> bytes:
    """Bound stdout, elapsed time, and stderr without passing input through a pipe."""
    import threading
    import signal
    options = {"start_new_session": True} if os.name == "posix" else {
        "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, shell=False, **options)
    pieces: list[bytes] = []
    failure: list[Exception] = []
    def read() -> None:
        try:
            pieces.append(child.stdout.read(limit + 1))
        except Exception as error:
            failure.append(error)
        finally:
            child.stdout.close()
    def stop_tree() -> None:
        if os.name == "posix":
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        else:
            try:
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=3, shell=False)
            except (OSError, subprocess.SubprocessError): pass
        if child.poll() is None:
            try: child.kill()
            except OSError: pass
        try: child.wait(timeout=3)
        except subprocess.TimeoutExpired: pass
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(timeout)
    if reader.is_alive() or (pieces and len(pieces[0]) > limit):
        stop_tree()
        reader.join(1)
        raise TimeoutError("command timed out or exceeded output bound")
    try:
        code = child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        stop_tree()
        raise TimeoutError("command did not exit")
    if failure or code not in allowed_exit_codes or not pieces:
        raise RuntimeError("bounded command failed")
    return pieces[0]


_FILE_WORKSPACE_BOOTSTRAP = r'''import json,os,pathlib,pwd,re,sys
root=pathlib.Path(sys.argv[1]); nonce=sys.argv[2]; name=sys.argv[3]
if re.fullmatch('[0-9a-f]{32}',nonce) is None: raise SystemExit(2)
if not root.is_absolute() or root.resolve(strict=True)!=root or not root.is_dir(): raise SystemExit(3)
if root.stat().st_uid!=os.getuid() or root.stat().st_mode&0o077 or pwd.getpwuid(os.getuid()).pw_name!=name: raise SystemExit(4)
work=root/('lfg-owned-matched-'+nonce[:18]); work.mkdir(mode=0o700)
print(json.dumps({'uid':os.getuid(),'gid':os.getgid(),'groups':sorted(os.getgroups()),'outer_netns_inode':os.stat('/proc/self/ns/net').st_ino},separators=(',',':')))
'''
_FILE_STAGE_VERIFY = r'''import hashlib,os,pathlib,stat,sys
p=pathlib.Path(sys.argv[1]); expected=sys.argv[2]; limit=int(sys.argv[3])
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
try:
 s=os.fstat(fd)
 if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid() or s.st_nlink!=1 or s.st_size>limit: raise SystemExit(2)
 os.fchmod(fd,0o600)
 with os.fdopen(fd,'rb',closefd=False) as f: raw=f.read(limit+1)
 if len(raw)>limit or hashlib.sha256(raw).hexdigest()!=expected: raise SystemExit(3)
finally: os.close(fd)
'''


def _file_evaluate(config: dict[str, Any], payload: bytes, run_id: str,
                   local_work: Path) -> dict[str, Any]:
    """Run the file protocol. Call only after source review and host coordination."""
    import threading
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("invalid run identity")
    root = config["private_source_root"]
    python = config["remote_python_path"]
    systemd = config["systemd_run_path"]
    for value in (root, python, systemd):
        if (not isinstance(value, str) or re.fullmatch(r"/[A-Za-z0-9_./-]+", value) is None
                or ".." in PurePosixPath(value).parts):
            raise ValueError("unsafe remote path")
    work = str(PurePosixPath(root) / ("lfg-owned-matched-" + run_id[:18]))
    destination = _ssh_destination(config["remote_user"], config["remote_host"])
    ssh = ["ssh", "-T", "-oBatchMode=yes", "-oStrictHostKeyChecking=yes", "-oConnectTimeout=8",
           "-p", str(config["ssh_port"]), destination]
    scp = ["scp", "-q", "-oBatchMode=yes", "-oStrictHostKeyChecking=yes", "-oConnectTimeout=8",
           "-P", str(config["ssh_port"])]
    preflight_deadline = config["file_preflight_deadline_seconds"]
    transfer_deadline = config["file_transfer_deadline_seconds"]
    cleanup_deadline = config["file_cleanup_deadline_seconds"]
    value = _json(payload, "worker payload")
    if not isinstance(value, dict) or value.get("run_id") != run_id or len(payload) > config["max_payload_bytes"]:
        raise ValueError("worker payload identity or size is invalid")
    rows = value.get("binding", {}).get("rows")
    if type(rows) is not int or rows < 0:
        raise ValueError("invalid payload row count")
    inference_deadline = min(config["total_runtime_deadline_seconds"],
                             rows * 2 * config["per_request_deadline_seconds"] + 60)
    local_work = local_work.resolve(strict=True)
    files: list[Path] = []
    observed: dict[str, Any] | None = None
    made = False
    started = False
    transfer_started = False
    result = None
    cleanup = None
    static_sha = ""
    stage = "workspace_creation"
    def remote(argv: list[str], timeout: int, limit: int = 65536) -> bytes:
        return _bounded_command(ssh + [shlex.join(argv)], timeout, limit)
    def copy(local: Path, remote_name: str, timeout: int) -> None:
        child = subprocess.run(scp + [str(local), destination + ":" + work + "/" + remote_name],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=timeout, shell=False)
        if child.returncode != 0:
            raise RuntimeError("private file transfer failed")
    def local_file(name: str, raw: bytes) -> Path:
        path = local_work / (run_id + "-" + name)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        files.append(path)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        return path
    def action(name: str, *extra: str, limit: int = 65536) -> dict[str, Any]:
        timeout = (preflight_deadline + 30 if name == "start" else
                   cleanup_deadline * 2 + 20 if name == "stop" else 15)
        raw = remote([python, "-I", work + "/control.py", name, work, run_id, static_sha, *extra],
                     max(10, timeout), limit)
        parsed = _json(raw, "file control response")
        if not isinstance(parsed, dict):
            raise ValueError("invalid file control response")
        return parsed
    try:
        initial = _json(remote([python, "-I", "-c", _FILE_WORKSPACE_BOOTSTRAP, root, run_id,
                                config["remote_user"]], 20, 4096), "workspace identity")
        made = True
        if (not isinstance(initial, dict) or type(initial.get("uid")) is not int or initial["uid"] <= 0
                or type(initial.get("gid")) is not int or not isinstance(initial.get("groups"), list)
                or any(type(x) is not int for x in initial["groups"])
                or type(initial.get("outer_netns_inode")) is not int):
            raise ValueError("workspace owner identity is invalid")
        stage = "static_code_transfer"
        worker = _file_worker_source().encode("utf-8")
        control = _file_control_source().encode("utf-8")
        plan = {k: v for k, v in value.items() if k != "records"}
        static = {**initial, "nonce": run_id, "unit": "lfg-owned-matched-" + run_id[:18] + ".service",
                  "run_directory": work, "private_source_root": root, "plan": plan,
                  "payload_sha256": sha(payload), "payload_bytes": len(payload),
                  "worker_sha256": sha(worker), "max_payload_bytes": config["max_payload_bytes"],
                  "max_result_bytes": config["max_response_bytes"],
                  "max_binary_bytes": config["file_max_binary_bytes"],
                  "max_metadata_bytes": config["file_max_metadata_bytes"],
                  "transfer_deadline_seconds": preflight_deadline + transfer_deadline,
                  "cleanup_deadline_seconds": cleanup_deadline,
                  "runtime_deadline_seconds": preflight_deadline + transfer_deadline + inference_deadline + cleanup_deadline + 30,
                  "cpu_quota_percent": config["resource_cpu_quota_percent"],
                  "memory_max_bytes": config["resource_memory_max_bytes"],
                  "systemd_run_path": systemd, "coexisting_service_unit": config["coexisting_service_unit"],
                  "expected_remote_architecture": config["expected_remote_architecture"],
                  "coexisting_listener_port": config["coexisting_listener_port"],
                  "min_memory_available_bytes": config["min_memory_available_bytes"],
                  "min_disk_free_bytes": config["min_disk_free_bytes"],
                  "min_gpu_free_bytes": config["file_min_gpu_free_bytes"]}
        static_raw = json.dumps(static, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        if len(static_raw) > 65536 or len(worker) > 1048576 or len(control) > 1048576:
            raise ValueError("worker code or static configuration exceeds bound")
        static_sha = sha(static_raw)
        for name, raw in (("control.py", control), ("worker.py", worker), ("static.json", static_raw)):
            path = local_file(name, raw)
            copy(path, name, transfer_deadline)
            remote([python, "-I", "-c", _FILE_STAGE_VERIFY, work + "/" + name, sha(raw), str(len(raw))], 20, 1024)
        started = True  # A launch timeout has an unknown outcome and requires cleanup.
        stage = "service_launch"
        if action("start").get("started") is not True:
            raise RuntimeError("transient service did not start")
        deadline = time.monotonic() + preflight_deadline
        stage = "host_preflight"
        while time.monotonic() < deadline:
            try:
                observed = action("check")
                if observed.get("ready") is True:
                    break
            except (RuntimeError, ValueError):
                pass
            if action("state").get("active") is not True:
                raise RuntimeError("service exited before preflight")
            time.sleep(.2)
        if not isinstance(observed, dict) or observed.get("ready") is not True:
            raise TimeoutError("file preflight did not complete")
        _validate_unit_cgroup(observed.get("cgroup"), static["unit"])
        if type(observed.get("netns_inode")) is not int or observed["netns_inode"] == initial["outer_netns_inode"]:
            raise ValueError("host preflight namespace mismatch")
        path = local_file("payload", payload)
        transfer_started = True
        stage = "payload_transfer"
        copy(path, "payload.partial", transfer_deadline)
        stage = "payload_promotion"
        if action("promote").get("promoted") is not True:
            raise RuntimeError("payload promotion failed")
        deadline = time.monotonic() + inference_deadline
        stage = "result_wait"
        while time.monotonic() < deadline:
            response = action("result", limit=config["max_response_bytes"] + 4096)
            if response.get("ready") is True:
                result = response.get("result")
                break
            if action("state").get("active") is not True:
                raise RuntimeError("service exited before result")
            time.sleep(.2)
        if not isinstance(result, dict):
            raise TimeoutError("file worker result deadline expired")
        _validate_payload_identity(payload, result.get("worker_payload_sha256"))
        _validate_worker_aggregate(result, value["binding"]["rows"],
                                   _payload_output_text_byte_bucket_boundaries(value))
    except Exception as error:
        kind = ("timeout" if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) else
                "invalid" if isinstance(error, ValueError) else
                "io_error" if isinstance(error, OSError) else "failed")
        result = _unknown_execution(value, "supervisor_file_" + stage + "_" + kind,
                                    "unknown_after_file_transfer" if transfer_started else "not_transferred_before_preflight")
    finally:
        if made and started and static_sha:
            try:
                cg = observed.get("cgroup", "") if isinstance(observed, dict) else ""
                ns = str(observed.get("netns_inode", 0)) if isinstance(observed, dict) else "0"
                cleanup = action("stop", cg, ns)
            except Exception:
                cleanup = None
        for path in files:
            try: path.unlink(missing_ok=True)
            except OSError: pass
    if not isinstance(result, dict):
        result = _unknown_execution(value, "supervisor_file_protocol_failure", "not_transferred_before_preflight")
    cleanup_ok = isinstance(cleanup, dict) and cleanup.get("verified") is True
    if result.get("supervisor_failure") is not True:
        worker_ok = result.get("cleanup_verified") is True
        result.update({"payload_transfer_state": "received_by_isolated_worker", "worker_input_identity_verified": True,
                       "file_preflight": observed,
                       "unit_cleanup_verified": cleanup_ok, "worker_cleanup": cleanup,
                       "cleanup_verified": worker_ok and cleanup_ok,
                       "coexisting_service_snapshots_match": cleanup.get("coexisting_service_snapshots_match") if cleanup else None,
                       "coexisting_service_intervening_state": "unknown_between_point_in_time_snapshots",
                       "host_before": cleanup.get("host_before") if cleanup else None,
                       "host_after": cleanup.get("host_after") if cleanup else None})
        if not result["cleanup_verified"] or result["coexisting_service_snapshots_match"] is not True:
            result.update({"aborted": True, "abort_reason": "supervisor_cleanup_or_coexistence_unverified", "supervisor_failure": True})
    else:
        result["unit_cleanup_verified"] = cleanup_ok
        result["worker_cleanup"] = cleanup
    result["file_preflight"] = observed
    result["remote_workspace_retained"] = True if made else None
    result["remote_workspace_location"] = work if made else None
    if not made:
        result["remote_workspace_location_candidate"] = work
        result["remote_workspace_status"] = "unknown_after_workspace_creation_attempt"
    if cleanup_ok:
        try:
            if action("clean").get("removed") is True:
                result["remote_workspace_retained"] = False
                result["remote_workspace_location"] = None
        except Exception:
            result["remote_workspace_status"] = "cleanup_unconfirmed"
    return result


def _empty_probe_payload(config: dict[str, Any], classical_config: dict[str, Any], run_id: str) -> bytes:
    """An empty production envelope, with no corpus reads or inference rows."""
    labels = sorted(classical_config["selected_labels"])
    if classical_config["target_field"] != "intents" or len(labels) != 1:
        raise ValueError("probe requires one configured intent target")
    root = Path(__file__).resolve().parents[2]
    assessment = _json((root / "crates/lfg-core/config/assessment.v1.json").read_bytes(), "assessment config")
    rules = _json((root / "crates/lfg-core/config/preflight.v1.json").read_bytes(), "rule config")
    rule_ids = [rule.get("intent") for rule in rules.get("rules", []) if isinstance(rule, dict)]
    fallback = rules.get("fallback", {}).get("intent")
    if (not rule_ids or any(not isinstance(x, str) for x in rule_ids) or len(set(rule_ids)) != len(rule_ids)
            or labels[0] not in rule_ids or not isinstance(fallback, str) or fallback in rule_ids):
        raise ValueError("probe rule identity is invalid")
    remote_fields = ("private_source_root", "service_binary", "service_binary_sha256", "service_config_template", "service_config_template_sha256",
                     "metadata_directory", "expected_model_sha256", "expected_framing_sha256",
                     "expected_cached_manifest_sha256", "expected_runtime_config_sha256",
                     "expected_training_manifest_sha256", "coexisting_service_unit", "resource_cpu_quota_percent",
                     "resource_memory_max_bytes", "min_memory_available_bytes", "min_disk_free_bytes",
                     "per_request_deadline_seconds", "total_runtime_deadline_seconds", "max_response_bytes")
    empty_digest = sha(b"[]")
    payload = {"schema_version": "lfg.owned-matched-worker-input.v1", "run_id": run_id,
               "binding": {"classical_report_sha256": config["classical_report_sha256"],
                           "classical_config_sha256": config["classical_config_sha256"],
                           "comparison_12b_report_sha256": config["comparison_12b_report_sha256"],
                           "comparison_12b_config_sha256": config["comparison_12b_config_sha256"],
                           "validation_sha256": None, "validation_record_id_set_sha256": empty_digest,
                           "target_mask_sha256": empty_digest, "target_label": labels[0], "rows": 0,
                           "assessment_sha256": _canonical_digest(assessment), "rules_sha256": _canonical_digest(rules),
                           "rule_ids": rule_ids, "fallback_id": fallback},
               "remote": {field: config[field] for field in remote_fields},
               "diagnostics": {"output_text_utf8_byte_bucket_boundaries":
                               _validated_output_text_byte_bucket_boundaries(
                                   config.get("output_text_utf8_byte_bucket_boundaries")
                                   if "output_text_utf8_byte_bucket_boundaries" in config else None)},
               "records": []}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def run_file_probe(config_path: Path, output: Path) -> dict[str, Any]:
    config, config_digest = load_config(config_path)
    raw = _read_bounded(Path(config["classical_config"]), config["max_config_bytes"], "classical config")
    if sha(raw) != config["classical_config_sha256"]:
        raise ValueError("classical config digest mismatch")
    classical_config, _ = classical_comparison.load_config(Path(config["classical_config"]), config["max_config_bytes"])
    run_id = uuid.uuid4().hex
    payload = _empty_probe_payload(config, classical_config, run_id)
    output_dir = _private_report_directory(output)
    result = _file_evaluate(config, payload, run_id, output_dir)
    passed = (result.get("rows_total") == 0 and result.get("rows_attempted") == 0
              and result.get("deterministic_requests_started") == 0 and result.get("owned_requests_started") == 0
              and result.get("aborted") is False and result.get("cleanup_verified") is True
              and result.get("coexisting_service_snapshots_match") is True
              and result.get("remote_workspace_retained") is False)
    report = {"schema_version": "lfg.owned-file-probe.v1", "config_sha256": config_digest,
              "adapter_sha256": sha(Path(__file__).read_bytes()),
              "worker_code_sha256": sha(_file_worker_source().encode()),
              "control_code_sha256": sha(_file_control_source().encode()),
              "passed": passed, "private_rows": 0, "result": result}
    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(encoded) > config["report_max_bytes"]:
        raise ValueError("probe report exceeds bound")
    target = output_dir / "owned-file-probe.json"
    with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    return {"passed": passed, "report_path": str(target), "report_sha256": sha(encoded)}


def _validate_probe(probe_raw: bytes, config_digest: str,
                    output_text_byte_bucket_boundaries: list[int] | None = None) -> dict[str, Any]:
    """Bind a successful no-input run to the current file protocol and config."""
    probe = _json(probe_raw, "probe report")
    worker_sha = sha(_file_worker_source().encode())
    control_sha = sha(_file_control_source().encode())
    if (not isinstance(probe, dict) or probe.get("schema_version") != "lfg.owned-file-probe.v1"
            or probe.get("passed") is not True or probe.get("private_rows") != 0
            or probe.get("adapter_sha256") != sha(Path(__file__).read_bytes())
            or probe.get("config_sha256") != config_digest or probe.get("worker_code_sha256") != worker_sha
            or probe.get("control_code_sha256") != control_sha):
        raise ValueError("matching successful no-input probe is required")
    result = probe.get("result")
    if (not isinstance(result, dict) or result.get("rows_total") != 0 or result.get("rows_attempted") != 0
            or result.get("deterministic_requests_started") != 0 or result.get("owned_requests_started") != 0
            or result.get("aborted") is not False or result.get("cleanup_verified") is not True
            or result.get("coexisting_service_snapshots_match") is not True
            or result.get("remote_workspace_retained") is not False):
        raise ValueError("no-input probe result does not establish the required lifecycle")
    _validate_worker_aggregate(result, 0, output_text_byte_bucket_boundaries)
    return {"probe_report_sha256": sha(probe_raw), "worker_code_sha256": worker_sha,
            "control_code_sha256": control_sha}


def _ssh_worker(config: dict[str, Any], payload: bytes, run_id: str) -> dict[str, Any]:
    # The scope-and-pipe transport below failed independent security review.
    # It must not transfer private input while the file-handshake replacement,
    # its real no-input probe, and independent review remain incomplete.
    raise RuntimeError("owned validation transport disabled pending reviewed file handshake")
    import base64
    import queue
    import threading
    import zlib
    worker = _remote_worker_source()
    worker_b64 = base64.b64encode(worker.encode("utf-8")).decode("ascii")
    wrapper = {"run_id": run_id, "worker_b64": worker_b64, "payload": _json(payload, "worker payload"),
               "remote": {key: config[key] for key in ("remote_home", "private_source_root", "coexisting_service_unit",
                         "coexisting_listener_port", "host_snapshot_deadline_seconds", "min_memory_available_bytes",
                         "min_disk_free_bytes", "total_runtime_deadline_seconds", "max_response_bytes")}}
    wrapped=(json.dumps(wrapper,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n").encode("utf-8")
    if len(wrapped)>config["max_payload_bytes"]: raise ValueError("remote worker envelope exceeds configured byte limit")
    supervisor_b64=base64.b64encode(zlib.compress(_remote_supervisor_body().encode("utf-8"),9)).decode("ascii")
    settings={"run_id":run_id,"remote_user":config["remote_user"],"systemd_run_path":config["systemd_run_path"],
              "cpu_quota":config["resource_cpu_quota_percent"],"memory_max":config["resource_memory_max_bytes"],
              "preflight_deadline":max(15,config["host_snapshot_deadline_seconds"]),
              "run_deadline":config["total_runtime_deadline_seconds"]+60,
              "max_payload":config["max_payload_bytes"],"max_response":config["max_response_bytes"],
              "supervisor_z_b64":supervisor_b64}
    settings_b64=base64.b64encode(json.dumps(settings,sort_keys=True,separators=(",",":" )).encode()).decode("ascii")
    launcher_b64=base64.b64encode(zlib.compress(_remote_scope_launcher_body().encode("utf-8"),9)).decode("ascii")
    remote_command=shlex.join([config["remote_python_path"],"-I","-c",
                                "import base64,zlib;exec(zlib.decompress(base64.b64decode('"+launcher_b64+"'))) ",settings_b64])
    args=["ssh","-T","-oBatchMode=yes","-oConnectTimeout=8","-p",str(config["ssh_port"]),
          _ssh_destination(config["remote_user"],config["remote_host"]),remote_command]
    timeout=config["total_runtime_deadline_seconds"]+config["host_snapshot_deadline_seconds"]*2+120
    try:
        proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=False,bufsize=0)
    except Exception:
        value=_unknown_execution(_json(payload,"worker payload"),"supervisor_ssh_launch_failed","not_transferred_before_preflight")
        return _validate_worker_aggregate(value,value["rows_total"])
    events=queue.Queue()
    def read_stdout():
        try:
            while True:
                line=proc.stdout.readline(config["max_response_bytes"]+1)
                if not line: events.put(None); return
                if len(line)>config["max_response_bytes"]: events.put(RuntimeError("remote line exceeded limit")); return
                events.put(line)
        except Exception as error: events.put(error)
    def drain_stderr():
        try:
            while proc.stderr.read(65536): pass
        except Exception: pass
    threading.Thread(target=read_stdout,daemon=True).start()
    threading.Thread(target=drain_stderr,daemon=True).start()
    sent=False
    try:
        first=events.get(timeout=max(30,config["host_snapshot_deadline_seconds"]+15))
        if first is None or isinstance(first,Exception):
            raise RuntimeError("isolated scope failed before preflight")
        preflight=_validate_scope_preflight(_json(first,"scope preflight"),config["remote_user"],run_id)
        proc.stdin.write(wrapped); proc.stdin.close(); sent=True
        result_line=events.get(timeout=timeout)
        if result_line is None or isinstance(result_line,Exception):
            raise RuntimeError("isolated scope returned no bounded aggregate")
        value=_json(result_line,"remote aggregate")
        proc.wait(timeout=10)
        if events.get(timeout=5) is not None: raise RuntimeError("unexpected extra remote output")
        if proc.returncode!=0: raise RuntimeError("isolated scope launcher returned failure")
        value["scope_preflight"]={k:preflight[k] for k in ("username","uid","gid","netns_inode","outer_netns_inode","scope_unit","capabilities_empty","no_new_privs","owner_groups_verified","scope_cgroup_observed")}
        worker_payload = _json(payload, "worker payload")
        return _validate_worker_aggregate(value, worker_payload["binding"]["rows"],
                                          _payload_output_text_byte_bucket_boundaries(worker_payload))
    except Exception as error:
        try: proc.stdin.close()
        except Exception: pass
        try: proc.wait(timeout=5)
        except Exception:
            try: proc.terminate(); proc.wait(timeout=3)
            except Exception:
                try: proc.kill()
                except Exception: pass
        timed_out=isinstance(error,(TimeoutError,queue.Empty,subprocess.TimeoutExpired))
        transfer=("unknown_after_client_timeout" if timed_out else "unknown_after_scope_failure") if sent else "not_transferred_before_preflight"
        reason="supervisor_client_timeout" if timed_out else ("supervisor_preflight_rejected" if not sent else "supervisor_scope_transport_or_lifecycle_failure")
        unit="lfg-owned-matched-"+run_id[:18]+".scope"
        cleanup=None
        try:
            cleanup_script=_remote_cleanup_script(unit,config["remote_home"])
            cleanup_command=_ssh_remote_command(config["remote_python_path"],
                "import base64;exec(base64.b64decode('"+base64.b64encode(cleanup_script.encode()).decode()+"'))")
            cleanup_args=["ssh","-T","-oBatchMode=yes","-oConnectTimeout=8","-p",str(config["ssh_port"]),
                          _ssh_destination(config["remote_user"],config["remote_host"]),cleanup_command]
            cleanup_result=subprocess.run(cleanup_args,capture_output=True,timeout=45,shell=False)
            if cleanup_result.returncode==0 and len(cleanup_result.stdout)<=4096: cleanup=_json(cleanup_result.stdout,"scope cleanup status")
        except Exception: cleanup=None
        value=_unknown_execution(_json(payload,"worker payload"),reason,transfer)
        value.update({"remote_unit":unit,"scope_cleanup":cleanup or {"verified":False,"unit_presence":"unknown"},
                      "unit_cleanup_verified":isinstance(cleanup,dict) and cleanup.get("verified") is True,
                      "remote_workspace_location_candidate":str(PurePosixPath(config["private_source_root"])/("lfg-owned-matched-"+run_id[:18]))})
        return _validate_worker_aggregate(value,value["rows_total"])


def _validate_owned_response_diagnostics(value: Any, total: int, boundaries: list[int],
                                         owned_requests_started: Any,
                                         expected_accepted_responses: int) -> None:
    stop_keys = {"stop_sequence", "byte_limit", "context_limit", "other", "unavailable"}
    shape_keys = {"valid_json_object", "valid_json_non_object", "invalid_json", "unavailable_or_invalid_type"}
    expected_keys = {"accepted_responses", "stop_reason_counts", "output_text_shape_counts",
                     "output_text_utf8_byte_bucket_counts", "output_text_utf8_bytes_unavailable_or_invalid"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("remote response diagnostics schema is invalid")
    accepted = value["accepted_responses"]
    if (type(accepted) is not int or not 0 <= accepted <= total
            or accepted != expected_accepted_responses
            or type(owned_requests_started) is not int or not accepted <= owned_requests_started <= total):
        raise ValueError("remote response diagnostics count is invalid")
    for field, keys in (("stop_reason_counts", stop_keys), ("output_text_shape_counts", shape_keys)):
        counts = value[field]
        if (not isinstance(counts, dict) or set(counts) != keys
                or any(type(count) is not int or not 0 <= count <= total for count in counts.values())
                or sum(counts.values()) != accepted):
            raise ValueError("remote response diagnostics categories are invalid")
    byte_buckets = value["output_text_utf8_byte_bucket_counts"]
    unavailable_bytes = value["output_text_utf8_bytes_unavailable_or_invalid"]
    if (not isinstance(byte_buckets, list) or len(byte_buckets) != len(boundaries) + 1
            or any(type(count) is not int or not 0 <= count <= total for count in byte_buckets)
            or type(unavailable_bytes) is not int or not 0 <= unavailable_bytes <= total
            or sum(byte_buckets) + unavailable_bytes != accepted):
        raise ValueError("remote byte diagnostics buckets are invalid")


def _validate_worker_aggregate(value: Any, total: int,
                               output_text_byte_bucket_boundaries: list[int] | None = None) -> dict[str, Any]:
    attempted = value.get("rows_attempted") if isinstance(value, dict) else None
    known_attempts = type(attempted) is int and 0 <= attempted <= total
    unknown_after_supervisor_failure = (isinstance(value, dict) and attempted is None
                                        and value.get("supervisor_failure") is True
                                        and isinstance(value.get("abort_reason"), str)
                                        and value["abort_reason"].startswith("supervisor_"))
    if (not isinstance(value, dict) or value.get("rows_total") != total
            or not (known_attempts or unknown_after_supervisor_failure)
            or type(value.get("aborted")) is not bool
            or (known_attempts and attempted != total and value["aborted"] is not True)
            or (unknown_after_supervisor_failure and value["aborted"] is not True)):
        raise ValueError("remote aggregate does not safely account for the bound validation rows")
    boundaries = _validated_output_text_byte_bucket_boundaries(output_text_byte_bucket_boundaries)
    if value.get("supervisor_failure") is True:
        # A supervisor or transport failure makes any worker-returned diagnostics untrusted.
        value["owned_response_diagnostics"] = None
    else:
        diag = value.get("owned_response_diagnostics")
        if type(value.get("owned_requests_started")) is not int or value["owned_requests_started"] > attempted:
            raise ValueError("remote owned request accounting is invalid")
        status_counts = value.get("owned_response_status_counts")
        allowed_statuses = {"ok", "truncated", "context_limit", "invalid_input", "malformed_output",
                            "incomplete_output", "timeout", "inference_error", "unavailable", "busy",
                            "no_terminal_response"}
        if (not isinstance(status_counts, dict) or not set(status_counts) <= allowed_statuses
                or any(type(count) is not int or not 0 <= count <= total for count in status_counts.values())
                or sum(status_counts.values()) != value["owned_requests_started"]):
            raise ValueError("remote owned response status accounting is invalid")
        accepted_responses = sum(count for status, count in status_counts.items()
                                 if status != "no_terminal_response")
        _validate_owned_response_diagnostics(diag, total, boundaries,
                                             value.get("owned_requests_started"), accepted_responses)
    return value


def _report(config: dict[str, Any], config_digest: str, bound: dict[str, Any], result: dict[str, Any],
            payload: bytes, code_sha256: str, output_dir: Path) -> dict[str, Any]:
    result = _validate_worker_aggregate(
        result, len(bound["validation_rows"]),
        _validated_output_text_byte_bucket_boundaries(
            config.get("output_text_utf8_byte_bucket_boundaries")
            if "output_text_utf8_byte_bucket_boundaries" in config else None))
    baseline = bound["classical_report"]
    report12 = bound["comparison_12b_report"]
    selected = sorted(bound["classical_config"]["selected_labels"])
    report = {
        "schema_version": REPORT_SCHEMA, "promotion_eligible": False, "qualification_claim": False,
        "task": {"field": bound["classical_config"]["target_field"], "labels": selected,
                 "taxonomy_sha256": bound["corpus"]["taxonomy_sha256"],
                 "validation_rows": len(bound["validation_rows"]),
                 "validation_record_id_set_sha256": bound["validation_id_set_sha256"],
                 "target_mask_sha256": bound["target_mask_sha256"],
                 "target_observation_support": baseline["task"]["validation_support"]},
        "baseline": {"classical_report_sha256": config["classical_report_sha256"],
                     "classical_config_sha256": bound["classical_config_sha256"],
                     "comparison_12b_report_sha256": config["comparison_12b_report_sha256"],
                     "comparison_12b_config_sha256": bound["comparison_12b_config_sha256"],
                     "comparison_12b_host": bound["comparison_12b_config"].get("remote_host"),
                     "comparison_12b_user": bound["comparison_12b_config"].get("remote_user"),
                     "classical_aggregate_metrics": {name: value.get("metrics") for name, value in baseline["algorithms"].items()},
                     "12b_aggregate_metrics": report12.get("metrics"),
                     "paired_owned_12b_intersection": "unavailable; report has no protected per-row predictions"},
        "deterministic": {"kind": "exhaustive_configured_rule_hit_behavior", **result.get("deterministic_metrics", {}),
                          "status_counts": result.get("deterministic_rule_status_counts"),
                          "effective_identity": result.get("deterministic_identity"),
                          "config_files_sha256": bound["deterministic_files_sha256"],
                          "assessment_sha256": bound["deterministic_assessment_sha256"],
                          "rules_sha256": bound["deterministic_rules_sha256"],
                          "latency_ms": result.get("deterministic_latency_ms")},
        "owned": {"mapping": result.get("owned_mapping"), "status_counts": result.get("owned_response_status_counts"),
                  "lifecycle_outcome_counts": result.get("owned_lifecycle_outcome_counts"),
                  "mapping_outcome_counts": result.get("owned_mapping_outcome_counts"),
                  "response_diagnostics": ({**result["owned_response_diagnostics"],
                                            "output_text_utf8_byte_bucket_boundaries":
                                                _validated_output_text_byte_bucket_boundaries(
                                                    config.get("output_text_utf8_byte_bucket_boundaries")
                                                    if "output_text_utf8_byte_bucket_boundaries" in config else None)}
                                           if result.get("owned_response_diagnostics") is not None else None),
                  "metrics": result.get("owned_metrics"), "engine_identity": result.get("owned_engine_identity"),
                  "configured_limits": result.get("owned_status_limits"),
                  "latency_ms": {"end_to_end": result.get("owned_end_to_end_latency_ms"),
                                 "handler_reported": result.get("owned_handler_elapsed_ms")}},
        "execution": {"rows_attempted": result.get("rows_attempted"), "rows_total": result.get("rows_total"),
                      "owned_host": config["remote_host"], "owned_user": config["remote_user"],
                      "rows_unattempted": ((result.get("rows_total", 0) - result["rows_attempted"])
                                           if type(result.get("rows_attempted")) is int else None),
                      "aborted": result.get("aborted"), "abort_reason": result.get("abort_reason"),
                      "supervisor_failure": result.get("supervisor_failure"),
                      "remote_workspace_retained": result.get("remote_workspace_retained"),
                      "remote_workspace_status": result.get("remote_workspace_status"),
                      "remote_workspace_location": result.get("remote_workspace_location"),
                      "remote_workspace_location_candidate": result.get("remote_workspace_location_candidate"),
                      "worker_cleanup": result.get("worker_cleanup"),
                      "scope_cleanup": result.get("scope_cleanup"),
                      "scope_cleanup_verified": result.get("scope_cleanup_verified"),
                      "unit_cleanup_verified": result.get("unit_cleanup_verified"),
                      "owned_service_cleanup_verified": result.get("owned_service_cleanup_verified"),
                      "owned_listener_closed_verified": result.get("owned_listener_closed_verified"),
                      "worker_input_identity_verified": result.get("worker_input_identity_verified"),
                      "worker_validation_record_id_set_sha256": result.get("worker_validation_record_id_set_sha256"),
                      "worker_target_mask_sha256": result.get("worker_target_mask_sha256"),
                      "supervisor_payload_sha256": result.get("supervisor_payload_sha256"),
                      "deterministic_requests_started": result.get("deterministic_requests_started"),
                      "owned_requests_started": result.get("owned_requests_started"),
                      "single_flight": True, "owned_deadline_seconds": config["per_request_deadline_seconds"],
                      "file_preflight": result.get("file_preflight"),
                      "isolated_network": result.get("namespace"),
                      "runtime_identity": result.get("runtime"),
                      "worker_resource_sample": result.get("worker_resource_sample"),
                      "host_before": result.get("host_before"), "host_after": result.get("host_after"),
                      "coexisting_service_snapshots_match": result.get("coexisting_service_snapshots_match"),
                      "coexisting_service_intervening_state": result.get("coexisting_service_intervening_state"),
                      "service_cleanup_verified": result.get("cleanup_verified"),
                      "service_exit_code": result.get("service_exit_code"),
                      "unit_exit_code": result.get("unit_exit_code"),
                      "payload_sha256": sha(payload), "remote_payload_sha256": result.get("worker_payload_sha256")},
        "provenance": {"screened_manifest_sha256": bound["corpus"]["manifest_sha256"],
                       "source_service_template_sha256": config["service_config_template_sha256"],
                       "plan_sha256": baseline["data"]["plan_sha256"],
                       "screening_policy_sha256": bound["corpus"]["policy_sha256"],
                       "screening_code_sha256": bound["corpus"]["screen_code_sha256"],
                       "review_ledger_sha256": baseline["data"]["review_ledger_sha256"],
                       "train_sha256": bound["corpus"]["train_sha256"],
                       "validation_sha256": bound["corpus"]["validation_sha256"],
                       "baseline_review_and_history_claims": bound["review_provenance"],
                       "redaction_complete": False,
                       "payload_transfer_state": result.get("payload_transfer_state", "received_by_isolated_supervisor"),
                       "private_input_policy": "no public/cloud endpoint configured; screened payload is staged in a private file after host preflight and service stdout/stderr are discarded",
                       "report_content_policy": "aggregate-only; report has no raw prompt, owned output, token, or per-row prediction",
                       "remote_persistence_state": ("worker aggregate received; workspace cleanup is reported separately"
                                                    if result.get("supervisor_failure") is not True
                                                    else "unknown because supervisor or transfer did not return a verified worker aggregate")},
        "limitations": ["development validation labels and history retain the pinned baseline's review state and claims; they are not independently verified here",
                        "this set is development validation, not independent final evaluation",
                        "owned framing and output schema do not prove exact LFG intent taxonomy, so owned predictions remain unknown",
                        "deterministic metrics are rule-hit outcomes, not learned-model quality",
                        "the 12B aggregate report lacks per-row predictions, so paired intersections cannot be computed",
                        "no heldout data were opened and no training or distillation was run"],
        "config_sha256": config_digest,
        "comparator_code_sha256": code_sha256,
        "comparator_git_revision": _git_revision(),
    }
    return report


def _git_revision() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    relative = Path(__file__).resolve().relative_to(repo).as_posix()
    try:
        tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", relative],
                                 check=True, capture_output=True, text=True, timeout=10, shell=False)
        status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", relative],
                                check=True, capture_output=True, text=True, timeout=10, shell=False)
        if not tracked.stdout.strip() or status.stdout.strip():
            return None
        result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True, timeout=10, shell=False)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run(config_path: Path, output: Path, probe_path: Path | None = None) -> dict[str, Any]:
    global CURRENT_STAGE
    CURRENT_STAGE = "configuration"
    config, config_digest = load_config(config_path)
    probe = None
    if probe_path is not None:
        probe = _validate_probe(
            _read_bounded(probe_path, config["report_max_bytes"], "probe report"), config_digest,
            _validated_output_text_byte_bucket_boundaries(
                config.get("output_text_utf8_byte_bucket_boundaries")
                if "output_text_utf8_byte_bucket_boundaries" in config else None))
    CURRENT_STAGE = "baseline_binding"
    bound = _bind_inputs(config)
    run_id = uuid.uuid4().hex
    payload = _make_payload(config, bound, run_id)
    CURRENT_STAGE = "private_report_directory"
    output_dir = _private_report_directory(output)
    try:
        CURRENT_STAGE = "isolated_remote_evaluation"
        result = (_file_evaluate(config, payload, run_id, output_dir) if probe is not None
                  else _ssh_worker(config, payload, run_id))
        if result.get("payload_transfer_state") in ("unknown_after_client_timeout", "unknown_after_scope_failure", "not_transferred_before_preflight", "unknown_after_file_transfer"):
            if result.get("worker_payload_sha256") is not None:
                raise ValueError("unconfirmed SSH transfer cannot carry a remote payload receipt")
        elif result.get("worker_payload_sha256") is not None:
            _validate_payload_identity(payload, result.get("worker_payload_sha256"))
        elif result.get("supervisor_payload_sha256") is not None:
            _validate_payload_identity(payload, result.get("supervisor_payload_sha256"))
        else:
            raise ValueError("confirmed transfer has no observed supervisor or worker payload digest")
        supervisor_failure = result.get("supervisor_failure") is True
        if (not supervisor_failure and (result.get("cleanup_verified") is not True
                or result.get("coexisting_service_snapshots_match") is not True)):
            raise ValueError("isolated service cleanup or point-in-time coexistence snapshots were not verified")
        if (not supervisor_failure
                and result.get("runtime", {}).get("binary_sha256") != config["service_binary_sha256"]):
            raise ValueError("owned service binary identity changed during run")
        CURRENT_STAGE = "private_report_write"
        report = _report(config, config_digest, bound, result, payload, sha(Path(__file__).read_bytes()), output_dir)
        report["provenance"]["file_protocol_probe"] = probe
        encoded = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        if len(encoded) > config["report_max_bytes"]:
            raise ValueError("private report exceeds configured byte limit")
        target = output_dir / "owned-matched-validation.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(target, flags, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return {"report_path": str(target), "report_sha256": sha(encoded),
                "rows": report["task"]["validation_rows"],
                "owned_coverage": report["owned"]["metrics"].get("coverage"),
                "deterministic_coverage": report["deterministic"].get("coverage"),
                "report": report}
    except Exception:
        # Do not leave an empty/misleading report directory after a failed run.
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="new owner-private output directory outside Git")
    parser.add_argument("--file-probe", action="store_true",
                        help="run the empty production file protocol after target-host coordination")
    parser.add_argument("--probe-report", type=Path, help="successful private no-input probe report")
    args = parser.parse_args()
    try:
        if args.file_probe:
            if args.probe_report:
                raise ValueError("probe does not accept a probe report")
            result = run_file_probe(args.config, args.output)
            print(json.dumps(result, sort_keys=True))
            if not result["passed"]:
                raise SystemExit(1)
            return
        result = run(args.config, args.output, args.probe_report)
    except Exception:
        print(json.dumps({"stage": CURRENT_STAGE, "error": "matched owned evaluation failed; private details suppressed"}), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({k: result[k] for k in ("report_path", "report_sha256", "rows", "owned_coverage", "deterministic_coverage")}, sort_keys=True))


if __name__ == "__main__":
    main()
