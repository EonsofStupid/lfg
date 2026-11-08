#!/usr/bin/env python3
"""Run a private, local-only configured model comparison on a bound dev split."""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import math
import os
import platform
import queue
import csv
import re
import shlex
import shutil
import socket
import statistics
import stat
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "lfg.local-12b-comparison-config.v2"
REPORT_SCHEMA = "lfg.local-12b-comparison-report.v1"
RECEIPT_SCHEMA = "lfg.private-acquisition-receipt.v1"
HEX = re.compile(r"^[0-9a-f]{64}$")
MAX_CONFIG_BYTES = 65_536
CURRENT_STAGE = "configuration"


class RequestLimitError(ValueError):
    pass


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"invalid {label} JSON") from None


def _read_bounded(path: Path, limit: int, label: str) -> bytes:
    if type(limit) is not int or limit < 1:
        raise ValueError(f"invalid {label} size limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"{label} exceeds configured size limit")
    return raw


def _safe_absolute_posix_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (path.is_absolute() and ".." not in path.parts
            and bool(re.fullmatch(r"/[A-Za-z0-9_./ -]+", value)))


def _within_remote_home(path: str, home: str) -> bool:
    value, root = PurePosixPath(path), PurePosixPath(home)
    return (_safe_absolute_posix_path(path) and root != PurePosixPath("/")
            and len(root.parts) >= 3 and value != root and value.is_relative_to(root))


def _valid_ssh_host(host: str) -> bool:
    return isinstance(host, str) and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", host) is not None


def _valid_ssh_user(user: str) -> bool:
    return isinstance(user, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", user) is not None


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    cfg_raw = _read_bounded(path, MAX_CONFIG_BYTES, "configuration")
    cfg = _json(cfg_raw, "configuration")
    required = {
        "schema_version", "revision", "baseline_report", "baseline_report_sha256",
        "classical_config", "classical_config_sha256", "acquisition_receipt",
        "acquisition_receipt_sha256", "remote_acquisition_receipt", "model_store", "model_name",
        "server_binary_sha256", "server_version", "endpoint_url", "keep_alive", "max_report_bytes",
        "max_baseline_report_bytes", "max_source_bytes", "max_total_source_bytes",
        "max_manifest_bytes", "max_screened_file_bytes", "max_policy_bytes",
        "max_ledger_bytes", "max_model_manifest_bytes", "max_model_blob_bytes",
        "hash_chunk_bytes", "max_server_binary_bytes", "min_memory_available_bytes", "min_disk_free_bytes",
        "min_gpu_free_bytes", "num_ctx", "num_predict", "timeout_seconds",
        "max_request_bytes", "max_response_bytes", "system_prompt", "prompt_revision",
        "seed", "temperature", "top_k", "top_p", "repeat_penalty", "remote_host",
        "remote_user", "remote_workdir", "remote_model_store", "remote_adapter_path",
        "remote_python_path", "server_binary_path", "runtime_port", "ssh_port",
        "worker_start_timeout_seconds", "resource_memory_max_bytes", "resource_cpu_quota_percent",
        "max_payload_bytes", "remote_uid",
        "gpu_memory_policy", "expected_upstream_revision", "expected_registry_manifest_url",
        "expected_manifest_sha256", "expected_model_blob_sha256", "expected_model_blob_bytes",
        "expected_license_name", "expected_quantization", "remote_home", "remote_system_service",
        "remote_systemctl_path", "remote_sudo_path", "remote_systemd_run_path", "remote_unshare_path",
        "remote_bash_path", "remote_ip_path", "remote_setpriv_path", "remote_true_path",
        "remote_scope_tasks_max", "worker_ready_wait_margin_seconds",
        "artifact_install_timeout_seconds", "batch_overhead_margin_seconds",
        "remote_expected_hostname", "remote_expected_architecture",
        "remote_nvidia_smi_path",
    }
    if not isinstance(cfg, dict) or not required <= set(cfg) or cfg.get("schema_version") != SCHEMA:
        raise ValueError("invalid comparison configuration")
    for field in ("revision", "baseline_report", "classical_config", "acquisition_receipt",
                  "model_store", "model_name", "server_version", "endpoint_url", "keep_alive", "system_prompt",
                  "prompt_revision", "remote_acquisition_receipt", "remote_host", "remote_user",
                  "remote_workdir", "remote_model_store", "remote_adapter_path", "remote_python_path",
                  "server_binary_path", "gpu_memory_policy", "expected_upstream_revision",
                  "expected_registry_manifest_url", "expected_license_name", "expected_quantization",
                  "remote_home", "remote_system_service", "remote_systemctl_path", "remote_sudo_path",
                  "remote_systemd_run_path", "remote_unshare_path", "remote_bash_path", "remote_ip_path",
                  "remote_setpriv_path", "remote_true_path", "remote_expected_hostname",
                  "remote_expected_architecture", "remote_nvidia_smi_path"):
        if not isinstance(cfg[field], str) or not cfg[field].strip():
            raise ValueError(f"invalid {field}")
    for field in ("baseline_report_sha256", "classical_config_sha256", "acquisition_receipt_sha256",
                  "server_binary_sha256", "expected_manifest_sha256", "expected_model_blob_sha256"):
        if not isinstance(cfg[field], str) or not HEX.fullmatch(cfg[field]):
            raise ValueError(f"invalid {field}")
    for field in ("max_report_bytes", "max_baseline_report_bytes", "max_source_bytes",
                  "max_total_source_bytes", "max_manifest_bytes", "max_screened_file_bytes",
                  "max_policy_bytes", "max_ledger_bytes", "max_model_manifest_bytes",
                  "max_model_blob_bytes", "hash_chunk_bytes", "max_server_binary_bytes", "min_memory_available_bytes",
                  "min_disk_free_bytes", "num_ctx", "num_predict",
                  "max_request_bytes", "max_response_bytes", "worker_start_timeout_seconds",
                  "resource_memory_max_bytes", "resource_cpu_quota_percent", "max_payload_bytes", "remote_uid",
                  "expected_model_blob_bytes", "remote_scope_tasks_max", "worker_ready_wait_margin_seconds",
                  "artifact_install_timeout_seconds", "batch_overhead_margin_seconds"):
        if type(cfg[field]) is not int or cfg[field] < 1:
            raise ValueError(f"invalid {field}")
        if field in {"worker_start_timeout_seconds", "worker_ready_wait_margin_seconds",
                     "artifact_install_timeout_seconds", "batch_overhead_margin_seconds"} \
                and cfg[field] > sys.maxsize // 1_000_000_000:
            raise ValueError(f"{field} exceeds safe platform timeout arithmetic")
    if type(cfg["runtime_port"]) is not int or not 1024 <= cfg["runtime_port"] <= 65535:
        raise ValueError("invalid runtime_port")
    if type(cfg["ssh_port"]) is not int or not 1 <= cfg["ssh_port"] <= 65535:
        raise ValueError("invalid ssh_port")
    if type(cfg["min_gpu_free_bytes"]) is not int or cfg["min_gpu_free_bytes"] < 0:
        raise ValueError("invalid min_gpu_free_bytes")
    if cfg["min_gpu_free_bytes"] == 0 and cfg.get("gpu_memory_policy") != "shared_unified_memory_unavailable":
        raise ValueError("unknown GPU memory requires an explicit shared-memory policy")
    if cfg["resource_cpu_quota_percent"] > sys.maxsize // 100:
        raise ValueError("resource CPU quota exceeds safe systemd arithmetic")
    safe_seconds = sys.maxsize / 1_000_000_000
    if (cfg["worker_start_timeout_seconds"] + cfg["worker_ready_wait_margin_seconds"] > safe_seconds
            or cfg["artifact_install_timeout_seconds"] + 10 > safe_seconds):
        raise ValueError("combined timeout exceeds safe platform arithmetic")
    for field in ("resource_memory_max_bytes", "max_model_blob_bytes", "remote_scope_tasks_max"):
        if cfg[field] > sys.maxsize:
            raise ValueError(f"{field} exceeds safe platform arithmetic")
    timeout = cfg["timeout_seconds"]
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
            or timeout > sys.maxsize / 1_000_000_000):
        raise ValueError("invalid timeout_seconds")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9._-]+)?", cfg["model_name"]):
        raise ValueError("invalid configured model identity")
    revision = cfg["expected_upstream_revision"]
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision):
        raise ValueError("invalid configured upstream revision")
    registry_url = urlsplit(cfg["expected_registry_manifest_url"])
    if (registry_url.scheme != "https" or registry_url.netloc != "registry.ollama.ai"
            or not registry_url.path.startswith("/v2/") or "/manifests/" not in registry_url.path
            or registry_url.query or registry_url.fragment):
        raise ValueError("invalid configured registry manifest identity")
    if ":" not in cfg["model_name"]:
        raise ValueError("configured model identity must include an immutable tag")
    repository, model_tag = cfg["model_name"].rsplit(":", 1)
    if "/" not in repository:
        repository = "library/" + repository
    if (repository.count("/") != 1
            or registry_url.path != f"/v2/{repository}/manifests/{model_tag}"):
        raise ValueError("registry manifest URL does not match the configured model identity")
    if cfg["expected_model_blob_bytes"] > cfg["max_model_blob_bytes"]:
        raise ValueError("model blob size pin exceeds configured artifact byte limit")
    if (not _safe_absolute_posix_path(cfg["remote_home"])
            or not _within_remote_home(cfg["remote_acquisition_receipt"], cfg["remote_home"])
            or any(not _safe_absolute_posix_path(cfg[field]) for field in
                   ("remote_workdir", "remote_model_store", "remote_adapter_path", "remote_python_path",
                    "server_binary_path", "remote_systemctl_path", "remote_sudo_path",
                    "remote_systemd_run_path", "remote_unshare_path", "remote_bash_path", "remote_ip_path",
                    "remote_setpriv_path", "remote_true_path", "remote_nvidia_smi_path"))
            or not _within_remote_home(cfg["remote_workdir"], cfg["remote_home"])
            or not _within_remote_home(cfg["remote_model_store"], cfg["remote_home"])
            or not _within_remote_home(cfg["remote_adapter_path"], cfg["remote_home"])):
        raise ValueError("remote paths must be safe absolute paths inside the configured user home")
    if not (_valid_ssh_user(cfg["remote_user"]) and _valid_ssh_host(cfg["remote_host"])):
        raise ValueError("invalid SSH user or host")
    if not _valid_ssh_host(cfg["remote_expected_hostname"]):
        raise ValueError("invalid configured remote hostname identity")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cfg["remote_expected_architecture"]):
        raise ValueError("invalid configured remote architecture identity")
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", cfg["remote_system_service"]):
        raise ValueError("invalid configured service unit name")
    if "{text}" not in cfg["system_prompt"] or "{labels}" not in cfg["system_prompt"]:
        raise ValueError("system_prompt must contain {text} and {labels}")
    if cfg["num_predict"] < 2:
        raise ValueError("num_predict must permit a complete response")
    if type(cfg["seed"]) is not int or type(cfg["top_k"]) is not int or cfg["top_k"] < 1:
        raise ValueError("invalid deterministic sampling configuration")
    for field, low, high in (("temperature", 0.0, None), ("top_p", 0.0, 1.0),
                             ("repeat_penalty", 0.0, None)):
        value = cfg[field]
        if (type(value) not in (int, float) or not math.isfinite(value) or value < low
                or (high is not None and value > high)
                or (field == "repeat_penalty" and value == 0)):
            raise ValueError(f"invalid {field}")
    return cfg, sha(cfg_raw)


def output_schema(labels: list[str]) -> dict[str, Any]:
    if not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("invalid selected labels")
    if len(labels) != len(set(labels)):
        raise ValueError("selected labels are not unique")
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "labels": {"type": "array", "items": {"type": "string", "enum": labels},
                       "uniqueItems": True},
            "abstain": {"type": "boolean"},
        },
        "required": ["labels", "abstain"],
    }


def parse_prediction(raw: str, done: Any, done_reason: Any, allowed: set[str]) -> tuple[str, list[str] | None]:
    if done is not True:
        return "incomplete", None
    if done_reason == "length":
        return "truncated", None
    if done_reason not in (None, "stop", "eos"):
        return "incomplete", None
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return "malformed", None
    if isinstance(value, dict) and (value.get("refusal") or value.get("refused") is True):
        return "refusal", None
    if (not isinstance(value, dict) or set(value) != {"labels", "abstain"}
            or not isinstance(value.get("labels"), list) or type(value.get("abstain")) is not bool
            or any(not isinstance(label, str) or label not in allowed for label in value["labels"])
            or len(value["labels"]) != len(set(value["labels"]))):
        return "invalid_schema", None
    if value["abstain"]:
        return ("abstain", None) if not value["labels"] else ("invalid_schema", None)
    return "answered", value["labels"]


def _loopback_endpoint(url: str) -> tuple[str, int, str]:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("endpoint must be plain HTTP to a literal loopback address")
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
    except ValueError:
        raise ValueError("endpoint hostname must be a literal loopback IP") from None
    if not address.is_loopback:
        raise ValueError("endpoint must be loopback-only")
    port = parsed.port or 80
    if not (1 <= port <= 65535):
        raise ValueError("invalid endpoint port")
    path = parsed.path or "/"
    if path != "/api/chat":
        raise ValueError("endpoint path must be Ollama /api/chat")
    return address.compressed, port, path


def _read_proc_status(pid: int) -> dict[str, str]:
    fields = {}
    for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key] = value.strip()
    return fields


def _process_rss(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _descendant_pids(root_pid: int) -> list[int]:
    parents = {}
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            values = _read_proc_status(int(status_path.parent.name))
            parents[int(status_path.parent.name)] = int(values.get("PPid", "-1"))
        except (OSError, ValueError):
            continue
    found, pending = {root_pid}, [root_pid]
    while pending:
        parent = pending.pop()
        children = [pid for pid, ppid in parents.items() if ppid == parent and pid not in found]
        found.update(children)
        pending.extend(children)
    return sorted(found)


def _route_is_private() -> bool:
    try:
        if set(name for _, name in socket.if_nameindex()) - {"lo"}:
            return False
        if not (int(Path("/sys/class/net/lo/flags").read_text(encoding="ascii").strip(), 16) & 1):
            return False
        for line in Path("/proc/net/route").read_text(encoding="ascii").splitlines()[1:]:
            parts = line.split()
            if parts and parts[0] != "lo":
                return False
        for line in Path("/proc/net/ipv6_route").read_text(encoding="ascii").splitlines():
            parts = line.split()
            if parts and parts[-1] != "lo":
                return False
    except OSError:
        return False
    return True


def _private_directory(path: Path) -> bool:
    try:
        st = path.stat()
        return (path.is_dir() and st.st_uid == os.geteuid() and (st.st_mode & 0o077) == 0
                and not path.is_symlink())
    except OSError:
        return False


def verify_runtime_boundary(config: dict[str, Any]) -> dict[str, Any]:
    if not sys.platform.startswith("linux") or not Path("/proc/self/ns/net").exists():
        raise ValueError("verified Linux network namespace is required")
    address, port, _ = _loopback_endpoint(config["endpoint_url"])
    if not _route_is_private():
        raise ValueError("network namespace has non-loopback interfaces or routes")
    pid = config["server_pid"]
    self_status, server_status = _read_proc_status(os.getpid()), _read_proc_status(pid)
    if self_status.get("Uid", "").split()[0] != server_status.get("Uid", "").split()[0]:
        raise ValueError("server process is not owned by the invoking user")
    if os.stat("/proc/self/ns/net").st_ino != os.stat(f"/proc/{pid}/ns/net").st_ino:
        raise ValueError("server and client do not share the private network namespace")
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").lower()
    if b"ollama" not in cmdline or b"serve" not in cmdline:
        raise ValueError("expected isolated Ollama server process was not found")
    env = {}
    for entry in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        key, sep, value = entry.partition(b"=")
        if sep and key in (b"OLLAMA_NO_CLOUD", b"OLLAMA_MODELS", b"OLLAMA_HOST"):
            env[key.decode("ascii")] = value.decode("utf-8", "strict")
    model_store_path = Path(config["model_store"])
    if model_store_path.is_symlink():
        raise ValueError("model store must not be a symlink")
    model_store = model_store_path.resolve(strict=True)
    if (env.get("OLLAMA_NO_CLOUD") != "1"
            or Path(env.get("OLLAMA_MODELS", "")).resolve(strict=True) != model_store
            or env.get("OLLAMA_HOST") not in (f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}")
            or not _private_directory(model_store)):
        raise ValueError("Ollama server is not configured for the verified private model store and loopback")
    binary = Path(f"/proc/{pid}/exe").resolve(strict=True)
    if binary != Path(config["server_binary_path"]).resolve(strict=True):
        raise ValueError("isolated Ollama executable path differs from configured path")
    digest, _ = _hash_file(binary, config["max_server_binary_bytes"], config["hash_chunk_bytes"])
    if digest != config["server_binary_sha256"]:
        raise ValueError("Ollama binary digest mismatch")
    try:
        result = subprocess.run([str(binary), "--version"], check=True, capture_output=True,
                                text=True, timeout=10, shell=False)
    except (OSError, subprocess.SubprocessError):
        raise ValueError("Ollama version could not be verified") from None
    if config["server_version"] not in result.stdout + result.stderr:
        raise ValueError("Ollama version does not match configured version")
    rss = [_process_rss(item) for item in _descendant_pids(pid)]
    return {"boundary": "same_uid_private_network_namespace_without_nonloopback_routes",
            "server_pid": pid, "server_binary_sha256": digest,
            "server_version": config["server_version"],
            "network_namespace_inode": os.stat(f"/proc/{pid}/ns/net").st_ino,
            "process_tree_rss_bytes": sum(value for value in rss if value is not None),
            "process_tree_rss_scope": "server and observed descendants; summed RSS can double-count shared pages"}


def resource_snapshot(config: dict[str, Any], enforce: bool = True) -> dict[str, Any]:
    mem_available = None
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            mem_available = int(line.split()[1]) * 1024
            break
    disk_free = shutil.disk_usage(config["model_store"]).free
    gpu_free = None
    gpu_devices = []
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.free,memory.total",
                                 "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=5, shell=False)
        values = []
        for line in result.stdout.splitlines():
            name, free, total = (part.strip() for part in line.split(",", 2))
            free_bytes, total_bytes = int(free) * 1024 * 1024, int(total) * 1024 * 1024
            values.append(free_bytes)
            gpu_devices.append({"name": name, "free_bytes": free_bytes, "total_bytes": total_bytes})
        gpu_free = min(values) if values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        gpu_free = None
    cgroup_memory_available = None
    try:
        limit_raw = Path("/sys/fs/cgroup/memory.max").read_text(encoding="ascii").strip()
        current_raw = Path("/sys/fs/cgroup/memory.current").read_text(encoding="ascii").strip()
        if limit_raw != "max":
            cgroup_memory_available = max(0, int(limit_raw) - int(current_raw))
    except (OSError, ValueError):
        pass
    effective_available = min(value for value in (mem_available, cgroup_memory_available) if value is not None) if any(
        value is not None for value in (mem_available, cgroup_memory_available)) else None
    if enforce and (effective_available is None or effective_available < config["min_memory_available_bytes"]):
        raise ValueError("current memory availability is below the configured coexistence floor")
    if enforce and disk_free < config["min_disk_free_bytes"]:
        raise ValueError("current disk availability is below the configured coexistence floor")
    if enforce and config["min_gpu_free_bytes"] > 0 and (gpu_free is None or gpu_free < config["min_gpu_free_bytes"]):
        raise ValueError("GPU availability is unknown or below the configured coexistence floor")
    if enforce and gpu_free is None and config["gpu_memory_policy"] != "shared_unified_memory_unavailable":
        raise ValueError("GPU memory is unknown and no shared-memory policy was declared")
    return {"host_memory_available_bytes": mem_available,
            "cgroup_memory_available_bytes": cgroup_memory_available,
            "effective_memory_available_bytes": effective_available,
            "model_store_disk_free_bytes": disk_free,
            "minimum_gpu_free_bytes_across_devices": gpu_free,
            "gpu_devices": gpu_devices,
            "gpu_attribution": ("system-wide free memory only; no per-process GPU attribution"
                                if gpu_free is not None else
                                "unknown; platform did not expose GPU memory and unified shared-memory policy was declared")}


def _hash_file(path: Path, limit: int, chunk_size: int) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("model artifact is not a regular file")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(min(chunk_size, limit + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError("model artifact exceeds configured byte limit")
            digest.update(chunk)
    return digest.hexdigest(), size


def _no_symlink_child(root: Path, *parts: str) -> Path:
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("model store contains a symbolic link")
    try:
        current.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        raise ValueError("model store path escapes its private root") from None
    return current


def verify_acquisition(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    receipt_path = Path(config["acquisition_receipt"]).resolve(strict=True)
    receipt_raw = _read_bounded(receipt_path, config["max_report_bytes"], "acquisition receipt")
    receipt_digest = sha(receipt_raw)
    if receipt_digest != config["acquisition_receipt_sha256"]:
        raise ValueError("acquisition receipt digest mismatch")
    receipt = _json(receipt_raw, "acquisition receipt")
    if (not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("outcome") != "verified_downloaded_not_loaded"):
        raise ValueError("acquisition receipt is not a verified download receipt")
    manifest = receipt.get("manifest")
    if (not isinstance(manifest, dict) or manifest.get("sha256") != config["expected_manifest_sha256"]
            or type(manifest.get("bytes")) is not int or not isinstance(manifest.get("file"), str)):
        raise ValueError("receipt manifest does not match the configured artifact identity")
    objects = receipt.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("receipt lacks verified artifact objects")
    model = [item for item in objects if isinstance(item, dict)
             and isinstance(item.get("digest"), str)
             and item["digest"].removeprefix("sha256:") == config["expected_model_blob_sha256"]]
    if (len(model) != 1 or model[0].get("bytes") != config["expected_model_blob_bytes"]
            or model[0].get("sha256_verified") is not True):
        raise ValueError("receipt lacks the configured verified model layer")
    if (not isinstance(receipt.get("sources"), dict) or not isinstance(receipt.get("provenance"), dict)
            or not isinstance(receipt.get("staging_locator"), str)):
        raise ValueError("acquisition receipt provenance is incomplete")
    sources = receipt["sources"]
    registry = sources.get("registry_manifest")
    if (sources.get("upstream_model_revision") != config["expected_upstream_revision"]
            or not isinstance(registry, str)
            or registry != config["expected_registry_manifest_url"]):
        raise ValueError("receipt does not bind the configured upstream and registry identities")
    return receipt, receipt_digest


def verify_installed_artifact(config: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    store_configured = Path(config["model_store"])
    if store_configured.is_symlink() or not _private_directory(store_configured):
        raise ValueError("model store is not owner-private")
    store = store_configured.resolve(strict=True)
    registry = urlsplit(config["expected_registry_manifest_url"])
    manifest_parts = registry.path.strip("/").split("/")
    if len(manifest_parts) != 5 or manifest_parts[0] != "v2" or manifest_parts[3] != "manifests":
        raise ValueError("configured registry manifest path is not a supported OCI layout")
    manifest_file = _no_symlink_child(store, "manifests", registry.netloc, *manifest_parts[1:3],
                                      manifest_parts[4])
    manifest_sha, manifest_size = _hash_file(manifest_file, config["max_model_manifest_bytes"],
                                             config["hash_chunk_bytes"])
    if manifest_sha != receipt["manifest"]["sha256"] or manifest_size != receipt["manifest"]["bytes"]:
        raise ValueError("installed model manifest differs from the private acquisition receipt")
    manifest = _json(_read_bounded(manifest_file, config["max_model_manifest_bytes"], "model manifest"),
                     "model manifest")
    layers = manifest.get("layers")
    model_config = manifest.get("config")
    if not isinstance(layers, list) or not isinstance(model_config, dict):
        raise ValueError("invalid installed model manifest")
    expected = {item["digest"].removeprefix("sha256:"): item for item in receipt["objects"]
                if isinstance(item, dict) and isinstance(item.get("digest"), str)
                and HEX.fullmatch(item["digest"].removeprefix("sha256:"))}
    entries = [model_config] + layers
    actual_digests = set()
    for entry in entries:
        digest = entry.get("digest", "").removeprefix("sha256:")
        if not HEX.fullmatch(digest) or digest in actual_digests or digest not in expected:
            raise ValueError("installed manifest layer is missing from acquisition receipt")
        actual_digests.add(digest)
        receipt_item = expected[digest]
        if (receipt_item.get("sha256_verified") is not True or type(entry.get("size")) is not int
                or entry["size"] != receipt_item.get("bytes")):
            raise ValueError("installed manifest layer size or digest receipt is invalid")
        blob_path = _no_symlink_child(store, "blobs", f"sha256-{digest}")
        blob_sha, blob_size = _hash_file(blob_path, config["max_model_blob_bytes"], config["hash_chunk_bytes"])
        if blob_sha != digest or blob_size != entry["size"]:
            raise ValueError("installed model layer digest or byte count mismatch")
    if actual_digests != set(expected):
        raise ValueError("receipt and installed manifest layer sets differ")
    license_info = receipt.get("license")
    license_sha = (license_info.get("sha256", "").removeprefix("sha256:")
                   if isinstance(license_info, dict) and isinstance(license_info.get("sha256"), str) else "")
    if (not isinstance(license_info, dict)
            or license_info.get("identified_as") != config["expected_license_name"]
            or license_info.get("notice_retained_in_verified_blob") is not True
            or license_sha not in actual_digests):
        raise ValueError("license notice or identification is not verified")
    return {"model_name": config["model_name"], "manifest_sha256": manifest_sha,
            "manifest_bytes": manifest_size,
            "verified_layer_count": len(actual_digests),
            "model_layer_sha256": config["expected_model_blob_sha256"],
            "model_layer_bytes": config["expected_model_blob_bytes"],
            "license": config["expected_license_name"],
            "quantization": config["expected_quantization"]}


def _http_call(config: dict[str, Any], body: bytes | None, method: str = "POST",
               inventory: bool = False, path_override: str | None = None) -> tuple[int, bytes]:
    host, port, path = _loopback_endpoint(config["endpoint_url"])
    if inventory:
        path = "/api/ps"
    if path_override is not None:
        if path_override not in ("/api/version", "/api/ps", "/api/chat"):
            raise ValueError("unsupported internal Ollama path")
        path = path_override
    if body is not None and len(body) > config["max_request_bytes"]:
        raise RequestLimitError("model request exceeds configured byte limit")
    timeout_seconds = float(config["timeout_seconds"])
    deadline = time.monotonic() + timeout_seconds
    connection = http.client.HTTPConnection(host, port, timeout=timeout_seconds)
    expired = threading.Event()

    def enforce_deadline() -> None:
        expired.set()
        sock = connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    watchdog = threading.Timer(timeout_seconds, enforce_deadline)
    watchdog.daemon = True
    watchdog.start()
    try:
        headers = {"Accept": "application/json", "Connection": "close"}
        if body is not None:
            headers.update({"Content-Type": "application/json", "Content-Length": str(len(body))})
        connection.request(method, path, body=body, headers=headers)
        if expired.is_set():
            raise TimeoutError("model endpoint deadline exceeded")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("model endpoint deadline exceeded")
        if connection.sock is not None:
            connection.sock.settimeout(remaining)
        response = connection.getresponse()
        if 300 <= response.status < 400:
            raise ValueError("model endpoint redirect refused")
        if response.status != 200:
            return response.status, b""
        raw = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("model endpoint deadline exceeded")
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            chunk = response.read1(min(65_536, config["max_response_bytes"] + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > config["max_response_bytes"]:
                raise ValueError("model response exceeds configured byte limit")
        if time.monotonic() > deadline:
            raise TimeoutError("model endpoint deadline exceeded")
        return response.status, bytes(raw)
    except (OSError, http.client.HTTPException):
        if expired.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("model endpoint deadline exceeded") from None
        raise
    finally:
        watchdog.cancel()
        connection.close()


def verify_loaded_model(config: dict[str, Any], artifact_identity: dict[str, Any]) -> dict[str, Any]:
    status, raw = _http_call(config, None, method="GET", inventory=True)
    if status != 200:
        raise ValueError("isolated runtime model inventory request failed")
    value = _json(raw, "runtime model inventory")
    models = value.get("models") if isinstance(value, dict) else None
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise ValueError("isolated runtime must have exactly one loaded model")
    model = models[0]
    digest = str(model.get("digest", "")).removeprefix("sha256:")
    reported_size = model.get("size")
    if (model.get("name") != config["model_name"] or digest != artifact_identity["manifest_sha256"]
            or type(reported_size) is not int or reported_size < artifact_identity["model_layer_bytes"]):
        raise ValueError("loaded model identity does not match the verified artifact")
    return {"reported_model_name": model["name"], "reported_manifest_sha256": digest,
            "reported_size_bytes": reported_size,
            "reported_size_scope": "runtime API size; not weight layer size or process resident memory",
            "identity_match": True}


def _schema_request(config: dict[str, Any], label_schema: dict[str, Any], screened_text: str) -> bytes:
    labels = ", ".join(label_schema["properties"]["labels"]["items"]["enum"])
    prompt = config["system_prompt"].replace("{labels}", labels).replace("{text}", screened_text)
    request = {"model": config["model_name"], "messages": [
        {"role": "user", "content": prompt}], "format": label_schema, "stream": False,
        "keep_alive": config["keep_alive"], "options": {"temperature": config["temperature"],
                                         "num_ctx": config["num_ctx"],
                                         "num_predict": config["num_predict"], "seed": config["seed"],
                                         "top_k": config["top_k"], "top_p": config["top_p"],
                                         "repeat_penalty": config["repeat_penalty"]}}
    return (json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _stats(values: list[float | None]) -> dict[str, Any]:
    observed = [value for value in values if type(value) in (int, float) and math.isfinite(value)]
    if not observed:
        return {"count": 0, "missing_count": len(values), "median_ms": None,
                "p95_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(observed)
    return {"count": len(observed), "missing_count": len(values) - len(observed),
            "median_ms": statistics.median(observed),
            "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
            "min_ms": min(observed), "max_ms": max(observed)}


NOT_STARTED_OUTCOMES = {"request_too_large_not_started", "not_attempted_after_abort"}


def _http_requests_started(outcomes: list[str]) -> int:
    return sum(outcome not in NOT_STARTED_OUTCOMES for outcome in outcomes)


def _validate_final_timings(outcomes: Any, timings: Any, progress: list[dict[str, Any]], rows: int) -> None:
    if (not isinstance(outcomes, list) or not isinstance(timings, list)
            or len(outcomes) != rows or len(timings) != rows or len(progress) != rows):
        raise ValueError("worker final timing/result row counts are inconsistent")
    for index, (outcome, elapsed, observed) in enumerate(zip(outcomes, timings, progress, strict=True)):
        if not isinstance(outcome, str) or not isinstance(observed, dict):
            raise ValueError("worker final timing/result structure is invalid")
        not_started = outcome in NOT_STARTED_OUTCOMES
        valid_elapsed = (elapsed is None if not_started else
                         type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0)
        if (not valid_elapsed or observed.get("index") != index
                or observed.get("outcome") != outcome or observed.get("timing_ms") != elapsed):
            raise ValueError("worker final timing does not match validated progress observations")


def _response_prediction_matrix(raw_predictions: list[list[Any]], labels: list[str], rows: int) -> list[list[int]]:
    if len(raw_predictions) != rows:
        raise ValueError("worker prediction row count is inconsistent")
    normalized = []
    for row in raw_predictions:
        if not isinstance(row, list) or len(row) != len(labels):
            raise ValueError("worker prediction label count is inconsistent")
        if any(value is not None and (type(value) is not int or value not in (0, 1)) for value in row):
            raise ValueError("worker prediction contains an invalid label value")
        normalized.append([-1 if value is None else value for value in row])
    return normalized


def _git_revision_for_comparator() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--",
                                str(Path(__file__).resolve())], check=True, capture_output=True,
                               timeout=10, shell=False).stdout
        if dirty:
            return None
        revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                  capture_output=True, text=True, timeout=10, shell=False).stdout.strip()
        return revision if re.fullmatch(r"[0-9a-f]{40,64}", revision) else None
    except (OSError, subprocess.SubprocessError):
        return None


def _baseline_review_provenance(baseline_report: dict[str, Any]) -> dict[str, Any]:
    data = baseline_report.get("data") if isinstance(baseline_report.get("data"), dict) else {}
    raw_states = data.get("label_review_state")
    fields = ("rows", "target_array_nonempty", "target_array_empty", "target_array_missing_or_null",
              "verified_true", "verified_false", "labels_pending_true", "labels_pending_false",
              "labels_pending_absent", "target_completeness_true", "target_completeness_false",
              "target_completeness_absent")
    review_state = None
    if isinstance(raw_states, dict):
        sanitized = {}
        for split in ("train", "validation"):
            counts = raw_states.get(split)
            if not isinstance(counts, dict):
                sanitized = {}
                break
            if any(type(counts.get(field)) is not int or counts[field] < 0 for field in fields):
                sanitized = {}
                break
            sanitized[split] = {field: counts[field] for field in fields}
        if sanitized:
            review_state = sanitized
    group_use_claim = data.get("validation_group_use_claim")
    if not isinstance(group_use_claim, str) or group_use_claim not in {"previously_used", "reported_unused", "unknown"}:
        group_use_claim = "unknown"
    group_claim_verified = data.get("validation_group_use_claim_verified")
    if type(group_claim_verified) is not bool:
        group_claim_verified = None
    independence_claim = data.get("independence_claim")
    if not isinstance(independence_claim, str) or not independence_claim.strip():
        independence_claim = "unknown; no baseline claim supplied"
    elif "source independence remains unverified" in independence_claim:
        independence_claim = "plan separation is claimed; source independence remains unverified"
    else:
        independence_claim = "baseline claim supplied; meaning not recognized by this adapter"
    independence_verified = data.get("validation_independence_verified")
    if type(independence_verified) is not bool:
        independence_verified = None
    all_marked_verified = None
    if review_state is not None:
        all_marked_verified = all(review_state[split]["verified_true"] == review_state[split]["rows"]
                                  for split in ("train", "validation"))
    return {"label_review_state": review_state,
            "all_rows_marked_verified": all_marked_verified,
            "validation_group_use_claim": group_use_claim,
            "validation_group_use_claim_verified": group_claim_verified,
            "validation_independence_claim": independence_claim,
            "validation_independence_verified": independence_verified}


def _review_history_limitations(review: dict[str, Any]) -> list[str]:
    limits = ["label-review and validation-history flags copied from the baseline are claims; this adapter does not independently audit them",
              "source schema completeness claims are not independently verified",
              "no final test or heldout data were opened",
              "conversion linkage from the configured artifact to upstream weights is not independently proven unless the receipt explicitly establishes it",
              "per-process GPU memory attribution was not measured",
              "paired classical same-answered-subset metrics unavailable without per-row baseline predictions"]
    if review["label_review_state"] is None:
        limits.append("baseline label review state is absent or unsupported; label review status is unknown")
    claim = review["validation_group_use_claim"]
    verified = review["validation_group_use_claim_verified"]
    verification_text = ("baseline marks the history claim verified" if verified is True else
                         "baseline marks the history claim unverified" if verified is False else
                         "history-claim verification status is unknown")
    if claim == "previously_used":
        limits.append(f"baseline reports prior validation-group use; {verification_text}; results are exploratory")
    elif claim == "reported_unused":
        limits.append(f"baseline reports validation groups unused; {verification_text}")
    else:
        limits.append("validation-group use history is unknown in the baseline report")
    if review["validation_independence_verified"] is True:
        pass
    elif review["validation_independence_verified"] is False:
        limits.append("baseline explicitly marks validation independence as unverified")
    else:
        limits.append("baseline does not provide a validation-independence verification status")
    return limits


def _compute_metrics(y_true: list[list[int]], y_pred: list[list[int]],
                     target_known: list[list[bool]], labels: list[str]) -> dict[str, Any]:
    rows, width = len(y_true), len(labels)
    if any(len(row) != width for row in y_true + y_pred + target_known):
        raise ValueError("metric matrices have inconsistent label dimensions")
    by_label, totals = {}, {key: 0 for key in ("tp", "fp", "fn", "tn", "unknown_predictions",
                                                "unknown_targets", "unknown_positive")}
    standard = {key: [] for key in ("precision", "recall", "f1")}
    supported = {key: [] for key in ("precision", "recall", "f1")}
    known_targets = scored = fully_scored_rows = subset_correct = 0
    for index, label in enumerate(labels):
        tp = fp = fn = tn = unknown_predictions = unknown_positive = 0
        support = negative_support = scored_support = predicted_positive = 0
        for row in range(rows):
            target_known_here = target_known[row][index]
            value, prediction = y_true[row][index], y_pred[row][index]
            if target_known_here:
                known_targets += 1
                if value == 1:
                    support += 1
                else:
                    negative_support += 1
                if prediction < 0:
                    unknown_predictions += 1
                    unknown_positive += int(value == 1)
                    continue
                scored += 1
                scored_support += 1
                predicted_positive += int(prediction == 1)
                tp += int(value == 1 and prediction == 1)
                fp += int(value == 0 and prediction == 1)
                fn += int(value == 1 and prediction == 0)
                tn += int(value == 0 and prediction == 0)
        precision_denominator, recall_denominator = tp + fp, tp + fn
        f1_denominator = 2 * tp + fp + fn
        precision = tp / precision_denominator if precision_denominator else None
        recall = tp / recall_denominator if recall_denominator else None
        f1 = 2 * tp / f1_denominator if f1_denominator else None
        by_label[label] = {"support": support, "negative_support": negative_support,
                           "scored_support": scored_support, "predicted_positive": predicted_positive,
                           "unknown_predictions": unknown_predictions,
                           "unknown_targets": sum(not target_known[row][index] for row in range(rows)),
                           "unknown_positive_targets": unknown_positive, "tp": tp, "fp": fp,
                           "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}
        for name, value in (("precision", precision), ("recall", recall), ("f1", f1)):
            if value is not None:
                standard[name].append(value)
                if support > 0:
                    supported[name].append(value)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["tn"] += tn
        totals["unknown_predictions"] += unknown_predictions
        totals["unknown_positive"] += unknown_positive
        totals["unknown_targets"] += sum(not target_known[row][index] for row in range(rows))
    for row in range(rows):
        fully_known = all(target_known[row][index] and y_pred[row][index] >= 0 for index in range(width))
        if fully_known:
            fully_scored_rows += 1
            subset_correct += int(all(y_true[row][index] == y_pred[row][index] for index in range(width)))
    total_cells = rows * width
    micro_p_denominator = totals["tp"] + totals["fp"]
    micro_r_denominator = totals["tp"] + totals["fn"]
    micro_f_denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
    macro = {key: (sum(values) / len(values) if values else None) for key, values in standard.items()}
    macro_supported = {key: (sum(values) / len(values) if values else None) for key, values in supported.items()}
    return {"examples": rows,
            "subset_accuracy_on_fully_scored": subset_correct / fully_scored_rows if fully_scored_rows else None,
            "scored_label_predictions": scored, "total_label_predictions": total_cells,
            "target_coverage": known_targets / total_cells if total_cells else None,
            "prediction_coverage_on_known_targets": scored / known_targets if known_targets else None,
            "overall_coverage": scored / total_cells if total_cells else None,
            "macro_precision": macro["precision"], "macro_recall": macro["recall"], "macro_f1": macro["f1"],
            "macro_precision_supported_labels": macro_supported["precision"],
            "macro_recall_supported_labels": macro_supported["recall"],
            "macro_f1_supported_labels": macro_supported["f1"],
            "macro_metric_label_coverage": {key: {"defined_labels": len(values), "total_labels": width}
                                            for key, values in standard.items()},
            "micro_precision_on_covered": totals["tp"] / micro_p_denominator if micro_p_denominator else None,
            "micro_recall_on_covered": totals["tp"] / micro_r_denominator if micro_r_denominator else None,
            "micro_f1_on_covered": 2 * totals["tp"] / micro_f_denominator if micro_f_denominator else None,
            "labels": by_label, "unknown_or_abstained": totals["unknown_predictions"],
            "unknown_target_cells": totals["unknown_targets"], "unknown_positive_targets": totals["unknown_positive"]}


def _baseline_id_hashes(rows: list[dict]) -> list[str]:
    return [sha(row["id"].encode("utf-8")) for row in sorted(rows, key=lambda row: row["id"])]


def _target_mask_digest(rows: list[dict], known) -> str:
    material = [[sha(row["id"].encode("utf-8")), [bool(value) for value in mask]]
                for row, mask in zip(rows, known, strict=True)]
    return sha(json.dumps(material, separators=(",", ":")).encode("utf-8"))


def _target_scoring_policy(y_train, known_train, labels: list[str], constant_behavior: str) -> dict[str, dict[str, Any]]:
    policy = {}
    for index, label in enumerate(labels):
        values = y_train[known_train[:, index], index]
        classes = set(values.tolist())
        if not classes:
            mode, constant = "unsupported_abstain", None
        elif len(classes) > 1:
            mode, constant = "model_prediction", None
        elif constant_behavior == "abstain":
            mode, constant = "constant_target_abstain", int(next(iter(classes)))
        else:
            mode, constant = "constant_target_majority_fallback", int(next(iter(classes)))
        policy[label] = {"mode": mode, "constant_train_value": constant,
                         "known_train_rows": int(len(values)),
                         "positive_train_rows": int(values.sum()),
                         "negative_train_rows": int(len(values) - values.sum())}
    return policy


def _bind_baseline(config: dict[str, Any]) -> tuple[dict, dict, list[dict], list[dict], Any, Any, Any, Any, str]:
    import classifier_comparison as classical
    report_path = Path(config["baseline_report"]).resolve(strict=True)
    report_raw = _read_bounded(report_path, config["max_baseline_report_bytes"], "baseline report")
    if sha(report_raw) != config["baseline_report_sha256"]:
        raise ValueError("baseline report digest mismatch")
    report = _json(report_raw, "baseline report")
    if (not isinstance(report, dict) or report.get("schema_version") != "lfg.classifier-comparison-report.v1"
            or report.get("promotion_eligible") is not False):
        raise ValueError("unsupported classical baseline report")
    source_path = Path(classical.__file__).resolve(strict=True)
    if sha(source_path.read_bytes()) != report.get("code_sha256"):
        raise ValueError("baseline report code is not the current pinned local comparison")
    classical_config_path = Path(config["classical_config"]).resolve(strict=True)
    cfg_raw = _read_bounded(classical_config_path, MAX_CONFIG_BYTES, "classical configuration")
    if sha(cfg_raw) != config["classical_config_sha256"] or sha(cfg_raw) != report.get("config_sha256"):
        raise ValueError("classical configuration digest does not match baseline report")
    cc, cc_digest = classical.load_config(classical_config_path, MAX_CONFIG_BYTES)
    train, validation, corpus, taxonomy, source_identity = classical.load_screened_corpus(
        Path(cc["screened_corpus"]), Path(cc["label_plan"]), cc["max_rows"], cc["max_text_bytes"],
        config["max_source_bytes"], config["max_total_source_bytes"], config["max_manifest_bytes"],
        config["max_screened_file_bytes"], cc["near_duplicate_similarity"], cc["max_near_duplicate_pairs"],
        cc["max_near_duplicate_work"], Path(cc["screen_policy"]), config["max_policy_bytes"])
    ledger, ledger_digest = classical._load_review_ledger(Path(cc["review_ledger"]), config["max_ledger_bytes"])
    if set(ledger) != {row["id"] for row in train + validation}:
        raise ValueError("classical review ledger selection changed")
    field, labels = cc["target_field"], sorted(cc["selected_labels"])
    if report.get("task", {}).get("field") != field or report.get("task", {}).get("labels") != labels:
        raise ValueError("baseline task or selected labels changed")
    if report.get("task", {}).get("taxonomy_sha256") != corpus["taxonomy_sha256"]:
        raise ValueError("baseline taxonomy identity changed")
    algorithms = report.get("algorithms")
    if (not isinstance(algorithms, dict)
            or set(algorithms) != {"logistic_regression", "knn", "decision_tree"}
            or any(not isinstance(result, dict) or result.get("status") not in
                   ("completed", "constant_target_fallback_only", "no_supported_targets")
                   for result in algorithms.values())):
        raise ValueError("classical baseline is incomplete or has failed estimators")
    if report.get("config_sha256") != cc_digest:
        raise ValueError("baseline config identity changed")
    if (report.get("data", {}).get("plan_sha256") != source_identity["plan_sha256"]
            or report["data"].get("taxonomy_sha256") != corpus["taxonomy_sha256"]
            or report["data"].get("manifest_sha256") != corpus["manifest_sha256"]
            or report["data"].get("train_sha256") != corpus["train_sha256"]
            or report["data"].get("validation_sha256") != corpus["validation_sha256"]
            or report["data"].get("policy_sha256") != corpus["policy_sha256"]
            or report["data"].get("screen_code_sha256") != corpus["screen_code_sha256"]
            or report["data"].get("review_ledger_sha256") != ledger_digest
            or report["data"].get("rejected_label_handling") != cc["rejected_label_handling"]):
        raise ValueError("baseline data, screening, or review-ledger identity changed")
    recorded_review = report["data"].get("label_review_state")
    if (recorded_review is not None
            and recorded_review != {"train": classical._label_review_state(train, field),
                                    "validation": classical._label_review_state(validation, field)}):
        raise ValueError("baseline label review counts no longer match the selected source rows")
    recorded_group_claim = report["data"].get("validation_group_use_claim")
    if recorded_group_claim is not None and recorded_group_claim != cc["validation_group_use_claim"]:
        raise ValueError("baseline validation-use claim differs from its configured claim")
    if (len(train) != report["data"].get("train_rows") or len(validation) != report["data"].get("validation_rows")
            or _baseline_id_hashes(train) != report["data"].get("train_record_id_sha256")
            or _baseline_id_hashes(validation) != report["data"].get("validation_record_id_sha256")):
        raise ValueError("baseline selected row identity changed")
    _, y_train, known_train = classical._matrix(train, field, labels, taxonomy)
    _, y_validation, known_validation = classical._matrix(validation, field, labels, taxonomy)
    if cc["rejected_label_handling"] == "exclude":
        known_train[[ledger[row["id"]] == "rejected" for row in train], :] = False
        known_validation[[ledger[row["id"]] == "rejected" for row in validation], :] = False
    decisions = {decision: sum(value == decision for value in ledger.values())
                 for decision in ("pending", "approved", "rejected")}
    excluded = {"train": sum(ledger[row["id"]] == "rejected" for row in train)
                if cc["rejected_label_handling"] == "exclude" else 0,
                "validation": sum(ledger[row["id"]] == "rejected" for row in validation)
                if cc["rejected_label_handling"] == "exclude" else 0}
    if (report["data"].get("review_ledger_decisions") != decisions
            or report["data"].get("rejected_rows_excluded") != excluded):
        raise ValueError("baseline review decisions or rejected-label denominator changed")
    for index, label in enumerate(labels):
        expected_train = report["task"]["training_support"][label]
        expected_validation = report["task"]["validation_support"][label]
        train_values = y_train[known_train[:, index], index]
        validation_values = y_validation[known_validation[:, index], index]
        for expected, values, name in ((expected_train, train_values, "train"),
                                       (expected_validation, validation_values, "validation")):
            if (int(values.sum()) != expected.get("positive")
                    or len(values) - int(values.sum()) != expected.get("negative")
                    or (len(train) if name == "train" else len(validation)) - len(values) != expected.get("unknown_rows")):
                raise ValueError("baseline per-label observation mask no longer matches source report")
    return report, cc, train, validation, y_train, known_train, y_validation, known_validation, taxonomy, corpus, cc_digest


def _prediction_http(config: dict[str, Any], request: bytes, labels: set[str]) -> tuple[str, list[str] | None, float]:
    started = time.monotonic()
    status, raw = _http_call(config, request)
    elapsed = (time.monotonic() - started) * 1000
    if status != 200:
        return "http_error", None, elapsed
    envelope = _json(raw, "model response")
    if not isinstance(envelope, dict):
        return "malformed", None, elapsed
    if envelope.get("model") != config["model_name"]:
        return "identity_mismatch", None, elapsed
    if envelope.get("done") is not True:
        return "incomplete", None, elapsed
    message = envelope.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    outcome, prediction = parse_prediction(content, envelope.get("done"), envelope.get("done_reason"), labels)
    return outcome, prediction, elapsed


def _copy_verified_artifact(config: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    store = Path(config["model_store"])
    if store.is_symlink():
        raise ValueError("private model store must not be a symbolic link")
    if store.exists():
        return verify_installed_artifact(config, receipt)
    store.mkdir(mode=0o700, parents=True, exist_ok=False)
    if not _private_directory(store):
        raise ValueError("new model store is not owner-private")
    staging = Path(receipt["staging_locator"])
    if staging.is_symlink() or not _private_directory(staging):
        raise ValueError("acquisition staging directory is not owner-private")
    manifest_meta = receipt["manifest"]
    manifest_name = manifest_meta.get("file")
    if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
        raise ValueError("invalid acquisition manifest name")
    source_manifest = _no_symlink_child(staging, manifest_name)
    manifest_sha, manifest_size = _hash_file(source_manifest, config["max_model_manifest_bytes"],
                                             config["hash_chunk_bytes"])
    if manifest_sha != manifest_meta["sha256"] or manifest_size != manifest_meta["bytes"]:
        raise ValueError("staged model manifest differs from acquisition receipt")
    manifest = _json(_read_bounded(source_manifest, config["max_model_manifest_bytes"], "model manifest"),
                     "model manifest")
    registry = urlsplit(config["expected_registry_manifest_url"])
    manifest_parts = registry.path.strip("/").split("/")
    if len(manifest_parts) != 5 or manifest_parts[0] != "v2" or manifest_parts[3] != "manifests":
        raise ValueError("configured registry manifest path is not a supported OCI layout")
    target_manifest = store / "manifests" / registry.netloc / manifest_parts[1] / manifest_parts[2] / manifest_parts[4]
    target_manifest.parent.mkdir(mode=0o700, parents=True)
    (store / "blobs").mkdir(mode=0o700)
    expected = {item["digest"].removeprefix("sha256:"): item for item in receipt["objects"]}
    entries = [manifest.get("config")] + manifest.get("layers", [])
    if not isinstance(entries[0], dict) or not isinstance(manifest.get("layers"), list):
        raise ValueError("staged registry manifest has an invalid layer structure")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid registry layer entry")
        digest = entry.get("digest", "").removeprefix("sha256:")
        receipt_item = expected.get(digest)
        if (not HEX.fullmatch(digest) or digest in seen or receipt_item is None
                or receipt_item.get("sha256_verified") is not True
                or type(entry.get("size")) is not int or entry["size"] != receipt_item.get("bytes")):
            raise ValueError("registry layer is not bound by the verified receipt")
        seen.add(digest)
        source = _no_symlink_child(staging, "objects", "sha256", digest)
        source_sha, source_size = _hash_file(source, config["max_model_blob_bytes"], config["hash_chunk_bytes"])
        if source_sha != digest or source_size != entry["size"]:
            raise ValueError("staged model layer failed digest or size verification")
        target = store / "blobs" / f"sha256-{digest}"
        shutil.copyfile(source, target)
        os.chmod(target, 0o600)
        copied_sha, copied_size = _hash_file(target, config["max_model_blob_bytes"], config["hash_chunk_bytes"])
        if copied_sha != digest or copied_size != source_size:
            raise ValueError("private model layer copy failed verification")
    if seen != set(expected):
        raise ValueError("receipt and registry layer sets differ")
    shutil.copyfile(source_manifest, target_manifest)
    os.chmod(target_manifest, 0o600)
    return verify_installed_artifact(config, receipt)


def _mask_digest_from_hashes(record_hashes: list[str], masks: list[list[bool]]) -> str:
    return sha(json.dumps([[record_hash, mask] for record_hash, mask in zip(record_hashes, masks, strict=True)],
                          separators=(",", ":")).encode("utf-8"))


def _make_payload(config: dict[str, Any], report: dict, classical_config: dict,
                  train_rows: list[dict], validation_rows: list[dict], y_train, train_known,
                  y_validation, validation_known, corpus: dict, classical_digest: str) -> dict[str, Any]:
    labels = sorted(classical_config["selected_labels"])
    record_hashes = [sha(row["id"].encode("utf-8")) for row in validation_rows]
    if sorted(record_hashes) != sorted(report["data"]["validation_record_id_sha256"]):
        raise ValueError("validation record identity changed before worker payload creation")
    masks = [[bool(item) for item in row] for row in validation_known.tolist()]
    values = [[int(value) if masks[row_index][label_index] else None
               for label_index, value in enumerate(row)]
              for row_index, row in enumerate(y_validation.tolist())]
    texts = []
    for row in validation_rows:
        text = row["text"]
        if not isinstance(text, str):
            raise ValueError("screened validation text is not UTF-8 text")
        texts.append({"text": text, "text_sha256": sha(text.encode("utf-8"))})
    payload = {
        "schema_version": "lfg.local-12b-worker-input.v1",
        "binding": {"baseline_report_sha256": config["baseline_report_sha256"],
                    "classical_config_sha256": classical_digest,
                    "task_field": classical_config["target_field"], "labels": labels,
                    "taxonomy_sha256": corpus["taxonomy_sha256"],
                    "train_record_count": len(train_rows),
                    "train_target_mask_sha256": _target_mask_digest(train_rows, train_known),
                    "validation_record_id_sha256": report["data"]["validation_record_id_sha256"],
                    "validation_target_mask_sha256": _target_mask_digest(validation_rows, validation_known)},
        "records": [{"id_sha256": record_hashes[i], **texts[i],
                     "target_values": values[i], "target_known": masks[i]}
                    for i in range(len(validation_rows))],
        "max_request_bytes": config["max_request_bytes"],
    }
    return payload


def _validate_worker_payload(payload: Any, config: dict[str, Any]) -> tuple[list[str], list[dict]]:
    if (not isinstance(payload, dict) or payload.get("schema_version") != "lfg.local-12b-worker-input.v1"
            or not isinstance(payload.get("binding"), dict) or not isinstance(payload.get("records"), list)
            or not payload["records"]):
        raise ValueError("invalid private worker input")
    binding = payload["binding"]
    labels = binding.get("labels")
    if (not isinstance(labels, list) or not labels or any(not isinstance(x, str) for x in labels)
            or len(labels) != len(set(labels))):
        raise ValueError("invalid worker target labels")
    ids, masks = [], []
    if len(payload["records"]) != len(binding.get("validation_record_id_sha256", [])):
        raise ValueError("worker record count differs from bound validation IDs")
    if type(payload.get("max_request_bytes")) is not int or payload["max_request_bytes"] != config["max_request_bytes"]:
        raise ValueError("worker request size limit changed")
    for record in payload["records"]:
        if not isinstance(record, dict):
            raise ValueError("invalid worker validation row")
        text, text_digest, record_id = record.get("text"), record.get("text_sha256"), record.get("id_sha256")
        known, values = record.get("target_known"), record.get("target_values")
        if (not isinstance(text, str) or sha(text.encode("utf-8")) != text_digest
                or not isinstance(record_id, str) or not HEX.fullmatch(record_id)
                or not isinstance(known, list) or not isinstance(values, list)
                or len(known) != len(labels) or len(values) != len(labels)
                or any(type(item) is not bool for item in known)
                or any((item is not None and (type(item) is not int or item not in (0, 1))) for item in values)
                or any((known[i] and values[i] is None) or (not known[i] and values[i] is not None)
                       for i in range(len(labels)))):
            raise ValueError("worker text, target, or digest validation failed")
        ids.append(record_id)
        masks.append(known)
    if sorted(ids) != sorted(binding["validation_record_id_sha256"]):
        raise ValueError("worker validation IDs differ from the bound baseline")
    if _mask_digest_from_hashes(ids, masks) != binding.get("validation_target_mask_sha256"):
        raise ValueError("worker target observation mask differs from local baseline")
    return labels, payload["records"]


def _warmup(config: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps({"model": config["model_name"], "messages": [{"role": "user", "content": ""}],
                       "stream": False, "keep_alive": config["keep_alive"],
                       "options": {"num_ctx": config["num_ctx"], "num_predict": 1,
                                   "temperature": 0.0}}, separators=(",", ":")).encode("utf-8")
    status, raw = _http_call(config, body)
    if status != 200:
        raise ValueError("empty-input model warmup failed")
    response = _json(raw, "empty-input model warmup")
    if (not isinstance(response, dict) or response.get("model") != config["model_name"]
            or response.get("done") is not True):
        raise ValueError("empty-input model warmup identity or completion is invalid")
    return {"status": "completed", "input": "empty user content; no private record"}


def _worker(config: dict[str, Any]) -> None:
    import signal

    receipt, receipt_digest = verify_acquisition(config)
    if (not sys.platform.startswith("linux") or not _route_is_private()
            or os.geteuid() != config["remote_uid"]):
        raise ValueError("worker is not running as the configured user in a loopback-only namespace")
    resources_before = resource_snapshot(config)
    artifact = verify_installed_artifact(config, receipt)
    address, port, _ = _loopback_endpoint(config["endpoint_url"])
    if address != "127.0.0.1":
        raise ValueError("worker endpoint must use IPv4 loopback in its private namespace")
    server = subprocess.Popen([config["server_binary_path"], "serve"], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False,
                              start_new_session=True,
                              env={**os.environ, "OLLAMA_NO_CLOUD": "1",
                                   "OLLAMA_MODELS": config["model_store"],
                                   "OLLAMA_HOST": f"127.0.0.1:{port}"})
    try:
        deadline = time.monotonic() + config["worker_start_timeout_seconds"]
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise ValueError("task-owned Ollama server exited before readiness")
            try:
                status, version_raw = _http_call(config, None, method="GET", path_override="/api/version")
                if status == 200:
                    version = _json(version_raw, "Ollama version response")
                    if not isinstance(version, dict) or version.get("version") != config["server_version"]:
                        raise ValueError("isolated Ollama runtime version differs from pinned version")
                    break
            except (OSError, http.client.HTTPException, TimeoutError):
                pass
            time.sleep(0.25)
        else:
            raise TimeoutError("isolated Ollama runtime did not become ready")
        warmup = _warmup(config)
        runtime_config = {**config, "server_pid": server.pid}
        identity = verify_loaded_model(config, artifact)
        boundary = verify_runtime_boundary(runtime_config)
        ready = {"phase": "ready", "netns_inode": boundary["network_namespace_inode"],
                 "server_pid": server.pid, "port": port, "model_manifest_sha256": identity["reported_manifest_sha256"],
                 "artifact_receipt_sha256": receipt_digest,
                 "server_binary_sha256": boundary["server_binary_sha256"],
                 "server_version": boundary["server_version"], "warmup": warmup,
                 "worker_code_sha256": config["worker_code_sha256"],
                 "comparator_git_revision": config.get("comparator_git_revision"),
                 "resources_before": resources_before}
        print(json.dumps(ready, sort_keys=True, separators=(",", ":")), flush=True)
        payload_raw = sys.stdin.buffer.readline(config["max_payload_bytes"] + 1)
        if not payload_raw or len(payload_raw) > config["max_payload_bytes"] or not payload_raw.endswith(b"\n"):
            raise ValueError("bounded worker input was absent, oversized, or incomplete")
        payload = _json(payload_raw, "worker input")
        labels, records = _validate_worker_payload(payload, config)
        schema = output_schema(labels)
        results, predictions, timings = [], [], []
        aborted, abort_reason = False, None
        for index, record in enumerate(records):
            if aborted:
                results.append("not_attempted_after_abort")
                predictions.append([None] * len(labels))
                timings.append(None)
                print(json.dumps({"phase": "progress", "index": index, "outcome": "not_attempted_after_abort",
                                  "prediction": [None] * len(labels), "timing_ms": None}, separators=(",", ":")),
                      flush=True)
                continue
            request = _schema_request(config, schema, record["text"])
            started = time.monotonic()
            try:
                outcome, predicted, elapsed = _prediction_http(config, request, set(labels))
            except RequestLimitError:
                outcome, predicted, elapsed = "request_too_large_not_started", None, None
            except TimeoutError:
                outcome, predicted, elapsed = "timeout_unknown_worker_lifecycle", None, (time.monotonic() - started) * 1000
                aborted, abort_reason = True, "request deadline expired; worker lifecycle unknown"
            except (OSError, ValueError, http.client.HTTPException):
                outcome, predicted, elapsed = "transport_or_protocol_error", None, (time.monotonic() - started) * 1000
                aborted, abort_reason = True, "transport/protocol outcome ambiguous"
            row_prediction = [None] * len(labels)
            if outcome == "answered" and predicted is not None:
                row_prediction = [int(label in predicted) for label in labels]
            elif outcome in ("identity_mismatch", "http_error", "incomplete"):
                aborted = True
                abort_reason = "worker response identity, status, or lifecycle was not trustworthy"
            results.append(outcome)
            predictions.append(row_prediction)
            timings.append(elapsed)
            print(json.dumps({"phase": "progress", "index": index, "outcome": outcome,
                              "prediction": row_prediction, "timing_ms": elapsed},
                             separators=(",", ":")), flush=True)
        resources_after = resource_snapshot(config, enforce=False)
        response = {"phase": "result", "payload_sha256": sha(payload_raw),
                    "validation_record_id_sha256": payload["binding"]["validation_record_id_sha256"],
                    "target_mask_sha256": payload["binding"]["validation_target_mask_sha256"],
                    "outcomes": results, "predictions": predictions, "timing_ms": timings,
                    "aborted": aborted, "abort_reason": abort_reason,
                    "loaded_model": identity, "artifact": artifact,
                    "resources_after": resources_after}
        encoded = json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > config["max_response_bytes"]:
            raise ValueError("worker aggregate exceeded configured response limit")
        response["server_cleanup_verified"] = False
        response["server_exit_code"] = None
    finally:
        child_pids = _descendant_pids(server.pid)
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            server.wait(timeout=5)
        lingering = [pid for pid in child_pids if Path(f"/proc/{pid}").exists()]
        if lingering:
            raise ValueError("task-owned Ollama child process remained after cleanup")
        response["server_cleanup_verified"] = server.poll() is not None
        response["server_exit_code"] = server.returncode
    if not response["server_cleanup_verified"]:
        raise ValueError("task-owned Ollama server did not exit cleanly")
    encoded = json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > config["max_response_bytes"]:
        raise ValueError("worker aggregate exceeded configured response limit")
    print(encoded, flush=True)


def _read_protocol_line(process: subprocess.Popen, lines: queue.Queue, timeout: float) -> dict[str, Any]:
    try:
        line = lines.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError("isolated worker did not produce protocol output before deadline") from None
    if line is None:
        raise ValueError("isolated worker exited without a protocol result")
    if len(line) > 1_048_576:
        raise ValueError("isolated worker protocol line exceeds maximum size")
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        raise ValueError("isolated worker returned invalid protocol JSON") from None
    if not isinstance(value, dict):
        raise ValueError("isolated worker protocol must be an object")
    return value


def _private_report_directory(path: Path) -> Path:
    if any((ancestor / ".git").exists() for ancestor in (path, *path.parents)):
        raise ValueError("report output must be outside every Git worktree")
    if path.exists() or path.is_symlink():
        raise ValueError("report directory must be new")
    # This helper is used only for a newly named, task-owned report directory.
    # Establish the final ACL before creating any private report bytes.
    path.mkdir(mode=0o700, parents=False, exist_ok=False)
    if os.name == "posix":
        if path.stat().st_uid != os.geteuid() or path.stat().st_mode & 0o077:
            raise ValueError("report directory is not owner-private")
    elif os.name == "nt":
        who = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], check=True,
                             capture_output=True, text=True, timeout=10, shell=False).stdout
        who_row = next(csv.reader([who.strip()]), [])
        if len(who_row) != 2 or not re.fullmatch(r"S-1-[0-9-]+", who_row[1]):
            raise ValueError("current Windows user SID could not be verified")
        user_name, sid = who_row[0], who_row[1]
        # Reset inherited and explicit grants from the temp parent before
        # installing the exact intended principals. /inheritance:r alone only
        # drops inherited ACEs; it leaves an explicit LogonSessionId ACE behind.
        subprocess.run(["icacls", str(path), "/reset"],
                       check=True, capture_output=True, text=True, timeout=10, shell=False)
        subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r",
                        f"*{sid}:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F"],
                       check=True, capture_output=True, text=True, timeout=10, shell=False)
        acl = subprocess.run(["icacls", str(path)], check=True, capture_output=True, text=True,
                             timeout=10, shell=False).stdout
        acl_lines = acl.splitlines()
        first = acl_lines[0].partition(str(path))[2].strip() if acl_lines else ""
        entries = []
        for line in ([first] + [item.strip() for item in acl_lines[1:]]):
            match = re.fullmatch(r"(.+?):((?:\([^)]+\))+)", line)
            if match:
                entries.append(match.groups())
        allowed = {user_name.casefold(), "builtin\\administrators", "nt authority\\system"}
        principals = {name.strip().casefold() for name, _ in entries}
        # The descriptor owner is separate from its access list. Require it to
        # remain the current user, even if the ACL text happens to look right.
        path_literal = str(path).replace("'", "''")
        owner_command = (f"(New-Object System.Security.Principal.NTAccount((Get-Acl -LiteralPath '{path_literal}').Owner))."
                         "Translate([System.Security.Principal.SecurityIdentifier]).Value")
        owner = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", owner_command],
                               check=True, capture_output=True, text=True, timeout=10, shell=False).stdout.strip()
        if (principals != allowed or any("F" not in rights for _, rights in entries)
                or "(I)" in acl or owner.casefold() != sid.casefold()):
            raise ValueError("report directory ACL is not restricted to owner, SYSTEM, and Administrators")
    return path


def _verify_private_report_directory(path: Path) -> bool:
    """Verify the report-specific owner/SYSTEM/Administrators policy without changing it."""
    try:
        if path.is_symlink() or not path.is_dir() or any((p / ".git").exists() for p in (path, *path.parents)):
            return False
        if os.name == "posix":
            info = path.stat(follow_symlinks=False)
            return info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o700
        if os.name != "nt":
            return False
        who = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], check=True,
                             capture_output=True, text=True, timeout=10, shell=False).stdout
        row = next(csv.reader([who.strip()]), [])
        if len(row) != 2 or not re.fullmatch(r"S-1-[0-9-]+", row[1]):
            return False
        user_name, sid = row
        acl = subprocess.run(["icacls", str(path)], check=True, capture_output=True,
                             text=True, timeout=10, shell=False).stdout
        lines = acl.splitlines()
        first = lines[0].partition(str(path))[2].strip() if lines else ""
        entries = []
        for line in [first] + [item.strip() for item in lines[1:]]:
            match = re.fullmatch(r"(.+?):((?:\([^)]+\))+)" , line)
            if match:
                entries.append(match.groups())
        allowed = {user_name.casefold(), "builtin\\administrators", "nt authority\\system"}
        if ({name.strip().casefold() for name, _rights in entries} != allowed
                or any("F" not in rights for _name, rights in entries) or "(I)" in acl):
            return False
        literal = str(path).replace("'", "''")
        command = (f"(New-Object System.Security.Principal.NTAccount((Get-Acl -LiteralPath '{literal}').Owner))."
                   "Translate([System.Security.Principal.SecurityIdentifier]).Value")
        owner = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                               check=True, capture_output=True, text=True, timeout=10, shell=False).stdout.strip()
        return owner.casefold() == sid.casefold()
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _ssh_run(config: dict[str, Any], script: str, timeout: float = 30) -> subprocess.CompletedProcess:
    import base64
    if (not _valid_ssh_host(config.get("remote_host")) or not _valid_ssh_user(config.get("remote_user"))
            or not _safe_absolute_posix_path(config.get("remote_python_path", ""))):
        raise ValueError("unsafe SSH destination or remote interpreter path")
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    command = " ".join((shlex.quote(config["remote_python_path"]), "-I", "-c",
                         shlex.quote(f"import base64;exec(base64.b64decode('{encoded}'))")))
    argv = ["ssh", "-T", "-oBatchMode=yes", "-oConnectTimeout=10", "-p", str(config["ssh_port"]),
            f"{config['remote_user']}@{config['remote_host']}", command]
    return subprocess.run(argv, check=False, capture_output=True, timeout=timeout, shell=False)


def _remote_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    script = r'''import json,os,platform,shutil,subprocess
OLLAMA=__OLLAMA__
HOME=__HOME__
SERVICE=__SERVICE__
SYSTEMCTL=__SYSTEMCTL__
NVIDIA=__NVIDIA__
def cmd(a):
 try:
  return subprocess.run(a,check=True,capture_output=True,text=True,timeout=8).stdout.strip()
 except Exception:
  return None
ps=cmd([OLLAMA,'ps'])
mem=None
for line in open('/proc/meminfo'):
 if line.startswith('MemAvailable:'): mem=int(line.split()[1])*1024; break
gpu=None
try:
 out=cmd([NVIDIA,'--query-gpu=name,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'])
 if out: gpu=[x.strip() for x in out.splitlines()]
except Exception: pass
print(json.dumps({'hostname':platform.node(),'architecture':platform.machine(),'uid':os.geteuid(),
 'netns_inode':os.stat('/proc/self/ns/net').st_ino,'memory_available_bytes':mem,
 'home_disk_free_bytes':shutil.disk_usage(HOME).free,'gpu_inventory':gpu,
 'ollama_service_pid':cmd([SYSTEMCTL,'show','-p','MainPID','--value',SERVICE]),
 'ollama_version':cmd([OLLAMA,'--version']),
 'loaded_model_count':(len([x for x in ps.splitlines()[1:] if x.strip()]) if ps is not None else None)},separators=(',',':')))
'''
    script = (script.replace("__OLLAMA__", json.dumps(config["server_binary_path"]))
              .replace("__HOME__", json.dumps(config["remote_home"]))
              .replace("__SERVICE__", json.dumps(config["remote_system_service"]))
              .replace("__SYSTEMCTL__", json.dumps(config["remote_systemctl_path"]))
              .replace("__NVIDIA__", json.dumps(config["remote_nvidia_smi_path"])))
    result = _ssh_run(config, script, timeout=20)
    if result.returncode != 0:
        raise ValueError("remote host metadata preflight failed")
    value = _json(result.stdout.strip(), "remote host snapshot")
    if (value.get("hostname") != config["remote_expected_hostname"] or value.get("uid") != config["remote_uid"]
            or not isinstance(value.get("memory_available_bytes"), int)
            or not isinstance(value.get("home_disk_free_bytes"), int)
            or type(value.get("loaded_model_count")) is not int
            or not isinstance(value.get("netns_inode"), int)):
        raise ValueError("remote host identity or capacity could not be verified")
    if value["loaded_model_count"] != 0:
        raise ValueError("existing Ollama service has loaded models; safe coexistence is not established")
    return value


def _remote_port_closed(config: dict[str, Any]) -> bool:
    script = ("import errno,socket,sys; s=socket.socket(); s.settimeout(2); "
              f"r=s.connect_ex(('127.0.0.1',{config['runtime_port']})); s.close(); "
              "sys.exit(0 if r == errno.ECONNREFUSED else 3)")
    return _ssh_run(config, script, timeout=10).returncode == 0


def _remote_install(config: dict[str, Any], remote_config_path: str) -> dict[str, Any]:
    adapter = config["remote_adapter_path"]
    script = ("import subprocess,sys; r=subprocess.run(["
              f"{config['remote_python_path']!r},'-I',{adapter!r},'--install','--config',{remote_config_path!r}],"
              f"capture_output=True,timeout={config['artifact_install_timeout_seconds']}); "
              "sys.stdout.buffer.write(r.stdout); sys.exit(r.returncode)")
    result = _ssh_run(config, script, timeout=config["artifact_install_timeout_seconds"] + 10)
    if result.returncode != 0:
        raise ValueError("verified private artifact installation failed")
    return _json(result.stdout.strip(), "private artifact installation result")


def _queue_protocol(process: subprocess.Popen) -> queue.Queue:
    # Four bounded 1 MiB lines cap reader buffering even if a child floods stdout.
    lines: queue.Queue = queue.Queue(maxsize=4)

    def enqueue(value: str | None) -> bool:
        try:
            lines.put(value, timeout=1)
            return True
        except queue.Full:
            if process.poll() is None:
                process.kill()
            return False

    def reader() -> None:
        assert process.stdout is not None
        while True:
            line = process.stdout.readline(1_048_577)
            if not line:
                break
            if len(line) > 1_048_576:
                enqueue("__oversized_protocol_line__")
                break
            try:
                if not enqueue(line.decode("utf-8").rstrip("\r\n")):
                    return
            except UnicodeDecodeError:
                enqueue("__invalid_utf8_protocol_line__")
                break
        enqueue(None)

    threading.Thread(target=reader, daemon=True).start()
    return lines


def _write_worker_config(path: Path, config: dict[str, Any]) -> None:
    raw = (json.dumps(config, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("worker configuration exceeds bounded size")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _remote_cleanup(config: dict[str, Any], workdir: str, unit: str) -> None:
    root = PurePosixPath(config["remote_workdir"])
    path = PurePosixPath(workdir)
    if (not _within_remote_home(workdir, config["remote_home"]) or path.parent != root
            or not re.fullmatch(r"run-[0-9a-f]{32}", path.name)):
        raise ValueError("refusing cleanup outside this task's private remote directory")
    script = ("import os,shutil,subprocess; "
              f"subprocess.run([{config['remote_sudo_path']!r},'-n',{config['remote_systemctl_path']!r},'stop',{unit!r}],capture_output=True,timeout=15); "
              f"shutil.rmtree({workdir!r},ignore_errors=False)")
    result = _ssh_run(config, script, timeout=30)
    if result.returncode != 0:
        raise ValueError("task-owned remote work directory cleanup failed")


def _worker_remote_command(config: dict[str, Any], adapter_path: str,
                           worker_config_path: str, unit: str) -> str:
    paths = ("remote_sudo_path", "remote_systemd_run_path", "remote_unshare_path",
             "remote_bash_path", "remote_ip_path", "remote_setpriv_path", "remote_python_path")
    if any(not _safe_absolute_posix_path(config[field]) for field in paths):
        raise ValueError("unsafe remote executable path in worker command")
    if (not _valid_ssh_user(config["remote_user"]) or not _valid_ssh_host(config["remote_host"])
            or not _within_remote_home(adapter_path, config["remote_home"])
            or not _within_remote_home(worker_config_path, config["remote_home"])
            or not re.fullmatch(r"lfg12b-[0-9a-f]{20}\.scope", unit)):
        raise ValueError("unsafe remote worker command identity or path")
    inner = (shlex.join([config["remote_ip_path"], "link", "set", "lo", "up"])
             + " && exec " + shlex.join([config["remote_setpriv_path"], "--reuid=" + str(config["remote_uid"]),
                                          "--regid=" + str(config["remote_uid"]), "--init-groups",
                                          config["remote_python_path"], "-I", adapter_path,
                                          "--worker", "--config", worker_config_path]))
    argv = [config["remote_sudo_path"], "-n", config["remote_systemd_run_path"],
            "--scope", "--collect", "--quiet", f"--unit={unit}",
            "-p", f"MemoryMax={config['resource_memory_max_bytes']}",
            "-p", f"CPUQuota={config['resource_cpu_quota_percent']}%",
            "-p", f"TasksMax={config['remote_scope_tasks_max']}", "--",
            config["remote_unshare_path"], "--net", "--fork", "--mount-proc",
            config["remote_bash_path"], "-c", inner]
    return shlex.join(argv)


def _assemble_report(config: dict[str, Any], config_digest: str, receipt: dict, receipt_digest: str,
                    baseline_report: dict, classical_config: dict, train_rows: list[dict],
                    validation_rows: list[dict], y_train, train_known, y_validation, target_known,
                    corpus: dict, classical_config_digest: str, payload: dict, payload_raw: bytes,
                    ready: dict, result: dict, host_before: dict, host_after: dict) -> dict[str, Any]:
    labels = sorted(classical_config["selected_labels"])
    outcomes = result["outcomes"]
    y_true = [[int(value) if value is not None else 0 for value in row]
              for row in (y_validation.tolist() if hasattr(y_validation, "tolist") else y_validation)]
    known = [[bool(value) for value in row]
             for row in (target_known.tolist() if hasattr(target_known, "tolist") else target_known)]
    predictions = _response_prediction_matrix(result["predictions"], labels, len(validation_rows))
    policy = _target_scoring_policy(y_train, train_known, labels, classical_config["constant_target_behavior"])
    review = _baseline_review_provenance(baseline_report)
    resources_before = ready.get("resources_before", {})
    resources_after = result.get("resources_after", {})
    metrics = _compute_metrics(y_true, predictions, known, labels)
    status_counts = {status: outcomes.count(status) for status in sorted(set(outcomes))}
    lifecycle = {"completed_responses": sum(status in {"answered", "abstain", "malformed", "invalid_schema",
                                                       "refusal", "truncated", "identity_mismatch"}
                                             for status in outcomes),
                 "not_started": sum(status in {"request_too_large_not_started", "not_attempted_after_abort"}
                                    for status in outcomes),
                 "unknown_or_ambiguous": sum(status in {"timeout_unknown_worker_lifecycle", "transport_or_protocol_error",
                                                        "incomplete", "http_error"} for status in outcomes)}
    target_id_hashes = [row["id_sha256"] for row in payload["records"]]
    report = {
        "schema_version": REPORT_SCHEMA, "promotion_eligible": False, "qualification_claim": False,
        "task": {"field": classical_config["target_field"], "labels": labels,
                 "taxonomy_sha256": corpus["taxonomy_sha256"], "validation_rows": len(validation_rows),
                 "validation_record_id_sha256": baseline_report["data"]["validation_record_id_sha256"],
                 "target_observation_support": baseline_report["task"]["validation_support"],
                 "target_scoring_policy": policy,
                 "target_mask_sha256": {"train": _target_mask_digest(train_rows, train_known),
                                        "validation": _target_mask_digest(validation_rows, target_known)},
                 "target_mask_identity": "local source re-screen and per-row ID/text digests were rebound; worker recomputed validation text, ID-set, and mask digests"},
        "baseline": {"report_sha256": config["baseline_report_sha256"],
                     "report_code_sha256": baseline_report["code_sha256"],
                     "config_sha256": classical_config_digest,
                     "algorithms": {name: value.get("metrics") for name, value in baseline_report["algorithms"].items()},
                     "paired_classical_metrics": "unavailable; original aggregate report has no per-row predictions"},
        "model": {"identity": config["model_name"], "quantization": config["expected_quantization"],
                  "receipt_sha256": receipt_digest, "artifact": result["artifact"],
                  "runtime": result["loaded_model"], "server_binary_sha256": config["server_binary_sha256"],
                  "server_version": config["server_version"], "runtime_boundary": "distinct private network namespace; loopback only",
                  "inference_device": "unknown; runtime response did not report actual CPU/GPU placement",
                  "source_revision": receipt.get("sources", {}).get("upstream_model_revision"),
                  "exact_full_precision_conversion_linkage_independently_proven": (
                      receipt.get("provenance", {}).get("exact_conversion_linkage_to_upstream_revision_independently_proven") is True),
                  "license": config["expected_license_name"]},
        "inference": {"rows_considered": len(outcomes), "http_requests_started": _http_requests_started(outcomes),
                      "validation_rows": len(validation_rows), "outcome_counts": status_counts,
                      "worker_lifecycle": lifecycle, "aborted": result["aborted"],
                      "abort_reason": result["abort_reason"], "timing_ms": _stats(result["timing_ms"]),
                      "warmup": ready["warmup"], "cold_model_load_time_ms": None,
                      "timing_scope": "sequential local HTTP request through complete Ollama response; model loaded by empty-input warmup first",
                      "prompt_revision": config["prompt_revision"],
                      "prompt_sha256": sha(config["system_prompt"].encode("utf-8")),
                      "schema_sha256": sha(json.dumps(output_schema(labels), sort_keys=True, separators=(",", ":")).encode()),
                      "num_ctx": config["num_ctx"], "num_predict": config["num_predict"],
                      "temperature": config["temperature"], "seed": config["seed"], "top_k": config["top_k"],
                      "top_p": config["top_p"], "repeat_penalty": config["repeat_penalty"],
                      "keep_alive": config["keep_alive"], "max_request_bytes": config["max_request_bytes"],
                      "max_response_bytes": config["max_response_bytes"], "timeout_seconds": config["timeout_seconds"],
                      "payload_sha256": sha(payload_raw), "response_record_id_set_sha256": sha(
                          json.dumps(sorted(target_id_hashes), separators=(",", ":")).encode())},
        "metrics": metrics,
        "resources": {"remote_before": resources_before, "remote_after": resources_after,
                      "configured_transient_scope": {"memory_max_bytes": config["resource_memory_max_bytes"],
                                                       "cpu_quota_percent": config["resource_cpu_quota_percent"],
                                                       "scope_started_successfully": True},
                      "per_run_cpu_time_usec": None,
                      "per_run_peak_memory_bytes": None,
                      "cgroup_usage_scope": "unknown; the worker did not collect per-run cgroup CPU or peak-memory counters",
                      "cgroup_memory_available_bytes_before": resources_before.get("cgroup_memory_available_bytes"),
                      "cgroup_memory_available_bytes_after": resources_after.get("cgroup_memory_available_bytes"),
                      "gpu_memory_attribution": {"before": resources_before.get("gpu_attribution", "unknown"),
                                                  "after": resources_after.get("gpu_attribution", "unknown"),
                                                  "per_process": "unknown; no per-process GPU memory measurement was collected"},
                      "model_layer_size_bytes": result["artifact"]["model_layer_bytes"],
                      "model_layer_size_scope": "quantized model blob size; not runtime memory"},
        "provenance": {"baseline_report_sha256": config["baseline_report_sha256"],
                       "classical_config_sha256": classical_config_digest,
                       "plan_sha256": baseline_report["data"]["plan_sha256"], "manifest_sha256": corpus["manifest_sha256"],
                       "policy_sha256": corpus["policy_sha256"], "screen_code_sha256": corpus["screen_code_sha256"],
                       "review_ledger_sha256": baseline_report["data"]["review_ledger_sha256"],
                       "train_sha256": corpus["train_sha256"], "validation_sha256": corpus["validation_sha256"],
                       "taxonomy_sha256": corpus["taxonomy_sha256"],
                       "rejected_label_handling": classical_config["rejected_label_handling"],
                       "rejected_rows_excluded": baseline_report["data"]["rejected_rows_excluded"],
                       **review,
                       "privacy": "screened text sent once over authenticated SSH stdin only after private namespace, artifact, model identity, and host-loopback denial checks; no text/output logs or public/cloud service"},
        "limitations": _review_history_limitations(review),
        "host_state": {"before": {k: host_before[k] for k in ("hostname", "architecture", "memory_available_bytes",
                         "home_disk_free_bytes", "gpu_inventory", "ollama_service_pid", "ollama_version", "loaded_model_count")},
                       "after": {k: host_after[k] for k in ("hostname", "architecture", "memory_available_bytes",
                         "home_disk_free_bytes", "gpu_inventory", "ollama_service_pid", "ollama_version", "loaded_model_count")},
                       "sampled_service_comparisons": {"ollama_service_pid_matches": host_before["ollama_service_pid"] == host_after["ollama_service_pid"],
                                                       "ollama_version_matches": host_before["ollama_version"] == host_after["ollama_version"],
                                                       "loaded_model_count_matches": host_before["loaded_model_count"] == host_after["loaded_model_count"],
                                                       "loaded_model_identities": "not applicable; count was zero at both snapshots"},
                       "observation_scope": "point-in-time snapshots only; intervening intervals and other services were not monitored"},
        "comparator": {"code_sha256": config["worker_code_sha256"],
                       "git_revision": config.get("comparator_git_revision")},
        "config_sha256": config_digest,
    }
    return report


def run(config_path: Path, output: Path) -> dict[str, Any]:
    global CURRENT_STAGE
    CURRENT_STAGE = "local_artifact_receipt"
    config, config_digest = load_config(config_path)
    comparator_code_sha256 = sha(Path(__file__).read_bytes())
    config = {**config, "worker_code_sha256": comparator_code_sha256,
              "comparator_git_revision": _git_revision_for_comparator()}
    receipt, receipt_digest = verify_acquisition(config)
    CURRENT_STAGE = "baseline_binding"
    (baseline_report, classical_config, train_rows, validation_rows, y_train, train_known,
     y_validation, target_known, taxonomy, corpus, classical_config_digest) = _bind_baseline(config)
    labels = sorted(classical_config["selected_labels"])
    payload = _make_payload(config, baseline_report, classical_config, train_rows, validation_rows,
                            y_train, train_known, y_validation, target_known, corpus, classical_config_digest)
    payload_raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    if len(payload_raw) > config["max_payload_bytes"]:
        raise ValueError("screened worker input exceeds configured payload size")

    CURRENT_STAGE = "private_report_directory"
    output_dir = _private_report_directory(output)
    run_id = uuid.uuid4().hex
    base = config["remote_workdir"].rstrip("/")
    workdir = f"{base}/run-{run_id}"
    model_store = config["remote_model_store"]
    if not _within_remote_home(base, config["remote_home"]):
        raise ValueError("remote work directory must remain inside the configured user home")
    adapter_path = f"{workdir}/compare_local_12b.py"
    config_path_remote = f"{workdir}/worker-config.json"
    worker_config = {**config, "acquisition_receipt": config["remote_acquisition_receipt"],
                     "model_store": model_store, "remote_model_store": model_store,
                     "remote_workdir": workdir, "remote_adapter_path": adapter_path,
                     "endpoint_url": f"http://127.0.0.1:{config['runtime_port']}/api/chat",
                     "worker_code_sha256": sha(Path(__file__).read_bytes())}
    worker_config_raw = (json.dumps(worker_config, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(worker_config_raw) > MAX_CONFIG_BYTES:
        raise ValueError("remote worker configuration exceeds bounded size")

    CURRENT_STAGE = "fresh_host_capacity_and_service_check"
    host_before = _remote_snapshot(config)
    if (host_before["architecture"] != config["remote_expected_architecture"]
            or host_before["memory_available_bytes"] < config["min_memory_available_bytes"]
            or host_before["home_disk_free_bytes"] < config["min_disk_free_bytes"]):
        raise ValueError("fresh remote capacity is below the configured coexistence floor")
    if not _remote_port_closed(config):
        raise ValueError("configured host loopback port is already occupied or not demonstrably closed")
    unit = f"lfg12b-{run_id[:20]}.scope"
    CURRENT_STAGE = "transient_resource_control_preflight"
    preflight_argv = [config["remote_sudo_path"], "-n", config["remote_systemd_run_path"],
                      "--scope", "--collect", "--quiet", f"--unit=probe-{run_id[:12]}.scope",
                      "-p", f"MemoryMax={config['resource_memory_max_bytes']}",
                      "-p", f"CPUQuota={config['resource_cpu_quota_percent']}%",
                      "-p", f"TasksMax={config['remote_scope_tasks_max']}", "--", config["remote_true_path"]]
    preflight_script = f"import subprocess,sys; r=subprocess.run({preflight_argv!r},capture_output=True,timeout=30); sys.exit(r.returncode)"
    if _ssh_run(config, preflight_script, timeout=40).returncode != 0:
        raise ValueError("temporary cgroup resource limit preflight failed")

    work_created = False
    process = None
    ready = None
    result = None
    try:
        CURRENT_STAGE = "private_remote_workspace"
        create_script = ("import os,pathlib; p=pathlib.Path(" + repr(workdir) + "); "
                         "p.parent.mkdir(mode=0o700,parents=True,exist_ok=True); p.mkdir(mode=0o700); "
                         "os.chmod(p,0o700)")
        if _ssh_run(config, create_script).returncode != 0:
            raise ValueError("private remote task directory creation failed")
        work_created = True
        local_adapter = Path(__file__).resolve(strict=True)
        scp_args = ["scp", "-oBatchMode=yes", "-oConnectTimeout=10", "-P", str(config["ssh_port"]), str(local_adapter),
                    f"{config['remote_user']}@{config['remote_host']}:{adapter_path}"]
        if subprocess.run(scp_args, check=False, capture_output=True, timeout=120, shell=False).returncode != 0:
            raise ValueError("comparator transfer to the private worker failed")
        worker_config_path = output_dir / ".worker-config.json"
        _write_worker_config(worker_config_path, worker_config)
        scp_cfg = ["scp", "-oBatchMode=yes", "-oConnectTimeout=10", "-P", str(config["ssh_port"]), str(worker_config_path),
                   f"{config['remote_user']}@{config['remote_host']}:{config_path_remote}"]
        if subprocess.run(scp_cfg, check=False, capture_output=True, timeout=30, shell=False).returncode != 0:
            raise ValueError("private worker configuration transfer failed")
        protect_script = ("import os,pathlib,sys; w=pathlib.Path(" + repr(workdir) + "); "
                          "a=pathlib.Path(" + repr(adapter_path) + "); c=pathlib.Path(" + repr(config_path_remote) + "); "
                          "os.chmod(a,0o600); os.chmod(c,0o600); "
                          "sys.exit(0 if w.stat().st_uid == " + str(config["remote_uid"]) +
                          " and (w.stat().st_mode & 0o777) == 0o700 and (a.stat().st_mode & 0o077) == 0 "
                          "and (c.stat().st_mode & 0o077) == 0 else 5)")
        if _ssh_run(config, protect_script).returncode != 0:
            raise ValueError("remote worker directory or files are not owner-private")
        CURRENT_STAGE = "verified_artifact_install"
        _remote_install({**config, "remote_adapter_path": adapter_path}, config_path_remote)
        if not _remote_port_closed(config):
            raise ValueError("configured host loopback port became occupied before worker start")
        CURRENT_STAGE = "isolated_worker_startup"
        worker_command = _worker_remote_command(config, adapter_path, config_path_remote, unit)
        process = subprocess.Popen(["ssh", "-T", "-oBatchMode=yes", "-oConnectTimeout=10", "-p", str(config["ssh_port"]),
                                    f"{config['remote_user']}@{config['remote_host']}", worker_command],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   shell=False)
        lines = _queue_protocol(process)
        ready = _read_protocol_line(process, lines,
                                    config["worker_start_timeout_seconds"] + config["worker_ready_wait_margin_seconds"])
        if ready.get("phase") != "ready":
            raise ValueError("isolated worker did not prove readiness and model identity")
        if (ready.get("model_manifest_sha256") != config["expected_manifest_sha256"]
                or ready.get("artifact_receipt_sha256") != receipt_digest
                or ready.get("server_binary_sha256") != config["server_binary_sha256"]
                or ready.get("server_version") != config["server_version"]
                or ready.get("worker_code_sha256") != comparator_code_sha256
                or ready.get("comparator_git_revision") != config.get("comparator_git_revision")
                or ready.get("netns_inode") == host_before["netns_inode"]
                or ready.get("port") != config["runtime_port"]):
            raise ValueError("worker readiness did not match expected private runtime identity")
        CURRENT_STAGE = "outside_namespace_loopback_probe"
        if not _remote_port_closed(config):
            raise ValueError("host-network process can reach the worker loopback listener")
        CURRENT_STAGE = "screened_validation_inference"
        assert process.stdin is not None
        process.stdin.write(payload_raw)
        process.stdin.flush()
        process.stdin.close()
        progress = []
        batch_timeout = (config["timeout_seconds"] * (len(validation_rows) + 2)
                         + config["batch_overhead_margin_seconds"])
        if not math.isfinite(batch_timeout) or batch_timeout > sys.maxsize / 1_000_000_000:
            raise ValueError("total batch timeout exceeds safe platform arithmetic")
        deadline = time.monotonic() + batch_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("isolated request batch exceeded total configured deadline")
            message = _read_protocol_line(process, lines, remaining)
            if message.get("phase") == "progress":
                status = message.get("outcome")
                elapsed = message.get("timing_ms")
                timing_valid = ((status in NOT_STARTED_OUTCOMES and elapsed is None)
                                or (status not in NOT_STARTED_OUTCOMES
                                    and type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0))
                if (message.get("index") != len(progress)
                        or message.get("outcome") not in {"answered", "abstain", "malformed", "invalid_schema",
                                                         "refusal", "truncated", "incomplete", "identity_mismatch",
                                                         "http_error", "timeout_unknown_worker_lifecycle",
                                                         "transport_or_protocol_error", "request_too_large_not_started",
                                                         "not_attempted_after_abort"}
                        or not isinstance(message.get("prediction"), list)
                        or len(message["prediction"]) != len(labels) or not timing_valid):
                    raise ValueError("worker progress protocol is incomplete or contradictory")
                progress.append(message)
                continue
            result = message
            break
        return_code = process.wait(timeout=30)
        if return_code != 0 or result.get("phase") != "result" or result.get("server_cleanup_verified") is not True:
            raise ValueError("isolated inference worker failed after the private request batch")
        if (result.get("payload_sha256") != sha(payload_raw)
                or result.get("validation_record_id_sha256") != payload["binding"]["validation_record_id_sha256"]
                or result.get("target_mask_sha256") != payload["binding"]["validation_target_mask_sha256"]
                or len(result.get("predictions", [])) != len(validation_rows)
                or len(result.get("outcomes", [])) != len(validation_rows)):
            raise ValueError("worker results do not bind to the submitted records and masks")
        _validate_final_timings(result.get("outcomes"), result.get("timing_ms"), progress,
                                len(validation_rows))
        if (len(progress) != len(result["outcomes"])
                or [item["outcome"] for item in progress] != result["outcomes"]
                or [item["prediction"] for item in progress] != result["predictions"]):
            raise ValueError("worker progress and final result disagree")
        host_after = _remote_snapshot(config)
        if (host_after["ollama_service_pid"] != host_before["ollama_service_pid"]
                or host_after["ollama_version"] != host_before["ollama_version"]
                or host_after["loaded_model_count"] != host_before["loaded_model_count"]):
            raise ValueError("existing Ollama service state changed during isolated comparison")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        if work_created:
            _remote_cleanup(config, workdir, unit)
        try:
            (output_dir / ".worker-config.json").unlink()
        except FileNotFoundError:
            pass

    CURRENT_STAGE = "aggregate_report"
    report = _assemble_report(config, config_digest, receipt, receipt_digest, baseline_report,
                              classical_config, train_rows, validation_rows, y_train, train_known,
                              y_validation, target_known, corpus, classical_config_digest, payload,
                                  payload_raw, ready, result, host_before, host_after)
    output_path = output_dir / "local-12b-report.json"
    encoded_report = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded_report) > config["max_report_bytes"]:
        raise ValueError("aggregate report exceeds configured report size")
    CURRENT_STAGE = "private_report_write"
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded_report)
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.worker and args.install:
        raise SystemExit("worker and install modes are mutually exclusive")
    try:
        config, config_digest = load_config(args.config)
        if args.worker:
            if config.get("worker_code_sha256") != sha(Path(__file__).read_bytes()):
                raise ValueError("worker code identity mismatch")
            _worker(config)
            return
        if args.install:
            config = {**config, "acquisition_receipt": config["remote_acquisition_receipt"],
                      "model_store": config["remote_model_store"]}
            receipt, receipt_digest = verify_acquisition(config)
            artifact = _copy_verified_artifact(config, receipt)
            print(json.dumps({"status": "private_artifact_installed_and_verified",
                              "receipt_sha256": receipt_digest, "artifact": artifact,
                              "config_sha256": config_digest}, sort_keys=True, separators=(",", ":")))
            return
        if args.output is None:
            raise ValueError("an owner-private output directory is required for local coordinator mode")
        report = run(args.config, args.output)
    except (OSError, ValueError, TypeError, KeyError, IndexError, TimeoutError,
            http.client.HTTPException, subprocess.SubprocessError, ImportError) as exc:
        print(json.dumps({"status": "stopped", "stage": CURRENT_STAGE,
                          "error_class": type(exc).__name__, "private_data_echoed": False},
                         sort_keys=True), file=sys.stderr)
        raise SystemExit("local comparison stopped; inspect the sanitized stage result without sharing prompt/output data") from None
    result = json.dumps({"schema_version": report["schema_version"],
                         "report_sha256": sha((json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()),
                         "http_requests_started": report["inference"]["http_requests_started"],
                         "promotion_eligible": False}, sort_keys=True)
    print(result)


if __name__ == "__main__":
    main()
