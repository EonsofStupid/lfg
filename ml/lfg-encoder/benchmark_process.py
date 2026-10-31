#!/usr/bin/env python3
"""Benchmark the real baseline process on a bounded, operator-selected corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != "lfg.process-benchmark-config.v1":
        raise ValueError("unsupported benchmark configuration")
    for field in ("measured_runs", "repetitions", "max_records", "max_dataset_bytes", "max_batch_bytes", "timeout_seconds"):
        if type(config.get(field)) is not int or config[field] < 1:
            raise ValueError(f"invalid {field}")
    if type(config.get("warmup_runs")) is not int or config["warmup_runs"] < 0:
        raise ValueError("invalid warmup_runs")
    return config, sha(raw)


def prepare_batch(dataset: bytes, config: dict[str, Any]) -> tuple[bytes, int, int]:
    if len(dataset) > config["max_dataset_bytes"]:
        raise ValueError("dataset exceeds configured byte limit")
    rows = [json.loads(line) for line in dataset.decode("utf-8").splitlines() if line.strip()]
    if not rows or len(rows) > config["max_records"]:
        raise ValueError("dataset record count exceeds configured limits")
    ids: set[str] = set()
    templates = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in ids:
            raise ValueError("dataset IDs must be nonempty and unique")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("dataset text must be nonempty")
        ids.add(row["id"])
        templates.append(row["text"])
    parts: list[bytes] = []
    size = 0
    for repetition in range(config["repetitions"]):
        for index, prompt in enumerate(templates):
            line = (json.dumps({"id": f"{repetition}:{index}", "text": prompt}, ensure_ascii=False) + "\n").encode("utf-8")
            size += len(line)
            if size > config["max_batch_bytes"]:
                raise ValueError("batch exceeds configured byte limit")
            parts.append(line)
    return b"".join(parts), len(rows), len(rows) * config["repetitions"]


def windows_counters(process: subprocess.Popen) -> dict[str, int | None]:
    if os.name != "nt":
        return {"process_cpu_100ns": None, "process_cycles": None}
    import ctypes
    import ctypes.wintypes as wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.QueryProcessCycleTime.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_ulonglong)]
    kernel.QueryProcessCycleTime.restype = wintypes.BOOL
    creation, exit_time, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
    ok_time = kernel.GetProcessTimes(int(process._handle), ctypes.byref(creation), ctypes.byref(exit_time),
                                     ctypes.byref(kernel_time), ctypes.byref(user_time))
    cycles = ctypes.c_ulonglong()
    ok_cycles = kernel.QueryProcessCycleTime(int(process._handle), ctypes.byref(cycles))
    ticks = lambda value: (value.dwHighDateTime << 32) | value.dwLowDateTime
    return {"process_cpu_100ns": ticks(kernel_time) + ticks(user_time) if ok_time else None,
            "process_cycles": cycles.value if ok_cycles else None}


def run_process(binary: Path, payload: bytes, timeout: int, capture: bool = False) -> tuple[bytes, dict[str, Any]]:
    started = time.perf_counter_ns()
    process = subprocess.Popen([str(binary), "baseline-predict"], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE if capture else subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        stdout, _ = process.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise ValueError("benchmark process timed out") from None
    elapsed = time.perf_counter_ns() - started
    counters = windows_counters(process)
    if process.returncode != 0:
        raise ValueError(f"baseline process failed with exit code {process.returncode}")
    return stdout or b"", {"elapsed_ns": elapsed, **counters}


def run(binary: Path, dataset: Path, config_path: Path) -> dict[str, Any]:
    config, config_hash = read_config(config_path)
    binary = binary.resolve(strict=True)
    binary_hash = sha(binary.read_bytes())
    override_paths = {key: Path(os.environ[key]) for key in ("LFG_PREFLIGHT_CONFIG", "LFG_OPERATIONS_CONFIG") if key in os.environ}
    override_hashes = {key: sha(path.read_bytes()) for key, path in override_paths.items()}
    if dataset.stat().st_size > config["max_dataset_bytes"]:
        raise ValueError("dataset exceeds configured byte limit")
    raw = dataset.read_bytes()
    payload, rows, requests = prepare_batch(raw, config)
    first = payload.splitlines()[0] + b"\n"
    probe, _ = run_process(binary, first, config["timeout_seconds"], capture=True)
    model = json.loads(probe)["model"]
    for _ in range(config["warmup_runs"]):
        run_process(binary, payload, config["timeout_seconds"])
    samples = [run_process(binary, payload, config["timeout_seconds"])[1] for _ in range(config["measured_runs"])]
    after, _ = run_process(binary, first, config["timeout_seconds"], capture=True)
    if (json.loads(after)["model"] != model or sha(binary.read_bytes()) != binary_hash or
            sha(dataset.read_bytes()) != sha(raw) or
            sha(config_path.read_bytes()) != config_hash or
            {key: sha(path.read_bytes()) for key, path in override_paths.items()} != override_hashes):
        raise ValueError("benchmark inputs or effective encoder changed during measurement")
    def median(field: str) -> float | None:
        values = [sample[field] for sample in samples]
        return None if any(value is None for value in values) else statistics.median(values)
    return {"schema_version": "lfg.process-benchmark.v1", "promotion_eligible": False,
            "binary_sha256": binary_hash, "dataset_sha256": sha(raw), "config_sha256": config_hash,
            "benchmark_sha256": sha(Path(__file__).read_bytes()),
            "encoder": model, "override_sha256": override_hashes, "platform": platform.platform(), "machine": platform.machine(),
            "python_version": platform.python_version(),
            "processor": platform.processor() or None, "logical_cpus": os.cpu_count(),
            "rows": rows, "requests_per_run": requests, "warmup_runs": config["warmup_runs"],
            "measured_runs": config["measured_runs"], "samples": samples,
            "median_elapsed_ns": median("elapsed_ns"), "median_process_cpu_100ns": median("process_cpu_100ns"),
            "median_process_cycles": median("process_cycles"),
            "counter_source": "Win32 GetProcessTimes and QueryProcessCycleTime" if os.name == "nt" else None,
            "measurement_scope": "whole process including startup, input parsing, encoding and output serialization"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new private report path")
    args = parser.parse_args()
    try:
        result = run(args.binary, args.dataset, args.config)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
        print(json.dumps(result, allow_nan=False))
    except (OSError, ValueError, KeyError, UnicodeError, TypeError) as error:
        raise SystemExit(f"benchmark rejected: {error}") from error


if __name__ == "__main__":
    main()
