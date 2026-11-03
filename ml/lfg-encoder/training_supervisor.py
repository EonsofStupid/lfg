#!/usr/bin/env python3
"""Launch one configured training run and recover only from a complete checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import train


ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "train.py"
COMPLETE = "supervised-complete.json"
REQUIRED_OPTIONS = {"--dataset", "--output", "--base-model", "--model-revision",
                    "--model-manifest", "--environment-lock", "--wheelhouse"}


def hash_file(path: Path) -> str:
    before = path.stat()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular, non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str, argparse.Namespace]:
    if not path.is_absolute():
        raise ValueError("service configuration path must be absolute")
    raw = path.read_bytes()
    value = json.loads(raw)
    fields = {"schema_version", "python", "trainer_sha256", "trainer_arguments", "recovery", "resource_limits"}
    if not isinstance(value, dict) or not fields <= value.keys() or value["schema_version"] != "lfg.reviewed-training-service.v1":
        raise ValueError("invalid training service configuration")
    python = value["python"]
    if not isinstance(python, str) or not Path(python).is_absolute() or not Path(python).is_file():
        raise ValueError("python must name an existing absolute interpreter path")
    if not isinstance(value["trainer_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["trainer_sha256"]):
        raise ValueError("trainer_sha256 must be lowercase SHA-256")
    if hash_file(TRAINER) != value["trainer_sha256"]:
        raise ValueError("trainer digest does not match service configuration")
    if value["recovery"] not in ("resume", "manual"):
        raise ValueError("recovery must be resume or manual")
    limits = value["resource_limits"]
    limit_fields = {"cpu_quota_percent", "memory_max_bytes", "tasks_max"}
    if (not isinstance(limits, dict) or not limit_fields <= limits.keys()
            or any(type(limits[name]) is not int or limits[name] < 1 for name in limit_fields)):
        raise ValueError("positive CPU, memory and task limits are required")
    arguments = value["trainer_arguments"]
    if not isinstance(arguments, list) or not all(isinstance(item, str) and item for item in arguments):
        raise ValueError("trainer_arguments must be a list of nonempty strings")
    supplied = {item.split("=", 1)[0] for item in arguments if item.startswith("--")}
    if not REQUIRED_OPTIONS <= supplied:
        raise ValueError("training requires explicit dataset, output, model and environment inputs")
    parser = train.build_parser()
    parser.allow_abbrev = False
    try:
        parsed = parser.parse_args(["train", *arguments])
    except SystemExit as error:
        raise ValueError("invalid trainer arguments") from error
    mode = value.get("training_mode", "authorized" if "--allow-below-gate" in supplied else "reviewed")
    if mode not in ("authorized", "reviewed"):
        raise ValueError("training_mode must be authorized or reviewed")
    if (parsed.resume or (mode == "reviewed" and "--allow-below-gate" in supplied)
            or (mode == "authorized" and "--require-reviewed" in supplied)):
        raise ValueError("training mode conflicts with trainer arguments or manually supplied resume")
    if mode == "reviewed" and parsed.audited_plan is None:
        raise ValueError("reviewed training requires an audited plan")
    for name in ("dataset", "output", "base_model", "model_manifest", "environment_lock", "wheelhouse",
                 "training_config", "promotion_policy", "support_policy"):
        if not Path(getattr(parsed, name)).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    if parsed.audited_plan is not None and not Path(parsed.audited_plan).is_absolute():
        raise ValueError("audited_plan must be an absolute path")
    if not re.fullmatch(r"[0-9a-f]{40}", parsed.model_revision or ""):
        raise ValueError("model revision must be a full lowercase commit ID")
    train.validate_input_options(False, parsed.environment_lock, parsed.wheelhouse,
                                 parsed.model_manifest, parsed.environment_allowed_extra)
    if path.read_bytes() != raw:
        raise ValueError("service configuration changed while reading")
    return value, hashlib.sha256(raw).hexdigest(), parsed


def current_cgroup() -> Path:
    if sys.platform != "linux":
        raise ValueError("reviewed training service requires Linux cgroup v2")
    groups = [line.split("::", 1)[1] for line in Path("/proc/self/cgroup").read_text().splitlines()
              if line.startswith("0::")]
    if len(groups) != 1:
        raise ValueError("cannot determine the current cgroup v2 path")
    group = PurePosixPath(groups[0])
    if not group.is_absolute() or ".." in group.parts:
        raise ValueError("invalid cgroup path")
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, separator, after = line.partition(" - ")
        if not separator or not after.startswith("cgroup2 "):
            continue
        fields = before.split()
        if len(fields) < 5:
            continue
        def unescape(value: str) -> str:
            return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)
        root = PurePosixPath(unescape(fields[3]))
        mount = Path(unescape(fields[4])).resolve()
        try:
            relative = group.relative_to(root)
        except ValueError:
            continue
        candidate = (mount / str(relative)).resolve()
        if candidate.is_relative_to(mount):
            mounts.append((len(root.parts), candidate))
    if not mounts:
        raise ValueError("cannot locate the current cgroup v2 mount")
    return max(mounts, key=lambda item: item[0])[1]


def verify_cgroup_limits(limits: dict[str, int], cgroup: Path | None = None) -> dict[str, int]:
    directory = cgroup if cgroup is not None else current_cgroup()
    cpu = (directory / "cpu.max").read_text().split()
    memory = (directory / "memory.max").read_text().strip()
    tasks = (directory / "pids.max").read_text().strip()
    if len(cpu) != 2 or not all(value.isdecimal() for value in (*cpu, memory, tasks)):
        raise ValueError("CPU, memory and task cgroup limits must all be finite")
    quota, period = map(int, cpu)
    memory_bytes, task_count = int(memory), int(tasks)
    if (min(quota, period, memory_bytes, task_count) < 1 or quota * 100 > limits["cpu_quota_percent"] * period
            or memory_bytes > limits["memory_max_bytes"] or task_count > limits["tasks_max"]):
        raise ValueError("effective cgroup limits exceed the configured maximums")
    return {"cpu_quota_us": quota, "cpu_period_us": period, "memory_max_bytes": memory_bytes,
            "tasks_max": task_count}


def output_files(output: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise ValueError("training output contains a symbolic link")
        if path.is_file() and path.name != COMPLETE:
            files[path.name] = hash_file(path)
    required = {"run-manifest.json", "trainer_state.json", "labels.json"}
    if not required <= files.keys() or any((output / name).stat().st_size == 0 for name in required):
        raise ValueError("finished training output is incomplete")
    monolithic = {name for name in ("model.safetensors", "pytorch_model.bin")
                  if name in files and (output / name).stat().st_size > 0}
    if not monolithic:
        indexes = {"model.safetensors.index.json", "pytorch_model.bin.index.json"} & files.keys()
        if not indexes:
            raise ValueError("finished training output is missing weights")
        for name in indexes:
            index = json.loads((output / name).read_bytes())
            mapping = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(mapping, dict) or not mapping or any(
                not isinstance(shard, str) or Path(shard).name != shard or shard not in files
                or (output / shard).stat().st_size == 0 for shard in mapping.values()
            ):
                raise ValueError("finished training output has missing or invalid weight shards")
    return files


def completion(output: Path, config_sha: str, trainer_sha: str, supervisor_sha: str) -> dict[str, Any]:
    return {"schema_version": "lfg.supervised-training-complete.v1", "service_config_sha256": config_sha,
            "trainer_sha256": trainer_sha, "supervisor_sha256": supervisor_sha,
            "output_files": output_files(output)}


def choose_mode(output: Path, recovery: str, config_sha: str, trainer_sha: str, supervisor_sha: str) -> str:
    marker = output / COMPLETE
    if output.is_symlink() or marker.is_symlink():
        raise ValueError("output and completion marker cannot be symbolic links")
    if marker.exists():
        if json.loads(marker.read_bytes()) != completion(output, config_sha, trainer_sha, supervisor_sha):
            raise ValueError("completed output or service configuration changed")
        return "complete"
    if not output.exists() or (output.is_dir() and not any(output.iterdir())):
        return "start"
    if not output.is_dir() or (output / "trainer_state.json").exists():
        raise ValueError("ambiguous output; inspect completed files before retrying")
    if recovery != "resume" or not (output / "run-manifest.json").is_file():
        raise ValueError("existing output requires explicit recovery and its original run manifest")
    candidates = [path for path in output.iterdir() if re.fullmatch(r"checkpoint-([0-9]+)", path.name)]
    if any(path.is_symlink() or not path.is_dir() for path in candidates):
        raise ValueError("checkpoint path is not a regular directory")
    checkpoints = [(int(path.name.removeprefix("checkpoint-")), path) for path in candidates]
    if len({number for number, _ in checkpoints}) != len(checkpoints):
        raise ValueError("checkpoint numbers are ambiguous")
    if not checkpoints:
        raise ValueError("no checkpoint is available; choose a new output after inspection")
    latest = max(checkpoints, key=lambda item: item[0])[1]
    train.validate_checkpoint(latest)
    return "resume"


def execute(config_path: Path) -> int:
    config, config_sha, parsed = load_config(config_path)
    effective_limits = verify_cgroup_limits(config["resource_limits"])
    print("effective cgroup limits: " + json.dumps(effective_limits, sort_keys=True), flush=True)
    supervisor_sha = hash_file(Path(__file__))
    output = Path(parsed.output)
    with train.training_run_lock(output):
        mode = choose_mode(output, config["recovery"], config_sha, config["trainer_sha256"], supervisor_sha)
    if mode == "complete":
        print("supervised run already completed; output identities match", flush=True)
        return 0
    command = [config["python"], str(TRAINER), "train", *config["trainer_arguments"]]
    supplied = {item.split("=", 1)[0] for item in config["trainer_arguments"] if item.startswith("--")}
    training_mode = config.get("training_mode", "authorized" if "--allow-below-gate" in supplied else "reviewed")
    if training_mode == "authorized" and "--allow-below-gate" not in supplied:
        command.append("--allow-below-gate")
    elif training_mode == "reviewed" and "--require-reviewed" not in supplied:
        command.append("--require-reviewed")
    if training_mode == "authorized" and "--verify-installed-payload" not in supplied:
        command.append("--verify-installed-payload")
    if not parsed.development_plan_only:
        command.append("--development-plan-only")
    if mode == "resume":
        command.append("--resume")
    print(f"starting {training_mode} training ({mode}); environment and resource limits remain enforced", flush=True)
    child = subprocess.Popen(command)
    previous = {}
    def forward(signum: int, _frame: Any) -> None:
        if child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, forward)
        result = child.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if result != 0:
        print(f"trainer exited with status {result}; inspect its logs and checkpoint", file=sys.stderr, flush=True)
        return 1 if result < 0 else 2
    with train.training_run_lock(output):
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != config_sha:
            raise ValueError("service configuration changed during training")
        if hash_file(TRAINER) != config["trainer_sha256"]:
            raise ValueError("trainer changed during training")
        if hash_file(Path(__file__)) != supervisor_sha:
            raise ValueError("supervisor changed during training")
        result_record = completion(output, config_sha, config["trainer_sha256"], supervisor_sha)
        with (output / COMPLETE).open("x", encoding="utf-8") as stream:
            json.dump(result_record, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    print("training finished; independent evaluation remains required before promotion", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        raise SystemExit(execute(args.config))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(f"training service rejected: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
