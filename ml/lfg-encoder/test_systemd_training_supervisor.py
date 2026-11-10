"""Opt-in systemd recovery test; fixture files are not model training data."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path


@unittest.skipUnless(sys.platform == "linux" and os.environ.get("LFG_TEST_SYSTEMD_USER") == "1",
                     "requires an explicitly enabled Linux user systemd manager")
class SystemdTrainingSupervisorTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("LFG_TEST_INSTALLED_TRAINING_TEMPLATE") == "1",
                         "requires an explicitly installed matching service template")
    def test_installed_template_runs_real_trainer_gate_with_explicit_limits(self):
        source = Path(__file__).resolve().parent
        unit_file = Path.home() / ".config/systemd/user/lfg-reviewed-training@.service"
        self.assertEqual(unit_file.read_bytes(), (source / "systemd/lfg-reviewed-training@.service").read_bytes())
        instance = "proof-" + uuid.uuid4().hex
        unit = f"lfg-reviewed-training@{instance}.service"
        env_dir = Path.home() / ".config/lfg-training"
        env_file = env_dir / f"{instance}.env"
        dropin = unit_file.parent / f"{unit}.d"
        limits = dropin / "limits.conf"

        def systemctl(*args, check=True):
            return subprocess.run(["systemctl", "--user", *args], text=True, capture_output=True, check=check)

        with tempfile.TemporaryDirectory(prefix="lfg installed service test ") as temporary:
            root = Path(temporary)
            for name in ("train.py", "training_supervisor.py", "promotion-policy.json",
                         "training-config.v1.json", "training-support-policy.v1.json"):
                shutil.copy2(source / name, root / name)
            dataset = root / "empty.jsonl"
            dataset.write_bytes(b"")
            output = root / "run"
            arguments = ["--dataset", str(dataset), "--output", str(output),
                         "--audited-plan", str(root / "plan"), "--base-model", str(root / "model"),
                         "--model-revision", "a" * 40, "--model-manifest", str(root / "manifest"),
                         "--environment-lock", str(root / "lock"), "--wheelhouse", str(root / "wheels")]
            config = {"schema_version": "lfg.reviewed-training-service.v1", "python": str(Path(sys.executable).resolve()),
                      "trainer_sha256": hashlib.sha256((root / "train.py").read_bytes()).hexdigest(),
                      "trainer_arguments": arguments, "recovery": "manual",
                      "resource_limits": {"cpu_quota_percent": 50, "memory_max_bytes": 134217728, "tasks_max": 32}}
            config_path = root / "service.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            env_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Exclusively created names prevent this test from replacing an existing instance.
            dropin.mkdir(mode=0o700)
            env_created = False
            unit_started = False
            try:
                def quoted(path):
                    return '"' + str(path).replace('\\', '\\\\').replace('"', '\\"') + '"'
                with env_file.open("x", encoding="utf-8") as stream:
                    env_created = True
                    stream.write("LFG_TRAIN_SUPERVISOR=" + quoted(root / "training_supervisor.py") + "\n")
                    stream.write("LFG_TRAIN_CONFIG=" + quoted(config_path) + "\n")
                env_file.chmod(0o600)
                with limits.open("x", encoding="utf-8") as stream:
                    stream.write("[Service]\nCPUQuota=50%\nMemoryMax=128M\nTasksMax=32\n")
                systemctl("daemon-reload")
                unit_started = True
                systemctl("start", unit, check=False)
                for _ in range(40):
                    state = systemctl("show", unit, "-p", "ActiveState", "-p", "Result", "-p", "ExecMainStatus", "-p", "NRestarts")
                    values = dict(line.split("=", 1) for line in state.stdout.splitlines())
                    if values.get("ActiveState") == "failed":
                        break
                    time.sleep(0.25)
                else:
                    self.fail("installed service fixture did not stop")
                journal = subprocess.run(["journalctl", "--user", "-u", unit, "--no-pager"],
                                         text=True, capture_output=True, check=True).stdout
                self.assertEqual(values["Result"], "exit-code")
                self.assertEqual(values["ExecMainStatus"], "2")
                self.assertEqual(values["NRestarts"], "0")
                minimum = json.loads((root / "promotion-policy.json").read_bytes())["minimum_human_verified_examples"]
                self.assertIn(f"0 verified examples; {minimum} required", journal)
                self.assertFalse(output.exists())
            finally:
                if unit_started:
                    systemctl("stop", unit, check=False)
                    systemctl("reset-failed", unit, check=False)
                if env_created:
                    env_file.unlink(missing_ok=True)
                limits.unlink(missing_ok=True)
                dropin.rmdir()
                systemctl("daemon-reload", check=False)

    def test_killed_child_resumes_once_and_completed_run_is_idempotent(self):
        source = Path(__file__).resolve().parent
        unit = f"lfg-supervisor-test-{uuid.uuid4().hex}.service"
        with tempfile.TemporaryDirectory(prefix="lfg supervisor test ") as temporary:
            root = Path(temporary)
            shutil.copy2(source / "training_supervisor.py", root / "training_supervisor.py")
            shutil.copy2(source / "train.py", root / "real_train.py")
            fixture = root / "train.py"
            fixture.write_text('''import json, os, signal, sys
from pathlib import Path
from real_train import build_parser, validate_input_options, validate_checkpoint, training_run_lock
if __name__ == "__main__":
    args = build_parser().parse_args()
    output = Path(args.output)
    with training_run_lock(output):
        output.mkdir(exist_ok=True)
        with (output.parent / "calls").open("a") as calls:
            calls.write("resume\\n" if args.resume else "start\\n")
            calls.flush()
            os.fsync(calls.fileno())
        if not args.resume:
            (output / "run-manifest.json").write_text("{}")
            checkpoint = output / "checkpoint-1"
            checkpoint.mkdir()
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
                (checkpoint / name).write_bytes(b"test fixture")
            (checkpoint / "trainer_state.json").write_text('{"global_step": 1}')
            os.kill(os.getpid(), signal.SIGKILL)
        for name in ("trainer_state.json", "labels.json", "model.safetensors"):
            (output / name).write_bytes(b"test fixture")
''', encoding="utf-8")
            output = root / "run"
            arguments = ["--dataset", str(root / "dataset"), "--output", str(output),
                         "--audited-plan", str(root / "plan"), "--base-model", str(root / "model"),
                         "--model-revision", "a" * 40, "--model-manifest", str(root / "manifest"),
                         "--environment-lock", str(root / "lock"), "--wheelhouse", str(root / "wheels")]
            config = {"schema_version": "lfg.reviewed-training-service.v1", "python": str(Path(sys.executable).resolve()),
                      "trainer_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(), "trainer_arguments": arguments,
                      "recovery": "resume", "resource_limits": {"cpu_quota_percent": 50,
                          "memory_max_bytes": 134217728, "tasks_max": 32}}
            config_path = root / "service.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            def systemctl(*args):
                return subprocess.run(["systemctl", "--user", *args], text=True, capture_output=True, check=True)

            def wait_for_success():
                for _ in range(60):
                    result = systemctl("show", unit, "-p", "ActiveState", "-p", "SubState", "-p", "Result", "-p", "NRestarts")
                    state = dict(line.split("=", 1) for line in result.stdout.splitlines())
                    if state.get("ActiveState") == "active" and state.get("SubState") == "exited":
                        return state
                    if state.get("ActiveState") == "failed":
                        self.fail(subprocess.run(["journalctl", "--user", "-u", unit, "--no-pager"],
                                                 text=True, capture_output=True).stdout)
                    time.sleep(0.5)
                self.fail("systemd fixture did not finish")

            try:
                subprocess.run(["systemd-run", "--user", "--unit=" + unit, "--remain-after-exit",
                                "--property=Restart=on-failure", "--property=RestartSec=1s",
                                "--property=RestartPreventExitStatus=2", "--property=CPUQuota=50%",
                                "--property=MemoryMax=128M", "--property=TasksMax=32",
                                "--property=RuntimeMaxSec=30s", "--property=NoNewPrivileges=yes",
                                sys.executable, str(root / "training_supervisor.py"), "--config", str(config_path)],
                               check=True, capture_output=True, text=True)
                state = wait_for_success()
                self.assertEqual(state["Result"], "success")
                self.assertGreaterEqual(int(state["NRestarts"]), 1)
                self.assertEqual((root / "calls").read_text().splitlines(), ["start", "resume"])
                marker = (output / "supervised-complete.json").read_bytes()
                systemctl("restart", unit)
                wait_for_success()
                self.assertEqual((root / "calls").read_text().splitlines(), ["start", "resume"])
                self.assertEqual((output / "supervised-complete.json").read_bytes(), marker)
            finally:
                subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)


if __name__ == "__main__":
    unittest.main()
