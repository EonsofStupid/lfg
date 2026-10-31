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
