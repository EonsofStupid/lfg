"""Opt-in service-manager test; it does not start a model or use project data."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

import train


@unittest.skipUnless(os.environ.get("LFG_TEST_SYSTEMD_USER") == "1", "systemd user test not selected")
class SystemdTrainingLockTests(unittest.TestCase):
    def test_service_restart_reacquires_lock_after_sigkill(self):
        if sys.platform != "linux":
            self.skipTest("requires Linux systemd user manager")
        unit = "lfg-lock-test-" + uuid.uuid4().hex + ".service"
        with tempfile.TemporaryDirectory(prefix="lfg-lock-test-") as directory:
            root = Path(directory)
            fixture = root / "fixture.py"
            fixture.write_text(
                "import os,signal,sys\nfrom pathlib import Path\n"
                "sys.path.insert(0,sys.argv[1])\nimport train\nroot=Path(sys.argv[2])\n"
                "with train.training_run_lock(root/'run'):\n"
                "    first=root/'first'\n"
                "    if not first.exists():\n"
                "        with first.open('x') as f:\n"
                "            f.write('first'); f.flush(); os.fsync(f.fileno())\n"
                "        os.kill(os.getpid(),signal.SIGKILL)\n"
                "    (root/'recovered').write_text('recovered')\n", encoding="utf-8")
            command = ["systemd-run", "--user", "--unit=" + unit, "--remain-after-exit",
                       "--property=Restart=on-failure", "--property=RestartSec=1s",
                       "--property=CPUQuota=50%", "--property=MemoryMax=128M",
                       "--property=TasksMax=32", "--property=RuntimeMaxSec=30s",
                       "--property=NoNewPrivileges=yes", sys.executable, str(fixture),
                       str(Path(train.__file__).resolve().parent), str(root)]
            attempted = False
            try:
                attempted = True
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
                deadline = time.monotonic() + 30
                state = {}
                while time.monotonic() < deadline:
                    result = subprocess.run(["systemctl", "--user", "show", unit,
                                             "-p", "ActiveState", "-p", "SubState", "-p", "Result",
                                             "-p", "NRestarts", "-p", "CPUQuotaPerSecUSec",
                                             "-p", "MemoryMax", "-p", "TasksMax", "-p", "NoNewPrivileges"],
                                            check=True, capture_output=True, text=True, timeout=5)
                    state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
                    if state.get("SubState") == "exited" and (root / "recovered").exists():
                        break
                    if state.get("ActiveState") == "failed":
                        self.fail(f"test unit failed: {state}")
                    time.sleep(0.2)
                self.assertEqual(state.get("Result"), "success", state)
                self.assertEqual(state.get("SubState"), "exited", state)
                self.assertGreaterEqual(int(state.get("NRestarts", "0")), 1)
                self.assertEqual(state.get("CPUQuotaPerSecUSec"), "500ms")
                self.assertEqual(state.get("MemoryMax"), str(128 * 1024 * 1024))
                self.assertEqual(state.get("TasksMax"), "32")
                self.assertEqual(state.get("NoNewPrivileges"), "yes")
                self.assertEqual((root / "run.lock").read_bytes(), train.RUN_LOCK_MAGIC)
            finally:
                if attempted:
                    subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True, timeout=15, check=False)


if __name__ == "__main__":
    unittest.main()
