import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import training_supervisor as supervisor


class TrainingSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "run"
        self.config_path = self.root / "service.json"
        self.config = {
            "schema_version": "lfg.reviewed-training-service.v1",
            "python": str(Path(sys.executable).resolve()),
            "trainer_sha256": supervisor.hash_file(supervisor.TRAINER),
            "recovery": "resume",
            "resource_limits": {"cpu_quota_percent": 50, "memory_max_bytes": 134217728, "tasks_max": 32},
            "trainer_arguments": ["--dataset", str(self.root / "train.jsonl"),
                                  "--output", str(self.output),
                                  "--audited-plan", str(self.root / "plan.json"),
                                  "--base-model", str(self.root / "model"),
                                  "--model-revision", "a" * 40,
                                  "--model-manifest", str(self.root / "model.json"),
                                  "--environment-lock", str(self.root / "lock.txt"),
                                  "--wheelhouse", str(self.root / "wheels")],
        }
        self.write_config()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def checkpoint(self, index, complete=True):
        path = self.output / f"checkpoint-{index}"
        path.mkdir(parents=True)
        if complete:
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
                (path / name).write_bytes(b"fixture")
            (path / "trainer_state.json").write_text('{"global_step": 1}', encoding="utf-8")
        return path

    def finished_output(self):
        self.output.mkdir(exist_ok=True)
        for name in ("run-manifest.json", "trainer_state.json", "labels.json", "model.safetensors"):
            (self.output / name).write_bytes(b"test-only-output")

    def test_config_rejects_ambiguous_resume_and_unpinned_inputs(self):
        supervisor.load_config(self.config_path)
        for extra in (["--allow-below"], ["--resume"], ["--unknown"]):
            with self.subTest(extra=extra):
                self.config["trainer_arguments"] += extra
                self.write_config()
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(ValueError):
                    supervisor.load_config(self.config_path)
                del self.config["trainer_arguments"][-len(extra):]
        self.config["trainer_sha256"] = "0" * 64
        self.write_config()
        with self.assertRaisesRegex(ValueError, "trainer digest"):
            supervisor.load_config(self.config_path)

    def test_authorized_mode_does_not_require_reviewed_plan_or_gate(self):
        self.config["training_mode"] = "authorized"
        self.config["extra_metadata"] = {"owner": "fixture"}
        self.config["resource_limits"]["extra_metadata"] = "fixture"
        index = self.config["trainer_arguments"].index("--audited-plan")
        del self.config["trainer_arguments"][index:index + 2]
        self.write_config()
        supervisor.load_config(self.config_path)
        child = SimpleNamespace(wait=lambda: -9, poll=lambda: -9)
        with patch.object(supervisor, "verify_cgroup_limits", return_value={}), patch.object(subprocess, "Popen", return_value=child) as launch:
            self.assertEqual(supervisor.execute(self.config_path), 1)
        command = launch.call_args.args[0]
        self.assertIn("--allow-below-gate", command)
        self.assertIn("--verify-installed-payload", command)
        self.assertNotIn("--require-reviewed", command)

    def test_reviewed_mode_remains_explicit_and_requires_audited_plan(self):
        child = SimpleNamespace(wait=lambda: -9, poll=lambda: -9)
        with patch.object(supervisor, "verify_cgroup_limits", return_value={}), patch.object(subprocess, "Popen", return_value=child) as launch:
            self.assertEqual(supervisor.execute(self.config_path), 1)
        self.assertIn("--require-reviewed", launch.call_args.args[0])
        index = self.config["trainer_arguments"].index("--audited-plan")
        del self.config["trainer_arguments"][index:index + 2]
        self.write_config()
        with self.assertRaisesRegex(ValueError, "audited plan"):
            supervisor.load_config(self.config_path)

    def test_conflicting_modes_are_rejected(self):
        self.config["training_mode"] = "reviewed"
        self.config["trainer_arguments"].append("--allow-below-gate")
        self.write_config()
        with self.assertRaisesRegex(ValueError, "conflicts"):
            supervisor.load_config(self.config_path)
        self.config["training_mode"] = "authorized"
        self.config["trainer_arguments"][-1] = "--require-reviewed"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "conflicts"):
            supervisor.load_config(self.config_path)

    def test_start_resume_and_incomplete_checkpoints_fail_closed(self):
        digest = "b" * 64
        trainer_digest = self.config["trainer_sha256"]
        supervisor_digest = supervisor.hash_file(Path(supervisor.__file__))
        self.assertEqual(supervisor.choose_mode(self.output, "resume", digest, trainer_digest, supervisor_digest), "start")
        self.checkpoint(1)
        (self.output / "run-manifest.json").write_text("{}", encoding="utf-8")
        self.assertEqual(supervisor.choose_mode(self.output, "resume", digest, trainer_digest, supervisor_digest), "resume")
        with self.assertRaisesRegex(ValueError, "explicit recovery"):
            supervisor.choose_mode(self.output, "manual", digest, trainer_digest, supervisor_digest)
        self.checkpoint(2, complete=False)
        with self.assertRaisesRegex(ValueError, "incomplete checkpoint"):
            supervisor.choose_mode(self.output, "resume", digest, trainer_digest, supervisor_digest)
        (self.output / "trainer_state.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ambiguous output"):
            supervisor.choose_mode(self.output, "resume", digest, trainer_digest, supervisor_digest)

    def test_success_records_output_identity_and_refuses_drift(self):
        def run_fixture():
            self.finished_output()
            return 0
        child = SimpleNamespace(wait=run_fixture, poll=lambda: 0)
        with patch.object(supervisor, "verify_cgroup_limits", return_value={}), patch.object(subprocess, "Popen", return_value=child):
            self.assertEqual(supervisor.execute(self.config_path), 0)
        marker = json.loads((self.output / supervisor.COMPLETE).read_bytes())
        self.assertEqual(marker["schema_version"], "lfg.supervised-training-complete.v1")
        with patch.object(supervisor, "verify_cgroup_limits", return_value={}):
            self.assertEqual(supervisor.execute(self.config_path), 0)
        (self.output / "labels.json").write_text("changed", encoding="utf-8")
        with patch.object(supervisor, "verify_cgroup_limits", return_value={}), self.assertRaisesRegex(ValueError, "changed"):
            supervisor.execute(self.config_path)

    def test_child_error_is_permanent_and_signal_is_restartable(self):
        for status, expected in ((1, 2), (-9, 1)):
            with self.subTest(status=status):
                child = SimpleNamespace(wait=lambda: status, poll=lambda: status)
                with patch.object(supervisor, "verify_cgroup_limits", return_value={}), patch.object(subprocess, "Popen", return_value=child):
                    self.assertEqual(supervisor.execute(self.config_path), expected)
                self.assertFalse((self.output / supervisor.COMPLETE).exists())

    def test_resume_passes_flag_to_trainer_and_signal_does_not_mark_complete(self):
        self.checkpoint(1)
        (self.output / "run-manifest.json").write_text("{}", encoding="utf-8")
        child = SimpleNamespace(wait=lambda: -9, poll=lambda: -9)
        with patch.object(supervisor, "verify_cgroup_limits", return_value={}), patch.object(subprocess, "Popen", return_value=child) as launch:
            self.assertEqual(supervisor.execute(self.config_path), 1)
        self.assertIn("--development-plan-only", launch.call_args.args[0])
        self.assertEqual(launch.call_args.args[0][-1], "--resume")
        self.assertFalse((self.output / supervisor.COMPLETE).exists())

    def test_sharded_output_requires_every_named_weight_file(self):
        self.finished_output()
        (self.output / "model.safetensors").unlink()
        index = self.output / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"weight": "model-01.safetensors"}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing or invalid weight shards"):
            supervisor.output_files(self.output)
        (self.output / "model-01.safetensors").write_bytes(b"fixture")
        supervisor.output_files(self.output)
        index.write_text(json.dumps({"weight_map": {"weight": "../outside"}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing or invalid weight shards"):
            supervisor.output_files(self.output)

    def test_limits_must_be_finite_and_within_configured_maxima(self):
        cgroup = self.root / "cgroup"
        cgroup.mkdir()
        limits = self.config["resource_limits"]
        (cgroup / "cpu.max").write_text("50000 100000", encoding="utf-8")
        (cgroup / "memory.max").write_text("134217728", encoding="utf-8")
        (cgroup / "pids.max").write_text("32", encoding="utf-8")
        self.assertEqual(supervisor.verify_cgroup_limits(limits, cgroup)["cpu_quota_us"], 50000)
        (cgroup / "cpu.max").write_text("max 100000", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "finite"):
            supervisor.verify_cgroup_limits(limits, cgroup)
        (cgroup / "cpu.max").write_text("60000 100000", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exceed"):
            supervisor.verify_cgroup_limits(limits, cgroup)


if __name__ == "__main__":
    unittest.main()
