# Supervised reviewed training on Linux

The repository includes a configurable systemd user-service template and
`ml/lfg-encoder/training_supervisor.py`. They run one existing reviewed trainer,
enforce finite CPU, memory and task limits, and recover only from a complete
checkpoint. The template is opt-in: installing it does not start training.
It is not a trained model, an evaluation, or a production-qualified service.

The launcher rejects `--allow-below-gate` and chooses `--resume` itself. The
trainer still requires a reviewed corpus, audit, immutable model revision,
model manifest and locked environment, and applies its existing eligibility
checks. Use a [development-only corpus plan](CORPUS-REVIEW.md#development-only-assembly)
and keep the final holdout separate: the trainer also accepts legacy plans that
read test data. A training completion marker is not a quality or release gate.

## Configure an instance

Use a trusted Linux host with cgroup v2 and a working systemd user manager, a
durable local output filesystem, an approved GPU training environment and the
exact source files for the experiment. The launcher uses Python 3.9 or newer;
the configured trainer interpreter must also satisfy the trainer's locked
environment. Refer to [the Spark environment](SPARK-LOCKED-ENVIRONMENT.md)
and [corpus review](CORPUS-REVIEW.md). Do not put credentials in configuration
or logs. Systemd's inherited environment is not fully attested by the launcher.

1. Copy `ml/lfg-encoder/systemd/training-service.example.json` into a private
   directory. Replace every placeholder and every zero resource limit. Set
   `python` to the absolute trainer environment interpreter, and set
   `trainer_sha256` to `sha256sum` of the adjacent `train.py`. All paths must be
   absolute; the model revision must be its full 40-character commit ID. Keep
   the exact supervisor, trainer, policies and experiment inputs stable for the
   lifetime of the run. Choose a fresh output directory. The example's CUDA
   device and `pip` allowance are examples to review, not universal defaults.
2. Place the service template at
   `~/.config/systemd/user/lfg-reviewed-training@.service`. Check for an
   existing file before copying; don't overwrite another installation. Create
   `~/.config/lfg-training/INSTANCE.env`, with private permissions, containing
   the absolute paths below. Quote values with spaces using systemd
   EnvironmentFile syntax. Use a simple instance name such as `project1`.

   ```ini
   LFG_TRAIN_SUPERVISOR="/absolute/path/to/ml/lfg-encoder/training_supervisor.py"
   LFG_TRAIN_CONFIG="/absolute/path/to/private/service.json"
   ```

3. Create `~/.config/systemd/user/lfg-reviewed-training@INSTANCE.service.d/limits.conf`.
   Set actual values for the host and workload; this is a template, not an
   operational configuration:

   ```ini
   [Service]
   CPUQuota=CHOOSE_PERCENT%
   MemoryMax=CHOOSE_BYTES
   TasksMax=CHOOSE_COUNT
   ```

   Set each value no higher than its corresponding `resource_limits` value in
   the JSON. The launcher checks its own cgroup's finite `cpu.max`, `memory.max`
   and `pids.max` before creating training output; ancestor-only limits do not
   satisfy that check. System memory limits do not establish a GPU VRAM limit or
   isolation from other GPU workloads. Set GPU allocation and scheduling policy
   separately. Tune restart and timeout settings through a drop-in if needed.
4. After the corpus and inputs pass their checks, run `systemctl --user
   daemon-reload`, then `systemctl --user enable --now
   lfg-reviewed-training@INSTANCE.service`. Check `systemctl --user status
   lfg-reviewed-training@INSTANCE.service` and `journalctl --user -u
   lfg-reviewed-training@INSTANCE.service`. To stop it, use `systemctl --user
   stop lfg-reviewed-training@INSTANCE.service`. Persistence across logout or
   reboot also depends on the host's user-manager and linger configuration.

## Recovery and evidence

The default service restarts signal-terminated runs, subject to systemd's
restart limit. The launcher exits with status 2 for a permanent trainer error or
configuration failure, which prevents automatic restart. With `recovery` set
to `resume`, it requires a run manifest and the numerically latest complete
checkpoint; the trainer independently checks whether the checkpoint and
manifest match its inputs. `manual` instead stops when output already exists.
Before the first checkpoint, after an incomplete latest checkpoint, or when
finished-looking output lacks a completion marker, inspect the run and choose a
new output or a deliberate manual recovery. Do not delete a lock while a
trainer may hold it. Network filesystem lock semantics are not qualified.

After a successful child exit, the supervisor records hashes of its config,
code, trainer and root output files. It refuses to rerun an intact completed
output and rejects drift. This is an integrity check for trusted local files,
not adversarial authentication, scientific validation or model qualification.
If the process dies after training finishes but before writing the marker,
the result is ambiguous and requires inspection. The GPU training and reboot
paths must be validated for the actual reviewed run and machine.

On 2026-10-31, the Spark ran the real launcher under a bounded user service
against an empty test input: the trainer rejected zero verified records against
the 500-record gate, and systemd did not retry the permanent failure. A second
service with no resource limits was rejected before a trainer launched. A
separate systemd test used a **test-only fixture**, killed its first process,
resumed from its fixture checkpoint, and verified completion was not repeated.
Those checks transferred no project data and did not train a model. They do not
establish actual model checkpoint recovery, reboot persistence or model quality.

The template was then installed, disabled, on `Clyffy-Ring` alongside a
versioned copy of the merged service source. A temporary instance of that
installed template used an EnvironmentFile and per-instance limits drop-in to
launch the real trainer with an empty dataset. The configuration path contained
spaces. The trainer stopped at the reviewed-data gate, the unit exited with
status 2 and zero restarts, and no training output was created. The temporary
instance and its drop-in were removed after the check. No production training
instance was enabled. Rerun the installed-template and fixture-recovery tests
from matching source on a suitable Linux user manager with:

```bash
LFG_TEST_SYSTEMD_USER=1 LFG_TEST_INSTALLED_TRAINING_TEMPLATE=1 \
  python3 -m unittest discover -s ml/lfg-encoder -p test_systemd_training_supervisor.py -v
```

The installed-template test requires the installed template bytes to match the
source under test. It creates a unique temporary instance and configuration,
starts and stops only that instance, and reloads the user manager. Its limits
and empty input are test settings, not recommended production values.

A separate Windows check passed the real local 23-record LFG/DevPulse
development dataset through the actual trainer subprocess. All 23 records
were unverified, so the 500-record gate stopped it before output creation.
This check mocked the Linux cgroup verification because it ran on Windows;
its other model and environment paths were unused placeholders. It did not
train a model or establish a reviewed corpus.
