# Private Spark training service

The owned-model supervisor runs one explicitly configured, finite distillation
or assessment fine-tuning job in checkpointed segments. The model, teacher,
corpus, device and training settings come from private configuration. Extra service-configuration fields
are accepted but do not acquire behavior unless the service implements them.
The service verifies all published checkpoint tensor digests, selects the
highest valid completed step and passes its state digest to the trainer, which
checks its full continuation identity. An OS lock prevents two supervisors
from owning the same state directory. A completed run remains available for
status inspection; it does not automatically replay its corpus. Use a new
state directory for a new job or changed service/trainer configuration.

The [service configuration](../../ml/lfg-encoder/owned-training-service.example.json)
and [systemd unit](../../ml/lfg-encoder/systemd/owned-training.service.example)
are examples to configure for the host, not recommended fixed limits. Keep the
data, configs, checkpoints, outputs and logs private and outside Git. Select
an authorized teacher and review its terms and byte mapping. Screening can
miss secrets. Run only the training split; reserve independent evaluation data
before generating labels or selecting models. CPU and memory cgroups constrain
the process; the PyTorch CUDA allocator fraction is not GPU isolation and does
not reserve GPU capacity. Disk retention is not automatic: every checkpoint
is retained, so provision and monitor disk space.

If the last checkpoint reaches the configured final step but the trainer dies
before publishing the final model manifest, the supervisor resumes that
checkpoint and finalizes the result without taking another optimizer step.
A crash can also leave temporary checkpoint directories, which are not selected
for recovery. Checkpoints bind trainer implementation hashes: updating trainer
code does not silently migrate older unfinished jobs.

To supervise assessment fine-tuning, use the separate
[assessment service configuration](../../ml/lfg-encoder/owned-assessment-service.example.json).
It accepts an owned starting checkpoint and a prepared private assessment
dataset; it does not turn teacher claims into verified outcomes.

`owned_training_dashboard.py` serves only sanitized progress from a single
status file and binds to loopback. If shared through a proxy, configure an
appropriate authenticated private network and TLS; the dashboard itself has
no authentication or access control. Its status is not assessment quality,
and it does not control training, operate a queue, or expose examples.

`local_teacher_service.py` starts the local teacher collector and chooses its
resume mode when an output already exists. The collector keeps incomplete
records and resumes only after verifying their identities. Each service handles
one finite selected batch; creating new batches and transferring completed
traces to an SFT trainer are not automated. Teacher assessments are unreviewed
proposals with unknown outcomes, not verified labels. Per-request options such
as model reasoning mode can be supplied through `request_extensions` in the
collector configuration; fields controlling the selected model, source
messages, response format, streaming and token limit cannot be overridden.
The selector can accept an `exclude_ids` list to avoid collecting the same
requests in a later batch; the exclusion list is part of the hashed selection
configuration.

## 2026-11-02 local cluster evidence

On the owner's `clyffy-03` GB10, an isolated service completed 10,000
distillation steps from random initialization for a 3,726,336-parameter owned
student. Its screened training input contained 15 documents and 489,022 bytes,
including recovered authored notes and 598 training-side chat requests. The
byte-aligned teacher was a separate 1,927,936-parameter autocomplete model;
it is not LFG. A deliberate service restart after step 1,000 resumed training
and advanced beyond that step. This is one restart test, not proof of recovery
from every hardware or filesystem failure.

On the lab host, the existing local Nemotron Lightning service generated 21
assessment proposals for screened real training requests from 11 declared
project labels. They remain unreviewed, with unknown outcomes. The collector
initially rejected incomplete reasoning-only responses; configuring a direct
answer through the model's request-level chat-template option produced complete
records. NVIDIA documents this option in its [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4).
The running local snapshot reports OpenMDW 1.1; that [license](https://raw.githubusercontent.com/OpenMDW/OpenMDW/refs/heads/main/1.1/LICENSE.OpenMDW-1.1)
does not impose output-use restrictions, but the local weight bytes were not
independently pinned or verified.

Those 21 proposals were prepared and used to fine-tune the owned checkpoint
for 1,000 steps. The resulting student has 4,480,000 parameters and 485,028
supervised byte positions; extending its context reused 128 position rows and
randomly initialized 2,944 more. A separate PyTorch GPU format check sampled
10 of 47 screened development validation requests and produced invalid JSON
for all 10; eight outputs stopped at a newline and two hit the 512-byte limit.
This checks output format only, not assessment accuracy or native-binary
performance. The validation records share declared provenance and their
independence is not proven. No final held-out test was opened. The model is
not a usable, qualified LFG release.

A second local-teacher batch was selected from 200 additional training
requests, excluding the first 21. It is a finite, resumable collection job;
preparing its outputs for training and triggering the next fine-tune are not
yet automated. The installed, already completed Spark services used the prior
trainer implementation; the final-checkpoint recovery improvement in this
change applies to new jobs, not migrations of their pinned checkpoints.
