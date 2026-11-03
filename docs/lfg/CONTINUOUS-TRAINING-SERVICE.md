# Private Spark training service

The owned-model supervisor runs one explicitly configured, finite distillation
or assessment fine-tuning job in checkpointed segments. The model, teacher,
corpus, device and training settings come from private configuration. Extra service-configuration fields
are accepted but do not acquire behavior unless the service implements them.
The service selects the highest published checkpoint step by directory name, verifies its metadata and
tensor digests and passes its state digest to the trainer, which
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

`owned_training_chain.py` can wait for an explicitly configured owned
pretraining job, verify its completed manifest and weight digest, and launch
the configured assessment fine-tune under the same service supervisor. Its
[configuration](../../ml/lfg-encoder/owned-training-chain.example.json)
specifies the existing dataset and all paths and limits. It does not fetch new
teacher batches or select models automatically.
`owned_evaluation_chain.py` can similarly wait for that fine-tune, verify its
result and run one configured private development-format evaluation. A
matching completed report is reused; a changed checkpoint, dataset, policy or
evaluation configuration is rejected. Format results do not measure whether
the assessment is correct.

`owned_training_dashboard.py` serves sanitized training progress and optional
aggregate teacher and format-evaluation status, and binds to loopback. If shared through a proxy, configure an
appropriate authenticated private network and TLS; the dashboard itself has
no authentication or access control. Its status is not assessment quality,
and it does not control training, operate a queue, or expose examples.
The lab deployment routes `/lfg-training/` through the existing tailnet-only
HTTPS proxy to this dashboard. Tailnet access controls that route; it does not
inherit the Clyffy application's login. The public site and Clyffy application
do not yet embed the dashboard or control training.

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

`stage_owned_source_corpus.py` selects tracked files from explicitly configured
project roots and path/extension rules, screens their contents, and writes a
private train-only document plan plus source-path and content-digest
provenance. It checks source bytes again before finalizing. The recorded Git
HEAD is a claim about the checkout, not proof that working-tree bytes matched
that commit. Oversized, duplicate, unavailable, short and failed-screening
files are counted; screening can miss secrets. Configure limits and source
selection for each run and keep the resulting corpus outside Git.

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
requests, excluding the first 21. One request exhausted even a 4,096-token
response limit without a complete JSON assessment. `split_teacher_collection.py`
preserved the 112 preceding records in a separately identified selection; the
collector revalidated them before finalizing that prefix. The incomplete
request remains recorded for separate follow-up, and the remaining 87 requests
were assigned to another collection run. That run encountered a second
incomplete response after 34 more records; those records were revalidated and
the request was recorded separately. The 52 remaining requests use a new
explicit policy that records invalid responses and continues; HTTP server
errors still trigger a supervised retry.

The 112 validated records plus the first 21 were prepared as 133 unreviewed,
unknown-outcome training examples. A 2,000-step fine-tune from the owned byte
checkpoint produced 928,565 supervised byte positions. On the same 10-request
development format check, nine outputs were invalid JSON and one was JSON
missing required fields. This is not a task-accuracy result or evidence of a
usable release. The split requires a contiguous prefix and intentionally does
not infer or invent a response for an excluded request.

The remaining 52-request collection completed with no additional failures.
Combining its records with the prior verified prefixes produced 219 unreviewed
training examples. A 3,000-step assessment fine-tune completed with 1,384,398
supervised byte positions. On the same 10 development requests, all outputs
were invalid JSON with a 512-byte output bound. A second run allowing 2,048
new bytes also produced 10 invalid JSON outputs: six stopped at a newline,
three reached the output bound and one reached the model's context limit.
This remains a format check rather than assessment-quality evaluation.
Another 131 eligible requests were selected after excluding prior IDs and
auditing exact normalized text overlap, then queued on the local teacher.
Those 131 come mainly from two declared projects, so they do not establish
broad source independence or sufficient training diversity.

That last batch produced 130 valid proposals and one incomplete response;
the three incomplete requests across both batches remain recorded. Combining
the valid proposals yielded 349 train examples. A 5,000-step owned assessment
fine-tune completed after the context was configured to 4,096 bytes to fit the
full selected examples. Its PyTorch format check on the same 10 development
requests, with a 2,048-byte generation bound, returned one JSON object
containing the required fields and nine invalid JSON outputs. All stopped at
a newline. Field correctness, types, assessment quality and native-binary
behavior were not established.

An expanded local training selection added 604 tracked Rust files from the
owner's RRFlow, DevPULSE and LFG checkouts. It contained 11,512,022 screened
bytes; two oversized and two short files were skipped. Combined with the
previous authored material, the resulting train-only plan has 619 documents
and 12,001,044 screened bytes. Pattern screening reported redactions and is
not proof that all secrets were removed. A separately configured 100,000-step
run of a 19,264,512-parameter owned byte model was started from random weights
on `clyffy-03`, using the existing small byte-aligned teacher. This is a new
language-model pretraining experiment, not the previously fine-tuned 349-case
assessment checkpoint. The run completed all 100,000 steps and wrote final
weights. Its assessment quality has not been established, and neither model
is the intended 650 MB release.
The expanded run resumed from a completed step-1,500 checkpoint after an
intentional service restart and then continued training. This does not prove
recovery from power loss or all storage failures.

Two additional user services chain from that run: the first verified the
completed byte-model weights and completed assessment fine-tuning using the
prepared 349-example dataset. The second ran the existing 10-request private
development-format check: two responses were JSON objects containing the
required fields and eight were invalid JSON. Seven responses hit the byte
limit and three hit a stop sequence. This reused development set does not
establish independent generalization, field correctness, recommendation accuracy
or native-binary behavior. Their control flow
was also exercised with completed-checkpoint software fixtures on the Spark.
The dashboard switches to the assessment job when its status appears and
displays the resulting aggregate format count when the report is available.

Each teacher collection is a finite, resumable job. The configured
[continual data handoff](CONTINUAL-DATA-HANDOFF.md) publishes completed
collections and queues new datasets; the live first update on the 369-record
dataset completed after the initial assessment model. The queue returned to
idle; its separate evaluation found four of ten development responses contained
the required fields, five were invalid JSON and one missed required fields.
The completed
earlier assessment services used
the prior trainer implementation; the final-checkpoint recovery improvement
applies to new jobs, not migrations of their pinned checkpoints.
