# Continual private teacher-data handoff

The lab publisher combines completed local teacher collections into a
train-only assessment dataset. Collector configurations must be explicitly
approved and use the configured assessment instruction; held-out and
validation collections are rejected. The existing assessment builder checks
source digests and rejects duplicate IDs and normalized requests. Output
claims remain unreviewed and outcomes unknown unless the source says
otherwise; hashes do not verify those claims or prove complete redaction.

The [publisher configuration](../../ml/lfg-encoder/teacher-dataset-publisher.example.json)
defines source paths, approved collector configs, limits, dataset plan and an
approved HTTPS destination. It uses TLS certificate validation, rejects
redirects, disables environment proxies and reads its bearer token from a
private file. An unchanged dataset is not repeatedly sent within the configured
resend interval. New collections or changes to the plan cause a new build.
The example [publisher service](../../ml/lfg-encoder/systemd/teacher-dataset-publisher.service.example)
and [timer](../../ml/lfg-encoder/systemd/teacher-dataset-publisher.timer.example)
show the lifecycle; set the executable paths, schedule and limits for the host.

`teacher_dataset_receiver_server.py` binds only to loopback and accepts a
bounded, token-authenticated dataset submission. The receiver rejects a
manifest declaring test data or anything except a train file, verifies the
train-file digest and unique record IDs, and atomically publishes the inbox
directory. Repeated delivery of identical bytes is idempotent. The deployment
must provide private TLS and network access controls; the server itself does
not supply TLS. The token and all datasets, manifests, logs and derived model
weights must remain private and outside Git.
An example [receiver service](../../ml/lfg-encoder/systemd/teacher-dataset-receiver.service.example)
shows the loopback binding and configurable size limits.

The [dataset queue](../../ml/lfg-encoder/owned-dataset-queue.example.json)
waits for the configured initial owned assessment checkpoint, confirms that
checkpoint identifies the configured baseline training dataset, and then
looks for delivered datasets that include its existing records plus new ones.
It rejects unapproved revisions and competing branches. An explicit setting can
permit revised records; removing existing records is not automatically
supported. Each update fine-tunes the previous owned checkpoint under the
existing checkpointed supervisor. State and deterministic job identities
allow a completed update to be recovered after a restart without replaying it.
An optional policy sizes context from the actual full examples, up to
configured context and attention limits. Context growth initializes new
positional rows and is not evidence of long-context understanding.
The example [queue service](../../ml/lfg-encoder/systemd/owned-dataset-queue.service.example)
uses systemd resource controls; the CUDA allocator limit remains a per-process
setting, not GPU isolation.

The [independent update evaluator](../../ml/lfg-encoder/owned-update-evaluator.example.json)
monitors completed queue jobs and runs the configured development-format
check for each. Evaluation does not block training or promote a checkpoint.
The same development examples can be reused for iteration, so they are not a
final independent test. The dashboard exposes aggregate queue progress, format
counts and recent loss from completed initial-training checkpoints, not source
records or labels. Loss is a training statistic, not assessment accuracy.
The example [evaluator service](../../ml/lfg-encoder/systemd/owned-update-evaluator.service.example)
runs independently from the training queue.

The dashboard can also display a separately configured training status via
`--additional-status`. It exposes only the same bounded, sanitized progress
fields as the primary status. The lab's teacher panel can point at the
currently selected collection rather than a completed earlier batch.

## Local deployment evidence

The owner's lab and `clyffy-03` Spark use an authenticated dataset-only HTTPS
endpoint over the existing private tailnet. Tailscale SSH and its check policy
were not modified. A completed 349-record dataset was delivered twice and the
receiver recognized the second delivery as identical. A later collection of
20 additional screened real requests produced a 369-record dataset, which the
publisher delivered to the Spark inbox after a manual service trigger; its
periodic timer is enabled. The queue verified that
it contains the prior 349 records and measured the required context. The
initial 349-record assessment fine-tune completed. The queue picked up
the 369-record dataset and completed the subsequent 5,000-step fine-tune,
with all 369 examples fitting the configured context. The queue returned to
idle. Its separate check on the reused 10-request development set returned
four JSON objects with the required fields, five invalid JSON responses and
one JSON object missing required fields. It did not measure recommendation
accuracy or qualify the checkpoint for promotion.

The transfer receiver, publisher, queue restart behavior and independent
evaluator were tested with isolated software fixtures; those fixtures are not
training data. There is no automatic discovery or approval of arbitrary user
projects, no automatic retry of individually incomplete teacher answers, and
no automatic deployment or promotion of newly trained weights. Adding new
collector configurations, selecting authorized sources and expanding resource
limits remain explicit configuration work.
