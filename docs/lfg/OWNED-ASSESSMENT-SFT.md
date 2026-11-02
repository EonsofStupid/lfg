# Owned student assessment fine-tuning

`ml/lfg-encoder/train_owned_assessment_sft.py` fine-tunes an owned byte student
on the training file produced by [assessment distillation preparation](ASSESSMENT-DISTILLATION.md).
It checks the dataset and checkpoint digests, reads **only** `train.jsonl`, and
rejects a dataset manifest that declares a test file. A declared validation
file is not opened. The dataset's teacher, outcome, review, authorization and
redaction fields remain supplied claims; the trainer does not verify them or
convert them into ground truth. It checks each training row against the
manifest's declared source, project and group and checks for declared
train/validation group or source overlap. That does not prove true project
identity or semantic independence. The output is always unqualified.

The configuration supplies the system/user/assistant framing, review and
outcome sampling weights, learning settings, resource limits, CPU or CUDA
device, and context policy. Unknown configuration and input fields are
allowed. A zero category weight excludes that category; a positive unreviewed
weight permits exploratory training without claiming a human review. Choose
weights and a training source deliberately. Do not treat a recorded model
setting, synthetic fixture, or unknown outcome as a successful assessment.

Loss is computed only for the assistant response and configured suffix. The
trainer samples examples according to the configured weights and samples an
end position uniformly within each response. Within a sampled window it
weights each target by the inverse of the number of windows that can contain
it. In expectation this gives each response byte equal weight within that
example, instead of favoring earlier bytes. Padding and prompt bytes do not
contribute to the loss.

By default, a complete framed example must fit within the configured context;
`allow_truncated_prompt: true` explicitly permits sliding windows that may
omit part or all of the original prompt at later response positions. Counts of
examples affected are recorded. The context cannot shrink the starting
checkpoint. If it grows, existing positional rows are copied and new rows are
randomly initialized using the configured seed; those rows are not pretrained.
Extending context is not evidence that the model understands long requests.
The configured attention-element limit is a rough tensor-shape estimate, not a
bound on allocator, optimizer, model or device memory.

```powershell
python ml/lfg-encoder/train_owned_assessment_sft.py `
  --dataset PRIVATE_PREPARED_DATASET --checkpoint PRIVATE_OWNED_CHECKPOINT `
  --config PRIVATE_SFT_CONFIG.json --output NEW_PRIVATE_OUTPUT_DIRECTORY
```

The [example configuration](../../ml/lfg-encoder/owned-assessment-sft.example.json)
illustrates the format; its values and framing are not recommended production
settings. The output directory must be new, and the entire directory must
be ignored and untracked when inside Git. It contains weights and aggregate
provenance, not raw examples, but trained weights and digests can disclose
private data. Keep the inputs, configuration, output and logs local. A failed
run can leave an incomplete output directory. `--max-config-bytes` can adjust
the configuration-file size limit.

The manifest records the source checkpoint, training file and configuration
digests, code hashes, status/outcome claim counts, context changes, sampled
supervised positions, loss history and runtime settings. It does not assert
that data is properly redacted, the source claims are true, the model is useful,
or evaluation has occurred. CUDA is used only when explicitly selected and
available; the CUDA device index, PyTorch allocator fraction, deterministic
algorithms and TF32 are configurable. The allocator fraction is not a hard GPU
or host-memory isolation limit and does not cover CUDA context or non-PyTorch
allocations. The manifest records PyTorch's peak allocated and reserved GPU
bytes for CUDA runs. Identical results across devices or software versions are
not guaranteed. This version
does not provide a serving or update path for the resulting assessment model.

## Checkpoints and resume

At `checkpoint_every` steps and at the end of an invocation, the trainer writes
an immutable step directory with model weights, AdamW state, random states,
sample progress and input identities. Tensor data uses safetensors; optimizer
checkpoints are not loaded with Python pickle. `latest.json` points to the last
completed checkpoint and records its state-file digest. The trainer writes the
step in a temporary directory before renaming it and updates that pointer only
afterward. A failed write can leave an incomplete `.writing-*` directory; the
previous completed checkpoint remains available. These filesystem operations
do not guarantee durability after every possible power loss.

Resume into a **new** private output directory, using the checkpoint directory
and state SHA-256 from the prior run's `latest.json` (or a separately computed
digest for a deliberately selected older checkpoint):

```powershell
python ml/lfg-encoder/train_owned_assessment_sft.py `
  --dataset PRIVATE_PREPARED_DATASET --checkpoint PRIVATE_STARTING_CHECKPOINT `
  --config PRIVATE_SFT_CONFIG.json --output NEW_PRIVATE_OUTPUT_DIRECTORY `
  --resume PRIVATE_CHECKPOINT_STEP_DIRECTORY --resume-state-sha256 STATE_SHA256
```

An optional `--max-steps-this-run` ends an invocation early after writing a
checkpoint, allowing an operator to schedule bounded segments. Resumption
checks the checkpoint's state and tensor digests, source data, initial weights,
implementation, Python/PyTorch/safetensors versions, device and objective configuration. Total steps, checkpoint
frequency and resource limits can be changed; model, framing, sampling and
optimization settings must match. A hash is an integrity check, not a signature
or authentication of the source. The tool does not launch itself automatically
or supervise a persistent service.

All checkpoints are retained. Each includes model weights and optimizer state,
which can require several times the model's size; choose the interval and
provision private disk space accordingly. Per-file size bounds are configurable,
but snapshot serialization also temporarily holds copies in memory, and there is
no automatic retention policy or total disk quota. A partial run
has no final `manifest.json` or final weights. Only a completed run produces
those files.

Software fixtures verify masked loss, weighting, input separation, repeated
CPU execution, optimizer-state resume, corruption rejection and recovery from
a simulated checkpoint write failure. One fixture-trained checkpoint was also
exported to ONNX and matched the experimental Rust probe; this tests format and
numerical compatibility only. In a separate **read-only** check on 2026-11-02, the existing
single unreviewed real-request pilot had a 792-byte framed prompt and a
423-byte response. The original checkpoint had context 64. The illustrative
2,048-byte setting accommodates the complete example but initializes 1,984
additional position rows; the expanded model has 101,376 parameters. No real
assessment fine-tuning or quality evaluation was performed by this check. One
unreviewed example is not a representative training or evaluation corpus.
