# Owned student: initial CPU experiment

`ml/lfg-encoder/owned_student.py` defines a newly initialized, configurable
causal byte model with gated depthwise short convolutions and optional causal
attention blocks. It is an LFG research candidate, not a rebranded pretrained
model. The example configuration is intentionally tiny and is not the target
650 MB model. There is no incremental inference cache, quantization, Rust
runtime integration, distributed training, tokenizer training or hardware
architecture search in this experiment.

The same module implements a forward decoupled Top-K loss following the
[LFM2 technical report](https://arxiv.org/html/2511.23404v1#S3.SS3.SSS2).
It separates untempered top-set membership KL from a teacher-mass-weighted,
temperature-scaled conditional KL. The input must include the teacher's
untempered full-vocabulary probabilities for its selected tokens **and** the
remaining probability mass; Top-K logits alone do not determine that mass.
Teacher and student token indices must refer to the same vocabulary. This byte
student cannot directly consume logits from an arbitrary subword-tokenized
teacher; an aligned teacher or a justified alignment method is still needed.
The original CPU experiment below uses next-byte cross-entropy only.

`distill_owned_student.py` adds an online CPU path that combines next-byte
cross-entropy and weighted Top-K distillation. It supports a local
`OwnedByteStudent` checkpoint with a compatible implementation hash, or a
verified local GPT-2 snapshot whose 256 token IDs the operator explicitly
asserts correspond to raw byte values. A 256-token vocabulary alone does not
prove that mapping. The GPT-2 adapter uses the installed Transformers code,
loads local safetensors directly, and does not run remote model code. The
teacher manifest and safetensors bytes
are checked before use and rechecked after training, the teacher is frozen, and
the student is independently initialized. Top-K selection uses a stable tie
order; both the selected probabilities and the remaining full-vocabulary mass
are computed from the teacher. Teacher and student architecture, source,
configuration, code and output hashes are recorded. The manifest's identity
claims are not a signature or proof of teacher quality. The teacher must have
at least the student's context length. An externally trained subword model is
not supported by this path. For GPT-2, specify `--teacher-format gpt2-byte`,
the full `--teacher-revision`, a snapshot `--teacher-manifest`, and the snapshot
directory as `--teacher-weights`.

```powershell
python ml/lfg-encoder/distill_owned_student.py `
  --student-config STUDENT_CONFIG.json `
  --distillation-config DISTILLATION_CONFIG.json `
  --source PRIVATE_REDACTED_SOURCE `
  --teacher-manifest PRIVATE_TEACHER_MANIFEST.json `
  --teacher-weights PRIVATE_TEACHER.safetensors `
  --output NEW_PRIVATE_OUTPUT_DIRECTORY
```

On 2026-11-02, two local distillation runs used the same inspected README and the
tiny earlier owned checkpoint as teacher. Each trained a 37,888-parameter
student for 80 steps on 20,480 byte-token positions. The teacher had itself
seen the same file and has no established task quality; this is an end-to-end
distillation mechanics check, not evidence of a useful teacher or generalizing
model. The runs produced identical weights and loss sequences on the local CPU.
Neither validation nor assessment training was performed.
A cross-entropy-only control using the same student configuration, seed and
source produced different weights, confirming that the added objective affected
the update path; it does not show that distillation improved model quality.

The same day, a pinned local snapshot of [BananaMind Completor V1](https://huggingface.co/Banaxi-Tech/BananaMind-Completor-V1)
provided a separately pretrained teacher. Its model card describes an Apache
2.0 1,927,936-parameter GPT-2 autocomplete model with raw byte IDs 0–255;
its source dataset has separately documented terms. Snapshot revision
`454c7fa5a18e047a53a693a547712dd5e8895156`, license, notice, configuration and
weights were fetched and verified against the source metadata. The owner's
inspected README supplied 11,637 input bytes. The independently initialized
37,888-parameter student trained for 80 steps on 20,480 byte-token positions.
This is local teacher distillation, not adoption of the teacher as LFG; no
source text was sent to the provider. The teacher was designed for autocomplete,
and this tiny run has no independent task-quality evaluation or baseline
comparison. Two runs with the same inputs produced identical weight hashes and
loss sequences on this machine. The software environment is recorded but not
wheel-locked; cross-platform reproducibility is not established.
For a separate next-byte development measurement, see
[owned byte-model evaluation](OWNED-BYTE-DEVELOPMENT-EVALUATION.md).
For configurable screened project-group training, see
[multi-document development training](MULTIDOCUMENT-DEVELOPMENT-TRAINING.md).

`train_owned_student.py` accepts an explicitly selected, already redacted local
file and configuration. It uses random initialization, CPU PyTorch, a
deterministic-algorithm request, seeded sampling, AdamW and next-byte targets.
It saves a new private safetensors file and a manifest with source, code and
configuration hashes, loss history, runtime identity and limitations. The
single-file, in-memory loader is bounded by configuration and is an experiment,
not a scalable corpus loader or resumable training service. There is no
validation split or assessment target in this path. The operator must establish
authorization, licensing and redaction for each input. Output weights can
memorize source content; store them locally and outside Git. Windows access is
controlled by the parent directory ACL. An output inside a Git worktree must be
ignored and not tracked. A failed write can leave an incomplete new directory.

```powershell
python ml/lfg-encoder/train_owned_student.py `
  --config EXPERIMENT_CONFIG.json --source PRIVATE_REDACTED_SOURCE `
  --output NEW_PRIVATE_OUTPUT_DIRECTORY
```

## Recorded local run, 2026-11-02

The owner-authorized LFG repository README was manually inspected and screened
with the example local secret-pattern policy; it produced no pattern matches.
That screening is not proof that all private information or secrets are absent.
The data and derived weights remain local. The run used 11,637 source bytes,
37,888 model parameters, 80 CPU steps and 20,480 byte-token training positions.
The first sampled-batch loss was 5.725779 and the final sampled-batch loss was
3.543533. These are different training batches, not held-out performance.

Two runs with the current implementation, identical source, configuration and
local software produced identical weight hashes and loss sequences. The run did not use a teacher,
evaluate assessments, measure baseline quality, or establish that the model can
generalize. PyTorch documents that [deterministic settings do not guarantee
reproducibility across releases or platforms](https://pytorch.org/docs/stable/notes/randomness.html).

The next substantive steps are a real, license-reviewed open pretraining corpus,
a capable aligned teacher and scalable Top-K data path, reproducible larger-scale and Spark
execution, project-separated assessment targets and evaluation, target-machine
latency and memory measurements, and a native embedded runtime. None of those
requirements is satisfied by the recorded mechanics run.

One research lead is [EvaByte's model card](https://huggingface.co/EvaByte/EvaByte),
which lists a 6.5B byte-level model under Apache 2.0 and describes byte IDs
offset by 64 plus special tokens. Its example requires custom model code. It is
not downloaded, assessed, integrated, or selected as a teacher here; its exact
license, code, byte/special-token probability mapping and hardware requirements
still need verification. Separately, [cross-tokenizer byte-level distillation](https://arxiv.org/abs/2604.07466)
is an active research direction, not evidence that arbitrary teacher logits can
be used with the present 256-way loss unchanged.
