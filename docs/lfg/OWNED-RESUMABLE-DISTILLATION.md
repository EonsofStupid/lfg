# Resumable owned-student distillation

`train_owned_distill_resumable.py` trains an independently initialized owned
byte student on explicitly selected local project documents, combining
next-byte cross-entropy with the existing byte-aligned Top-K teacher objective.
It supports the existing owned-byte and verified local GPT-2-byte teacher
formats. The teacher is a source of training signal, not LFG's shipped model.
The operator must select an authorized teacher and verify its terms and byte
mapping. A 256-token vocabulary alone does not prove byte alignment.

The student, distillation, screening and runtime settings come from separate
configurations. CPU and CUDA are explicit; a missing requested CUDA device is
an error, never a CPU fallback. The CUDA device index, PyTorch allocator
fraction, deterministic algorithms and TF32 are configurable. The allocator
fraction is not GPU isolation and does not cover context or non-PyTorch memory.
The runtime configuration also controls checkpoint frequency and per-file
size limits. Existing corpus bounds, parameter limits, teacher limits and
redaction policy still apply.

The private, new output directory contains atomic step snapshots with model,
optimizer, sampler and random states. A resume requires the expected checkpoint
state digest and matching source, teacher, implementation, objective, device
and software identities. Total training steps and checkpoint frequency can be
changed. Completed runs also write `weights.safetensors` and `manifest.json`;
partial runs do not. Checkpoints are retained and can use several times the
model size each. This is resumable execution, not an automatic training service;
allocate private disk and use a host supervisor or scheduler for persistence.

`stage_owned_documents.py` creates a private training-host copy of only the
selected, screened training documents and a matching relative-path plan. It
does not open or copy validation or test sources. The staging manifest records
the source plan, screening policy, screened content and code identities. The
operator must protect and transfer this directory only to an authorized host.
Pattern screening is incomplete by design and does not establish that every
secret was found or that a person reviewed the documents.

```powershell
python ml/lfg-encoder/stage_owned_documents.py --plan PRIVATE_PLAN.json `
  --policy PRIVATE_SCREEN_POLICY.json --student-config PRIVATE_STUDENT_CONFIG.json `
  --output NEW_PRIVATE_STAGE_DIRECTORY

python ml/lfg-encoder/train_owned_distill_resumable.py `
  --student-config PRIVATE_STUDENT_CONFIG.json --distillation-config PRIVATE_DISTILL_CONFIG.json `
  --runtime-config PRIVATE_RUNTIME_CONFIG.json --corpus-plan PRIVATE_PLAN.json `
  --screen-policy PRIVATE_SCREEN_POLICY.json --teacher-format gpt2-byte `
  --teacher-manifest PRIVATE_TEACHER_MANIFEST.json --teacher-weights PRIVATE_TEACHER_SNAPSHOT `
  --teacher-revision PINNED_REVISION --output NEW_PRIVATE_OUTPUT_DIRECTORY
```

For a bounded invocation, `--max-steps-this-run` saves a checkpoint and stops;
resume with `--resume CHECKPOINT_STEP_DIRECTORY --resume-state-sha256 SHA256`
into a new output directory. Logs, source files, snapshots, weights and hashes
are private. The trainer does not evaluate a validation or test set, train an
assessment response, establish model quality or promote a model. Identical
results across hardware and software versions are not guaranteed.

Software fixtures verify separation, staging, resume identity and repeatable
CPU execution. A local integration check on 2026-11-02 used the previously
screened owner-project development documents and pinned byte teacher for one
CPU step, then resumed for one more: 512 byte-token positions in total. It did
not open the declared validation source. This was a pipeline check, not evidence
of a useful or qualified model.
