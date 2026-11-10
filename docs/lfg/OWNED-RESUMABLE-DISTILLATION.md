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

## Spark real-project run, 2026-11-02

On the owner's GB10, a separately initialized 679,680-parameter student ran
400 steps over three authorized, screened project documents, totaling 65,946
bytes. The run processed 102,400 next-byte training positions using a pinned
1,927,936-parameter external byte teacher; that teacher is not the LFG model.
The private screening policy replaced 25 long opaque strings and 43 lines
containing credential-related terms. Automated screening can miss secrets;
there was no claim of complete redaction or human review.

The run resumed twice from private checkpoints, most recently at step 22. An
independent uninterrupted execution on the same host and software produced the
same weight digest and loss history. The peak PyTorch GPU allocation reported
by the initial segment was 93,159,424 bytes, with a 0.01 allocator fraction;
the host scope also limited CPU, memory and tasks. The existing SWAG service
was observed running before and after, which does not establish zero
interference. The installed environment was not wheel-locked. This confirms
repeatability for this run, not across machines or software versions.

No validation or test source was opened by training. This is byte-level
pretraining and distillation, not training on assessment answers or proof of
assessment quality. Three small documents and a 0.68-million-parameter student
are far short of the intended model and representative training corpus. The
inputs, checkpoints and weights remain private and outside Git.

The same-architecture cross-entropy control was trained for the same number
of steps and sampled from the same screened corpus on the Spark CPU. On the
previously inspected SWAG development document described in
[the byte evaluation](OWNED-BYTE-DEVELOPMENT-EVALUATION.md), the distilled
student scored 3.633 bits per byte and 32.20% next-byte accuracy; the control
scored 3.694 and 32.07%, respectively. Both were evaluated with 64-byte
windows over 6,019 predicted bytes. The pretrained teacher scored 3.514 and
51.20%. The control and candidate were trained on different devices, so the
comparison includes possible device effects. This is one non-blind development
document from a related project, with unknown overlap; it is not an independent
assessment evaluation or a reliable estimate of product quality. The report
and model identities are private.

An additional 400-step GB10 run used the same student architecture and
teacher with a new corpus of 13 screened documents: the three project
documents above and the ten recovered, owner-authored question-bank,
curriculum, routing and addendum notes. The original notes were checked
against their private archive manifest before staging. The 189,715-byte corpus
used four uniformly sampled project groups, including a separate owner-notes
group. Screening the added material replaced one long opaque string and 21
credential-context lines; rescanning the staged corpus on the training host
made no further replacements. This remains pattern screening, not a guarantee
of complete redaction. The run processed 102,400 byte positions; it did not
open a test source or train assessment answers. SWAG was observed running before
and after the bounded run. Reading the authored questions as byte text does not
establish that the student understands or can apply them.

On the same previously inspected SWAG development document and 64-byte-window
setup, the notes-inclusive student scored 3.628 bits per byte and 32.35%
next-byte accuracy, compared with 3.684 and 31.67% for its matching
cross-entropy control. Its 3.628 bits per byte compares with 3.633 for the
earlier project-only student on this one document. Those small differences
are not evidence of a reliable improvement, question answering ability, or
independent assessment quality; training devices differ between each student
and its CPU control, and the evaluation document has been used repeatedly.

An optimized Windows executable containing this notes-inclusive student and the
branch's built UI passed empty-working-directory health and UI smoke checks.
Its embedded native next-byte logits matched the PyTorch checkpoint at lengths
1, 4, 32, 64 and 128, with maximum absolute difference 0.00000287; 16 greedy
generated bytes also matched. On three short project-document prefixes, 60
sequential 16-byte generations measured median 58.33 ms and p95 152.42 ms
while the machine was also building software. Threads were not controlled and
the benchmark does not test assessment behavior. The executable contains
derived private weights and should be protected accordingly. It has not been
tested on a clean machine or compared across independent builds.

Separately, two isolated Windows release builds of commit
`4e4ef27263486e76227ebd399cd6d8943ef2ea7c`, using the earlier three-project
checkpoint (without the authored notes) and the same local toolchain, produced
identical 34,760,704-byte executables and UI assets. Both binaries had SHA-256
`303132cc74bba53bfc7b49c37e469dbf4506d1b95da5aaa3fc8b7dcf05c941ba`.
One matched the source checkpoint in native next-byte comparisons through 128
bytes. This is same-machine, same-toolchain reproducibility for that commit
and artifact, not a clean-machine test or reproducibility evidence for the
later notes-inclusive or assessment-pilot models.
