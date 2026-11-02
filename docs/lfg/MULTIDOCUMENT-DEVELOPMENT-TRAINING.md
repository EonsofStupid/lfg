# Screened multi-document development training

The owned-student and byte-aligned distillation trainers accept either the
existing single-file `--source` or a private `--corpus-plan` paired with
`--screen-policy`. A development plan identifies each selected file, its
declared project and project group, split and expected SHA-256. The plan allows
train and validation entries only; a `test` entry is rejected before any source
file is opened. Validation entries are recorded by hash but are not opened by
the training loader. Declared groups, paths and hashes cannot cross splits.
These checks do not prove that related projects, sessions, paraphrases or
teacher pretraining data are independent.

Training files are read locally, checked against the plan, screened in memory
with the operator-selected policy, and checked again after training. The model
receives the screened bytes; source text is not copied to its manifest. Exact
screened duplicates in training are rejected, and sampling windows never cross
document boundaries. Configured source and output byte limits bound the current
in-memory implementation. The policy's regex rules can miss secrets or match
ordinary text; use a reviewed policy and protect source, plan, hashes and
weights. Results explicitly retain `redaction_complete: false` and
`human_reviewed: false`. See [screening limits](PRIVATE-CORPUS-SCREENING.md).

The optional plan `sampling` field can select `byte_uniform` (the default),
`document_uniform`, or `group_uniform`. Group-uniform sampling chooses a
declared training group uniformly, then weights windows by their availability
within that group. The choice is recorded in the corpus identity. It does not
establish representativeness or measure whether groups are truly independent.

Copy the [plan template](../../ml/lfg-encoder/document-development-plan.example.json)
to a private location and replace every placeholder. On PowerShell,
`(Get-FileHash -Algorithm SHA256 -LiteralPath YOUR_FILE).Hash.ToLowerInvariant()`
gives the required digest. Use distinct project groups for validation; paths
may be absolute or relative to the plan. These entries are explicit selections,
not an automatic claim that the files are authorized or secret-free.

```powershell
python ml/lfg-encoder/distill_owned_student.py `
  --student-config PRIVATE_STUDENT_CONFIG.json `
  --distillation-config PRIVATE_DISTILLATION_CONFIG.json `
  --corpus-plan PRIVATE_DEVELOPMENT_PLAN.json `
  --screen-policy PRIVATE_SCREEN_POLICY.json `
  --teacher-format gpt2-byte --teacher-revision PINNED_REVISION `
  --teacher-manifest PRIVATE_TEACHER_MANIFEST.json `
  --teacher-weights PRIVATE_TEACHER_SNAPSHOT_DIRECTORY `
  --output NEW_PRIVATE_OUTPUT_DIRECTORY
```

The cross-entropy control accepts the same plan and policy through
`train_owned_student.py`. The byte evaluator checks every recorded original and
screened training-file hash against the selected evaluation file and reports
whether it matches a declared validation hash. That is exact-file protection,
not a semantic leakage audit. All derived artifacts stay local and outside Git.
The implementation does not yet stream large corpora, resume checkpoints,
train assessment targets, or provide a production training service.

## Local development run, 2026-11-02

The private plan selected three owner-authorized documents from LFG, RRFlow and
DevPulse: 69,752 screened UTF-8 bytes in total. The example policy found no
credential-shaped matches; that does not establish complete redaction. The
previously inspected SWAG development document was declared as validation and
not opened during training. Makers-Impulse was not supplied. Equal group
sampling prevents the larger RRFlow file from receiving all of the extra weight
solely because it is longer.
Replaying the configured seeded sampler yielded 544 RRFlow, 529 DevPulse and
527 LFG windows out of 1,600. These are replayed selection counts, not separately
instrumented training telemetry.

The same 37,888-parameter student configuration trained for 400 CPU steps
(102,400 sampled byte positions) with and without the pinned pretrained byte
autocomplete teacher. On the previously used 6,019-byte-target SWAG development
measurement, the distilled model scored **3.995 bits per byte** and **27.51%**
next-byte accuracy; the cross-entropy control scored **4.063 bits per byte** and
**27.15%**. The teacher scored 3.514 bits per byte and 51.20% accuracy. These
figures come from one related, already inspected document and do not establish a
statistically reliable improvement, independent generalization or assessment
quality. The run is neither a 650 MB model nor a Spark or release evaluation.
Two runs of each training configuration reproduced their weights and loss
sequences locally; the environment was not wheel-locked and cross-platform
reproduction has not been established.
