# Reviewed native comparison

`evaluate_reviewed_native.py` compares the deterministic baseline with an
operator-selected ONNX shadow artifact using test records from an audited
corpus plan. It launches both commands from the selected LFG binary and sends
one request at a time over local stdio. The report contains aggregate counts,
model and input identities, per-label precision/recall/F1, and positive and
negative test support; it contains no request text or per-request predictions.
Hashes and source metadata can still be sensitive, so keep the report private.

The tool checks that every train, validation and test record has a current
review attestation, the training dataset matches the audited train partition,
and the model run records that same audit, dataset, split, locked-wheel payload
evidence and model-snapshot evidence. It rejects below-gate runs and older
manifests without payload evidence. The evidence does not cover generated
bytecode, scripts, bootstrap tools, shadowing files or the interpreter. It checks that the
ONNX manifest names the selected source model artifact and taxonomy, verifies
the declared ONNX files, and binds both prediction producers and the native
model identity to the selected binary and runtime configuration. It repeats
input checks after inference. These hashes establish consistency, not the
authenticity of an operator-supplied artifact, reviewer identity, authorship,
or proof that two projects are unrelated.

The current evaluator also requires the run manifest to declare the trained
label groups and bind the assessment-target validator and audited target
coverage. Historical run manifests without these identities need their matching
historical evaluator; they cannot be silently treated as current runs. Target
coverage in the training manifest is limited to train and validation; the
standalone audit can report all splits when run on a complete plan. This
comparison scores intent, capability and risk only. Reviewed complexity, model,
effort and agent-count judgments are not yet predicted or scored here.

For example, after obtaining a genuinely reviewed, separated corpus and a
trained model with its exported ONNX artifact, choose a new private report:

```powershell
python ml/lfg-encoder/evaluate_reviewed_native.py `
  --plan REVIEWED_PLAN.json --training-dataset REVIEWED_TRAIN.jsonl `
  --model TRAINED_MODEL_DIR --artifact ONNX_ARTIFACT_DIR `
  --binary LFG_BINARY --runtime-config NATIVE_RUNTIME_CONFIG.json `
  --config ml/lfg-encoder/benchmark-persistent.v1.json `
  --mode embedded --output NEW_PRIVATE_REPORT.json
```

Use `--mode external` when the selected binary loads the provided ONNX artifact
from disk. `embedded` verifies that the binary reports the supplied artifact's
manifest digest; the selected artifact directory is still required to verify
source lineage. The existing benchmark configuration provides request timeout
and dataset/response limits. This is an accuracy comparison, not the
persistent CPU performance benchmark. Add `--calibration-config
ml/lfg-encoder/calibration.v1.json` to request native per-label sigmoid scores
and aggregate empirical calibration statistics. The score output is explicitly
uncalibrated and is enabled only for that offline request; it is not a claim of
confidence calibration or a live-routing change.

The native candidate must report valid tokenization metadata and no truncated
evaluation inputs. Older binaries that do not report this metadata, and
predictions with tokenizer overflow, are rejected by this evaluator. Select or
build a compatible artifact and binary for the intended input lengths; the
check does not establish that the model understands every part of an input.

The configuration supplies bin edges from zero to one. For every label, the
evaluator computes empirical binary expected calibration error (ECE),
`sum((bin_count / observations) * abs(mean_score - positive_rate))`, and the
mean squared probability error (Brier score). Bins are left-closed and
right-open, except that the last includes one. The report also gives the mean
of per-label ECE values and the ECE over pooled label observations; these are
different summaries. It verifies that the scores cover the whole taxonomy and
produce exactly the emitted labels at the artifact's configured thresholds.
Per-label support and bin counts remain essential: a score computed from a
small or unrepresentative test set does not establish calibration. Individual
scores are held in process memory, while the report contains aggregate values.

A report always has `promotion_eligible: false`: separate quality, security,
latency, memory, quantization and release checks remain necessary. Low or absent label
support makes any metric correspondingly weak; no support threshold is
automatically treated as sufficient.

The legacy complete-plan audit reads the declared test records before training to verify the
plan. The training code does not feed them to the model, but this process does
not enforce a blinded test or prevent an operator from inspecting them. Project
and family separation are declarations and normalized exact-text checks; they
cannot detect every related conversation or paraphrase. Independent holdout
selection and access control remain necessary.
The newer development-only plan avoids selecting a test workspace during
training, but this evaluator rejects it. See [development-only assembly](CORPUS-REVIEW.md#development-only-assembly)
and the separate experimental workflow below.

## Controlled classifier holdout

`evaluate_classifier_holdout.py` separates the custodian's test commitment,
development candidate freeze, and final test. It uses the same reviewed native
comparison and model, environment, binary and artifact identity checks as the
complete-plan evaluator. Reports remain unqualified and contain aggregate
metrics, not request text or individual predictions. Hashes, counts, project
names and group names can be sensitive metadata; keep reports private.

1. A holdout custodian exports a genuinely reviewed, **test-only** workspace to a
   new private directory using `review_corpus.py export`, then commits its plan:

```powershell
python ml/lfg-encoder/evaluate_classifier_holdout.py commit-holdout `
  --holdout-plan PRIVATE_TEST_EXPORT/plan.json --output NEW_PRIVATE_COMMITMENT.json
```

The commitment checks current review attestations, verified flags, duplicate
normalized text within test, and source identities. Record the resulting
`output_sha256` independently, and keep test files away from the candidate
operator. Export includes current approvals only: deciding which records to
include must be done independently of candidate performance.

2. After a reviewed development run has produced its model and ONNX artifact,
   freeze it using only its development plan, reviewed training dataset and the
   commitment. The same native binary compares baseline and candidate on
   validation examples; the freeze accepts no holdout-plan argument.

```powershell
python ml/lfg-encoder/evaluate_classifier_holdout.py freeze `
  --plan PRIVATE_DEVELOPMENT/plan.json --training-dataset PRIVATE_DEVELOPMENT/training.jsonl `
  --model TRAINED_MODEL_DIR --artifact ONNX_ARTIFACT_DIR --binary LFG_BINARY `
  --runtime-config NATIVE_RUNTIME_CONFIG.json --config ml/lfg-encoder/benchmark-persistent.v1.json `
  --mode embedded --commitment NEW_PRIVATE_COMMITMENT.json --commitment-sha256 TRUSTED_COMMITMENT_SHA256 `
  --output NEW_PRIVATE_FREEZE.json
```

Use `--mode external` if the binary loads the ONNX artifact from disk. Record
the resulting freeze digest independently. An optional `--calibration-config`
is a frozen input and must be identical for test.

3. The final evaluator repeats the same arguments with command `test`, a new
   output, and `--freeze NEW_PRIVATE_FREEZE.json --freeze-sha256 TRUSTED_FREEZE_SHA256
   --holdout-plan PRIVATE_TEST_EXPORT/plan.json`. It first reruns and checks the
   frozen validation comparison before opening the test plan. It then checks
   committed bytes, identical taxonomy, distinct source files and ledgers,
   current approvals, and declared project, group, family, ID and normalized
   exact-text separation across all splits. It rechecks inputs after inference.

These checks do not establish that sources are independent, that a reviewer is
who they claim, that rights permit use, or that someone could not inspect the
holdout, change the commitment, or repeat a favorable test. They cannot find all
related conversations or paraphrases. Keep roles and access separate and record
digests outside the machine. The code accepts trusted local files and binaries;
temporary audit files include paths and review metadata. It does not set a
statistical adequacy threshold, qualify latency or memory, or authorize release.
The workflow has been exercised with test fixtures and simulated approvals; no
real reviewed model or holdout has yet completed it.

On 2026-10-31 the tool rejected the actual local review workspace before
inference because its training records have no current approvals. No report was
created. A separate local check confirmed that the existing native binary's
identity digest matched its declared artifact, runtime and taxonomy components
for one real, unreviewed DevPulse excerpt; that is a wire-identity check, not a
reviewed evaluation or quality measurement.

With the optional score output, the Windows x86-64 MSVC experimental release
binary SHA-256 `20673c099ad2f722ad686013218b1e579528af65413761a565c71f8a5315202c`
was compared with its source PyTorch model on the existing 15 unreviewed
DevPulse excerpts. Across their five output labels, the maximum absolute score
difference was `5.96e-8` and no threshold decision changed. This is numerical
agreement on that sample, not ground-truth calibration or accuracy. The
embedded weights derive from private project data and are not release-qualified.
The 123,309,568-byte binary was rebuilt offline in a fresh target directory on
the same machine with the same selected artifact, UI assets and static-runtime
flags; the two SHA-256 hashes matched. Its static import table listed Windows
system libraries and no ONNX Runtime or MSVC runtime DLL. This is not a
cross-machine or clean-installation qualification.

Three non-isolated, paired persistent-process runs on the same excerpts compared
the same embedded command with and without `--include-scores`: each process
handled 150 measured requests (15 unique excerpts repeated ten times, after
warmup). Without scores, medians were 14.94–15.50 ms and p95s 28.69–28.81 ms;
with scores, medians were 14.96–15.40 ms and p95s 28.15–28.69 ms. Process
cycles ranged from 48.9–50.6 million per request without scores and 48.5–50.2
million with scores. These overlapping observations do not establish zero
overhead, a 512-token p95, or general performance. The aggregate reports and
score-parity report remain in `dataset/private`.

Separately, a fixture-only plan with simulated review approvals and one test
record was run end to end through the actual native binary and a separately
exported fixture ONNX artifact. The command returned a non-promotable report;
the fixture contained no positive `sensitive-data` examples, so its security
recall was unavailable. This verifies the wiring, not real-data quality,
human review, calibration, or release readiness. The fixture report remains in
`target/lfg-reviewed-native-eval-fixture-v1.json` on the development machine.
A second fixture-only run exercised the optional score and ECE/Brier path; its
single test record cannot establish calibration.
