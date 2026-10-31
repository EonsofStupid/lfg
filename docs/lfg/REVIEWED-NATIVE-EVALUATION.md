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
and the model run records that same audit, dataset, split, environment evidence
and model-snapshot evidence. It rejects below-gate runs. It checks that the
ONNX manifest names the selected source model artifact and taxonomy, verifies
the declared ONNX files, and binds both prediction producers and the native
model identity to the selected binary and runtime configuration. It repeats
input checks after inference. These hashes establish consistency, not the
authenticity of an operator-supplied artifact, reviewer identity, authorship,
or proof that two projects are unrelated.

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
persistent CPU performance benchmark. The native command currently returns
binary labels, so calibration is unavailable. A report always has
`promotion_eligible: false`: separate calibrated quality, security, latency,
memory, quantization and release checks remain necessary. Low or absent label
support makes any metric correspondingly weak; no support threshold is
automatically treated as sufficient.

The current audit reads the declared test records before training to verify the
plan. The training code does not feed them to the model, but this process does
not enforce a blinded test or prevent an operator from inspecting them. Project
and family separation are declarations and normalized exact-text checks; they
cannot detect every related conversation or paraphrase. Independent holdout
selection and access control remain necessary.

On 2026-10-31 the tool rejected the actual local review workspace before
inference because its training records have no current approvals. No report was
created. A separate local check confirmed that the existing native binary's
identity digest matched its declared artifact, runtime and taxonomy components
for one real, unreviewed DevPulse excerpt; that is a wire-identity check, not a
reviewed evaluation or quality measurement.

Separately, a fixture-only plan with simulated review approvals and one test
record was run end to end through the actual native binary and a separately
exported fixture ONNX artifact. The command returned a non-promotable report;
the fixture contained no positive `sensitive-data` examples, so its security
recall was unavailable. This verifies the wiring, not real-data quality,
human review, calibration, or release readiness. The fixture report remains in
`target/lfg-reviewed-native-eval-fixture-v1.json` on the development machine.
