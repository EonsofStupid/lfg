# Experimental per-node exemplar extension

The owner's [LFG v3 notes](https://docs.google.com/document/d/1NCIJK8wBlnzAa43OuWLktQup3RtksqDvM8n2ckPWj5U/edit) describe a frozen encoder and exemplar similarity as an extension at LFG kernel escalations. `ml/lfg-encoder/node_exemplars.py` is an offline experimental implementation of that path. It is separate from the earlier global multi-label classifier. It does not fine-tune encoder weights, change the LFG kernel, promote rules, or provide a native or self-contained runtime.

## How it works

The operator selects a local, link-free model snapshot and manifest, a reviewed
per-node corpus and admission policy, and a versioned configuration. The encoder
loads locally with remote code disabled, runs on CPU in evaluation mode, and
produces attention-mask mean-pooled, L2-normalized vectors. Inputs exceeding the
configured byte or token limits fail; they are not silently truncated. The
configuration also bounds batch size, CPU threads, embedding dimensions, item
counts, pack bytes and prediction bytes. The selected model must be suitable for
this pooling method; the tool does not determine that automatically.

An experimental [native encoder probe](NODE-ENCODER-NATIVE-PROBE.md) can run
the frozen encoder in the Rust binary and compare its vectors with Python. A
separate [native suggestion command](NATIVE-NODE-SUGGESTION.md) can load a
pinned exemplar pack for offline escalation experiments.

After the [review and admission gate](NODE-REVIEW-AND-ADMISSION.md) passes,
`build` encodes only approved **training** records into a new private exemplar
pack. It stores item digests, labels and vectors, not raw requests. Vectors and
digests can themselves be sensitive and must remain access-controlled. The pack
identifies the snapshot, review inputs, model manifest, environment, encoder
configuration and builder. It is marked `promotionEligible: false`. The gate
currently reads supplied test records to check support and separation; it is not
a blinded holdout. Declarations and hashes do not authenticate a reviewer, model
publisher or source rights, and a person with write access can forge artifacts.

The `predict` command requires the trusted pack digest and the same audited
inputs, model, encoder settings and locked environment. It encodes approved
validation requests and compares each vector against the stored training
vectors. For each option, its score is the highest dot product with one of that
option's unit vectors. The best option is suggested only when its score meets
`minimumSimilarity` and its lead over the next option exceeds `minimumMargin`;
otherwise the result is null. These scores are **uncalibrated similarities**,
not probabilities. Decision thresholds can be changed through configuration
without rebuilding vectors; changing the encoder settings requires a new pack.
The checked-in [example configuration](../../ml/lfg-encoder/node-exemplar-config.example.json)
is a schema and mechanics example, not a calibrated recommendation.

Predictions are written to a new private JSONL file compatible with the
[node validation runner](NODE-VALIDATION.md), which scores them only when the
LFG kernel escalated at the target node. That runner consumes the supplied file;
it does not itself execute or authenticate the candidate. The current path uses
a closed choice list from one exact snapshot: adding categories still requires
a new snapshot and matching review; it is not yet the complete dynamic category
or snapshot distribution system described in the notes.

## Running the tools

Choose a compatible locked Python environment, exact local model snapshot and
manifest, and an access-controlled location for outputs. The operator is
responsible for checking model provenance and license. `build` and `predict`
verify the supplied wheelhouse and installed wheel payload before and after the
operation using the existing training input checker. These checks can take
minutes and do not attest the interpreter, every generated file, external native
libraries, or concurrent hostile modifications. On Windows, a typical shape is:

```powershell
VENV/Scripts/python.exe ml/lfg-encoder/node_exemplars.py build `
  --input PRIVATE_QUEUE.jsonl PRIVATE_LEDGER.json `
  --snapshot SNAPSHOT.json --policy ADMISSION_POLICY.json --node NODE_ID `
  --model LOCAL_MODEL_DIR --model-manifest MODEL_MANIFEST.json --revision MODEL_REVISION `
  --config EXEMPLAR_CONFIG.json --environment-lock ENVIRONMENT_LOCK.txt `
  --wheelhouse WHEELHOUSE --allowed-extra pip --output NEW_PRIVATE_PACK.json

VENV/Scripts/python.exe ml/lfg-encoder/node_exemplars.py predict `
  --input PRIVATE_QUEUE.jsonl PRIVATE_LEDGER.json `
  --snapshot SNAPSHOT.json --policy ADMISSION_POLICY.json --node NODE_ID `
  --model LOCAL_MODEL_DIR --model-manifest MODEL_MANIFEST.json --revision MODEL_REVISION `
  --config EXEMPLAR_CONFIG.json --environment-lock ENVIRONMENT_LOCK.txt `
  --wheelhouse WHEELHOUSE --allowed-extra pip --pack PRIVATE_PACK.json `
  --pack-sha256 TRUSTED_PACK_SHA256 --output NEW_PRIVATE_PREDICTIONS.jsonl
```

Repeat `--input` for every relevant queue and ledger. `--allowed-extra` should
name only explicitly permitted bootstrap packages. Both outputs are exclusive:
they do not overwrite existing files. A failed write can leave a partial file;
inspect it and choose a new path. Limits bound input and output sizes, but the
implementation holds embeddings and serialized artifacts in memory and is not
yet qualified for large corpora. Floating-point results can vary by environment;
same-process repetition is not cross-platform reproducibility.

To check a selected frozen encoder without producing a pack, use `probe` with
`--queue`, `--ledger`, `--snapshot`, the model and manifest flags above,
`--config`, and `--repetitions 2`. It may read pending requests but uses no labels
and saves no vectors. It reports token ranges, dimension and whether two passes
were exactly equal in the same process. This is a functional probe, not training,
accuracy, calibration or latency evidence.

## Observed local checks

On 2026-10-31, the build command was refused against the real LFG queue because
its 11 records had zero approved labels; no real exemplar pack was created. A
local CPU probe ran the frozen `sentence-transformers/all-MiniLM-L6-v2` snapshot
at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` on those 11 requests.
It produced 384-dimensional vectors with token counts from 17 to 145; two
passes were exactly equal in that process. No labels or vectors were saved.

On the Spark, the same configurable CPU path loaded the separate locked
`answerdotai/ModernBERT-base` snapshot at revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8` and processed one test-only queue
containing the owner's published LFG golden example. It produced a 768-dimensional
vector and two equal passes in that process. No private LFG requests were sent to
the Spark. Both observations are functional checks of selected local models,
not evidence of suitable embedding quality, GPU training, performance, or a
qualified release. Test-only vectors and simulated review decisions were also
used for unit tests on Windows and Spark; they are not training data.
