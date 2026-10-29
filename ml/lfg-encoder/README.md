# LFG encoder experiment

This is the offline, non-authoritative training lane for the Rust
MasterClyffy CPU encoder. Runtime prompts are not collected here. The checked-in
corpus is intentionally authored synthetic data.

These experiments support the assessment and routing objective in the root
[README](../../README.md). The classifier is exploratory and is not the authoritative runtime encoder.
Any learned candidate needs task-specific evaluation before integration or
promotion.

The [draft labeling guide](LABELING.md) and
[RRFlow review batch](dataset/rrflow-review.v1.jsonl) are a starting point for
human review; their constructed examples are not a training-ready corpus.
The separate [PR review batch](dataset/rrflow-pr-review.v1.jsonl) contains
actual PR titles and excerpts linked to outcome journals, not original prompts.

## Current lane (classifier)

The promotion order is fixed:

1. Validate the corpus with `python3 train.py validate`.
2. Train with `python3 train.py train --base-model MODEL` only after the sample gate.
3. Export a versioned ONNX artifact and manifest.
4. Benchmark through the Rust runtime.
5. Promote only when `promotion-policy.json` passes.

Training requires `requirements.txt`. Validation uses only the Python standard
library and does not download a model.

## Starting a reproducible classifier experiment

Start with this classifier, which predicts the existing intent, capability, and
risk labels. The current six synthetic examples are a plumbing check, not a
representative evaluation set. The `verified` field is an attestation that a
human checked the text and all labels; the script counts it but cannot verify
the review happened. The sample gate alone does not make a model promotion-ready.

1. Agree on a versioned label guide: define each label, positive and negative
   examples, and how to handle multiple intents and uncertain cases. Record who
   reviewed each example outside this minimal schema. Only mark `verified` true
   after human review. Use synthetic or explicitly authorized, sanitized prompts;
   do not commit private prompts or secrets.
2. Assemble a representative corpus with hard negatives, paraphrases, multi-step
   requests, and security cases. Keep related examples together when splitting
   data to reduce leakage. Reserve an independently reviewed test set before
   training; do not use it to tune the model. `train.py` groups its development
   split by the supplied `family` values; it cannot detect related records given
   different family names or enforce a separate project-level test set.
3. Measure the Rust rule encoder on the same held-out examples. Compare per-label
   precision and recall, intent/capability macro F1, security recall, calibration,
   CPU latency, and memory. Define acceptable thresholds and the target hardware
   before deciding whether a model improves the product. Agent count, model class,
   and complexity need separate labels and evaluations; this classifier does not
   currently learn them.
4. Train a candidate, export and benchmark it, then evaluate it independently
   against the baseline and the policy. The current script does not implement
   those evaluation, ONNX export, or promotion checks. Keep the rule encoder in
   charge until those checks and runtime integration exist and pass.

For a local development smoke test on Windows, install Python 3.10 or newer,
then from the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ml/lfg-encoder/requirements.txt
.\.venv\Scripts\python.exe ml/lfg-encoder/train.py validate
.\.venv\Scripts\python.exe ml/lfg-encoder/train.py train --allow-below-gate
```

Training downloads the default base model and its dependencies and can be slow on
CPU. The six-example run tests plumbing only; its output is ineligible for
promotion and is ignored by Git. The training script currently uses an unpinned
model revision and dependency ranges, so save the exact model revision, package
versions, dataset hash, seed, and evaluation split for any serious experiment.
The `--allow-below-gate` flag is for development only and allows unverified rows
into the experiment. Without it, training requires the policy's minimum number
of records explicitly marked `verified` and excludes unverified rows. A run
requires at least two families and writes the seed, dataset hash, row IDs and
family assignments to `split-manifest.json` in the ignored output directory.
This family split is for development validation, not a final test: it is not
stratified and can leave rare labels absent from one side. Reserve a separately
reviewed, project-separated test set for final evaluation.
