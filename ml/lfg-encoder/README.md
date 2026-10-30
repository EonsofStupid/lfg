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
2. Train with a pinned model revision only after the sample gate.
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
then from the repository root. Supply the full immutable commit for a test model
as `MODEL_COMMIT` (the example model and override are for plumbing only):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ml/lfg-encoder/requirements.txt
.\.venv\Scripts\python.exe ml/lfg-encoder/train.py validate
.\.venv\Scripts\python.exe ml/lfg-encoder/train.py train --model-revision MODEL_COMMIT --output ml/lfg-encoder/artifacts/smoke --allow-below-gate --checkpoint-steps 1
```

Training downloads the selected pinned model and can be slow on CPU. The
six-example run tests plumbing only; its output is ineligible for promotion and
is ignored by Git. `requirements.txt` specifies dependency ranges, not a lock;
capture and reuse an exact environment for a serious experiment.
The `--allow-below-gate` flag is for development only and allows unverified rows
into the experiment. Without it, training requires the policy's minimum number
of records explicitly marked `verified` and excludes unverified rows. A run
requires at least two families and writes the seed, dataset hash, row IDs and
family assignments to `split-manifest.json` in the ignored output directory.
This family split is for development validation, not a final test: it is not
stratified and can leave rare labels absent from one side. Reserve a separately
reviewed, project-separated test set for final evaluation.

## Checkpointing and restart

Training settings live in `training-config.v1.json` and can be overridden by
the CLI. Every run needs a new output directory and a full 40-character model
commit. A run manifest records the model revision, dataset and source hashes,
split, settings, Python, package versions, OS and visible GPU names. These are
evidence for reproducing an experiment, not a guarantee of bitwise-identical
GPU results. The manifest and split contain IDs and family names: protect them
as project data and do not put sensitive information in those fields.

Checkpoints are written every configured number of optimizer steps. To resume,
use the exact same command and add `--resume`; it verifies the existing manifest
and requires the latest checkpoint's trainer, optimizer, scheduler and RNG
state. The dataset, code, policy, configuration and recorded environment must be
unchanged. The same total epoch setting is required; resume finishes that run,
it does not silently extend it. A different experiment gets a new directory.
An interruption before the first checkpoint cannot be resumed; start a new run.
Mismatched pretrained weights are rejected by default. If a different-sized
classification head must be initialized for a test, explicitly enable
`--allow-mismatched-sizes` and inspect every model-loading warning; the option
can also allow other mismatched weights, so it is not a promotion guarantee.

An exclusive lock rejects concurrent trainers for one output. An unclean kill
can leave `<output>.lock`. Verify the process is no longer running before
manually removing that lock; never remove it while a trainer is active. Saving
can be interrupted, so the latest checkpoint may be incomplete and is rejected.
Do not assume a failed checkpoint or a loss value qualifies a model.

For a Spark, run this in a persistent `tmux` session or an operator-managed job
service with checkpoint storage on a durable filesystem. This repository does
not install or supervise a service, and the workflow is not yet qualified for
production on a Spark. Checkpointing does not guarantee a save at the instant of a crash;
work since the last completed save is lost. Configure the interval for the
machine's storage cost and acceptable loss, and test interruption/restart on
the actual host before relying on it. Use one trainer per output directory.

One [GB10 infrastructure smoke test](../../docs/lfg/SPARK-SMOKE-EVIDENCE.md)
has since exercised training, interruption, resume and fresh-process GPU reload;
it does not qualify the image, dataset or model for production.
