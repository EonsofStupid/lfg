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
The [corpus review guide](../../docs/lfg/CORPUS-REVIEW.md#local-codex-project-session-messages)
also documents an inventory of real project-session messages awaiting review.
The separate [PR review batch](dataset/rrflow-pr-review.v1.jsonl) contains
actual PR titles and excerpts linked to outcome journals, not original prompts.
For a bounded real-corpus measurement of the baseline process, including
Windows CPU-time and cycle counters, see the [process benchmark](../../docs/lfg/CPU-PROCESS-BENCHMARK.md).
The [persistent comparison](../../docs/lfg/PERSISTENT-CPU-BENCHMARK.md) measures
repeated local requests after process warmup.
The [CPU training repeatability experiment](../../docs/lfg/REPRODUCIBLE-CPU-TRAINING.md)
records two clean environments, a hash-locked wheelhouse and the real-request runs.
The [GB10 environment record](../../docs/lfg/SPARK-LOCKED-ENVIRONMENT.md)
documents a separate ARM64 wheel lock, pinned model snapshot and runtime probe.
For the first below-gate run using real conversation requests and its negative
diagnostic result, see the [development experiment](../../docs/lfg/REAL-REQUEST-DEVELOPMENT-EXPERIMENT.md).

## Current lane (classifier)

The promotion order is fixed:

1. Audit a reviewed, project-separated corpus plan and validate the assembled
   training snapshot with `python3 train.py validate --dataset DATASET`.
2. Train with the audited plan and a pinned model revision only after the sample gate.
3. Export a versioned ONNX artifact and manifest.
4. Benchmark through the Rust runtime.
5. Promote only when `promotion-policy.json` passes.

Training requires `requirements.txt`. Validation uses only the Python standard
library and does not download a model.
Reviewed training also requires an operator-selected wheel lock, matching
wheelhouse and local model snapshot manifest. See the [bound training inputs](../../docs/lfg/REPRODUCIBLE-CPU-TRAINING.md#bound-training-inputs).

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
   split by the supplied `family` values only in development override mode. The
   audited mode uses the plan's separate reviewed validation project; neither
   mode can detect related records given different family names.
3. Measure the Rust rule encoder on the same held-out examples. Compare per-label
   precision and recall, intent/capability macro F1, security recall, calibration,
   CPU latency, and memory. Define acceptable thresholds and the target hardware
   before deciding whether a model improves the product. Agent count, model class,
   and complexity need separate labels and evaluations; this classifier does not
   currently learn them.
4. Train a candidate, export and benchmark it, then evaluate it independently
   against the baseline and the policy. The experimental evaluator below can
   report classification errors. ONNX export and the optional
   [native CPU shadow command](../../docs/lfg/ONNX-CPU-SHADOW.md) are experimental;
   live routing integration, calibration, independent-test qualification and
   promotion checks are still missing. Keep the rule encoder in charge until
   those checks pass.

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
of verified records and an audited project-separated plan with current review
attestations for every train, validation, and test example. The input must
exactly match its reviewed train partition and contain no unverified rows. See
[corpus review and separation](../../docs/lfg/CORPUS-REVIEW.md). A run writes
the seed, dataset hash, row IDs and family assignments to `split-manifest.json`
in the ignored output directory. Development override requires at least two
families.
With `--allow-below-gate`, the family split is for development only and is not
stratified. Without that override, the complete audited train partition trains
the model and the separate audited validation partition selects checkpoints. The
trainer rejects validation labels absent from the training partition instead of
silently discarding them. For every configured taxonomy label, it also requires
positive and negative examples in both partitions, using configurable minimums
from `training-support-policy.v1.json` or `--support-policy`. A threshold of
one only establishes basic coverage; it does not qualify model quality. The
policy digest and counts are included in the run manifest. A separate reviewed
test project remains reserved for
final evaluation and is not fed to the trainer. The audit still reads the test
file; genuinely blinded evaluation needs independent access controls.

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

## Compare a candidate with the Rust baseline

The Rust binary's `baseline-predict` reads dataset JSONL on stdin and emits one
prediction per example. `predict_model.py` uses a saved experimental model on
CPU and emits the same schema. Both outputs contain example IDs and input
SHA-256 hashes, not prompt text. These identifiers and hashes can still reveal
information through guessing: keep predictions and reports private, and never
publish them without review. Thresholds, CPU threads, input limits and batch
size are configured in `prediction-config.v1.json`; its default threshold is an
uncalibrated development setting, not a quality claim.

On Windows from the repository root, first build the Rust server and use a
separate dataset and model directory. Replace `MODEL_DIR` and `DATASET` with
actual paths; output files in `target` are Git-ignored. The commands use
`cmd /c` for byte-preserving redirection in Windows PowerShell:

```powershell
cargo build -p lfg-server
cmd /c "target\debug\lfg-server.exe baseline-predict < DATASET > target\baseline.jsonl"
cmd /c ".venv\Scripts\python.exe ml\lfg-encoder\predict_model.py --dataset DATASET --model MODEL_DIR --taxonomy ml\lfg-encoder\evaluation-taxonomy.v1.json --config ml\lfg-encoder\prediction-config.v1.json > target\candidate.jsonl"
.\.venv\Scripts\python.exe ml/lfg-encoder/evaluate.py --dataset DATASET --taxonomy ml/lfg-encoder/evaluation-taxonomy.v1.json --predictions target/baseline.jsonl --training-split MODEL_DIR/split-manifest.json
.\.venv\Scripts\python.exe ml/lfg-encoder/evaluate.py --dataset DATASET --taxonomy ml/lfg-encoder/evaluation-taxonomy.v1.json --predictions target/candidate.jsonl --training-split MODEL_DIR/split-manifest.json
```

These examples assume paths without spaces; quote paths as appropriate for the
shell. For unreviewed draft records, `--allow-unverified` is required on each
evaluation command. Results then describe agreement with draft labels only.
The evaluator rejects mismatched IDs, input hashes, model identities, unknown
labels and declared overlap with a supplied training split. A supplied split
only checks declared IDs and family names; it cannot establish independent
projects, provenance, human review, unseen paraphrases, or absence of other
leakage. Without a split, separation is `not_checked`. Every report is marked
ineligible for promotion until those requirements and promotion gates exist.
The separate [reviewed native comparison](../../docs/lfg/REVIEWED-NATIVE-EVALUATION.md)
binds an audited test partition to its training run, ONNX artifact and binary;
it likewise cannot qualify a model without the remaining evidence.

The configurable taxonomy defines the label universe. Per-label counts include
false positives for labels with no positive examples; macro F1 averages only
labels with positive reference support, while micro F1 includes all declared
labels. Empty denominators are `null`. Compare models using the same dataset
and taxonomy hashes and inspect the support for each label. The default
predictions are binary; the reviewed native comparison can optionally measure
empirical ECE and Brier score from uncalibrated native scores using explicitly
configured bins. This does not fit a calibrator or establish quality. Runtime
CPU cycles, memory, quantization loss, and independently reviewed final-test
quality require separate evidence.
See the [exploratory comparison](../../docs/lfg/EVALUATION-SMOKE.md) for a
recorded same-data smoke test and its limits.
