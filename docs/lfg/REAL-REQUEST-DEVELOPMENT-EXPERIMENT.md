# Real-request development experiment, 2026-10-30

This is an exploratory training and runtime experiment, **not** a reviewed
corpus, a valid independent quality evaluation, or a release candidate. It is
the first recorded run in this repository to train a pretrained encoder using
requests copied from this LFG conversation rather than synthetic fixtures.
The resulting classifier is ineligible for promotion.

## Inputs and boundaries

The private, Git-ignored development dataset contains 11 user messages copied
from the current LFG conversation and 12 excerpts attributed to the operator
in DevPulse project documents. All labels and implied negative labels were
proposed by an AI and remain unverified. Conversation messages can depend on
prior context; the original complete requests behind the DevPulse excerpts and
their authorship were not independently recovered. This is a convenience
sample, not representative project coverage. In particular it has no positive
`sensitive-data` labels and only five of the current taxonomy's labels.

All LFG messages were assigned to one family and all DevPulse excerpts to a
different family. A development split selected 11 LFG messages for training
and 12 DevPulse excerpts for validation. The seed was chosen to obtain this
project separation, not as evidence of a statistically representative split.
The same validation excerpts selected the best checkpoint; they cannot serve
as a final test or unbiased quality estimate. Makers Impulse was not supplied
to this run and remains reserved for later external-project evaluation; its
previously inspected examples are not a blinded final test.

The local input is
`ml/lfg-encoder/dataset/private/lfg-devpulse-development.v1.jsonl`, SHA-256
`6a5e0fbdd5caee3162f85524d1aaf29f24cfd7af39c9673cb79e068c71cbfe2a`.
The private assembly script and a `pip freeze` snapshot are stored alongside
it. These files contain or describe private project data and must not be
committed. The source messages are present in this conversation; an
independently exportable transcript has not been established.

## Training and runtime

On Windows x86-64, the run used the cached pretrained
`sentence-transformers/all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The pretrained encoder was loaded
locally without network access, with a newly initialized classification head.
The cached base `model.safetensors` SHA-256 was
`53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.
The run used CPU PyTorch 2.14.1, Transformers 4.57.6, a two-thread environment,
seed 730, three epochs, maximum length 192, batch size two, and checkpointing
every six steps. None of the 23 inputs exceeded 192 tokens. It explicitly used
`--allow-below-gate`; all 23 records were unverified and the 500-human-reviewed
sample gate did not pass. No Spark or GPU training occurred in this run.

The run manifest, checkpoints, tokenizer, and 90,872,572-byte weights are in
`target/lfg-real-conversation-dev-v1/`. The final weights' SHA-256 is
`21dd690709aa19a96265456c279674ecce14351e547e91c2c78aaa85e00ce5e8`.
The manifest captures package versions and inputs, but the inherited Python
environment was not built from an exact dependency lock or immutable image;
this run is traceable, not demonstrated reproducible on another machine.

The local command, shown for tracing rather than as a production recipe, was:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:TOKENIZERS_PARALLELISM = 'false'
target/lfg-training-venv/Scripts/python.exe ml/lfg-encoder/train.py train `
  --dataset ml/lfg-encoder/dataset/private/lfg-devpulse-development.v1.jsonl `
  --base-model sentence-transformers/all-MiniLM-L6-v2 `
  --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 `
  --output target/lfg-real-conversation-dev-v1 --seed 730 --epochs 3 `
  --max-length 192 --batch-size 2 --checkpoint-steps 6 --allow-below-gate
```

A repeat must choose a new output directory and restore the same input,
model files, code, dependencies, and settings. Matching those inputs still does
not establish bitwise reproducibility.

The model was exported to
`target/lfg-real-conversation-onnx-v1/` with manifest SHA-256
`f28f78e72442df23c46fce129c50c8121f2976eb28c6c9104274fe7db22c22e8`.
On those 12 validation excerpts, the Python exporter compared 24 batch-one
and batch-two cases across token lengths 28–144: maximum absolute logit
difference was `8.94e-8`, with zero threshold-decision mismatches. The Rust
native CPU shadow binary also matched Python decisions on all 12 excerpts.
This proves agreement on this sample, not model accuracy or broad ONNX shape
coverage. The export used the legacy tracing path and produced a tracer warning.
The weights remain separate from the binary, and live routing is unchanged.

## Diagnostic comparison

All 12 validation excerpts were provisionally labeled `build.software` and
`filesystem.write`. At the configured 0.5 threshold, the rule baseline emitted
each of those labels for two excerpts; the trained candidate emitted each for
none. These are disagreements with provisional labels, not verified errors.
The set has no positive security labels, no reviewed negatives for these
categories, and no independent evaluation; it cannot establish security recall,
generalization, or whether either system is fit for use.

With the same feature-enabled Windows binary and 12 excerpts, one warmup and
three sequential new-process runs gave median whole-process elapsed times of
29.9 ms for the rule baseline and 722.3 ms for the trained native candidate.
Median process CPU time was 15.6 ms versus 656.3 ms, and median process cycle
counts were 92.2 million versus 2.108 billion. These figures include process
startup, artifact loading, input parsing, inference, and output serialization;
the runs were not interleaved or isolated. They are not persistent-server
latency, p95/p99, memory, or a performance qualification.

Private per-example predictions, parity results, and benchmark reports remain
under `ml/lfg-encoder/dataset/private/`. Before a serious model can be trained
and qualified, obtain representative authorized requests and diverse reviewed
labels, retain a genuinely independent test, lock the training environment,
and perform quality, security, and CPU measurements against the baseline.
