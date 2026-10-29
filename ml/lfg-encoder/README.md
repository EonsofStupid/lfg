# LFG encoder experiment

This is the offline, non-authoritative training lane for the Rust
MasterClyffy CPU encoder. Runtime prompts are not collected here. The checked-in
corpus is intentionally authored synthetic data.

These experiments support the assessment and routing objective in the root
[README](../../README.md). The classifier and CPU LLM paths are exploratory;
neither is currently the authoritative runtime encoder. The coding-oriented
teacher and student work below also explores general model capabilities, and
its contribution to LFG's assessment role needs to be demonstrated through
task-specific evaluation before integration or promotion.

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
   training; do not use it to tune the model. The current random split in
   `train.py` is only a development split and does not enforce these protections.
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
The `--allow-below-gate` flag is for development only. Without it, training
requires the policy's minimum number of records explicitly marked `verified`.

## CPU LLM lane (2B -> ~650MB target)

If your goal is a true local CPU model in the 450-650MB range, this repo now
supports a practical setup path that starts from a 2B Hugging Face base model,
then converts and quantizes for CPU execution.

### Recommended objective

- Start point for LFG code specialization:
  `Qwen/Qwen2.5-Coder-1.5B-Instruct` (recommended default in this repo).
- Optional teacher stage for stronger coding behavior:
  a 7B-class coder model used for supervised data generation/distillation.
- Deployment shape: GGUF + llama.cpp for CPU-only runtime.
- Size target: ~650MB with aggressive quantization (Q2/IQ2 family), or
  ~900MB-1.3GB with higher quality quantization (Q3/Q4 family).

### Reality check on model size

- A "2B model at 650MB" is achievable, but only with aggressive quantization.
- For better quality at ~650MB, a common strategy is:
  1) train/distill from the 2B teacher,
  2) deploy a smaller student model (roughly 0.5B-1B) at a higher-bit quant.

That second path usually gives stronger quality per byte than forcing a 2B
model into very low-bit quantization.

### Build script

Use the new script:

```powershell
.\build_cpu_llm.ps1 -ModelId "Qwen/Qwen2.5-Coder-1.5B-Instruct" -QuantPreset iq2_xs
```

Script file: `ml/lfg-encoder/build_cpu_llm.ps1`

What it does:

1. Downloads a Hugging Face model snapshot locally.
2. Converts HF weights to F16 GGUF.
3. Quantizes to your selected preset.
4. Emits size and run-command hints for CPU execution.

### Prerequisites

- Python 3.10+
- Hugging Face CLI (`pip install huggingface_hub[cli]`)
- A local llama.cpp checkout with built binaries
  (`llama-cli` and `llama-quantize`)

Example llama.cpp build on Windows:

```powershell
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -S . -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release
```

### Quantization presets (expected range, 2B class)

- `q4_k_m`: best quality baseline, usually above the 650MB target.
- `q3_k_m`: middle ground.
- `q2_k`: near the target size, quality trade-off.
- `iq2_xs`: smallest preset in this script, strongest quality trade-off.

Exact file size depends on tokenizer, architecture, and conversion details.

## Cutting-edge training shape (recommended)

For the roadmap you described (7B teacher -> 2B/1.5B student -> ~650MB CPU artifact),
use this order:

1. Teacher setup (7B coder): generate high-quality instruction/solution/rubric
   traces for code tasks, tests, and repair loops.
2. Student SFT: train the deployable student (`Qwen2.5-Coder-1.5B-Instruct`
   baseline, or a true 2B student when selected) on those traces.
3. Preference alignment: run pairwise ranking or DPO-style passes on
   correctness and tool-use behavior.
4. CPU export + quantization: convert to GGUF, then test `q4_k_m`, `q3_k_m`,
   `q2_k`, and `iq2_xs`.
5. Promotion gate: choose the smallest artifact that passes your eval floor
   (task pass-rate, compile/test success, latency, and memory ceilings).

### KV-cache and runtime techniques to prioritize

- Paged KV cache for long sessions.
- Prefix cache reuse across repeated prompts/tool scaffolds.
- Sliding-window attention policy for bounded memory growth.
- Quantized KV cache where supported by your runtime stack.
- Speculative decoding only if net CPU throughput improves in benchmark.

## Distillation scaffold (7B teacher -> deployable student)

Use `distill.py` to validate teacher traces and build SFT-ready JSONL data:

```powershell
python3 distill.py init-template
python3 distill.py validate-traces --traces dataset/teacher-traces.v1.jsonl
python3 distill.py build-sft --traces dataset/teacher-traces.v1.jsonl
```

Script file: `ml/lfg-encoder/distill.py`

### Teacher trace schema

Each JSONL record is a single object with:

- `traceId` (unique string)
- `prompt` (task/request text)
- `response` (teacher answer/code)
- `language` (`rust|typescript|python|go|javascript|shell|mixed`)
- `difficulty` (`easy|medium|hard`)
- `checksPassed` (boolean; true when tests/checks passed)
- `sourceModel` (teacher model identifier, e.g. your 7B run)
- `tags` (string list for routing/eval grouping)

`build-sft` emits chat-style records (`messages` + `metadata`) split into:

- `dataset/sft-train.v1.jsonl`
- `dataset/sft-eval.v1.jsonl`

## Automation targets

From the repository root, you can run:

```bash
make ml-install
make ml-validate
make ml-distill-template
make ml-distill-validate
make ml-distill-build
```

For containerized deployment, `make image-build` uses the root
`Dockerfile`, which installs Python 3 and the dependencies in
`ml/lfg-encoder/requirements.txt`.
