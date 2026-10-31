# ModernBERT ONNX compatibility probe

On 2026-10-31, the [locked Spark environment](SPARK-LOCKED-ENVIRONMENT.md)
exported the pinned ModernBERT-base snapshot with a newly initialized, untrained
13-label classification head. No project prompts or training records were used.
This is a structural compatibility probe, not training data, a quality result,
a performance benchmark, or a release candidate. The output manifest explicitly
marks it `probe_only` and ineligible for promotion; no model weights are committed.

The configured legacy PyTorch exporter produced a self-contained opset 17 ONNX
graph. The export used one CPU thread, seed 731, and the pinned model revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`. The snapshot manifest SHA-256 was
`4a126906b3785c898995a2a78b0748ee84c14b3aeca157c9e943332cd61e22a6`.
The [probe manifest](evidence/modernbert-onnx-probe.v3.manifest.json) has SHA-256
`bccc965f4fe6adb317d46a84567ff396f6b200cc2e2a64e4020a6c702ec64f2a` and records
the inputs, output file digests, and observations. ONNX Runtime CPU and PyTorch
agreed within the configured tolerances for 1×1, 1×8, padded 2×32 and 1×512
artificial token shapes. The largest absolute logit difference was 0.00002230;
all threshold decisions matched on those shapes. The exporter issued legacy and
tracing warnings, so these finite observations do not prove correctness for all
inputs or shapes.

An isolated aarch64 `lfg-server` release build from repository commit
`4c24b1543e2ffa3be460db4520308b7615bb2c58` used Cargo's locked dependencies,
Rust 1.97.1 and the `onnx-shadow` feature. The binary SHA-256 was
`5882911bde187eb581bba9f9d5d509b4420024e5ca4a1f49cd9be7ed29435a79`.
It loaded the graph with LFG's tract runtime and processed artificial unknown-token
strings. Against ONNX Runtime, the 3-, 61- and 512-token cases matched all 13
threshold decisions; the largest absolute sigmoid-score difference was
0.00000686. This is a native CPU shadow-path check on one ARM64 machine, not a
trained or embedded binary, cross-platform qualification, or latency benchmark.

The checked-in probe scripts reproduce these checks using operator-selected
paths and settings. From the repository root, with the locked Python environment
active and the verified model snapshot available, an example is:

```bash
python ml/lfg-encoder/probe_untrained_onnx.py \
  --model MODEL_SNAPSHOT --model-manifest MODEL_MANIFEST.json \
  --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
  --taxonomy ml/lfg-encoder/evaluation-taxonomy.v1.json \
  --prediction-config ml/lfg-encoder/prediction-config.v1.json \
  --export-config ml/lfg-encoder/onnx-export-config.v1.json \
  --output NEW_PRIVATE_PROBE_DIRECTORY --max-length 512 --seed 731 \
  --shape 1:1 --shape 1:8 --shape 2:32 --shape 1:512

python ml/lfg-encoder/probe_native_untrained_onnx.py \
  --artifact NEW_PRIVATE_PROBE_DIRECTORY --manifest-sha256 OUTPUT_MANIFEST_SHA256 \
  --binary PATH_TO_LFG_SERVER --runtime-config crates/lfg-server/config/onnx-runtime.example.json \
  --unknown-repetitions 1 --unknown-repetitions 30 --unknown-repetitions 510 \
  --score-tolerance 0.0001 --timeout-seconds 240
```

The second script checks the manifest's probe-only marker and compares native
scores and threshold decisions with ONNX Runtime. It does not perform model
evaluation or authorize promotion. Reviewed real examples, independent splits,
actual training and evaluation, resource and latency measurements, and release
verification remain necessary.
