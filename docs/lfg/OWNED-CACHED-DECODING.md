# Owned student cached decoding experiment

`ml/lfg-encoder/owned_cached_decode.py` is a Python reference for incremental
inference with the experimental, independently initialized `OwnedByteStudent`.
For each layer it retains the last `conv_kernel - 1` convolution inputs; for
attention layers it also retains the keys and values for the request. Sessions
have independent state, take one byte per batch item, and reject inputs beyond
the configured context. Because this model uses absolute position embeddings,
the implementation does not pretend that dropping old cache entries is
equivalent to recomputing a sliding window. The model must remain in evaluation
mode with unchanged weights for the lifetime of a session.

This is not a Rust runtime, cache sharding, quantization, or production serving.
The current Python reference concatenates growing attention tensors at each
step; a native implementation should benchmark bounded preallocation and its
memory costs rather than assume this implementation is optimal. There is no
prompt-level task quality or concurrent-service measurement here.

## Separate ONNX prefill and decode graphs

`ml/lfg-encoder/export_owned_cached_onnx.py` exports a versioned experimental
artifact with separate `prefill.onnx` and `decode.onnx` graphs. The prefill
graph accepts `tokens` as `int64 [batch, sequence]` and returns the last-token
`float32 [batch, 256]` logits plus cache tensors. The decode graph accepts one
`int64 [batch]` token, a rank-zero `int64` absolute `position`, and those cache
tensors; it returns next-token logits and the updated caches. Convolution state
is named `conv_{layer}`; attention key and value state are `key_{layer}` and
`value_{layer}`. Decode outputs add `_out` to each state name.

The `manifest.json` declares the state order, dtype, shape, per-layer capacity,
empty attention-state shape, and the maximum cache tensor bytes per batch. The
consumer must validate byte token values, batch/rank/dtype/fixed dimensions,
prefill length, `position < context`, and that each attention cache length
equals `position`. The ONNX graphs do not add a serving wrapper that checks
those relationships. The model uses absolute positions: the caller increments
the absolute position, retains all attention history, and rejects context
overflow. There is no cache eviction. The cache byte figure excludes model
weights, allocator overhead, temporary graph tensors, and runtime buffers; it
is not a process-memory cap.

Export with a verified private checkpoint and already-framed local source
bytes. Keep the export configuration, sources, measurements, ONNX files, and
checkpoint outside Git:

```powershell
python ml/lfg-encoder/export_owned_cached_onnx.py `
  --checkpoint PRIVATE_CHECKPOINT_DIRECTORY `
  --config PRIVATE_EXPORT_CONFIG.json `
  --source PRIVATE_FRAMED_SOURCE.bin `
  --additional-source ANOTHER_PRIVATE_FRAMED_SOURCE.bin `
  --output NEW_PRIVATE_OUTPUT_DIRECTORY `
  --decode-steps 4 --warmup 0 --repeats 3
```

The config loader accepts at most 1 MiB; model, checkpoint, and manifest sizes
use their explicit configuration limits. Export verifies checkpoint weights
and source model code against the checkpoint manifest, checks that all inputs
remain unchanged during export, checks ONNX files, and compares configured
prefill lengths and each measured decode step with both full-prefix PyTorch and
the Python cache reference. It records artifact/input hashes only in the
private report and emits no prompt or generated text. `promotion_eligible`,
`native_runtime_checked`, and quality evaluation are false in the manifest.

One private local experiment used the owner-authorized 573-byte and 603-byte
framed development inputs with the actual private checkpoint (context 4,352,
width 512, eight layers). ONNX Runtime 1.31.1 with one CPU thread checked
prefill lengths 1, 4, 32, 64, and 573, plus four greedy decode steps for each
input, against both references. Greedy bytes matched exactly; maximum absolute
logit and cache differences were 1.82e-5 and 1.05e-5 respectively. Median
prefill times were 388.6 ms and 367.0 ms. Median four-byte decode batches were
34.4 ms and 30.8 ms (8.60 ms and 7.70 ms per byte). This run used no warmup and
three repeats. It did not measure a full-prefix timing baseline, evaluate
assessment quality, or establish a reliable speedup. These measurements are a
single local CPU case and must not be generalized to other hardware or loads.
For this checkpoint and full context, the declared cache tensor upper bound is
71,335,936 bytes per batch item. It does not include other runtime memory. The
raw private experiment report and inputs remain outside Git.

Run the local benchmark against an existing private checkpoint and an
authorized local source prefix:

```powershell
python ml/lfg-encoder/benchmark_owned_cached_decode.py `
  --checkpoint PRIVATE_CHECKPOINT_DIRECTORY --source PRIVATE_SOURCE_FILE `
  --length 64 --threads 1 --warmup 3 --repeats 20
```

The values shown are an example, not a runtime policy. The script verifies the
weights and model implementation against the checkpoint manifest, compares
cached logits with full-prefix logits, alternates the measurement order, and
reports median and nearest-rank p95 elapsed time for processing **all** prefixes
of the selected sequence. It records implementation, checkpoint and input-prefix
digests. These digests and learned weights can be sensitive; keep the report and
checkpoint local. This is a microbenchmark on a single request, not end-to-end
latency or throughput under concurrent load.

On 2026-11-02, a local Windows CPU run used the prior private 37,888-parameter
multi-document distilled checkpoint and the first 64 bytes of the owner's LFG
README. With PyTorch 2.14.1+cpu, one thread, three warmups and 20 alternating
repeats, the largest cached-versus-full-prefix logit difference was 0.00000191.
Full-prefix median/p95 elapsed time for the 64 positions was 46.23/59.19 ms;
cached median/p95 was 39.83/47.10 ms. The machine was not isolated, the sample
is small, and these results are neither a statistically established speedup nor
an assessment of the target-sized model. The source is already part of the
development material, so this is not an independent quality evaluation.
