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
