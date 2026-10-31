# Persistent kernel baseline

`benchmark_kernel_persistent.py` runs the actual `kernel-predict` command in one
long-lived, locally spawned process. It accepts a selected JSONL file and an
explicit choice node, sends each text with a fresh request ID, and checks the
returned input hash and snapshot identity. It compares the complete kernel
result for repeated inputs, then reports aggregate latency by whether the chosen
node selected an option, escalated, or was not reached. No labels or quality
scores are inferred. The output contains hashes and aggregate counts but no
request text or per-request results; treat that metadata as potentially sensitive.

```powershell
python ml/lfg-encoder/benchmark_kernel_persistent.py `
  --binary PATH_TO_LFG_BINARY --dataset PRIVATE_JSONL --id-field ID_FIELD `
  --config ml/lfg-encoder/benchmark-persistent.v1.json `
  --snapshot crates/lfg-core/config/kernel-core.v1.json `
  --node operation --output NEW_PRIVATE_REPORT.json
```

The configured warmup and measured rounds repeat the selected inputs; limits and
timeouts are supplied by the benchmark configuration. The command uses a
bounded number of unique generated request IDs and does not create a journal.
The report binds the executable, input, snapshot, benchmark code and optional
operations override by hash, checking them again after measurement. Hashes are
not authenticity or permission proofs. An existing report is never overwritten.
Use trusted local files and a private output location.

The timed request includes local stdio, scheduling and serialization. The first
response includes process startup, with possibly warm filesystem caches. CPU
time and cycles are child-process counters sampled around the measured requests;
on Windows, working set includes mapped and shared memory. Other platforms or
failed counter queries return unavailable values. The benchmark is sequential,
and it does not establish concurrent throughput or model performance.

## Local diagnostic, 2026-10-31

Three sequential runs used the optimized Windows x86-64 binary built with
`embedded-ui,onnx-shadow`, SHA-256
`d06fcc33a5806abb4769246b1d0189104a355915cd628be66cf3cc216c22372e`.
Each used the 11 local, unreviewed LFG requests in the node-review queue, one
warmup round and ten measured rounds. On each run, two distinct requests were
resolved and nine escalated at `operation`. The 110 measurements per run are
repetitions of those 11 inputs, not independent tasks.

| Measurement | Range across three runs |
| --- | ---: |
| Median request latency | 0.066–0.103 ms |
| p95 request latency | 0.097–0.153 ms |
| Process cycles, total for 110 measured requests | 7.18–11.11 million |
| Working set after measurement | 7.78–7.96 MiB |
| Time to first response | 29.6–362.3 ms |

The measured CPU time was 15.625 ms in every run, too coarse for a useful
per-request CPU-time estimate. The machine was not isolated from other work;
the variations, small sample, and repeated inputs limit interpretation. This is
not a production threshold, an independent quality evaluation, or a comparison
with an admitted learned extension. No simulated exemplar pack was used. Raw
aggregate reports remain under the Git-ignored private dataset directory.
