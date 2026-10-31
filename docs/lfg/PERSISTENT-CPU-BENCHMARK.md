# Persistent local CPU comparison

`benchmark_persistent.py` starts two operator-selected prediction commands from
the same LFG executable and keeps both processes alive. It sends one request at
a time over local stdin/stdout, alternating command order for successive inputs.
The versioned configuration controls warmup rounds, measured rounds, dataset and
response limits, and the per-request timeout. It verifies each response's ID,
input digest, producer and stable model identity; it does not compare predictions
to ground-truth labels. Reports contain no prompt text or per-example results,
but hashes and model identities can still be sensitive metadata. Output files
are created exclusively and should stay in a private directory.

For example, select a binary built with `embedded-onnx` and an authorized local
JSONL dataset with `id` and `text` fields:

```powershell
python ml/lfg-encoder/benchmark_persistent.py `
  --binary PATH_TO_LFG_BINARY --dataset PRIVATE_JSONL `
  --config ml/lfg-encoder/benchmark-persistent.v1.json --output NEW_PRIVATE_REPORT.json `
  --baseline-arg=baseline-predict --candidate-arg=embedded-onnx-predict
```

Each argument is supplied separately. Use trusted local executables and avoid
placing credentials in command arguments. The tool times each request from
enqueueing its JSONL input through receiving its response, including local IPC,
Python thread scheduling and serialization. It reports median and nearest-rank
p95/p99; the first response time separately includes process startup and model
loading. The binary is hashed before measurement, so that first response is a
new-process measurement with potentially warm filesystem caches, not a cold
machine or cold-disk measurement. Process counters are sampled after warmup and
after the measured requests. On Windows, CPU time and cycles use the same APIs
as the [whole-process benchmark](CPU-PROCESS-BENCHMARK.md), while current and
peak working set use [GetProcessMemoryInfo](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo).
Working set includes mapped and shared pages; it is not private memory. On other
platforms or failed counter calls, those metrics are reported as unavailable.

## Local experiment, 2026-10-31

Three sequential runs used the same 123,282,944-byte Windows x86-64 binary and
the embedded real-request experimental model described in
[native CPU shadow inference](ONNX-CPU-SHADOW.md#optional-model-embedded-in-the-binary).
Each run used the same 15 attributed, unreviewed DevPulse excerpts: one warmup
round followed by ten measured rounds, or 150 requests per command. The 15
unique inputs were repeated, and their untruncated tokenizer lengths ranged
from 17 to 104 tokens. Both commands used the same binary; their process-start
costs include hashing it for identity.

| Observation | Deterministic baseline | Embedded experimental model |
| --- | ---: | ---: |
| Median response, range across runs | 0.124–0.142 ms | 18.002–19.354 ms |
| p95 response, range across runs | 0.186–0.210 ms | 34.562–39.077 ms |
| Process cycles per measured request, range | 150,643–165,079 | 58.4–62.6 million |
| Working set after measured requests | 6.0–6.1 MiB | 203.2–203.4 MiB |
| Peak working set for process lifetime | 6.0–6.1 MiB | 279.8–279.9 MiB |

The baseline's measured CPU-time delta was reported as zero in these short
intervals despite nonzero cycle counts, so it is not a useful CPU-time estimate.
For the model, total measured CPU time was 2,922–3,063 ms per 150-request run.
First-response times varied from 87–113 ms for the baseline and 386–411 ms for
the model. The runs were not isolated from other machine activity. Earlier
measurements using a prior revision of this benchmark varied substantially,
reinforcing that these are local diagnostics rather than stable performance
guarantees.

These 150 samples are repetitions of 15 related excerpts, not 150 independent
tasks. The model was trained on only 11 unreviewed messages, cannot predict the
full taxonomy, and has not passed a quality or security evaluation. In
particular, these timings do not establish a 512-token p95, a concurrent-user
service result, production memory requirements, or release eligibility. The
normal LFG service still uses the deterministic encoder; this benchmark tests
long-lived offline commands rather than live routing. Raw aggregate reports are
stored under `ml/lfg-encoder/dataset/private/embedded-persistent-benchmark.v5.json`
through `v7.json`.
