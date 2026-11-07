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

## Broader real-request diagnostic, 2026-10-31

Three sequential runs compared the same Windows embedded experimental-model
binary with its deterministic baseline on 422 unreviewed, real project
requests from the private training candidate queue. Each run used one warmup
and two measured passes: 844 measured requests per command, consisting of 422
unique inputs repeated twice. The binary SHA-256 was
`20673c099ad2f722ad686013218b1e579528af65413761a565c71f8a5315202c`.
The requests stayed on the local machine; the reports contain aggregate
timings and identities, not prompt text. The runs were not isolated from other
machine activity and are not a production performance guarantee.

| Observation | Deterministic baseline | Embedded experimental model |
| --- | ---: | ---: |
| Median response, range across runs | 0.127–0.129 ms | 10.04–10.35 ms |
| p95 response, range across runs | 0.184–0.202 ms | 58.39–64.94 ms |
| p99 response, range across runs | 0.255–0.320 ms | 77.10–84.00 ms |
| Process cycles per measured request, range | 141,644–148,436 | 49.19–51.53 million |
| Working set after measurement | 6.2 MiB | 203.1–204.1 MiB |
| Peak process working set | 6.2 MiB | 279.9–280.0 MiB |

The model tokenizer reported untruncated request lengths from 3 to 5,603
tokens, median 25.5 and p95 151. Sixteen of the 422 requests exceeded the
experimental artifact's 192-token maximum; the old binary silently truncated
those requests. The latency comparison therefore covers full inputs for the
baseline and partial inputs for those model requests. The model was trained on
unreviewed development data, lacks the full current taxonomy, and has not
passed accuracy or security evaluation. These measurements are neither quality
evidence nor a release qualification. The three aggregate reports, benchmark
config and token-length summary remain in `dataset/private`.

New native ONNX predictions include processed token count, overflow chunk count
and a truncation flag. The benchmark aggregates these statuses for measured
requests and reports missing metadata as `unknown`. They identify tokenizer
truncation; they do not prove that a model understood the complete request.
The deterministic baseline does not emit tokenizer metadata, so its status is
`unknown` in this tokenization-specific count.
In a local functional check using the same experimental artifact, an updated
debug binary reported a 53-token real request as complete and a 389-token real
request as truncated to 192 processed tokens with two overflow chunks. The
requests remained private and unreviewed. This confirms the flag for those two
cases, not all tokenizer inputs or model quality. A separate two-request debug
benchmark counted one complete and one truncated candidate response; the
baseline's two tokenization statuses were unknown.

An updated local optimized Windows binary with embedded UI and the same
unqualified model had SHA-256
`42942aaf8619cc8755e73e8920a6954f580426bddb6692ff2129ba96f4a4c7d5`
and was 125,399,552 bytes. Its static import table listed Windows system
libraries and no ONNX Runtime or MSVC runtime DLL. A copy served the embedded
UI and developer-statistics endpoint from an otherwise empty working directory,
and ran two real unreviewed inputs without supplying external model files.
This was one local build using existing UI assets, not a clean-host or
reproducibility qualification. The executable contains weights derived from
private project data and is not suitable for distribution.

Three more non-isolated runs used that updated binary and the same 422-request
corpus and settings. Baseline median latency ranged from 0.130–0.144 ms and
p95 from 0.196–0.210 ms; the embedded model's median ranged from 10.33–12.04 ms
and p95 from 60.80–72.56 ms. Baseline cycles per request ranged from
147,425–169,323; model cycles ranged from 49.97–56.38 million. Each run's 844
measured model responses included 812 complete and 32 truncated tokenizations,
representing the same 16 over-limit inputs repeated twice. These are local
resource observations, not quality or release evidence. Build evidence and
aggregate reports remain in the local private output directories.

## Configurable overflow abstention

Three paired, alternating, sequential runs compared `report` and `abstain`
policies on the same 422 real, unreviewed training-source requests, with one
warmup and two measured passes each. Both processes used the same experimental
embedded model and binary; the configuration digest differs by policy. Each
run measured 844 requests: `report` predicted all 844, including 32 truncated
inputs, while `abstain` predicted 812 and abstained on those 32. The 32 are
16 distinct requests repeated twice. These policies do different work and their
outputs are not equivalent.

Across the three runs, median latency was 9.51–9.60 ms for `report` and
8.75–8.95 ms for `abstain`; p95 was 55.52–60.07 ms and 41.01–42.80 ms,
respectively. Process cycles per measured request were 46.88–48.34 million
and 38.82–39.96 million. This local measurement includes IPC and tokenization,
was not isolated, and another Rust test build overlapped the first run. It is
not a quality comparison or a claim about other workloads. The binary still
loads the model at startup, and abstention does not analyze the full input.
The model remains unreviewed and the private-derived binary must not be
distributed. Aggregate reports and runtime configurations remain private.
