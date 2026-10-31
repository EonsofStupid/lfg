# Real-source baseline process benchmark

`benchmark_process.py` feeds the deterministic `baseline-predict` command an
operator-selected JSONL corpus. It constructs a bounded batch with unique
request IDs and runs the release process using settings from a versioned JSON
configuration. Prompt text and labels are not written to its report. The report
identifies the binary, dataset, benchmark code, settings, effective encoder and
any explicitly configured preflight/operations files by digest. It refuses to
overwrite an existing report and checks those inputs again after the runs.

The measured interval covers process startup, input transfer, JSON parsing,
encoding, and output serialization to the null device. Each warmup and measured
run starts a **new process**; warmups can warm operating-system caches but do
not warm a persistent LFG process. On Windows, the child process handle supplies
CPU time through [GetProcessTimes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes)
and cycles through [QueryProcessCycleTime](https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-queryprocesscycletime).
Both counters include the whole process and all its threads, including kernel
work. Other platforms and failed counter calls report unavailable values, not
zeros. CPU time, cycles, and elapsed time are distinct measures; CPU time can
exceed elapsed time when threads run concurrently. Counter resolution and CPU
frequency/scheduling affect comparisons.

On 2026-10-30, the Windows x86-64 release binary recorded in
[the real-source diagnostic](EVALUATION-SMOKE.md#real-source-devpulse-diagnostic)
ran the 12 DevPulse excerpts, repeated 1,000 times per process, with one warmup
and five measured runs. On the reported 24-logical-CPU Intel machine, the median
whole-process elapsed time was **89.4 ms**, CPU time was **78.1 ms**, and cycle
count was **273,876,174** per 12,000-request run. The median elapsed time per
request after amortizing startup and I/O is about **7.4 microseconds**; it is
not a measurement of individual-request latency. The five elapsed times ranged
from 80.8 to 94.8 ms. The report is stored locally as
`ml/lfg-encoder/dataset/private/devpulse-process-benchmark.v3.json` and remains
Git-ignored.

The excerpts are a small, related, unreviewed sample. This run does not measure
p95/p99 request latency, memory, a persistent server, concurrent users, a trained
model, or accuracy. It is not a performance qualification or a target. Compare
future runs on the same machine and corpus with the same methodology, while
separately evaluating output quality and resource use.

For a new private report, adjust the bounded settings as needed and run:

```powershell
python ml/lfg-encoder/benchmark_process.py --binary PATH_TO_LFG_BINARY --dataset PRIVATE_JSONL --config ml/lfg-encoder/benchmark-process.v1.json --output NEW_PRIVATE_REPORT.json
```

The report contains text-derived hashes and model/configuration identities;
keep it with the other private evaluation metadata.
