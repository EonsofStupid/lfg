# Developer observability

The inspector view displays process-local development statistics for the
preflight, classification and assessment handlers, the rules each assessment
fired, and what LFG has learned about models from `GET /api/lfg/knowledge`. It
offers browser-session controls for
analysis debounce and displayed history, bounded by the server's configured
`devTools` ranges. Reset returns to server defaults; these controls do not write
or change project configuration. They can help compare behavior but do not by
themselves establish an optimal setting.

The `devTools` object in [`operations.v1.json`](../../crates/lfg-runtime/config/operations.v1.json)
controls collection, the trace ring capacity, the statistics polling interval,
and the range of UI tuning controls. The bundled configuration enables
collection. Set `statsEnabled` to `false` in an `LFG_OPERATIONS_CONFIG` override
and restart to disable it. Existing operations files without `devTools` remain
valid and disable the developer tools. Settings are loaded at startup.

`GET /api/lfg/dev-stats` returns a no-store snapshot of the bounded in-memory
ring, lifetime recorded/evicted counts, recent per-operation samples, errors,
and nearest-rank p50/p95 elapsed times. It includes the effective operations,
preflight encoder, and classifier digests to identify the active configuration.
The sequence identifies a trace within this server process; it is not a
distributed trace ID. Restart discards the ring and resets its counts.

Only operation, outcome, timestamp, elapsed microseconds and, for assessments,
the IDs of the rules that fired are retained—no
prompt text, session IDs, provider content, or credential values. Preflight
traces separate time waiting for the runtime lock from handler work;
classification has no runtime-lock queue measurement. Elapsed time starts after
HTTP decoding and ends before response serialization and network transmission.
Malformed requests rejected by the HTTP extractor and other endpoints are not
counted. The trace store itself adds overhead; polling can also contend for the
runtime lock. Percentiles use the most recent retained traces, include errors,
and are not stable performance claims on small samples.

CPU time and hardware cycle counters are not yet measured: `cpuTimeMicros` is
`null`, and the UI says unavailable. Optimizing CPU burst efficiency requires a
separate benchmark on declared hardware with a fixed corpus, warmup, concurrency,
quality checks, and actual CPU/cycle measurement. The current statistics are
useful for locating latency changes, not proving CPU efficiency.

## Keep local diagnostics

Inspector's **Download diagnostic snapshot** saves the currently retained
statistics to a JSON file. It explicitly selects the documented fields; it
does not download the prompt or the complete UI state. The result is a point
in time, not an ongoing log.

To record periodic snapshots while developing, run the same LFG executable in
a second terminal. Choose the server address, a **new** local output file and
limits appropriate for your machine:

```powershell
.\lfg-server.exe dev-log --server 127.0.0.1:4317 --output .\lfg-dev-log.jsonl `
  --interval-ms 2000 --timeout-ms 3000 --max-response-bytes 1048576 --max-bytes 10485760
```

The values above are examples, not fixed performance settings. Only loopback
addresses are accepted. The logger polls at the selected interval and writes
only changed snapshots, so idle polling does not fill the file. Each line has
schema `lfg.dev-log-entry.v1`, a capture time and one validated
`lfg.dev-stats.v1` snapshot. The logger rejects unknown
snapshot fields, unexpected operation/outcome values, malformed digests and
oversized responses. It creates a new file without overwriting an existing one,
flushes each entry, and stops before the next entry would exceed `--max-bytes`.
Stop it with Ctrl+C. If the server becomes unavailable or returns an incompatible
snapshot, the logger stops with an error; prior flushed entries remain. Start a
new file to resume.

For compatibility with the separate running assessment prototype, the logger
accepts its assessment operation and assessment configuration digest, but
omits its optional rule-detail field from the saved trace. This avoids copying
potentially identifying configuration values into the log.

Snapshots repeat the retained ring, so their traces overlap. Events evicted
between polls are lost; `evicted` and `totalRecorded` describe each server
snapshot. A restart resets these counters and the current protocol does not
provide a unique process identity, so the log is not an audit or distributed
trace. File permissions and retention are the operator's responsibility;
timing and activity metadata can still be sensitive. Logging is opt-in and does
not transmit data to RRFlow or a cloud service.

The server currently has no authentication layer on this endpoint. Its default
listen address is loopback; if configured to listen on a network, the operator
must account for exposure of timing and activity metadata. No remote export,
distributed tracing, profile comparison, automatic server-side persistence, or
server-side configuration editor is implemented here.

## Owned model run measurements

The experimental owned-assessment route is separately protected by its
loopback, peer, Host/Origin and session-token checks. Successful and handled
failure responses may include `lfg.owned-run-measurement.v1`. Its
measurement object's status, engine ID, and stop reason repeat the bounded
identifiers in the ordinary response. `handlerElapsedMs` starts after Axum has
extracted the request body and ends before response serialization and context
digest attachment. It is handler wall time and may include child-process
startup; it is not CPU time or model-only latency. Body-limit rejections made
by the Axum extractor have no measurement. `outputTextUtf8Bytes` counts only
output text present in the response. A `null` means that count is unavailable.
Worker start and successful completion come from the actual child spawn/wait
path. `workerCompletedSuccessfully: false` means the child did not successfully
finish, including a killed timeout; `null` means unknown.
The engine identities and raw proposal status remain in the ordinary owned
response. No prompt, project context, output text, tokens, or private run
history is added to `/api/lfg/dev-stats`.

## Paired HTTP assessment benchmark

`ml/lfg-encoder/benchmark_paired_assessment.py` runs a bounded, sequential
loopback comparison of `/api/lfg/assess` and the protected
`/api/lfg/assess/owned` route. The same selected UTF-8 text is sent to both;
the owned request has no project context, and deterministic declarations are
operator-configured (an empty list is valid). Warmups and measured rounds
alternate request order, with one request in flight at a time. The benchmark
does not tune, train, or evaluate proposal quality.

Create a JSON config with the required `lfg.paired-http-benchmark-config.v1`
fields below; there are no implicit defaults. Unknown extension fields are
accepted but ignored by benchmark behavior. The config document itself is
limited to 64 KiB. Configured record, round, and byte limits are used directly
without an additional fixed dataset-size ceiling, subject to platform-safe
integer arithmetic. The config, screening policy, and dataset must be outside
the repository. `dataset_path` names one explicit file; no directory or glob
is expanded. The configured split must be
`train`, `training`, `dev`, or `development`; validation and heldout split
names are not accepted. Split and `dataset_source_ref` are operator claims,
not independently verified. A development split claim is not a final-holdout
classification or a quality evaluation. The tool applies the configured screening policy to selected rows
before sending the same screened text to both paths. The source and screened
ID/text digests and redaction counts are reported without the text. The policy
digest is recorded; screening matches are not proof that all sensitive data
was removed. The tracked `ml/lfg-encoder/private-corpus-screen.example.json`
is a starting point to copy and review into a private policy file. Use reviewed
patterns because Python regular expressions do not provide hard per-match CPU
limits. The service target must be a
literal loopback IP over HTTP. Supply the high-entropy session token through
the named environment variable or a separately protected file; never place it
in the URL, config, or command line.

```json
{
  "schema_version": "lfg.paired-http-benchmark-config.v1",
  "target_base_url": "http://127.0.0.1:8090",
  "owned_token_env": "LFG_OWNED_SESSION_TOKEN",
  "owned_token_file": null,
  "dataset_path": "C:/private-data/train.jsonl",
  "dataset_split": "train",
  "dataset_source_ref": "owner-selected:train",
  "screening_policy_path": "C:/private-data/screen-policy.json",
  "selection_seed": 42,
  "max_records": 100,
  "max_source_records": 10000,
  "max_dataset_bytes": 16777216,
  "max_input_bytes": 65536,
  "max_request_bytes": 131072,
  "max_response_bytes": 1048576,
  "max_report_bytes": 1048576,
  "client_timeout_seconds": 35,
  "server_timeout_overhead_ms": 5000,
  "warmup_rounds": 1,
  "measured_rounds": 3,
  "declared_subscriptions": [],
  "first_order": "deterministic_first"
}
```

Run it with `python ml/lfg-encoder/benchmark_paired_assessment.py --config
<private-config.json> --output <new-report.json>`. The report path must be in
an existing, private directory outside the repository; creation is exclusive
and permissions are checked. Reports contain status counts and denominators,
latency summaries, request settings, digests, and service identities, but no
prompt text, project context, response text, tokens, or raw response bodies.
Client timeout or ambiguous transport failure aborts further measurement and
marks contamination unknown.

End-to-end timing starts immediately before opening the HTTP request, after
request JSON and headers are prepared, and ends after reading a bounded
response body, including bounded HTTP error bodies. The deterministic
`durationMicros` and owned `handlerElapsedMs` are separate endpoint-reported
handler times. The owned timer starts after Axum body extraction and ends
before response serialization and context-digest attachment; it may include
child startup. These values are not CPU or model-only time. The current HTTP
contracts do not attest the server binary hash. CPU, hardware cycles, and
memory remain unknown because this tool does not sample verified isolated
intervals. These measurements are not a quality result or promotion evidence.

Both owned routes set `Cache-Control: no-store`, including authorization and
body extraction errors. Inspector shows a side-by-side current-input proposal
comparison. Deterministic
rules use the request without optional project context; the owned model may
receive that additional context. The different framing is stated in the UI,
and the comparison does not establish equivalent-input quality. Edits to the
request, context, or subscription declarations clear the owned result and its
feedback association. Owned outputs remain unverified proposals.
