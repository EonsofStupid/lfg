# Developer observability

The inspector view displays process-local development statistics for the
preflight and classification handlers. It offers browser-session controls for
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

Only operation, outcome, timestamp, and elapsed microseconds are retained—no
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
