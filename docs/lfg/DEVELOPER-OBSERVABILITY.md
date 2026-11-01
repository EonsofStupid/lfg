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

The server currently has no authentication layer on this endpoint. Its default
listen address is loopback; if configured to listen on a network, the operator
must account for exposure of timing and activity metadata. No remote export,
distributed tracing, profile comparison, persisted diagnostics, or server-side
configuration editor is implemented here.
