# Operational configuration plan

Status: implemented. Windows verification covers
the Rust workspace, strict Clippy, UI tests/build, and a server-process test
with non-default settings. Other platforms and the sibling Go implementation
have not been tested in this pass.

## Goal

Move the remaining operator-tunable limits and live-UI polling settings for the
current LFG server into a validated, versioned configuration. One effective
configuration must drive server admission, replay, event streaming, and UI
behavior. The server must expose its revision and digest so a caller can tell
which settings produced a result.

## Scope and boundaries

- Configure default listen address and benchmark iteration limits, maximum prompt size in **UTF-8 bytes**, maximum HTTP request-body
  bytes, default and maximum signal page sizes, event broadcast capacity, stream
  keepalive interval, UI debounce interval, and UI timing-history length.
- Ship an editable JSON baseline embedded in the binary; allow an explicit
  project file through `LFG_OPERATIONS_CONFIG`. Load once at startup. An invalid
  explicit file fails before opening the signal store or binding a port.
- The runtime enforces the same prompt and page limits as the HTTP surface.
  The UI reads effective values from status and measures UTF-8 bytes; it does not
  silently substitute a different limit. An oversized prompt is not submitted.
- Apply the configured HTTP body limit at the JSON extractor. Body bytes include
  the JSON envelope and escaping, so the configured body limit is independent
  of the decoded prompt limit; clients can receive a body-size rejection first.
- Preserve protocol versions, hash-chain semantics, validated configuration
  files, explicit command-line overrides, and the existing provider boundary.
  Fixed protocol identifiers, mathematical identities, and visual style choices
  are not project policy. The benchmark's bundled prompt is an editable fixture
  with a `--prompt-file` override. The benchmark remains a smoke test, not CPU
  qualification.
- Validation must reject zero or inconsistent limits, invalid UI timer ranges,
  and capacity values that the broadcast implementation cannot represent.
  Configuration changes do not hot-reload and do not authorize actions.

## Verification

1. Unit tests cover malformed settings and runtime prompt/page enforcement.
2. A real server process uses non-default settings for classification, preflight,
   request-body and replay limits; invalid files fail before state creation.
3. UI tests verify non-default debounce and UTF-8 size behavior, including a
   multibyte prompt, plus runtime status loading. Run the Rust workspace suite,
   strict Clippy, UI tests and UI build.

## Still separate

Provider model evidence, per-project binding discovery, signed update
distribution, CPU cycle measurements, MCP transport, full self-contained UI
packaging, and RRFlow/SWAG integration remain separate work. This pass does not
claim a trained model, calibrated scores, or end-to-end CPU optimization.
