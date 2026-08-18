# LFG shadow-parity gate

MasterClyffy in Shippin is the Rust authority. GoClyffy remains a compatibility client until both sides pass their immutable protocol fixtures and the shared behavioral projection below.

| Gate | GoClyffy v0.2 | Rust v0.3 |
| --- | --- | --- |
| Provider-neutral agents | `internal/lfg/protocol_test.go` | `lfg-core` validation + parity fixture |
| Plan committed before provider binding | happy-path JSONL fixture | hash-chained runtime test |
| Unknown or invalid streams fail closed | Codex JSONL tests | `lfg-codex` JSONL tests |
| Raw prompt absent from control ledger | service tests | JSONL reopen/privacy tests |
| Semantic promotion requires samples and independent graders | evaluation tests | runtime and HTTP promotion-gate tests |

The shared projection is [`fixtures/lfg/shadow-parity.v1.json`](../../fixtures/lfg/shadow-parity.v1.json). It compares stable behavior, not byte-identical envelopes; v0.3 intentionally adds RROs, recursive content-key rejection, and the canonical Rust graph reducer.

Run the gate with:

```text
cargo test --workspace
(cd ../GoClyffy && go test ./...)
```

Do not retire the Go path until both commands pass on the same revision and real traffic has run in shadow mode without a contract-level divergence.
