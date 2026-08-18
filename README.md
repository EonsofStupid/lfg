# LFG

LFG ("Looking For Group") — the CPU prompt-recipe encoder, routing/signal
runtime, and visualizer. Extracted out of `shippin` so it can be developed,
run, and trained on its own.

Rust-authoritative; `GoClyffy` (in a sibling project) still hosts an older
shadow-parity Go implementation kept in sync via
`docs/lfg/SHADOW-PARITY.md` and `fixtures/lfg/shadow-parity.v1.json` until it
can be retired.

## Run locally

Start the server:

```powershell
cargo run -p lfg-server -- serve
```

Start the visualizer:

```powershell
cd ui/master-clyffy-web
npm install
npm run dev
```

The Vite dev server (port 5174) proxies `/api/*` to `http://127.0.0.1:4317`,
where `lfg-server serve` listens by default.

Check activation without consuming a model run:

```bash
curl -sS http://127.0.0.1:4317/api/shippin/intelligence/lfg/status
```

## Encoder

The active encoder today is `lfg_core::CpuRuleEncoder` — deterministic and
rule-based. `ml/lfg-encoder/` is the offline training lane for a future
ONNX-exported encoder (see its own README for the promotion order). It is not
yet wired into `lfg-runtime`.

Benchmark the current encoder:

```powershell
cargo run -p lfg-server -- lfg-bench --iterations 1000
```

## Environment

- `LFG_STATE_DIR` — where signal JSONL history is written (defaults to a temp dir)
- `MASTER_CLYFFY_UI_DIR` — static UI directory `lfg-server serve` falls back to when no dev proxy is in front of it (defaults to `ui/master-clyffy-web/dist`)

## Layout

```text
crates/
  lfg-core/      data model + CPU rule encoder
  lfg-codex/     Codex CLI provider adapter
  lfg-runtime/   signal store, graph builder, experiment/promotion logic
  lfg-server/    axum HTTP host (extracted from shippin-api) + CLI
ui/master-clyffy-web/   the visualizer (React + @xyflow/react)
ml/lfg-encoder/         offline ONNX training lane (not yet wired)
docs/lfg/, fixtures/lfg/  shadow-parity gate vs GoClyffy
```
