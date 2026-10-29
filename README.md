# LFG

LFG ("Looking For Group") — the CPU prompt-recipe encoder, routing/signal
runtime, and visualizer. Extracted out of `shippin` so it can be developed,
run, and trained on its own.

## Project objective

Build a self-contained, CPU-oriented binary and UI that help an operator or
downstream orchestration system assess a prompt before execution: how complex
the work is, where it may need to be decomposed, and what model class, agent
count, and reasoning depth may be appropriate. LFG should make that assessment
visible and traceable so it can inform a better execution plan.

The intended product is an operational assessment and routing overlay for a
larger reasoning workflow. It should visualize task strands and boundaries,
surface intent and routing signals, and preserve a clean signal trail. Its
recommendations are inputs to the operator or downstream agents; producing the
final answer remains their responsibility. In RRFlow's target architecture,
LFG supplies optional CPU-derived observations to SWAG, the separate System One
controller; the RRFlow engine retains authorization and execution authority.

### Development path and current status

- **Current foundation:** the active CPU encoder is deterministic and rule-based.
  The Rust runtime includes signal, graph, and evaluation/promotion machinery;
  the UI visualizes the resulting workflow. The full assessment and routing
  experience above is the objective, not a claim that every recommendation is
  already available or validated.
- **First, establish a controlled baseline:** use explicit rules, protocol
  checks, and evaluation gates to establish reliable behavior. Rule-based
  encoding is not itself supervised machine learning.
- **Then, evaluate learned behavior:** the offline classifier training lane and
  future model experiments can explore improvements. Learned or adaptive
  behavior, including any unsupervised approach, should gain authority only
  after the signal path, evaluation criteria, and safety checks are proven.

Success means an operator can inspect an assessment, understand the signals behind
it, and use it to choose an appropriate reasoning plan; the recommendations must
be evaluated against real tasks before they are trusted. Concrete quality,
latency, and resource thresholds still need to be defined.

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
