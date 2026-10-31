# LFG

LFG ("Looking For Group") — the CPU prompt-recipe encoder, routing/signal
runtime, and visualizer. Extracted out of `shippin` so it can be developed,
run, and trained on its own.

## Project objective

Build a self-contained, CPU-oriented classification system and UI that provide
fast, inspectable observations for an operator or downstream orchestration
system. The recovered LFG v3 notes describe a deterministic five-operation
kernel that produces an address and recipe, with an evaluated suggestion
extension for cases that escalate. Later project requirements include tracing,
model and effort profiles, cloud/local use, and configurable integrations.
See [notes alignment](docs/lfg/NOTES-ALIGNMENT.md) for the design and the gap
between it and the current implementation.

The intended product is an operational assessment and routing overlay for a
larger reasoning workflow. It should visualize task strands and boundaries,
surface intent and routing signals, and preserve a clean signal trail. Its
recommendations are inputs to the operator or downstream agents; producing the
final answer remains their responsibility. LFG's observations do not grant
authorization. The roles and integration boundaries described by the recovered
notes and the current RRFlow repository need reconciliation before integration
can be claimed complete.

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
  the local CPU LLM experiment explore possible improvements. Learned or adaptive
  behavior, including any unsupervised approach, should gain authority only
  after the signal path, evaluation criteria, and safety checks are proven.

Success means an operator can inspect an assessment, understand the signals behind
it, and use it to choose an appropriate reasoning plan; the recommendations must
be evaluated against real tasks before they are trusted. Concrete quality,
latency, and resource thresholds still need to be defined.

Rules, scoring parameters, and model catalog entries are configured rather than
fixed in code. See [LFG configuration](docs/lfg/CONFIGURATION.md) for the
versioned files, project overrides, and current limitations.
The [assessment and update design](docs/lfg/ASSESSMENT-AND-UPDATES.md) records
the intended MCP, project profile, CPU measurement, and artifact-update paths.
The [product and delivery target](docs/lfg/PRODUCT-AND-DELIVERY.md) defines the
self-contained binary, local/hosted use, cloud integration, and developer UI.

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

For a release build that embeds the browser assets in the executable, see
[bundled UI builds](docs/lfg/BUNDLED-UI.md). The UI build is a build-time
prerequisite, not a runtime Node.js requirement.

Check activation without consuming a model run:

```bash
curl -sS http://127.0.0.1:4317/api/shippin/intelligence/lfg/status
```

## Build and deployment

Build the release binary:

```powershell
cargo build --release -p lfg-server
```

Binary path:

- `target/release/lfg-server`

Use the automation targets (Linux/macOS or WSL with `make`):

```bash
make build-binary
make build-ui
make image-build
make image-sbom
make image-run
```

The deployment image installs `python3` and `pip`, builds `lfg-server`, copies
`ml/lfg-encoder`, installs its requirements so the distillation/training
utilities can run inside the container, and bundles the production UI build.

Windows-first deployment flow:

```powershell
.\scripts\deploy.ps1 -Action All
```

That script builds the Rust binary, builds the UI, builds the container image,
runs the container, performs a health smoke test, and generates an SBOM at
`output\sbom\lfg.spdx`.

To generate only an SBOM:

```powershell
.\scripts\deploy.ps1 -Action GenerateSbom -ImageName lfg:latest
```

The script uses `syft` when available and falls back to Docker Scout.

To review a policy baseline update (dry-run):

```powershell
python scripts/refresh-sbom-policy-baseline.py --sbom output/sbom/lfg-policy.json --policy security/sbom-policy.json
```

You can also run a GitHub-hosted dry-run review from
[sbom-baseline-review.yml](d:/projects/lfg/.github/workflows/sbom-baseline-review.yml)
(`workflow_dispatch`). It builds a review image, generates an SPDX JSON SBOM,
applies the current policy gate, and uploads both the SBOM and dry-run baseline
report as workflow artifacts.
On pull requests that touch SBOM policy surfaces, the same workflow runs and
updates a sticky PR comment with the dry-run baseline summary.

To apply a baseline update (explicitly acknowledged):

```powershell
python scripts/refresh-sbom-policy-baseline.py `
  --sbom output/sbom/lfg-policy.json `
  --policy security/sbom-policy.json `
  --write `
  --confirm I_ACKNOWLEDGE_BASELINE_UPDATE `
  --reason "Accepted dependency set after runtime image update"
```

## Release provenance

The container is stamped with OCI provenance labels:

- `org.opencontainers.image.version`
- `org.opencontainers.image.revision`
- `org.opencontainers.image.created`

Release automation is in [release.yml](d:/projects/lfg/.github/workflows/release.yml).
Pushing a `v*` tag builds a versioned image archive, emits an SPDX SBOM, and
attaches both artifacts to the GitHub release.

SBOM drift is enforced using [sbom-policy.json](d:/projects/lfg/security/sbom-policy.json)
and [validate-sbom-policy.py](d:/projects/lfg/scripts/validate-sbom-policy.py)
in both CI and release workflows.
The policy enforces package-count drift limits and approved package ecosystems
(`deb`, `pypi`, `generic`, `rpm`) plus namespace allowlists, so unexpected
package types or sources fail the pipeline.

For signed release tags, use:

```powershell
.\scripts\release-tag.ps1 -VersionTag v0.1.0 -Message "LFG v0.1.0" -AcknowledgeChecklist
git push origin v0.1.0
```

By default, `release-tag.ps1` enforces pre-release checks via
`scripts/pre-release-check.ps1` and requires acknowledgement of
[RELEASE_CHECKLIST.md](d:/projects/lfg/.github/RELEASE_CHECKLIST.md).
Use `-SkipChecks` only for emergency procedures.

## Encoder

The active encoder today is `lfg_core::CpuRuleEncoder` — deterministic and
rule-based. `ml/lfg-encoder/` is the offline training lane for a future
ONNX-exported encoder (see its own README for the promotion order). It is not
yet wired into `lfg-runtime`.

For offline comparison, an optional build can embed a selected experimental
ONNX artifact in the executable. See [native CPU shadow inference](docs/lfg/ONNX-CPU-SHADOW.md);
this does not enable live model routing or qualify an artifact for release.

Benchmark the current encoder:

```powershell
cargo run -p lfg-server -- lfg-bench --iterations 1000
```

## Environment

- `LFG_STATE_DIR` — where signal JSONL history is written (defaults to a temp dir)
- `MASTER_CLYFFY_UI_DIR` — optional external UI directory; overrides embedded assets when supplied. Without the embedded feature, it defaults to the checkout's `ui/master-clyffy-web/dist`.
- `LFG_DEVPULSE_CONTEXT_CONFIG` — optional configuration for reading a local DevPulse provider roster; see [project intake](docs/lfg/APP-INTAKE.md).

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
