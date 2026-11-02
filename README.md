# LFG

LFG ("Looking For Group") — the CPU prompt-recipe encoder, routing/signal
runtime, and visualizer. Extracted out of `shippin` so it can be developed,
run, and trained on its own.

## Project objective

LFG is a tuned model and a standalone product. For each prompt it reports four
results: the **intent**, the **complexity**, the **models** suggested, and the
**total agents** the work needs, with the evidence behind each. It ships as a
self-contained, CPU-oriented binary with its own UI. It is useful on its own
and can be combined with SWAG and RRFlow.

LFG's model is our own. It is distilled and fine-tuned on the owner's real
projects and chats, targets about 650 MB, and is bound with the runtime and UI
into one clean binary. We apply Liquid AI's published efficiency research
(hardware-in-the-loop architecture search, short-convolution hybrids, Top-K
distillation, RIPL's real-time latency targets) to build our own model rather
than adopting theirs. The same binary runs locally on the user's machine or as
a network service that clients ping. It hits above its weight by
incorporating organizational data over time: relationships between projects,
prompts, models, agents and outcomes, held in organized cache shards and folded
into its weights by fine-tuning. See
[model and distillation](docs/lfg/MODEL-AND-DISTILLATION.md).

Every user gets LFG's suggestion baseline. In their spare time they can have
LFG train on their own chats and projects that they mark as successful, so it
adapts and creates custom weights for itself. Both the baseline and the
adaptation are configurable.

When combined, SWAG accepts LFG's assessment as a data-point suggestion. What
counts as a confirmation, and how much extra weight a confirmed suggestion
earns, are configurable. The extra weight applies only once LFG's suggestions
have been shown to be reliable on real tasks. RRFlow keeps authorization and budgets; an LFG suggestion never grants
permission or dispatches work by itself, and LFG does not produce the final
answer.

The UI must be easy for a non-specialist to read and configure. For each prompt
it shows, in order: the prompt received, the formula applied to it, the
brackets the prompt fell into, the agents and models suggested, and the
subscriptions recognized. AI agents that manage LFG need the same operations as
clear, documented steps.

The recovered LFG v3 notes describe a deterministic five-operation kernel that
produces an address and recipe, with an evaluated suggestion extension for
cases that escalate. See [notes alignment](docs/lfg/NOTES-ALIGNMENT.md) for the
design and the gap between it and the current implementation.

### Development path and current status

- **Current foundation:** runtime answers come from deterministic rules. The
  [one assessment](docs/lfg/ASSESSMENT.md) returns all four results with the
  formula behind them, and can derive provisional model listings and usage
  counts from local Codex and Claude files. A listing or past use does not prove
  current subscription entitlement or a successful outcome. The older preflight and category
  classifiers remain for existing integrations and can disagree with it.
- **Building the model:** `ml/lfg-encoder/` contains experimental training
  tools. The intent, capability and risk fine-tuner requires a GPU; an offline
  experimental assessment-head trainer can use CPU or CUDA with a frozen
  locally supplied encoder. Its weight pack has not been trained on reviewed
  user assessments and cannot yet produce runtime predictions. A plan-driven
  distillation builder prepares selected teacher traces, but there is no
  integrated assessment SFT trainer or qualified distilled model. The
  [model and distillation](docs/lfg/MODEL-AND-DISTILLATION.md#what-blocks-this-today)
  direction lists what must change. The assessment-head trainer currently
  requires reviewed train and validation records; the separate SFT preparation
  path allows explicitly selected unreviewed teacher traces. A trainer for those
  traces remains work to be done. Evaluation gates
  decide when a model version is trusted.
- **Product:** the self-contained binary, embedded UI, configuration and
  integrations are tracked in the [product and delivery target](docs/lfg/PRODUCT-AND-DELIVERY.md).

Success means a person can read an assessment, understand why LFG suggested
those brackets, agents and models, and change the configuration that produced
it, and that SWAG can rely on the suggestion once it is proven. Concrete
quality, latency, and resource thresholds still need to be defined.

Rules, scoring parameters, and model catalog entries are configured rather than
fixed in code. See [LFG configuration](docs/lfg/CONFIGURATION.md) for the
versioned files, project overrides, and current limitations.
The [assessment and update design](docs/lfg/ASSESSMENT-AND-UPDATES.md) records
the intended MCP, project profile, CPU measurement, and artifact-update paths.
The [product and delivery target](docs/lfg/PRODUCT-AND-DELIVERY.md) defines the
self-contained binary, local/hosted use, cloud integration, and developer UI.
The [developer-to-alpha execution plan](docs/lfg/DEVELOPER-TO-ALPHA-PLAN.md)
tracks the work from the current developer preview through real supervised
training, standalone qualification and RRFlow/SWAG integration.
The [integration and training research](docs/lfg/INTEGRATION-AND-TRAINING-RESEARCH.md)
reconciles that plan with the current LFG, RRFlow and SWAG implementations and
the next independently testable delivery packages.

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

Inspector can download its current diagnostic snapshot. To keep bounded local
timing and outcome logs in a second process, see
[developer observability](docs/lfg/DEVELOPER-OBSERVABILITY.md#keep-local-diagnostics).
These diagnostics do not include prompts or establish model quality.

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
