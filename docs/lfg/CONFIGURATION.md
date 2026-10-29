# LFG configuration and project profiles

LFG's current rule-based preflight and category classifier use versioned JSON
configuration. The bundled files are starting configurations, not trained or
calibrated claims. They are embedded so the binary can start without fetching a
configuration; a project can provide its own file and restart the process to
apply it. Configuration changes do not currently hot-reload or distribute
automatically.

| Setting | Bundled file | Override |
| --- | --- | --- |
| Preflight rules, categories and agent/branch bounds | [`preflight.v1.json`](../../crates/lfg-core/config/preflight.v1.json) | `LFG_PREFLIGHT_CONFIG` |
| Classification taxonomy, scoring, thresholds and model catalog | [`classifier.v2.json`](../../crates/lfg-core/config/classifier.v2.json) | `LFG_CLASSIFIER_CONFIG` |
| Runtime provider inventory and explicit replay fixtures | [`providers.v1.json`](../../crates/lfg-runtime/config/providers.v1.json) | `LFG_PROVIDER_CONFIG` |
| Semantic evaluation rubric | [`evaluation-rubric.v1.json`](../../crates/lfg-core/config/evaluation-rubric.v1.json) | `LFG_EVALUATION_RUBRIC` |
| Runtime promotion thresholds | [`promotion-policy.v1.json`](../../crates/lfg-core/config/promotion-policy.v1.json) | `LFG_PROMOTION_POLICY` |

Copy a file to a project-controlled location, modify it, and set the relevant
environment variable to its path before starting the server. The benchmark uses
the same preflight override. Invalid, missing, or unsupported explicitly
configured files cause startup to fail rather than silently using defaults.
Version 1 classifier configuration files do not satisfy the v2 contract; update
them explicitly. Preflight responses now use `lfg.preflight.v2`: uncalibrated
confidence is `null`, and the encoder abstains from a confidence claim.

The default model catalog is empty. For each project, populate the configured
catalog with the provider and model IDs actually available, subscription ID,
applicable effort levels, stated strengths, configured scores and cost tier,
and evidence source, basis and observation date. These entries are declarations,
not proof that a subscription is active or that a model meets a task's needs.
The [model profile template](project-model-profile.template.json) is a blank
intake form for an operator or integrating AI; it is deliberately not a live
configuration. It must be reviewed and translated into a catalog entry after
the unknown fields are resolved.
The current UI lets an operator mark configured subscription IDs as available
for the session; it does not discover or authenticate subscriptions. Do not put
credentials in these files. An empty or unavailable catalog produces no model
recommendation. An invalid routing priority also produces no recommendation.
The separate runtime provider inventory defaults to a configured Codex adapter
and replay-only fixtures; those entries do not prove an active subscription.
An override may be empty. Configured entries cannot declare themselves active,
and a requested fixture binding is omitted if no fixture supports the task's
required capabilities.

Classifier category percentages are normalized rule scores, not probabilities.
The current token estimate is a whitespace-delimited word count, not a model
tokenizer count. `durationMicros` is elapsed wall time, not CPU time or cycle
count. The default thresholds, inferred agent counts, and scoring weights are
heuristic and need evaluation before they can support quality claims.

The classifier returns the effective configuration revision and digest; the
preflight encoder manifest binds its rule configuration. Project configuration
is loaded once at startup, so changes require restart. Model profiles are not a
complete project subscription system: an editable per-project profile flow,
provider-specific verified capabilities and effort settings, update signing and
distribution, measured CPU-cycle telemetry, MCP exposure, and RRFlow/SWAG
integration remain to be built and tested.

This configuration pass does not yet cover every operational setting. The
current server and runtime still contain fixed input-size and signal-page
limits, and the UI still contains fixed polling and display settings. Those
need an explicit configuration and compatibility pass before the project can
claim that all operator-tunable behavior is configurable. Protocol constants
and mathematical operations are separate from operator-tunable policy.

An experiment cannot mix semantic grades produced by different rubric hashes;
changing the rubric requires a new experiment (or restoring the original
rubric to inspect the prior experiment's statistics). The separate offline
encoder training policy in `ml/lfg-encoder/promotion-policy.json` is not this
runtime promotion policy and remains a distinct experimental gate.
