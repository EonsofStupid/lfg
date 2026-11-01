# Assessment, model profiles, and updates: target design

This is a design target, not a claim that MCP, CPU-cycle measurement, automatic
profile discovery, remote updates, or RRFlow integration is implemented.

## Assessment chain

An authorized project request enters one bounded CPU analysis. LFG reports a
versioned assessment: intent, complexity, suggested models and total agents,
together with capability and risk signals, the evidence for each, and the
configuration and model identity that produced them. The result includes unknown and inconclusive states. An MCP
adapter should expose the same typed result through a tool with an explicit
input/output schema; it must not create a second classifier or convert an
observation into permission. The local stdio transport is the first candidate;
network transport is a separate configurable integration. The [MCP tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
and [transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
specifications define those protocol boundaries.

When integrated with RRFlow, LFG supplies its assessment as a suggestion. SWAG
accepts it as a data point and weights it more heavily after a confirmation
whose sources and weight are configurable, once LFG's suggestions are proven
reliable. `RrdEngine` retains
authorization, admission, budgets, and durable state. The project configuration identifies the provider
bindings and subscriptions the operator has made available; a claim that a
subscription exists is distinct from a verified usable binding. Unknown or
stale capabilities cannot meet a hard requirement.

## Profiles and selection

Each project needs a versioned profile containing its allowed providers and
model bindings, supported effort values for the actual interface, modalities,
locality, context and output limits, and cost/latency constraints. A separate
updateable model-evidence catalog should record the source, model revision,
interface, observation date, task family, method, sample size, and uncertainty
for every claimed strength or measured result. A project profile references
that evidence; it does not contain credentials. An integrating AI can help fill
the [blank intake template](project-model-profile.template.json), but its
unverified claims remain proposals until reviewed.

Effort labels and model capabilities are provider-, model-, and interface-
specific. Do not assume an API model's settings are automatically available
through a subscription or CLI. [OpenAI's reasoning guide](https://developers.openai.com/api/docs/guides/reasoning),
[Anthropic's effort guide](https://platform.claude.com/docs/en/build-with-claude/effort),
and [xAI's reasoning guide](https://docs.x.ai/developers/model-capabilities/text/reasoning)
are starting sources; profile values must retain their date and be checked
against the actual binding and representative evaluations.

A candidate formula for an evaluated selection policy is:

1. Build the eligible set from authorized, available bindings that satisfy every
   known hard modality, context, locality, budget, and deadline requirement.
   Unknown required values exclude the candidate; an empty set means abstain.
2. For each eligible `(model, effort)` pair, estimate task success `q`, total
   cost per accepted task `c`, and end-to-end latency `t` from comparable
   held-out observations. State sample counts and uncertainty. Missing estimates
   do not become zero or a guessed percentage.
3. If a versioned policy supplies nonnegative weights and valid normalization
   scales, compare `utility = wq*q - wc*(c/c_scale) - wt*(t/t_scale)`. Report
   the inputs, policy revision, uncertainty, and alternatives. The weights,
   scales, and admissibility limits are configuration, not universal constants.
   This is a proposed formula; it is not yet implemented or validated.

The present classifier's `rawCapability` and `toolLoopEfficiency` are only
user-supplied relative scores. They do not implement this evidence model and
must not be presented as calibrated success probabilities.

## Development validation of reviewed assessment targets

`ml/lfg-encoder/assessment_evaluate.py` compares **supplied** baseline and
candidate prediction files against reviewed complexity, model, effort and agent
count judgments. It accepts only a development plan containing separate train
and validation sources, and checks that every record in both splits has a
current review attestation and that the audit finds no within-split normalized
duplicates. It rejects a plan declaring a test source before opening its data.
Project and family separation are still declarations, not proof of independence.

Each newline-terminated prediction file must contain exactly one record per
validation example. Every record identifies the input text digest, current
review annotation digest, and the exact reviewed project-profile digest. It
also carries a consistent claimed producer and artifact SHA-256 across the
file. A record uses this shape, with values drawn from **that record's**
reviewed profile rather than a built-in provider catalog:

```json
{"schema_version":"lfg.assessment-prediction.v1","id":"RECORD_ID",
 "input_sha256":"INPUT_SHA256","annotation_sha256":"ANNOTATION_SHA256",
 "profile_sha256":"PROFILE_SHA256","producer_sha256":"PRODUCER_SHA256",
 "artifact_sha256":"ARTIFACT_SHA256",
 "values":{"complexity":null,"models":null,"agent_count":null}}
```

The SHA placeholders above stand for lowercase 64-character digests. A missing
reviewed profile requires `profile_sha256: null` and all three values null.
Null predictions mean abstention. A non-null model list must be nonempty and
contains `{ "id": "PROFILE_MODEL_ID", "effort": null }` objects; selected
models must have been declared available in that profile, and any effort must
be one of that model's configured values. A non-null list is compared as the
complete recommended set; the current schema does not express a known empty
recommendation. The profile is the one recorded during review, not proof of
current subscription access.

```powershell
python ml/lfg-encoder/assessment_evaluate.py --plan PRIVATE_DEVELOPMENT_PLAN.json `
  --baseline PRIVATE_BASELINE_PREDICTIONS.jsonl --candidate PRIVATE_CANDIDATE_PREDICTIONS.jsonl `
  --max-prediction-bytes OPERATOR_SELECTED_LIMIT --output NEW_PRIVATE_REPORT.json
```

The report counts known and unknown judgments separately. Complexity reports
coverage and accuracy among predictions; agent count reports coverage, exact
matches and mean absolute error; model sets report exact matches and micro
precision/recall, including a recall denominator that counts abstentions.
Effort is scored only for expected models with a known reviewed effort. These
are aggregate, per-record comparisons against each recorded project profile;
they do not prove that different projects' complexity definitions are
equivalent. Unknown targets are excluded, not converted to negative labels or
zero. Small samples and missing classes remain visible in the support counts.

The tool does **not** generate predictions, run a model, prove that prediction
inputs were blinded, authenticate reviewers, or qualify an artifact. Producer
and artifact hashes in the prediction file are claims; the report binds the
file bytes but cannot establish how they were produced. It does not assess the
held-out test split, set promotion thresholds, train these targets or provide
their native runtime inference. An experimental offline trainer now exists;
see [assessment training](ASSESSMENT-TRAINING.md). It produces a private
weight pack, but does not yet generate predictions or provide native inference.

On 2026-11-01 UTC, an optimistic read-only audit of the local training review
workspace found 598 candidates, 569 without even draft labels, and zero current
approvals or known reviewed assessment targets. The workspace is training-only
and was rejected by this validation tool's plan-scope gate. No assessment
metric or training result was generated from it.

## CPU burst efficiency

Measure before optimizing. For a declared corpus, hardware/OS, release build,
configuration, concurrency, warmup, and run count, report total elapsed time,
p50/p95/p99 request latency, throughput, memory, and actual process or thread
CPU time. When reliable cycle counters are available, report
`cycles_per_request = measured_cpu_cycles / completed_requests`, with counter
source and CPU/frequency conditions. `CPU_time_per_request = CPU_time / completed_requests`
is a separate measure. Wall time is neither CPU time nor cycles. Compare changes
with the same corpus and limits, and retain quality and resource regressions
alongside speed. No numerical efficiency target is established yet.

## Updates

Treat binary releases, rule/profile catalogs, and learned weights as separately
versioned artifacts. An update needs an immutable manifest identifying artifact
type, version, digest, compatibility requirements, source and evaluation
evidence. Users need an available-update notice, an inspectable change summary,
an explicit install policy, integrity/authenticity verification, atomic
activation, and a way to roll back. Offline operation remains possible using the
currently installed artifacts. Project-local preferences and secrets are not
overwritten by a global catalog update. No update service or signing mechanism
is implemented yet; the distribution channel and ownership remain to be chosen.
