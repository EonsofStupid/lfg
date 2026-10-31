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
accepts it as a data point and weights it more heavily after a second
confirmation, once LFG's suggestions are proven reliable. `RrdEngine` retains
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
