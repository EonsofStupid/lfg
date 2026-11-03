# LFG: grounded execution plan

Status: ready for the first bounded implementation task. The executable handoff is
[`handoffs/lfg-developer-feedback-v1.json`](handoffs/lfg-developer-feedback-v1.json).
Grounded against `03078c1170441a7e6027fd4d15a613db1565fd8a`; refresh its anchors
after relevant source changes. The owner's latest direction and `AGENTS.md` take precedence.

## Objective and evidence

Build LFG's **own** compact model and standalone product: classify prompt intent and
complexity, recommend models and agent counts with reasons, and offer a clear UI,
locally or as a network service, optionally integrating with RRFlow and SWAG. The
target is approximately 650 MB for the delivered model, not a claim that a 650-million
parameter model or a 650 MB qualified binary already exists. Keep model choice,
limits, deployment and update policy configurable. Build knowledge from permitted
projects, conversations, feedback and outcomes; do not rebrand another vendor's model.

At the baseline, `crates/lfg-core/src/assessment.rs` supplies the deterministic
assessment used by `crates/lfg-server/src/lib.rs` and the web UI. The owned byte-model
path is experimental and separate from that endpoint. Private review and training
tools already exist under `ml/lfg-encoder`; extend them before inventing another
corpus workflow. On the owner's Spark, a source-continuation job completed 100,000
steps, and a follow-up completed 5,000 steps. The observed follow-up report produced
the required fields in 2/10 cases; it explicitly did **not** evaluate assessment
accuracy. A separate older queue report showed 8/10 on a reused 10-case format set;
the numbers are not comparable quality benchmarks. The queue was idle at inspection.
There were 470 unreviewed teacher proposals and a screened source inventory of 2,082
documents. Screening is not proof all secrets were removed. A previous Windows native
prototype bundled an experimental checkpoint and UI; the UI still called the
deterministic path. A three-run, eight-byte CPU measurement around 4.22 seconds is a
diagnostic, not a release benchmark. Recheck all operational state before acting.

## Sequence and acceptance

1. **Real correction and outcome loop.** Add a private, revisioned assessment feedback
   journal and developer UI to capture what LFG suggested, what an owner corrected,
   provenance, and whether an outcome is unknown or actually observed. Reuse existing
   private review concepts, preserve extra fields, make limits and storage configurable,
   and avoid collecting raw prompt text unless explicitly enabled. Separate local and
   network access; do not expose private feedback without suitable authorization.
   Exercise save, reopen, edit, history and rollback with real API/UI tests. The first
   task in the handoff has exact source anchors and checks.
2. **Owned-model vertical slice.** Expose an explicitly selected experimental owned
   model through the assessment service and developer UI with bounded input/output,
   timeout and explicit unavailable, malformed and truncated outcomes. Keep the
   deterministic result available and identify each engine and checkpoint. Verify
   the actual model path, not a fabricated model response; a model's failure is a
   result to report. Preserve extension fields and configurable runtime limits.
3. **Real data and supervision.** Use the owner's authorized projects and chats
   locally, capture target-by-target owner corrections and confirmed outcomes, and
   recover the authored question/template workflow with provenance. Redact secrets,
   audit samples and record dataset/label lineage. Partition by project or conversation
   before derived examples, deduplicate across splits, and reserve independent evaluation
   projects. Unreviewed teacher proposals remain proposals, not ground truth. Do not
   commit private text, datasets, model weights or secrets. Continue authorized training
   while evaluations are developed; an evaluation gate controls trust, not permission
   to build or train.
4. **Evidence-based evaluation.** Freeze evaluation versions; report examples, coverage,
   unknown counts and per-field intent, complexity, model and agent-count performance
   against actual adjudicated targets, plus calibration and failure categories where
   supported. Compare deterministic baseline, teacher and owned model on the same
   eligible set. Reserve final holdouts from tuning. Record lineage, checkpoint hash,
   commands, runtime, memory and uncertainty. Never turn format compliance into an
   accuracy claim; version and review promotion thresholds with the owner.
5. **Hardware and deployment.** Profile on the target Spark and representative local
   CPUs with actual token/input distributions, cold and warm latency, p50/p95, throughput,
   CPU use, peak RSS, power if measured, artifact sizes and methodology. Explore original
   short-convolution hybrids, distillation, quantization and hardware-in-the-loop search
   against measured quality and latency. The 650 MB goal and single binary must be
   verified for each target; prototype measurements do not establish release readiness.
   Design persistent Spark execution with resource ceilings, resumable checkpoints,
   telemetry, failure reporting and stop controls; distinguish a running service from
   actual progress in training.
6. **Integration and distribution.** Expose versioned, extensible assessment and
   provenance contracts for RRFlow/SWAG and MCP, with explicit authentication and
   authorization for remote use. Support standalone mode. Plan signed model updates,
   integrity verification, compatibility, rollback and opt-in user adaptation; verify
   offline operation and platform packaging. Development UI must show engine, input
   provenance, limitations, timings and supervision state in understandable terms.

The handoff breaks these phases into bounded tasks and can be revised as evidence
changes. A task is complete only when code, integration and test results support its
acceptance criteria. Escalate stale anchors, conflicting requirements, repeated failed
checks, privacy/security decisions beyond the authorized scope and unverified claims.

For the hardware experiment, compare an owned single-forward-pass classifier with
the owned generative path on the *same* reviewed tasks, including the cost and quality
of producing understandable reasons. A single-pass head may reduce decode cost but
does not establish useful explanations or performance on our inputs. Separately test
offline teacher top-k logits and a memory-efficient chunked distillation loss against
online distillation, with compatible tokenization, logged retained probability mass,
private cache provenance and equal held-out targets. Verify throughput, memory and
quality on the actual Spark; published H200 results do not predict Spark speedups.
Measure caller-visible time to first output and end-to-end time separately from
model-only prefill, under documented input lengths, load, warmup and concurrency.

## Routing and improvement

The `clyffy-grounded-delivery` skill uses editable routes: Terra xhigh for grounding,
Luna for a bounded task and Terra xhigh for escalation. Real switching requires the
host's explicit subagent routing or the skill's explicit CLI dispatcher. A hook can
add advisory context, but cannot guarantee obedience or change the parent model.
Hooks require host trust; do not bypass it. Validate the handoff and anchors before
each dispatch, compare the diff to scope and independently review outcomes. For each
actual dispatch record selected model/effort, handoff hash, duration, process exit and
where available actual usage/cost; record unknown values as unknown. Link task-level
acceptance, failed checks, rework and escalations in a private results ledger. Compare
completed real tasks and their total review/rework effort across routes before changing
the routing policy; do not infer quality from speed or process exit.

## Research basis

- [Liquid Foundation Models 2 paper](https://arxiv.org/abs/2511.23404): evidence for
  investigating hybrid short-convolution designs and distillation; a research input,
  not proof of LFG's quality or an instruction to use their weights.
- [Liquid RIPL announcement](https://www.liquid.ai/blog/low-latency): vendor-reported
  sub-50 ms end-to-end results on its stated workloads and infrastructure; treat as
  a latency research target, not a promised LFG or CPU result.
- [Liquid d1 research](https://www.liquid.ai/blog/d1-open): published single-pass
  decision-model approach and device-specific benchmarks motivate a head-to-head
  experiment using an independently built LFG model, without adopting its weights.
- [Offline top-k and chunked distillation study](https://arxiv.org/abs/2608.03796):
  motivates testing cached teacher logits and loss memory tradeoffs; its reported
  hardware and workloads differ from our Spark and require independent validation.
- [Pipette performance methodology](https://pipette.liquid.ai/docs/methodology/performance-methodology):
  distinguishes prefill, time to first output and caller-visible end-to-end latency.
- [NVIDIA DGX Spark known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html):
  verify target-platform constraints and measure the actual configuration.
- [On leakage in code-generation benchmarks](https://arxiv.org/abs/2401.07930):
  motivate contamination checks and project-level independent evaluation.
- [Codex subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents),
  [hooks](https://learn.chatgpt.com/docs/hooks), and
  [skills](https://learn.chatgpt.com/docs/build-skills): host capabilities and limits.
- [OpenAI agent evals](https://developers.openai.com/api/docs/guides/agent-evals):
  measure actual end-to-end agent behavior, not only a model's self-report.

The owner's hypotheses about RIPL's internal KV-cache and sharding are hypotheses;
do not present them as verified architecture. Organizational knowledge storage and
Transformer KV cache have different lifecycle and access requirements; establish
their interfaces and measure the design rather than assuming they are interchangeable.
