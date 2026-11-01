# LFG model and distillation: owner direction

This is the owner's direction for LFG's model. It is not optional context, and
it is not a description of what exists today; the [current gaps](#what-blocks-this-today)
are listed at the end.

## What LFG's model is

LFG's model is **our own**. It is distilled and fine-tuned on the owner's real
projects and chats so that it *is* LFG: for each prompt it reports the intent,
the complexity, the models suggested and the total agents, with the reasons
behind each. The model targets **about 650 MB on disk** and is bound with the
Rust runtime and UI into **one clean binary**.

We study how Liquid AI makes small models efficient and apply that research to
build our own model. We do not adopt Liquid AI's models, or any other vendor's
model, as LFG. At a 650 MB budget the model is roughly 1.1B parameters at 4-bit
weights or roughly 600M at 8-bit; the architecture search below decides the
actual shape.

## Data and authorization

The owner authorizes using their own projects and chats to train and distill
LFG: project repositories and their journals, Codex and Claude sessions,
checkpoint prompts, and project documents such as DevPulse and RRFlow. The owner
chooses which projects train the model and which are held back to evaluate it.

Private data stays local and out of Git, and secrets are redacted before use.
Marking a chat or project as successful is the owner's attestation that it may
be used and that its outcome is a good example.

## Distillation

Stronger models available through the owner's subscriptions act as teachers.
They label the owner's data with the four results and the reasons for them; LFG's
model is the student.

- **Sequence-level distillation** from teacher outputs works with any teacher,
  including hosted APIs that return text rather than full logits.
- **Logit distillation** follows the LFM2 objective below. It needs teacher
  logits, so it uses teachers that run locally, for example on the GB10 Spark.
- **Several specialist teachers** can be distilled into one student, as in
  LFM3's multi-domain on-policy distillation.

## Research to apply

These are the published methods to learn from. Liquid AI's figures are
vendor-reported.

- **Hardware-in-the-loop architecture search** ([LFM2 Technical Report](https://arxiv.org/abs/2511.23404)).
  Candidate architectures run on the real target devices (llama.cpp Q4_0 and
  ExecuTorch 8da4w on a laptop CPU and a phone). They are measured for
  time-to-first-token, p50/p95 decode latency at batch 1, and peak memory at 4K
  and 32K context. Candidates that break a budget are discarded, and the rest
  are ranked on the quality–latency–memory Pareto frontier using real task
  scores rather than perplexity. **For LFG:** the targets are the owner's
  Windows x86-64 machines and the ARM64 Spark, the size budget is 650 MB, and
  quality is measured on the owner's held-back projects.
- **Hybrid backbone.** LFM2 combines gated short convolutions with a small
  number of grouped-query attention blocks. Convolution layers keep a tiny
  fixed state ("excellent cache behavior on CPUs"), so only the attention
  layers grow with context. The search space also included sliding-window and
  linear attention, state-space models (S4, Liquid-S4, Mamba, Mamba2) and CfC
  liquid time-constant blocks. An [LFM3 analysis](https://local-ai-zone.github.io/blog/lfm3-deep-dive.html)
  reports 13 convolution, 5 sliding-window and 6 global attention layers. It
  also reports a per-head sigmoid gate and four norms per layer, which keep
  activations bounded and so make 4-bit quantization safer.
- **Top-K distillation objective.** LFM2 stores only the teacher's top 32
  logits per token. The loss splits into matching the teacher's total top-K
  probability mass, plus a temperature-scaled match within the top-K set, and
  is balanced against ordinary next-token loss. This avoids the student being
  pushed toward tokens outside the teacher's support.
- **Post-training.** Training data is ordered by difficulty, followed by
  length-normalized preference optimization, then model merging (TIES, DARE and
  similar). Merging also suits combining a user's custom weights with the
  shipped baseline.
- **Real-time serving** ([RIPL](https://www.liquid.ai/blog/low-latency), announced 2026-10-22).
  RIPL puts small models directly in a live request path and returns complete
  responses in under 50 ms at p95 under continuous load. Its listed uses
  include search intent classification. RIPL is closed source and
  GPU-only, and its serving mechanics were not public in the material
  reviewed. The owner's working hypothesis is that RIPL relies on a distinctive
  KV-cache design and custom-organized shards; investigating that is a research
  priority. **For LFG:** adopt its target (complete answers within a fixed p95
  budget under load). Design our own CPU engine around the model's small fixed
  state and organized cache shards (below).

## Organizational knowledge: hitting above its weight

LFG stays small but performs above its size because it absorbs the owner's and
each user's organizational data over time and builds relationships within it.
That means knowing which prompts belong to which projects, which models and
agent counts worked for which kinds of work, and which outcomes were
successful. The kernel journal, the runtime graph builder and the project
outcome journals are the starting points for that record.

Two layers carry this knowledge:

- **Organized cache shards** (fast, updated often). Organizational context is
  grouped into shards, for example per project, domain or team. Each shard's
  attention key/value cache and convolution state are precomputed and stored.
  At request time LFG loads only the relevant shards instead of re-reading the
  context. A short-convolution hybrid makes this cheap, because only its few
  attention layers hold per-token cache and the convolution layers keep a
  small fixed state.
- **Weights** (slow, periodic). As relationships stabilize, they are folded
  into the model through fine-tuning and merged custom weights, so that
  knowledge no longer needs to be loaded at all.

Shard layout, cache format, invalidation when source data changes, and when
knowledge moves from shards into weights are research and design work to
measure, not settled choices.

## Baseline and custom weights

Every user receives LFG's suggestion baseline: the distilled model and LFG's
rules. In their spare time, a user can point LFG at their own chats and
projects that they mark as successful. LFG then trains custom weights for that
user, for example a small adapter merged with the baseline. The baseline stays
intact and the custom weights can be undone.

All of this is configurable: which sources are used, when training runs (for
example, only when the machine is idle), and resource limits. Training can run
on the user's CPU, on a GPU if one is present, or on the network service.

## Confirmation and trust

When combined with SWAG, LFG's assessment is a data-point suggestion. Which
confirmations count, such as another source agreeing, the user approving it, or
a successful outcome, and how much extra weight each earns are configuration.
Extra weight applies only as LFG's measured reliability on the user's
successful work supports it. Evaluation gates decide trust and promotion; they
do not block training on authorized data.

## Local and network use

The same binary runs:

- **locally:** in-process on the user's machine, optimized so this is the
  normal case;
- **as a network service:** clients ping it for assessments and a health
  status.

A client can be told to switch to local when the service is unreachable or
when local meets the latency budget.

## What blocks this today

As of 2026-10-31:

1. `distill.py build-sft` is disabled pending provenance, data-use
   authorization, review and split checks. This document records the data-use
   authorization; enable it for the owner's projects.
2. The label guide limits training data to synthetic or explicitly authorized
   prompts. The owner's projects are now explicitly authorized.
3. The original labels cover intent, capability and risk. A separate assessment
   schema supports complexity, models, effort and agent counts, but the local
   review workspace has no reviewed targets for them.
4. The trainer fine-tunes off-the-shelf encoders (MiniLM, ModernBERT). There is
   no architecture of our own, no hardware-in-the-loop search and no 650 MB
   budget.
5. The fine-tuner is GPU-only. An experimental offline assessment-head trainer
   supports CPU and CUDA, but its weights cannot yet be used by the runtime;
   there is no complete user-side custom-weight training path. The intended
   learned network-service mode does not yet exist.
6. The runtime's answers come from rules split across two classifiers. No
   distilled model is bound into the binary.
7. There is no organizational relationship record, cache-shard store, or
   periodic fold of organizational knowledge into the weights.
