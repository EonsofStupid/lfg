# LFG: evidence anchors

Read this before choosing model, data or product work. It ties each decision to
something that has worked elsewhere and to one measure here.

## Why this exists

From 18 August to 3 November 2026 the project recorded 61 measured results. None
was scored against human review or a verified outcome, so no run could show that
one version was better than the last. A reviewed held-out test was a stated next
step in plans dated 29, 30 and 31 October and 1, 2 and 3 November. Of the 60 PRs
merged from #99 to #162, about 5 were mainly measurement.

## The rule

Before starting, name:

1. **The anchor.** A row under [anchors](#anchors), or a new row with its source.
   If nothing qualifies, add it to [bets](#bets) with the test that would settle it.
2. **The scorecard number** it should move.

Afterward, report that number against the same rows, including when it doesn't move.

This decides trust. It does not block training or building. Preprints and vendor
numbers can justify running a test, but they can't establish a plan. Agreement with
teacher proposals, format checks and how a build feels in use are not quality results.

## Scorecard

[Step 3 of the order of work](MODEL-AND-DISTILLATION.md#order-of-work) defines the
per-version report. These parts are fixed:

- **Targets, in the order they become available.** First, realized effort per typed
  prompt: tool calls, tokens and time, including follow-on turns. Next, what actually
  ran and what happened: model used, subagents started, work merged or reverted. Then
  owner review, for intent and for cases outcomes can't settle. About 100 reviewed
  examples measure a rate to within 10 points at 95%.
- **Rows.** Constant, prompt length, rules, hashed-text linear model, opponents,
  teacher, dev model, owned model. A model has to beat the cheap rows and the
  opponents to earn its cost. Write predictions and fix each system's one output
  before scoring, and include random-score and shuffled-target rows that must sit
  near 0.
- **Splits.** Whole projects unseen, and later turns. Intervals come from resampling
  whole projects. The reserved test project stays closed until a version is frozen.

### First result

Measured 3 November 2026 on 1,274 typed prompts from local Claude Code and Codex
logs, 22 September to 3 November, across 40 working directories. Each number is the
rank correlation with realized tool calls, within each tool, with a 95% interval.

| Row | All prompts (1,274) | 40+ characters (891) |
| --- | --- | --- |
| Prompt length | +0.06 (-0.03 to +0.14) | +0.25 (+0.15 to +0.31) |
| Rules | +0.12 (+0.04 to +0.19) | +0.33 (+0.22 to +0.40) |
| Hashed text, whole project unseen | +0.34 (+0.27 to +0.43) | +0.33 (+0.28 to +0.42) |

- **The untuned rules carry signal on substantive prompts.** Median tool calls by
  bracket, trivial to major, run 7, 11, 21, 28, 50 in Claude Code and 23, 34, 46, 28,
  35 in Codex. Rules beat prompt length by 0.10 (+0.05 to +0.15) on those prompts.
- **Short follow-ups are the gap.** 30% of prompts are under 40 characters, and they
  depend on conversation context the rules don't see.
- **The project offset is promising, not shown.** On later turns, adding a shrunken
  offset to text moved the correlation from +0.29 to +0.33, and the interval on that
  change spans zero (-0.02 to +0.12).
- **Agent count is mixed.** It tracks subagents started at +0.25 in Claude Code and
  about 0 in Codex.
- **Model suggestions can't be scored yet.** One model ran 94% of the Claude Code
  prompts, Codex split between two by choice rather than at random, and neither log
  records whether the work succeeded.

Limits: this is one person's logs and not a held-out test. Tool calls measure work
done, not whether it was right, and depend on each agent's habits. The teacher, dev
model and owned-model rows aren't added yet. For scale only, on other data and another
target, developers reached 0.50 and Claude Sonnet 4.5 reached 0.44 on 1,000 JIRA tasks
([Anthropic's own figures](https://www.anthropic.com/research/estimating-productivity-gains)).

### Opponent panel

Measured 3 November 2026 on 1,270 of those prompts, redacted first. The machine running
opponents never held the outcomes. Predictions and each system's output were fixed before
scoring. Each number is the same rank correlation; intervals run about ±0.08 at 40+
characters and ±0.15 under 40.

| Row | Size | All (1,270) | 40+ characters (887) | Under 40 (383) |
| --- | --- | --- | --- | --- |
| Hashed text, fit on unseen projects | about 130 KB of weights | +0.34 | +0.33 | +0.35 |
| [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), frozen, ridge fit on the same folds | 133 MB | +0.34 | +0.32 | +0.31 |
| [Clef-flash](https://huggingface.co/Cloudflare/clef-flash), one fixed `score` question | 19 GB | +0.24 | +0.39 | +0.07 |
| [Laya](https://huggingface.co/convaiinnovations/laya) `multilingual`, Clef-flash's question | 644 MB | +0.24 | +0.21 | +0.24 |
| [NVIDIA classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier), `prompt_complexity_score` | 736 MB | +0.18 | +0.22 | +0.21 |
| Laya `typed-decisions`, tuned on other work, same question | 843 MB | +0.17 | +0.23 | +0.10 |
| [NadirClaw DistilBERT](https://huggingface.co/nadirclaw/nadirclaw-distilbert), P(complex) | 268 MB | +0.13 | +0.27 | +0.01 |
| Rules | none | +0.12 | +0.33 | -0.17 |
| [Skala](https://huggingface.co/TakamaruAI/skala), calibrated complex score | 23 MB | +0.06 | +0.19 | -0.09 |
| Prompt length | none | +0.06 | +0.25 | -0.23 |

- **Nothing clears hashed text on any split**, by a paired 95% interval that resamples whole
  projects. On all prompts hashed text beats Clef-flash by 0.11 (+0.01 to +0.24). Better
  embeddings add nothing: bge minus hashed is -0.00 (-0.05 to +0.06).
- **Rules beat the three small purpose-built gauges at 40+ characters**: NVIDIA by 0.12, NadirClaw
  by 0.07, Skala by 0.15, each with an interval above zero. Clef-flash's lead over rules there,
  +0.05 (-0.04 to +0.15), is not established.
- **The closest rival in size ties the largest on all prompts.** Laya's 644 MB build, untuned for
  this work, matches Clef-flash at +0.24 (difference -0.00, -0.11 to +0.11) and beats it on short
  prompts by 0.17 (+0.03 to +0.32). On 40+ characters it is clearly behind Clef-flash by 0.18 and
  rules by 0.14. This is one question on one target, so it doesn't rank them generally.
- **The bar for a 650 MB model** is the best row on each split: +0.39 at 40+ characters, +0.35
  under 40, and +0.34 on all prompts. No row leads all three.
- **Rules go negative on short prompts** because they read short as small. Median tool calls
  under 40 characters against 40+ are 13 and 14 in Claude Code, 42 and 23 in Codex.
- **The written decision applies:** stop adding text models to the effort output and put the
  work into session and mid-run signals. LFG's own model hasn't been scored on this panel, and
  this measures effort only, not intent or reasons.
- **Audits:** random and shuffled rows sat within noise of 0, and a second implementation of
  the metric agreed to 3 decimals. Of nine written predictions, five held.

Limits: one person's logs, now seen, so not a held-out test. Ten rows on three splits with no
correction for multiple comparisons. Confirm any gain on prompts logged after this date. Latency
on a Spark, one prompt at a time, p50: Skala 2 ms on CPU, bge 9 ms on CPU, NadirClaw 11 ms on CPU,
NVIDIA 20 ms on CPU, Laya 9 ms and 20 ms on GPU, Clef-flash 213 ms on GPU.

**Locality has to be proven, not assumed.** ONNX Runtime 1.31.1 sends telemetry to
`mobile.events.data.microsoft.com` by default, from native code, past a Python-level network
block, and `disable_telemetry_events()` did not stop it. `ORT_DISABLE_TELEMETRY=1`, set before
import, did, and `ml/lfg-encoder` now sets it at every import. Private runs use a kernel-level
block and report outside connections. The web UI serves its own fonts, so it makes no outside
requests.

## Anchors

**A** means shipped at real scale, usually with the operator's own numbers. **B** is
peer-reviewed or independently replicated. **C** is a single lab or a vendor. **D** is
nothing found. Sources were read through a summarizing tool, so check a figure at its
source before publishing it.

| Decision | What worked | What went wrong | So LFG |
| --- | --- | --- | --- |
| Keep the large model off the typing path | **A** [Gmail Smart Compose](https://arxiv.org/abs/1906.00080): p90 under 60 ms; shipped an LSTM although Transformers scored better | **B** [Choosing predicted words correlated with slower typing](https://userinterfaces.aalto.fi/typing37k), 37,370 typists, observational | Rules and a hashed linear model per keystroke; large model at pause or submit |
| Size | **A** [Apple's on-device model](https://arxiv.org/abs/2507.13575): about 3B at 2 bits per weight by quantization-aware training; MMLU 67.8 to 64.4, IFEval 85.1 to 82.3 | **C** [1-bit Bonsai 4B](https://huggingface.co/prism-ml/Ternary-Bonsai-4B-mlx-2bit) fits at 0.57 GB but scores 62.7 against 77.1, by the vendor's table | About 2.6B at 2 bits is the budget with precedent; 3–4B is a bet. Train at low bit depth. 4 bits, as in step 5, holds about 1.3B |
| Distillation | **A** [Hinton et al.](https://arxiv.org/abs/1503.02531): about 700M examples, distilled 60.8% against ensemble 61.1%, from an ensemble to one model rather than large to small | **B** [Imitation models](https://arxiv.org/abs/2305.15717) rated competitive by crowd workers while closing little of the gap on benchmarks | Hundreds of teacher proposals can train a head or adapter, not transfer capability. Never judge by how output reads |
| Actions from a fixed catalog | **A** [Alexa](https://arxiv.org/abs/1804.08064): 97.5% on 20 domains, 92.5–93.8% on 1,500 after reranking, with millions of utterances | **B** [HINT3](https://arxiv.org/abs/2009.13833): real bots with 328–600 training queries reached 48–84% in scope, against 90+ on clean benchmarks | Small catalog, shortlist, explicit "none of these". Code does the work, and deterministic gates decide what may fire |
| Decision logging | **A** [Decision Service](https://arxiv.org/abs/1606.03966): over 25% CTR gain at MSN; records context, action and probability at decision time; a wrong logged probability distorted train-test results 3.0x | **A** [Azure Personalizer](https://learn.microsoft.com/azure/ai-services/personalizer/what-is-personalizer) asked for about 1,000 events a day and has been retired. **D** for single-user learning | Log candidates, order, seed and decision now. Learn across pooled use, not from one user's bandit |
| Complexity and effort | **B** [COCOMO II Bayesian calibration](https://doi.org/10.1109/32.799958): expert priors plus data, within 30% of actuals 75% of the time, on structured inputs | **B** [Deep-SE replication](https://arxiv.org/abs/2201.05401): beat Median in 8 of 42 cases and TF/IDF-SVM in 9 of 32 | Coarse ordinal with stated uncertainty. Fit from the current weights as the prior. Learned models must beat the cheap rows |
| Project influence | **A** [LinkedIn GLMix](https://www.kdd.org/kdd2016/subtopic/view/glmix-generalized-linear-mixed-models-for-large-scale-response-prediction): global plus per-entity coefficients, 20–40% more job applications. [Credibility theory](https://doi.org/10.1017/S0515036100008989) in insurance | **A** [A fixed split point lost performance](https://www.casact.org/abstract/recent-review-and-changes-ncci-individual-risk-experience-rating-plan) in workers' compensation rating as claims grew | Global formula plus a project offset weighted by n/(n+k), with no switch at a threshold |
| Growth per project | **A** [Meta CodeCompose](https://arxiv.org/abs/2305.12050): fine-tuned on internal code, 1.4–4.1x over public-only training, 16K developers | Fine-tuned Copilot models were [reported discontinued](https://github.com/orgs/community/discussions/161278), no official date found. [OpenAI is winding down fine-tuning](https://developers.openai.com/api/docs/guides/distillation). **D** for a per-project fine-tune with published benefit | Fine-tune pooled. Grow per project through retrieval and offsets first |
| Suggest rather than switch silently | **C** [Cursor Router](https://cursor.com/blog/router): about 60% lower cost in online tests, no sample sizes given | **A** for the event: [ChatGPT's router was removed for free users](https://the-decoder.com/openais-gpt-5-router-rollback-shows-why-ai-requires-unlearning-old-habits) after reportedly hurting daily use, from an anonymous source | Recommend, and the person keeps control. Count latency and cache-miss cost |
| Showing benefit | **A** [Field experiments](https://papers.ssrn.com/abstract=4945566): 4,867 developers, 26.08% more tasks (SE 10.3%) | **C** [METR](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/): 19% slower while believing 20% faster. Its [2026 update](https://metr.org/blog/2026-02-24-uplift-update/) calls its newer data "only very weak evidence" | Dogfooding finds bugs, not benefit. Randomize the overlay per session to measure it |
| Learning from use | **A** [Alexa self-learning](https://arxiv.org/abs/1911.02557): defect rate down over 30% in production, from what people did next, pooled | **A** [GPT-4o rollback](https://openai.com/index/expanding-on-sycophancy/): thumbs feedback weakened the primary reward while offline tests and A/B results looked fine | Cap "marked successful". Delayed outcomes and fixed behavior tests decide trust, and training continues |

## Bets

These have no shipped precedent yet. They are worth testing and must not be described
as proven.

| Bet | Evidence so far | What settles it |
| --- | --- | --- |
| A 650 MB typed classifier rivals Clef-flash | [Laya](https://huggingface.co/convaiinnovations/laya) has a 644 MB, 322M build, and its 421M build went from 0.362 near chance to 0.766 after fine-tuning, on 2,000 decisions, by its own table. Untuned for this work, the 644 MB build [ties Clef-flash on all prompts](#opponent-panel) and loses on 40+ characters. [d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M), 587M, scored 15.95 against Clef-flash's 57.1 on a self-run Decision Index | A row at 650 MB or smaller that beats every opponent on unseen projects, on each split |
| 3–4B capability in 650 MB | Vendor numbers only | A scorecard row for a frozen low-bit base |
| Adapter on a 1-bit base | A [ternary method](https://arxiv.org/abs/2608.24469) works on native ternary models, untested on Bonsai | Base against base plus adapter on unseen projects |
| Per-project weights | Nothing shipped with published benefit; [Code2LoRA](https://arxiv.org/abs/2606.06492v2) on one language and task | Project learning curve against retrieval plus offset |
| KV cache as the knowledge store | [RIPL's page](https://www.liquid.ai/blog/low-latency) never mentions a cache; an 8K-token shard takes about 100–350 MB | Break-even tests for the cache role only |
| A small local model choosing actions by ID | Shipped systems use purpose-built classifiers; [small-LLM results](https://arxiv.org/html/2603.00030) needed an RTX 4090 | Latency and wrong-action rate on target CPUs |
| A prompt overlay helps people | No in-situ test; nothing on non-technical users | Randomized on and off in real use |
| Pre-rendered, AI-chosen UI | Nothing published | Instrumented display-only panels |
