# One assessment

`POST /api/lfg/assess` returns LFG's four results for a prompt in one response:
intent, complexity, total agents and suggested models, with the formula behind
each. It is the only assessment the default UI shows. The older preflight and
category classifiers remain for existing integrations and can disagree with it;
new work should build on this one.

The separately versioned [integration observation](OBSERVATION-CONTRACT.md)
projects this analysis without consulting local subscription history and
accepts opaque extra request fields. It is an advisory interface. RRFlow's
draft engine integration calls it; SWAG and MCP don't yet.

These are rule outputs, not calibrated probabilities. `confidence` stays `null`
until measured evidence can support a number. An operator can bind a
[trained model](TRAINED-LEVEL-SCORE.md) whose work level chooses the bracket;
the response then carries a `modelScore` block beside the rule score.

## The formula

Complexity is a sum of points. The weights are in
[`assessment.v1.json`](../../crates/lfg-core/config/assessment.v1.json).

| Counted | Meaning |
| --- | --- |
| Kinds of work | Intents matched by the shared preflight rules plus `extraIntentWords` |
| Separate tasks | Clauses that contain an action word |
| Length | Words divided by `wordsPerLengthPoint`, up to `lengthCap` |
| Wide-reaching words | Matches from `scopeWords`, up to `scopeCap` |
| Sensitive content | Risk rules matched |

The score selects a bracket. Each bracket sets the agent range, the model role
it suits and a target effort. Total agents is the number of tasks, kept within
the bracket's range. Words match whole words with the configured endings, so
`test` matches `tests` but not `latest`. The older classifiers match substrings.

## How models are chosen

No model list is maintained by hand. LFG learns what is available and how it is
used from the provider tools' own files, and refreshes while it runs.

| Source | What it gives |
| --- | --- |
| `~/.codex/models_cache.json` | Codex's list: models offered, descriptions, rank, effort levels |
| `~/.codex/sessions/` | Codex prompts, with the model and effort used for each |
| `~/.claude.json` | Claude plan and extra model options |
| `~/.claude/projects/*/` | Claude prompts, with the model and effort used for each |
| Explicit Codex sign-in check | Whether Codex is signed in when the user requests a check |
| `publishedModels` in the configuration | Optional operator-supplied entries; the bundled configuration has no hand-maintained model roster |

A model's role comes from its provider's description, using `roleRules`. A model
whose description says nothing inherits from a sibling in the same family, so
`gpt-6.1-terra` takes its role from `gpt-5.6-terra`. Hidden models and models a
provider list no longer offers are never suggested. A model seen only in old
session history has unknown current availability and is not treated as offered.
Subagent sessions, such as
Codex guardian reviews, are not counted as the user's choice.

Models listed by a provider or configured by the operator are scored only when
the user has declared the corresponding subscription available. A sign-in
check, local history or provider list does not independently verify a paid
entitlement; a user declaration also remains unverified.

```text
score = (1 - usageInfluence) × fit + usageInfluence × habit
```

- **fit** is how close the model's role is to the role the bracket suits.
- **habit** is the model's share of the user's similar requests, with older use
  counting for less (`usageHalfLifeDays`).

`usageInfluence` is 0.3 by default, so a fast model can win a trivial request even
when the user habitually sends those to a frontier model. The result says so.
Set it higher to follow habit more closely, or set `effortFollowsHabit` to use the
effort the user usually picks. Ties go to the provider's own ordering.

Prompts are selected by the same rules as the private inventories in
`ml/lfg-encoder/inventory_codex_sessions.py` and `inventory_claude_sessions.py`.

## Privacy

Prompt text is scored in memory and dropped. Model and subscription metadata,
dates and counts are kept in memory; this collection does not transmit them.
LFG parses `~/.claude.json`
while retaining only the plan and model option fields; the whole file is read by
the parser, but other account fields are not retained. It does not open separate
credential files.
Traces record the IDs of the rules that fired, never prompt words.

Learning starts with `lfg-server serve`. `LFG_LEARNING=off` disables it. A server
bound to a non-loopback address does not learn unless `LFG_LEARNING=on`, because
`GET /api/lfg/knowledge` would expose the summary to anyone who can reach it.
`CODEX_HOME` and `CLAUDE_CONFIG_DIR` change where it looks.
Background refresh does not launch the Codex CLI; sign-in checks happen only
when explicitly requested through the existing provider status workflow.

History discovery and incremental reading are bounded by the `knowledgeScan`
settings in the assessment configuration. File and directory quotas are shared
between configured sources, and the byte budget rotates across discovered files.
When a quota, oversized line, retained-state limit, incomplete write, read error
or metadata limit prevents full collection, the knowledge state is `partial`.
The limits are operator configuration, not measured performance targets. A
partial or missing provider source is not evidence that a subscription or model
is currently available. File shrinkage and modified same-size files reset their
cached counts; a longer replacement with unchanged initial content may not be
detected by this prototype and should not be treated as verified provenance.

## Known limits

- It reads the prompt alone. A short follow-up such as "yes, do it" rates as
  trivial even when it continues large work.
- Habit shows what the user chose, not what succeeded. Outcomes are the next
  step in the [order of work](MODEL-AND-DISTILLATION.md#order-of-work).
- Bracket thresholds and weights are starting values, not fitted to data.
- Committing a plan still uses the preflight agent count, so the committed graph
  can differ from the assessment.
- Claude has no sign-in check yet, so it shows as found rather than signed in.
- Roles depend on provider wording. A new description with none of the
  `roleRules` words leaves the role unknown until a rule covers it.
