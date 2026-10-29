# LFG classifier labeling guide, draft v1

This guide proposes how to label the current classifier dataset. It is a draft
for human review, not an established taxonomy. LFG's broader target includes
complexity, language, modality, and task category; the current classifier schema
only covers intent, capability, and risk labels. It cannot yet train model-class,
agent-count, or complexity predictions.

## Unit of annotation

One record is one request as presented before execution. Label what the request
asks for and the capabilities needed to satisfy it, not what a later solution
actually did. A source document or outcome can explain a request but is not
automatically part of the input text. Multiple intents and capabilities can
apply. An absent label is a negative in this training format, not an unknown;
do not mark a record verified if any relevant label is uncertain.

The current schema requires at least one intent and lists for capabilities and
risks. Use an empty risk list only when the text supports a negative for every
risk in this version of the taxonomy. An empty capability list means none of
the capabilities in this limited taxonomy apply; it does not mean the task
requires no capabilities. Do not invent new labels during annotation without
first versioning the taxonomy and agreeing how to evaluate them.

## Intent labels

| Label | Apply when the request asks to | Do not apply merely because |
| --- | --- | --- |
| `respond.general` | Explain, summarize, or answer without a more specific supported intent. | A response will accompany another requested action. |
| `build.software` | Create or modify software, configuration, or code-bearing tooling. | It discusses software or asks for a read-only explanation. |
| `verify.outcome` | Test, reproduce, or check whether a concrete behavior or result holds. | The request uses the word "test" as a topic without asking for verification. |
| `research.web` | Seek information that requires accessing external web sources. | The request asks about an existing local document or mentions a URL. |
| `create.image` | Produce or edit an image. | The request mentions a UI, visualization, or image as a topic. |
| `evaluate.semantic` | Compare or grade the quality of candidates, outputs, or approaches against criteria. | A simple pass/fail test is requested without a quality assessment. |

## Capability labels

| Label | Apply when fulfilling the request requires | Boundary |
| --- | --- | --- |
| `text.generate` | A general explanatory or summary response. | Use for `respond.general`; do not add automatically to every task. |
| `filesystem.write` | Creating or modifying project files. | Do not infer writes for read-only investigation or explanation. |
| `test.execute` | Running tests, experiments, or commands to reproduce/check behavior. | A purely conceptual request to explain a test does not require execution. |
| `web.access` | Accessing external web sources. | Reading local documents alone is not web access. |
| `image.generate` | Generating or editing an image. | Discussing an image alone does not require it. |
| `evaluation.grade` | Assessing candidate quality against a rubric or comparative criteria. | An ordinary boolean check belongs to verification rather than semantic grading. |

These labels are a coarse prototype: they do not cover all possible tools and
must not be treated as permissions. Determine the minimum capability implied by
the request; if plausible interpretations differ, leave the record unverified
and record the question. For example, an investigation might be answerable from
an existing report or might require running a reproduction.

## Risk label

`sensitive-data` means the request contains or asks to expose or manipulate
secret or private data in a way that needs special handling. Merely discussing
authentication, prompt injection, or a token budget is not enough. Never place a
real secret in an example. This one label is not a complete security taxonomy:
an empty list does not mean a task is safe or authorized.

## Review procedure

1. Check the request and source reference, remove or replace private details,
   and distinguish an original request from a constructed or paraphrased one.
2. Apply every supported label from the text alone, including negatives. Record
   uncertainty, missing taxonomy, and disagreements in a separate review record.
3. A human reviewer checks the request and all labels; only then may `verified`
   become `true`. Keep reviewer identity, date, and decision in a suitable
   access-controlled review log. The boolean is an attestation, not proof.
4. Group variants, follow-up requests, and related records into one family before
   assigning splits. Keep whole projects or time periods for a final held-out
   test where practical. Do not use the held-out set for label or model tuning.
5. Record source revision, allowed use, dataset and split digests, counts, and
   exclusions before training. Human review does not by itself grant broader
   publication, redistribution, or remote-training permission.

## Current RRFlow review batch

[`dataset/rrflow-review.v1.jsonl`](dataset/rrflow-review.v1.jsonl) contains
25 draft, constructed examples from seven source records, not 25 independent
original user prompts. Every label and negative is provisional. The `family`
field groups related examples for future splitting; no train, validation, or
test assignment has been made. The available trainer currently performs a random
row split and does not enforce family or project separation: do not use that
split for a credible held-out result.

Source project: `rrflow-development`, local checkout HEAD
`dd225072bcc639c89b0eef968625cc73303ea998`. The read-only status check reported
no tracked changes, though Git could not read the user's global ignore file.
SHA-256 hashes below identify the source files actually read; paths are relative
to the RRFlow checkout. These are provenance references, not training examples.

| Source path | SHA-256 |
| --- | --- |
| `docs/evidence/change-journals/gate-c/scan-resistant-page-cache-integration-evidence-c-06j.md` | `dfc46585070a64ba454dbf58dbce84adc504aaa1ddedd6c8da01f50fb5cab09b` |
| `docs/evidence/change-journals/gate-c/c-07g-desktop-qualification-failure.md` | `8e27bc4c3668e0133cebc0723c52e5bd3623bb5146f4645be2f4961c58357d8e` |
| `docs/reference/inference/lfg-swag.md` | `a03e5eae417ced3222b7104ac488e9058ad4cdbf4eb2c966e512734aac3fa68e` |
| `docs/roadmap/rrflow-1.0/executable-dependency-spine.md` | `f242e3be7309b2c81d9472789777097ccb79e4ac4b7e0e47a5ff844502f76337` |
| `docs/objectives/rrflow-1.0-alpha.md` | `9977d06b00ec199c8060f4686ed8bc4cd897553cbe1dcfed508d7510ded0c234` |
| `docs/reference/inference/learning/controller-training.md` | `fbab449ff8e6d6ad38447998b900166334cb4909a5b36290c6f8e86c97a3208b` |
| `docs/roadmap/rrflow-1.0/gate-g.md` | `58043c4876c804e4b3a2779364ad2cfc31f7f0aafe0c7852e6939c81895a3bcc` |

Check `rrflow-review-002` first: it assumes executing a test is required to
investigate a failure, but the request does not say whether existing evidence
would suffice. Either clarify the request or remove the execution label if a
reviewer determines it is not necessary. Review all other records under the same
rule; the draft labels should not be converted into verified labels as a batch.

## RRFlow PR source batch

[`dataset/rrflow-pr-review.v1.jsonl`](dataset/rrflow-pr-review.v1.jsonl)
contains five actual PR titles or excerpts captured on 2026-10-29, with links
to their PRs and local outcome journals. They are authored project descriptions,
not demonstrated original user prompts. PR bodies can be edited; their current
contents and the journals' claims about a pre-edit brief do not independently
prove the text's historical timestamp. The `outcome` field describes reported
evidence, not an independently rerun result or a training input. Unknowns and
remaining qualification are retained rather than labeling every merged PR a
complete success.

All five rows require human review, including whether their wording supports
the proposed labels. Four linked inference-budget PRs intentionally share one
family. Before combining this batch with the constructed batch, reconcile
cross-file relationships: for example, `rrflow-g-budget-control` overlaps the
inference-budget series and `rrflow-c07g-desktop` appears in both. Similar source
text, dependencies, and outcome evidence must remain in the same split even if
the current family names differ. Prefer a different project as the final test.
The current training script ignores source/outcome metadata and only tokenizes
`text`; it does not validate these provenance or split requirements.

## Makers Impulse source and evaluation boundary

The local, Git-ignored `dataset/private/makers-impulse-review.v1.jsonl` contains
five exact prompts preserved in Replit assistant-checkpoint commit messages from
the Makers Impulse checkout at HEAD `f92162507e421c2c9a4cb971329a35c0e095b5c0`.
Each row identifies the source commit and includes a SHA-256 of the captured
prompt, changed paths from that commit, and tentative labels. Commit metadata
attributes the text to a user but does not independently establish authorship.
The file stays local unless separately reviewed and intentionally shared.

All five records are from the same recorded session and should be treated as
one correlated family. Reserve Makers Impulse as an external-project evaluation
source rather than training on it alongside RRFlow. These five inspected
examples are review examples, not a blinded final test: assemble and review a
separate unseen set before freezing final evaluation, and do not use it for
training, label design, or model selection. The current trainer does not enforce
this project boundary.

Changed files show what the checkpoint recorded, not whether the request was
fulfilled. For example, the checkpoint associated with a Biome rewrite request
has no changed paths, despite its title. No tests or outcomes have been verified
for these examples. The requests also reveal gaps in the current schema: it has
no explicit audit intent or fields for read-only scope, sequencing constraints,
or requested verification method. Do not interpret missing labels as proof
that those aspects are absent.
