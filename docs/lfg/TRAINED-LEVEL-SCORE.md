# Trained level score

An operator can bind a trained model to [the assessment](ASSESSMENT.md). The
model reads the prompt and gives a probability for each of five levels of work.
Its expected level chooses the complexity bracket, and the bracket then sets the
agent range, the model role and the effort, exactly as the rule score does
today. The rules still run and are still reported. The answer stays advisory,
and `confidence` stays null.

Nothing changes unless the assessment configuration binds a model. The bundled
configuration binds none, and its digest is unchanged.

## How it fits together

| Piece | Where | What it pins |
| --- | --- | --- |
| Assessment configuration | `LFG_ASSESSMENT_CONFIG` | `modelScore.manifestSha256`, level brackets and cut points, and optional `structuredInput`. All are part of the assessment digest. |
| Runtime configuration | `LFG_MODEL_SCORE_CONFIG`, see the [example](../../crates/lfg-server/config/model-score-runtime.example.json) | The model directory, byte limits, how many prompts may run at once, and CPU threads per prompt. |
| Model directory `manifest.json` | `lfg.model-score-manifest.v1` | SHA-256 of `model.onnx`, `tokenizer.json` and `input.json`, the input and output names, the level count and provenance. |
| `input.json` | `lfg.dev-model-input.v1` | The fixed prefix of token IDs holding the question and its option markers, the token room left for the prompt, and the separator and mask tokens. |

At startup LFG reads the manifest and checks its digest against the assessment
configuration, then checks every file against the manifest. Only then does it
load the tokenizer and the ONNX graph, once, into `tract`. LFG never loads ONNX
Runtime. A mismatch stops startup. A build without the `onnx-shadow` feature
refuses a configuration that binds a model.

For each prompt, LFG removes the mask token from the text, tokenizes it without
special tokens, and keeps as many tokens as the room allows. It then builds the
prefix, the prompt's tokens and the separator, and runs the graph.
`modelScore.truncated` is true when the prompt was longer than the room, and
observations then report `status: partial`. Inference runs off the async
runtime, under a limit of concurrent prompts. A full limit returns `model_busy`
rather than queueing.

```json
"structuredInput": {"jsonFields": ["title", "goal"], "separator": "\n"},
"modelScore": {
  "id": "lfg-dev-text-20261106",
  "manifestSha256": "SHA256_OF_THE_MODEL_DIRECTORY_MANIFEST",
  "levelBrackets": ["trivial", "simple", "moderate", "complex", "major"],
  "levelCutPoints": [0.5, 1.5, 2.5, 3.5]
}
```

The level is the number of cut points the expected level reaches. With these
cut points it is the expected level rounded to the nearest whole level, and
halves round up.

## What a bound server never does

- **No silent fallback:** it never answers an observation from the rules alone.
  A missing, busy or failing model returns HTTP 503 with `model_unavailable`,
  `model_busy`, `model_failed` or `model_output_invalid`. `/api/lfg/assess`
  still answers for people, with `modelScore.status` set to `unavailable`.
- **No wasted model runs:** stale or invalid observation requests are refused
  before the model runs, including requests pinned to another configuration's
  digest.
- **No prompt text in records:** reason IDs and errors carry identifiers only,
  never prompt text.

## The current dev model

`lfg-dev-text-20261106` is Laya's `multilingual` open weights (mmBERT, 322M
parameters, Apache-2.0) fine-tuned on 1,270 of the owner's prompts and the work
that followed them. It is a dev model, not LFG's own base model. Its weights,
data and configuration stay in the owner's private folders.

The same recipe, scored on prompts from projects it hadn't seen, gave a rank
correlation with realized tool calls of +0.36. On later prompts, after training
only on earlier ones, it gave +0.34. Both figures come from the 4 November
scorecard, recorded in the evidence page in PR #164. These are one person's
logs, not a held-out test.

Measurements on 6 November:

| Measure | Result |
| --- | --- |
| Size | 644 MB. Weights are stored as fp16 and cast to fp32 when loaded, so the arithmetic is fp32. Against the fp32 export: rank correlation 1.000, largest difference 0.0019 levels, 99.6% the same level. 8-bit weights were too lossy (rank 0.81–0.84) and are not used. |
| LFG against PyTorch | On 1,249 of the owner's prompts, the largest difference in expected level was 0.0068 and the mean 0.0027. Expected levels are reported to two decimals. The largest probability difference was 0.0008, and 99.6% got the same level. 21 prompts were over the 12,000-byte prompt limit, and 45 were longer than the model reads and were cut. The public samples, including French, Chinese and text containing the mask token, agreed within 0.006. |
| CPU time per prompt | Intel Core Ultra 9 275HX, `tract`, 117 prompts: p50 920 ms on 1 thread, 614 ms on 4, 583 ms on 8 and 588 ms on 16. p95 at 8 threads: 963 ms. |
| Memory | About 850 MB resident. Load takes about 9 seconds. |

Most of the time goes to the fixed question: 150 of the 204 tokens in an
average input. A shorter question (130 tokens on average) was tried on
6 November with the same recipe and the same project folds. It was worse on
every group, and clearly worse on later prompts: +0.20 against +0.34
(difference −0.14, 95% range −0.32 to −0.02). Don't shorten the question to
save time. Faster options still untried: a GPU runtime, or a smaller
student distilled from this model.

## Using it with RRFlow

RRFlow's draft engine integration (`rrflow-development` PR #249) calls
`POST /api/lfg/observations` over literal loopback. It sends its task
projection as JSON, so set `structuredInput` to `["title", "goal"]`, or the
rules and the model will read IDs, digests and cursors. RRFlow uses the
intent, role, effort and reason IDs. The model's choice reaches it through the
role and effort.

1. **Start LFG:** run the server built with `--features onnx-shadow`, with both
   configuration variables set and `LFG_LEARNING=off`.
2. **Read the digests:** `GET /api/lfg/observations/capabilities` reports
   `assessmentSha256`, `rulesSha256` and the model's `manifestSha256`.
3. **Pin them in RRFlow:** put the digests in the endpoint profile's
   `selection_policy.lfg` (`assessment_sha256`, `rules_sha256`). A model change
   changes the assessment digest, so RRFlow then sees `configuration_changed`
   instead of a different answer.
4. **Set the timeout:** a `timeout_ms` of at least 3,000 leaves room for the
   p95 above.
5. **Match the input limit:** keep `maximum_input_bytes` at or below LFG's
   `maxPromptBytes` (12,000 by default).

Checked on 6 November against PR #249's head (`f17d62b9`), with a scratch
copy of RRFlow:

| LFG | RRFlow pins | Live transport test | Live engine flow |
| --- | --- | --- | --- |
| This build, rules only | Its fixture's, unchanged | Passed | Passed |
| This build, dev model bound | The model configuration's | Passed (0.61 s) | Passed (1.41 s) |

RRFlow's decoder accepts the `rules-and-trained-model` analyzer and the extra
`modelScore` block; it carries the model's role and effort into the
controller packet.

RRFlow-side issues seen in that branch, and not changed here:

- **Budget stop:** one failed LFG call stops the task's budget.
- **Held permit:** an uncertain failure can leave the controller permit held.
- **Attempt IDs:** an attempt ID containing `:` fails before the call.
- **No deadline:** `deadlineUnixMs` is never sent.
- **Input limit:** the installed profile's `maximum_input_bytes` (65,536) is
  larger than LFG's default prompt limit.
