# LFG integration observation v1

This local developer interface projects the same assessment used by
`POST /api/lfg/assess` into a bounded, advisory-only response. It is an LFG
interface, not RRFlow admission, SWAG selection, MCP, provider dispatch, or
proof of authorization. By default the assessment is deterministic rules; an
operator can bind a trained model to choose the bracket, as described in
[trained level score](TRAINED-LEVEL-SCORE.md). It does not read local provider histories,
sign-in checks or subscription declarations. It returns a role and effort, not
a selected provider/model or permission to use one.

## Discover and submit

`GET /api/lfg/observations/capabilities` returns the request and response schema
names, the analyzer identifier (`deterministic-rules`, or
`rules-and-trained-model` when the configuration binds a model), effective assessment and
preflight-rule revision and SHA-256 digests, configured input and response byte
limits, the scoring-prefix byte limit and, when bound, the model's `id` and
`manifestSha256`. Both digests are required on a call
so a changed configuration is reported as a conflict instead of silently
producing a different answer. Digests use LFG's `digest_value`: recursively
sort JSON object keys, preserve array order, serialize compact UTF-8 JSON with
the project's serde version, then SHA-256. Treat digest as an opaque identity;
the capabilities endpoint is authoritative for the running process. This does
not identify the server executable, prove the provider identity, or attest a
future output.

`POST /api/lfg/observations` accepts UTF-8 JSON. Compute `inputSha256` over the
exact UTF-8 bytes of the decoded `text`, with no Unicode normalization. Example
values below are placeholders; callers must obtain live configuration hashes and
compute the input digest. `requestId` is an ASCII alphanumeric, `.`, `_`, or
`-` identifier, bounded by the configured prompt-byte limit. It is caller
supplied and must not contain private text or secrets.

```json
{
  "schemaVersion": "lfg.observation-request.v1",
  "requestId": "task-123",
  "revision": 1,
  "text": "A request selected by the caller",
  "inputSha256": "SHA256_OF_EXACT_UTF8_TEXT",
  "expectedAssessmentSha256": "FROM_CAPABILITIES",
  "expectedRulesSha256": "FROM_CAPABILITIES",
  "deadlineUnixMs": 2000000000000,
  "futureCallerField": {"example": "not interpreted"}
}
```

`deadlineUnixMs` is optional. The server checks it before and after synchronous
analysis and refuses to return an expired result. It is not a preemptive CPU
deadline: a request that overruns can still consume CPU until analysis returns.
Input and serialized response limits come from the running operations
configuration. The HTTP body has its own configured bound; an input at the
maximum prompt limit may still exceed it once JSON overhead is included.

Additional top-level fields are accepted as opaque data, counted in
`ignoredExtensionCount`, and ignored. They are not returned or placed in traces,
and cannot alter validation, permissions or the assessment. This forward
compatibility does not mean that an LFG project profile, RRFlow observation or
other extension has been processed. New behavior requires a specified,
implemented and tested field or a new schema version. The entire HTTP payload,
including extra fields, remains subject to the configured body limit.

The response uses `lfg.observation.v1`, carries the request ID/revision, input
and configuration hashes, analyzer kind, numeric rule-point score, complexity
bracket, agent range and suggestion, role, effort and configured reason IDs.
`status: partial` means scoring used only the configured prefix, or the bound
model read only part of the text; `intent.status: inconclusive` means the
general intent fallback fired. A score is not a
probability, `confidence` is null, and `advisory` is always true. Hashes of
predictable prompt text can still disclose information by guessing and must be
treated as sensitive metadata. Input text and extension values are not echoed.

### Optional trained model and structured input

These change behaviour only when the assessment configuration sets them, and
each is part of the assessment digest, so a pinned `expectedAssessmentSha256`
also pins them.

- `structuredInput` (`jsonFields`, `separator`): when `text` is a JSON object
  with any of those fields as non-empty strings, LFG assesses only their values,
  joined in order. RRFlow's engine sends its task projection as JSON, so
  `["title", "goal"]` keeps IDs, digests and cursors out of the assessment. The
  reason IDs then include `input:structured`. `inputSha256` still covers the
  exact `text` sent.
- `modelScore` (`id`, `manifestSha256`, `levelBrackets`, `levelCutPoints`): the
  model's expected level picks the bracket, which then sets the agent range,
  role and effort. `complexity.score` stays the rule score in
  `configured-rule-points`, and `confidence` stays null. The response gains a
  `modelScore` object with `status`, `id`, `manifestSha256`, `expectedLevel`,
  `probabilities`, `level`, `bracket`, `rulesBracket`, `elapsedMicros` and
  `truncated`. The reason IDs include `score-model:available`,
  `score-model-level:<n>` and `bracket-source:model`. A configuration that binds
  a model never answers from the rules alone: without a valid model output the
  observation fails with `model_unavailable`, `model_busy`, `model_failed` or
  `model_output_invalid` (HTTP 503). Request and digest checks run first, so a
  stale or invalid request never costs a model run.

Clients that ignore unknown response fields, as RRFlow's decoder does, keep
working; they see the model's choice through the bracket, role and effort.

The server reports `unsupported_schema`, `invalid_input`,
`input_digest_mismatch`, `configuration_changed`, `deadline_expired`,
`response_limit_exceeded`, `nonfinite_score` or one of the model codes above,
where applicable. Error responses contain static codes,
not prompt content. Normal HTTP size and JSON parsing errors can also occur.
Capability and successful observation responses have `Cache-Control: no-store`.

## Integration boundary and limits

RRFlow's draft engine integration (`rrflow-development` PR #249) calls this
endpoint over literal loopback, checkpoints the validated advisory in the task
attempt, and passes intent, role, effort and reason IDs to endpoint selection.
It ignores the complexity score, agent counts and unknown fields, and it pins
both digests in its endpoint profile. The engine remains responsible for
admission, binding the actual task identity, and authorizing any later
proposal or effect. Caller
IDs and hashes supplied to LFG are claims. This interface currently has no
authenticated hosted mode, hard CPU/memory isolation,
project-profile binding, context reference, persistence, distributed tracing or
cross-process conformance guarantee. Keep it on a trusted local boundary until
those capabilities exist and have been verified.
