# LFG integration observation v1

This local developer interface projects the same deterministic assessment used
by `POST /api/lfg/assess` into a bounded, advisory-only response. It is an LFG
interface, not RRFlow admission, SWAG selection, MCP, a trained model, provider
dispatch, or proof of authorization. It does not read local provider histories,
sign-in checks or subscription declarations. It returns a role and effort, not
a selected provider/model or permission to use one.

## Discover and submit

`GET /api/lfg/observations/capabilities` returns the request and response schema
names, `deterministic-rules` analyzer identifier, effective assessment and
preflight-rule revision and SHA-256 digests, configured input and response byte
limits, and the scoring-prefix byte limit. Both digests are required on a call
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
`status: partial` means scoring used only the configured prefix; `intent.status:
inconclusive` means the general intent fallback fired. A score is not a
probability, `confidence` is null, and `advisory` is always true. Hashes of
predictable prompt text can still disclose information by guessing and must be
treated as sensitive metadata. Input text and extension values are not echoed.

The server reports `unsupported_schema`, `invalid_input`,
`input_digest_mismatch`, `configuration_changed`, `deadline_expired`, or
`response_limit_exceeded` or `nonfinite_score` where applicable. Error responses contain static codes,
not prompt content. Normal HTTP size and JSON parsing errors can also occur.
Capability and successful observation responses have `Cache-Control: no-store`.

## Integration boundary and limits

The current RRFlow `EndpointSelectionRequest` has a different, strict schema;
this endpoint does not change that request or feed an observation to the engine.
The engine must define its own authenticated observation admission, bind the
actual task/source identity, and authorize any later proposal or effect. Caller
IDs and hashes supplied to LFG are claims. This interface currently has no
authenticated hosted mode, model artifact identity, hard CPU/memory isolation,
project-profile binding, context reference, persistence, distributed tracing or
cross-process conformance guarantee. Keep it on a trusted local boundary until
those capabilities exist and have been verified.
