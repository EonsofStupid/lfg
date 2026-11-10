# Explicit teacher label claims

`ml/lfg-encoder/teacher_explicit_labels.py` defines a pure, versioned wire
contract for a teacher proposal over selected `(field, label)` identities. It
does not call a model, collect project data, train LFG, verify an outcome, or
establish model quality.

Each decision is explicitly `positive`, `negative`, or `unknown`. The response
may omit selected identities; the parser returns every selected identity in
the pinned input order and fills every omission with `unknown`. A response-level
refusal, truncation, incomplete result, invalid JSON, malformed identity,
duplicate decision, or taxonomy mismatch cannot produce asserted labels.
Abstention also yields all unknowns and cannot accompany positive or negative
claims. The parser rejects duplicate JSON object keys and unknown top-level
fields. Errors and invalid results contain no raw request or response text.

The generated JSON Schema pins the taxonomy digest and makes each field/label
pair atomic. Schema adherence is only structural validation; the producer must
still validate responses and bind the selected identities, taxonomy, prompt,
model, configuration, code, source records and data split into its own
provenance. Inert extension data is accepted only through the declared
`extensions` object, bounded, and never changes a label.

`parse_response` returns a `ParsedResponse`. Its mapping contains the ordinary
sanitized summary, including only an `extensions_present` boolean. Default
`repr`, `str`, and `json.dumps(result)` do not include extension names or
values. For a successfully answered or abstained response, a caller that
deliberately needs the accepted extension values may call
`result.private_extensions()`. That returns a detached copy, so later mutation
does not change the parsed result. Invalid and non-answer outcomes expose an
empty private extension object. The private accessor contains untrusted model
output and must be handled as sensitive input; do not print it or place it in
logs, Git, or public reports.

Pickling a `ParsedResponse` is deliberately rejected with a generic `TypeError`
so default object-state serialization cannot copy private extension slots.
When only a sanitized summary is needed, serialize `dict(result)`. The isolated
teacher producer persists accepted extensions in the same owner-private,
source-bound checkpoint record as each explicit progress event. The extension
value and label event are fsynced together before the controller ACKs the
worker. The record binds row/source identity, attempt and worker event digest,
model, taxonomy, schema, prompt and parser module; the value has its own
canonical digest and configured byte bound. The `extensions` field may be
omitted even when its byte budget is zero or one; a present empty object uses
two canonical bytes and must fit. Preflight includes the maximum configured
extension bytes in both the worker event and private checkpoint record, along
with output, acknowledgment and framing bounds. Missing extensions remain
distinct from a present empty object. Recovery validates the binding and integrity
before exposing values. Deliberate retrieval is available through
`read_private_teacher_extensions` in `train_teacher_proposals.py`; its result
keeps a redacted representation and offers explicit accessors. These values
remain untrusted inert data and never affect labels, selection, training weight
or trust. This API reduces accidental disclosure; it cannot prevent deliberate
introspection or a caller from explicitly exposing retrieved values.

```python
result = parse_response(raw_response, selected_labels, taxonomy_digest)
summary = json.dumps(result)  # sanitized public summary
if result["status"] in {"answered", "abstain"}:
    private_extensions = result.private_extensions()  # handle as sensitive data
```

The parser itself does not write private data to disk. The producer writes only
the accepted extension object, not the raw model response, using its verified
owner-private checkpoint and hash-bound provenance. It keeps extension values
out of Git, stdout/stderr diagnostics, ordinary public event summaries and
published reports. Windows process-death durability is checked by the
checkpoint protocol, but Python cannot guarantee power-loss durability of
Windows directory metadata or underlying storage hardware. Forwarding
extensions must preserve them as inert data: they must not alter labels,
selection, training weight, or trust.

Ollama's current structured-output documentation accepts a JSON Schema through
the local `format` field and recommends validating the returned JSON. This
module generates Draft 2020-12 schema keywords (`const`, `oneOf`,
`additionalProperties`) to pin field/label pairs. Compatibility of those
keywords with the exact installed server and selected model remains unverified;
the future producer must check that runtime before sending project data.
Source: <https://docs.ollama.com/capabilities/structured-outputs>.

These are unreviewed teacher claims. A positive or explicit negative is not a
verified project outcome or ground truth. The current teacher collection and
owned training paths are not wired to this module by this contract-only step.
Future producer work must preserve the existing isolated worker, source and
model hash bindings, durable progress, train-only selection, privacy and
resume checks. Future training work must consume only explicitly asserted
positive/negative claims with source-bound provenance, retain their unreviewed
origin and configured weight, detect conflicts with source labels, and leave
unknowns out of target cells. Never interpret an omitted label as negative.

## Isolated teacher producer integration

The isolated teacher collector has an opt-in `teacher_response_mode` equal to
`lfg.teacher-explicit-labels.v1`. Omission retains legacy positive-list mode.
The opt-in mode requires `assume_exhaustive_labels: false` and a bounded,
operator-supplied `teacher_prompt_contract` containing `{identities}` and
`{text}` slots. The producer supplies ordered atomic identity pairs and only
the screened row text to that contract. Its generated JSON Schema is sent in
the local runtime `format` field. The request remains subject to existing
request/prompt/response and protocol bounds; schema support by a particular
runtime/model is not assumed or qualified here.

Only a complete HTTP response with the pinned model identity, accepted
completion reason, and no refusal is parsed. Each positive, explicit negative,
or unknown becomes a source-bound unreviewed claim. Omitted identities stay
unknown. Nonanswers leave every state unknown. New-mode state keys use the
canonical JSON serialization of `[field, label]`; conflicts are arrays of
atomic field/label objects, so delimiter collisions cannot alias identities.
Legacy output remains positive-list compatible and ambiguous colon-joined
identities are rejected.

The worker stages the exact parser module bytes and verifies their SHA-256 in
its isolated `python -I` bootstrap. Mode, ordered taxonomy/schema, prompt and
module identities bind to the worker config, empty-row probe, ready receipt,
payload, per-row durable event, ACK/checkpoint lifecycle, terminal report and
resume key. A changed identity rejects old progress while preserving it for
inspection. The producer preserves inert extension values only in its
owner-private checkpoint; public reports contain no raw responses or extension data. The collected
labels remain teacher claims, not verified outcomes, ground truth or a quality
qualification. This integration does not itself bridge proposals into LFG
training or run private collection.
