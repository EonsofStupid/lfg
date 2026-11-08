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
When only a sanitized summary is needed, serialize `dict(result)`. To persist
private extensions, retrieve them explicitly and pass them through the future
producer's verified private storage and provenance path. This API reduces
accidental disclosure; it cannot prevent deliberate introspection or a caller
from explicitly exposing values returned by `private_extensions()`.

```python
result = parse_response(raw_response, selected_labels, taxonomy_digest)
summary = json.dumps(result)  # sanitized public summary
if result["status"] in {"answered", "abstain"}:
    private_extensions = result.private_extensions()  # handle as sensitive data
```

The parser does not write private data to disk. A future producer must store
any extension values it needs using verified owner-only permissions and
hash-bound provenance, and must keep raw responses out of logs, Git and public
reports. Forwarding extensions must preserve them as inert data: they must not
alter labels, selection, training weight, or trust.

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

Legacy positive-list mode remains unchanged until an explicitly configured,
versioned producer mode is implemented and separately tested end to end. This
schema module alone does not collect labels, increase training coverage, train
weights, score heldout examples, or qualify a model.
