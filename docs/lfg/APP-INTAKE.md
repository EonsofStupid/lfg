# Project intake: current contract and boundary

The bundled questions and their source paths are defined in `crates/lfg-server/config/intake-template.v1.json`. Set `LFG_INTAKE_TEMPLATE` to a JSON file using the same schema to replace them at server startup; an invalid configuration stops startup. `GET /api/lfg/intake/template` returns the effective template and its digest. Questions use JSON pointers into an app snapshot, a value kind, and a required flag. A missing or incorrectly typed value is reported rather than guessed. An explicitly supplied empty array or `false` is a value.

### Sources and current coverage

The bundled questions are a working draft, not a verified transcription of the
owner's complete original question set. The project conversation requested
per-project AI subscriptions and model/effort profiles, CPU efficiency,
MCP/RRFlow/SWAG connectivity, tracing and developer statistics, and local or
hosted deployment with configurable model and weight updates. These requests
inform the provider, model, tools, integration, deployment, observability and
update questions. The current RRFlow repository's
`docs/reference/inference/lfg-swag.md` defines bounded authorized inputs and
observable outputs with provenance and explicit missing/inconclusive states;
those requirements inform the analysis input and output questions. It also
states that engine-admitted LFG/SWAG execution is unimplemented. These are
derived questions, not evidence that the original notes have all been found or
that the owner has approved this wording.

The owner's separate authored governance questions have since been recovered
and are available in the [boundary question catalog](BOUNDARY-QUESTION-BANK.md).
They are not the intake questionnaire described on this page.

Every answer is filled only from the supplied snapshot path. Adding questions
does not connect those paths to an app: the LFG UI currently supplies the
request, project ID, and configured model declarations, plus the optional
DevPulse provider projection described below. It does not populate the new
deployment, analysis, observability or update paths. An integrating app must
explicitly supply these facts; absent or unknown facts stay missing. The UI
does not yet edit or save the missing answers as a per-project profile.

A host can submit a snapshot to `POST /api/lfg/intake/preview`, for example:

```json
{
  "schemaVersion": "lfg.app-snapshot.v1",
  "sourceApp": "host-app",
  "facts": {
    "project": {"id": "project-id", "objective": "the host's recorded objective"},
    "request": {"text": "the user's actual request"},
    "providers": [],
    "models": []
  }
}
```

The arrays in this example mean the host has actually determined they are empty; omit fields whose values are unknown. The response contains only the configured answers, their status, the template revision and digest, and the count of required answers still needed. The endpoint does not persist the snapshot or add it to training signals. Its response is marked `no-store`.

The LFG web UI sends its own typed request, current project ID, configured model catalog and user-declared subscription selections. These declarations do not verify subscription entitlements.

An optional DevPulse adapter reads the existing local agent seam's `GET /state` provider projection. Copy `crates/lfg-server/config/devpulse-context.example.json`, set its loopback endpoint and limits for the user's instance, and point `LFG_DEVPULSE_CONTEXT_CONFIG` at the copy before starting LFG. No adapter is enabled by default. The endpoint must be a literal loopback socket address; redirects and proxies are disabled, and response size and request time are bounded by the configuration. The UI polls at the configured interval while the intake panel is open. The context response identifies the effective configuration by revision and digest. Invalid configuration fails startup. If the seam is unreachable or its response is invalid, the UI marks provider information unknown rather than substituting a stale roster.

The seam's response also contains other state; the adapter discards it and exposes only each provider ID, activation flag, and qualified capability slugs. DevPulse's activation may mean history-only access; it does not establish a usable model or subscription entitlement. Model entries remain separately configured user declarations, and the classifier does not yet use the DevPulse roster for recommendations. The connection is read-only and does not modify the DevPulse registry. It does not yet capture operator feedback for training, configure models through the UI, or integrate RRFlow or SWAG. DevPulse's seam and this LFG API have no authentication; another local process could impersonate the configured seam, and hosted use requires authentication and authorization.

On 2026-10-30, an isolated process using the actual DevPulse seam and fabric crates at commit `cb7baf0756cbff1ca7a4c18bef81ebdae1e0ee57` served a deliberately test-provided history-only roster. The LFG release binary read that roster and combined it with a real, unverified DevPulse project excerpt in an intake preview; the model question stayed missing. This verifies the wire path, not a live user's signed-in providers or the complete DevPulse application. A Windows release with this adapter was 5,519,360 bytes versus 4,994,560 bytes for the earlier release. Short sequential performance checks varied substantially and are inconclusive; they did not measure a live seam or production load.

Do not send credentials. Configurable key and prefix checks reject some credential-shaped input, but cannot detect all secrets. The API currently has no authentication and is intended for the existing local deployment boundary; hosted use requires appropriate authentication, authorization, and transport protection before exposing it.
