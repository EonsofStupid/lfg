# Project intake: current contract and boundary

The bundled questions and their source paths are defined in `crates/lfg-server/config/intake-template.v1.json`. Set `LFG_INTAKE_TEMPLATE` to a JSON file using the same schema to replace them at server startup; an invalid configuration stops startup. `GET /api/lfg/intake/template` returns the effective template and its digest. Questions use JSON pointers into an app snapshot, a value kind, and a required flag. A missing or incorrectly typed value is reported rather than guessed. An explicitly supplied empty array or `false` is a value.

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

The LFG web UI currently sends its own typed request, current project ID, configured model catalog and user-declared subscription selections. These declarations do not verify subscription entitlements. No DevPulse, RRFlow, or SWAG host sends this snapshot yet; no automatic provider-registry synchronization, operator-feedback training capture, or editable intake configuration UI is implemented. A real integration must obtain facts from its authoritative host and preserve the distinction between configured, activated, and execution-qualified providers.

Do not send credentials. Configurable key and prefix checks reject some credential-shaped input, but cannot detect all secrets. The API currently has no authentication and is intended for the existing local deployment boundary; hosted use requires appropriate authentication, authorization, and transport protection before exposing it.
