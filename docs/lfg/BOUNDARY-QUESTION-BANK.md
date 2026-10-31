# Authored boundary questions and attunement

The owner's draft **Boundary Question Bank — Core Predicates v0.2**, six stack
drafts and **Domain Boundaries v0.1** describe a governance question battery.
The [configurable catalog](../../crates/lfg-server/config/boundary-question-banks.v1.json)
transcribes their 425 numbered questions, sections, source terms, and explicit
`Binds to:` artifact names. These are source drafts, not ratified policies;
their technical references and version-specific assertions have not been
independently checked. The numbered questions are not training examples.
On 2026-10-31, an independent text comparison against the retrieved drafts
matched all 425 numbered question texts and source terms in order. This check
did not validate the drafts' technical claims or transcribe the unnumbered
curriculum and routing material.

| Bank ID | Source draft | Questions |
| --- | --- | ---: |
| `core` | Core Predicates v0.2 | 67 |
| `rust` | Rust v0.1 | 53 |
| `go` | Go v0.1 | 57 |
| `nextjs` | Next.js App Router v0.1 | 38 |
| `react` | React v0.1 | 46 |
| `typescript` | TypeScript v0.1 | 43 |
| `vanilla-js` | JavaScript v0.1 | 52 |
| `boundaries` | User, admin, chat and AI v0.1 | 69 |

`GET /api/lfg/boundaries/questions` returns the effective catalog and digest.
For an alternative catalog, set `LFG_BOUNDARY_QUESTION_BANK` to a file with
the same schema before starting the server. Invalid configuration rejects
startup. The question text, banks and artifact names are data, not a compiled
list in the handler.

An integrating app can submit its selected banks, optional exact section names,
and project artifacts to `POST /api/lfg/boundaries/attune`:

```json
{
  "schemaVersion": "lfg.boundary-attunement.v1",
  "sourceApp": "host-app",
  "selections": [{"bankId": "core", "sections": ["A"]}],
  "artifacts": {"module inventory": ["src/auth", "src/session"]}
}
```

The preview attaches the supplied artifact to each question whose binding name
matches; other required bindings are listed as missing. A question without an
explicit source binding is marked `not_required`, meaning only that this
draft did not name a project-artifact binding; it is not an answer. Null or
empty string, array or object values remain missing. The response identifies the
source app and catalog digest and is `no-store`. The server neither persists
nor verifies the artifacts; it applies the intake template's configurable
credential-shaped input checks, which cannot detect all secrets. Do not send
credentials. This local API has no authentication; hosted use needs appropriate
authentication, authorization and transport protection.

To read artifacts from an integrating app's existing facts instead, add an
`artifactPaths` object to a copied catalog, mapping binding names to JSON
pointers. For example, `"artifactPaths": {"module inventory":
"/project/modules"}` lets the request supply `"facts": {"project":
{"modules": ["src/auth"]}}`. The response marks this as `app_fact` and
includes the configured pointer; directly supplied artifacts are marked
`app_artifact`. A directly supplied key takes precedence even if its value is
empty or null, so an explicit missing value is never silently replaced. Unknown
binding names and malformed pointers reject catalog startup. The bundled
catalog does not guess an app's schema: its mappings must be configured for the
integrating app, and the app must send the facts.

On 2026-10-31, a Windows optimized build with the UI embedded served the UI and
the attunement endpoint from one process. With an operator-provided mapping for
`module inventory`, a request containing nine actual LFG server source paths
filled that binding for the core reuse question as `app_fact`. The endpoint did
not answer the question. That local build was 5,707,776 bytes with SHA-256
`d220e4706376f3a16db0a0fac83b1bbc77eee3e9d5d6329c1d7571454d9b14ae`.
This is a functional smoke check, not a reproducible release, trained binary,
performance result, or integration with another app.

Bank and section selection is explicit. Automatic stack detection, project
artifact discovery, maturity gating, question execution, yes/no or inconclusive
answers, attunement persistence, and RRFlow/SWAG/MCP integration are not yet
implemented. The existing app intake form is a separate project configuration
questionnaire; it is not this authored governance battery. Source documents also
contain curricula, lexicons, routing rules, gating proposals and open items;
those are not fully transcribed or implemented by the numbered-question catalog.
