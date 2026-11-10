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

On 2026-11-02, the source search located the [original Drive folder](https://drive.google.com/drive/folders/1Tp0fWzSkiH5a9gpbhGeQzRi4pgkjPJmh),
including the six stack curriculum drafts and the separate
[stack-routing draft](https://docs.google.com/document/d/1r0KlUR9_ZK-9IJjhAs7uze0ENdxqU8hZ7Ay3-Kop67w/edit).
A private local archive of ten retrieved notes, including the RRFlow/LFG
addendum, is under `ml/lfg-encoder/dataset/private/source-notes-20261102` in
the original checkout. Its manifest records source URLs, modification times,
and SHA-256 hashes of the UTF-8, LF-normalized copies. It is Git-ignored and is
not bundled with LFG. Retrieval preserves the source material but does not
mean the previously omitted curriculum and routing sections have been converted
into executable configuration or validated against the current code.

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

In Inspector, importing a local `boundary-project` preview now offers its
supplied artifacts as **proposed** answers beside the original catalog questions.
The operator can edit an answer and explicitly mark it reviewed. A proposal is
not proof that an artifact answers the question; imported provenance is an
operator-supplied claim, not an independent verification. A project ID and an
explicit save create a browser-local profile revision containing the preview,
catalog digest, draft answers and review flags. Loading an earlier revision and
saving it creates a new revision, preserving the prior history. The configured
browser import byte limit also caps the combined saved profile store; a save
that would exceed it is rejected without replacing the existing store.

Profiles are plaintext in the browser's local storage, scoped to its origin and
browser profile. Do not save secrets there. Browser storage is neither encrypted
project storage nor an authenticated multi-user service; clearing site data,
changing browser profiles or changing origin can make the saved history
unavailable. Review flags are local assertions. A stale head is rejected when
detected, but this is not an atomic multi-tab or multi-machine transaction.
There is not yet a server-side profile API or portable sync.

An isolated run of the unified development binary on 2026-11-02 generated a
private preview from the actual configured RRFlow checkout. For the selected
core and Rust banks, it reported four supplied bindings, eleven missing
bindings and 105 questions without an explicit binding. These are counts of
question entries, not answers or verified assertions. The private preview is
`ml/lfg-encoder/dataset/private/rrflow-project-preview-20261102.json` in the
original checkout; it is not included in the repository or binary. The browser
test observed imported evidence as unreviewed proposals. Persistence and
rollback were exercised separately with test data; no RRFlow project answers
were approved on the owner's behalf.

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

For local project evidence, `boundary-project` uses an explicitly selected
configuration to collect file paths and JSON values and feeds them into this
same catalog. The checked-in example is scoped to LFG itself:

```powershell
lfg-server boundary-project --root D:\projects\lfg --config crates/lfg-server/config/project-attunement.lfg.example.json
```

Copy and adapt that configuration for another project. Each source maps a
catalog binding to an inventory of files below a named directory, a JSON
pointer, or the complete UTF-8 text of an explicitly selected file. For example,
`{"kind":"text","binding":"workspace layout","path":"Cargo.toml"}` attaches
the workspace manifest to the corresponding authored Rust question. The AI or
reviewer still needs to interpret that evidence; the manifest is not an answer.
Paths are relative to the selected root. An
inventory extension list of `[]` includes every file; otherwise list extensions
without dots. Set `recursive` and the entry, depth, and JSON file byte limits
deliberately. Entry limits are shared across inventories and count directories
as well as files. Exceeding a limit or encountering a symlink is an error,
rather than a partial inventory presented as complete. An absent path or JSON
pointer stays missing. A JSON file is read in full to select the pointer, but
only the selected value is returned.

The result includes the catalog digest, configuration digest, source paths,
inventory path-list digests or file SHA-256 hashes, and the ordinary
attunement preview. An inventory digest covers names, not file contents. The
command writes the preview to standard output and does not persist it; paths
and selected values may themselves be sensitive. A text source includes the
entire file, not an excerpt; only select files whose complete contents may be
included in the preview. Existing credential-shaped checks reject some inputs
but cannot detect secrets embedded inside text. The configured root,
files and local machine must be trusted: the path checks are not a sandbox
against concurrent filesystem changes. No automatic stack detection or
judgment of question answers is performed.

The LFG UI's **Inspector → Boundary questions** panel displays the authored
catalog, grouped by bank. To inspect a project preview there, save the output
from the configured command to a private JSON file, for example:

```powershell
lfg-server boundary-project --root D:\localdev\devlabs\rrflow --config crates/lfg-server/config/project-attunement.rrflow.example.json |
  Set-Content -Encoding utf8 target/rrflow-question-preview-private.json
```

Choose that file in the panel. The browser reads it locally and checks that it
names the currently served catalog and known question IDs; it displays question
text from the catalog, not from the imported file. The panel distinguishes
supplied artifacts, missing artifacts, and questions omitted from the preview.
Source paths and hashes shown are claims in the imported file, not independently
verified evidence. The file can contain complete selected texts and project
paths, and remains the operator's responsibility to store privately. Import is
limited by the server's configured `maxHttpBodyBytes` value, even though the
file is not sent to the server. Rerun and reimport to reflect project changes.
This manual preview workflow does not connect RRFlow or another app directly,
persist review decisions, answer the questions, or create training examples.

On 2026-10-31, the example ran against the local LFG checkout: it collected
12 Rust source paths for `module inventory` and 45 paths for `file tree`,
supplying those bindings to three of the 120 selected core and Rust questions.
Nine other distinct binding names remained missing. This is a local functional
check against real project paths, not human review, a training corpus, or a
completed app integration.

The same binary was also run against the authorized RRFlow checkout with a
separate configuration selecting `crates/kernel/rrd-core/src` and
`crates/kernel`. It collected 14 Rust module paths and 31 file paths and
attached the two scoped inventories to three questions; the other nine binding
names remained missing. That earlier run read project paths only, not the file contents,
and does not establish an integrated RRFlow deployment. The adaptable example
is `crates/lfg-server/config/project-attunement.rrflow.example.json`.

After adding the text source, a local debug build ran that RRFlow configuration
again. It attached the selected checkout's `Cargo.toml` to the Rust workspace
question; both its returned UTF-8 content and reported SHA-256 matched the
file. Eight distinct bindings remained missing among the 120 selected
questions. This supplies evidence only; it does not assess the question,
establish deployed integration, or provide reviewed training examples.

Bank and section selection is explicit. Automatic stack detection, broad project
artifact discovery, maturity gating, question execution, yes/no or inconclusive
answers, attunement persistence, and RRFlow/SWAG/MCP integration are not yet
implemented. The existing app intake form is a separate project configuration
questionnaire; it is not this authored governance battery. Source documents also
contain curricula, lexicons, routing rules, gating proposals and open items;
those are not fully transcribed or implemented by the numbered-question catalog.
