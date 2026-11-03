# Private Codex turn evidence

`ml/lfg-encoder/codex_turn_evidence.py` links selected, inventoried Codex user
requests to preceding `turn_context` records with the same recorded turn ID.
It recovers the configured model and effort, not a recommendation, proof of
user preference, successful outcome, agent count, reviewed assessment, or
training label. Provider settings and session histories may themselves contain
private information. Keep the output local and out of Git, and review and redact
it before using it in training. Pattern screening or a hash is not proof that
secrets are absent.

The operator selects an existing development plan and configures the input,
prefix, line, record, context and metadata bounds. The full plan is checked for
test splits before any source is opened. This tool reads selected plan sources
and, for Codex records, verifies the stored original session prefix and selected
message hashes. It matches only the exact recorded turn ID; an absent ID or
conflicting model/effort settings yields unknown context. Repeated contexts with
the same turn ID and identical model/effort values are represented by their
latest matching line and a count. The source file is checked for changes while
read, but is not locked afterward.

The output contains candidate IDs, declared project and family, split, original
text and session-prefix hashes, hashed turn IDs, model/effort settings and the
context line and hash. The extractor does not copy prompt text or source-file
paths, but input-provided IDs, project names and model metadata may themselves
contain private information. Each
row records plan, configuration, implementation and verifier hashes, Python
version, `outcome: unknown`, `agent_count: null`, `review_status: unreviewed`,
and `promotion_eligible: false`. These hashes are sensitive and are not
anonymization; predictable text can be guessed. Windows permissions follow the
parent directory's ACL. An output inside a Git worktree must be ignored and not
already tracked, and an existing file is never overwritten.

Project and project-group split declarations, and family and normalized-text
crossings for the selected Codex candidates, are checked. This does not verify
the true project identity, all cross-provider duplicates, shared context,
semantic leakage, ownership, or evaluation independence. Claude candidates are
not processed. A source changing during extraction causes the run to fail; use
a new output path after the session stabilizes. The tool has no background
collection, and does not modify session files or the review workspace.

```powershell
python ml/lfg-encoder/codex_turn_evidence.py `
  --plan PRIVATE_DEVELOPMENT_PLAN.json `
  --config PRIVATE_REVIEWED_CONFIGURATION.json `
  --output NEW_PRIVATE_EVIDENCE.jsonl
```

The [example configuration](../../ml/lfg-encoder/codex-turn-evidence.example.json)
is a starting point; adapt the limits to the source and machine. The program
prints aggregate counts and hashes, not the model names or requests.

## Explicit user-requested routing evidence

`user_requested_observation.py` defines a separate
`lfg.user-requested-routing-observation.v1` contract. It can verify one
caller-selected inventory identity and exact UTF-8 byte spans in its selected
user message. The operator config supplies phase labels; a caller supplies one
configured phase label and the interpretation that links it to an evidence
span. A configured phase label is a category and may also occur in the source.
Optional model/effort wording must match selected source bytes exactly. The
caller also supplies an explicit polarity from a configurable set; the code
does not infer polarity from negation, quotation, or surrounding wording.
Multiple or conflicting claims are retained separately. Every claim remains
unreviewed and is not promoted to an actual preference without separate
review. The transferable value stores only
the span offsets and hashes, not the source path, user message, or selected
span text. The inventory-time whole-file digest is retained as an inventory
claim; when a session grows, this helper verifies the selected prefix/line and
does not claim that trailing bytes were unchanged or append-only. Model and
effort names are not mapped to provider IDs or a catalog.

The observation keeps provider entitlement, executor effort, execution,
outcome, and total agent count unknown/null, with review unreviewed and
promotion ineligible. It is not a teacher trace, success label, or evidence that
the request was executed. The pure consumer validates and summarizes only the
`user_requested` schema; it does not fill unknowns, create teacher response
rows, infer success, or assign training weights. Safe extension data stays in
an inert `extensions` object; reserved semantic keys and detected credential
keys are rejected, and values are screened with both the baseline and supplied policy.
Screening is heuristic, not proof that secrets are absent. The consumer does
not reopen source files or authenticate an observation hash; source identity is
checked by the builder, while a hash alone is not an independent signature.

All byte, span, extension, observation, and reserved-manifest limits come from
the supplied exact-key config. If an operator supplies a reserved-session
manifest, the module checks its unknown/partial coverage and exclusion before
opening the selected source, pins its digest, then rereads it after building
the in-memory observation. The owner-only manifest reader fails closed on
Windows. When a nondefault screening policy is used, supply the same policy to
manifest pinning and observation construction. The final reread narrows but does not lock out a concurrent manifest
change after return; keep the manifest in a protected directory and avoid
concurrent edits. No observation writer is integrated here. The source-bound
writer and this branch's ancestry require independent review before any real
owner observation can be materialized. The tests use synthetic fixture data
only.

On POSIX, selected-source verification opens through no-follow directory and
file handles and rejects multiply-linked session files before reading bytes.
Reserved manifests may also declare pre-open inventory identities: selected
text digest, selected line number and digest, and prefix byte count and digest.
Any one matching selected-text digest, selected-line identity, or prefix
identity excludes the candidate before opening it, even if the inventory's
whole-file digest or path changes; an exact whole-file digest is also an
independent match. Coverage remains `unknown` or `partial`.
This pre-open comparison relies on a trusted, protected inventory manifest and
candidate inventory record; these hashes do not authenticate their source. A forged identity cannot be
validated against source bytes without opening those bytes; this is not an
absolute no-heldout-access guarantee against malicious metadata. Only these
declared digests/coordinates and path-derived group IDs are compared.

The Windows source-bound builder, verifier, observation builder, and reserved
manifest readers now fail closed before path access because no proven
reparse-point-safe traversal is available. This is a functional limitation on
Windows direct-library use as well as the CLI. These changes do not
retroactively revalidate or qualify any previously materialized batch or
observation.

On 2026-11-02, the local development plan's 555 selected Codex candidates (520
training and 35 validation) all had an exact turn-ID match with model and effort
present. The remaining 90 plan candidates are Claude messages and were not
processed. Repeated but consistent context records were present. The held-out
Makers-Impulse plan was not supplied or opened. The output remains private,
unreviewed usage evidence; no assessment heads or owned LFG model were trained
from it. Exact matching describes recorded execution settings only, and does not
establish which settings were optimal or that a task succeeded.
Two runs with the same inputs, code and Python version produced identical
output bytes. That local check does not establish cross-platform reproducibility.

## Source-bound current provider snapshot

`source_bound_snapshot.py` separately verifies one selected Codex user-message
line and its exact prefix digest, then records the current `models_cache.json`
digest, fetched time, listed IDs, descriptions, and provider-supported effort
fields. Model rows are compared against the reconstructed cache on
revalidation, and provider extension strings are screened with the configured
policy before private persistence. Session exclusion identity is based on a
canonical project-label-independent path; the inventory's project association
is explicitly unverified. A full session-file digest may drift when later log events append; the
snapshot keeps that inventory-time claim separate from the current whole-file
digest, which is unknown. The selected prefix and line must still verify.

`exploratory_train` is an explicit collection assignment when no reserved final
test has been independently verified. It leaves `heldout_separation`, paid
entitlement, and historical effort unknown. The collector's source-bound
preflight binds the staged projection, one request, context, policy, and
selection manifest before any teacher call. Staging omits full source paths,
unselected cache extensions, and all validation/test data. See
[local teacher collection](LOCAL-ASSESSMENT-TEACHER.md) for the privacy and
remaining evaluation limits.
