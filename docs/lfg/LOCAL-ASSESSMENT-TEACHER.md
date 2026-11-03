# Local assessment teacher collection

The owner authorizes distillation on selected private projects and chats.
`select_teacher_requests.py` deterministically samples train requests across
project labels from a previously screened dataset, applies the selected local
screening policy again and writes a new private selection. It verifies the
train file's recorded digest and does not open validation or test files. The
sampling seed, per-project and total limits, sizes and policy are configuration.
Project labels are source claims, not proof of independent projects or chats.

`collect_local_teacher.py` sends those selected requests, one at a time, to an
explicitly configured OpenAI-compatible chat endpoint on a literal loopback
IP. It does not use a proxy or follow redirects. Its API key is read from the
configured environment variable and is not written to the configuration,
records or report. Set the served model, identity description, instruction,
source origin, required response fields, timeout and size/token limits in a private copy of
the [example configuration](../../ml/lfg-encoder/local-teacher-collection.example.json).
The loopback restriction makes the private-data boundary explicit; run the
collector on the machine hosting the authorized teacher.

The collector requires a complete JSON response from the configured model,
screens string values and keys, and stores an immutable private record per
request. Source, config, policy and code hashes are recorded. `--resume` reuses
verified records only when those identities still match; an OS lock prevents
two collector processes from sharing one output. A failed request can leave a
partial directory and no final manifest. Completed per-project JSONL files can
be passed as the configured `owner_chat` or `owner_project` train sources to `distill.py build-sft` after the
operator supplies a plan. Record metadata preserves source claims; the
collector cannot independently establish original session provenance.

Every output is explicitly **unreviewed**, with **unknown outcome** and no
promotion eligibility. Teacher identity is an operator claim and the served
weights may not be pinned. Screening is pattern-based and can miss secrets;
the selection, teacher responses, logs and derived weights must stay private.
Prompt text may appear in the local teacher service's own logs. The tool does
not establish that a teacher's recommendation is optimal or a task succeeded.
Extra response and configuration fields are accepted; required fields must be
present after screening. Completed traces are not an independent evaluation.

Optional `request_extensions` are passed to the local chat endpoint, for
example to configure the selected model's chat-template reasoning mode. The
collector rejects extensions that would override its model, messages, token
limit, temperature, response format or streaming behavior. Extensions and
their values are part of the immutable collection configuration and therefore
part of the recorded run identity.

Optional project context can be supplied with `--project-context`. The
`lfg.project-training-context.v1` file has a `projects` object keyed by the
selected request's project label. Each entry has UTF-8 `text` and lowercase
`sha256` equal to the digest of those exact bytes. The collector validates the
file and each entry, applies the configured screen policy, requires an entry
for every selected project, and records the context file digest in the
immutable run identity. `max_context_bytes` bounds each context and the fully
rendered teacher user request.

Context is operator-supplied opaque transport. The context file has no source
record identity or split field, so the software cannot establish that its
contents are derived only from train-split material or prove held-out material
was excluded. The caller must attest that boundary; source-bound context
provenance is not implemented.

Teacher collection and `/api/lfg/assess/owned` share the same user rendering:
context text, two newline characters, then request text. The API accepts
optional `context` and `contextSha256` alongside the existing `text`; it
rejects a missing or incorrect digest, empty context, or context over its
configured bound. Without context fields, the owned API and collector retain
the legacy prompt bytes. Teacher traces store the rendered request verbatim,
and the existing distillation and SFT paths train on those exact bytes.
The collector also records a digest of its exact teacher system prompt;
distillation requires a supplied digest to match the plan system prompt and
preserves trace extensions in SFT metadata. Context-bearing traces require the
teacher system digest and it must match the distillation plan. The current SFT
trainer records the system prompt digest and a canonical digest of all six
framing delimiters; the owned service recomputes and checks those values against
the serving system and delimiter fields. Legacy artifacts without these links
remain compatible but are explicitly unlinked.
Context-file and trace extension fields are retained as metadata and do not
change rendering or routing behavior.

Legacy `--project-context` remains an opaque transport contract: its digest
proves byte identity only. The optional source-bound path adds a separately
verified snapshot and collector preflight for one exact Codex user message and
the current provider-tool cache. It verifies the selected source line and byte
prefix, preserves the source's original whole-file digest claim separately
from the current unknown whole-file digest, and binds the screened request,
provider context, selection manifest, and staged copy with SHA-256 identities.
Append-only session-log growth after the selected line is allowed; a changed
prefix or selected line fails validation.

Source-bound snapshots preserve current listed model metadata as provider
claims. A provider listing does not establish entitlement; subscription
entitlement and historical session effort remain `unknown`. Every persisted
provider string, including IDs, labels, descriptions, effort names, timestamps
and extension strings, is screened by the configured policy. If screening
would change an ID or effort identity, collection fails; other provider text
is stored only in screened form. Safe extension values remain inert, and
source digests continue to bind the original cache bytes. The staged
projection carries the original cache digest and selected provider fields,
omits raw source paths and provider extension payloads, and records that it is
a copy of an origin-verified snapshot. Project association comes from an
inventory label and is explicitly marked unverified. Session group identity
does not include that mutable label. Windows writes fail closed unless the
dedicated output directory has a protected current-user, SYSTEM, and
Administrators ACL, untrusted ancestors cannot delete child paths or change
ACLs/take ownership, and the original open file handle still has the same
identity as the named path. Ancestors may grant read/traverse or create-only
rights that cannot replace an already existing path component. Writes use the
original handle without reopening the path.

When no separately verified reserved final-test group exists, the caller can
explicitly use `exploratory_train`. This is an exploratory collection assignment
with `heldout_separation: unknown`; it does not claim an independent holdout or
blinded evaluation. Known reserved project candidates are rejected before the
selected session is opened, and no test or validation source is opened. Known
session exclusions use project-label-independent session IDs; normalized
project-label aliases are an additional preflight check, but inventory labels
are unverified and do not prove project identity. Reserved-set coverage is
therefore incomplete. The Makers Impulse candidate external-project group
remains excluded from this selection. A separate, independently reserved
evaluation group is still needed before any quality claim or promotion.
The CLI accepts `--reserved-session-ids` with a private
`lfg.reserved-session-groups.v1` manifest containing SHA-256 session IDs and an
`unknown` or `partial` coverage marker. It binds the manifest digest into the
snapshot and rejects matching IDs before opening the session source. It will
not accept a completeness claim without separate verifiable evidence.

The collector's `--source-bound-snapshot` and `--source-bound-bundle` options
recheck the staged snapshot, request, context, bundle, and screening-policy
digests after train-row preflight, immediately before collection, and before
finalizing the run. Traces retain the snapshot and bundle digests with the
context provenance. The originating Windows session is verified before
materialization; the staged projection is not a live lock on that source, so
the trace describes the verified copy, not an immutable source file. Screening
is still a claim, not proof that all secrets are absent. Teacher outputs remain
unreviewed proposals with unknown outcomes. No subscription, project success,
optimality, or evaluation result is inferred.

Context files, selections, snapshots, traces, and resulting weights must remain
private and outside Git. The current request-plus-context byte estimate is
close to the checkpoint context limit; do not train or claim a fitting SFT
example until the actual framed bytes/tokens and response are measured without
truncation.

The optional `on_incomplete` policy defaults to `fail`. When explicitly set to
`skip`, an incomplete or invalid individual response is recorded as a failure
and the collector continues; the final manifest reports requested, successful,
and failed counts. HTTP server errors still fail the job for supervision to
retry. Failed records contain an identifier and digest, not a fabricated label.
For subsequent batches the selector can exclude prior request IDs and
SHA-256 digests of normalized, rescreened request text. Exact normalized
deduplication does not prove semantic or conversation independence.

For short follow-ups, `contextual_teacher_selection.py` can instead verify the
original source prefix and combine the current user message with a bounded
number of preceding user messages from the same source. It reads only declared
training source files, checks the screened candidate and source digests, screens
the combined text with the configured policy, and emits the same private
selection format accepted by the collector. Selection limits, framing and
exclusions are configurable in
[the example](../../ml/lfg-encoder/contextual-teacher-selection.example.json).
Codex context contains user messages only; Claude context follows available
same-session parent links. Either can omit important assistant responses, and
an authentic conversation is not proof that a teacher's assessment is correct
or that the project outcome was successful. Pattern screening remains
incomplete, so the selection, source paths, outputs and weights stay private.

```powershell
python ml/lfg-encoder/contextual_teacher_selection.py --plan PRIVATE_TRAIN_PLAN.json `
  --dataset PRIVATE_SCREENED_DATASET --policy PRIVATE_SCREEN_POLICY.json `
  --config PRIVATE_CONTEXT_SELECTION_CONFIG.json --output NEW_PRIVATE_SELECTION
```

On 2026-11-02, a selection from the owner's already screened training split
produced 95 additional contextual requests across 11 declared project labels;
four candidates lacked usable prior context and two failed source verification.
Eighty-four selected context windows were truncated by the configured bounds.
The validation and test sources were not opened. The selection remains
unreviewed. Its resumable local teacher job produced 94 responses and recorded
one incomplete response. The resulting 470-record combined dataset was
delivered to the Spark and accepted for another checkpointed fine-tune.

```powershell
python ml/lfg-encoder/select_teacher_requests.py --dataset PRIVATE_SCREENED_DATASET `
  --policy PRIVATE_SCREEN_POLICY.json --config PRIVATE_SELECTION_CONFIG.json `
  --output NEW_PRIVATE_SELECTION_DIRECTORY

# Run on the teacher host with the configured key already in its environment.
python ml/lfg-encoder/collect_local_teacher.py --selection PRIVATE_SELECTION_DIRECTORY `
  --config PRIVATE_TEACHER_CONFIG.json --policy PRIVATE_SCREEN_POLICY.json `
  --output NEW_PRIVATE_COLLECTION_DIRECTORY
```

An initial one-request attempt on 2026-11-02 reached the owner's existing
local SWAG endpoint but failed with an HTTP error and produced no completed
trace. The container was observed running immediately before and after that
attempt; a subsequent check found it exited with status 0. The cause has not
been established. A later local log inspection found an `EngineDeadError`, an
HTTP 500 and a shutdown marker; those observations do not establish what caused
the engine failure. The service was not restarted and no further requests were
sent. The pilot selection contained one screened real request; a separate
balanced selection contains 21 requests across 11 declared project labels,
but no teacher responses have been collected for it.
