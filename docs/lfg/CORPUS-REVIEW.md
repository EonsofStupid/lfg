# Corpus review and separation

The training and evaluation corpus should use authorized, representative
project examples with reviewed labels and a held-out evaluation source. A
project's commit message, change description, or generated summary is not
automatically the user's original prompt, an independent example, or a
verified outcome. Keep that distinction in source provenance.

`corpus_audit.py` accepts a versioned operator-authored plan. Each source names
a JSONL file, a declared project, and a `train`, `validation`, or `test`
partition. All paths are resolved relative to the plan; absolute paths are
also accepted. The [plan template](../../ml/lfg-encoder/corpus-plan.template.json)
starts with no sources. When copying it elsewhere, update the taxonomy path as
well. Use an ignored, access-controlled location for plans, datasets, review
queues and training snapshots that refer to private project material.

Sources can declare a `project_group` to keep related repositories or versions
of the same product in one split. It defaults to the declared project name;
every source for one project must use the same group. The audit rejects a group
that appears in multiple splits and records the effective group for each source.
Groups are declarations: omitted or incorrect relationships are not discovered
automatically and cannot establish independence.

The audit checks unique example IDs, source-project declarations, taxonomy
membership, project and family separation, and normalized exact-text overlap
across splits. It reports within-split exact duplicates and draft label support.
It cannot identify every paraphrase, related project, shared history, uncertain
label, or unauthorized source. Project names and source records are operator
declarations, not independently verified identities. The output, especially
example IDs and text-derived hashes, is sensitive metadata even though it does
not contain the prompt text.

For a review workflow:

1. Populate a private plan, preserving source and project metadata on each
   example. Decide project separation before making a final holdout. Audit with
   `python ml/lfg-encoder/corpus_audit.py --plan PLAN`.
2. After drafting labels, create a **new** pending-only queue with
   `--review-queue PRIVATE_QUEUE.jsonl`. It never approves examples. Review the
   referenced records and their source against the labeling guide. A person
   records `decision: "approved"`, a reviewer identifier and an ISO-8601
   timestamp with timezone, and sets the source record's `verified` flag only
   after review. Record disagreements or exclusions outside the training set.
3. Point the plan's `review_ledger` at that private JSONL queue and rerun the
   audit. The approval digest binds the record, declared project and split, and
   taxonomy. Editing text, labels, provenance, family, split or taxonomy makes
   the old approval stale; create a new queue and review again. The `verified`
   flag itself is excluded from the digest so it can be set after review.
   Queues are created exclusively and never overwrite an existing file.
4. Once every declared train, validation and test record is approved and
   verified, create a **new** training snapshot with `--training-output
   PRIVATE_TRAIN.jsonl`. Assembly repeats the audit and rejects changed inputs,
   duplicate normalized text within a split, and incomplete review. The
   training command additionally enforces the configured sample threshold and
   `--audited-plan PLAN`; the input must exactly match the audited train rows.
   Audited training loads the separate validation partition and verifies its
   source hashes and audit again. It rejects validation labels missing from the
   training partition; test rows are not supplied to the trainer. Before loading
   a model, it also checks positive and negative counts for every taxonomy label
   in both training and validation. The default requirements are in
   `ml/lfg-encoder/training-support-policy.v1.json`; select a different file
   with `--support-policy` to raise defaults or set per-label thresholds. Every
   threshold must be a positive integer. A minimum of one is a basic coverage
   check, not adequate statistical support or evidence of quality.
   The full taxonomy is included in the trained model, and the policy digest
   and support counts are bound to the run manifest. The test partition is not
   consulted for these support checks.
   Its run manifest records audit, plan, taxonomy, ledger, source and code
   digests. Changing them prevents an ordinary resume.

An approval is an attestation, not cryptographic proof that a person reviewed a
record, that a source is licensed, or that a label is correct. The audit reads
the declared test file; access controls and a separate evaluation procedure are
still required for a genuinely blinded test. Sample counts alone do not qualify
a model. `--allow-below-gate` intentionally bypasses the normal corpus gate for
development, and its artifacts remain experimental. A different versioned
promotion-policy file can be selected with `--promotion-policy`; changing its
sample threshold does not establish adequate data or quality.

## Local review workspace

`review_corpus.py` provides a terminal-based review workflow for an existing
unreviewed plan. It copies the plan, taxonomy and sources into a new local
workspace and starts every decision as pending. It refuses to initialize from
an existing ledger or records already marked verified. It never contacts a
network service. The workspace contains the original private text: choose an
access-controlled directory and do not commit it. Git ignore is not encryption;
on Windows, access depends on the directory's inherited ACL.

```powershell
python ml/lfg-encoder/review_corpus.py init --plan PRIVATE_PLAN.json --workspace NEW_PRIVATE_DIRECTORY
python ml/lfg-encoder/review_corpus.py review --workspace PRIVATE_DIRECTORY --reviewer REVIEWER_ID
python ml/lfg-encoder/review_corpus.py status --workspace PRIVATE_DIRECTORY
python ml/lfg-encoder/review_corpus.py export --workspace PRIVATE_DIRECTORY --output NEW_PRIVATE_EXPORT_DIRECTORY
```

Review displays the request, source, review note and proposed labels. The
reviewer can edit all label groups, skip, or explicitly approve or reject an
item. Approving requires typing `APPROVE`; rejecting requires `REJECT` and is
not reversible within that workspace. Blank label groups mean negatives, not
unknowns. If a request, attribution, label, source permission or potential
secret is uncertain, skip or reject it rather than approving it. Review the
actual source and context separately; the tool displays provenance but does
not establish authorship, licensing or secret removal. The reviewer identifier
and time are recorded attestations, not identity verification.

Decisions are bound to the displayed record, taxonomy, project and split. An
approval made stale by an edit is not exported. If a process stops between
writing a source and its ledger, the item stays pending and must be explicitly
reviewed again. The workspace lock coordinates this tool's processes only;
avoid editing workspace files concurrently. If a crash leaves `review.lock`,
first verify no reviewer is running before removing it. Export creates a new
plan containing only current approvals and excludes pending, rejected and stale
items. It does not assert that there are enough examples or suitable project
splits for training: the existing corpus audit, label support and sample gates
still apply.

### Combining reviewed workspaces

`combine_reviewed_workspaces.py` joins current approvals from separately reviewed
workspaces into a new private plan, ledger, taxonomy and train-only JSONL. It
locks each workspace while reading, checks matching taxonomies, exports only
current approvals, and applies the existing audit to the combined result. It
rejects duplicate IDs, cross-split project/group/family or normalized-text
overlap, within-split normalized duplicates and missing partitions; stale
approvals are excluded. Every supplied workspace must contribute at least one current
approval. It creates the output directory exclusively after staging and auditing;
if an error occurs during the final copy, inspect the incomplete output and use
a new path. Its manifest binds input audit, code, policy and output digests.

Check the actual private workspaces without creating an output:

```powershell
python ml/lfg-encoder/combine_reviewed_workspaces.py `
  --workspace ml/lfg-encoder/dataset/private/codex-session-review-draft.v1/workspace `
  --workspace ml/lfg-encoder/dataset/private/onemanwho-validation-review-draft.v1/workspace `
  --workspace ml/lfg-encoder/dataset/private/makers-heldout-review-draft.v4/workspace `
  --status
```

After genuine reviews, replace `--status` with `--output NEW_PRIVATE_DIRECTORY`.
The output's `plan.json` and `training.jsonl` can then be supplied to the audited
training command. The sample status uses `promotion-policy.json` by default;
select the same policy intended for training with `--promotion-policy`. The
candidate shortfall is optimistic: it counts every record in the training
workspaces, including pending, rejected and stale records, so it is a lower
bound on how much additional data may be needed. The count alone does not check
label support, independence or model quality. Assembly can produce a reviewed
snapshot before it meets the training sample and label-support gates; it does
not bypass those gates or mark the snapshot promotable. Audit currently reads the
test records and therefore does not enforce a blinded holdout.

On 2026-10-31 the status for these three workspaces was 226 train, 35 validation
and 785 test records, all pending. The default policy required 500 reviewed
training records: at least 274 additional candidates are needed even if every
current training candidate is approved. Assembly correctly rejected the current
workspaces without creating an output. A test-only fixture exercised approved
records through the combined plan and the existing training/validation audit;
that fixture is not human-reviewed data or quality evidence.

The current real-request review workspace is
`dataset/private/real-request-review-workspace.v1`. It was initialized with
38 pending items and no approvals. Its LFG validation group was used for an
earlier experiment and it has no test source; review alone cannot turn this
plan into an independent final evaluation.

### Local browser review

The same workspace can be reviewed in a browser using the Python training
tools. This is a local corpus-review interface, not the server's developer UI
or an LLM configuration editor:

```powershell
python ml/lfg-encoder/review_web.py --workspace ml/lfg-encoder/dataset/private/real-request-review-workspace.v1 --reviewer REVIEWER_ID
```

Open the complete URL printed in the terminal. The server chooses a loopback
port by default; `--port` selects one explicitly. Its configurable request,
display and timeout limits come from `review-web-config.v1.json`, or an
operator-selected `--config` file. It runs until Ctrl+C and holds the same
workspace lock as the terminal tool. A session token in the URL authorizes
access to the local workspace for that server session; do not share it. The
page removes the token from the address bar after loading and uses no external
resources. The server rejects non-loopback host headers, requests without the
token, and cross-origin decisions. Responses are marked `no-store`.

The browser displays one pending request, source and proposed labels at a time.
For a `codex-session-user-text` record, **Check original source** rereads the
bounded original session prefix, verifies its recorded hash and selected line,
and compares the original message with the review copy. The source check is
on demand; it does not save an approval or establish authorship, context,
permission, privacy or label correctness. Later content appended to the session
does not invalidate the checked prefix. A changed or unavailable prefix, a
different review copy, unsupported source type, or an interrupted review write
produces an error. The browser does not display additional session messages.
The `source_verification` object in the web configuration sets the maximum
session-prefix and line sizes. Configurations without that optional object
disable this check; it is independent of whether a reviewer can decide a
record. Local source paths and hashes remain private metadata.
Approving requires the attestation checkbox and typing `APPROVE`; rejection
requires typing `REJECT`. The underlying workspace checks the displayed digest,
taxonomy, project and split before saving. Skipped items remain pending.
To remove sensitive text, select it in the request, enter a replacement and
choose **Replace selection**. Repeat as needed and review the resulting request
and its labels before approving. Replacements stay in browser memory until
approval. The saved record binds sequential character ranges and replacement
text to hashes of the request before and after the changes; its original
source provenance stays attached. A redacted request that duplicates another
workspace record is rejected. The tool does not decide whether redactions are
complete or preserve meaning. Source files and draft copies still contain the
original text, so approving a redaction does not erase private data. The
terminal review path does not offer redaction.
Decisions, export and subsequent training use the same rules as the terminal
workflow above. The interface does not verify source authorship, user identity,
data rights or secrets, and it does not create approvals on its own. Local
processes or browser extensions with access to the session URL or workspace
may access private data; this is not a hosted authentication boundary.

`smoke_reviewed_gate.py` tests the gate through an actual short model-training
call using invented fixture records and simulated approvals. It creates a new
output directory, requires an explicitly chosen local model snapshot, immutable
revision and bound environment inputs, and disables Hugging Face network access. It deliberately lowers a
copy of the policy's sample threshold for its two training fixtures; its
output must never be used as a release candidate or as evidence of human
review. For example, with a compatible local model snapshot and environment:

```powershell
.\.venv\Scripts\python.exe ml/lfg-encoder/smoke_reviewed_gate.py --base-model LOCAL_MODEL_DIR --model-revision COMMIT --model-manifest MODEL_MANIFEST --environment-lock LOCK_FILE --wheelhouse WHEELHOUSE --environment-allowed-extra pip --output target/review-gate-smoke --epochs 1 --max-length 32 --batch-size 1 --checkpoint-steps 1
```

The requested output directory must not exist. A model that already has a
different-sized classification head may require the explicit
`--allow-mismatched-sizes` option; inspect all model-loading warnings. The
simulated approvals and fixture policy are stored with the test artifact so
they cannot be confused with the actual project review ledger.
The current fixture deliberately assigns artificial labels to exercise both
positive and negative support for all taxonomy labels in its two training and
two validation records; those labels make no semantic claims.
On 2026-10-31, the fixture-only audited path also completed with the new
wheelhouse and local-model input checks. Its simulated approvals and output
remain test fixtures, not evidence of a reviewed corpus.

On 2026-10-30, this smoke command completed offline on the Windows development
machine using the cached tiny random BERT revision recorded in the
[Spark smoke evidence](SPARK-SMOKE-EVIDENCE.md). It assembled two simulated
reviewed training examples, bound the fixture audit into the run manifest, and
saved model weights. This exercises the gate; the fixture approvals, two-example
threshold and resulting model are not real training or quality evidence.
After the audited-validation fix, a separate offline fixture run checked that
both train records reached training, the validation record came from the
declared validation project, and the test record was not passed to either
training or validation. It explicitly replaced the tiny test model's mismatched
classification head. This proves the split wiring, not model quality.

## Current local inventory

### Local Codex project-session messages

The local session history is another possible source of original requests. The
`inventory_codex_sessions.py` tool selects session files only when their
`session_meta` working directory falls within an explicitly supplied project
root. Within those files it inventories only single-text user messages whose
record metadata explicitly identifies the content kind as `user.text`. It
does not copy message text, assign labels, approve records, or choose splits. It
skips an entire selected file if its JSON is malformed, the configured byte
limits are exceeded, or the file changes while being scanned. The file is read
in bounded lines, including for large sessions. A working directory and the
`user.text` marker are source claims, not independent proof of authorship,
project ownership, data rights, safety, or representativeness. Messages may be
follow-ups, duplicates, or include private data and secrets.

The inventory records the original session file's full SHA-256 at scan time,
plus the SHA-256 and byte length of the source prefix through each selected
message and the individual line and text hashes. The prefix lets an inspector
verify an earlier message after an ordinary append; it does not assert that the
file's later content is unchanged. A rewritten prefix or changed line fails
inspection. The inspector checks the marker and hashes before displaying the
text, but does not verify labels, attribution, context, or data rights.

For example, from the repository root, choose the session directory, explicit
project roots, and a new private output file (the CLI accepts multiple
`--project-root` arguments):

```powershell
python ml/lfg-encoder/inventory_codex_sessions.py --sessions-root SESSION_DIR --project-root project=PROJECT_DIR --output PRIVATE_NEW_INVENTORY.jsonl
python ml/lfg-encoder/inspect_codex_inventory.py --inventory PRIVATE_NEW_INVENTORY.jsonl --record 1
python ml/lfg-encoder/inspect_codex_inventory.py --inventory PRIVATE_NEW_INVENTORY.jsonl --record 1 --show-text
```

Paths must be absolute for project roots; project roots cannot overlap. Limits
are configurable with `--max-session-bytes` and `--max-line-bytes`. Output is
created exclusively. The last command intentionally prints private text: use it
only in a private terminal. Paths, line numbers and hashes in the inventory are
also sensitive. Git ignore is not encryption or access control.

To prepare a separate review workspace, provide an assignments JSON file that
explicitly chooses a group and provisional split for every included project:

```json
{
  "schema_version": "lfg.codex-review-assignments.v1",
  "projects": {
    "project-name": {"project_group": "related-projects", "split": "train"}
  }
}
```

Then select a new private output directory:

```powershell
python ml/lfg-encoder/prepare_codex_review.py --inventory PRIVATE_NEW_INVENTORY.jsonl --assignments ASSIGNMENTS.json --taxonomy ml/lfg-encoder/evaluation-taxonomy.v1.json --output PRIVATE_NEW_DIRECTORY
python ml/lfg-encoder/review_web.py --workspace PRIVATE_NEW_DIRECTORY/workspace --reviewer REVIEWER_ID
```

Preparation verifies the selected source prefixes in batches, excludes
normalized exact-text duplicates within a split, and refuses duplicate text or
a declared project group across splits. It conservatively groups messages from
the same session into a family; relationships across sessions still require
human investigation. Supply `--reference-plan PLAN` once per existing plan to
exclude normalized text already present there and reject project, group or
session assignments that cross a reference split. The reference plans and the
new workspace must use the same taxonomy. Only supplied reference material is
checked; this cannot detect paraphrases, related sessions, undisclosed prior
exposure or an incorrectly declared group. A private preparation manifest binds
the inventory, assignments, reference audits, source verifier, preparation code,
limits and generated workspace file digests. The output includes raw, potentially sensitive text and
source paths in both the draft files and the review workspace; keep the whole
directory private. If output creation is interrupted, inspect the partial
directory and choose a new path. Missing labels are represented as
`labels_pending`, not negative examples. The normal dataset loader rejects
these records even in development mode. Approval requires a reviewer to select
at least one intent and attest to all labels, source context and privacy. An
unresolved message should be skipped or rejected. The browser review tool can
replace selected sensitive spans before approval; the reviewer must verify the
result, its labels, and whether the redaction is sufficient. Nothing is approved
by the preparation command.

On 2026-10-31 a scan scoped to the local LFG, RRFlow and DevPulse directories
created `dataset/private/codex-project-session-inventory.v4.jsonl`. It scanned
26 selected session files and excluded 46 outside the configured roots. It
identified 271 candidate messages: 198 RRFlow, 43 DevPulse and 30 LFG, with 44
exact text repetitions across the total. Per project, the corresponding unique
exact-text counts are 192, 10 and 28; the RRFlow candidates came from two
sessions, DevPulse from fourteen and LFG from one. These counts do not establish
independence: some messages are short follow-ups, conversations can overlap, and
LFG and RRFlow work may be related. None of these inventory entries has been
approved, labeled, sanitized or assigned a split. One entry from each project
passed prefix verification locally. Some messages may overlap the 38 pending
records or earlier experiments; reconcile them before any training or holdout
assignment. Makers Impulse remains a candidate external-project holdout and
was not included in this scan.

The private `dataset/private/codex-session-review-draft.v1/workspace` was then
created from that inventory using one provisional training group for all three
related projects. It contains 226 pending unlabeled records: 190 RRFlow, 27
LFG and 9 DevPulse; 45 normalized duplicates were excluded. The audit reports
zero approvals and verified records, zero validation or test examples, and no
promotion eligibility. Source prefix verification proves the copied text came
from the inventoried records at preparation time; it does not prove the text is
complete, safe, independent, or attributable to the project owner. Some of this
material overlaps earlier exploratory use. It is review material, not evidence
of a trained or qualified model. A loopback server smoke check displayed the
unlabeled state for one record from each project without recording a decision;
the training loader rejected the pending-label file.

An additional metadata-only session inventory across selected local projects
found 170 candidate messages in 11 sessions. OneManWho had 36 messages from
one session; Shippin/Vaultix directories contain related product histories and
must not be assumed independent merely because their paths differ. Using the
current Codex training draft, Makers test draft and earlier LFG/DevPulse
development plan as references, one normalized duplicate was excluded and a
private OneManWho validation candidate workspace was created at
`dataset/private/onemanwho-validation-review-draft.v1/workspace`. Its 35
records have no labels or approvals. Two preparations with the same inputs on
this machine yielded identical manifests and workspace files.

A temporary combined audit of the three separate workspaces found 226 train,
35 validation and 785 test records, no normalized exact-text overlap or
within-split duplicates, and zero approvals. These assignments remain
provisional. In particular, the entire validation candidate comes from one
session and cannot demonstrate broad or statistically reliable validation;
shared context, source rights, sensitive information, label support and true
project independence still need review. The combined audit does not create a
training plan or qualify a model. The metadata-only inventory is stored at
`dataset/private/additional-project-session-inventory.v1.jsonl` and its explicit
candidate assignment at `dataset/private/onemanwho-validation-assignments.v1.json`.

An exploratory plan on this machine audited 25 constructed RRFlow examples,
four DevPulse excerpts attributed to an operator, and five Makers Impulse
commit-message prompts. All 34 are unverified, and this plan references no
human-review ledger. The first batch contains reconstructed examples rather than original
prompts; the five Makers examples were already inspected and are not a blinded
test. All 34 have zero draft positive `sensitive-data` labels. A pending-only
queue is stored in the Git-ignored private dataset directory. Its provisional
split assignments are for testing this workflow, not a final split decision.

Repository history contains more material, but commit counts are not counts of
valid prompts or independent labeled examples. Representative positive and
negative security cases, human review, independently held-out projects, and
assessment of reuse and privacy remain necessary before a serious training run.

An additional local draft contains 12 operator-attributed excerpts from seven
DevPulse documents at commit `cb7baf0756cbff1ca7a4c18bef81ebdae1e0ee57`.
Each record includes source path, line range, source and text hashes, and the
normalization used to join wrapped quote lines. The stored spans and hashes were
checked against the pinned Git blobs. The original complete requests and their
authorship have not been independently recovered. Their labels and negatives
are AI-proposed, and every record remains unverified. The file is Git-ignored
at `dataset/private/devpulse-attributed-draft.cb7baf07.jsonl`.

A new local provisional plan and pending-only review queue include those rows
without changing the original files. The audit reports 25 reconstructed RRFlow
training examples, 16 DevPulse validation excerpts, and five previously
inspected Makers test examples. None of the 46 is approved or verified, all 16
validation rows have the same provisional intent and capability labels, and
none of the three splits has a positive `sensitive-data` label. These counts
are not evidence of adequate variety, independence, or a blinded test.

The later 11-message LFG / 12-excerpt DevPulse development run also fails the
default label-support requirements: its training group has no positive
`sensitive-data` examples, and its validation group contains only positive
`build.software` and `filesystem.write` proposals. The labels are unverified,
so running this count on them is a diagnostic, not reviewed support. See the
[experiment](REAL-REQUEST-DEVELOPMENT-EXPERIMENT.md).

### Additional attributed DevPulse excerpts

On 2026-10-30, 16 quotations attributed to the operator were selected from
11 DevPulse plan documents at commit `cb7baf0756cbff1ca7a4c18bef81ebdae1e0ee57`.
One was wholly contained in an earlier draft excerpt and was excluded from the
candidate review set. The 15 remaining private records preserve document paths,
line spans, blob hashes, normalized quote text and text hashes. Their spans and
hashes were checked against the pinned Git blobs. The project documents call
the passages verbatim, but the original complete requests and authorship were
not independently recovered. These may be fragments, follow-ups or overlapping
ideas, not 15 independent examples. Exact-text checks do not detect all such
relationships. None has been approved, independently sanitized or verified.

The Git-ignored file is
`dataset/private/devpulse-quoted-deduplicated.cb7baf07.v3.jsonl`, SHA-256
`75b706fc397145691593b7eeb5e9faa6794b7151490eaaca6ea7f604c2e7510d`.
The discarded-overlap source file and the local curation and verification
scripts are also retained in the private directory for traceability. Draft
labels and negatives are AI proposals: 14 are marked as software building and
one as web research. That research excerpt does not establish that external
web access is required; the proposed labels need human review. No record has
a positive `sensitive-data` proposal.

A separate local plan, `dataset/private/real-requests-review.plan.v1.json`,
places these 15 and the 12 earlier DevPulse excerpts in a provisional training
group and the 11 LFG conversation messages in a provisional validation group.
It has no test source. Its pending queue contains 38 records and zero approvals;
the audit found no normalized exact duplicates within the declared splits.
These project assignments are a review aid, not proof of source independence.
The LFG messages trained the earlier experimental model, so this validation
group is not an untouched holdout for that model or a final evaluation set.
The draft DevPulse training group has no `respond.general`, `verify.outcome`,
`text.generate`, or `sensitive-data` positives, while the LFG validation group
proposes the first three. Even after review, this plan cannot pass the training
and coverage gates without more suitable data and an independent test source.

For triage only, the existing rule baseline produced `build.software` on two of
the 14 new excerpts with that proposed label, and `research.web` on the one
with that proposed label. This is comparison with unreviewed guesses, not a
verified accuracy measure. The earlier experimental LFG-trained model also
emitted `build.software` on two of the 14, and did not emit `research.web` on
the one research proposal; its output head does not include that label at all.
This is not an independent test: the model's checkpoint was selected on other
DevPulse excerpts, and the quoted requests may be related. Predictions and
reports remain private.
The candidate Makers Impulse holdout was not read for this new plan.

### Makers checkpoint inventory

`inventory_checkpoint_prompts.py` scans a pinned local Git commit for Replit
checkpoint messages with a strictly delimited `User prompt` section. It prints
aggregate metadata by default. With an explicit `--output`, it exclusively
creates a JSONL file containing the extracted text, source commit and time,
text hash, and hashed session identifier. It assigns no labels, approval, or
split. The output is a source inventory, not the training JSONL schema or a
claim of verified authorship or outcome. Exact duplicates are flagged, not
deleted; sessions and related follow-ups can still overlap in ways the scanner
does not identify. Unrecognized message formats are skipped and their commit
IDs are listed for review.

On 2026-10-30 the scanner read Makers Impulse at commit
`f92162507e421c2c9a4cb971329a35c0e095b5c0`: 1,608 reachable commits, 945
with a prompt marker, 941 strictly extracted, four skipped for format review,
33 exact repeated texts, and 63 recorded sessions. A new Git-ignored local
file, `dataset/private/makers-checkpoint-inventory.f9216250.jsonl`, contains
the unreviewed extracts. Its contents may contain private data or secrets and
have not been sanitized. Git ignore is not encryption or access control. Do not
publish or send the file to a remote trainer without reviewing the material and
its intended use. The 941 records are not independent labeled samples; the
previous five were already inspected, and the rest have not been reviewed.

The whole Makers project remains a candidate external-project evaluation source;
do not train or tune against it merely because more text is available. Establish
the holdout and relationships before inspecting or labeling the remaining
records, then have a human review authorized examples under the labeling guide.
The four skipped messages need separate format inspection. A separate diverse,
reviewed training corpus is still required.

On 2026-10-31 `prepare_checkpoint_review.py` re-extracted the pinned primary
Makers history and compared it with the saved inventory. It excluded the session
containing the five previously inspected prompts (126 records), then excluded
seven records matching the normalized text of either the current Codex review
draft or the earlier LFG/DevPulse development plan and 23 further normalized
duplicates. The resulting private review workspace is
`dataset/private/makers-heldout-review-draft.v4/workspace`: 785 pending unlabeled
records, zero approvals, provisionally assigned to test. The earlier `.v1`
workspace was checked only against the Codex draft; `.v2` and `.v3` predate the
preparation manifest. Use `.v4`. The counts are
records, not independent tasks. A temporary combined audit with the 226-record
Codex draft found no normalized exact overlap and no within-split duplicates;
it contained no validation partition and did not qualify either set.
Repeating the preparation with the same inputs in `.v5` produced a byte-identical
manifest, plan, taxonomy, candidate records and pending ledger on this machine.
This checks repeatability, not labeling or independence.

The preparation tool takes one or more explicit reference plans. It excludes
entire recorded sessions associated with an operator-provided list of previously
inspected commits, their normalized text wherever repeated, normalized matches
to all reference-plan records, and remaining normalized duplicates. It requires
the candidate project and group to be absent from those plans and the taxonomy
to match. The output is a new private, pending-only review workspace; it does
not add the test data to the training plan, approve records, or assign labels.
Its preparation manifest binds the source and review inputs, audit and code
digests, workspace files, size limit, and aggregate exclusion counts; retain the
original private inputs to reproduce or investigate it.
The exclusions are only as complete as the supplied inspection list and reference
plans. Shared context, paraphrases, related conversations, previous unseen
exposure, source authorship and rights are not proven by these checks. This
workspace is a candidate for a future holdout, not evidence of a blinded or
independent test. Keep it out of model development and tuning; a separate
validation group and actual human review are still required.

For example, the local candidate was prepared with:

```powershell
python ml/lfg-encoder/prepare_checkpoint_review.py `
  --repo 'D:\dev\dev-and-sort\Makers-Impulse\Makers-Impulse' `
  --revision f92162507e421c2c9a4cb971329a35c0e095b5c0 `
  --project Makers-Impulse --project-group makers-impulse `
  --source-inventory ml/lfg-encoder/dataset/private/makers-checkpoint-inventory.f9216250.jsonl `
  --exclusions ml/lfg-encoder/dataset/private/makers-impulse-review.v1.jsonl `
  --reference-plan ml/lfg-encoder/dataset/private/codex-session-review-draft.v1/workspace/plan.json `
  --reference-plan ml/lfg-encoder/dataset/private/real-requests-review.plan.v1.json `
  --taxonomy ml/lfg-encoder/dataset/private/codex-session-review-draft.v1/workspace/taxonomy.json `
  --output NEW_PRIVATE_DIRECTORY --max-history-bytes 104857600
```

The new directory must not exist. It and its workspace contain private original
prompts; Git ignore is not encryption. The browser review command above accepts
the workspace path. Its automatic source-check button currently supports Codex
session records only, not these Git checkpoint records. Review their source
commits and surrounding context separately before making any decision.

A second local checkout, `D:/dev/dev-and-sort/makers (2)/makers`, at commit
`ee9831d27407258865ff3676ed8f702e2770e6ff`, contains 19 strictly extracted
checkpoint prompts in one recorded session, with one exact repetition. Its
Git-ignored private inventory is `dataset/private/makers-secondary-inventory.ee9831d2.jsonl`.
The two inventories have no shared prompt commit, exact prompt hash, or session
hash, and their reachable histories have no shared commit. Both package files
nevertheless identify the same Makers Impulse product. Treat them as related
and use the same project group until their lineage and data relationships are
reviewed; different repository names or hashes are not proof of independence.

Run the inventory without writing prompt text:

```powershell
python ml/lfg-encoder/inventory_checkpoint_prompts.py --repo REPO_PATH --revision FULL_COMMIT_SHA --project PROJECT_NAME
```

Add `--output PRIVATE_NEW_FILE.jsonl` only when a private review copy is needed.
The destination must not already exist. The scanner reports source and code
hashes, but those hashes and session identifiers are also sensitive metadata.
