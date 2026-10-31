# Per-node review and data admission

The experimental kernel's choice nodes need their own reviewed outcomes. The
existing global multi-label encoder and its review ledger do not provide those
outcomes. `ml/lfg-encoder/node_review.py` provides a separate local review and
admission workflow for the pending queue produced by
[`kernel-journal-review-queue`](KERNEL-CORRECTION-JOURNAL.md). It does not train
a model, authenticate reviewers, verify source rights, sanitize requests, or
establish model quality.

## Review

The queue contains plaintext and the original source context must be inspected
separately. Select an access-controlled directory; Git ignore is not encryption
or access control. The snapshot must be the exact bytes used by the journal,
including line endings.

### Local review screen

For an existing pending ledger, `node_review_web.py` serves a browser screen on
`127.0.0.1` with a one-time bearer token in the printed URL. It shows one request,
its provenance and configured choice guidance; it hides the kernel's prior
choice and state. The server does not send data to a model or record decisions
unless the reviewer explicitly submits one. It is a local review tool, not a
hosted multi-user service. Other local processes or browser extensions with
access to the token or files may see the data; the files and ledger are not
encrypted or signed.
Source notes or outside context may still bias a reviewer; hiding the baseline
does not make this a blinded review.

First generate a **new draft** guide from the exact snapshot. `--node` can be
repeated; omit it to generate entries for every choice node. The template
contains the actual node and option names but leaves questions, meanings and
boundaries blank rather than inventing definitions:

```powershell
python ml/lfg-encoder/node_review_web.py template `
  --snapshot SNAPSHOT.json --node NODE_ID --output NEW_PRIVATE_GUIDE.json
```

A knowledgeable reviewer must define the question, meaning and boundary for
each option, examine their source and suitability, and set a new revision and
`status` to `ready`. A draft guide allows browsing and rejection but disables
approval. `ready` is a local declaration, not proof of expert review or an
approved taxonomy. The guide must match the snapshot and cover the queue's nodes.

Then start the screen with an existing queue and ledger. A supplied source JSONL
is optional. When its file digest and the matching request text agree with the
queue, the screen offers its project and family as editable, source-provided
values. It does not guess a choice, related-project group or split. Inspect the
original source, surrounding context and privacy before deciding; a digest match
is not proof of authorship or permission. This screen does not redact text: if
redaction is needed, reject or skip and prepare a new sanitized source and queue.

```powershell
python ml/lfg-encoder/node_review_web.py serve `
  --queue PRIVATE_QUEUE.jsonl --ledger PRIVATE_LEDGER.json --snapshot SNAPSHOT.json `
  --guide PRIVATE_GUIDE.json --source PRIVATE_SOURCE.jsonl --reviewer REVIEWER_ID
```

Open the printed URL locally and stop the server with Ctrl+C. Approval requires
a ready guide, a selected option, project, group, family, predetermined split,
attestation and typed confirmation. The ledger records the guide digest; its
first guide-bound approval upgrades the ledger to `lfg.node-review-ledger.v2`
while preserving prior decisions. Older tools that only accept v1 may reject the
upgraded ledger. Keep the exact guide bytes with the review evidence. Guide and
reviewer identities are assertions and can be forged by someone with file access;
the audit binds the ledger bytes but does not authenticate the guide or reviewer.
The API checks a loopback Host, a bearer token and same-origin POSTs; it is not
designed for a reverse proxy or remote deployment.

The current bundled snapshot is an example, not a validated labeling guide. In
the owner's [LFG v3 notes](https://docs.google.com/document/d/1NCIJK8wBlnzAa43OuWLktQup3RtksqDvM8n2ckPWj5U/edit),
`tool_call` is also used for a model's function-call signal; the example
snapshot's text patterns do not establish how to label every user request.
Resolve that distinction before declaring a guide ready or using these labels.
For the current real queue a private, blank `operation` draft guide was generated
at `dataset/private/kernel-node-review-guide.operation.draft.v1.json`; no real
decision was recorded. An actual source-file check matched one pending item and
its project and family metadata. A subsequent metadata-only check matched all
11 pending items to that file and found project and family fields for each; the
ledger remained unchanged. These are wiring checks, not source review.

### Terminal review

To start a **new** pending ledger:

```powershell
python ml/lfg-encoder/node_review.py init --queue PRIVATE_QUEUE.jsonl `
  --snapshot SNAPSHOT.json --ledger NEW_PRIVATE_LEDGER.json

python ml/lfg-encoder/node_review.py list --queue PRIVATE_QUEUE.jsonl `
  --snapshot SNAPSHOT.json --ledger NEW_PRIVATE_LEDGER.json

python ml/lfg-encoder/node_review.py show --queue PRIVATE_QUEUE.jsonl `
  --snapshot SNAPSHOT.json --ledger NEW_PRIVATE_LEDGER.json `
  --item-sha256 ITEM_SHA256 --include-text
```

`list` displays item digests and status, not request text. `show` hides the
baseline choice and state unless `--include-baseline` is set, and hides text
unless `--include-text` is set. This reduces one source of reviewer anchoring;
the queue itself contains baseline results, so this does not establish blind
review. Hashes and identifiers can also be sensitive.

A reviewer should examine the request, source, surrounding context, data-use
rights, and the meaning of the snapshot's choices. If no option is appropriate
or the meaning is unclear, do not force a label. Approval is an explicit local
attestation that this review was performed and the choice and assignments were
considered; it is not proof of identity, permission, truth, independence, or
correctness. A reviewer can explicitly approve or reject a pending item:

```powershell
python ml/lfg-encoder/node_review.py decide --queue PRIVATE_QUEUE.jsonl `
  --snapshot SNAPSHOT.json --ledger PRIVATE_LEDGER.json --item-sha256 ITEM_SHA256 `
  --decision approved --reviewer REVIEWER_ID --choice CHOICE --project PROJECT `
  --project-group RELATED_PROJECT_GROUP --family RELATED_REQUEST_FAMILY `
  --split train --attest

python ml/lfg-encoder/node_review.py decide --queue PRIVATE_QUEUE.jsonl `
  --snapshot SNAPSHOT.json --ledger PRIVATE_LEDGER.json --item-sha256 ITEM_SHA256 `
  --decision rejected --reviewer REVIEWER_ID --attest
```

Use `validation` or `test` when the predetermined source separation calls for
it; do not assign splits based on model performance. Approvals must select an
actual option from that snapshot. Rejection has no label or split. Decisions are
not editable through this command: use a new ledger and repeat the review if a
decision must change. The tool uses an exclusive sidecar lock and atomic ledger
replacement for cooperating processes. If a crash leaves `LEDGER.json.lock`,
verify that no review process is running before removing it. The files are not
signed; someone with write access can forge an attestation or replace the full
ledger. Avoid concurrent external edits.

## Audit and export

Use a separately selected, versioned admission policy and include **every**
relevant queue and ledger as a repeated `--input QUEUE LEDGER` pair:

```powershell
python ml/lfg-encoder/node_review.py audit `
  --input PRIVATE_QUEUE.jsonl PRIVATE_LEDGER.json `
  --snapshot SNAPSHOT.json --policy PRIVATE_POLICY.json --node NODE_ID
```

With the v1 policy, admission requires all supplied items for the target node to have a
decision, at least one approved item, and the configured minimum number of
approved examples for **every** option and project-group count in train,
validation and test. It rejects normalized exact-text duplicates within the
target split and flags project, declared project-group, family, normalized text
or request-ID overlap across splits. It also checks approved records for other
nodes in the supplied queues for cross-split conflicts. Unlisted queues, related
projects with different declarations, paraphrases, shared history, and previous
exposure cannot be detected reliably. Declared groups and families are not proof
of independence.

The checked-in [example policy](../../ml/lfg-encoder/node-admission-policy.example.json)
uses one per option and one project group per split solely to demonstrate the
schema. It is not statistically adequate, a recommended production threshold,
or a quality gate. Set thresholds appropriate to the actual evaluation plan.
Passing admission is never a claim that a model is trained or qualified.

For development without opening a held-out test queue, use a v2 policy such as
the [development policy example](../../ml/lfg-encoder/node-development-policy.example.json).
It requires support and separate project groups only for train and validation,
and produces `lfg.node-admission-report.v2` with `scope: development` and
`holdoutAssessment: not_performed`. Supply only queues and ledgers explicitly
reserved for development. Before opening any queue, the audit reads all supplied
ledgers and rejects a test-assigned record; it checks that those ledgers did not
change before their queues are read. Pending and rejected records have no split,
so the tool cannot determine whether they belong to a hidden holdout. Operators
must physically separate holdout files and avoid supplying mixed queues or
ledgers. This does not establish that a holdout exists, that related sources are
independent, or that no test information has previously been seen. A separate,
controlled final evaluation and leakage check remain necessary. The example
thresholds of one are for mechanics only and are not sufficient for quality.
Admission reports bind the audit script digest. Changing the audit tool or
policy changes that identity; rebuild any development pack from the newly
audited inputs rather than treating an older pack as a current audit result.

After admission, `export` takes the same inputs plus `--output NEW_PRIVATE.json`.
It creates a new file containing only approved train and validation texts,
choices and provenance; test texts are not exported. It refuses to overwrite an
existing file or export while gates fail. This format is a per-node handoff and
is **not** consumed by the existing global encoder trainer. An interruption
during writing can leave a partial file; inspect it and choose a new path. The
v1 audit and export process itself reads test records and reports test support.
The v2 development policy does not require test records but cannot prove they
were kept separate. Genuine blinding and an independent evaluation process
remain necessary. The report records input, snapshot, policy
and tool hashes plus Python and Unicode versions; these hashes identify the
bytes used, not their trustworthiness. Filesystem checks do not protect against
hostile concurrent changes.

The separate [node validation runner](NODE-VALIDATION.md) can evaluate admitted
validation decisions against the selected deterministic kernel binary, and can
compare supplied candidate predictions on escalations. It does not train or
execute a candidate model.

## Actual local state

On 2026-10-31, a private pending ledger was initialized from the existing 11
real LFG observation items. The audit reported 11 pending, zero approved, zero
support for either `operation` option in every split, and failed both support and
project-group gates. Export was refused and no training input was created.
These are real requests, but the two-option example snapshot is not a validated
labeling guide and the records are not reviewed training data. They were not
sent to the Spark. The review and admission mechanics were tested on Windows
and the Spark using explicitly test-only fixtures.

On 2026-10-31, a separate private queue was prepared from the two existing
training-only review workspaces. Of 520 pending source records, 83 previously
triage-flagged records and four over the example snapshot's input limit were
deferred; 11 exact-text matches were already present in the original queue.
These are preparation decisions, not final dataset exclusions. The remaining
422 requests cover 12 declared projects and 17 session families. Their recorded
source prefixes and message text matched 17 local session files during a
read-only check. A repeated grouped check reported 34 records unverified with
`ValueError` but did not retain the reason; a subsequent full check matched all
422. Both aggregate reports are retained in the private directory. These checks
do not lock the sessions or establish authorship, privacy, rights, context,
independence or suitable labels.

The example LFG kernel produced 435 pending node observations for those 422
requests: 422 at `operation`, 11 at `work-kind` and two at `domain`. Its
`operation` choice selected `tool_call` for 20 user-message requests; those are
unreviewed kernel outputs, not verified labels. The notes' signal-normalization
draft uses `tool_call` for a model's structured action signal, whereas the example
snapshot can select that option from prompt text. The intended meaning of this
choice must be resolved before a reviewer can consistently label it.

The private files are under
`dataset/private/kernel-real-train-pending-20261031-v1/`: `source.jsonl`,
`observations.jsonl`, `queue.jsonl`, `ledger.json`, and a contextual draft
guide. The ledger has 435 pending items and zero approvals. The guide records
the open questions and remains `draft`, so the browser disables approval.
Source and queue contain plaintext, paths and hashes and require access controls.
These 422 requests were not sent to the Spark, and this training-source queue is
not an independent validation or test set. Avoid combining overlapping queue
versions; run the admission audit on every selected queue once real reviews
and suitable choice definitions exist.
