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
including line endings. Start a **new** pending ledger:

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
