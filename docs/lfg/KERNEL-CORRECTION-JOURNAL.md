# Local kernel observation and correction journal

The `kernel-predict` stream can optionally write an operator-selected local
JSONL journal. Nothing is journaled unless `--journal` and
`--journal-max-bytes` are supplied. Each observation records the input digest,
request ID, snapshot identity, address, state and choice-node observations.
It does not record prompt text or Extract values by default. An explicit
`--capture-text` records the full prompt in plaintext for later review. Use an
access-controlled, Git-ignored location and review privacy and data-use rights
before enabling capture. Request IDs, hashes, paths and addresses can themselves
be sensitive; a digest is not anonymization.

For example, with an authorized private JSONL file containing `id` and `text`:

```powershell
cmd /c 'target\release\lfg-server.exe kernel-predict --max-examples 1000 --journal "PRIVATE_JOURNAL.jsonl" --journal-max-bytes 10485760 < "PRIVATE_REQUESTS.jsonl"'

target/release/lfg-server.exe kernel-journal-status `
  --journal PRIVATE_JOURNAL.jsonl --journal-max-bytes 10485760

target/release/lfg-server.exe kernel-journal-show `
  --journal PRIVATE_JOURNAL.jsonl --journal-max-bytes 10485760 --observation 1
```

Predictions include `journal_sequence` when successfully journaled. `show`
returns the observation and valid choices from the matching snapshot. It hides
captured text unless `--include-text` is requested. For a custom snapshot,
pass the same `--snapshot` to prediction, show, correction and export; the
snapshot digest must match. The source and surrounding context still need to be
reviewed separately. After that review, a person can explicitly record a choice:

```powershell
target/release/lfg-server.exe kernel-correct `
  --journal PRIVATE_JOURNAL.jsonl --journal-max-bytes 10485760 `
  --observation 1 --node NODE_ID --choice OPTION --reviewer REVIEWER_ID --attest
```

The correction must refer to a choice node reached or escalated to by the
observation and an option in its exact snapshot. `--attest` and `REVIEWER_ID`
are local assertions; the CLI does not authenticate a person or establish that
a review occurred. It does not grant permission or change the kernel.

If text was not captured in the journal, an operator can supply the original
private JSONL source instead. Its rows must have unique, nonempty `id` and
`text` fields. The tool checks exact UTF-8 text hashes against each matching
observation; it never changes the journal or treats a match as approval:

```powershell
target/release/lfg-server.exe kernel-journal-reconcile `
  --journal PRIVATE_JOURNAL.jsonl --journal-max-bytes 10485760 `
  --source-jsonl PRIVATE_REQUESTS.jsonl --source-max-bytes 10485760

target/release/lfg-server.exe kernel-journal-show `
  --journal PRIVATE_JOURNAL.jsonl --journal-max-bytes 10485760 --observation 1 `
  --source-jsonl PRIVATE_REQUESTS.jsonl --source-max-bytes 10485760 --include-text
```

`reconcile` reports matched and missing counts plus source and journal digests,
without returning the text. It rejects any matching ID whose text changed.
`show` also checks the source and only displays its text when explicitly asked.
The check identifies the selected file bytes, not authorship, rights, complete
context or reviewer identity. Source size is bounded and the source is reread
to detect ordinary changes during the operation; this is not a lock against
hostile concurrent modification. Inspect the original source and context before
attesting to a correction.

To prepare uncorrected choice observations for a person to inspect, create a
new private queue from the exact matching source and snapshot:

```powershell
target/release/lfg-server.exe kernel-journal-review-queue `
  --journal PRIVATE_JOURNAL.jsonl --journal-max-bytes 10485760 `
  --source-jsonl PRIVATE_REQUESTS.jsonl --source-max-bytes 10485760 `
  --output NEW_PRIVATE_REVIEW_QUEUE.jsonl --output-max-bytes 10485760 --include-text
```

This command writes the requests in plaintext, so the explicit `--include-text`
flag is required. It reports aggregate counts without printing requests. It
requires the source to cover the entire journal and each observation to match
the selected snapshot, rejects ambiguous existing corrections, and does not
overwrite an existing output. The output has one pending item per reached or
escalated choice node that has no correction. It lists the node's snapshot
options and the baseline choice if one was made; a null baseline choice means
the node did not select an option. The reviewed choice, reviewer, project,
family and split are all unset. A snapshot option is not evidence that it is
an appropriate label for a request: review its meaning and the original context,
and skip an item if the option set is inadequate or the decision is uncertain.

The queue is a point-in-time display aid, not a ledger. Editing it does not
record a correction. After examining an item and its source, an actual reviewer
can use its observation sequence and node with `kernel-correct` above; the
journal then enforces the snapshot and choice checks. Recreate the queue at a
new path to omit already corrected nodes. A queued item or correction is not
automatically approved for training. The output is bounded, but a failed write
may leave a partial file; inspect it and use a new path. Source and journal
digests are consistency checks, not proof of authorship, review or rights.

When a correction has been recorded and matching text is available from either
the journal or the explicitly selected source, export pending per-node
candidates to a **new** private file:

```powershell
target/release/lfg-server.exe kernel-journal-export `
  --journal PRIVATE_JOURNAL.jsonl --journal-max-bytes 10485760 `
  --source-jsonl PRIVATE_REQUESTS.jsonl --source-max-bytes 10485760 `
  --output NEW_PRIVATE_CANDIDATES.jsonl --output-max-bytes 10485760
```

The exporter refuses multiple corrections for the same observation and node,
unknown snapshots or choices, an existing output, over-limit output, and an
empty result. The source flags are optional when the journal already contains
captured text. It excludes corrected observations for which neither source has
text and reports their count. Candidate records identify the text origin and,
when used, the source file digest. This export uses the versioned
`lfg.node-correction-candidate.v2` schema; it supersedes the earlier captured-text
only candidate format. Candidates remain `pending_review`, with project, family
and split unset; they are not accepted by the existing training loader and are
not approved labels. Per-node review, source and rights checks, split assignment,
label support, and a separate evaluation procedure must be implemented before
these can train or qualify a model.

Entries form a SHA-256-linked sequence. Opening the file verifies the chain,
size, complete lines and correction references before writing. An OS file lock
excludes cooperating writers for the lifetime of the process; other software
that ignores the lock can still interfere. Each append calls `sync_data`.
Opening currently reads the whole bounded journal, so startup cost grows with
the file; per-event synchronization also has a cost. Neither has been benchmarked
or qualified for a high-throughput persistent service.
This is not an authenticated or encrypted log: an actor who can rewrite the
whole file can recompute the chain or truncate a valid suffix. Preserve a trusted
tail digest externally if that threat matters. A crash can leave a partial line;
the next open rejects it rather than silently truncating it. Inspect and recover
from a backup before continuing. A failed batch may already have appended prior
rows, so check its exit status and journal before retrying. Output export may
leave a partial new file if the filesystem fails; inspect it before retrying.

On 2026-10-31 the journal recorded 11 previously inspected, unreviewed LFG
conversation requests locally: 11 observations, zero corrections, zero
captured texts. The reconciliation command matched all 11 request IDs and input
digests against the local source, and an explicit local show check confirmed
that source text is hidden by default and available on request. The local status
check verified the chain, and export with no corrections was rejected without
creating a file. This is real
observation provenance, not a reviewed training dataset. The journal remains
under `dataset/private` and was not sent to the Spark or committed.
The new queue command was run locally on those same 11 records. It created
`dataset/private/kernel-node-review-queue-real.v1.jsonl` with 11 pending
`operation` items: two baseline selections and nine without a baseline choice.
All reviewer, reviewed-choice and split fields are unset; the journal is
unchanged. This private file contains the original text and has not been
sanitized. The bundled two-option example snapshot is not a validated labeling
guide. These records still need source/context review, useful per-node label
definitions, decisions, project/family and split assignment, and a per-node
training and evaluation gate before they can support model development.
The separate [per-node review and admission workflow](NODE-REVIEW-AND-ADMISSION.md)
can bind explicit human decisions to this queue, audit declared split
separation and support, and export admitted train/validation inputs. It does not
train or qualify a model.
The journal unit tests and CLI/locking tests also passed in a separate offline
ARM64 Spark build using test fixtures only. This does not establish production
durability or authenticate reviewers.
A separate test-only journal created on Windows was verified and given a
simulated correction on the Spark, then verified again on Windows. No real
project text was used in that cross-platform fixture, and its simulated
attestation is not review or training evidence.
An optimized Windows build with the UI embedded verified the real journal's
aggregate status and the existing kernel golden. Its SHA-256 was
`cf16e4f3971954e9a413ecfc479fa6e5722ebe7e43a54e3e6bf00f1453865c4f`
and its size was 7,900,672 bytes. This was a local smoke check, not a qualified
or reproducible release.
