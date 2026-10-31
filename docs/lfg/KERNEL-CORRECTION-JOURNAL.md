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

If full text was explicitly captured and a correction was recorded, export
pending per-node candidates to a **new** private file:

```powershell
target/release/lfg-server.exe kernel-journal-export `
  --journal PRIVATE_JOURNAL.jsonl --journal-max-bytes 10485760 `
  --output NEW_PRIVATE_CANDIDATES.jsonl --output-max-bytes 10485760
```

The exporter refuses multiple corrections for the same observation and node,
unknown snapshots or choices, an existing output, over-limit output, and an
empty result. It excludes corrected observations without captured text and
reports their count. Candidates remain `pending_review`, with project, family
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
captured texts. Each recorded request ID and input digest was checked against
the local source record. The local status check verified the chain, and export with no
corrected captured text was rejected without creating a file. This is real
observation provenance, not a reviewed training dataset. The journal remains
under `dataset/private` and was not sent to the Spark or committed.
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
