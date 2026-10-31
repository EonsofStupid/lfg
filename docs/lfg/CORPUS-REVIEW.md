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
   training partition; test rows are not supplied to the trainer. This does not
   establish adequate positive or negative support for every label.
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

`smoke_reviewed_gate.py` tests the gate through an actual short model-training
call using invented fixture records and simulated approvals. It creates a new
output directory, requires an explicitly chosen cached model and immutable
revision, and disables Hugging Face network access. It deliberately lowers a
copy of the policy's sample threshold for its two training fixtures; its
output must never be used as a release candidate or as evidence of human
review. For example, with a compatible test model already cached:

```powershell
.\.venv\Scripts\python.exe ml/lfg-encoder/smoke_reviewed_gate.py --base-model MODEL --model-revision COMMIT --output target/review-gate-smoke --epochs 1 --max-length 32 --batch-size 1 --checkpoint-steps 1
```

The requested output directory must not exist. A model that already has a
different-sized classification head may require the explicit
`--allow-mismatched-sizes` option; inspect all model-loading warnings. The
simulated approvals and fixture policy are stored with the test artifact so
they cannot be confused with the actual project review ledger.

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
