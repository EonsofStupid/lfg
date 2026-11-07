# Partial-label masked supervision

`ml/lfg-encoder/masked_supervision.py` is an explicit experimental path for
exporting partial judgments and, in a separately configured future run,
training the existing Transformers encoder with masked binary cross-entropy.
It does not change `train.py`'s complete-record behavior and it does not make
that third-party encoder the LFG-owned model.

## Export policy

Use a versioned `lfg.masked-supervision-config.v1` JSON file. It names the
workspace, screening policy, optional label guide, selected taxonomy labels,
target-source choices, cell weights, near-duplicate policy, and operator-set
resource limits. Extension fields are accepted but inert. Use a new output
directory outside every Git worktree. The exporter requires owner-private
workspace/output access, a train-only plan (optional `validation` only), and
verifies plan, source, taxonomy, ledger, guide, policy, and sidecar identities
before and after writing. The output manifest is written last and binds the
export file digests to the config and source inventory.

```powershell
python ml/lfg-encoder/masked_supervision.py export --config PRIVATE_CONFIG.json
```

The exporter checks project, project-group, family, and exact screened text
separation across train and development validation. A bounded near-duplicate
audit also compares screened text across those splits. The split labels and
source ownership remain operator/source claims; the exporter does not access
test or holdout data and does not establish independent evaluation.

Each exported row has a nullable target, boolean `known_mask`, per-cell weight,
origin, and reviewer claim for every selected label column. A current partial
event's explicit positive or negative judgment is a known target. Its explicit
`unknown` remains unknown. Accepting a proposal is not a positive target;
rejecting a proposal is not a negative target and suppresses the unreviewed
draft guess. A current full review uses the complete reviewed label arrays.

`allow_unreviewed_proposals` explicitly enables owner-authorized development
draft labels. An explicitly present label is then a positive target. Omitted
labels remain unknown by default. The operator may name fields in
`unreviewed_complete_array_fields` to declare legacy arrays complete and allow
their omitted labels to become negative targets; the manifest records that
assumption. A per-cell target is trainable only if `known_mask` is true and its
weight is positive. Unknown targets stay JSON `null` and have zero weight.
Screening is recomputed locally before export; the report records changed-row
counts and policy digest, never prompt text in the command summary.

The export contains all eligible rows, including rows whose targets are all
unknown. It reports known/unknown and target-origin counts. Empty or unknown
targets are never converted to negatives, and a teacher proposal is never
described as a verified outcome. The exporter does not invent human review or
claim source rights, screening validity, or label accuracy are independently
verified. Row IDs, project/group names, family names, and reviewer claims are
stored as domain-separated digests; source artifact paths and content hashes,
the plan digest, and the source-copy comparison retain reconstruction
provenance without echoing those metadata strings into the exported rows.

## Optional masked training

Training requires a separate `lfg.masked-training-config.v1` file, an immutable
40-character model revision, an explicit model identity, and the selected
export manifest's exact SHA-256 in `export_manifest_sha256`. The trainer checks
that pin and revalidates the export's split digests again immediately before
fitting. Both unreviewed
proposal targets and partial-review targets have separate opt-ins. It refuses
to fit if no known training target remains after those choices. Example command
for a future authorized run:

```powershell
python ml/lfg-encoder/masked_supervision.py train --config PRIVATE_TRAIN_CONFIG.json
```

The trainer computes a stable weighted BCE-with-logits mean over known cells in
each training batch. Validation loss is normalized over the total known cell
weight across the entire validation split; per-batch values are scaled so the
Transformers example-weighted reducer returns that global weighted-cell ratio.
All-unknown validation rows stay in the coverage denominator when another row
has known cells. Unknown cells have exactly zero gradient. It currently requires
one CUDA process/device and gradient accumulation of one; distributed training
and larger accumulation are rejected because their exact cross-batch known-cell
normalization is not implemented. Validation rows are retained only when they
contain known target cells. Without usable known validation, there is no
validation metric or selected checkpoint claim. Validation statistics count
only known cells and undefined per-label values remain null.

The run manifest records the export/config/source digests, model revision,
runtime package/device evidence, target-source policy, mask semantics and
normalization. This task adds the path and fixture tests only; it does not run
model fitting. Experimental results are not promotion eligible and do not
qualify the LFG-owned binary, model, latency, or quality.
