# Owned multilabel contract and training adapter

`owned_multilabel_contract.py` defines `lfg.owned-multilabel-response.v1` and
converts the pinned, owner-private masked export's **train split only** into
one atomic example for each selected known target cell. A positive or negative
source target becomes one explicit decision. Unknown cells are counted but
produce no example, decision token, or abstention target. Missing decisions in
a model response mean unknown; they never mean negative. Full-response
abstention is optional and cannot coexist with positive or negative decisions.
Selected field/label identities must exist in the pinned taxonomy. Duplicate or
out-of-selection decisions are rejected. Extension objects are retained but
inert. Unknown response/config extension fields are retained where the
contract permits them, subject to a 64 KiB serialized bound; the adapter does
not assign them behavior. Responses carry a required taxonomy digest and an
optional guide digest, which the private dataset reader checks against its
manifest before accepting targets.

The operator configuration pins the source export manifest, taxonomy, selected
columns, prompt template, output path and resource bounds. The template must
bind both `{field}` and `{label}`; its digest is retained with the template in
the private manifest. Effective example weight is the product of the configured
per-label (or default) multiplier and the source cell weight. Provenance keeps
source, project and family identity claims, target origin, reviewer/status
claims, screening/redaction identity and source digests. These digests preserve
the claims supplied by the source; they do not independently verify them.

Use an absolute private config path:

```json
{
  "schema_version": "lfg.owned-multilabel-training-config.v1",
  "export_dir": "C:/private/masked-export",
  "export_manifest_sha256": "<64 lowercase hex characters>",
  "taxonomy_file": "C:/private/evaluation-taxonomy.v1.json",
  "taxonomy_sha256": "<64 lowercase hex characters>",
  "selected_columns": [{"field": "intents", "label": "build.software"}],
  "instruction_template": "For field {field}, decide whether label {label} applies to the project request. Return one explicit positive, negative, or unknown decision.",
  "output": "C:/private/new-multilabel-training-run",
  "max_rows": 100000,
  "max_output_bytes": 1073741824,
  "default_weight": 1.0,
  "per_label_weights": {}
}
```

Run `python owned_multilabel_contract.py build --config <private-config>` from
`ml/lfg-encoder`. The adapter refuses an existing output directory and paths
inside a Git worktree; it writes `train.jsonl` and `manifest.json` with owner-only
permissions where the host supports them. Standard output contains only the
manifest digest, aggregate positive/negative/unknown counts, row count and
weight totals. It does not print output paths, per-label support, prompts or
label text. Readers must pin the manifest digest when calling
`load_training_dataset(directory, expected_manifest_sha256)`. The owned SFT
trainer has a separate `lfg.owned-multilabel-sft-config.v1` mode. It requires
`expected_manifest_sha256` and `multilabel_system_prompt`, plus the shared SFT
training, framing, context, resource, checkpoint and optimizer settings. Pass
the adapter output directory as `--dataset`; do not reuse the assessment or
user-request config schema for this mode. Six framing delimiters wrap the
configured system prompt, the row's exact instruction and screened text, and
that row's serialized `target_response`. The target response alone is learned;
prompt and padding bytes are masked. The adapter-supplied effective weight is
used once for weighted record sampling. Unknown cells are omitted entirely and
remain counted as unknown in source support claims.

The trainer loads the adapter through `load_training_dataset`, checks the
caller-pinned manifest and row-file digests, and rechecks the bytes around
loading and after execution. It binds manifest, train bytes, configuration,
framing, taxonomy and guide digests, source identity claims, code digests and
source checkpoint digests into coverage and resumable checkpoint identity.
Coverage records aggregate support and states that source quality is
unverified. It does not emit prompts or target text. Checkpoints and trained
weights must be written under an ignored, untracked private output directory.
The validation and heldout splits are never opened by this trainer mode.
The live `masked_supervision.py` loader code digest is recorded separately from
the adapter manifest's loader digest claim and is checked again before a run
can complete. Completed multilabel results can seed another run only after
their weights, architecture, internally linked historical code digests,
framing and source provenance validate. Historical adapter and loader digests
remain provenance; new runs bind the current runtime code, and resume requires
that exact runtime identity. If the config is missing or invalid, filesystem
errors are redacted conservatively. Private filesystem errors from this mode
are reported without printing filesystem paths.

This integration makes the adapter consumable; it does not qualify the
classifier. Unreviewed labels and zero-negative datasets remain trainable, but
they cannot establish discrimination or independent accuracy. Promotion stays
disabled until a separate evaluation supports it. Existing assessment and
user-request SFT config schemas retain their prior behavior.

## Optional legacy non-exhaustive teacher overlay

The owned multilabel config may include a `legacy_teacher_overlay` object to
combine source annotations with a separately pinned, source-bound legacy local
teacher report. This path accepts only `lfg.train-teacher-proposals.v1` reports
in `complete` or `partial_aborted_ambiguous` operational state whose explicit
claims still say teacher suggestions are unreviewed and not promotion eligible.
It validates the report and proposal file digests, their internal publication
binding, source export/config/train/taxonomy identities, source lineage,
selected-column order, ordered selected row IDs, source row evidence and the
model/prompt/schema provenance. These checks establish package integrity and
lineage consistency; they do not validate the meaning of the teacher labels.
The producer places taxonomy and source lineage under
`report.provenance.taxonomy_sha256` and
`report.provenance.source_lineage`; selected columns are within that nested
lineage. The bridge follows that published schema. It also requires the
producer's uncertainty claim `omitted_labels_are_negative: false` and records
the producer's `heldout_accessed: false` claim without reading heldout data.
Partial reports may lack worker/payload digests when the terminal worker result
was not verified; such rows remain operationally uncertain teacher claims.
The report directory must pass the producer's existing owner/SYSTEM/
Administrators policy on Windows (or owner-only `0700` on POSIX); each report
file separately retains its stricter owner-private check. Source artifact
inventory hashing follows the producer's serializer: terminal-verified reports
use its completed-report JSON encoding, while terminal-unverified recovered
reports use its canonical recovery encoding. Non-ASCII artifact paths are
covered by the corresponding synthetic cases.

Example fields (use actual absolute private paths and digests in a local config):

```json
{
  "legacy_teacher_overlay": {
    "schema_version": "lfg.legacy-teacher-overlay-config.v1",
    "report_manifest": "C:/private/teacher-report/manifest.json",
    "report_manifest_sha256": "<64 lowercase hex characters>",
    "proposals_file": "C:/private/teacher-report/proposals.jsonl",
    "proposals_sha256": "<64 lowercase hex characters>",
    "default_weight": 1.0,
    "per_label_weights": {},
    "conflict_policy": "retain_source_and_skip_conflicting_teacher_positive",
    "max_manifest_bytes": 67108864,
    "max_proposals_bytes": 2147483648
  }
}
```

Only an explicit `proposed_positive` from an `answered` non-exhaustive row can
add a target, and only where the pinned source cell is unknown. Omitted,
unknown, non-answered, abstained, truncated, incomplete and not-attempted
states remain unknown; none produces a negative. Existing source positives
remain source examples and a teacher agreement is counted without adding a
duplicate sample. A teacher positive that conflicts with a known source
negative is recorded as a conflict and skipped; the source negative remains
unchanged. Other conflict policies are rejected until they are deliberately
implemented and reviewed.

Added targets use origin `unreviewed_local_teacher_positive`, keep the original
source origin and reviewer claim separately, and bind each row to the report,
proposal file and exact proposal line/index, proposal and comparison config,
model name/manifest, prompt contract, output schema and adapter/comparator code
digests. Teacher weight is `source cell base × owned-configured multiplier ×
teacher-overlay multiplier`; because a teacher target only fills an unknown
source cell, its source base is `1.0`. The existing trainer then applies the
resulting row weight once during weighted sampling. The manifest records the
teacher weight policy and aggregate state counts, including unknown cells.
Partial operational status and an unverified terminal claim are preserved as
uncertainty; they are not quality gates and do not promote the model. With no
overlay configured, the original source-only builder and loader contract is
unchanged. Private report/proposal contents and paths must stay outside Git.

Training support is cell-level and may contain no negative labels. Unknown and
unreviewed status remain explicit claims; use does not imply quality, review,
verification, independent validation or promotion. This atomic-query framing is
a supervision format, not evidence that one model response can infer all labels
or that any latency target has been met. The existing free-form intent behavior
is unchanged.
