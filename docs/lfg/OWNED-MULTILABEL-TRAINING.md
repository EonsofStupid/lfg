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

Training support is cell-level and may contain no negative labels. Unknown and
unreviewed status remain explicit claims; use does not imply quality, review,
verification, independent validation or promotion. This atomic-query framing is
a supervision format, not evidence that one model response can infer all labels
or that any latency target has been met. The existing free-form intent behavior
is unchanged.
