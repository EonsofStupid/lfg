# Local classical classifier comparison

`ml/lfg-encoder/classifier_comparison.py` runs logistic-regression, KNN, and
decision-tree baselines against a screened local development corpus. It fits
each vectorizer and classifier on the train split and applies the frozen
vectorizer to the validation split. It does not access a test split, write
per-example predictions, retain model files, or call the LFG service.

The runner requires a screening manifest and the exact development plan used
to create it. It joins screened text to the original explicit target arrays by
source-record digest, requires a one-to-one match, checks project, group, family,
and normalized-text separation, and recomputes each screened string with the
configured policy and current screening code. A missing target array is always
unknown. An empty array is unknown unless the row carries an explicit
`target_completeness[field]: true` attestation. When completeness is explicitly
false, present positive labels remain known and omitted labels stay unknown.
Legacy nonempty arrays without a field-level completeness flag are treated as
complete under the source schema; this remains a claim, not independent
verification. Intent arrays stay multilabel. Select the target field and label
set in a local configuration file. Selected labels outside the taxonomy fail.
Screened text is used for fitting, while source target arrays remain unchanged.
Any redaction can affect label meaning; screening does not certify that text is
fully redacted or that labels remain correct.

Create an isolated dependency environment outside the repository and install
the pinned comparison dependencies. The pins satisfy the encoder's declared
scikit-learn `>=1.7,<2` and NumPy `>=2,<3` ranges:

```powershell
py -3.11 -m venv $env:LOCALAPPDATA\LFG\venvs\classifier-comparison
& $env:LOCALAPPDATA\LFG\venvs\classifier-comparison\Scripts\python.exe -m pip install -r ml/lfg-encoder/requirements-classifier-comparison.txt
```

The configuration is a private JSON file with schema version
`lfg.classifier-comparison-config.v1`. Required keys are:

- `revision`, `screened_corpus`, `label_plan`, `target_field`, `selected_labels`
- `screen_policy`, `constant_target_behavior` (`abstain` or `majority_class`)
- `review_ledger`, `max_ledger_bytes`
- `rejected_label_handling` (`exclude` or explicit `include_exploratory`)
- `validation_group_use_claim` (`previously_used`, `reported_unused`, or `unknown`)
- `seed`, `max_rows`, `max_text_bytes`, `max_source_bytes`,
  `max_total_source_bytes`,
  `max_manifest_bytes`, `max_screened_file_bytes`, `max_policy_bytes`, `max_features`
- `tfidf_ngram_range`, `logistic_c`, `logistic_max_iter`, `knn_neighbors`,
  `knn_weights`, `knn_p`, `tree_criterion`, `tree_max_depth`
- `near_duplicate_similarity`, `max_near_duplicate_pairs`, and
  `max_near_duplicate_work`, which bound the configured similarity check

Set all bounds and model parameters explicitly for the corpus and available
machine. Unknown extension fields are accepted but inert. Constant targets
either abstain or use one configured majority-class fallback across all three
algorithms; ties select negative. Empty or missing labels never enter training
or validation as negatives. Rejected ledger rows are excluded by default;
`include_exploratory` is an explicit opt-in and is recorded in the report.
Pending rows remain fit eligible, with review state reported as a claim. The report records
the config digest, input/output digests, taxonomy digest, train/validation
counts and group claims, source label uncertainty, parameters, environment,
per-target positive/negative/unknown support and confusion counts, covered-only
macro/micro precision, recall and F1. Metrics with no scored denominator are
null, and macro values average defined labels while reporting their coverage.
Standard macro precision includes labels
with false-positive predictions even when they have no positive truth support;
a separate supported-label macro is also reported. The report records
fit/vectorize/predict elapsed time and
classifier status. It binds the review ledger digest, checks that the ledger
covers exactly selected rows, and reports per-split review and target-array
states. The timer excludes loading, imports, metric calculation, and
report writing; it is not per-request latency.
KNN's retained sparse feature-array and target-array bytes are measured
separately. Model artifacts are not persisted, so persisted model artifact
bytes are zero; this does not represent total runtime memory.

The report destination must be outside every Git worktree. The runner creates
an owner-only report directory and verifies its ACL on Windows before writing;
the exact report file is checked again after creation. If these checks fail,
the report is not written.

Run it with a new owner-private output directory:

```powershell
& $env:LOCALAPPDATA\LFG\venvs\classifier-comparison\Scripts\python.exe `
  ml/lfg-encoder/classifier_comparison.py `
  --config C:\path\to\private-comparison-config.json `
  --output C:\Users\USER\AppData\Local\LFG\private-exports\NEW-COMPARISON
```

The output is descriptive exploratory evidence. It records per-partition
`verified`, `labels_pending`, and target-completeness flags, without treating
those claims as independent verification. The configured validation-use claim
is also explicitly unverified. No quality threshold, promotion, model
qualification, or verified outcome follows from this comparison. LFG and
owned-model results are reported unavailable unless captured separately by
their actual local prediction paths using matching task and input semantics.
