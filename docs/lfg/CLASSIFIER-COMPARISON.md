# Local classical classifier comparison

`ml/lfg-encoder/classifier_comparison.py` runs logistic-regression, KNN, and
decision-tree baselines against a screened local development corpus. It fits
each vectorizer and classifier on the train split and applies the frozen
vectorizer to the validation split. It does not access a test split, write
per-example predictions, retain model files, or call the LFG service.

The runner requires a screening manifest and the exact development plan used
to create it. It joins screened text to the original explicit target arrays by
source-record digest. Missing target arrays are errors; an explicit empty array
remains empty. Intent arrays stay multilabel. Select the target field and label
set in a local configuration file. Selected labels outside the taxonomy fail.
Screened text is used for fitting, while the original target arrays remain
unchanged. Any redaction can affect label meaning; screening does not certify
that text is fully redacted or that labels remain correct.

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
- `seed`, `max_rows`, `max_text_bytes`, `max_source_bytes`,
  `max_total_source_bytes`,
  `max_manifest_bytes`, `max_screened_file_bytes`, `max_features`
- `tfidf_ngram_range`, `logistic_c`, `logistic_max_iter`, `knn_neighbors`,
  `knn_weights`, `knn_p`, `tree_criterion`, `tree_max_depth`
- `near_duplicate_similarity`, a configured similarity boundary used to reject
  suspicious cross-split near duplicates

Set all bounds and model parameters explicitly for the corpus and available
machine. Unknown extension fields are accepted but inert. The report records
the config digest, input/output digests, taxonomy digest, train/validation
counts and group claims, source label uncertainty, parameters, environment,
per-target support and confusion counts, wall time, and classifier status.
KNN's retained sparse feature-array and target-array bytes are measured
separately. Model artifacts are not persisted, so persisted model artifact
bytes are zero; this does not represent total runtime memory.

Run it with a new owner-private output directory:

```powershell
& $env:LOCALAPPDATA\LFG\venvs\classifier-comparison\Scripts\python.exe `
  ml/lfg-encoder/classifier_comparison.py `
  --config C:\path\to\private-comparison-config.json `
  --output C:\Users\USER\AppData\Local\LFG\private-exports\NEW-COMPARISON
```

The output is descriptive exploratory evidence. Unreviewed or teacher-proposed
labels remain unverified. Previous use of a validation group prevents a claim
of independent final evaluation. No quality threshold, promotion, model
qualification, or verified outcome follows from this comparison. LFG and
owned-model results are reported unavailable unless captured separately by
their actual local prediction paths using matching task and input semantics.
