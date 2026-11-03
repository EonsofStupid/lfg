# Source-bound user model and effort requests

`ml/lfg-encoder/user_request_targets.py` joins an existing text-free
`lfg.user-requested-routing-observation.v1` to an exact source-bound provider
snapshot. It returns an in-memory `lfg.user-requested-model-effort-targets.v1`
record; it does not write prompt text or source paths. The caller supplies an
unreviewed relation list that connects one model span with at most one optional
effort span. The importer checks both spans against the verified UTF-8 source
bytes and uses only a unique exact match to a provider model ID/display name and
that model's listed effort levels. Provider listing remains distinct from
subscription entitlement.

Only caller-interpreted `affirmed` or `negated` model mentions produce binary
request targets (`1` or `0`). An effort target is emitted only when it is
uniquely matched, explicitly affirmed/negated, and related to an affirmed model
request. Quoted, uncertain, redaction-requiring, unmatched and ambiguous claims
stay `null`; no unmentioned candidate is made negative. The record marks the
interpretation unreviewed and keeps the source split, heldout separation,
subscription entitlement, execution, outcome and total-agent count unknown.

The importer requires the source snapshot to have been explicitly selected for
the exploratory train path before reopening that source. The output still has
`source_split.assignment: unknown`: a dataset plan must separately assign any
record to train, validation or test. A test-assigned snapshot is rejected
before source revalidation. This is not an automated split-reservation or
semantic-independence proof.

This target schema is not consumed by `assessment_train.py` and is not a
`review-assessment-targets.v1` record. Those targets mean recommendation
suitability and treat omitted available models as negatives. The new records
mean only that a caller interpreted an explicit user request in this prompt.
They are also not `assessment_distill.py` SFT examples: that builder requires a
complete assistant response, while this observation has no generated or
verified assistant target. A separate request-target training adapter is
required before these targets can train either the assessment heads or the
owned generative model. No quality claim follows from target import.
