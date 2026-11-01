# Experimental assessment training

`ml/lfg-encoder/assessment_train.py` fits experimental assessment heads for
complexity, model selection, effort, and agent count. It uses reviewed request
and project-profile data with a frozen, locally supplied encoder. A project
profile supplies its own choices; there is no built-in provider or model roster.
The trainer does not discover a user's subscriptions or populate the profiles.

The result is a private weight pack, not a qualified model or release. There
is currently no prediction generator, native binary inference, rounding or
abstention policy, training checkpoint/resume, or weight update distribution
for this pack. Its choice scores are uncalibrated numbers, not probabilities.
The count head produces a continuous number. These gaps must be addressed and
evaluated before claiming that the binary uses the learned assessments.

## Data and review gates

The input must be a development plan with separate train and validation sources,
current verified and approved review attestations for every record, and no
within-split normalized duplicates. Plans containing a test source are rejected
before its data is opened. The trainer checks configured minimum support for
each target in both splits. Only training examples are passed to the encoder
and fitted; validation is used for support checks, not training. Project and
family separation are declarations and require independent review.

An unknown answer generates no positive or negative examples; zero agents is a
known count. For a reviewed non-null model list, available models omitted from
that list are treated as negative selections. Effort examples are generated
only for selected models whose effort was reviewed. If available candidates have
indistinguishable descriptions, the trainer fails instead of merging them.
Subscription IDs are excluded from descriptors by default; an operator can
explicitly enable them in the configuration when needed to distinguish seats.
Doing so can make derived weights reveal more about private subscriptions.

The example configuration's support minima of one are there to exercise the
mechanism. They do not establish a representative corpus, generalization, or
release quality. Set task-appropriate minima and maintain an independently
reviewed held-out test set and explicit promotion criteria.

## Method and reproducibility

The pinned local encoder uses attention-mask mean pooling and L2 normalization.
The `lfg.assessment-candidate-features.v1` row is `[q, p, c, q * c, 1]`, where
`q` embeds the request, `p` embeds the project profile's sorted complexity
labels, available model descriptions and effort levels, and agent range, and
`c` embeds the candidate. Candidate descriptions are canonical JSON containing
the candidate kind and applicable label or provider, model, label and effort;
subscription ID is included only when configured. For agent count `c` and
`q * c` are zero. Each head solves squared error plus configured ridge penalty
on the full weight vector. Model sets use one binary target per available model;
complexity and known effort use one binary target per configured choice.

The configuration explicitly selects CPU or CUDA. A requested unavailable CUDA
device fails; there is no automatic fallback. Torch deterministic algorithms
are requested, and CUDA TF32 is disabled. CUDA may require the operator to set
`CUBLAS_WORKSPACE_CONFIG` before launching; this value and device information
are recorded. Deterministic settings do not guarantee identical results across
hardware or software versions. The configured tensor estimate does not bound
the encoder, allocator, Python objects or solver workspace. Token limits are
enforced without truncation.

A separately bounded, fixture-only check of the encoder and head-fitting path
on a Windows RTX 5090 is described in [the local CUDA lab](LOCAL-CUDA-LAB.md).
That result does not establish training quality or Spark capacity.

Before and after fitting, the trainer checks the audited inputs, source code,
configuration, locked installed wheel payload and pinned model snapshot. The
output must be new and outside the snapshot and plan directory. It contains
weights, support counts and provenance, not raw request text, but derived
weights and metadata may reveal information from training data. Protect the
pack accordingly; excluding text is not anonymization.

After preparing reviewed data and the locked local environment described in
[the Spark environment notes](SPARK-LOCKED-ENVIRONMENT.md), a typical invocation
is:

```powershell
python ml/lfg-encoder/assessment_train.py --plan PRIVATE_DEVELOPMENT_PLAN.json `
  --config OPERATOR_CONFIG.json --model LOCAL_MODEL_DIRECTORY `
  --model-manifest LOCAL_MODEL_MANIFEST.json --revision PINNED_MODEL_REVISION `
  --environment-lock LOCKED_REQUIREMENTS.txt --wheelhouse LOCAL_WHEELHOUSE `
  --output NEW_PRIVATE_WEIGHT_PACK.json
```

Additional installed packages must be explicitly allowed with repeatable
`--allowed-extra` arguments. See the existing environment verifier for the
accepted model manifest and lock formats. Never put private packs in Git.

As of the latest local audit, the combined training workspace has 598 pending
records and no current approved assessments. It has no validation split and is
rejected by the plan-scope gate. No assessment heads have been trained on those
records. The automated trainer tests use fixtures to exercise code paths; they
are not a real training run or evidence of assessment quality.
