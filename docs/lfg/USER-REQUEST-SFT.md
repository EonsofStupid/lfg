# Teacherless user-request SFT

`ml/lfg-encoder/user_request_sft.py` builds one private SFT record from one
source-bound provider snapshot and a caller's explicit span interpretation. The
CLI takes a private caller-input JSON with this shape:

```json
{
  "schema_version": "lfg.user-requested-sft-caller-input.v1",
  "spans": [],
  "interpretation": {},
  "relations": {}
}
```

The caller supplies exact UTF-8 byte spans and polarity through the existing
`user_requested_observation` contract; the relations connect those model and
effort spans. The adapter creates the observation in memory, revalidates the
selected source, snapshot and provider cache, checks the reservation evidence,
and revalidates the caller/config inputs before writing. It stores no source
path in the dataset. A separate observation file is not needed.

The assistant target has separate `requested` and `unknown` arrays. The
`requested` array contains caller-linked model claims that uniquely match the
pinned provider listing and have explicit `affirmed` or `negated` polarity.
Explicitly related effort is included only when it uniquely matches the same
model's observed supported efforts. Ambiguous, unmatched, quoted, uncertain,
and unresolved annotated spans appear in `unknown` with their mention and a
categorical abstention status. These entries are not positive or negative model
targets. Unmentioned candidates stay absent by task definition; they are not
materialized as negative candidate labels. The caller's annotations are not
proof that every model or effort mention in the source was annotated. If screening would alter source
text, this span-bound builder refuses the record rather than risk offset drift.
The provider candidate projection is screened before storage.

An effort span linked to a negated model is retained as
`model_not_affirmed` unknown: this task does not claim the effort is a positive
setting for a model the caller explicitly negated. An effort linked to quoted
or uncertain model polarity remains `model_polarity_unknown`.

The dataset records the selected verified source identity (or a separately
verified POSIX projection identity), line, prompt, provider-cache, relation,
target, screening-policy, adapter and framing digests. It explicitly retains
`review_status: unreviewed`, unknown outcome, entitlement, execution and agent
count, exploratory train assignment, unknown heldout separation, and disabled
promotion. A caller interpretation remains an unreviewed claim. The target is
mechanically serialized from those claims; it is not a teacher response or a
verified preference.

The trainer hashes each actual UTF-8 request slice referenced by the sparse
target and unknown entries. Projection records also require the full request
bytes to match the verified POSIX projection digest and the unverified Windows
text-digest claim. Caller and SFT config extension fields are bounded, inert,
screened, and preserved; they do not override known configuration behavior.

The user-request dataset uses a distinct schema and is read only with the
`lfg.owned-user-request-sft-config.v1` task config. The trainer requires a
positive configurable weight for unreviewed examples and does not use the
assessment outcome weights for this task. Existing assessment SFT data continues
through its existing directory schema and validation path. User-request output
uses its own checkpoint and result schemas and is marked incompatible with the
current owned-assessment service.

Build one private dataset file on POSIX:

```text
python ml/lfg-encoder/user_request_sft.py \
  --caller-input PRIVATE_CALLER_INPUT.json \
  --provider-snapshot PRIVATE_SOURCE_BOUND_SNAPSHOT.json \
  --observation-config USER_REQUEST_OBSERVATION_CONFIG.json \
  --sft-config ml/lfg-encoder/user-request-sft.example.json \
  --screen-policy ml/lfg-encoder/private-corpus-screen.example.json \
  --output PRIVATE_DIRECTORY/user-request.json
```

The output parent must already be an owner-only POSIX directory and ignored and
untracked by Git. Windows output and user-request training fail closed until
private path safety is verified there. Inputs and output are byte-limited; the
CLI emits only an aggregate count and dataset digest. Keep input, dataset,
checkpoint and trained weights private and local.

Train with the separate task configuration and an owned checkpoint:

```text
python ml/lfg-encoder/train_owned_assessment_sft.py \
  --dataset PRIVATE_DIRECTORY/user-request.json \
  --checkpoint PRIVATE_BASE_CHECKPOINT \
  --config ml/lfg-encoder/owned-user-request-sft.example.json \
  --output PRIVATE_DIRECTORY/user-request-checkpoint
```

The existing inference service still expects the assessment output schema and
does not serve this task. A task-specific service route, live provider-candidate
input, and separate task evaluation are future work. The example config and
fixtures demonstrate the data path only; a passing fixture test is not evidence
of useful generalization. The adapter currently accepts one source request per
dataset file. It does not automatically assemble a representative corpus.

For a Windows-origin request that cannot be safely verified on its source
host, the caller may add `windows_origin_projection_claims` to the caller input.
It contains `schema_version: lfg.windows-origin-projection-claims.v1`,
`origin_platform: windows`, `origin_verification: unverified_claim`, and
SHA-256 claims named `line_sha256_claim`, `prefix_sha256_claim`, and
`user_text_sha256_claim`. The text digest must match the verified POSIX
projection bytes. An origin locator digest, source line number, and source
prefix byte count may be included together as further unverified claims.
The adapter accepts it only with a separately verified POSIX file containing
one selected user message, no later bytes, and text unchanged by the configured
screening policy. The POSIX projection digest is verified source evidence; the
Windows line, prefix, and text digests remain separate `unverified_claim`
fields. The dataset calls the source a projection, never the Windows original.
Split assignment and heldout independence remain unknown. Do not transfer a
full session to create this projection.
