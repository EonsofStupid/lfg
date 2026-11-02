# Preparing assessment teacher traces

`ml/lfg-encoder/distill.py build-sft` converts explicitly selected local
teacher traces into a private SFT dataset. Its task and system instruction come
from the operator's plan. To train LFG's assessment behavior, choose teacher
responses that actually assess the intended request and project profile; the
builder cannot infer that from arbitrary coding-assistant replies. It does not
run a teacher, fetch subscriptions, fill app templates, train a student, or
produce runtime weights.

The owner has authorized using their own projects and chats. The operator
chooses sources and train or validation splits; an unreviewed trace may be
prepared and keeps its `unreviewed` and outcome claims. A train-only plan works.
Validation, when supplied, is not training input. The builder rejects an
original-source digest declared in both splits, but declared project groups,
source digests and exact normalized-request checks cannot prove independent projects,
conversations or paraphrases. A source declared as `test` is rejected before
the tool opens any source. Reserve held-out data outside the development plan.
The manifest marks independence unproven, claims unverified and promotion
ineligible. No review or success claim is authenticated or established by this
tool.

## Input

Create a **private** UTF-8 JSONL file containing teacher assessments that have
already been selected and redacted. Each line has exactly these fields:

```json
{"id":"RECORD_ID","request":"REDACTED_REQUEST_AND_PROJECT_CONTEXT","response":"REDACTED_TEACHER_ASSESSMENT","teacher":"OPERATOR_SELECTED_TEACHER_ID","review_status":"unreviewed","outcome":"unknown","source_ref":"ORIGINAL_PROJECT_OR_SESSION_REFERENCE","source_sha256":"ORIGINAL_SOURCE_SHA256"}
```

The example is a format illustration, not training data. `review_status` is
`unreviewed` or `reviewed`; `outcome` is `success`, `unknown` or `failure`.
These are supplied claims. IDs must be unique across the plan, and the same
normalized request cannot appear twice, including across splits. Teacher
responses are preserved as supplied; the tool does not invent missing answers.
The source reference and original-source SHA-256 are supplied claims for later
traceability; this builder does not open the original source or verify them.

Create a private JSON plan alongside the source. A minimal train-only plan has
this structure; replace every uppercase placeholder and set resource limits to
values appropriate for the machine:

```json
{
  "schema_version": "lfg.assessment-distillation-plan.v1",
  "revision": "OPERATOR_REVISION",
  "task": "OPERATOR_TASK_ID",
  "system_prompt": "OPERATOR_SUPPLIED_LFG_ASSESSMENT_INSTRUCTION",
  "authorization_attestation": "OPERATOR_SOURCE_SELECTION_AND_AUTHORIZATION",
  "redaction_attestation": "OPERATOR_REDACTION_PROCESS",
  "forbidden_literals": [],
  "include_outcomes": ["success", "unknown"],
  "max_source_bytes": 1000000,
  "max_total_bytes": 1000000,
  "max_examples": 1000,
  "sources": [{"path":"PRIVATE_SOURCE.jsonl","sha256":"SOURCE_SHA256",
               "project":"PROJECT_ID","project_group":"PROJECT_GROUP_ID",
               "split":"train","origin":"owner_chat"}]
}
```

`origin` is `owner_chat` or `owner_project`. Sources require the SHA-256 of
their exact bytes; PowerShell's `(Get-FileHash PRIVATE_SOURCE.jsonl -Algorithm
SHA256).Hash.ToLowerInvariant()` obtains it. `forbidden_literals` optionally
lists case-sensitive strings that must not appear in exported content or
metadata. It is a limited extra check, not a secret scanner or proof that all
secrets were removed. Do not place real secret values in a tracked plan. The
operator must ensure redaction before use and protect both plan and dataset.
Source origin, authorization and redaction are recorded attestations; hashing a
file does not prove them.

```powershell
python ml/lfg-encoder/distill.py build-sft --plan PRIVATE_PLAN.json `
  --output-dir NEW_PRIVATE_OUTPUT_DIRECTORY
```

The same inputs can be passed through the project entry points with
`make ml-distill-build DISTILL_PLAN=PRIVATE_PLAN.json DISTILL_OUTPUT_DIR=NEW_PRIVATE_OUTPUT_DIRECTORY`
or `scripts/deploy.ps1 -Action DistillBuild -DistillPlan PRIVATE_PLAN.json
-DistillOutputDirectory NEW_PRIVATE_OUTPUT_DIRECTORY`.

The output directory must not exist. Inside any Git working tree it must be
Git-ignored; Git must be available to check the location. The tool writes `train.jsonl`,
an optional `validation.jsonl`, and a manifest with input, builder and output
digests and coverage counts. A missing manifest indicates an incomplete write.
Do not commit or publish these files; learned weights can also disclose private
data. The builder checks configured literal and size limits, source identity,
project-group consistency and normalized duplicates before output. It checks
input identity again before writing, but does not lock the source files and
cannot prove that they did not change between checks.

The fixture tests exercise preparation only. As of the local audit, the existing
review workspace has no reviewed assessment targets, and no assessment SFT
dataset or student training run has been produced from it by this tool.

On 2026-11-02, a local development-only audit combined the existing private
training workspace with two Onemanwho validation workspaces. It reported 598
training requests and 47 validation requests, with zero approved decisions or
verified flags in those sources. The development plan and audit are in
`ml/lfg-encoder/dataset/private/split-reservation-20261102-v1` in the original
checkout; they reference private files and are not distributed. The separate
Makers-Impulse test source was reserved by path and file digest outside that
development plan. Its existing workspace contained 785 pending requests, but
none were used to construct the development dataset or derive targets. The
reservation records declared group separation, not proof of semantic or
conversation independence; a near-duplicate and relationship audit remains
necessary. A separate local separation audit read the reserved source only to
check IDs, declared project/family assignments and exact normalized-text
overlap; those checks passed. It did not expose test text to a model or use it
for training, and does not prove semantic or conversational independence.
No teacher assessment was generated, redaction was not attested by
this audit, and no final training or evaluation corpus was created.

A separate train-only pipeline pilot was prepared from the user's actual
2026-11-02 request in this conversation, with one assessment authored by the
current assistant. Its private source, plan and prepared output are under
`ml/lfg-encoder/dataset/private/current-request-assessment-pilot-20261102-v1`
in the original checkout. The request is a local transcription of the message,
not independently verified against a session file. The assessment is explicitly
`unreviewed`, its outcome is `unknown`, and unknown model/effort/agent choices
remain null. The manifest is not promotion eligible. This single example only
exercises preparation of a real input with a labeled AI proposal; it does not
establish redaction completeness, model quality, dataset sufficiency, validation
or any training run. It was not added to the held-out test data.
