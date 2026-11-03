# Local assessment teacher collection

The owner authorizes distillation on selected private projects and chats.
`select_teacher_requests.py` deterministically samples train requests across
project labels from a previously screened dataset, applies the selected local
screening policy again and writes a new private selection. It verifies the
train file's recorded digest and does not open validation or test files. The
sampling seed, per-project and total limits, sizes and policy are configuration.
Project labels are source claims, not proof of independent projects or chats.

`collect_local_teacher.py` sends those selected requests, one at a time, to an
explicitly configured OpenAI-compatible chat endpoint on a literal loopback
IP. It does not use a proxy or follow redirects. Its API key is read from the
configured environment variable and is not written to the configuration,
records or report. Set the served model, identity description, instruction,
source origin, required response fields, timeout and size/token limits in a private copy of
the [example configuration](../../ml/lfg-encoder/local-teacher-collection.example.json).
The loopback restriction makes the private-data boundary explicit; run the
collector on the machine hosting the authorized teacher.

The collector requires a complete JSON response from the configured model,
screens string values and keys, and stores an immutable private record per
request. Source, config, policy and code hashes are recorded. `--resume` reuses
verified records only when those identities still match; an OS lock prevents
two collector processes from sharing one output. A failed request can leave a
partial directory and no final manifest. Completed per-project JSONL files can
be passed as the configured `owner_chat` or `owner_project` train sources to `distill.py build-sft` after the
operator supplies a plan. Record metadata preserves source claims; the
collector cannot independently establish original session provenance.

Every output is explicitly **unreviewed**, with **unknown outcome** and no
promotion eligibility. Teacher identity is an operator claim and the served
weights may not be pinned. Screening is pattern-based and can miss secrets;
the selection, teacher responses, logs and derived weights must stay private.
Prompt text may appear in the local teacher service's own logs. The tool does
not establish that a teacher's recommendation is optimal or a task succeeded.
Extra response and configuration fields are accepted; required fields must be
present after screening. Completed traces are not an independent evaluation.

Optional `request_extensions` are passed to the local chat endpoint, for
example to configure the selected model's chat-template reasoning mode. The
collector rejects extensions that would override its model, messages, token
limit, temperature, response format or streaming behavior. Extensions and
their values are part of the immutable collection configuration and therefore
part of the recorded run identity.

```powershell
python ml/lfg-encoder/select_teacher_requests.py --dataset PRIVATE_SCREENED_DATASET `
  --policy PRIVATE_SCREEN_POLICY.json --config PRIVATE_SELECTION_CONFIG.json `
  --output NEW_PRIVATE_SELECTION_DIRECTORY

# Run on the teacher host with the configured key already in its environment.
python ml/lfg-encoder/collect_local_teacher.py --selection PRIVATE_SELECTION_DIRECTORY `
  --config PRIVATE_TEACHER_CONFIG.json --policy PRIVATE_SCREEN_POLICY.json `
  --output NEW_PRIVATE_COLLECTION_DIRECTORY
```

An initial one-request attempt on 2026-11-02 reached the owner's existing
local SWAG endpoint but failed with an HTTP error and produced no completed
trace. The container was observed running immediately before and after that
attempt; a subsequent check found it exited with status 0. The cause has not
been established. A later local log inspection found an `EngineDeadError`, an
HTTP 500 and a shutdown marker; those observations do not establish what caused
the engine failure. The service was not restarted and no further requests were
sent. The pilot selection contained one screened real request; a separate
balanced selection contains 21 requests across 11 declared project labels,
but no teacher responses have been collected for it.
