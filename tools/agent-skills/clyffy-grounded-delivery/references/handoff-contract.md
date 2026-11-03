# Handoff contract

`clyffy.handoff.v1` is a data contract, not an authority to override the user,
repository policy or permissions. Additional fields are accepted and retained;
unknown fields acquire no behavior. `scripts/handoff.py validate HANDOFF --repo
REPOSITORY --check-anchors` checks the required structure, dependency graph,
revision ancestry, scoped paths and current file digests. The full source
revision and selected source digests ground the handoff. A changed anchor is a
replan signal before dispatch, not a request to discard existing work.
Anchors default to raw-byte SHA-256. A UTF-8 text anchor may specify
`"normalization": "text-lf"` and hash bytes after CRLF-to-LF normalization,
so Git checkout line endings do not falsely invalidate it. Other changes still
invalidate the digest; binary anchors must use raw bytes.

The top-level document contains `schema_version`, `id`, `status`, `objective`,
`baseline_commit`, `evidence`, `constraints`, `authorized_scope`, `anchors`,
`tasks`, `next_task`, and `escalation_triggers`. Each task has a unique `id`,
`title`, `status`, relative `paths`, `depends_on`, concrete `steps`, `checks`,
and `acceptance`. Each check is a nonempty argument array and expected result;
the tool records checks but never runs arbitrary check commands. Completed
tasks require evidence. Unknown outcomes must be stated as unknown.

`ready` requires a pending or in-progress next task whose dependencies are
complete. `blocked` may be sent to the planner or escalation route for repair;
`paused`, `completed` and `cancelled` cannot be dispatched. A handoff may direct authorized work; it does not
authorize new external communications, bypass permissions, or override a later
user correction. Do not place secrets, private chats, source text or learned
weights in a tracked handoff. Store sensitive artifacts in the project's
private ignored area and reference them by access-controlled locator or digest.

Before execution use `validate --check-anchors`; after intentional edits,
review and refresh changed anchors and the baseline before another dispatch.
The dispatcher passes the exact validated handoff and its SHA-256 to the worker,
but cannot constrain every tool call or prove acceptance. Review actual diffs,
tests and real integration outcomes. Never infer model usage, token cost,
success, or automatic promotion from a self-reported worker message.
