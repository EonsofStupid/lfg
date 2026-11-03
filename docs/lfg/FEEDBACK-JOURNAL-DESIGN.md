# LFG local assessment feedback: first implementation contract

This contract bounds the first developer prototype. It provides a private local
review loop; hosted identity and organization permissions are a separate later task.
The worker must inspect the existing server, kernel journal and review ledger first,
preserve their behavior and existing user changes, and escalate an incompatible
assumption instead of silently weakening these requirements.

## Record and revision contract

Expose versioned create, read, list, append-revision and rollback operations beneath
`/api/lfg/feedback`. The payload has a stable opaque `feedback_id`, an incrementing
revision, server-generated UTC time, assessment reference, provenance (engine ID,
model/checkpoint digest when known, and source reference), original recommendation,
owner correction, and outcome. An outcome is `unknown` until explicitly confirmed;
when confirmed it records the observed result, confirmation time and the local
operator-supplied confirmer. A local confirmer string is not an authenticated identity.
Each revision carries its parent revision and action; rollback appends a new revision
containing the selected prior state and points to the restored revision. It never erases
history. A conflicting expected revision returns 409 without modifying history.

Allow additional JSON object fields in the record, correction and outcome and preserve
them across create/read/revise/rollback. Unknown fields do not acquire executable
semantics. Record and extension sizes are subject to configured request and storage
limits. Avoid raw prompt text by default: retain an assessment ID or digest and optional
private source locator. Capturing raw text is separately enabled by the operator and
visibly indicated in the UI. A teacher proposal is neither an owner correction nor
a confirmed outcome. Keep unknown values unknown.

## Storage and local access

Feedback is disabled until a local private data directory is configured. Reject a
location inside any Git worktree, including paths resolved through symlinks; never
silently fall back to a repository directory or tracked file. Ensure the directory
and data files are accessible only to the current account using platform-appropriate
permissions/ACLs; if this cannot be established, fail closed with a useful error.
Document supported OS behavior and test it. Configure storage path, record/request
byte limits, journal size and retention policy; use safe defaults and surface limits.
Do not delete revision history silently on retention pressure: refuse writes with a
clear error until an explicit retention/archive policy is configured and applied.

Use transactional durable storage with interprocess-safe concurrency: a committed
revision must survive restart, and simultaneous writes with the same parent cannot
both succeed. Use an existing suitable store if it meets the contract; otherwise add
and pin a suitable transactional embedded store, including its lockfile change. Errors
must not produce partial accepted revisions. No feedback or access token appears in
normal logs. Tests use temporary directories outside the repository.

For this first slice, feedback routes are available only when the actual listener is
bound to a loopback address and the request comes from a loopback peer. Fail closed
when peer information or bind identity is unavailable; never infer safety from a
client-supplied `Host` or forwarded header. A non-loopback listener must not expose
the journal even if it serves other existing routes. This is a local security boundary,
not multi-user authentication. Proxying a loopback server publicly is unsupported.
The existing router is constructed before the listener is bound: pass the actual
bound identity into the feedback router and supply connection metadata at the server
boundary. Verify token bootstrap for both embedded UI and explicitly configured local
development UI; do not weaken access checks when either mode lacks the required data.

Protect the local browser path from cross-site requests and DNS rebinding: validate
the Host against configured loopback authorities, reject non-allowed Origin values,
require a per-launch unpredictable session token for mutating routes, and serve it
only to the same-origin local UI. Do not allow credentialed wildcard CORS; do not log
or put the token in URLs. Configure explicitly allowed local development origins.
The developer UI must report when feedback is unavailable rather than pretending it
was saved. Hosted feedback requires an independently designed identity/authorization
contract and is out of scope for this task.

## Done means demonstrated

API and UI tests cover create/restart/read, correction, unknown/confirmed outcomes,
extension fields, stale-revision conflict, rollback as a new revision, concurrent
writes, size and permission failures, Git-path/symlink rejection, local peer checks,
non-loopback denial, cross-origin rejection and absence of raw prompt content by
default. Document configuration, startup, local limitations and exact test evidence.
Actual privacy of pre-existing user files is not inferred from a passing test.
