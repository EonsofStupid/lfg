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
boundary. Require the same configured local secret for embedded and explicitly
configured development UI; do not weaken access checks when either mode lacks it.

Protect the local browser path from cross-site requests and DNS rebinding: validate
the Host against configured loopback authorities, reject non-allowed Origin values,
and require a high-entropy operator-configured secret on every feedback route,
including status, list and read. Send it in `X-LFG-Session-Token`; never bootstrap
or return it over HTTP, log it, or put it in URLs. Do not allow credentialed
wildcard CORS. Configure explicitly allowed local development origins. The
developer UI must accept the secret out of band and hold it only in memory, and
must report when feedback is unavailable rather than pretending it was saved.
Hosted feedback requires an independently designed identity/authorization
contract and is out of scope for this task.

## Done means demonstrated

API and UI tests cover create/restart/read, correction, unknown/confirmed outcomes,
extension fields, stale-revision conflict, rollback as a new revision, concurrent
writes, size and permission failures, Git-path/symlink rejection, local peer checks,
non-loopback denial, cross-origin rejection and absence of raw prompt content by
default. Document configuration, startup, local limitations and exact test evidence.
Actual privacy of pre-existing user files is not inferred from a passing test.

## Backend startup settings and implementation boundary

The server keeps feedback disabled unless `LFG_FEEDBACK_DIR` names a private
directory with an existing parent. On Unix, the server creates a new directory
atomically with mode `0700`; on restart it accepts only a directory and SQLite
database owned by the current effective user with no group or other permission
bits. Parent directories must be owned by the current account or root and may
not be group/other writable unless the sticky bit protects the entry (as with
`/tmp`). This protects against other unprivileged accounts; the same account and
root remain trusted. It rejects symlinked final directories and database files,
Git worktree paths (including resolved paths), and unsafe existing SQLite sidecars before opening the database. Do not
point this setting at a shared or pre-existing data directory.

`LFG_FEEDBACK_MAX_RECORD_BYTES` defaults to 262144,
`LFG_FEEDBACK_MAX_JOURNAL_BYTES` defaults to 67108864, and
`LFG_FEEDBACK_BUSY_TIMEOUT_MS` defaults to 5000. `LFG_FEEDBACK_MAX_PAGE_SIZE`
defaults to 100 and can be configured up to 10000; list and history requests
default to 50 items per page. The journal limit counts the
UTF-8 serialized state bytes for every retained revision, including rollback
revisions. The counter is updated in the same immediate SQLite transaction as
each write, so concurrent writers cannot exceed the configured logical payload
limit. This is not a hard physical disk quota: SQLite pages, indexes, WAL and
shared-memory files add storage overhead. No revisions are pruned automatically.
Set `LFG_FEEDBACK_TOKEN` to an operator-generated random secret of at least 32
non-whitespace bytes before startup. The server refuses feedback setup if the
secret is missing or weak. Enter the same value into the developer UI; status,
list, read, history and writes all require it in `X-LFG-Session-Token`. The API
never returns or logs this secret. Store the environment configuration in the
local service manager's protected settings, not in Git or a command transcript.
`LFG_FEEDBACK_ALLOWED_ORIGINS` accepts a comma-separated list of local HTTP
development origins; `LFG_FEEDBACK_ALLOWED_HOSTS` accepts up to 16 local
development authorities such as `localhost:5173`. Other values are ignored.
The configured secret is not per-launch; authorization is local operator access,
not an authenticated person or multi-user identity system.

The API routes are `GET /api/lfg/feedback/status`, `GET|POST
/api/lfg/feedback`, `GET /api/lfg/feedback/{feedback_id}`, `POST
/api/lfg/feedback/{feedback_id}/revisions`, and `POST
/api/lfg/feedback/{feedback_id}/rollback`. Status reports `defaultPageSize` and
`maxPageSize`. List requests accept `limit` and `cursor`; responses contain
`{schemaVersion,records,nextCursor,hasMore}`. Records are ordered by immutable
`feedbackId` ascending. A cursor continues strictly after the last ID returned,
so updates do not move records between pages. Traversal is not a snapshot: new
IDs after the cursor may appear; new IDs before it require restarting from page
one. Read requests accept `historyLimit` and `historyBefore`; the current record
is always returned while `history` contains latest-first entries. Responses also
include `historyNextBefore` and `historyHasMore`; continue with the revision
number in `historyNextBefore` to load strictly older history. Every history entry
has the complete `state`. Create and append responses contain `{record}`.
Revision writes send `{expectedRevision,state}`; rollback sends
`{expectedRevision,restoreRevision}`. Stale writes return HTTP 409.

The route never obtains assessment prompt text from the assessment engine. The
client sends a selected recommendation snapshot, correction, outcome, provenance
and extensions; manually entered correction and outcome text is intentionally
persisted. `provenance.sourceRef` may contain `sha256:<digest>` for a separately
available source; this link does not guarantee that source can later be recovered.
The server does not semantically screen arbitrary extension values for
prompt-like content. The UI should omit prompt-bearing assessment fields by
default.

Per-record revision history is paginated, though a large page still reflects the
configured per-record and global logical payload limits. Operators who raise
those limits substantially should keep the UI page size bounded.

On Windows, the server opens a guarded local directory only after checking the
effective current-user ACL on the final directory and existing database files,
rejecting Git paths, junctions, reparse points, hard links and insecure existing
paths. The guard remains alive with the database connection and rechecks files
after WAL initialization. Paths require an existing safe parent; unsafe existing
directories are refused rather than re-permissioned. The Windows-specific tests
cover directory creation/reopen, Git markers, junctions, database sidecars and
ACL inheritance. Hosted access and public reverse-proxy use are unsupported.

For an interactive Windows developer run, from a private PowerShell session:

```powershell
$env:LFG_FEEDBACK_DIR = Join-Path $env:LOCALAPPDATA 'LFG-feedback'
$feedbackBytes = New-Object byte[] 32
$feedbackRng = [Security.Cryptography.RandomNumberGenerator]::Create()
$feedbackRng.GetBytes($feedbackBytes)
$feedbackRng.Dispose()
$env:LFG_FEEDBACK_TOKEN = -join ($feedbackBytes | ForEach-Object { $_.ToString('x2') })
$env:LFG_FEEDBACK_TOKEN  # Copy the token from this private, non-transcribed console.
.\target\debug\lfg-server.exe serve --bind 127.0.0.1:4317
```

Use a dedicated directory that does not already contain shared data. Open the
local URL, switch to Inspector, and paste the copied token into **Connect feedback**.
The token remains in the running server's environment and the browser's memory;
restarting the server with a new token
requires reconnecting. For unattended operation, use protected service-manager
environment settings and retain the token securely instead of relying on a
terminal session. These instructions configure feedback; they do not train or
qualify a model.
