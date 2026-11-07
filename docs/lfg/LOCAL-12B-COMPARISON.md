# Local 12B development comparison

`ml/lfg-encoder/compare_local_12b.py` compares a configured, hash-pinned local
Ollama model artifact with an existing classical comparison on the exact same
screened development records, selected labels, target masks, and review-ledger
policy. It is a task-specific exploratory comparison. It does not train LFG,
promote the 12B model, or establish general model quality.

## Execution boundary

The local coordinator re-runs the existing private-corpus loader and screening
checks, verifies the private baseline report, and binds the selected records to
their report, taxonomy, source, screening, policy, and ledger digests. It sends
only the already-screened text, per-record text digests, opaque record-ID
digests, and the exact per-label target values and observation masks. Unknown
target cells remain unknown.

The coordinator transfers no prompt through command arguments, shell history,
logs, or a public model endpoint. It uses authenticated SSH stdin/stdout. Before
writing screened text, it requires a newly created network namespace with only
loopback, a task-owned Ollama process under the configured user, a separate
owner-only model store, a verified loaded model digest, and a host-network
connection refusal to the namespace's loopback listener. A transient systemd
scope applies configured memory and CPU limits. The worker returns aggregate
metrics and status/prediction values through the private pipe; it does not save
prompt text or model response text. Task-created processes are stopped and
checked before the run is reported complete.

The public model artifact is checked against its private acquisition receipt,
registry manifest, object hashes and sizes, and license notice. The exact
quantized-to-upstream weight conversion is not independently proven. The model
name and tag, registry URL, upstream revision claim, and manifest and
model-layer digests and sizes are required configuration pins bound to the
acquisition receipt. License identity is checked against the receipt and its
verified license layer. Quantization is an operator-supplied artifact
description, not an independently parsed or measured property. These are not
a built-in model catalog, and configured names do not independently prove
model lineage. The existing shared Ollama service and cache are not used by
inference.

## Configuration and use

The input is a bounded JSON object with `schema_version` set to
`lfg.local-12b-comparison-config.v2`. It supplies local baseline/config/receipt
paths and their SHA-256 digests; the selected revision and prompt template;
source, report, request, response, and payload byte limits; configured model
name/tag, registry manifest URL, upstream revision claim, manifest and model
layer hashes/size, license name, and quantization description; context,
generation, sampling, and per-request timeout settings; the literal loopback port; remote
SSH host/user, validated executable paths, private artifact/store paths, and
service unit; expected Ollama binary digest/version; minimum observed
memory/disk availability; explicit GPU memory policy; and transient cgroup
memory/CPU and task limits. Readiness, artifact-install, and batch-overhead
timeouts are configurable. Timeout values are checked against safe platform
arithmetic, not a fixed deployment-scale cap. The runner uses shell quoting
for remote commands and rejects unsafe host/user/path syntax. Unknown extension fields are accepted and
ignored. They do not override recognized fields.

The runtime endpoint must be an IPv4 loopback `/api/chat` address. The adapter
does not follow redirects, use proxies, retry, or contact a model registry. It
copies only receipt-verified public artifacts from the private acquisition
stage to a new private Ollama store. A missing digest, owner boundary, namespace,
resource floor, or runtime identity stops the run before screened records are
sent.

Run it with the project's pinned Python environment:

```text
python ml/lfg-encoder/compare_local_12b.py --config PRIVATE_CONFIG --output NEW_PRIVATE_REPORT_DIRECTORY
```

The report directory must be new, outside Git worktrees, and ACL-restricted to
the current owner, SYSTEM, and Administrators. The report contains no raw text,
per-record identifiers, or per-record predictions. It records aggregate
coverage and metrics, answered/abstained/failed/unattempted counts, timing
summaries only for started requests (with missing timing counts), model and
runtime identities, configuration and provenance digests, and resource
measurements with their limits. Comparator code SHA-256 is recorded; a Git
revision is recorded only when the comparator file is clean and Git can verify
it.

## Reading results

Empty label arrays count as negatives only for completed, valid model responses
with `abstain=false`. Refusal, abstention, malformed or out-of-schema output,
truncation, transport failure, and unattempted rows remain separate outcomes.
Metrics use only known target cells and report prediction coverage; subset
accuracy uses rows with every target known and every prediction completed.
The 12B metrics score only its actual validated responses against known target
cells. Classical constant-target fallback policy is not used to create or
replace 12B predictions. The existing classical report lacks per-record
predictions, so same-answered-row paired classical metrics remain unavailable.

Warmup uses empty user content before any private record. The request timing
summary covers only sequential local HTTP requests that started, through the
completed Ollama response or observed failure; non-started requests have no
zero-duration placeholder. It excludes model startup, private-source binding,
SSH setup, and report writing. Before/after host values are point-in-time
snapshots; intervening intervals and other services are not continuously
observed. A timeout or ambiguous transport failure stops subsequent
requests. A failed or partial run cannot be read as a complete denominator.

Unreviewed labels remain unverified proposals. Previously used development
validation is exploratory and is not an independent final test. No final
holdout is opened. GPU memory and per-process GPU use are reported as unknown
when the platform exposes only shared unified memory; the runtime did not report
whether inference ran on CPU or GPU. System-wide inventory is not per-process
attribution. Host RAM and disk snapshots, configured transient scope limits,
and unknown per-run CPU/memory totals are reported separately; these do not
imply a reservation or an LFG model size.
