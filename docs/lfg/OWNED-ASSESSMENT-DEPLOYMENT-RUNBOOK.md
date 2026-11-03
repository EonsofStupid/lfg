# Private loopback assessment service deployment runbook

This runbook covers a single Linux host running the experimental owned
assessment API and bundled UI as a systemd **user** service. It keeps the
listener on loopback, stores credentials and feedback outside Git, and uses
systemd cgroup controls selected from measurements on the target host.

It does not qualify the model, establish assessment accuracy, provide GPU
isolation, or turn the loopback API into an authenticated network service. Do
not expose this service on a LAN or public interface. Use a separately designed
and tested authentication boundary for network hosting.

## Prerequisites and operator choices

Before deployment, record privately (not in this repository):

- The target hostname or SSH alias, Linux user, CPU architecture, systemd user
  manager, and whether user lingering is already enabled.
- A target-compatible `lfg-server` binary with the bundled UI and required
  owned cached inference features. Confirm the feature set and build identity
  from release evidence. The experimental `cached_onnx` embedded build uses
  `owned-byte-onnx`, `embedded-owned-cached-byte`, and `embedded-ui`. The copy
  steps here cover embedded mode; directory artifact mode needs its own
  verified graph-package staging procedure.
- The matching framing file, training manifest, and native runtime config, plus
  their trusted expected SHA-256 digests. Do not copy source credentials,
  feedback, state, logs, training datasets, or separate weights. The embedded
  build carries its cached graphs in the binary; directory artifact mode has
  different artifact-copy requirements.
- A free loopback port, unique service name, private installation path, and
  resource limits selected from measured headroom and coexistence needs.
- The output policy that matches the checkpoint's trained response contract.
  The example config is a starting point, not a universal checkpoint policy.

On the operator workstation, choose `SSH_ALIAS`, `REMOTE_PRIVATE_ROOT`,
`UNIT_NAME`, `REMOTE_PORT`, and `LOCAL_PORT`. Use absolute paths on the target
for `REMOTE_PRIVATE_ROOT`; keep it outside Git. On the target, set
`PRIVATE_ROOT` to the same path and export `UNIT_NAME` (without `.service`),
`REMOTE_PORT`, `CPU_QUOTA`, `MEMORY_HIGH`, `MEMORY_MAX`, and `SMOKE_TIMEOUT`.
Choose instance-specific resource values. Use a remote private path without
spaces for the `scp` examples. Do not copy example hostname, user, port, or path
values from another installation.

Keep source identities and artifact digests in a private operator record.
Redact paths and metadata before sharing. Do not save tokens, prompt text,
model response text, or private logs in Git.

## Choose and verify the resource envelope

Take a baseline before starting the new service. Measure available memory,
swap, CPU load, GPU utilization and memory where the host exposes them, and
health and latency for each existing service that must coexist. Confirm the
identity of each endpoint and its loopback/public binding; do not assume an
unlabeled port belongs to a named service. Repeat the same checks during a
bounded inference and after restart.

Choose `CPUQuota`, `MemoryHigh`, and `MemoryMax` from those measurements and
coexistence requirements. `MemoryHigh` applies pressure; `MemoryMax` is a
last-resort limit that can terminate the service. Keep margin for model graphs,
cache tensors, runtime workspace and allocator overhead. The API's
`maxCacheStateBytesPerRequest` bounds declared cache tensors only; it is not a
process memory limit. `workerThreads` and `maxConcurrent` are not hard CPU or
RAM quotas. A CPU quota does not isolate shared GPU memory or GPU compute.

One reference installation used `CPUQuota=200%`, `MemoryHigh=4G`,
`MemoryMax=8G`, `workerThreads=1`, and `maxConcurrent=1`, after a bounded
coexistence run. These are tested settings for that installation, **not product
defaults or recommendations for another host**. Re-measure rather than copying
them. This runbook intentionally leaves the values as operator inputs.

## Prepare private directories and fresh credentials

Run commands on the target host as the account that will own the user service.
Set an instance-specific `PRIVATE_ROOT` outside every Git worktree. Its
canonical path and all existing ancestors must be owned by the service account
or root; no ancestor may be group/world writable unless it has the sticky bit.
The feedback store enforces these ancestor checks.

```bash
set -euo pipefail
umask 077
INSTANCE_ID='replace-with-a-local-instance-name'
PRIVATE_ROOT="${PRIVATE_ROOT:-$HOME/private/lfg/$INSTANCE_ID}"
install -d -m 700 \
  "$PRIVATE_ROOT/staging" "$PRIVATE_ROOT/install" "$PRIVATE_ROOT/config" \
  "$PRIVATE_ROOT/state" "$PRIVATE_ROOT/feedback"
namei -l "$PRIVATE_ROOT/feedback"
: "${UNIT_NAME:?set a unique user-service name without the .service suffix}"
: "${REMOTE_PORT:?set a free loopback port}"
: "${CPU_QUOTA:?set a measured CPUQuota value}"
: "${MEMORY_HIGH:?set a measured MemoryHigh value}"
: "${MEMORY_MAX:?set a measured MemoryMax value}"
: "${SMOKE_TIMEOUT:?set a bounded smoke timeout, such as 120s}"
export PRIVATE_ROOT UNIT_NAME REMOTE_PORT CPU_QUOTA MEMORY_HIGH MEMORY_MAX SMOKE_TIMEOUT
```

Run target-side snippets in the same shell, or set and export their required
non-secret variables again in each new terminal. In particular, the UI/auth
and post-restart checks need `PRIVATE_ROOT` and `REMOTE_PORT` in their process
environment.

If `namei` shows an unsafe owner or writable ancestor, choose a safe private
path. Do not change ownership or permissions on unrelated parent directories
to make the service start. Keep the private root and its child directories at
mode `0700`.

Generate two fresh, independent 256-bit credentials on the target. This writes
them directly to a mode-`0600` environment file and prints neither value. Do
not source a credential file with shell tracing enabled.

```bash
python3 - "$PRIVATE_ROOT/config/credentials.env" <<'PY'
import os, secrets, sys
path = sys.argv[1]
values = []
for _ in range(2):
    while True:
        value = secrets.token_hex(32)
        if len(value.encode()) >= 32 and len(set(value.encode())) >= 12:
            values.append(value)
            break
if values[0] == values[1]:
    raise SystemExit('credential generation did not produce distinct values')
data = (
    'LFG_OWNED_ASSESSMENT_TOKEN=' + values[0] + '\n' +
    'LFG_FEEDBACK_TOKEN=' + values[1] + '\n'
).encode()
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'wb') as handle:
    handle.write(data)
PY
chmod 600 "$PRIVATE_ROOT/config/credentials.env"
```

For local retrieval, read one variable directly to a trusted terminal; do not
paste its output into chat, tickets, shell history, or logs:

```bash
sed -n 's/^LFG_OWNED_ASSESSMENT_TOKEN=//p' "$PRIVATE_ROOT/config/credentials.env"
sed -n 's/^LFG_FEEDBACK_TOKEN=//p' "$PRIVATE_ROOT/config/credentials.env"
```

## Stage and verify only the required artifacts

Copy the target-compatible binary and the three matching metadata files to the
private staging directory. Define `SSH_ALIAS`, `REMOTE_PRIVATE_ROOT`,
`BINARY`, `FRAMING_FILE`, `TRAINING_MANIFEST`, and `RUNTIME_CONFIG` in the
operator workstation shell. For example:

```bash
: "${SSH_ALIAS:?set the target SSH alias}"
: "${REMOTE_PRIVATE_ROOT:?set the absolute private target path}"
: "${BINARY:?set the verified target binary path}"
: "${FRAMING_FILE:?set the matching framing metadata path}"
: "${TRAINING_MANIFEST:?set the matching training manifest path}"
: "${RUNTIME_CONFIG:?set the matching runtime config path}"
scp "$BINARY" "$SSH_ALIAS:$REMOTE_PRIVATE_ROOT/staging/lfg-server"
scp "$FRAMING_FILE" "$SSH_ALIAS:$REMOTE_PRIVATE_ROOT/staging/framing.json"
scp "$TRAINING_MANIFEST" "$SSH_ALIAS:$REMOTE_PRIVATE_ROOT/staging/training.json"
scp "$RUNTIME_CONFIG" "$SSH_ALIAS:$REMOTE_PRIVATE_ROOT/staging/runtime.json"
```

Back on the target, make staged files owner-only before reading or comparing
them:

```bash
chmod 700 "$PRIVATE_ROOT/staging" "$PRIVATE_ROOT/install"
chmod 600 "$PRIVATE_ROOT/staging/lfg-server" \
  "$PRIVATE_ROOT/staging/framing.json" \
  "$PRIVATE_ROOT/staging/training.json" \
  "$PRIVATE_ROOT/staging/runtime.json"
```

Set the four expected digests from trusted release or export evidence, not
from an unverified copy. Compare them at the source and on the target before
installation. The hash command prints values for comparison; acceptance must
be against the independently recorded expected digests:

```bash
sha256sum "$PRIVATE_ROOT/staging/lfg-server" \
  "$PRIVATE_ROOT/staging/framing.json" \
  "$PRIVATE_ROOT/staging/training.json" \
  "$PRIVATE_ROOT/staging/runtime.json"
```

Compare each reported value with its independently recorded expected digest.
Then install the verified files:

```bash
install -m 700 "$PRIVATE_ROOT/staging/lfg-server" "$PRIVATE_ROOT/install/lfg-server"
install -m 600 "$PRIVATE_ROOT/staging/framing.json" "$PRIVATE_ROOT/install/framing.json"
install -m 600 "$PRIVATE_ROOT/staging/training.json" "$PRIVATE_ROOT/install/training.json"
install -m 600 "$PRIVATE_ROOT/staging/runtime.json" "$PRIVATE_ROOT/install/runtime.json"
sha256sum "$PRIVATE_ROOT/install/lfg-server" "$PRIVATE_ROOT/install/framing.json" \
  "$PRIVATE_ROOT/install/training.json" "$PRIVATE_ROOT/install/runtime.json"
```

Install only after every digest matches its independently obtained value. Keep
the installed executable mode `0700` and metadata mode `0600`. Run `file` and
the installed binary's help on the target to confirm architecture and the
expected `serve` command:

```bash
file "$PRIVATE_ROOT/install/lfg-server"
"$PRIVATE_ROOT/install/lfg-server" --help
```

A digest match does not establish model quality or artifact provenance by
itself.

Do not copy a source `service.json` or environment file as the target config:
they can contain source paths or sensitive metadata. Create a target config
with target-local paths, current artifact pins, and the output policy that
matches the selected checkpoint. For cached embedded mode, set `backend` to
`cached_onnx`, `artifactMode` to `embedded`, and pin `expectedModelSha256`,
`expectedCachedManifestSha256`, and `expectedFramingSha256`. Set `framingFile`,
`trainingManifestFile`, and `runtimeConfigFile` to the installed metadata
paths. These steps target embedded mode; do not select directory mode without
staging and verifying the manifest and both graph files as a separate artifact
package. Keep the config mode `0600`.

Set request, context, output, body, graph, cache, worker, and timeout bounds
from the checkpoint contract and measured workload. Confirm they satisfy the
runtime's limits. In particular, `maxGeneratedBytes` must fit the runtime's
generation bound and stop suffix; `maxWorkerOutputBytes` must fit the maximum
escaped worker response. Do not carry an example's small generation limit or
resource settings forward by habit.

The repository example is disabled and is not a complete cached-backend
config. Create a private output-policy JSON array that matches the checkpoint,
then export the expected artifact digests and measured numeric limits listed
below. This generator fixes `artifactMode` to `embedded`, constructs a
target-local cached config, validates positive numeric bounds, and writes the
file owner-only without printing config values:

```bash
: "${OUTPUT_POLICY_FILE:?set a private JSON array of required output fields}"
: "${EXPECTED_MODEL_SHA256:?set the trusted model/source-weight digest}"
: "${EXPECTED_CACHED_MANIFEST_SHA256:?set the trusted cached manifest digest}"
: "${EXPECTED_FRAMING_SHA256:?set the trusted framing-file digest}"
: "${MAX_RUNTIME_CONFIG_BYTES:?set the runtime config byte limit}"
: "${MAX_METADATA_BYTES:?set the metadata byte limit}"
: "${MAX_HTTP_BODY_BYTES:?set the HTTP body byte limit}"
: "${MAX_REQUEST_BYTES:?set the request text byte limit}"
: "${MAX_CONTEXT_BYTES:?set the framed context byte limit}"
: "${MAX_GENERATED_BYTES:?set the output byte limit}"
: "${MAX_WORKER_OUTPUT_BYTES:?set the serialized worker output limit}"
: "${TIMEOUT_MS:?set the measured request timeout}"
: "${MAX_CONCURRENT:?set the concurrency bound}"
: "${WORKER_THREADS:?set the requested worker thread count}"
: "${MAX_CACHED_GRAPH_BYTES:?set the cached graph byte bound}"
: "${MAX_CACHE_STATE_BYTES:?set the cache tensor state bound}"
export OUTPUT_POLICY_FILE EXPECTED_MODEL_SHA256 EXPECTED_CACHED_MANIFEST_SHA256
export EXPECTED_FRAMING_SHA256 MAX_RUNTIME_CONFIG_BYTES MAX_METADATA_BYTES
export MAX_HTTP_BODY_BYTES MAX_REQUEST_BYTES MAX_CONTEXT_BYTES MAX_GENERATED_BYTES
export MAX_WORKER_OUTPUT_BYTES TIMEOUT_MS MAX_CONCURRENT WORKER_THREADS
export MAX_CACHED_GRAPH_BYTES MAX_CACHE_STATE_BYTES
python3 - "$PRIVATE_ROOT/config/service.json" "$OUTPUT_POLICY_FILE" <<'PY'
import json, os, sys
target, policy_path = sys.argv[1:]
def positive(name):
    value = int(os.environ[name])
    if value <= 0:
        raise SystemExit(name + ' must be positive')
    return value
with open(policy_path, encoding='utf-8') as handle:
    policy = json.load(handle)
if not isinstance(policy, list) or not all(isinstance(item, dict) for item in policy):
    raise SystemExit('output policy must be a JSON array of field objects')
root = os.environ['PRIVATE_ROOT']
config = {
    'schemaVersion': 'lfg.owned-assessment-service.v1',
    'enabled': True,
    'framingFile': root + '/install/framing.json',
    'trainingManifestFile': root + '/install/training.json',
    'runtimeConfigFile': root + '/install/runtime.json',
    'maxRuntimeConfigBytes': positive('MAX_RUNTIME_CONFIG_BYTES'),
    'expectedModelSha256': os.environ['EXPECTED_MODEL_SHA256'],
    'expectedFramingSha256': os.environ['EXPECTED_FRAMING_SHA256'],
    'maxMetadataBytes': positive('MAX_METADATA_BYTES'),
    'requiredOutputFields': policy,
    'maxHttpBodyBytes': positive('MAX_HTTP_BODY_BYTES'),
    'maxRequestBytes': positive('MAX_REQUEST_BYTES'),
    'maxContextBytes': positive('MAX_CONTEXT_BYTES'),
    'maxGeneratedBytes': positive('MAX_GENERATED_BYTES'),
    'maxWorkerOutputBytes': positive('MAX_WORKER_OUTPUT_BYTES'),
    'timeoutMs': positive('TIMEOUT_MS'),
    'maxConcurrent': positive('MAX_CONCURRENT'),
    'workerThreads': positive('WORKER_THREADS'),
    'backend': 'cached_onnx',
    'artifactMode': 'embedded',
    'expectedCachedManifestSha256': os.environ['EXPECTED_CACHED_MANIFEST_SHA256'],
    'maxCachedGraphBytes': positive('MAX_CACHED_GRAPH_BYTES'),
    'maxCacheStateBytesPerRequest': positive('MAX_CACHE_STATE_BYTES'),
}
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w', encoding='utf-8') as handle:
    json.dump(config, handle, indent=2)
    handle.write('\n')
PY
chmod 600 "$PRIVATE_ROOT/config/service.json"
```

Create the bounded smoke config from the final target config so it retains the
same artifact pins, metadata paths, and output policy. This example uses eight
generated bytes and a 60-second inference timeout only for the smoke; change
those values to fit the selected runtime. The smoke generation limit must still
fit the configured assistant stop suffix. These are not serving defaults.

```bash
SMOKE_GENERATED_BYTES="${SMOKE_GENERATED_BYTES:-8}"
SMOKE_API_TIMEOUT_MS="${SMOKE_API_TIMEOUT_MS:-60000}"
python3 - "$PRIVATE_ROOT/config/service.json" "$PRIVATE_ROOT/config/smoke.json" \
  "$SMOKE_GENERATED_BYTES" "$SMOKE_API_TIMEOUT_MS" <<'PY'
import json, os, sys
source, target = sys.argv[1:3]
with open(source, encoding='utf-8') as handle:
    config = json.load(handle)
config['maxGeneratedBytes'] = int(sys.argv[3])
config['timeoutMs'] = int(sys.argv[4])
if config['maxGeneratedBytes'] <= 0 or config['timeoutMs'] <= 0:
    raise SystemExit('smoke bounds must be positive')
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w', encoding='utf-8') as handle:
    json.dump(config, handle, indent=2)
    handle.write('\n')
PY
chmod 600 "$PRIVATE_ROOT/config/smoke.json"
```

## Run a bounded pre-unit smoke

Check that the chosen loopback port is free, then create a separate smoke
config outside Git. It should pin the same model, cache manifest, framing,
training manifest, and runtime metadata as the final config, but use a small
`maxGeneratedBytes` and short `timeoutMs`. Keep its output policy and paths
consistent with the final service. The following check starts only a
temporary, resource-bounded scope; run the UI and authentication checks from a
second terminal while the foreground command is active. It automatically
expires after the selected smoke timeout.

```bash
ss -ltn "sport = :$REMOTE_PORT"
set -a
. "$PRIVATE_ROOT/config/credentials.env"
set +a
export LFG_OWNED_ASSESSMENT_CONFIG="$PRIVATE_ROOT/config/smoke.json"
export LFG_STATE_DIR="$PRIVATE_ROOT/state"
export LFG_FEEDBACK_DIR="$PRIVATE_ROOT/feedback"
systemd-run --user --scope --quiet --unit="${UNIT_NAME}-smoke" \
  --property="CPUQuota=$CPU_QUOTA" \
  --property="MemoryHigh=$MEMORY_HIGH" \
  --property="MemoryMax=$MEMORY_MAX" \
  /usr/bin/timeout --signal=TERM "$SMOKE_TIMEOUT" \
  "$PRIVATE_ROOT/install/lfg-server" serve --bind "127.0.0.1:$REMOTE_PORT"
```

After the smoke exits, confirm its scope is inactive and the port is released.
Do not create the permanent user unit if artifact pins, status, authentication,
UI, or coexistence checks fail. Keep testing processes and smoke scopes out of
the persistent service configuration.

## Install an independent systemd user unit

Choose a unique unit name and loopback port. Verify the port is free before
writing the unit. Render the placeholders below to literal paths and measured
values; systemd does not expand arbitrary shell variables in unit properties.

```ini
[Unit]
Description=LFG experimental owned cached assessment service
After=network.target

[Service]
Type=simple
WorkingDirectory=<PRIVATE_ROOT>/install
EnvironmentFile=<PRIVATE_ROOT>/config/credentials.env
Environment=LFG_OWNED_ASSESSMENT_CONFIG=<PRIVATE_ROOT>/config/service.json
Environment=LFG_STATE_DIR=<PRIVATE_ROOT>/state
Environment=LFG_FEEDBACK_DIR=<PRIVATE_ROOT>/feedback
ExecStart=<PRIVATE_ROOT>/install/lfg-server serve --bind 127.0.0.1:<PORT>
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
KillMode=control-group
UMask=0077
NoNewPrivileges=yes
CPUQuota=<MEASURED_CPU_QUOTA>
MemoryHigh=<MEASURED_MEMORY_HIGH>
MemoryMax=<MEASURED_MEMORY_MAX>

[Install]
WantedBy=default.target
```

Install this as
`$HOME/.config/systemd/user/$UNIT_NAME.service`, mode `0600`. Keep
secrets in `EnvironmentFile`; do not put tokens in `Environment=`, the unit,
`ExecStart`, URLs, or command-line arguments. The user service inherits its
account identity; do not add a `User=` override.

After rendering the template into the private config directory, install it
without changing unrelated unit files:

```bash
install -d -m 700 "$HOME/.config/systemd/user"
install -m 600 "$PRIVATE_ROOT/config/$UNIT_NAME.service" \
  "$HOME/.config/systemd/user/$UNIT_NAME.service"
```

**Filesystem namespace limitation:** a systemd filesystem namespace using
`ReadWritePaths=` or `ProtectSystem=` can make otherwise safe ancestors appear
unmapped to the feedback store's ownership checks. In the reference deployment,
feedback remained unavailable with those directives and became available
when that filesystem sandbox was removed. Keep `NoNewPrivileges`, the private
owner-only directory tree, and measured cgroup limits. If you add filesystem
sandboxing, verify feedback status and an authenticated write/read from inside
the actual service before relying on it; do not change unrelated system
directory ownership to work around a failure.

Reload and enable the new unit only after the bounded smoke succeeds:

```bash
systemctl --user daemon-reload
systemctl --user enable --now "$UNIT_NAME.service"
systemctl --user is-active "$UNIT_NAME.service"
systemctl --user is-enabled "$UNIT_NAME.service"
loginctl show-user "$(id -un)" -p Linger
```

If linger is disabled, arrange the appropriate administrator action for this
account; do not change other users' linger or units. An enabled unit plus
linger configures startup after reboot, but does not replace a reboot test.

## Verify UI, authentication, feedback, and cgroup limits

Use a local SSH tunnel for browser access:

```bash
: "${SSH_ALIAS:?set the target SSH alias}"
: "${LOCAL_PORT:?set a free local port}"
: "${REMOTE_PORT:?set the target loopback port}"
ssh -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$SSH_ALIAS"
```

Open `http://127.0.0.1:<LOCAL_PORT>/` locally. The server has no dedicated
`/health` route in the current binary; check the UI root, a JavaScript asset,
and the authenticated status APIs. Existing-service health URLs are
operator-specific and must be checked separately before, during, and after
load.

Test the two distinct tokens without printing either. This example reads the
owner-only file into process memory, checks successful and cross-token
requests, and prints only HTTP codes and status fields:

```bash
: "${PRIVATE_ROOT:?set the target private root}"
: "${REMOTE_PORT:?set the target loopback port}"
export PRIVATE_ROOT REMOTE_PORT
set -a
. "$PRIVATE_ROOT/config/credentials.env"
set +a
python3 - <<'PY'
import json, os, re, urllib.request, urllib.error
base = 'http://127.0.0.1:' + os.environ['REMOTE_PORT']
ui = urllib.request.urlopen(base + '/', timeout=10)
html = ui.read().decode('utf-8', 'replace')
scripts = re.findall('src="([^"]+)"', html)
script_codes = []
for path in scripts:
    with urllib.request.urlopen(base + path, timeout=10) as response:
        script_codes.append(response.status)
def get(path, token=None):
    headers = {} if token is None else {'X-LFG-Session-Token': token}
    request = urllib.request.Request(base + path, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
owned, owned_body = get('/api/lfg/assess/owned/status', os.environ['LFG_OWNED_ASSESSMENT_TOKEN'])
owned_cross, _ = get('/api/lfg/assess/owned/status', os.environ['LFG_FEEDBACK_TOKEN'])
owned_missing, _ = get('/api/lfg/assess/owned/status')
feedback, feedback_body = get('/api/lfg/feedback/status', os.environ['LFG_FEEDBACK_TOKEN'])
feedback_cross, _ = get('/api/lfg/feedback/status', os.environ['LFG_OWNED_ASSESSMENT_TOKEN'])
feedback_missing, _ = get('/api/lfg/feedback/status')
print(json.dumps({
    'ui_http': ui.status,
    'javascript_http': script_codes,
    'owned_http': owned,
    'owned_status': json.loads(owned_body).get('status'),
    'owned_cross_token_http': owned_cross,
    'owned_missing_token_http': owned_missing,
    'feedback_http': feedback,
    'feedback_available': json.loads(feedback_body).get('available'),
    'feedback_cross_token_http': feedback_cross,
    'feedback_missing_token_http': feedback_missing,
}, sort_keys=True))
PY
```

Expect authenticated status requests to return `200`, cross-token and
unauthenticated requests to return `403`, owned status `ready`, and feedback
`available: true`. If feedback is unavailable, inspect its sanitized `reason`
and the path ancestry from inside the actual service environment. Do not print
tokens or inspect prompt/response contents.

Check the effective cgroup values rather than relying only on the unit text:

```bash
systemctl --user show "$UNIT_NAME.service" \
  -p ControlGroup -p CPUQuotaPerSecUSec -p MemoryHigh -p MemoryMax
CONTROL_GROUP="$(systemctl --user show --value -p ControlGroup "$UNIT_NAME.service")"
cat "/sys/fs/cgroup$CONTROL_GROUP/cpu.max"
cat "/sys/fs/cgroup$CONTROL_GROUP/memory.high"
cat "/sys/fs/cgroup$CONTROL_GROUP/memory.max"
cat "/sys/fs/cgroup$CONTROL_GROUP/memory.peak"
cat "/sys/fs/cgroup$CONTROL_GROUP/cpu.stat"
```

The user hierarchy must expose the CPU and memory controllers. Record baseline,
in-flight and post-request health and resource measurements. Stop the new unit
if an existing workload degrades or host pressure crosses the operator's
pre-set safety thresholds. Cgroup limits do not reserve GPU memory or isolate
GPU compute.

An actual inference check is optional and must use an authorized request. Keep
prompt and response in memory; record only digests, typed status, elapsed time,
output length, and resource measurements. `status: "ok"` means the stop
condition and configured response-shape policy passed; it does not establish
that the answer is correct. Do not feed a deployment diagnostic into training
or treat it as user-confirmed feedback.

If persistence is part of deployment acceptance, create a private, sanitized
operational diagnostic using an authorized input/output digest only. Use
`assessmentRef` as a non-empty string, `provenance` and
`originalRecommendation` as JSON objects, and `outcome: {"status":"unknown"}`.
Mark the diagnostic unverified and not training-eligible. Do not store prompt or
response text. Set `INPUT_SHA256`, `OUTPUT_SHA256`, `INFERENCE_STATUS`,
`INFERENCE_ELAPSED_MS`, and `OUTPUT_BYTES` from the just-observed request; the
shell below sources only the private token file. The example stores an
operational diagnostic, not user-approved success or a training label.

```bash
: "${PRIVATE_ROOT:?set the target private root}"
: "${REMOTE_PORT:?set the target loopback port}"
export PRIVATE_ROOT REMOTE_PORT
set -a
. "$PRIVATE_ROOT/config/credentials.env"
set +a
export INPUT_SHA256 OUTPUT_SHA256 INFERENCE_STATUS INFERENCE_ELAPSED_MS OUTPUT_BYTES
python3 - <<'PY'
import hashlib, json, os, re, urllib.request, urllib.error
root = os.environ['PRIVATE_ROOT']
token = os.environ['LFG_FEEDBACK_TOKEN']
for name in ('INPUT_SHA256', 'OUTPUT_SHA256'):
    if not re.fullmatch(r'[0-9a-f]{64}', os.environ[name]):
        raise SystemExit(name + ' must be a lowercase SHA-256 digest')
base = 'http://127.0.0.1:' + os.environ['REMOTE_PORT']
def call(path, method='GET', body=None):
    headers = {'X-LFG-Session-Token': token}
    data = None
    if body is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(body, separators=(',', ':')).encode()
    request = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
state = {
    'assessmentRef': 'sha256:' + os.environ['INPUT_SHA256'],
    'provenance': {
        'kind': 'operational_diagnostic',
        'inputSha256': os.environ['INPUT_SHA256'],
        'outputSha256': os.environ['OUTPUT_SHA256'],
        'verifiedOutcome': False,
        'trainingEligible': False,
        'promptStored': False,
        'responseStored': False,
    },
    'originalRecommendation': {
        'status': os.environ['INFERENCE_STATUS'],
        'outputSha256': os.environ['OUTPUT_SHA256'],
        'elapsedMs': int(os.environ['INFERENCE_ELAPSED_MS']),
        'outputBytes': int(os.environ['OUTPUT_BYTES']),
    },
    'outcome': {'status': 'unknown'},
}
code, body = call('/api/lfg/feedback', 'POST', state)
if code != 201:
    raise SystemExit('feedback create returned HTTP ' + str(code))
record = json.loads(body)['record']
record_id = record['feedbackId']
id_path = os.path.join(root, 'state', 'diagnostic_feedback_id')
fd = os.open(id_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w', encoding='utf-8') as handle:
    handle.write(record_id + '\n')
read_code, read_body = call('/api/lfg/feedback/' + record_id)
loaded = json.loads(read_body).get('record', {}) if read_code == 200 else {}
verified = (
    loaded.get('assessmentRef') == state['assessmentRef'] and
    loaded.get('provenance', {}).get('outputSha256') == os.environ['OUTPUT_SHA256'] and
    loaded.get('outcome', {}).get('status') == 'unknown' and
    loaded.get('provenance', {}).get('trainingEligible') is False and
    'promptText' not in loaded and 'responseText' not in loaded
)
print(json.dumps({'create_http': code, 'read_http': read_code, 'digest_only_readback': verified}))
PY
```

Restart the unit, then verify the same private record ID and digests are still
available. In a fresh shell, set/export `PRIVATE_ROOT`, `REMOTE_PORT`,
`INPUT_SHA256`, and `OUTPUT_SHA256`, then source the credential file. This
second request prints only a boolean and HTTP status:

```bash
: "${PRIVATE_ROOT:?set the target private root}"
: "${REMOTE_PORT:?set the target loopback port}"
: "${INPUT_SHA256:?set the authorized request digest}"
: "${OUTPUT_SHA256:?set the response digest}"
export PRIVATE_ROOT REMOTE_PORT INPUT_SHA256 OUTPUT_SHA256
set -a
. "$PRIVATE_ROOT/config/credentials.env"
set +a
python3 - <<'PY'
import json, os, urllib.request
root = os.environ['PRIVATE_ROOT']
token = os.environ['LFG_FEEDBACK_TOKEN']
record_id = open(os.path.join(root, 'state', 'diagnostic_feedback_id'), encoding='utf-8').read().strip()
request = urllib.request.Request(
    'http://127.0.0.1:' + os.environ['REMOTE_PORT'] + '/api/lfg/feedback/' + record_id,
    headers={'X-LFG-Session-Token': token},
)
with urllib.request.urlopen(request, timeout=15) as response:
    payload = json.loads(response.read())
record = payload.get('record', {})
provenance = record.get('provenance', {})
persisted = (
    record.get('assessmentRef') == 'sha256:' + os.environ['INPUT_SHA256'] and
    provenance.get('outputSha256') == os.environ['OUTPUT_SHA256'] and
    provenance.get('trainingEligible') is False and
    record.get('outcome', {}).get('status') == 'unknown' and
    'promptText' not in record and 'responseText' not in record
)
print(json.dumps({'reload_http': response.status, 'digest_only_persisted': persisted}, sort_keys=True))
PY
```

Retain the diagnostic ID under `PRIVATE_ROOT/state/` and the SQLite journal
under its sibling `PRIVATE_ROOT/feedback/` directory. Keep both private.

## Restart, journal, and rollback

Restart and recheck the unit, both authenticated status routes, UI, feedback
record (if created), existing-service health, and effective cgroup values:

```bash
systemctl --user restart "$UNIT_NAME.service"
systemctl --user is-active "$UNIT_NAME.service"
systemctl --user is-enabled "$UNIT_NAME.service"
journalctl --user-unit="$UNIT_NAME.service" --since '15 minutes ago' --no-pager
```

The journal is the user manager's journal. Review it locally and keep it
private; do not paste raw logs into Git or chat. The server does not log
inference prompt or response text. Verify actual output through a sanitized
summary rather than dumping the response or the environment.

Rollback affects only this service:

```bash
systemctl --user disable --now "$UNIT_NAME.service"
rm -f "$HOME/.config/systemd/user/$UNIT_NAME.service"
systemctl --user daemon-reload
systemctl --user is-active "$UNIT_NAME.service" || true
ss -ltn "sport = :$REMOTE_PORT"
```

Confirm the listener is released and all pre-existing services remain healthy.
Keep the private state and feedback database until the operator's retention
policy authorizes their removal; do not use a broad recursive delete as part
of routine rollback. Preserve the pinned binary and metadata in a protected
staging area only as long as needed for recovery.

## Qualification limits

This is a single-node, loopback deployment. Its credentials protect local API
routes; they do not provide multi-user tenancy, remote identity, or a tested
network security boundary. An SSH tunnel carries traffic to the loopback
listener but does not make the API suitable for direct network exposure.

The bundled experimental model is not the planned approximately 650 MB
qualified LFG release. A completed request, persistent unit, feedback
journal, clean hash verification, or passing smoke establishes only that
specific path. It is not an accuracy evaluation, representative performance
qualification, or training outcome. Keep independent evaluation material
separate. Gates govern trust and promotion; they must not block authorized
training. Do not automatically promote or publish a checkpoint from this
deployment.
