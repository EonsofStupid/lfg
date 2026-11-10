# Owned student cached decoding experiment

`ml/lfg-encoder/owned_cached_decode.py` is a Python reference for incremental
inference with the experimental, independently initialized `OwnedByteStudent`.
For each layer it retains the last `conv_kernel - 1` convolution inputs; for
attention layers it also retains the keys and values for the request. Sessions
have independent state, take one byte per batch item, and reject inputs beyond
the configured context. Because this model uses absolute position embeddings,
the implementation does not pretend that dropping old cache entries is
equivalent to recomputing a sliding window. The model must remain in evaluation
mode with unchanged weights for the lifetime of a session.

This is not a Rust runtime, cache sharding, quantization, or production serving.
The current Python reference concatenates growing attention tensors at each
step; a native implementation should benchmark bounded preallocation and its
memory costs rather than assume this implementation is optimal. There is no
prompt-level task quality or concurrent-service measurement here.

## Native cached CLI and service

The Rust `owned-byte-cached-generate` command loads the two graphs once per
invocation, validates the pinned manifest and graph digests, checks their ONNX
tensor contracts and rejects external tensor data before tract loads either
graph. It accepts one JSONL request, uses the first prefill logits to choose the
first byte, and then advances one absolute byte position per decode call. It
does not evict history or attempt to decode beyond the context boundary. The
service invokes this command in a separate child process for each request, so
graph loading is repeated across requests; the in-memory cache belongs to one
request and is discarded when that child exits. This is an experimental
correctness path, not a persistent model server or a performance claim.

The child runtime configuration is byte-bounded and pinned by SHA-256 both at
service startup and when the child reads it. `--max-total-graph-bytes` bounds
the combined graph file size; `--max-cache-state-bytes-per-request` must cover
the manifest's maximum cache tensors for one batch item. The latter does not
bound peak RAM: graph weights, tract workspace, allocation overhead and
transient cache copies are additional. Graph loading currently materializes
bounded graph files in memory before tract compiles the runnable models.

Build the opt-in native runtime with `--features owned-byte-onnx`. To embed a
private export into a standalone binary, additionally enable
`embedded-owned-cached-byte` and provide the `LFG_EMBEDDED_OWNED_CACHED_DIR`,
`LFG_EMBEDDED_OWNED_CACHED_MANIFEST_SHA256`, and
`LFG_EMBEDDED_OWNED_CACHED_MAX_TOTAL_BYTES` build environment values. The
build script verifies the manifest digest, graph hashes and configured total
size before emitting the embedded byte arrays. Private graphs and build
outputs must stay out of Git. Embedding the weights increases executable size
by approximately the graph payload size and does not eliminate tract's runtime
workspace or compilation memory.

This embeds the cached manifest and two graph files only. The CLI still reads
the pinned native runtime JSON and one JSONL request from the paths supplied at
invocation. It does not embed the assessment framing, training manifest,
service configuration, session token, or browser UI. The service still needs
those private configuration files and an out-of-band token; embedded artifact
mode only removes the graph directory dependency. To embed UI assets as well,
build with `embedded-ui` and supply a built, reviewed UI distribution using
`LFG_UI_DIST` (or build it at `ui/master-clyffy-web/dist`). A graph-only binary
is therefore not the complete owner-facing standalone product.

## Local release integration check

On 2026-11-03, an isolated Linux release binary embedded the private cached
graphs and a five-file UI distribution from source commit
`c0e3efafa75ac085b729c828e825c06e0faf621f`. The staged UI archive SHA-256 was
`a48124e345c0fdd9bbeed4a821a992661cbe084175ca5dc8ce8dcbba47dccd57`; the UI
payload was 607,990 bytes. The resulting copied executable was 211,121,608
bytes. From an empty working directory, its UI root and a JavaScript asset both
returned HTTP 200, authenticated status was `ready`, and authenticated
inference returned `ok` with `stop_sequence` in 32,085 ms server time / 32,099
ms loopback client wall time. The output object matched the separately pinned
private suffix-stopped reference exactly. Its checked field types were string
for `intent` and `complexity`, array for `suggested_models` and `reasons`, and
integer for `agent_count`; the model list was empty and the reasons array had
13 entries. This checks artifact loading, stop-suffix handling, configured
schema validation, and serving the embedded UI. It does not evaluate
recommendation quality.

The test binary needed the external service JSON, framing JSON, training
manifest, runtime JSON, session token in the environment, and a private state
directory. The request arrived over loopback. It needed no graph directory or
UI directory at runtime and made no external network request. A network
namespace was unavailable on the test host, so this was not a hard network
isolation test. It was a Linux release build; Windows packaging and the
owner-facing Windows executable remain unverified.

## Separate ONNX prefill and decode graphs

`ml/lfg-encoder/export_owned_cached_onnx.py` exports a versioned experimental
artifact with separate `prefill.onnx` and `decode.onnx` graphs. The prefill
graph accepts `tokens` as `int64 [batch, sequence]` and returns the last-token
`float32 [batch, 256]` logits plus cache tensors. The decode graph accepts one
`int64 [batch]` token, a rank-zero `int64` absolute `position`, and those cache
tensors; it returns next-token logits and the updated caches. Convolution state
is named `conv_{layer}`; attention key and value state are `key_{layer}` and
`value_{layer}`. Decode outputs add `_out` to each state name.

The `manifest.json` declares the state order, dtype, shape, per-layer capacity,
empty attention-state shape, and the maximum cache tensor bytes per batch. The
consumer must validate byte token values, batch/rank/dtype/fixed dimensions,
prefill length, `position < context`, and that each attention cache length
equals `position`. The ONNX graphs do not add a serving wrapper that checks
those relationships. The model uses absolute positions: the caller increments
the absolute position, retains all attention history, and rejects context
overflow. There is no cache eviction. The cache byte figure excludes model
weights, allocator overhead, temporary graph tensors, and runtime buffers; it
is not a process-memory cap.

Export with a verified private checkpoint and already-framed local source
bytes. Keep the export configuration, sources, measurements, ONNX files, and
checkpoint outside Git:

New cached exports currently require a POSIX system and an existing private
parent directory owned by the current user. The exporter rejects symbolic-link
ancestors and unsafe writable ancestors, then creates a new mode-0700 output
directory. Windows cached export is disabled until a verified current-user ACL
creation and verification path is available; Python mode bits do not establish
Windows ACLs. Existing Windows exports are not changed and can still be used by
the native runtime. Run new exports on a trusted Linux host when available.

Every configured parity length and development input must be shorter than the
model context so at least one cached decode step can be checked. Full-context
prefill-only export measurements are not supported; such a request is rejected
before creating the output directory.

```powershell
python ml/lfg-encoder/export_owned_cached_onnx.py `
  --checkpoint PRIVATE_CHECKPOINT_DIRECTORY `
  --config PRIVATE_EXPORT_CONFIG.json `
  --source PRIVATE_FRAMED_SOURCE.bin `
  --additional-source ANOTHER_PRIVATE_FRAMED_SOURCE.bin `
  --output NEW_PRIVATE_OUTPUT_DIRECTORY `
  --decode-steps 4 --warmup 0 --repeats 3
```

The config loader accepts at most 1 MiB; model, checkpoint, and manifest sizes
use their explicit configuration limits. Export verifies checkpoint weights
and source model code against the checkpoint manifest, checks that all inputs
remain unchanged during export, checks ONNX files, and compares configured
prefill lengths and each measured decode step with both full-prefix PyTorch and
the Python cache reference. It records artifact/input hashes only in the
private report and emits no prompt or generated text. `promotion_eligible`,
`native_runtime_checked`, and quality evaluation are false in the manifest.

Cached export uses its own `owned-byte-cached-onnx-export.example.json` config.
The optional `graph_optimization_level` accepts ONNX Runtime's symbolic levels
`ORT_DISABLE_ALL`, `ORT_ENABLE_BASIC`, `ORT_ENABLE_EXTENDED`, and
`ORT_ENABLE_ALL`. When omitted, the exporter leaves `SessionOptions` at the
installed ONNX Runtime library default and records the request as
`unspecified`; it separately records the value read back from each prefill and
decode session, or `unknown` if that API is unavailable. The manifest and
measurement also record the ORT version, providers, and thread settings, and
the existing config hash binds the request. Unknown extension fields remain
inert. Invalid values of this recognized option fail before the private output
directory is created.

The optional boolean `failure_evidence` config field defaults to `false` and
does not change strict acceptance. When enabled, the first failed
`numpy.testing.assert_allclose` writes one bounded `failure-report.json` into
the already verified private export directory. It records only fixed phase and
comparison labels, source ordinal, prefix/decode position, a generated cache
state descriptor when applicable, shapes/dtypes, and safe aggregate counts and
error bounds. The original-dtype NumPy comparison, configured tolerances,
call order, equal-NaN behavior, and scalar rules remain the pass/fail check;
summary calculations never decide acceptance. Shape mismatches or unsupported
aggregates are marked unavailable. The report omits paths, source identifiers,
input/output values, coordinates, exception text, and tracebacks. If creation
fails, the strict export remains failed and the sanitized CLI result reports
`failure_report_written: false` with a fixed error category. A numerical
failure never writes the success manifest or measurement; partial ONNX graphs
must be treated as unverified artifacts, with promotion ineligible. With the
field omitted or `false`, no sidecar is written.

Report publication uses the output directory identity captured when the
private directory is created, opens that same directory without following the
final path component, and exclusively creates, writes, and syncs the sidecar
relative to its verified descriptor. The writer never removes a report by
filename after creation; if its path or entry changes concurrently, it reports
the uncertainty and preserves the competing entry. CLI startup import and
argument failures are returned as fixed categories without traceback or
argument echoes. A write interruption may leave an incomplete sidecar; a
`failure_report_written: false` result means its contents were not verified as
complete.

When invoked as the exporter script, Python output and ordinary process file
descriptors 1 and 2 stay redirected to the system null device for the full
process lifetime, including native C/C++ stream flushes during shutdown. The
fixed JSON result is written through saved status descriptors, so it remains
available without restoring descriptors that native buffers could later use.
This boundary applies only to the standalone script; importing the module or
calling its library API does not redirect the host application's streams.
Output emitted before the script starts (including interpreter startup hooks),
custom descriptors or files, OS crashes, and malicious native libraries are
outside this boundary.

The standalone `--help` option writes fixed public usage text through the saved
stdout channel; its program label does not expose the invoked script path.
Status delivery is checked: success and help return exit code 7 if the complete
status cannot be written. When the preferred status channel is unavailable
before writing begins, the CLI tries the other saved channel; a failed or
partial write is not replayed on the other channel. Export failures keep their
original nonzero exit status even if their fixed error status cannot be sent.

The sidecar describes only a failed `assert_allclose` comparison. Earlier
shape or finiteness guards and the separate greedy-byte `ValueError` retain
their existing rejection behavior and do not claim that a numeric failure
report exists.

This ORT setting controls only the exporter's ORT validation and benchmark
sessions and the matching ORT parity diagnostic. It does not configure the
native tract runtime or prove which individual graph rewrites occurred. The
native cached runtime continues to use tract and needs separate backend-specific
parity and performance evidence.

A later 10-work-item CPU prefill-isolation diagnostic on two screening-claimed
training sources observed one cache-value failure in 7,020,544 comparisons for
current-default ORT versus PyTorch incremental, and zero in 7,020,544 for
`ORT_DISABLE_ALL` in that comparison. The diagnostic compared prefixes only;
it did not rerun the strict cached exporter or its autoregressive decode
acceptance. It therefore does not show that disabling optimization fixes strict
cached parity, and it says nothing about model quality. The original strict
cached export remains failed pending an independently grounded rerun.

The legacy-policy regression uses a base64-stored copy of the pre-configuration
cached exporter at `ml/lfg-encoder/fixtures/legacy_owned_cached_exporter.b64`.
The decoded source SHA-256 is
`e8c3ef56f26abaadeef341f7f68201754b4aeb42561b2f64641d25cd16d2ea2f`; base64
storage keeps that source pin stable across platform line-ending conversions.
The test creates only synthetic model inputs and artifacts and does not depend
on repository history or claim model qualification.

One private local experiment used the owner-authorized 573-byte and 603-byte
framed development inputs with the actual private checkpoint (context 4,352,
width 512, eight layers). ONNX Runtime 1.31.1 with one CPU thread checked
prefill lengths 1, 4, 32, 64, and 573, plus four greedy decode steps for each
input, against both references. Later greedy-byte choices matched, but the original
check did not independently compare the first choice; maximum absolute logit and
cache differences were 1.82e-5 and 1.05e-5 respectively. Median
prefill times were 388.6 ms and 367.0 ms. Median four-byte decode batches were
34.4 ms and 30.8 ms (8.60 ms and 7.70 ms per byte). This run used no warmup and
three repeats. It did not measure a full-prefix timing baseline, evaluate
assessment quality, or establish a reliable speedup. These measurements are a
single local CPU case and must not be generalized to other hardware or loads.
For this checkpoint and full context, the declared cache tensor upper bound is
71,335,936 bytes per batch item. It does not include other runtime memory. The
raw private experiment report and inputs remain outside Git.

The exporter now checks every generated byte, including the first, against an
independently generated full-prefix greedy sequence. The earlier exporter
compared prefill logits within tolerance but did not independently check their
argmax, so its `greedy_bytes_exact` field alone did not establish first-byte
parity. The separate native tract experiment remains separate evidence and is
not a result of this exporter check.

On 2026-11-03, a real-checkpoint CPU export on the owner's Spark using PyTorch
2.11.0+cu130, ONNX 1.23.2 and ONNX Runtime 1.31.1 **failed** its original
absolute and relative tolerances of 1e-5: one of 1,024 cache-state values
failed, with a maximum absolute difference of 1.552701e-5 and relative
difference of 3.3856897e-5 among the reported violations. A separate diagnostic
retaining the original relative tolerance and using an absolute tolerance of
2e-5 completed on two authorized development inputs of 573 and 603 bytes. Four
greedy bytes for each matched the independent full-prefix reference; maximum
observed logit and cache differences were 1.811982e-5 and 1.621247e-5. The
diagnostic used one measured repeat and no warmup, and its output directory was
owned by the current user with mode 0700. It does **not** pass the original
tolerance policy, establish why the environments differ, qualify runtime
performance, or evaluate assessment accuracy. Both runs and their private
reports were kept separate; neither modified the starting checkpoint.

### Read-only parity diagnostics

`ml/lfg-encoder/diagnose_owned_cached_parity.py` measures existing cached ONNX
graphs without exporting, modifying, promoting, or certifying them. Its v2
private policy pins the checkpoint manifest and weights, model/cache/exporter
code, export config, graph bytes, and a train-source selection manifest. Before
inference it checks the exporter graph input/output names, types, order, ranks,
static dimensions, dynamic cache-position axes, and cache-state mapping. It
loads the verified weights and graph bytes from the already-hashed memory
buffers, then checks every source path again afterward and records mutations.
Pinned `owned_student` and `owned_cached_decode` modules are registered from
those captured bytes before loading the pinned exporter, then prior
`sys.modules` bindings are restored. The export config must retain the original
`atol=1e-5` and `rtol=1e-5`.

The selection manifest maps opaque source-ID digests to source digests and
screening-manifest digests. The screening records must match those identifiers
and hashes, but screening status remains an operator claim until independently
verified. Free-text metadata and unknown extension fields are ignored. Reports
contain allowlisted hashes and aggregate measurements; they omit paths, inputs,
logits, cache values, token choices, output text, and exception messages.

The report keeps teacher-forced parity on the full-reference stream separate
from independent greedy PyTorch and ONNX trajectories. It reports trajectory
agreement at each decision, first divergence, and each trajectory's same-history
comparison against its full-prefix model result. Per-tensor statistics use
NumPy's original-dtype `assert_allclose` comparison (`equal_nan=True`) for the
failure count, while nonfinite data receives a separate status. Absolute and
nonzero-reference relative errors, denominator, invalid shapes/dtypes, empty
tensors, crashes, and not-tested steps are identified independently. Descriptive
differences are calculated in float64 after original-dtype comparison; an
unrepresentable statistic is marked explicitly and serialized as JSON `null`.
Reports use strict JSON and reject NaN or Infinity. Invalid or nonfinite logits
stop the affected trajectories before any next-token argmax; prior safe
teacher-forced measurements remain recorded and later steps are marked not
tested. Decision rows cover the prefill decision at index zero and every
requested decode decision; decisions unavailable because of an invalid prefix,
timeout, or earlier failure are explicitly marked not tested. Teacher-forced
rows retain tensor statistics for an attempted invalid or nonfinite step, with
the row status set to `invalid` or `nonfinite`. The attempted counter includes
such rows; the completed counter includes only valid measured transitions, and
the combined decode completion count is the minimum of valid teacher-forced
and free-trajectory transitions. The free-trajectory attempted counter counts
decode calls launched; its completed counter advances only after all decision
comparisons for that step are valid. A failed free decision retains its row and
does not count as a completed transition.

The sample/work/prefix limits are enforced by the runner. `--max-seconds` is a
cooperative in-process timer: it checks between operations and cannot interrupt
a single PyTorch or ONNX call. Reports say explicitly that no hard wall-time,
CPU or memory limit was enforced. Any future real-input run needs separate
verified host supervision and resource/coexistence checks in a fresh handoff.
The report's external-prerequisite fields describe these limitations; they do
not enforce a real-input prohibition, verify owner authorization, or validate
screening claims. A real-input operator must establish those conditions before
starting a run.
Elapsed and CPU time include preflight, model/session setup and measured work;
peak RSS is unavailable. A private atomic partial report is updated during
execution. Incomplete, context-truncated, crashed, or input-mutated runs cannot
return `diagnostic_complete`; no safe report may be available if private output
setup itself fails. A complete diagnostic means all requested measurements ran,
not that numerical parity passed or model quality was established.

Output requires a new directory and the exporter's verified private POSIX
directory checks. Windows output remains disabled until current-user ACL
verification is implemented. Example invocation:

```powershell
python ml/lfg-encoder/diagnose_owned_cached_parity.py `
  --checkpoint PRIVATE_CHECKPOINT --export-config PRIVATE_EXPORT_CONFIG.json `
  --graphs PRIVATE_EXISTING_GRAPH_DIRECTORY --policy PRIVATE_PIN_POLICY.json `
  --selection-manifest PRIVATE_TRAIN_SELECTION.json `
  --source SOURCE_ID_SHA256=PRIVATE_SOURCE.bin `
  --screening-manifest SOURCE_ID_SHA256=PRIVATE_SCREENING.json `
  --output NEW_PRIVATE_DIAGNOSTIC_DIRECTORY --length 64 --length 573 `
  --decode-steps 4 --threads 1 --max-bytes 1000000000 `
  --max-work-items 8 --max-prefix-tokens 100000 --max-seconds 600
```

Tests exercise synthetic float32 tolerance boundaries and the public runner
against a tiny generated fixture on a compatible POSIX CPU environment. Fixture
results establish only that test path; they are not real-input evidence. No
private example, validation/test data, deployment, or GPU work is part of this
diagnostic implementation.

### Prefill parity isolation

`ml/lfg-encoder/diagnose_owned_cached_prefill_isolation.py` separates three
read-only CPU paths for the same pinned checkpoint and prefix: the retained
ONNX prefill graph, the pinned PyTorch `PrefillGraph`, and the incremental
PyTorch `DecodeSession`. For logits and every cache tensor, it records the
three pairwise comparisons independently. It preserves the exporter's original
`atol=1e-5` and `rtol=1e-5`; a strict-tolerance mismatch is a measured result,
not an incomplete run or a reason to change acceptance.

Two independent ONNX Runtime sessions are created for each run: the explicit
current optimization setting (`ORT_ENABLE_ALL`) and `ORT_DISABLE_ALL`. The
selected intra-op thread count is applied to both sessions and PyTorch; the
inter-op count is explicitly one. The report records the optimization level,
provider, thread settings, and Python/NumPy/PyTorch/ONNX/ONNX Runtime/safetensors
versions. The prefix-work bound counts the PyTorch full and incremental paths
and each requested ORT variant. A different result between
these sessions is an experiment, not proof that graph optimization caused a
parity difference.

When the configured intermediate-capture budget permits, bounded PyTorch
module hooks compare normalized block inputs/residual boundaries, input-gate
content, convolution and attention branches, and feed-forward operations at
matching sequence positions. The report gives aggregate differences and the
earliest nonzero-difference and earliest strict-tolerance-failing
layer/operation. Intermediate values are never written. The retained ONNX graph exposes no equivalent internal tensors, so the
tool does not claim an ONNX layer trace; it does not rewrite or instrument the
graph. If the activation capture estimate exceeds `--max-intermediate-bytes`,
cache/logit comparisons still run and the layer trace is marked `not_tested`.
This byte estimate covers anticipated captured tensor payloads only; it is not a
hard process-memory bound. The parent supervisor must independently enforce
memory and other host resource limits.

The runner reuses the v2 pinned policy, selection, screening-claim, graph,
checkpoint, model, cache-reference, and exporter validation. It enforces the
configured work-item, prefix-token, file-byte, intermediate-capture, thread,
and cooperative-time bounds and writes only a bounded strict-JSON aggregate
report. The cooperative timer cannot interrupt a single library call; a real
input run still requires a fresh handoff and independent OS-level CPU, memory,
PID, wall-time, privacy, authorization, and coexistence checks. The runner
does not itself verify those host controls. Screening and data quality remain
operator claims unless independently verified.

The shared diagnostic core is loaded by reading, hashing against its reviewed
raw SHA-256, and executing those exact captured bytes in a private module
object; it is not resolved by import name. A launcher must provide separate,
explicit reviewed raw-byte SHA-256 values for both the tool and shared core.
The accepted digest depends on the exact staged bytes: Git LF and Windows CRLF
forms each require their own separately reviewed digest. The tool does not
normalize line endings or derive trust from a digest it computes itself. Before
CLI use, an external bootstrap verifies both captured files and executes the
captured tool bytes. Set `EXPECTED_TOOL` and `EXPECTED_CORE` from the reviewed
handoff:

```sh
python - "$PWD/ml/lfg-encoder/diagnose_owned_cached_prefill_isolation.py" "$EXPECTED_TOOL" "$PWD/ml/lfg-encoder/diagnose_owned_cached_parity.py" "$EXPECTED_CORE" --checkpoint PRIVATE_CHECKPOINT ... <<'PY'
import hashlib, os, sys
from pathlib import Path
tool, tool_pin, core, core_pin = map(str, sys.argv[1:5])
tool_path, core_path = Path(tool).resolve(), Path(core).resolve()
tool_raw, core_raw = tool_path.read_bytes(), core_path.read_bytes()
if hashlib.sha256(tool_raw).hexdigest() != tool_pin:
    raise SystemExit("reviewed tool digest mismatch")
if hashlib.sha256(core_raw).hexdigest() != core_pin:
    raise SystemExit("reviewed core digest mismatch")
os.environ["LFG_PREFILL_ISOLATION_EXPECTED_SHA256"] = tool_pin
os.environ["LFG_PREFILL_CORE_EXPECTED_SHA256"] = core_pin
sys.argv = [str(tool_path), *sys.argv[5:]]
exec(compile(tool_raw, str(tool_path), "exec"), {"__name__": "__main__", "__file__": str(tool_path)})
PY
```

The report records tool/core digests and whether the supplied launcher pin
matches. The bootstrap is an external trust prerequisite; a report field alone
does not prove process origin. Work-item counts are disjoint: measured work is
completed (including valid tolerance failures), invalid inputs/comparisons,
incomplete work, crashes, in-progress work, and never-started `not_tested`
work sum to requested work. A skipped optional layer trace is separately
marked `not_tested` and is not counted as a sample work item.

Run synthetic tests only with a compatible POSIX CPU environment; no private
inference is implied by these tests:

```powershell
python -m unittest discover -s ml/lfg-encoder -p test_diagnose_owned_cached_prefill_isolation.py
```

Run the CLI only through the dual-pin bootstrap above, with the normal options
below; direct `python diagnose_owned_cached_prefill_isolation.py ...` execution
does not satisfy the trust prerequisite. The CLI requires both optimization
variants:

```text
--checkpoint PRIVATE_CHECKPOINT --export-config PRIVATE_EXPORT_CONFIG.json
--graphs PRIVATE_EXISTING_GRAPH_DIRECTORY --policy PRIVATE_PIN_POLICY.json
--selection-manifest PRIVATE_TRAIN_SELECTION.json
--source SOURCE_ID_SHA256=PRIVATE_SOURCE.bin
--screening-manifest SOURCE_ID_SHA256=PRIVATE_SCREENING.json
--output NEW_PRIVATE_ISOLATION_DIRECTORY --length 1 --length 64
--threads 1 --max-bytes 1000000000 --max-work-items 8
--max-prefix-tokens 100000 --max-intermediate-bytes 268435456
--max-seconds 600 --optimization-variant current_default
--optimization-variant disable_all
```

This diagnostic reports execution-path differences only. Greedy agreement,
synthetic fixtures, or a passing optimization variant does not establish
strict export parity, task quality, performance qualification, or release
readiness. The original exporter failure remains failed.

Run the local benchmark against an existing private checkpoint and an
authorized local source prefix:

```powershell
python ml/lfg-encoder/benchmark_owned_cached_decode.py `
  --checkpoint PRIVATE_CHECKPOINT_DIRECTORY --source PRIVATE_SOURCE_FILE `
  --length 64 --threads 1 --warmup 3 --repeats 20
```

The values shown are an example, not a runtime policy. The script verifies the
weights and model implementation against the checkpoint manifest, compares
cached logits with full-prefix logits, alternates the measurement order, and
reports median and nearest-rank p95 elapsed time for processing **all** prefixes
of the selected sequence. It records implementation, checkpoint and input-prefix
digests. These digests and learned weights can be sensitive; keep the report and
checkpoint local. This is a microbenchmark on a single request, not end-to-end
latency or throughput under concurrent load.

On 2026-11-02, a local Windows CPU run used the prior private 37,888-parameter
multi-document distilled checkpoint and the first 64 bytes of the owner's LFG
README. With PyTorch 2.14.1+cpu, one thread, three warmups and 20 alternating
repeats, the largest cached-versus-full-prefix logit difference was 0.00000191.
Full-prefix median/p95 elapsed time for the 64 positions was 46.23/59.19 ms;
cached median/p95 was 39.83/47.10 ms. The machine was not isolated, the sample
is small, and these results are neither a statistically established speedup nor
an assessment of the target-sized model. The source is already part of the
development material, so this is not an independent quality evaluation.
