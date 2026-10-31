# Experimental native exemplar suggestions

The optional `onnx-shadow` build can now use an explicitly selected
[per-node exemplar pack](NODE-EXEMPLAR-EXTENSION.md) with the
[native frozen encoder](NODE-ENCODER-NATIVE-PROBE.md). The command is an
offline, unpromoted experiment; it requires `--experimental` and does not
replace the LFG kernel answer, continue the tree, route work, or promote rules.
It does not prove that an attestation, model or artifact is authentic.

```powershell
cargo build --release --locked -p lfg-server --features "embedded-ui,onnx-shadow"
lfg-server onnx-node-suggest --experimental `
  --artifact ENCODER_ARTIFACT_DIR --manifest-sha256 TRUSTED_ENCODER_MANIFEST_SHA256 `
  --runtime-config NODE_RUNTIME_CONFIG.json `
  --exemplar-pack PRIVATE_PACK.json --pack-sha256 TRUSTED_PACK_SHA256 `
  --exemplar-config EXEMPLAR_CONFIG.json --snapshot SNAPSHOT.json
```

It reads `id` and `text` JSONL records on stdin. It first runs the deterministic
LFG kernel. Only when the kernel escalates at the pack's exact choice node does
it load the encoder, calculate an embedding, and compare it with that node's
exemplars. The encoder is initialized on the first eligible request and reused
for later eligible requests in the same process. The pack and small manifest are
read at startup. A resolved request receives `not_applicable`; an eligible
request receives either `suggested` or `abstained`. The response preserves the
baseline and marks the result `promotionEligible: false`. Similarity and margin
are uncalibrated scores, not probabilities. Kernel extracts in a detailed
response can include sensitive input text. Model-load or inference errors fail
the command; they do not silently produce a suggestion. As with other streaming
commands, earlier outputs can already have been emitted when a later item fails.

The command verifies the selected pack hash, snapshot, node, choices, exemplar
counts and unit vectors; checks that its encoder and source-model identities
match the selected ONNX manifest; and requires an admission report indicating
that the configured review gates passed. It checks the form of recorded review
and environment provenance but **does not re-run the review audit or authenticate
the pack**. A hash identifies bytes only when its expected value was obtained
from a trusted source. Anyone able to forge a pack and supply its hash can forge
its provenance. Keep packs, source data and expected digests access-controlled.

With the `onnx-shadow` feature, Rust callers can use
`lfg_server::exemplar_suggest::NodeSuggester` in-process. Construct `Options`
with the selected artifact, hashes and configuration files, provide a borrowed
`lfg_core::kernel::Engine` and an `OperationsConfig` to `open`, then call
`suggest` with request text. The instance stays bound to that engine and loads
the encoder lazily on the first eligible escalation. Its response retains the
kernel result, extension status, optional suggestion, input and artifact hashes,
and `promotionEligible: false`. The command uses this same path. This is an
experimental library seam in the server crate, not yet an RRFlow integration,
the final engine packaging, or an authenticated artifact distribution system.

Similarity and margin thresholds come from the selected exemplar configuration
and can change without rebuilding vectors. Encoder settings that affect vectors
must still match the pack and ONNX artifact. Limits in the selected runtime and
exemplar configurations apply; the experiment has not been benchmarked or
qualified for high-throughput use. The model, tokenizer and pack are external
files. The current binary does not provide an update service, embedding of this
node model, hosted authentication, or RRFlow/SWAG/MCP integration.

`--prediction-only` emits `lfg.node-prediction.v1` records for the existing
validation format. An ID used with the validation runner must be the matching
review item digest. For a stronger local check,
`ml/lfg-encoder/evaluate_native_exemplars.py` reruns the review audit, verifies
that the pack's recorded admission and training item identities match the
current audited inputs, then launches the selected binary for both the baseline
and native candidate on validation records. For example:

```powershell
python ml/lfg-encoder/evaluate_native_exemplars.py `
  --input PRIVATE_QUEUE.jsonl PRIVATE_LEDGER.json --snapshot SNAPSHOT.json `
  --policy ADMISSION_POLICY.json --node NODE_ID --binary LFG_BINARY `
  --artifact ENCODER_ARTIFACT_DIR --manifest-sha256 TRUSTED_ENCODER_MANIFEST_SHA256 `
  --runtime-config NODE_RUNTIME_CONFIG.json --pack PRIVATE_PACK.json `
  --pack-sha256 TRUSTED_PACK_SHA256 --exemplar-config EXEMPLAR_CONFIG.json `
  --max-items 1000 --max-input-bytes 10485760 --timeout-seconds 180
```

Repeat `--input` for all relevant queue/ledger pairs. The runner checks file
hashes and outputs aggregate per-node metrics without request text or vectors.
It observes that the selected process ran; it does not authenticate the binary,
reviewer, source rights or data independence. The current admission audit reads
test records to check support, so this is not a blinded final evaluation.
Performance, calibration and an independent holdout still require separate
qualification.

## Actual state and checks

On 2026-10-31, the real LFG queue still had 11 pending items and zero approved
labels. The native validation runner refused to evaluate it and produced no
quality metrics. No real exemplar pack has been built.

After the encoder refactor, the Windows native debug binary again matched the
frozen Python encoder on all 11 unreviewed local requests within the configured
`1e-4` tolerances, with maximum absolute difference approximately `2.91e-7`.
No request-derived vectors were saved. This checks numerical parity only.

For integration testing only, a pack was produced with the actual frozen encoder
and **simulated** review and environment evidence for two test-only inputs. The
Windows native command accepted that Python-built pack, matched the Python
similarity choice on an escalation, skipped the model for a resolved request,
and abstained when the configured margin was tightened. The same test-only pack
and encoder artifact exercised the native validation path on Windows and Spark;
no private LFG requests were transferred to Spark. These checks establish wiring
for the tested cases, not reviewed training, general parity, model accuracy or a
qualified release.

The optimized Windows build also passed the test-only native validation path.
It was 32,863,744 bytes with SHA-256
`9b09b8ed80834be369e4c4f9c39a5c1b571443f04e1bcabe4f42a7416a28d88a`;
the encoder and test pack remained external files. This was a functional check,
not a clean-machine installation, throughput benchmark, or release qualification.
