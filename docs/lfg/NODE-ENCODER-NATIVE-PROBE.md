# Experimental native node encoder probe

The optional `onnx-shadow` feature now includes an offline
`onnx-node-embedding` command. It loads an explicitly selected frozen encoder
and tokenizer from an external artifact and emits metadata for each request.
It can expose vectors only when `--include-embedding` is explicitly requested.
It does not classify an LFG kernel node, read an exemplar pack, route work,
promote an answer, or establish model quality. The encoder files are not yet
embedded in the executable; this is not a self-contained release.

## Export and parity

`ml/lfg-encoder/export_node_encoder.py` exports the frozen encoder and its
attention-mask mean pooling and L2 normalization to ONNX. It checks ONNX CPU
output against the selected Python encoder on each item in an explicitly
selected local queue. Pending requests are allowed **only for this functional
parity probe**: no labels are used, and the exported graph is traced using an
unknown-token input rather than a project request. It saves no request-derived
vectors. The selected model files and manifest are checked before and after;
the export environment must also be independently prepared and recorded.

```powershell
VENV/Scripts/python.exe ml/lfg-encoder/export_node_encoder.py `
  --queue PRIVATE_QUEUE.jsonl --ledger PRIVATE_LEDGER.json --snapshot SNAPSHOT.json `
  --model LOCAL_MODEL_DIR --model-manifest MODEL_MANIFEST.json --revision MODEL_REVISION `
  --config EXEMPLAR_CONFIG.json --export-config ONNX_EXPORT_CONFIG.json `
  --max-parity-items 100 --output NEW_PRIVATE_ARTIFACT_DIR
```

The new directory contains `model.onnx`, `tokenizer.json` and `manifest.json`.
An interruption can leave a partial directory; inspect it and use a new path.
The manifest binds file digests, model identity and the encoder settings that
affect vectors. Similarity thresholds are separate, so changing them does not
require re-exporting the encoder. The manifest is not a signature; acquire its
expected digest through a trusted channel. Treat the selected model, tokenizer,
binary and local filesystem as trusted inputs. Artifacts and text-derived
hashes can be sensitive.

Build and run the native probe with explicit limits and the independently
selected manifest digest:

```powershell
cargo build --release --locked -p lfg-server --features "embedded-ui,onnx-shadow"
lfg-server onnx-node-embedding --artifact ARTIFACT_DIR `
  --manifest-sha256 EXPECTED_SHA256 --runtime-config NODE_RUNTIME_CONFIG.json
```

The command accepts JSONL `id` and `text` objects on stdin and prints one
metadata object per input. Its configurable runtime limits are illustrated in
[`node-onnx-runtime.example.json`](../../crates/lfg-server/config/node-onnx-runtime.example.json).
It rejects duplicate IDs, oversized input, tokenizer truncation or padding,
unsupported graph inputs, mismatched hashes, invalid dimensions and non-finite
or non-normalized vectors. A batch can emit earlier results before a later input
fails, so check the exit status. `--include-embedding` returns the derived
vectors and must be handled as sensitive local data. The configured single
thread is a supported setting, not proof that every dependency or the operating
system creates only one thread. The model and graph are loaded in memory; no
startup, memory or throughput qualification has been performed.

For the same queue, `compare_node_encoder.py` runs the native command and the
selected Python encoder, compares vectors in memory, checks their identities,
and prints aggregate parity evidence without saving the vectors. It requires
the same model, snapshot, configuration, export configuration and artifact, plus
`--binary`, `--runtime-config`, `--manifest-sha256`, `--max-items`,
`--max-input-bytes` and `--timeout-seconds`. It does not authenticate the
artifact or establish classification accuracy. The process captures native
output in memory; configured input bounds are not a complete memory sandbox.
Both the export parity check and native probe currently run one request per
model invocation; they do not validate batched native inference.

```powershell
VENV/Scripts/python.exe ml/lfg-encoder/compare_node_encoder.py `
  --queue PRIVATE_QUEUE.jsonl --ledger PRIVATE_LEDGER.json --snapshot SNAPSHOT.json `
  --model LOCAL_MODEL_DIR --model-manifest MODEL_MANIFEST.json --revision MODEL_REVISION `
  --config EXEMPLAR_CONFIG.json --export-config ONNX_EXPORT_CONFIG.json `
  --artifact ARTIFACT_DIR --manifest-sha256 EXPECTED_SHA256 `
  --binary LFG_BINARY --runtime-config NODE_RUNTIME_CONFIG.json `
  --max-items 100 --max-input-bytes 10485760 --timeout-seconds 180
```

## Observed Windows check

On 2026-10-31, a local MiniLM snapshot was exported using an existing Python
environment that was **not** the locked release environment. ONNX Runtime and
Python agreed on all 11 previously inspected, unreviewed local LFG requests;
the maximum absolute vector difference was approximately `1.61e-7`. The Rust
native debug binary and Python also agreed on all 11 within the selected
absolute and relative tolerances of `1e-4`; the largest absolute difference was
approximately `2.91e-7`. The inputs ranged from 17 to 145 tokens and each
vector had 384 dimensions. No labels or request-derived vectors were saved.
The artifact manifest SHA-256 was
`b618f7f870ff2e98c417b061e9edef991aeebe2896ae796aa83cb028a42e7005`;
the ONNX model SHA-256 was
`d76931eb2f7abc05d5d4ac5b68386491bd9350493f20b7943182275f248b2838`.
The legacy ONNX export emitted a tracer warning. A separate test-only boundary
check accepted 256 tokens and rejected an over-limit request. These checks do
not prove every shape, cross-machine reproducibility, model quality, calibration,
or latency. The full real review ledger still has zero approved labels.

The optimized Windows build with `embedded-ui,onnx-shadow` also matched Python
on those 11 requests with the same maximum absolute difference. Its executable
was 32,645,120 bytes and SHA-256
`2fd1294c2f80c7f8fe72af9e98b7fc11ece9262c8408ff5f51122a1deec1135d`;
it still loaded the model and tokenizer from the external artifact. Two local
exports from the same model and environment produced identical model, tokenizer
and manifest hashes; the exporter was revised between those runs to stream its
file hash. This is limited same-machine evidence, not an independent clean build.

The same frozen ONNX artifact and tokenizer were copied to an isolated Spark
build; the graph was traced from an unknown token and was not trained on the
private requests. The Spark native debug binary matched the Windows native
vector for the owner's test-only golden request within `1e-4` absolute and
relative tolerances; the maximum absolute difference was approximately
`1.14e-7`. Both native unit tests passed there. No private LFG requests were
transferred. This is one cross-architecture fixture, not cross-machine
reproducibility or quality qualification.
