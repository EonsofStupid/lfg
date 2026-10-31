# Experimental native CPU inference

The optional `onnx-shadow` build feature adds an offline `onnx-predict` command
to the LFG binary. It uses a native Rust ONNX engine and tokenizer; the normal
server and rule encoder remain authoritative. It neither promotes a model nor
routes user work. The feature and its dependencies are omitted from the normal
build unless selected explicitly.

Build for the intended target and include the already-built UI if desired:

```powershell
cargo build --release --locked -p lfg-server --features "embedded-ui,onnx-shadow"
```

Export to a new directory using a local source model and a separately selected
parity dataset. The exporter compares its ONNX output with that source model;
parity does not establish ground-truth accuracy.

```powershell
python ml/lfg-encoder/export_onnx.py --model SOURCE_MODEL_DIR --parity-dataset PRIVATE_JSONL --taxonomy ml/lfg-encoder/evaluation-taxonomy.v1.json --prediction-config ml/lfg-encoder/prediction-config.v1.json --export-config ml/lfg-encoder/onnx-export-config.v1.json --output NEW_ARTIFACT_DIR
```

Run the command with an operator-selected runtime config and an independently
supplied digest of the artifact's `manifest.json`:

```powershell
lfg-server onnx-predict --artifact ARTIFACT_DIR --manifest-sha256 EXPECTED_SHA256 --runtime-config RUNTIME_CONFIG_JSON
```

The command reads JSONL objects with `id` and `text` on stdin and emits
metadata-only `lfg.evaluation-prediction.v1` objects on stdout, compatible with
the existing evaluator. It rejects duplicate IDs and inputs or example counts
outside configured limits. The example runtime configuration is
`crates/lfg-server/config/onnx-runtime.example.json`; native inference currently
supports a single CPU thread and rejects other requested thread counts.

The v2 export manifest records token length and tokenizer directions in addition
to file hashes. The runtime verifies the pinned manifest and the model,
tokenizer, labels, taxonomy and prediction configuration before use. It accepts
only the supported tokenizer and input contract and loads the model from bytes,
so external tensor files are not resolved. A digest detects changes only when
the expected digest came from a trusted source; it does not sign or authenticate
an artifact. Treat model files as trusted local input: this path does not provide
an operating-system sandbox or a memory/CPU isolation guarantee.

For a repeatable comparison with the source Python model on a local JSONL
dataset, use the separate comparison tool. It checks that the ONNX manifest
identifies the supplied source artifact, runs both paths offline, compares
label decisions and input digests, and writes only aggregate metadata to a
new private report. It does not establish ground-truth accuracy:

```powershell
python ml/lfg-encoder/compare_onnx_shadow.py --binary LFG_BINARY --artifact ONNX_DIR --model SOURCE_MODEL_DIR --dataset PRIVATE_JSONL --runtime-config RUNTIME_CONFIG_JSON --max-examples MAX --timeout-seconds SECONDS --output NEW_PRIVATE_REPORT.json
```

An opt-in `embedded-onnx` feature can instead include an explicitly selected
artifact in the executable. The separate external-artifact command remains
available in the same build. No production-trained artifact, qualified model,
automated update channel, live inference routing, calibration, or final
independent evaluation is provided by either command. Older v1 exports must be
re-exported; they do not record the input contract required by this runtime.

## Optional model embedded in the binary

Select an artifact and its independently obtained manifest SHA-256 at build
time. The source tree does not contain the weights or choose a model for the
user. For example, after building the UI assets:

```powershell
$env:LFG_EMBEDDED_ONNX_DIR = (Resolve-Path ARTIFACT_DIRECTORY).Path
$env:LFG_EMBEDDED_ONNX_SHA256 = 'EXPECTED_LOWERCASE_MANIFEST_SHA256'
cargo build --release --locked -p lfg-server --features 'embedded-ui,embedded-onnx'
```

The build requires a matching manifest digest and regular artifact files; at
runtime the command rechecks that digest and the required model, tokenizer,
label, taxonomy and prediction-configuration digests. The digest must come
from a trusted channel; it is not a signature. Run the resulting executable
from any working directory with `embedded-onnx-predict`, feeding the same JSONL
input as `onnx-predict`. It uses the bundled example runtime-limits file by
default; `--runtime-config PATH` selects another one. Runtime limits are not a
model quality policy, and this offline command cannot promote or route work.
The `onnx-predict` command can still use a separately selected external artifact
and runtime config. Replacing the embedded model requires rebuilding the binary;
there is no update service, signing, rollout, or rollback mechanism yet.

On 2026-10-31, a local Windows x86-64 MSVC release was built with embedded UI,
the `+crt-static` and `/Brepro` flags, and the real-request experimental model
from [the development experiment](REAL-REQUEST-DEVELOPMENT-EXPERIMENT.md).
The executable was 123,282,944 bytes, SHA-256
`54667e335bac83525cfcca6c1bfbcf433c85432342b41f353b78e7dc0b0f8dc4`.
The selected model was trained on only 11 unreviewed LFG messages and is not
qualified for use or publication; its weights may encode private project data.
A second build in a fresh target directory, using the same checkout, prepared
UI assets, model artifact, Rust 1.98.1 and release flags, ran with `--offline`
and produced the same 123,282,944-byte SHA-256. This is same-machine
repeatability, not a cross-machine reproducibility or clean-installation proof.
A copied executable served the UI and developer-stats endpoint from a separate
working directory. With no model or runtime-config files in that directory,
its embedded command matched the external-artifact command and Python on all
15 additional unreviewed DevPulse excerpts. The executable's static import
table listed Windows system libraries and no ONNX Runtime or MSVC runtime DLL;
this is not a clean-machine test or a guarantee about dynamic dependencies.
The same executable also loaded a separately supplied, differently pinned test
fixture through `onnx-predict`, confirming the external selection path remains
usable; that fixture is not a trained product model.
In separate fixture checks, a wrong build-time manifest digest was rejected,
and a binary built from a deliberately altered tokenizer was rejected at
runtime when its file digest did not match the manifest.

Using that same 123 MB binary, one warmup and three sequential new-process
runs on those 15 excerpts had median elapsed times of 70.9 ms for the rule
baseline and 582.0 ms for embedded model inference. Median process CPU time
was 62.5 ms versus 578.1 ms; median process cycle counts were 200.9 million
versus 1.742 billion. Both commands hash the executable as part of their
identity, so these numbers include reading the much larger binary, startup,
parsing, inference and serialization. They are not per-request latency,
memory, an isolated benchmark, or evidence of acceptable model quality.
The parity and performance reports remain in the private dataset directory.
For an interleaved, long-lived-process comparison that separates startup from
per-request timing, see [persistent CPU comparison](PERSISTENT-CPU-BENCHMARK.md).

## Local compatibility evidence

On 2026-10-30, the exporter produced a v2 artifact from the previously created
tiny test model using 12 real, unreviewed DevPulse excerpts for parity only.
Its PyTorch/ONNX Runtime comparison covered 24 cases in batches of one and two,
with zero decision mismatches. A separate native Rust prototype produced the
same token IDs and masks for all 12 excerpts; its maximum absolute logit
differences were `1.49e-8` versus PyTorch and `2.24e-8` versus ONNX Runtime.
All of these excerpts reached the fixture's 32-token limit, so this does not
establish broad variable-length or model-architecture coverage. An independent
graph probe also executed the fixture with 16- and 32-token tensor shapes.

The feature-enabled Windows release binary was copied with the artifact and
runtime config to a separate directory. Its prediction labels matched the
Python model for all 12 excerpts; a wrong manifest digest and a modified labels
file were rejected. The executable was 31,589,376 bytes. Its static Windows
import table did not list an ONNX Runtime DLL; it still imports operating-system
libraries. The reusable comparison tool independently repeated the 12-example
decision check and recorded the source, runtime and tool identities in a private
report. The export used PyTorch's legacy tracing path and emitted a tracer
warning; the tiny fixture result does not establish that other architectures
or input shapes export correctly. This is not a clean-machine installation test.

A short sequential comparison on the same machine and binary used three new
process runs of the 12 excerpts after one warmup for each command. Median elapsed
time was 31.7 ms for the deterministic baseline and 156.6 ms for the fixture
model; median process cycle counts were 94.0 million and 471.3 million,
respectively. These are whole-process measurements including artifact loading,
model preparation, parsing and output serialization, not persistent-server or
per-request latency. The sample is tiny, the weights were trained only on
invented fixture data, and the runs were not interleaved or isolated. This is
not evidence of model quality, a reliable performance comparison, or a release
threshold. The private report is stored at
`ml/lfg-encoder/dataset/private/onnx-shadow-process-comparison.v2.json`.
