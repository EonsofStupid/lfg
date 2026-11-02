# Native owned byte-model probe

This experiment exports the independently initialized owned byte student to
ONNX and can embed that artifact alongside the existing developer UI in the
Rust server executable. The model predicts the next byte; it does **not**
produce LFG's intended prompt assessment, model recommendations, agent count,
or reasons. The live assessment endpoint still uses its existing deterministic
implementation. There has been no task-quality evaluation or promotion.

`ml/lfg-encoder/export_owned_byte_onnx.py` verifies the private checkpoint and
its model implementation against their recorded digests, exports one dynamic
sequence-length model, rejects external tensor references, and checks CPU
ONNX Runtime logits against PyTorch on explicitly selected local source
prefixes. Set the export settings and parity lengths in a local configuration;
the [example](../../ml/lfg-encoder/owned-byte-onnx-export.example.json) is
illustrative. Output must be a new directory and, inside a Git worktree, both
files must be ignored and untracked. A failed export may leave an incomplete
directory without a manifest. The trace-based exporter checks only the
configured parity lengths; native input bounds are enforced separately.

```powershell
python ml/lfg-encoder/export_owned_byte_onnx.py `
  --checkpoint PRIVATE_CHECKPOINT_DIRECTORY --config EXPORT_CONFIG.json `
  --source PRIVATE_SOURCE_FILE --output NEW_PRIVATE_EXPORT_DIRECTORY
```

The Rust feature `owned-byte-onnx` adds `owned-byte-probe`, which loads the
separate export only after checking the caller-supplied manifest digest and the
manifest's model digest. `embedded-owned-byte` also includes the manifest and
model at build time. Both are opt-in offline JSONL probes requiring
`--experimental`. They reject empty or over-context inputs and enforce the
runtime's configured limits. The embedded probe has bundled defaults and can
instead use `--runtime-config`; the separate-artifact probe requires one.
Unknown JSON fields are allowed. The probes emit the predicted byte; requesting
`--include-logits` emits the final-position vector for local comparison.

```powershell
$env:LFG_EMBEDDED_OWNED_BYTE_DIR = 'PRIVATE_EXPORT_DIRECTORY'
$env:LFG_EMBEDDED_OWNED_BYTE_MANIFEST_SHA256 = (Get-FileHash `
  -Algorithm SHA256 "$env:LFG_EMBEDDED_OWNED_BYTE_DIR/manifest.json").Hash.ToLowerInvariant()
$env:LFG_UI_DIST = 'BUILT_UI_DIST_DIRECTORY'
cargo build --locked -p lfg-server --features embedded-owned-byte,embedded-ui
'{"id":"example","text":"Example input"}' | target/debug/lfg-server.exe `
  embedded-owned-byte-probe --experimental
```

`ml/lfg-encoder/compare_owned_byte_native.py` compares final-position native
logits and next-byte choices with the source PyTorch checkpoint, keeping the
selected source and vectors in local process memory. It supports embedded and
separate-artifact modes, operator-supplied lengths, tolerances and timeout.
Reports contain digests; those and derived weights can be sensitive. An
executable with these weights embedded can disclose the owner's training data
and must remain private. Build products are not independently reviewed or
distributed.

On 2026-11-02, the 37,888-parameter private multi-document checkpoint was
exported with source-prefix lengths 1, 4, 32 and 64 from the owner's LFG
README. ONNX Runtime's largest absolute difference from PyTorch was 0.00000167.
A Windows debug executable containing the model and UI was then built; both
its embedded and separate-artifact paths matched the checkpoint at those
lengths, with maximum absolute difference 0.00000191. A copied executable ran
its embedded probe from an otherwise empty directory, and its health endpoint,
embedded UI page and script returned HTTP 200. This verifies that specific
local executable and inputs, not a release build, target-sized model, operating
system compatibility, independent assessment quality, or CPU latency under
load. The private debug executable is under the ignored `target` directory.
