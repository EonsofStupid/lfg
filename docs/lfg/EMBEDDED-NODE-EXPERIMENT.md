# Embedded per-node suggestion experiment

The optional `embedded-node` build packages a selected frozen encoder, tokenizer,
per-node exemplar pack, snapshot and configurations into the executable. The
build selects inputs explicitly; no model or pack is checked into the source
tree. This is an experimental packaging path, not an approved model or release.

With UI assets already built, select artifacts and expected hashes obtained from
a trusted source, then build:

```powershell
$env:LFG_EMBEDDED_NODE_DIR = 'ENCODER_ARTIFACT_DIRECTORY'
$env:LFG_EMBEDDED_NODE_MANIFEST_SHA256 = 'EXPECTED_MANIFEST_SHA256'
$env:LFG_EMBEDDED_NODE_PACK = 'EXEMPLAR_PACK_JSON'
$env:LFG_EMBEDDED_NODE_PACK_SHA256 = 'EXPECTED_PACK_SHA256'
$env:LFG_EMBEDDED_NODE_SNAPSHOT = 'SNAPSHOT_JSON'
$env:LFG_EMBEDDED_NODE_RUNTIME = 'RUNTIME_CONFIG_JSON'
$env:LFG_EMBEDDED_NODE_CONFIG = 'EXEMPLAR_CONFIG_JSON'
cargo build --release --locked -p lfg-server --features 'embedded-ui,embedded-node'
```

The builder requires regular files and checks the selected manifest and pack
digests. On startup, the command checks those identities and the pack's
compatibility with the embedded snapshot, configuration and manifest. It checks
model and tokenizer bytes when it first loads them. A digest only binds bytes
to an expected digest; it does not establish publisher, reviewer, rights or
artifact authenticity. Pack contents may be sensitive even without raw text.

```powershell
lfg-server embedded-node-suggest --experimental
```

The command consumes the same `id`/`text` JSONL stream as
`onnx-node-suggest`, uses the selected embedded snapshot, and produces the same
unpromoted result schema. It loads the encoder only when the kernel escalates
at the pack's node. `--prediction-only` emits the validation format. The
optional `--runtime-config PATH` and `--exemplar-config PATH` flags override the
embedded defaults; incompatible encoder settings are rejected. The external
artifact command remains available for independently selected updates. Changing
embedded weights, pack or snapshot requires a rebuild; no signed update,
distribution or rollback mechanism is implemented. The current native node
runtime accepts only `cpuThreads: 1`; configurable multi-thread execution has
not yet been implemented or measured.

Rust callers with this feature can use `embedded_snapshot` and
`NodeSuggester::open_embedded`, supplying their own operations configuration and
optional config overrides. The result is still advisory and does not dispatch
work or establish quality. This is currently exposed from the server crate,
not a completed RRFlow, SWAG or MCP integration.

## Local packaging check

On 2026-10-31 a Windows release build embedded the actual frozen encoder and a
**test-only** pack built with simulated review evidence. A copied executable ran
an escalation from an empty working directory with no artifact files and agreed
with the external-artifact command on that test input. The binary was
124,350,976 bytes, SHA-256
`ae167a74cdad07b0002d9daee695c5fc795fa28118bfa53b63e4a4b78ae43d77`.
That local test executable must not be treated as a qualified release. A wrong
manifest or pack digest rejected the build, and an intentionally altered embedded model
was rejected on escalation without emitting a suggestion. This did not test a
clean machine, authenticate artifacts, evaluate real labels, or qualify performance.

Two further offline builds in separate target directories used the same inputs,
Rust and Cargo 1.98.1, `-C target-feature=+crt-static -C link-arg=/Brepro`, and
different build concurrency. Both produced 124,543,488-byte executables with
SHA-256 `881d9e2f8d49fb418c904348211707632300f830673400e8cebf0d99c683c342`.
This is same-machine repeatability of a **test-only** build, not cross-machine
reproducibility, a clean installation test, or permission to publish its contents.
