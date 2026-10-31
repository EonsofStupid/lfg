# Bundled UI builds

The `embedded-ui` feature embeds the built browser assets in the Rust server.
The executable serves them when `MASTER_CLYFFY_UI_DIR` is unset; setting that
variable explicitly selects an external UI directory instead. Without the
feature, the existing development behavior serves from the checkout or the
specified external directory. Configuration and state remain separately
configurable; state is written to `LFG_STATE_DIR` or the existing temporary
default. Provider integrations can still require their own tools or services.

Build on the target OS and architecture, with Node/npm and Rust installed:

```powershell
cd ui/master-clyffy-web
npm ci
npm run build
cd ../..
cargo build --release --locked -p lfg-server --features embedded-ui
```

The resulting executable is `target/release/lfg-server.exe` on Windows and
`target/release/lfg-server` on Unix-like systems. Start it with `serve` and
open its configured loopback address. `LFG_UI_DIST` can select a different
asset directory at **build time**; it must contain `index.html`. A missing
asset tree, missing index, or symlink inside the asset tree fails the bundled
build. Rebuild the binary after changing UI assets. Node and the UI asset
directory are not needed to serve the embedded UI after the build.

For a Windows x86-64 MSVC build with the C runtime linked into the executable,
the following is an explicit build configuration. `/Brepro` requests
deterministic linking; the target and flags are selected for this build rather
than imposed on other platforms. Rust recommends inspecting the output to
confirm static C runtime linkage on the chosen target. [Rust linkage reference](https://doc.rust-lang.org/reference/linkage.html#static-and-dynamic-c-runtimes)

```powershell
$previousRustFlags = $env:RUSTFLAGS
try {
    $env:RUSTFLAGS = '-C target-feature=+crt-static -C link-arg=/Brepro'
    cargo build --release --locked -p lfg-server --features embedded-ui --target x86_64-pc-windows-msvc
    if ($LASTEXITCODE -ne 0) { throw 'Release build failed' }
} finally {
    $env:RUSTFLAGS = $previousRustFlags
}
```

This executable is under `target/x86_64-pc-windows-msvc/release/`. It still
depends on Windows operating-system libraries; an optional provider adapter
can require additional software. The choice does not imply equivalent static
linking or qualification on macOS, Linux, or Windows ARM.

The embedded server returns missing API and asset paths as 404; non-asset paths
can fall back to the single-page application's index. The server remains
unauthenticated and loopback by default. Hosting it on a network needs a
separate, tested authentication and authorization boundary before exposing its
administrative and diagnostic endpoints.

The test suite copies an embedded build to an empty working directory, starts
it, and requests the page, JavaScript, and API. On 2026-10-30, two offline,
locked Windows x86-64 MSVC release builds with the flags above were made into
separate fresh target directories using the same checkout, dependency cache,
UI assets and rustc 1.98.1. Both executables had SHA-256
`3316fad1044dd8e25757cd95d69b12ebcc943c39349ea8ef661a25f77fc1c198`
and were 4,894,720 bytes. One copied executable served its UI and API from an
empty working directory. Import-table inspection showed Windows system DLLs
and no `VCRUNTIME140.dll`. Two consecutive UI builds using the same installed
Node 22.23.2/npm 10.9.8 dependency tree also produced identical five-file
asset hashes.

This checks same-machine repeatability and that the UI does not depend on the
checkout. It does not prove reproducibility on another machine, toolchain or
source path, installation on a clean machine, the absence of all system
dependencies, supported builds for every platform, or learned-model CPU
inference. Those remain release qualification work. Without `/Brepro`, two
otherwise matching local builds differed in a debug identifier, so ordinary
builds should not be assumed byte-reproducible.

For a repeatable same-machine Windows baseline build, choose a full committed
revision and a **new** output directory, then run:

```powershell
.\scripts\build-windows-baseline.ps1 -Commit FULL_COMMIT_SHA -OutputDir NEW_DIRECTORY -Jobs 2
```

The command archives that exact commit, extracts it into two separate source
directories, performs offline `npm ci` and UI builds, and builds two locked
release executables in separate target directories. It uses static CRT,
`/Brepro`, and path remapping, then compares the UI file hashes and executable
hashes. It writes `build-report.json` and retains both builds; it refuses an
existing output directory. It requires the selected Node, npm, Rust and MSVC
toolchain and complete dependency caches. It records tool versions but does not
attest the compiler, operating system, caches or upstream package publishers;
package-manager offline mode is not a network sandbox for build scripts.
By default it builds the deterministic baseline with the UI, without a model.
To embed an operator-selected ONNX artifact instead, supply both its directory
and independently selected manifest digest:

```powershell
.\scripts\build-windows-baseline.ps1 -Commit FULL_COMMIT_SHA -OutputDir NEW_PRIVATE_DIRECTORY `
  -Jobs 2 -ModelArtifactDir PRIVATE_ARTIFACT_DIRECTORY -ModelManifestSha256 EXPECTED_LOWERCASE_SHA256
```

The optional model build verifies the manifest and every listed file digest,
copies the complete manifested artifact to one shared snapshot for the two builds, rechecks that
snapshot, and builds with `embedded-ui,embedded-onnx`. The output retains the
snapshot and both executables; protect it as private when the weights derive
from private project data. Matching hashes establish same-machine repeatability
for these selected inputs, not model quality, artifact authenticity, clean-host
reproducibility, or release eligibility.

On 2026-10-31, two clean Windows x86-64 MSVC builds of commit
`fcfe4a53a105874d06e13e6e2845c87100cf4151` used Rust 1.98.1 and the same
private experimental model manifest
`f28f78e72442df23c46fce129c50c8121f2976eb28c6c9104274fe7db22c22e8`.
The separate source and target directories produced identical five-file UI
assets and 125,399,552-byte executables with SHA-256
`42942aaf8619cc8755e73e8920a6954f580426bddb6692ff2129ba96f4a4c7d5`.
The retained snapshot passed the existing evaluator's file-identity check for
all eight manifest entries. A copied first executable served its embedded UI
and developer statistics and processed two real unreviewed inputs without
external model files. Both executable hashes also matched an earlier local
in-place build. These are same-machine observations with a model trained on
unreviewed development data, not a qualified model, clean-host proof, or
permission to distribute its private-derived weights. The private build report
and artifacts remain under `target/lfg-embedded-repro-20261031-v3/`.
The command was also run end to end with an output path containing spaces; its
two executables matched each other and the digest reported below. npm warned
that one package's install script was not approved; the offline UI builds
still completed, but that warning is part of the observed environment.

On 2026-10-31, two offline builds of commit
`3e2c8fb294f20cfb17aeacddc9b1bc45053ae56f` in separate source and target paths
produced identical five-file UI outputs and 8,305,152-byte executables with
SHA-256 `0e66479251ad7a14b8fcce304f891e31a572b542bc30c71ee7e049f715be341a`.
The machine used rustc 1.98.1, Node 24.21.0 and npm 11.19.0. A copied binary
matched the LFG kernel golden and served the embedded index, JavaScript and
developer statistics endpoint from a separate local working directory with a
restricted environment. Its inspected import table listed Windows system DLLs,
not an MSVC runtime or ONNX Runtime DLL. This is not a clean-machine test,
an independent toolchain build, a signed release, or evidence of model quality.

## Linux ARM64 baseline

The following command rebuilds a Linux musl baseline twice using explicitly
selected, trusted source and prebuilt UI archives. The UI archive must contain
`dist/index.html`. Supply their SHA-256 digests, a prepared Cargo cache, a new
output directory, a locally supported musl target and a job count:

```bash
bash scripts/build-linux-baseline.sh SOURCE_TAR SOURCE_SHA256 UI_TAR UI_SHA256 \
  CARGO_CACHE NEW_OUTPUT_DIR aarch64-unknown-linux-musl 2
```

The script checks archive digests, extracts each into separate directories,
builds with Cargo's locked offline mode and path remapping, checks for ELF
interpreter and shared-library dependencies, and compares both executables.
It writes `build-report.json` when they match. Archives are trusted inputs;
their hashes do not establish origin or protect against malicious contents.
Prepare the selected Rust target, linker, toolchain and dependency cache first.
Offline Cargo mode does not sandbox build scripts from the network.

On 2026-10-31, the `Clyffy-Ring` Spark built commit
`fd594a8f8759790829ed5560f74e709ed25f38cd` using Rust 1.97.1 and an isolated
Cargo cache populated from the lock before offline builds. The source archive
SHA-256 was `c0fa3346a9c5ef7459d257c2fa368ae6a5924daa640e9b4c86b50dd4e554f4cf`.
The UI archive SHA-256 was
`5fd344c076a5bef1b1e8f4f36969505081a4529641fab34e430c967f113780dc`;
it contained the Windows-built UI assets from the prior check, and the UI
source was unchanged between those commits. Two separate Spark source and
target directories produced identical 10,009,008-byte ARM64 executables with
SHA-256 `e68989acee41152e82e5e72c976ec752af8e63684b336d8d92eb49424dbb552d`.
`readelf` reported no interpreter or `NEEDED` entries.
The reusable script was also run end to end under a path containing spaces and
produced the same digest; it rejected an invalid digest and an existing output
directory. Its report and separate builds remain on the Spark.

That binary matched the same LFG kernel golden inside a Docker `scratch`
container running as a non-root user with a read-only filesystem, no network,
and no bundled operating-system libraries. A separate scratch-container check
with a writable temporary filesystem and host networking bound to loopback
served the embedded index, JavaScript and developer statistics endpoint. These
checks share the Spark's kernel and do not prove behavior on a clean machine,
other Linux architectures, an authenticated hosted deployment, or a trained
model. No private project requests or model weights were transferred for them.
