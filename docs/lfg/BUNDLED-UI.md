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
