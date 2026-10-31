#!/usr/bin/env bash
# Rebuild a deterministic baseline from trusted, explicitly selected inputs.
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "Usage: $0 SOURCE_TAR SOURCE_SHA256 UI_TAR UI_SHA256 CARGO_HOME NEW_OUTPUT_DIR TARGET JOBS" >&2
  exit 2
fi
source_tar=$(realpath -- "$1")
source_sha=$2
ui_tar=$(realpath -- "$3")
ui_sha=$4
cargo_home=$(realpath -- "$5")
output=$6
target=$7
jobs=$8
[[ $source_sha =~ ^[0-9a-f]{64}$ && $ui_sha =~ ^[0-9a-f]{64}$ ]] || { echo 'Expected lowercase SHA-256 digests.' >&2; exit 2; }
[[ $target =~ ^[a-z0-9_]+-unknown-linux-musl$ ]] || { echo 'Select a Linux musl target.' >&2; exit 2; }
[[ $jobs =~ ^[1-9][0-9]*$ ]] && (( jobs <= 256 )) || { echo 'Jobs must be between 1 and 256.' >&2; exit 2; }
[[ $(sha256sum -- "$source_tar" | cut -d ' ' -f 1) == "$source_sha" ]] || { echo 'Source archive digest mismatch.' >&2; exit 1; }
[[ $(sha256sum -- "$ui_tar" | cut -d ' ' -f 1) == "$ui_sha" ]] || { echo 'UI archive digest mismatch.' >&2; exit 1; }
[[ -d $cargo_home ]] || { echo 'Cargo cache directory is missing.' >&2; exit 1; }
mkdir -- "$output" # Refuse an existing path; retain partial work on failure.
output=$(realpath -- "$output")

for suffix in a b; do
  source_dir="$output/source-$suffix"
  target_dir="$output/target-$suffix"
  mkdir -- "$source_dir"
  tar -xf "$source_tar" -C "$source_dir"
  tar -xf "$ui_tar" -C "$source_dir/ui/master-clyffy-web"
  ui_dir="$source_dir/ui/master-clyffy-web/dist"
  [[ -f $ui_dir/index.html ]] || { echo 'UI archive must supply dist/index.html.' >&2; exit 1; }
  flags=$(printf '%s\x1f%s' "--remap-path-prefix=$source_dir=/lfg-source" "--remap-path-prefix=$target_dir=/lfg-target")
  (
    cd -- "$source_dir"
    env -u RUSTFLAGS -u RUSTC_WRAPPER -u RUSTC_WORKSPACE_WRAPPER \
      CARGO_HOME="$cargo_home" CARGO_TARGET_DIR="$target_dir" CARGO_BUILD_JOBS="$jobs" \
      CARGO_ENCODED_RUSTFLAGS="$flags" LFG_UI_DIST="$ui_dir" \
      cargo build --offline --locked --release -p lfg-server --features embedded-ui --target "$target"
  )
  binary="$target_dir/$target/release/lfg-server"
  program_headers=$(readelf -l "$binary")
  dynamic_section=$(readelf -d "$binary")
  [[ $program_headers != *INTERP* && $dynamic_section != *NEEDED* ]] || { echo 'Binary has dynamic dependencies.' >&2; exit 1; }
done

binary_a="$output/target-a/$target/release/lfg-server"
binary_b="$output/target-b/$target/release/lfg-server"
if ! cmp -s -- "$binary_a" "$binary_b"; then
  sha256sum -- "$binary_a" "$binary_b" >&2
  echo 'Executables differ; inspect the retained builds.' >&2
  exit 1
fi
python3 - "$output" "$source_sha" "$ui_sha" "$target" "$jobs" "$binary_a" <<'PY'
import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
output, source_sha, ui_sha, target, jobs, binary = sys.argv[1:]
path = Path(binary)
report = {
    'schemaVersion': 'lfg.linux-baseline-build.v1',
    'sourceArchiveSha256': source_sha, 'uiArchiveSha256': ui_sha,
    'target': target, 'jobs': int(jobs), 'platform': platform.platform(),
    'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip(),
    'cargo': subprocess.check_output(['cargo', '--version'], text=True).strip(),
    'binarySha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'binaryBytes': path.stat().st_size, 'binariesIdentical': True,
    'scope': 'Same-machine offline rebuild; no model or release qualification.',
}
with (Path(output) / 'build-report.json').open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps(report))
PY
