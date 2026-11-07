#!/usr/bin/env bash
# Only discard known unused SDKs on the disposable GitHub-hosted image builder.
set -euo pipefail
if [[ "${RUNNER_ENVIRONMENT:-}" != "github-hosted" ]]; then
  echo 'Skipping hosted-image cleanup on a non-hosted runner.'
  exit 0
fi
echo 'Disk space before cleanup:'
df -h /
for sdk in /usr/local/lib/android /usr/share/dotnet /usr/local/share/boost; do
  if [[ -d "$sdk" && ! -L "$sdk" ]]; then
    sudo du -sh "$sdk"
    sudo rm -rf -- "$sdk"
  fi
done
echo 'Disk space after cleanup:'
df -h /
