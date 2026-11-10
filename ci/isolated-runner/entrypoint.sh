#!/bin/sh
set -eu
IFS= read -r jit_config
jit_config=$(printf '%s' "$jit_config" | tr -d '\r')
[ -n "$jit_config" ]
exec ./run.sh --jitconfig "$jit_config"
