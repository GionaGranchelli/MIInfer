#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s MODEL.gguf\n' "$0" >&2
    exit 2
fi

model_path=$1
if [[ ! -f "$model_path" ]]; then
    printf 'M5-B profile: model does not exist: %s\n' "$model_path" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
benchmark="$repo_root/build/mi50-release/miinfer-qwen3-decode-profile"
if [[ ! -x "$benchmark" ]]; then
    printf 'M5-B profile executable is missing: %s\n' "$benchmark" >&2
    printf 'build it with: cmake --build --preset mi50-release --target miinfer-qwen3-decode-profile\n' >&2
    exit 2
fi

exec "$repo_root/scripts/run-bench.sh" "$benchmark" "$model_path"
