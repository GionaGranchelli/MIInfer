#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s MODEL.gguf\n' "$0" >&2
    exit 2
fi

model_path=$1
if [[ ! -f "$model_path" ]]; then
    printf 'M5-A baseline: model does not exist: %s\n' "$model_path" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
benchmark="$repo_root/build/mi50-release/miinfer-qwen3-inference-bench"
if [[ ! -x "$benchmark" ]]; then
    printf 'M5-A baseline: build the mi50-release preset first\n' >&2
    exit 2
fi

exec "$repo_root/scripts/run-bench.sh" "$benchmark" "$model_path" \
    --prompt hello --generated-tokens 8 --warmup 1 --iterations 3
