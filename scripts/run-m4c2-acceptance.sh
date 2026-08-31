#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 MODEL.gguf" >&2
    exit 2
fi

model_path=$1
if [[ ! -f "$model_path" ]]; then
    echo "M4-C2 acceptance: model does not exist: $model_path" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
debug_binary="$repo_root/build/mi50-debug/miinfer-qwen3-decode-sequence-gpu-test"
release_binary="$repo_root/build/mi50-release/miinfer-qwen3-decode-sequence-gpu-test"
if [[ ! -x "$debug_binary" || ! -x "$release_binary" ]]; then
    echo "M4-C2 acceptance: build the mi50-debug and mi50-release presets first" >&2
    exit 2
fi

echo "M4-C2 physical acceptance: MI50 Debug"
debug_status=0
"$debug_binary" "$model_path" --gpu-only || debug_status=$?
echo "M4-C2 physical acceptance: MI50 Release"
release_status=0
"$release_binary" "$model_path" --gpu-only || release_status=$?
if [[ $debug_status -ne 0 || $release_status -ne 0 ]]; then
    echo "M4-C2 physical acceptance: FAIL (Debug=$debug_status Release=$release_status)" >&2
    exit 1
fi
echo "M4-C2 physical acceptance: PASS"
