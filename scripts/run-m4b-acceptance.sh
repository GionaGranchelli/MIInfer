#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 MODEL.gguf REFERENCE_TRACE_DIR" >&2
    exit 2
fi

model_path=$1
trace_path=$2

if [[ ! -f "$model_path" ]]; then
    echo "M4-B acceptance: model does not exist: $model_path" >&2
    exit 2
fi
if [[ ! -d "$trace_path" ]]; then
    echo "M4-B acceptance: reference trace directory does not exist: $trace_path" >&2
    exit 2
fi
for checkpoint in embedding.f32 layer-0.f32 layer-35.f32 final-norm.f32 logits.f32; do
    if [[ ! -f "$trace_path/$checkpoint" ]]; then
        echo "M4-B acceptance: required checkpoint is missing: $trace_path/$checkpoint" >&2
        exit 2
    fi
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
host_binary="$repo_root/build/mi50-release/miinfer-qwen3-forward-test"
gpu_binary="$repo_root/build/mi50-release/miinfer-qwen3-forward-gpu-test"
if [[ ! -x "$host_binary" || ! -x "$gpu_binary" ]]; then
    echo "M4-B acceptance: build the mi50-release preset first" >&2
    exit 2
fi

echo "M4-B physical acceptance: host reference comparison"
status=0
"$host_binary" "$model_path" "$trace_path" || status=$?
echo "M4-B physical acceptance: MI50 Release reference comparison"
"$gpu_binary" "$model_path" "$trace_path" || status=$?
if [[ $status -ne 0 ]]; then
    echo "M4-B physical acceptance: FAIL (see comparison output above)" >&2
    exit "$status"
fi
echo "M4-B physical acceptance: PASS"
