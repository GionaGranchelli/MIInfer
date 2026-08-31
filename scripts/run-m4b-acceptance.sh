#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 MODEL.gguf CPU_TRACE_DIR EXTERNAL_GPU_TRACE_DIR" >&2
    exit 2
fi

model_path=$1
cpu_trace_path=$2
gpu_trace_path=$3

if [[ ! -f "$model_path" ]]; then
    echo "M4-B acceptance: model does not exist: $model_path" >&2
    exit 2
fi
if [[ ! -d "$cpu_trace_path" ]]; then
    echo "M4-B acceptance: CPU reference trace directory does not exist: $cpu_trace_path" >&2
    exit 2
fi
if [[ ! -d "$gpu_trace_path" ]]; then
    echo "M4-B acceptance: external GPU trace directory does not exist: $gpu_trace_path" >&2
    exit 2
fi
for trace_path in "$cpu_trace_path" "$gpu_trace_path"; do
    for checkpoint in final-norm.f32 logits.f32; do
        if [[ ! -f "$trace_path/$checkpoint" ]]; then
            echo "M4-B acceptance: required checkpoint is missing: $trace_path/$checkpoint" >&2
            exit 2
        fi
    done
done

if [[ ! -f "$cpu_trace_path/embedding.f32" || ! -f "$cpu_trace_path/layer-35.f32" ]]; then
    echo "M4-B acceptance: CPU trace is missing structural checkpoints" >&2
    exit 2
fi
if [[ ! -f "$gpu_trace_path/layer-35.f32" ]]; then
    echo "M4-B acceptance: external GPU trace is missing layer-35.f32" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
host_binary="$repo_root/build/mi50-release/miinfer-qwen3-forward-test"
release_gpu_binary="$repo_root/build/mi50-release/miinfer-qwen3-forward-gpu-test"
debug_gpu_binary="$repo_root/build/mi50-debug/miinfer-qwen3-forward-gpu-test"
if [[ ! -x "$host_binary" || ! -x "$release_gpu_binary" || ! -x "$debug_gpu_binary" ]]; then
    echo "M4-B acceptance: build the mi50-release and mi50-debug presets first" >&2
    exit 2
fi

echo "M4-B24 physical acceptance: host strict comparison (diagnostic)"
"$host_binary" "$model_path" "$cpu_trace_path" || true
echo "M4-B24 physical acceptance: MI50 Debug backend-envelope comparison"
"$debug_gpu_binary" "$model_path" --backend-envelope "$cpu_trace_path" "$gpu_trace_path"
echo "M4-B24 physical acceptance: MI50 Release backend-envelope comparison"
"$release_gpu_binary" "$model_path" --backend-envelope "$cpu_trace_path" "$gpu_trace_path"
echo "M4-B physical acceptance: PASS"
