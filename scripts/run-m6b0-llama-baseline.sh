#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s MODEL.gguf\n' "$0" >&2
    exit 2
fi

model_path=$1
if [[ ! -f "$model_path" ]]; then
    printf 'M6-B0: model does not exist: %s\n' "$model_path" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
benchmark="${LLAMA_BENCH_BIN:-/home/fedora-workstation/Development/mi50-artifacts/m6b0-llama-c0bc8591/bin/llama-bench}"
if [[ ! -x "$benchmark" ]]; then
    printf 'M6-B0: llama-bench is not executable: %s\n' "$benchmark" >&2
    exit 2
fi

results_root="${M6B0_RESULTS_DIR:-/home/fedora-workstation/Development/mi50-artifacts/m6b0-results}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_dir="$results_root/$run_id"
mkdir -p "$run_dir"

"$script_dir/capture-env.sh" "$run_dir/environment-before.json"
telemetry_pid=""

stop_telemetry() {
    if [[ -n "${telemetry_pid:-}" ]]; then
        kill "$telemetry_pid" 2>/dev/null || true
        wait "$telemetry_pid" 2>/dev/null || true
        telemetry_pid=""
    fi
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    stop_telemetry
    exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$script_dir/sample-gpu.sh" "$run_dir/telemetry.jsonl" 250 &
telemetry_pid=$!
for _ in {1..40}; do
    [[ -s "$run_dir/telemetry.jsonl" ]] && break
    sleep 0.05
done

common_args=(
    -m "$model_path"
    -r 5
    -b 2048
    -ub 512
    -ngl 999
    -t 24
    -ctk f16
    -ctv f16
    -fa on
    -o jsonl
)

"$benchmark" "${common_args[@]}" -p 512 -n 0,64,128,256 \
    | tee "$run_dir/standard-pp512-tg.jsonl"
"$benchmark" "${common_args[@]}" -p 1 -n 0,64,256,1024 \
    -pg 1,64 -pg 1,256 -pg 1,1024 \
    | tee "$run_dir/context-p1-tg.jsonl"

stop_telemetry
"$script_dir/capture-env.sh" "$run_dir/environment-after.json"
printf 'M6-B0 results: %s\n' "$run_dir" >&2
