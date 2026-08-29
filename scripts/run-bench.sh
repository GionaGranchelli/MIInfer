#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    printf 'usage: %s BENCHMARK [ARGUMENT ...]\n' "$0" >&2
    exit 2
fi

benchmark=$1
shift
if [[ "$benchmark" != */* ]]; then
    benchmark_path=$(command -v "$benchmark" || true)
else
    benchmark_path=$benchmark
fi
if [[ -z "$benchmark_path" || ! -x "$benchmark_path" ]]; then
    printf 'benchmark executable is not executable or not found: %s\n' "$benchmark" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
results_root="${MIINFER_BENCH_RESULTS_DIR:-$repo_root/bench/results}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_dir="$results_root/$run_id"
mkdir -p "$run_dir"

"$script_dir/capture-env.sh" "$run_dir/environment-before.json"

telemetry_pid=""
if [[ "${MIINFER_TELEMETRY:-1}" != "0" ]]; then
    "$script_dir/sample-gpu.sh" "$run_dir/telemetry.jsonl" \
        "${MIINFER_TELEMETRY_INTERVAL_MS:-250}" &
    telemetry_pid=$!
    for _ in {1..40}; do
        [[ -s "$run_dir/telemetry.jsonl" ]] && break
        sleep 0.05
    done
fi

status=0
"$benchmark_path" "$@" --json-output "$run_dir/result.json" || status=$?

if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
fi

"$script_dir/capture-env.sh" "$run_dir/environment-after.json"

if [[ $status -ne 0 ]]; then
    printf 'benchmark failed with exit status %d; results retained in %s\n' "$status" "$run_dir" >&2
    exit "$status"
fi
if [[ ! -s "$run_dir/result.json" ]]; then
    printf 'benchmark succeeded but did not produce %s\n' "$run_dir/result.json" >&2
    exit 1
fi

printf 'benchmark results: %s\n' "$run_dir" >&2
