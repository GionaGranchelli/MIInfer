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

stop_telemetry() {
    if [[ -z "${telemetry_pid:-}" ]]; then
        return 0
    fi

    # The sampler handles TERM itself.  kill -0 avoids reporting an error when
    # it already exited, and wait reaps the child in either case.
    if kill -0 "$telemetry_pid" 2>/dev/null; then
        kill "$telemetry_pid" 2>/dev/null || true
    fi
    wait "$telemetry_pid" 2>/dev/null || true
    telemetry_pid=""
}

cleanup() {
    local exit_status=$?
    trap - EXIT INT TERM
    stop_telemetry
    exit "$exit_status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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

stop_telemetry

capture_after_status=0
"$script_dir/capture-env.sh" "$run_dir/environment-after.json" || capture_after_status=$?

if [[ $status -ne 0 ]]; then
    printf 'benchmark failed with exit status %d; results retained in %s\n' "$status" "$run_dir" >&2
    exit "$status"
fi
if [[ $capture_after_status -ne 0 ]]; then
    printf 'post-run environment capture failed with exit status %d; results retained in %s\n' \
        "$capture_after_status" "$run_dir" >&2
    exit "$capture_after_status"
fi
if [[ ! -s "$run_dir/result.json" ]]; then
    printf 'benchmark succeeded but did not produce %s\n' "$run_dir/result.json" >&2
    exit 1
fi

printf 'benchmark results: %s\n' "$run_dir" >&2
