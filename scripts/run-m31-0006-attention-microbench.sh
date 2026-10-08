#!/usr/bin/env bash
set -euo pipefail

if [[ ${M31_GPU_RUN_AUTHORIZED:-} != 1 ]]; then
    echo "GPU execution blocked: lead authorization is required (M31_GPU_RUN_AUTHORIZED=1)" >&2
    exit 2
fi
if (( $# < 3 )); then
    echo "usage: M31_GPU_RUN_AUTHORIZED=1 $0 MAX_SECONDS NEW_OUTPUT_DIR BENCHMARK [--device N] [--repeats N]" >&2
    exit 2
fi

max_seconds=$1
output_dir=$2
benchmark=$3
shift 3
if [[ ! $max_seconds =~ ^[0-9]+$ ]] || (( max_seconds < 10 || max_seconds > 300 )); then
    echo "MAX_SECONDS must be an integer from 10 through 300 (per matrix case)" >&2
    exit 2
fi
if [[ ! -x $benchmark ]]; then
    echo "benchmark is not executable: $benchmark" >&2
    exit 2
fi
command -v timeout >/dev/null || { echo "GNU timeout is required" >&2; exit 2; }
umask 077
mkdir -- "$output_dir"
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

metadata="$output_dir/run-metadata.txt"
{
    printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
    printf 'kernel=%s\n' "$(uname -a)"
    printf 'source_sha=%s\n' "$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'benchmark_source_sha256=%s\n' "$(sha256sum "$repo_root/bench/m31_0006_graph_attention_microbench.cpp" | awk '{print $1}')"
    printf 'benchmark_sha256=%s\n' "$(sha256sum "$benchmark" | awk '{print $1}')"
    if [[ -f $repo_root/container/Containerfile ]]; then
        printf 'containerfile_sha256=%s\n' "$(sha256sum "$repo_root/container/Containerfile" | awk '{print $1}')"
        printf 'container_image_ref=%s\n' "$(rg -o 'docker\.io/[^[:space:]@]+@sha256:[a-f0-9]+' "$repo_root/container/Containerfile" | head -n 1 || true)"
    fi
    printf 'worktree_status:\n'
    git -C "$repo_root" status --short 2>/dev/null || true
    printf 'max_seconds_per_case=%s\n' "$max_seconds"
    printf 'benchmark=%q\narguments=' "$benchmark"
    printf '%q ' "$@"
    printf '\n\n--- hipconfig ---\n'
    if command -v hipconfig >/dev/null; then hipconfig --full 2>&1 || true; else echo unavailable; fi
    printf '\n--- rocm-smi before ---\n'
    if command -v rocm-smi >/dev/null; then
        rocm-smi --showproductname --showmeminfo vram --showclocks --showtemp --showpower 2>&1 || true
    else echo unavailable; fi
} >"$metadata"

record_after() {
    {
        printf '\n--- rocm-smi after ---\n'
        if command -v rocm-smi >/dev/null; then
            rocm-smi --showproductname --showmeminfo vram --showclocks --showtemp --showpower 2>&1 || true
        else echo unavailable; fi
        printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
    } >>"$metadata"
}
trap record_after EXIT

for context in 8192 32768 65536 131072; do
    mode=baseline
    log="$output_dir/context-${context}-${mode}.log"
    printf 'case_start_utc=%s context=%s mode=%s\n' "$(date -u +%FT%TZ)" "$context" "$mode" | tee "$log"
    set +e
    timeout --signal=INT --kill-after=5s "${max_seconds}s" "$benchmark" \
        --context "$context" --mode "$mode" "$@" 2>&1 | tee -a "$log"
    result=${PIPESTATUS[0]}
    set -e
    printf 'case_exit=%s ended_utc=%s\n' "$result" "$(date -u +%FT%TZ)" | tee -a "$log"
    if (( result != 0 )); then
        echo "run stopped at context=$context mode=$mode (exit $result); raw case log: $log" >&2
        exit "$result"
    fi
done

echo "complete matrix and raw per-case repeats: $output_dir"
