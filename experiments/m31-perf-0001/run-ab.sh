#!/usr/bin/env bash
set -euo pipefail

root=/tmp/m31-perf-0001
runner="$root/runner/e2_0003_same_source_ab.py"
guard="$root/runner/run-m31-exploratory.sh"
model=/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
image=localhost/miinfer-dev:rocm-7.2.1
image_digest=sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e
model_sha=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
build_flags='Release; MIINFER_ENABLE_HIP=ON; MIINFER_BUILD_TESTS=ON; MIINFER_BUILD_BENCHMARKS=ON; MIINFER_BUILD_RESEARCH_TOOLS=OFF; CMAKE_HIP_ARCHITECTURES=gfx906; CMAKE_PREFIX_PATH=/opt/rocm-7.2.1; hipblas_DIR=/opt/rocm-7.2.1/lib/cmake/hipblas; hip_DIR=/opt/rocm-7.2.1/lib/cmake/hip; FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/deps/nlohmann_json-src'

run_call() {
    local label=$1 role=$2 mode=$3 phase=$4
    local sha binary pairs=1 warmup=1 iterations=1
    local -a mode_args=() capture_args=()
    case "$role" in
        control) sha=016472160c295eb1927b1ed6e095719ada6c3d18 ;;
        candidate) sha=8a9caec3366512e459a93bad68adde181225304b ;;
        *) echo "unknown role: $role" >&2; return 2 ;;
    esac
    binary="$root/bin/$role"
    if [[ "$mode" == eager ]]; then mode_args+=(--disable-hip-graph); fi
    if [[ "$phase" == measured ]]; then iterations=2; fi
    if [[ "$phase" == validation ]]; then
        warmup=0
        capture_args+=(--capture-raw-logits)
    fi
    echo "START $(date -u +%FT%TZ) $label"
    python3 "$runner" \
        --model "$model" --expected-model-sha "$model_sha" \
        --binary "$binary" --source-manifest "$root/manifests/source-manifest-$role.json" \
        --source-sha "$sha" --build-flags "$build_flags" \
        --image "$image" --expected-image-digest "$image_digest" \
        --guard "$guard" --output-dir "$root/results/$label" \
        --expected-bdf 0000:06:00.0 --context-profile 8K \
        --prompt-tokens 2048 --generate-tokens 5 --pairs "$pairs" \
        --iterations "$iterations" --warmup "$warmup" --expected-seconds 240 \
        "${mode_args[@]}" "${capture_args[@]}"
    echo "END $(date -u +%FT%TZ) $label"
}

for mode in eager graph; do
    if [[ "$mode" == eager && ${M31_PERF_RESUME_EAGER_WARMUPS:-0} != 1 ]]; then
        # Each warmup call also records its one post-warmup iteration as sample one.
        run_call "$mode-control-warmup" control "$mode" warmup
        run_call "$mode-candidate-warmup" candidate "$mode" warmup
    fi

    if [[ "$mode" == eager ]]; then
        # The two retained prelude samples are A1/B1; complete A/B/B/A/A/B in two-iteration batches.
        run_call "$mode-candidate-r1" candidate "$mode" measured
        run_call "$mode-control-r2" control "$mode" measured
        run_call "$mode-control-r3" control "$mode" measured
        run_call "$mode-candidate-r2" candidate "$mode" measured
    else
        run_call "$mode-control-r1" control "$mode" measured
        run_call "$mode-candidate-r1" candidate "$mode" measured
        run_call "$mode-candidate-r2" candidate "$mode" measured
        run_call "$mode-control-r2" control "$mode" measured
        run_call "$mode-control-r3" control "$mode" measured
        run_call "$mode-candidate-r3" candidate "$mode" measured
    fi

    run_call "$mode-control-validation" control "$mode" validation
    run_call "$mode-candidate-validation" candidate "$mode" validation
done
