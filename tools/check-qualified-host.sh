#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model_path=${1:-"$HOME/models/Qwen3.8-27B-Q4_K_M.gguf"}
image=${MIINFER_IMAGE:-localhost/miinfer-dev:rocm-7.2.1}
expected_model_sha=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
expected_image_digest=sha256:42851dac319afce41cf993e25f95005b7f2cd0a0f6abd32ad8f25cd876ec56df

if [[ ! -f $model_path ]]; then
    echo "FAIL model missing: $model_path" >&2
    exit 1
fi
runtime=${MIINFER_CONTAINER_RUNTIME:-}
if [[ -z $runtime ]]; then
    command -v podman >/dev/null && runtime=podman || runtime=docker
fi
command -v "$runtime" >/dev/null || { echo "FAIL container runtime missing: $runtime" >&2; exit 1; }

image_id=$($runtime image inspect "$image" --format '{{.Id}}')
image_base_digest=$($runtime image inspect "$image" --format '{{index .Config.Labels "org.opencontainers.image.base.digest"}}')
model_sha=$(sha256sum "$model_path" | awk '{print $1}')
gpu_count=$(rocminfo 2>/dev/null | awk '/^[[:space:]]*Name:[[:space:]]*gfx906[[:space:]]*$/{n++} END{print n+0}')
gpu_model=$(rocm-smi --showproductname 2>/dev/null | grep -m1 -E 'MI50|Instinct' || true)

echo "host=$(hostname)"
echo "host_os=$(uname -srmo)"
echo "cpu=$(lscpu 2>/dev/null | awk -F: '/Model name/{sub(/^[[:space:]]+/,"",$2); print $2; exit}' || true)"
echo "gpu_model=${gpu_model:-UNAVAILABLE}"
echo "gfx_target_count=$gpu_count"
echo "rocm_host=$(timeout 10 hipcc --version 2>/dev/null | awk '/HIP version/{print $0; exit}' || true)"
echo "kernel=$(uname -r)"
echo "pcie=$(lspci 2>/dev/null | grep -i -m1 -E 'Vega|MI50|AMD.*Display' || true)"
rocm-smi --showproductname --showmeminfo vram --showclocks --showpower --showtemp 2>/dev/null || true
echo "miinfer_sha=$(git -C "$repo_root" rev-parse HEAD)"
echo "image=$image"
echo "image_id=$image_id"
echo "image_base_digest=${image_base_digest:-UNAVAILABLE}"
echo "model=$model_path"
echo "model_sha=$model_sha"

[[ $gpu_count == 1 ]] || { echo 'FAIL exactly one gfx906 agent required'; exit 1; }
[[ $model_sha == "$expected_model_sha" ]] || { echo 'FAIL model SHA mismatch'; exit 1; }
[[ $image_base_digest == "$expected_image_digest" ]] || { echo "FAIL image base digest mismatch: $expected_image_digest"; exit 1; }

model_dir=$(dirname -- "$model_path")
$runtime run --rm --init \
    --device /dev/kfd --device /dev/dri \
    --security-opt=label=disable --group-add=keep-groups \
    -e MIINFER_EXPECTED_MODEL_SHA="$expected_model_sha" \
    -e MIINFER_EXPECTED_IMAGE_DIGEST="$expected_image_digest" \
    -v "$repo_root:/workspace/MIInfer:Z" \
    -v "$model_dir:/models:ro" \
    "$image" bash -lc '
set -euo pipefail
model=/models/"$(basename "$0")"
echo "container_rocm=$(hipcc --version | awk "/HIP version/{print; exit}")"
echo "container_arch=$(rocminfo | awk "/^[[:space:]]*Name:[[:space:]]*gfx906[[:space:]]*$/{n++} END{print n+0}")"
test "$(rocminfo | awk "/^[[:space:]]*Name:[[:space:]]*gfx906[[:space:]]*$/{n++} END{print n+0}")" = 1
test "$(sha256sum "$model" | awk "{print \$1}")" = "$MIINFER_EXPECTED_MODEL_SHA"
cmake -S . -B build/qualification-host -DCMAKE_BUILD_TYPE=Release -DMIINFER_ENABLE_HIP=OFF -DMIINFER_BUILD_TESTS=ON
cmake --build build/qualification-host --parallel 2
ctest --test-dir build/qualification-host --output-on-failure -L host-only
cmake -S . -B build/qualification-gpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx906 -DMIINFER_ENABLE_HIP=ON -DMIINFER_HIP_ARCHITECTURE=gfx906 -DMIINFER_TARGET_ARCH=gfx906 -DMIINFER_BUILD_TESTS=ON -DMIINFER_BUILD_BENCHMARKS=ON
cmake --build build/qualification-gpu --parallel 2
ctest --test-dir build/qualification-gpu --output-on-failure -R "hip-smoke|fp16-gemv-correctness|q4-q8-gemv-correctness"
build/qualification-gpu/miinfer-bench --warmup 2 --iterations 10 --elements 1048576
' "$model_path"
