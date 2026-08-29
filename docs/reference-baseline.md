# External gfx906 Reference Baseline

MIInfer's primary external comparison is the separate
[`milpster/gfx906-llama-cpp`](https://github.com/milpster/gfx906-llama-cpp)
repository. It is a reference and benchmark opponent, not a MIInfer runtime
dependency.

## Pin

```text
Repository: https://github.com/milpster/gfx906-llama-cpp.git
Branch: master
Commit: 6e4ef6c1a553b8f61ad77bba18e9ca05aa677295
Commit date: 2026-08-28T21:48:37+02:00
Pinned on: 2026-08-29
```

The commit was obtained from `git ls-remote` and verified by a shallow checkout.
Do not benchmark a moving branch and label it as this baseline.

## Checkout

Keep the reference checkout outside this repository:

```bash
git clone https://github.com/milpster/gfx906-llama-cpp.git gfx906-reference
cd gfx906-reference
git checkout 6e4ef6c1a553b8f61ad77bba18e9ca05aa677295
```

## MI50 build starting point

The pinned repository's `BUILD-VEGA20.md` is the source for its Vega20 build
settings. For a single MI50 baseline, begin with the following reduced
configuration and record any toolchain-specific changes in the experiment
record:

```bash
cmake -S . -B build-mi50 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx906 \
  -DGPU_TARGETS=gfx906 \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_LTO=OFF \
  -DGGML_VULKAN=OFF
cmake --build build-mi50 --target llama-cli llama-bench
```

The upstream guide also describes ROCm 6.1, custom rocBLAS, Vulkan, and a
multi-GPU configuration. Those are not silently assumed here. The physical
MI50 run must record the actual ROCm/HIP compiler, rocBLAS availability,
environment variables, GPU visibility, and final build command before results
are considered comparable.

## Option validation on the available host

The exact configure command above was used against the pinned commit on
2026-08-29. Fedora provides the required development metadata as
`hipblas-devel`, `hipblas-common-devel`, `rocblas-devel`, and `rocsolver-devel`;
the primary Fedora config filename is `hipblas-config.cmake` (lowercase).
Those packages are installed system-wide and CMake resolved
`hipblas_DIR=/usr/lib64/cmake/hipblas` and
`hipblas-common_DIR=/usr/share/cmake/hipblas-common` without a custom prefix.

| Option | State at pinned commit | Evidence |
| --- | --- | --- |
| `GGML_HIP` | ACTIVE | consumed by `ggml/src/ggml-hip` |
| `AMDGPU_TARGETS` | ACTIVE | forwarded to `GPU_TARGETS` / HIP architecture |
| `GPU_TARGETS` | ACTIVE | forwarded to `CMAKE_HIP_ARCHITECTURES` |
| `GGML_HIP_GRAPHS` | ACTIVE | HIP compile definition |
| `GGML_HIP_NO_VMM` | ACTIVE | HIP compile definition |
| `GGML_CUDA_FORCE_MMQ` | ACTIVE | HIP/CUDA MMQ compile definition |
| `GGML_CUDA_FA_ALL_QUANTS` | ACTIVE | HIP flash-attention compile definition |
| `GGML_LTO` | ACTIVE | declared ggml build option, OFF |
| `GGML_VULKAN` | ACTIVE | declared backend option, OFF for single-MI50 run |

`GGML_HIP_MMQ_MFMA` defaults to ON in the pinned source, but its MFMA path is
gated to CDNA targets; the gfx906 build therefore has no applicable CDNA MFMA
path. This default is recorded rather than silently changing the canonical
command.

## Toolchain preflight record

```text
ROCm: 7.1.52802-9999 (hipcc; system package layout)
HIP compiler: clang 20.0.0.rocm
rocBLAS: `rocblas-devel-7.1.1-7.fc44`; `rocsolver-devel-7.1.1-4.fc44`
hipBLAS: `hipblas-devel-7.1.0-6.fc44`; `hipblas-common-devel-7.1.0-3.fc44`
GPU visibility: with gfx802 isolated, rocminfo reports one gfx906 MI50 device
Environment variables: no HIP/HSA/ROCm/GGML/GPU overrides were set
CMake build: `/home/fedora-workstation/Development/mi50-artifacts/milpster-gfx906-reference-6e4ef6c/build-mi50`
Build: PASS — `llama-cli` and `llama-bench` built for gfx906 with system package discovery
```

## Baseline status

```text
Commit pinned: YES
Configure/build: PASS — `llama-cli` and `llama-bench` target gfx906
GPU execution on physical MI50: PASS with gfx802 isolated
Model selected: Qwen/Qwen3-8B at b968826d9c46dd6066d109eabc6255188de91218
F16 GGUF: CREATED — SHA256 recorded in `docs/qwen3-8b-baseline.md`
Correctness comparison: PASS — single-turn GPU smoke generation completed
Performance measurements: PASS — canonical PP/TG records retained below
```

## Initial baseline

The benchmark used the pinned reference commit, Qwen3-8B F16 GGUF, `-ngl 99`,
batch size 2048, ubatch size 512, F16 KV cache, and five repetitions. Each
case was run separately with `-p`/`-n` so the tool's default PP512+TG128 rows
were not mixed into the canonical record.

```text
PP512: 563.339377 t/s average; stddev 4.411476; samples 555.586, 565.649, 564.875, 564.200, 566.387
PP2048: 550.789369 t/s average; stddev 2.638194; samples 547.321, 550.102, 554.387, 552.126, 550.011
TG128: 31.598717 t/s average; stddev 0.040570; samples 31.5324, 31.6025, 31.6348, 31.5957, 31.6281
```

Raw JSONL and diagnostics are retained under
`/home/fedora-workstation/Development/mi50-artifacts/qwen3-8b-reference-baseline-20260829/canonical/`.
