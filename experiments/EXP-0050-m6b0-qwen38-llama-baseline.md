# EXP-0050 — M6-B0 Qwen3.8-27B llama.cpp MI50 baseline

## Question

What is the reproducible upstream llama.cpp performance baseline for the
selected Qwen3.8-27B Q4_K_M artifact on the target MI50?

## Baseline

Upstream llama.cpp was built from commit
`c0bc8591e8815c63cb01dd3f051a8b0df02501c` with the HIP gfx906 backend. No
MIInfer execution code was changed or used for these measurements.

## Model and environment

```text
GPU:       AMD Instinct MI60 / MI50, gfx906, 32 GiB VRAM
model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
size:      17,106,775,008 bytes (15.93 GiB)
sha256:    7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
ROCm/HIP:  7.1.52802-9999 / Clang 20.0.0.rocm
CPU:       Intel Xeon E5-2680 v3 @ 2.50 GHz
build:     Release, c0bc859
policy:    stable_peak
MCLK:      1000 MHz throughout telemetry
SCLK:      1282–1725 MHz observed in telemetry
repetitions: 5
KV:        F16 K/V
GPU offload: 999 layers, single device
batch/ubatch: 2048 / 512
threads:   24
FlashAttention: on
```

The source checkout was `/home/fedora-workstation/llama.cpp`; the separate
HIP build was configured in
`/home/fedora-workstation/Development/mi50-artifacts/m6b0-llama-c0bc8591`.
The canonical raw capture is retained outside Git at:

```text
/home/fedora-workstation/Development/mi50-artifacts/m6b0-results/20260902T130121Z-1146031/
```

The runner captured environment state before/after and 1,812 active telemetry
samples. The MIInfer working tree was dirty only because the B0 runner was
being added; the llama.cpp source commit and benchmark binary were fixed
independently.

## Commands

Build:

```bash
cmake -S /home/fedora-workstation/llama.cpp \
  -B /home/fedora-workstation/Development/mi50-artifacts/m6b0-llama-c0bc8591 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_NO_VMM=ON -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_LTO=OFF -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build /home/fedora-workstation/Development/mi50-artifacts/m6b0-llama-c0bc8591 \
  --target llama-bench -j2
```

Run:

```bash
scripts/run-m6b0-llama-baseline.sh \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
```

The runner uses five serial repetitions and records both standard isolated
PP/TG rows and explicit combined P1+TG rows. The standard command is:

```text
llama-bench -m MODEL -r 5 -b 2048 -ub 512 -ngl 999 -t 24 \
  -ctk f16 -ctv f16 -fa on -p 512 -n 0,64,128,256
```

The combined context rows add `-pg 1,64 -pg 1,256 -pg 1,1024`.

## Results

### Standard isolated controls

| Workload | Median tok/s | Mean tok/s | Raw tok/s |
|---|---:|---:|---|
| PP512 | 196.5850 | 196.1652 | 194.5300, 196.5850, 196.6290, 196.6010, 196.4800 |
| TG64 | 22.4888 | 22.5036 | 22.4888, 22.5492, 22.5827, 22.4259, 22.4716 |
| TG128 | 22.2873 | 22.2653 | 22.4699, 22.3430, 22.2873, 22.1569, 22.0696 |
| TG256 | 21.9009 | 21.8845 | 22.0051, 21.9623, 21.9009, 21.8737, 21.6806 |

### Combined context controls

| Workload | Median tok/s | Mean tok/s | Raw tok/s |
|---|---:|---:|---|
| P1 + TG64 | 21.8576 | 21.8495 | 21.8576, 21.8618, 21.8712, 21.8501, 21.8067 |
| P1 + TG256 | 21.9405 | 21.9471 | 21.9572, 21.9397, 21.9405, 21.9763, 21.9218 |
| P1 + TG1024 | 21.9107 | 21.9152 | 21.9223, 21.9107, 21.9332, 21.9019, 21.9081 |

Peak telemetry VRAM usage was 17,544,048,640 bytes (16.339 GiB). Junction
temperature reached 102 C and reported socket graphics power reached 280 W;
these observations qualify the absolute result and remain in the raw capture.
The policy was stable_peak, but actual SCLK varied between observed DPM levels.

## Interpretation

The real Qwen3.8-27B Q4_K_M model loads and runs with full GPU offload on the
MI50. Upstream decode is approximately 22 tok/s across the tested isolated and
P1 context regimes, while PP512 is approximately 196.6 tok/s. The combined
decode curve is flat through TG1024 in this run.

This is the external performance control, not a claim that MIInfer is
competitive yet. The next step is to run the functioning MIInfer GPU path
under the same model, hardware policy, and workload definitions.

## Decision

**KEEP / M6-B0 complete.** The pinned upstream llama.cpp HIP/gfx906 baseline is
reproducible, the model is confirmed runnable on the MI50, and standard plus
context-sensitive raw measurements are retained.

## Follow-up

M6-B1 should establish the MIInfer Qwen3.8-27B GPU profile and compare its
whole-token performance, GPU utilization, VRAM, and context behavior directly
against these controls.
