# EXP-0093 — M6-B1 Qwen3.8-27B native GPU generation baseline

## Question

What steady-state token-generation throughput does the complete native
Qwen3.8-27B MIInfer GPU path achieve on the MI50, and how does it compare with
the pinned llama.cpp baseline?

## Baseline and candidate

The MIInfer candidate is the production-shaped native qwen35 executor from
M6-A28: GPU Q4_K embedding, all 64 hybrid layers, final norm, Q8_K/Q6_K LM
head, GPU argmax, and only the selected token ID copied to the host.

The external baseline is llama.cpp commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c`
using its existing gfx906 HIP build and the same local model artifact.

## Environment

```text
GPU:          AMD Instinct MI50 / gfx906 (reported as MI60 / MI50)
VRAM:         32 GiB
Model:        /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Model bytes:  17,106,775,008
Quantization: Q4_K_M
Clock policy: stable_peak
ROCm:         7.1.52802-9999
Compiler:     Clang 20.0.0.rocm
MIInfer:      build/mi50-release
llama.cpp:    c0bc859
KV:           F16
```

MIInfer telemetry was sampled every 250 ms during TG64 and TG128. The sampled
state reached 1725 MHz SCLK and 1000 MHz MCLK; junction temperature peaked at
90 C for TG64 and 92 C for TG128. llama.cpp telemetry reached the same maximum
clock levels and peaked at 90 C. These runs were not voltage or OverDrive
experiments.

## Commands

MIInfer:

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --generate16
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --generate64
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --generate128
```

llama.cpp:

```bash
scripts/run-m6b0-llama-baseline.sh \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
```

The raw llama.cpp result directory is outside the repository at:
`/home/fedora-workstation/Development/mi50-artifacts/m6b0-results/20260904T211356Z-3185440`.
MIInfer telemetry is at `/tmp/m6b1-qwen35-generate64-telemetry.jsonl` and
`/tmp/m6b1-qwen35-generate128-telemetry.jsonl`.

## Results

MIInfer runs execute two reset/replay passes; timing covers only the token
loop, including GPU work, synchronization, argmax, and the 4-byte token-ID
readback.

| MIInfer run | First pass ms | First tok/s | Replay pass ms | Replay tok/s | Replay | Decode allocations |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| 16 tokens | 4726.10 | 3.3855 | 4728.82 | 3.3835 | PASS | 0 |
| 64 tokens | 18960.00 | 3.3755 | 18980.40 | 3.3719 | PASS | 0 |
| 128 tokens | 38098.80 | 3.3597 | 38168.30 | 3.3536 | PASS | 0 |

The 64/128-token steady-state averages are approximately **3.37/3.36
tok/s**. Device bytes after setup and peak device bytes were
`17,018,706,644`; no allocations occurred inside the decode loop.

Fresh llama.cpp medians from five repetitions:

| Workload | llama.cpp tok/s | llama.cpp ms/token |
| --- | ---: | ---: |
| PP512 | 196.2255 | 5.0962 |
| TG64 | 22.5665 | 44.3136 |
| TG128 | 22.5844 | 44.2784 |
| TG256 | 22.4795 | 44.4851 |
| P1 + TG64 | 22.1116 | 45.2255 |
| P1 + TG256 | 22.1165 | 45.2155 |
| P1 + TG1024 | 22.1364 | 45.1749 |

The primary TG64 ratio is approximately `22.5665 / 3.3737 = 6.69x` in
llama.cpp's favor. TG128 is approximately `6.73x` in llama.cpp's favor.

## Correctness and resource checks

* M6-A28 native generation replay: PASS for 16, 64, and 128 tokens.
* First/last generated tokens: `11/585`, `11/29517`, and `11/15016`.
* Combined state fingerprints were deterministic for every repeated run.
* Decode-loop allocations: `0`.
* Release CTest: **20/20**.
* No per-token recurrent/KV/logit host round-trip was observed; only the
  selected token ID is read back.
* MIInfer peak sampled VRAM: approximately `17.62 GB` from telemetry; the
  executor's tracked device allocation was `17,018,706,644` bytes.
* llama.cpp peak sampled VRAM: `17,543,536,640` bytes.

## Interpretation

Qwen3.8-27B text inference is operational on MIInfer and the native path is
deterministic, GPU-resident, and allocation-free during decode. It is not yet
performance-competitive: the current MIInfer TG64/TG128 throughput is about
15% of the pinned llama.cpp result.

The comparison is a bring-up baseline, not a final parity claim. MIInfer's
current executable uses the A28 fixture's single prompt token and a 128-token
cache-capacity harness, while llama-bench uses its benchmark prompt and
standard benchmark settings. Model, quantization, GPU, and clock policy are
matched; an exact workload-equivalent benchmark interface is still needed
before publishing a definitive M6 performance comparison.

## Decision

**MEASUREMENT-ONLY / BASELINE.** Native MIInfer generation passes its current
functional and resource checks, but the performance objective
(`MIInfer >= 1.05x llama.cpp`) is not met.

## Follow-up

M6-B2 should first build a workload-equivalent MIInfer benchmark/profile with
clean warmup and repeated TG64/TG128 measurements. Then optimize the largest
measured whole-token family; do not infer the next target from dispatch count
alone.
