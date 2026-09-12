# EXP-0325 — M25 beta/alpha batched FP32 hipBLAS port

## Hypothesis

The pinned oracle's two FP32 beta/alpha GEMMs might replace MIInfer's custom
dual projection in the wide recurrent prefill path and reduce the remaining
P512 gap.

## Baseline

Current `1e35ae9` uses one MIInfer-owned dual FP32 GEMM dispatch followed by
the existing `qwen35_prepare_beta_decay` kernel. The production shape is
`B512 x 48 x 5120`, with row-major weights and token-major input/output.

## Candidate

An opt-in implementation issued two FP32 `hipblasGemmEx` calls using the
oracle-compatible row-major/column-major reinterpretation, then reused the
same preparation kernel. No candidate code was retained.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
SCLK/HBM: 1606/1000 MHz (post-run state)
ROCm/driver: HIP 7.1.52802 / 7.1.10-200.fc44.x86_64
Compiler: clang 20.0.0.rocm
Kernel: 7.1.10-200.fc44.x86_64
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Model SHA-256: 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
Build: build/mi50-release
Benchmark: miinfer-m24-recurrent-layer-bakeoff MODEL control 512
Trace: rocprofv3 kernel trace, layer 0, B512
```

## Correctness

The candidate completed the production-shape one-layer execution without HIP
errors. The benchmark has no tensor comparison at this boundary, so this was
not promoted as a model-level correctness result.

## Results

Kernel traces, including the measured dispatches after warmup:

| Implementation | Projection dispatches | Preparation | Projection + preparation |
| --- | ---: | ---: | ---: |
| MIInfer dual | 992.798, 993.439 us | 5.920 us | ~999 us |
| two FP32 hipBLAS | 631.359 + 629.439 us | 4.640 us | ~1265 us |

The oracle-style two-launch projection is approximately 26.7% slower than
the native dual projection for this exact MI50 shape. Unprofiled whole-layer
samples were clock/state-sensitive and were not used as the decision metric.

## Interpretation

The oracle's two FP32 projection launches are a real contract match, but their
advantage does not transfer to this MI50 execution. MIInfer's dual projection
already reuses the normalized-input traversal and is the faster implementation
for the measured shape. The broad `prefill_prepare` profile family is not
evidence that this source-level port is a useful end-to-end optimization.

## Decision

**REJECT.** Keep the MIInfer dual beta/alpha projection and separate
preparation kernel. Do not reopen this path without a materially different
work-sharing or fused execution hypothesis.

## Follow-up

Return to the measured recurrent FFN Gate/Up and Down tail. Do not replace the
beta/alpha projection with the oracle's two-GEMM sequence.
