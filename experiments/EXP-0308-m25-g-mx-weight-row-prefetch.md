# EXP-0308 — M25-G mx weight-row prefetch

## Hypothesis

The pinned mx MMQ kernel keeps the current weight row in registers and loads
the next row from LDS before consuming it. Applying that row prefetch to the
MIInfer compact Mx kernel should reduce repeated weight-LDS reads.

## Motivation

EXP-0307 kept mx's affine DP4A ordering and left the FFN projections as the
largest remaining family. The row-prefetch schedule is the next small,
directly portable difference in the pinned `mmq_gemm_repacked_impl`.

## Baseline

The committed Mx legacy kernel, including the kept interleaved affine DP4A
ordering from EXP-0307:

| projection | B512 us |
|---|---:|
| Q4 gate/up | 6828.315 |
| Q5 SSM out | 2771.197 |
| Q6 down | 8132.794 |

## Candidate

Two uncommitted variants were tested:

1. Prefetch the current/next weight `lo`, `hi`, and scale slot for every
   quantization type.
2. Keep those extra registers only for the Q6 split-scale instantiation.

The weight layout, input staging, launch geometry, and output contract were
unchanged. Both variants were reverted after the primitive gate.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M GGUF
- model SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- ROCm system libraries, release build
- pinned mx source: `/home/fedora-workstation/Development/mx-llama.cpp`
- pinned mx commit: `2e9d29fe736969160f17476ec6f0a6298cee6966`

## Benchmark

```text
build/mi50-release/miinfer-m24-projection-bakeoff MODEL gate q4
build/mi50-release/miinfer-m24-projection-bakeoff MODEL ssm_out q5
build/mi50-release/miinfer-m24-projection-bakeoff MODEL down q6
```

Each result is the bakeoff's median after warmup and repeated iterations.
The CPU contract check passed for all shapes, with the established maximum
errors `0.00000060`, `0.00000167`, and `0.00000048` respectively.

## Results

| projection | baseline us | all-type prefetch us | Q6-only prefetch us |
|---|---:|---:|---:|
| Q4 gate/up | 6828.315 | 6882.874 | 6897.913 |
| Q5 SSM out | 2771.197 | 2732.317 | 2765.597 |
| Q6 down | 8132.794 | 7830.356 | 7848.633 |

The all-type variant regressed the dominant Q4 shape by `0.8%`; the
Q6-only variant regressed it by `1.0%`. The Q6 improvement was not enough to
justify the affine-path register cost, and no end-to-end run was warranted.

## Profiling

No PMC capture was taken. The shape split is consistent with added live
weight state increasing register pressure in the affine instantiations.

## Interpretation

The source schedule does not transfer as a general optimization to this
kernel. A Q6-specific implementation would require duplicating the hot
compute loop or another abstraction, which is not justified by the measured
end-to-end opportunity.

## Decision

**REJECT.** Revert both row-prefetch variants and retain EXP-0307's compact
Mx kernel as the default.

## Follow-up

Do not revisit row prefetch without compiler register/VGPR evidence or a
larger Q6-dominated workload.
