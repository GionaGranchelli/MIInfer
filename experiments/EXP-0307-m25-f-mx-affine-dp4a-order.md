# EXP-0307 — M25-F mx affine DP4A ordering

## Hypothesis

The Q4_K/Q5_K affine inner loop serializes all four low-half DP4A operations
before all four high-half operations. Matching mx's alternating low/high order
should expose independent gfx906 DP4A chains and reduce projection latency.

## Motivation

EXP-0305's profile left the FFN projections as the largest measured gap. The
pinned mx repacked kernel uses the alternating order in its affine path. This
is a one-loop source port with no layout, launch, allocation, or numerical
contract change.

## Baseline

The committed `mx_repacked_mmq_legacy_kernel` used:

```cpp
dot = lo.x*q0.x + lo.y*q0.y + lo.z*q0.z + lo.w*q0.w
    + hi.x*q1.x + hi.y*q1.y + hi.z*q1.z + hi.w*q1.w;
```

## Candidate

The affine Q4_K/Q5_K path now accumulates:

```cpp
dot = lo.x*q0.x;
dot = hi.x*q1.x;
dot = lo.y*q0.y;
dot = hi.y*q1.y;
dot = lo.z*q0.z;
dot = hi.z*q1.z;
dot = lo.w*q0.w;
dot = hi.w*q1.w;
```

This follows the pinned mx implementation at:

```text
/home/fedora-workstation/Development/mx-llama.cpp
commit 2e9d29fe736969160f17476ec6f0a6298cee6966
ggml/src/ggml-cuda/q8_repack/repack-kernels.cuh
```

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M GGUF
- model SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- ROCm system libraries, release build
- SCLK/MCLK `1606/1000 MHz`
- context capacity 1024
- exact 512-token repeated `The quick brown fox jumps over the lazy dog. `
  prompt, ending `The quick`
- Mx GDN and compact Mx projection paths enabled

## Benchmark

Primitive bakeoff:

```text
build/mi50-release/miinfer-m24-projection-bakeoff MODEL gate q4
build/mi50-release/miinfer-m24-projection-bakeoff MODEL ssm_out q5
build/mi50-release/miinfer-m24-projection-bakeoff MODEL down q6
```

End-to-end A/B used candidate/control order A3, B3, A4, B4 with
`--max-tokens 0`. Each case had a concurrent `rocm-smi --json` sampler; the
four captures contained 466 JSON samples total.

## Correctness

- Q4 contract max absolute error: `0.00000060`
- Q5 contract max absolute error: `0.00000167`
- Q6 contract max absolute error: `0.00000048`
- `qwen35-conv-batch-gpu` release CTest passed.
- All end-to-end cases returned status 0 and reported 512 prompt tokens.
- Candidate and control allocations were both `19,108,282,708 B`.

## Results

Primitive B512 medians:

| projection | baseline us | candidate us | candidate / baseline |
|---|---:|---:|---:|
| Q4 gate/up | 7852.154 | 6828.315 | 0.870x |
| Q5 SSM out | 3144.957 | 2771.197 | 0.881x |
| Q6 down | 8149.594 | 8132.794 | 0.998x |

The Q4/Q5 baseline values are the committed compact-Mx measurements from
EXP-0303. Q6 is unchanged because it uses the non-affine split-scale path.

Clock-qualified P512 A/B:

| order | path | prefill | generation |
|---|---|---:|---:|
| A3 | candidate | 2856.88 ms | 179.22 tok/s |
| B3 | control | 3058.81 ms | 167.39 tok/s |
| A4 | candidate | 2914.95 ms | 175.65 tok/s |
| B4 | control | 3061.78 ms | 167.22 tok/s |

Median candidate latency was `2885.92 ms` (`177.41 tok/s`) versus
`3060.30 ms` (`167.31 tok/s`) for control: `1.060x`, or `6.0%` faster, saving
`174.38 ms/P512`. All sampled SCLK/MCLK values were `1606/1000 MHz`; edge
temperature ranged from `33–41 C`.

## Interpretation

The instruction-order port improves the two dominant affine projection shapes
and carries the gain through end-to-end prefill. It does not affect recurrent
state or decode, and it leaves the project gate unmet: the best qualified
result is `177.41 tok/s` versus the `200 tok/s` target.

## Decision

**KEEP.** The interleaved affine DP4A order is now the default compact Mx
projection implementation.

## Follow-up

Profile the remaining FFN projection and attention tails after this change;
the next candidate must target a measured cost rather than add another
unqualified MMQ scheduling variation.
