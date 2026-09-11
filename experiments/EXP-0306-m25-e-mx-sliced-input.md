# EXP-0306 — M25-E sliced activation staging for mx MMQ

## Hypothesis

The default compact mx MMQ kernel stages a complete 144-byte activation block
for every K tile. Matching mx's narrower q8-repack staging—only the quantized
words, scales, and Q6 sums needed by the tile—should reduce LDS traffic and
speed up the wide FFN projections.

## Motivation

After EXP-0305, the measured P512 gap is concentrated in the FFN projections.
The pinned mx implementation stages sliced activation fields rather than the
complete MIInfer `MxQ8_1MmqBlock`, making this the smallest isolated port to
test before considering a larger kernel rewrite.

## Baseline

The committed default `mx_repacked_mmq_legacy_kernel` used by the compact mx
projection path. The comparison shapes are the Qwen3.8 wide FFN projections:

```text
Q4 gate/up:  K=8192, N=6144, B=512
Q5 SSM out:  K=8192, N=2048, B=512
Q6 down:     K=6144, N=8192, B=512
```

Historical baseline timings are from the same primitive bakeoff in
EXP-0303, on the MI50 release build.

## Candidate

An uncommitted probe changed only the legacy kernel's activation staging to
separate shared planes for `MxSharedInputRow`, per-row scales, and Q6 sums.
Weights, output layout, accumulation, launch geometry, and the runtime path
were unchanged. The probe was reverted after the primitive result.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M GGUF
- model SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- ROCm system libraries, release build
- pinned mx source: `/home/fedora-workstation/Development/mx-llama.cpp`
- pinned mx commit: `2e9d29fe736969160f17476ec6f0a6298cee6966`

## Benchmark

The isolated projection bakeoff was run with:

```text
build/mi50-release/miinfer-m24-projection-bakeoff \
  --case gate_q4 --batch 512
build/mi50-release/miinfer-m24-projection-bakeoff \
  --case ssm_out_q5 --batch 512
build/mi50-release/miinfer-m24-projection-bakeoff \
  --case down_q6 --batch 512
```

## Correctness

The CPU contract checks passed for every probed shape. Maximum reported
absolute errors were `0.00000060` for Q4 gate/up, `0.00000167` for Q5 SSM
out, and `0.00000048` for Q6 down.

## Results

| projection | baseline us | sliced-input us | candidate / baseline |
|---|---:|---:|---:|
| Q4 gate/up | 7852.154 | 10029.271 | 1.277x |
| Q5 SSM out | 3144.957 | 3862.557 | 1.228x |
| Q6 down | 8149.594 | 7483.675 | 0.918x |

The dominant Q4 and Q5 shapes regressed despite the Q6 improvement. No
end-to-end P512 run was warranted after the primitive gate failed.

## Profiling

No PMC capture was taken. The A/B timings show that the additional LDS planes
and address calculations cost more than the avoided activation payload reads
on the dominant Q4/Q5 paths.

## Interpretation

The source-level staging pattern is not a free optimization when transplanted
into the current MIInfer kernel. Its benefit is shape-dependent and does not
justify replacing the faster default path.

## Decision

**REJECT.** The probe was reverted before integration. Retain the existing
full-block staging kernel as the default compact mx path.

## Follow-up

If this area is revisited, compare complete kernel instruction/register
footprints or the exact mx repack implementation; do not reapply sliced
staging alone.
