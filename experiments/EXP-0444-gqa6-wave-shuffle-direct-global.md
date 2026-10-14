# EXP-0444 — GQA6 wave-shared direct-global Q8 operands (rejected)

**Base:** `8b90661` (direct-global GQA6 route)  
**Branch:** `perf/m31-gqa6-wave-shuffle-gpu2`  
**Target:** Machinist MI50 GPU2, BDF `0000:88:00.0`  
**Disposition:** Rejected at the compile/ISA gate. No GPU run was made.

## Hypothesis and candidate

Both 32-lane query halves in a Wave64 use the same KV head. The candidate
changed only operand delivery: lower-half lanes loaded each packed Q8 K/V
`uint2`, then two 32-bit shuffles per operand delivered the same bits to the
upper half. Lane 0 loaded each scale and broadcast it across the Wave64.
Head mapping, split ranges, key order, causal bounds, FP32 QK and softmax
updates, and output accumulation were unchanged. The route was opt-in and
had separate HIP Graph cache identity.

The maximum intended benefit was eliminating half the duplicate active-lane
K/V loads and scale loads in Stage1. Hot L2 and shuffle/predicate cost made the
actual gain uncertain; no speedup was assumed.

## Compile and ISA gate

The candidate and full-model harness compiled successfully for gfx906 using
the pinned local image `localhost/miinfer-m31-rocm721-libdw:20261014T094012Z`
(image ID `f5f33d8e8efbe77f0a08335140f9c9a162202067189499be9dbddd2eb214afa4`),
HIP `7.2.53211-e1a6bc5663`, AMD clang 22.0.0 ROCm 7.2.1. CMake required
explicit HIP/hipBLAS package paths. A first source-level lower-half guard
compiled, but ISA showed the compiler had kept the Q8 loads unconditional. One
bounded retry added an empty volatile memory-clobber marker inside each
lower-half load branch; it also compiled, but did not change the load masking.

| Kernel route | VGPR | SGPR | Spills | Private bytes | Fixed LDS | Wave |
|---|---:|---:|---:|---:|---:|---:|
| Direct-global control | 60 | 50 | 0 | 0 | 0 | 64 |
| Wave-shuffle candidate, final compile | 60 | 54 | 0 | 0 | 0 | 64 |

In candidate ISA, the eight unrolled Q8 K/V `global_load_dwordx2` instructions
are at PCs `0xBE024`–`0xBE0D4`. The first `s_and_saveexec_b64` appears at
`0xBE0DC`, after those operand loads, and masks the scale loads. Thus the
upper half still issues the duplicate global loads. The candidate also adds
`ds_bpermute_b32` instructions. The compiler-visible load-sharing requirement
failed, so timing and correctness gates were not entered.

## Evidence and disposition

The source snapshot, both build attempts, final code object, AMDHSA metadata,
and full disassembly are on Machinist at:

`/home/machinist/m31-gqa6-longctx-results/EXP-0444-wave-shuffle-gpu2/`

Final source patch SHA-256:
`9585234b387441d61b0200e7644068b09ab537b8e181510b52cc6fb935255c68`.
Candidate binary SHA-256:
`9b192199598c33d63649514175704ffb3fe118e2a7f2a82db94c02bf5418ca2e`.
HIP object SHA-256:
`f11bcf2e8e61e222294d52a8e339ecc6cff5f4ec127f00e37069f7156b96263d`.

The candidate kernel and route changes were reverted from this branch. This
report is the only intended commit; production defaults were never changed.
