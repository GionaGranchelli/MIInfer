# EXP-0424 — 128-bit packed GQA6 LDS staging

**Status:** EXACT; repeatable ~0.6% faster on the restored 73K-prefix suffix. Keep as an opt-in candidate; default unchanged.

## Hypothesis

The exact packed-U4 path copies K and V into LDS as `uint2` (64-bit) vectors.
Using `uint4` (128-bit) copies should reduce load/store instruction count while
leaving the byte layout, K/V contents, QK/V arithmetic, online-softmax order,
causal bounds, and Split-K partitions unchanged.

## Build and static checks

- Base pushed revision: `c0887d8` on `perf/m31-gqa6-longctx`.
- Candidate binary SHA-256: `a3c5bec5cc270824041eb2b51a06da736d950b0ffc873756c96239fd2ffb9673`.
- Image: `localhost/miinfer-m31-rocm721-libdw@sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`.
- The complete `miinfer-m31-fp16-full-model` target built successfully.
- The packed128 specialization reports 71 VGPR, 46 SGPR, 0 VGPR/SGPR spills,
  0 bytes of scratch, and 3 waves/SIMD. This matches the packed-U4 uint2
  specialization's 71 VGPR and 3 waves/SIMD.
- Extracted gfx906 code object confirms candidate `global_load_dwordx4` and
  `ds_write_b128`; the packed-U4 uint2 specialization uses `global_load_dwordx2`
  and `ds_write_b64` for its staging path.

## Workload and method

The pinned image and binary ran on both MI50s concurrently. On each card, the
same process alternated five packed-U4 uint2 controls and five packed-U4 uint4
candidates (control then candidate for every sample). Each path restored the
same Q8/Q8 checkpoint at 73,099 tokens and processed 1,024 tokenizer-backed
tokens from the same prompt file with HIP Graph replay (2 suffix tiles).

The measured time is suffix prefill only. Restore was separately reported at
about 2.4 seconds per run. Each card compared against itself; the two cards are
not pooled into one timing series.

## Results

| GPU / BDF | Packed-U4 uint2 median (range) | Packed-U4 uint4 median (range) | Speedup | Paired gain range |
| --- | ---: | ---: | ---: | ---: |
| GPU0 / `0000:04:00.0` | 20.541 s (20.516–20.544) | 20.419 s (20.413–20.425) | 1.0060x | 0.103–0.128 s |
| GPU2 / `0000:88:00.0` | 22.427 s (22.421–22.442) | 22.301 s (22.282–22.313) | 1.0057x | 0.108–0.156 s |

All five pairs on both cards passed full-logit equality, recurrent-state-trace
equality, greedy-token equality, and finite-value checks. Each run restored
73,099 tokens and reached position 74,123. In the first pair, route traces
confirmed 16 graph-attention layers dispatched packed-U4 uint2 for control and
16 dispatched packed-U4 uint4 for candidate on each card. Both containers and
thermal monitors exited successfully. Peak junction temperatures were 68 C on
GPU0 and 60 C on GPU2, below the 80 C warning threshold.

## Decision and limits

Keep the route opt-in for cumulative performance work. The improvement was
small but consistent across both cards and larger than the observed run ranges.
It does not isolate the fraction of Stage1 spent in staging and does not prove
a cold-prefill gain. No 64K cold run was repeated; the existing exact packed-U4
64K result remains 802.708 seconds, still far above the 180-second target.
Do not change the default or claim a 64K speedup from this suffix result.

Raw logs, temperatures, and guard records are retained on Machinist under:

- `/home/machinist/m31-gqa6-longctx-results/EXP-0424-u128-pair-20261014T184119Z/`
- `/home/machinist/m31-gqa6-longctx-results/EXP-0424-u128-repeat4-20261014T1845Z/`
