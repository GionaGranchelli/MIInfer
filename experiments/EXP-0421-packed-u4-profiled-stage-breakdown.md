# EXP-0421 — Packed U4 GQA6 late-context stage attribution

**Status:** PASS — profiler dispatches packed U4; Stage1 remains dominant
**Source:** `5a9c115afdfa65e0a53941c78a43f21ab086f78c` plus the profiler fix in
this change
**Target:** Qwen3.8-27B Q4_K_M, restored Q8/Q8 prefix 73,099, 1,024-token tile

## Discovery and fix

The first profile run exposed a profiler wiring gap: `forward_profiled()` chose
GQA6 but omitted the packed-LDS argument when launching it. The environment
selector therefore had no effect on profiled runs. `forward_profiled()` now
passes the existing packed-U4 selector into the same kernel launcher and emits
a route trace when requested. The ordinary forward route is unchanged.

The first run completed successfully as an un-packed GQA6 control. Its metadata
had `packed_lds=1`, but source inspection and the absence of a packed dispatch
trace establish that it ran the un-packed route. It is retained as a control,
not evidence for packed U4.

The packed profile binary SHA-256 is
`3beb5f14739d9bf174dee1f5abe0bddda10e096eeea970b677bbf9f5796d7c5c`.

## Measurement

Both profiles used the pinned ROCm 7.2.1 image, GPU2/BDF `0000:88:00.0`, the
same restored 73,099-token checkpoint, 1,024-token profile tile, Q8/Q8 cache,
32 splits, and no HIP Graph capture. The profile's final hidden state was
finite and all 16 attention layers reported GQA6. Packed dispatch tracing
reported all 16 layers in the candidate run.

| Measure | GQA6 U4 unpacked | GQA6 packed U4 |
| --- | ---: | ---: |
| Full profiled tile | 30,519.250 ms | 22,383.703 ms |
| GQA blocks | 27,108.388 ms | 18,980.285 ms |
| Causal attention | 26,141.788 ms | 18,014.752 ms |
| Stage1 | 26,120.017 ms | 17,993.273 ms |
| Stage2 | 20.602 ms | 20.436 ms |
| Attention share of tile | 85.657% | 80.482% |

Packed U4 cuts this profiled tile by 1.363x. Stage1 still consumes 80.5% of
the candidate tile; Stage2 is about 0.09%. This supports continuing to target
Stage1. The profile is one instrumented sample, not a production latency
distribution. The cold 64K candidate separately measured 802.708 s, exact
against the retained full-logit reference, so the remaining gap to the 180 s
target is 4.46x.

Artifacts on Machinist:

- Un-packed control profile: `/home/machinist/m31-gqa6-longctx-results/EXP-0421-packed-u4-profile-GPU2-20261014T1825Z/`
- Packed profile: `/home/machinist/m31-gqa6-longctx-results/EXP-0422-packed-u4-profile-GPU2-20261014T1830Z/`

## Next action

Screen a 64-split GQA6 Stage1 against the packed-U4 32-split control using the
same restored 73K session. Require exact full logits, recurrent-state trace,
generated tokens, and a material suffix win before any cold-prefill rerun.
