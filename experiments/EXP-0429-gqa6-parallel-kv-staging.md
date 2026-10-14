# EXP-0429 — Parallel K/V staging for packed GQA6

**Status:** REJECTED — exact, no material speed gain

## Hypothesis

The packed-U128 GQA6 loader stages K and V serially on the same 64 threads of a
192-thread CTA. Assigning K loads to one wave and V loads to the next doubles
the active loader waves while preserving the shared-memory layout and all
attention arithmetic. This may overlap independent memory operations.

## Implementation

Added opt-in `MIINFER_M31_Q8_GQA6_PARALLEL_KV_STAGE`. The original packed-U128
route remains the control. The candidate maps packed K vectors to CTA thread
IDs 0–63 and V vectors to IDs 64–127; both reach the existing barrier before
consumption. The session-pair benchmark toggles the selector between control
and candidate, and the HIP Graph cache key includes the selector. Arithmetic,
Split-K bounds, and shared-memory offsets are unchanged.

The candidate built with the pinned ROCm 7.2.1 image
`sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`.
It ran on the ROCm 7.14 image ID
`b8b8200afce88ce8d17bac793c37c50425a339b5408027d1ed8f19803a104173`, which
had just completed the EXP-0428 full-model smoke. Executable SHA-256:
`9d70c26920874ea085b7e2fbd54d15eb899c087ffd954a836bbdb7af5502b72f`.

## Matched restored-session screen

Each MI50 ran one packed-U128 control followed by the parallel-staging
candidate against the same copied Q8/Q8 checkpoint (73,099 tokens), model,
tokenized 1,024-token suffix, and capacity. The two cards ran concurrently;
comparisons are within each card. Dispatch logs showed 16 control and 16
candidate attention layers using graph state on each card. Both paths restored
the full prefix, replayed one suffix tile, reached position 74,123, and passed
finite logits, exact full-logit equality, exact recurrent-state trace, and
first/second token equality (18369, 198).

| MI50 / BDF | Control suffix | Candidate suffix | Speed ratio | Restore control / candidate | Peak junction |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU0 / `0000:04:00.0` | 20.519 s | 20.422 s | 1.005x | 3.695 / 2.463 s | 66 C |
| GPU2 / `0000:88:00.0` | 22.281 s | 22.278 s | 1.000x | 3.689 / 2.418 s | 58 C |

Both containers exited 0. Neither thermal guard warned or stopped a run. The
screen has one matched pair per card; the tiny latency differences are not a
qualified gain and fail the predeclared 10% continuation gate. Reject this
loader schedule and do not spend another run on a 4K suffix. Keep the existing
packed-U128 GQA6 baseline.

Raw logs and guards are on Machinist under
`/home/machinist/m31-gqa6-longctx-results/EXP-0429-parallel-kv-stage/gpu0-r1/`
and `gpu2-r1/`. This restored-suffix screen does not measure cold 64K prefill.
