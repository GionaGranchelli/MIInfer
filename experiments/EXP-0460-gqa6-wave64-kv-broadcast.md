# EXP-0460 — GQA6 Wave64 K/V broadcast, GPU2

## Objective

Reduce restored long-context Q8 GQA6 Stage 1 time on one MI50, then verify
that the candidate preserves the real coding-task result and persistent state.

## Current bottleneck and reference

EXP-0412 measured Stage 1 at 25.759 s of a 30.165 s 1,024-token restored
suffix (85.5%). EXP-0436 found 98.86% L2 hits, no reported LDS bank conflicts,
and high aggregate LDS-wait issue counts; it did not establish an HBM or wall-
time stall fraction. Packed-U128 GQA6 is the exact production-arithmetic
reference. Direct-global GQA6 avoids LDS and was faster than packed staging in
EXP-0457; corrected DPP reduced its restored suffix from 18.796 s to 17.938 s
in one exact pair (EXP-0456).

The current direct-global CTA has three Wave64s per KV head. In each wave, the
two 32-lane halves compute adjacent query heads but load the same eight-byte
K and V fragment independently. The candidate will have the lower half issue
each `uint2` K/V load once and broadcast its two 32-bit words to the upper
half with Wave64 `__shfl`. It will retain the two-half query/head mapping,
FP32 products, reduction tree, scales, per-key softmax/value update order,
split ranges, Stage 2, and state interface. No LDS staging or cross-token
reuse is introduced.

## Hypothesis and one variable

Avoiding half of the direct-global K/V load instructions per query-head pair
will reduce Stage 1 issue pressure enough to lower full suffix time by at
least 10%, without changing any arithmetic inputs or order. The only variable
is paired-half K/V operand broadcast versus duplicate per-half direct-global
loads.

## Gates

- **Compiler/resource:** gfx906; record VGPR, SGPR, spills, scratch, LDS,
  workgroup and wave size. Require zero spills and no unexpected scratch.
- **Dispatch:** graph cache must include the selector; require route traces in
  all 16 GQA layers during graph replay on GPU2/BDF `0000:88:00.0`.
- **Correctness:** matched restored 73,099-prefix / tokenizer-backed 1,024
  suffix. Require finite output, final position 74,123, exact logits and
  recurrent-state trace, and equal checked generated tokens. This is the
  arithmetic-preserving expectation; any difference is a reject.
- **Performance:** same-binary GPU2 control/candidate pair using direct-global
  GQA6, same model/session/prompt/image/environment and thermal guard. Keep
  only at >=10% lower suffix time. One pair is a kill screen.
- **Safety:** warn at 80 C and stop at 85 C. GPU0 remains unused/frozen.

## Amdahl and disposition

The maximum possible full-suffix improvement is bounded by Stage 1's measured
85.5% share. The removable operand-load fraction within Stage 1 is unknown;
therefore no larger gain is assumed. Stop if route, arithmetic exactness,
resource or >=10% timing gate fails. If it passes, run the defined coding task
before any cold 64K claim. Record and push either result.

## Result — rejected

GPU2/BDF `0000:88:00.0`, restored 73,099-token prefix plus 1,024-token suffix,
same binary, model, prompt, and graph-state path. The route trace confirmed all
16 layers used the direct-global control, then all 16 used Wave64 broadcast.
The single paired kill-screen measured 20.877 s control and 25.070 s candidate
(candidate/control 1.201; 4.193 s slower). This is one A/B pair, not a repeated
median or a stable estimate of regression size; it fails the predeclared 10%
keep gate and is rejected without further repeats.

Correctness passed: final position 74,123, exact logits, exact recurrent-state
trace, same checked generated tokens, finite output, and `accepted=1`. The
thermal guard remained below its 80 C warning threshold (reported peak 56 C).
No cold 64K run or coding-agent task was attempted. The feature remains opt-in
and must not be promoted.

The result falsifies the narrow hypothesis that removing duplicate per-half
K/V loads is sufficient to accelerate this restored suffix. It does not
identify the limiting instruction/resource. Next work should change the CTA
schedule to reuse each K/V tile across multiple query positions as well as GQA
heads, then compare that design against this exact control. Inspect generated
ISA and stage timing before deciding whether the tile is limited by shuffle,
softmax/dequantization, register occupancy, or split/merge overhead.
