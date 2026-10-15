# EXP-0461 — GQA6 dual-key Wave64 Stage1 schedule, GPU2

## Objective and active frontier

Test one materially different Stage1 attention schedule for the active M31
goal: improve long-context Qwen3.8-27B Q8-KV prefill and restored suffixes on
one MI50 while preserving the model, persistent-state, Split-K workspace,
HIP Graph, and Q8 cache contracts.

Base: `2227deb094d907bb19ec58667612ce1d8f388687` (`perf/m31-gqa6-pair-head-rewrite-gpu2`).
Candidate branch: `perf/m31-gqa6-dual-key-wave-gpu2`, isolated worktree
`/tmp/m31-gqa6-dual-key-wave-gpu2`.

## Bottleneck and controls

EXP-0412 attributed 85.5% of a restored 73,099-prefix/1,024-token suffix to
attention Stage1. Direct-global Q8 GQA6 is the comparison route; its current
matched GPU2 suffix control was 20.877 s in EXP-0460. The latest direct-global
route keeps the same six query heads per KV head, 32 splits, Q8 KV, and
production FP32 arithmetic. GPU0 is excluded.

The current CTA is `(64,3)`: three Wave64s per KV head, each wave's two
32-lane halves compute two query heads. Each half serially computes its own
key scores and updates its query's online softmax and PV state.

This schedule is distinct from:

- EXP-0459, which remapped all 64 lanes across dimensions to share operands
  for two query heads and changed the QK reduction.
- EXP-0460, which retained the query/head mapping and only broadcast K/V
  loads between the two halves.
- EXP-0360 / EXP-0413, which tested six-wave K/V or cross-token cooperative
  reuse through different CTA dataflows.
- EXP-0458, which combined four softmax/PV updates and failed numerical and
  performance gates by changing FP32 association.

No EXP-0450 report or matching commit was found in the available branch/ref
history during this pass.

## Candidate mapping and exact-order proof

Use a `(64,6)` CTA. Each Wave64 owns one Q head; its lower and upper 32-lane
halves each map the same 256 Q/K dimensions (`8` dimensions per lane) and
compute one of two adjacent QK dots independently. This exposes two key dots
per head concurrently while retaining the existing 32-lane XOR reduction,
Q8 scale placement, and FP32 arithmetic for each individual score.

For each pair `(p,p+1)`:

1. Lower half computes score `p`; upper half computes score `p+1`. Each half
   loads its own key/value Q8 fragment and corresponding scales directly from
   global memory. Q may be loaded in both halves.
2. Exchange the two scalar scores and the upper half's two packed V words to
   matching lower-half lanes using Wave64 shuffles. The lower half owns the
   running max/sum and output accumulator; the upper half does not update or
   store output state.
3. Lower-half lane 0 applies the existing online max/sum update to `p`,
   broadcasts its alpha/weight, then performs the existing PV update for
   `p`. It then repeats those same operations for `p+1`, using the exchanged
   score/V data. No score, softmax, denominator, or accumulator operations
   are combined or reordered.
4. On an odd-length split tail, the lower half executes the existing one-key
   path. Only the lower half writes partial max/sum/accumulator outputs.

The score and value shuffles add DS operations, but replace sequential QK
work for two keys with parallel half-wave work. No LDS or workspace is added;
Stage2 and state ownership remain unchanged.

## Feasibility/resource estimate before implementation

For the restored workload, Stage1 remains `1,024 × 32 × 4 = 131,072` CTAs.
Workgroup size rises from 192 to 384 threads (three to six Wave64s); query
heads per KV-head CTA remain six. Split-key intervals and the existing
`811,597,824 B` partial workspace do not change. Dynamic LDS and additional
global KV traffic are expected to remain zero. Baseline direct-global is
60 VGPR / 50 SGPR with no spills or private bytes. Candidate registers are
unknown until compiled; estimated additional live state is the second QK
score plus the exchanged V fragment, approximately 4–12 VGPR over baseline.
Reject if the object spills, adds private memory/LDS, emits excess peer
shuffles, or drops resident-wave capacity materially relative to baseline.

## Gates

- **Source/dispatch:** candidate is opt-in; graph identity includes the new
  selector; trace exactly 16 candidate dispatches in restored graph replay.
- **Compiler/resource:** pinned gfx906 object; record VGPR, SGPR, spills,
  private bytes, LDS, workgroup, and wave size. Inspect the QK/peer-shuffle
  sequence and prove the upper half has no output stores/state writes.
- **Correctness:** same GPU2 restored 73,099-token Q8 checkpoint plus the
  tokenizer-backed 1,024-token prompt suffix. Require final position 74,123,
  finite values, exact full logits and recurrent state, and matching checked
  tokens; any mismatch rejects this exact-order schedule.
- **Performance:** one same-binary direct-global control/candidate kill-screen
  on GPU2 with the same model, checkpoint, prompt, image, graph, and guard.
  Candidate must be at least 10% faster. One pair is a kill screen, not a
  repeated timing estimate. No cold-64K run unless this gate passes.
- **Safety:** GPU2 must be idle and verified as BDF `0000:88:00.0` before the
  run. Warn at 80 C; stop at 85 C. GPU0 remains unused.

## Result

Implementation, build/resource screen, and restored-session kill-screen are
pending.
