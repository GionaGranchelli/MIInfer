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
  nonempty 48-layer recurrent trace, finite logits/state, max-absolute error
  at most 0.01, cosine at least 0.9999 for logits and state, and matching
  first/second tokens. Report exact logits/state equality separately as a
  diagnostic; a numerical pass with changed exact outputs still requires the
  deterministic real coding-task behavior gate before any cold-64K run.
- **Performance:** one same-binary direct-global control/candidate kill-screen
  on GPU2 with the same model, checkpoint, prompt, image, graph, and guard.
  Candidate must be at least 10% faster. One pair is a kill screen, not a
  repeated timing estimate. No cold-64K run unless this gate passes.
- **Safety:** GPU2 must be idle and verified as BDF `0000:88:00.0` before the
  run. Warn at 80 C; stop at 85 C. GPU0 remains unused.

## Result

Implementation and pair-harness changes are pushed in `7b4088e` and `8dbae74`.
The full target compiled in the pinned ROCm 7.2.1 image. The final benchmark
binary SHA-256 is
`bb6ce430bcdbb8dac962f46e3e744778cd7d883040e06ebffad7aacd1fbcbd0f`; the
candidate gfx906 code object SHA-256 is
`3e01f6055a9aac9c402ed23459f7303f739b8067d61ecac3c1fcc26f5603ce0d`.

The candidate kernel has maximum workgroup size 384 and wave size 64, with
46 VGPR and 42 SGPR. It uses no private segment, spills, or LDS. The direct-
global control uses 60 VGPR and 50 SGPR, also with no spills/private segment/
LDS. The candidate object has 17 static `ds_bpermute_b32` and 4
`v_exp_f32` instructions; the control has 50 and 10 respectively. The static
counts are not dynamic instruction counts, but show that the cross-half
exchanges add instructions to a path that already performs four keys per
iteration.

One same-GPU2 restored-session kill-screen ran on BDF `0000:88:00.0` in the
pinned image digest
`sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`.
It restored the 73,099-token checkpoint and processed the tokenizer-backed
1,024-token suffix, ending at position 74,123 with one graph replay tile.
The control traced direct-global in 16/16 attention layers; the candidate
traced dual-key Wave64 in 16/16 layers. The prompt SHA-256 is
`fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`, and the
session SHA-256 is
`bbbe9543d7ccbb740cc3db1cda7bc959e8e95926587a31b9cb8e3510c557de96`.

Correctness passed both the strict numerical screen and exact diagnostic:
logits and recurrent-state max error were 0, cosine was 1.0, all 48 recurrent
layers were observed, outputs were finite, top two generated tokens matched,
and both runs ended at the expected position. Restore took 2.889 s for control
and 2.873 s for candidate. Suffix time was 20.882 s for control and 43.072 s
for candidate, a single-sample speed ratio of 0.485. This fails the predeclared
10% improvement gate decisively; no cold-64K run is justified. The run is a
kill-screen, not a stable timing estimate. Thermal telemetry peaked at 59 C,
below the 80 C warning threshold, and the run exited successfully.

**Decision:** reject dual-key Wave64 as a performance candidate. Keep the
selector disabled by default and retain this report as negative evidence; do
not run another timing sample or cold-64K test for this schedule.

Raw run artifacts are under
`/home/machinist/m31-gqa6-longctx-results/EXP-0461-dual-key-wave64-gpu2/pair-1024/`.
The runner was the built `miinfer-m31-fp16-full-model` with
`ROCR_VISIBLE_DEVICES=2`, `--q8-gqa6-dual-key-wave64 --tokens 1024`, and the
artifact `session.miinfer` plus `prompt.txt`; the pair harness selected
direct-global for control and dual-key for candidate. Model SHA-256 was
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`. The
thermal guard sampled every three seconds, warned at 80 C and stopped at 85 C;
maximum observed junction temperature was 59 C, and the process exited 0.

The measured result is consistent with, but does not prove, a schedule-cost
hypothesis: each CTA grows from three to six Wave64s for the same six query
heads, while peer-half exchanges add score and value shuffles. The design also
keeps independent direct-global K/V loads for the two key producers. These
costs may outweigh concurrent QK work or reduce resident CTA capacity. No
profiler attribution was collected, so these remain hypotheses rather than
measured causes.
