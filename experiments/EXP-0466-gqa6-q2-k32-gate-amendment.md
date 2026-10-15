# EXP-0466 — Q2×K32 resource-gate amendment and GPU2 kill screen

## Amendment and claim

This amendment reopens the EXP-0465 Q2×K32 candidate for one bounded GPU2
screen. EXP-0465's predeclared 72-VGPR ceiling is withdrawn as an evidence-free
cutoff; its source, compile data, and initial rejection remain unchanged. The
replacement resource gate is zero spills and zero private scratch, with
register and LDS occupancy reported before runtime measurement.

The claim under test is: **on the same GPU, the restored 73,099+1,024 suffix
produces exact outputs and the candidate is at least 10% faster than
direct-global.** This is one A→B kill-screen, not a repeated timing estimate.
No cold 64K run is authorized unless both gates pass.

## Candidate, executable, and resources

The candidate source is the EXP-0465 commit `356d2134c5dbef8c866a3e64ecb96cb5a049696a`
on branch `perf/m31-gqa6-q2-k32-gpu2-exp0466`. It uses the already-built
pinned ROCm 7.2.1 executable on Machinist:

```text
/tmp/m31-gqa6-stage1-exp0465/build/mi50-release/miinfer-m31-fp16-full-model
SHA-256 04f0836ef225e4b7634e8a8e43ee70b86734051488c24e818d38189d33e0e7df
```

The device object is recorded in EXP-0465 with SHA-256
`3fcc28218f20f7a19fbdee154e81ae46ea8f38e10e5c677357decbbb661255d1`.
The paired-token kernel uses 82 VGPR, 55 SGPR, zero VGPR/SGPR spills, zero
private bytes, and 16,512 B dynamic LDS for a 192-thread, three-wave CTA.
gfx906's 64 KiB LDS limit permits at most three such CTAs per CU:
`3 × 16,512 = 49,536 B`, while `4 × 16,512 = 66,048 B` exceeds 65,536 B.
That LDS cap is at most nine resident Wave64s per CU. The 82-VGPR count is
reported alongside this cap; it is not by itself treated as a blocker. The
resource decision is **eligible for the one screen** because there are no
spills or private scratch and the requested LDS is within the architectural
limit.

The exact model artifact SHA-256 is
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`. The
tokenized suffix prompt SHA-256 is
`fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`. The
Q8/Q8 persistent session is
`/home/machinist/m31-gqa6-longctx-results/EXP-0461-dual-key-wave64-gpu2/pair-1024/session.miinfer`,
2,573,173,193 bytes, SHA-256
`bbbe9543d7ccbb740cc3db1cda7bc959e8e95926587a31b9cb8e3510c557de96`;
its saved prefix is 73,099 tokens.

## GPU and host preflight

The selected card is Machinist GPU2, gfx906, PCI BDF `0000:88:00.0`; GPU0 is
`0000:04:00.0` and is excluded. Immediately before the run GPU2 was at 27 C
junction with 1000 MHz SCLK and 1000 MHz MCLK, and `rocm-smi --showpids`
reported no KFD processes. Host load was 1.41 on 56 CPUs. The only transient
CPU-heavy process seen during preflight was this experiment's model-file hash
check; it was stopped before GPU use. No competing inference process was
present. Warn at 80 C and stop at 85 C.

## Timed work and pass conditions

The pair harness in `bench/m31_fp16_full_model.cpp` runs direct-global control
then paired-token candidate on the same binary. Each side restores the same
Q8/Q8 session and replays the same 1,024 tokenizer-backed suffix through one
HIP Graph tile. The reported suffix time comes from
`GenerateStats::suffix_prefill_ms`; restore and decode time are reported
separately and excluded. The harness checks the restored prefix count,
dispatched suffix count, graph replay tile count, graph use, generated tokens,
finite values, logits, and recurrent-state trace.

Accept only if both runs route Stage1 through the intended kernel in all 16
attention layers, both end at position 74,123, all logits and recurrent-state
values are exactly equal, all compared values are finite, no state is lost,
and candidate suffix time is at most 90% of control. Record top tokens,
route counts, position, graph state, temperature/clocks, and raw output. Any
failed condition rejects Q2×K32; do not retry this candidate or run cold 64K.

The source change removes repeated Q8 LDS staging/barriers by sharing each
K32 tile across two adjacent tokens and six query heads while preserving the
ordered online-softmax recurrence. EXP-0436's high LDS issue counters and
98.86% L2 hit rate suggest LDS issue pressure is worth testing, but counters
do not attribute wall time. This single pair can accept or reject the
candidate; it cannot establish repeatability or prove a limiter.

## Result

**REJECT — exact but slower.** One same-GPU2 A→B kill-screen completed with
exit code 0. The executable reported BDF `0000:88:00.0`, gfx906. Both sides
restored 73,099 tokens, dispatched 1,024 tokenizer-backed suffix tokens,
replayed one HIP Graph tile, and ended at position 74,123. Route traces show
16/16 direct-global control dispatches and 16/16 paired-token candidate
dispatches, all with `graph_state=1`.

The measured suffix prefill times were 20.869 s for direct-global and 24.848 s
for Q2×K32. The candidate took 3.979 s longer (19.1% more latency); the
control/candidate ratio was 0.840. This misses the required 10% suffix-time
improvement. The pair harness printed its candidate label as
`q8_gqa6_direct_global_session` because both selectors were set for the pair;
the 16/16 route trace confirms the candidate actually dispatched paired-token
Stage1. The final pair summary's first duration is control and second is
candidate.

Correctness passed exactly: logits and recurrent-state trace were byte-for-
byte equal, generated tokens matched (`18369`, then `198`), all compared
values were finite, and the harness accepted the pair. Peak observed GPU2
junction temperature was 57 C, below the warning threshold; the run ended at
32 C with SCLK 1000 MHz and MCLK 1000 MHz. GPU0 remained excluded. There was
no concurrent KFD process during preflight. The initial delay before the
control timing was session-file inspect/restore work; it is outside the
suffix timing. No cold 64K run or repeat was performed.

The one-pair result establishes that Q2×K32 is not a performance candidate
under this workload, but it does not prove which hardware resource caused the
regression. EXP-0436's high LDS issue counters remain a hypothesis about the
cost; they are not wall-time attribution. Raw output and telemetry are at
`/home/machinist/m31-gqa6-longctx-results/EXP-0466-q2-k32-gpu2/pair-1024/`.
