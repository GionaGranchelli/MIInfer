# EXP-0454 — GQA6 four-key parallel QK with exact ordered PV

**Status:** REJECT — exact, but slower on the restored long-context screen

## Objective

Reduce restored long-context GQA6 Stage1 time by computing four independent
Q8 QK scores concurrently, while preserving the production dot reduction tree
and the sequential online-softmax/value-update order.

## Current full-model bottleneck

EXP-0412 attributes 25.759 s of a 30.165 s restored 1,024-token tile to
GQA6 Stage1 (85.4%). EXP-0436 measured high aggregate LDS-wait counters but
did not establish a wall-time stall fraction. EXP-0443 removed LDS staging
and improved the exact restored suffix by 7.8%. The intended control for this
screen was that direct-global Q8 path, but the session-pair harness did not
select it for the recorded control.

## Reference and difference

Intended control: production-arithmetic GQA6, direct-global Q8 K/V, 32 splits,
UNROLL=4, on `perf/m31-gqa6-cold64k-gpu0` base `3beb2bd4`.

Candidate: one query warp maps 32 lanes as four key groups of eight lanes.
Each lane computes the same four virtual original 8-element QK partials that
the production 32-lane mapping assigned to those original lanes. It locally
emulates the original XOR-16 then XOR-8 additions, performs physical XOR-4,
XOR-2, XOR-1 shuffles, then gathers each key score from lane `key*8` and
broadcasts the four scores. Existing output-lane ownership remains unchanged.
The existing four ordered score/online-softmax/value updates consume those
scores in the original order. No approximate arithmetic, alternate softmax,
KV format, split count, query tile, or K/V staging change is intended.

The intended per-four-key issue change is five QK reduction shuffles per key
(20 total) down to three physical shuffles shared by all four key groups;
score gathering remains four shuffles, and K-scale broadcast falls from four
to one. Query-fragment broadcasts add 32 shuffles once per CTA, outside the
per-key loop. This is a source-level estimate only; disassembly must confirm
the compiler keeps the lane schedule and the benchmark decides whether it
helps.

## Gates

- **Resource:** pinned ROCm 7.2.1 gfx906 object; inspect actual disassembly and
  metadata. Require zero spills/private bytes, no unexpected LDS, and no
  impossible VGPR occupancy. Reject before GPU execution otherwise.
- **Correctness:** first run the existing restored 73,099-token session plus
  1,024-token graph suffix on physical GPU2 BDF `0000:88:00.0`, Q8/Q8,
  32 splits. Require exact final logits and recurrent-state trace, same
  checked tokens, finite values, final position 74,123, 16/16 intended-route
  dispatches, and graph replay.
- **Performance:** one focused Stage1/full-tile comparison against direct
  global GQA6. Only if correct and faster, perform a second relevant
  long-suffix screen. Cold 64K is justified only by a material restored-suffix
  win and remaining GPU2/runtime budget.
- **Safety:** no GPU2 load until GPU0's clean pair completes; confirm GPU2 idle,
  selected BDF, request idleness, and thermal guard before the first run.
  Warn at 80 C; stop at 85 C. Do not stop or modify user services.

## Expected ceiling and risks

The theoretical Stage1 Amdahl ceiling is 85.4%, but this candidate changes
only how four QK scores are scheduled; it does not reduce arithmetic or
sequential PV work. The plausible gain is limited to score-latency overlap and
fewer physical wave shuffles. Main risks are increased per-thread Q/K live
state, poorer memory coalescing, compiler scalarization, and incorrect
reconstruction of the original reduction order. Any exactness or resource
failure rejects the candidate immediately.

## Result

The candidate compiled in the pinned ROCm 7.2.1 image
`localhost/miinfer-m31-rocm721-libdw@sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`.
The selected gfx906 specialization had 118 VGPR, 48 SGPR, zero VGPR/SGPR
spills, zero private bytes, and zero LDS bytes. The direct-global selector-off
sibling had 60 VGPR and 46 SGPR, also with no spills or LDS. The candidate
passed the hard spill/scratch gate but nearly doubled its vector register use.

One matched pair ran on physical GPU2, BDF `0000:88:00.0`, with the same
restored Q8 session prefix of 73,099 tokens and tokenizer-backed 1,024-token
suffix. Both used one graph replay tile and ended at position 74,123. **Erratum:
the control was packed-U128 GQA6, not direct-global GQA6.** The harness set
`MIINFER_M31_Q8_GQA6_PROD_ARITH=1`, `PACKED_LDS=1`, and `PACKED_LDS128=1`,
while leaving `DIRECT_GLOBAL=0`. The candidate toggled key-parallel4 on.

| Path | Restore | Suffix | Final logits/state/tokens |
| --- | ---: | ---: | --- |
| Packed-U128 GQA6 control | 2.411 s | 20.885 s | exact, finite; tokens 18369 / 198 |
| Key-parallel4 candidate | 2.443 s | 25.418 s | exact, finite; tokens 18369 / 198 |

This single sequential pair makes the candidate 21.7% slower than packed-U128
(control / candidate = 0.822x). It is not a direct-global comparison or a
repeatable performance estimate. It still rejects the candidate against the
control actually executed, and measured exactness confirms the arithmetic
contract for this one restored suffix.

Route-trace logging was not enabled, so no explicit 16/16 dispatch count is
claimed. The harness set key-parallel4 off for control and on for the
candidate; both report graph replay. The raw log, thermal telemetry, input
hashes, and exit status are on Machinist at
`/home/machinist/m31-attn-rewrite-gpu2/results/EXP-0454-gpu2-keyparallel4/`.
Peak junction temperature was 59 C; the 80 C warning and 85 C stop guard did
not act. The container exited 0 and GPU2 returned to zero KFD processes.

## Decision

**REJECT.** Revert the key-parallel4 source and harness changes. Keep this
report as the negative result, with the comparator corrected above. The
dataflow reduced the intended QK shuffle work in source but doubled VGPR use
and did not offset that cost against packed-U128 in the restored-suffix run.
The next candidate must compare against direct-global and enable explicit
route tracing before its performance result is interpreted.
