# MIInfer Current State

This document describes the **current implementation state** of MIInfer.

It is intentionally operational and should be updated whenever the active milestone, immediate target, or project constraints change.

For long-term direction, see:

* [`roadmap.md`](roadmap.md)
* [`architecture.md`](architecture.md)
* [`benchmarking.md`](benchmarking.md)
* [`hardware.md`](hardware.md)

---

# Current Phase

**M6-B44 — post-B41 profile and Q4 split-K rejection**

MIInfer now has a completed M6-A0 audit, a real M6-A1 fixture, validated
recurrent and full-attention layer executors, a four-layer hybrid composition,
and a complete 64-layer host forward for the selected
`Qwen3.8-27B-Q4_K_M.gguf` artifact. The stateful host harness now advances
the complete recurrent state and full-attention KV histories through positions
0–64, with external final-norm/logit checkpoints and greedy-token checks at
positions 0, 1, 2, 4, 8, 16, 32, and 64. M6-B0 establishes the external MI50
performance baseline. M6-A27 now composes all 64 qwen35 layers on the GPU
through the common executor. M6-A28 validates native autoregressive GPU
generation through 128 tokens; EXP-0093 records the first MIInfer throughput
baseline against the pinned llama.cpp control. EXP-0095 records a
production-selected Q5_K scale/min unpack-hoisting optimization with about
27% TG64/TG128 improvement, and EXP-0096 records a production-selected Q4_K
metadata-staging optimization with about 38% additional TG64/TG128
improvement and unchanged replay/resource checks. EXP-0099 records a
production-selected subgroup-structured Q5_K dot loop with about 26%
additional TG64/TG128 improvement and a 64.9% reduction in the profiled
recurrent Q5 output-projection stage. EXP-0100 records a further
production-selected paired-nibble Q5_K decoding change with about 2.7%
additional TG64/TG128 improvement.
EXP-0101 rejected a Wave64-local cached-attention score reduction: it was
neutral at TG64 and regressed TG128. EXP-0102 rejected Q6_K LM-head
index/scale hoisting: it preserved replay but stayed below the useful
end-to-end threshold and regressed TG128 slightly. The accepted production
path remains unchanged; the next target must be a materially different,
measured whole-token opportunity.
EXP-0103 tested reuse of the recurrent normalized Q8_K input between QKV and
gate projections. It preserved replay but improved TG64/TG128 by only 0.07% /
0.02%, below the useful threshold, so it was rejected and the separate
quantization path remains active.
EXP-0104 replaces the scalar Q4_K × Q8_K projection candidate with a
gfx906 Wave64 packed-dot4 path. It is production-selected after replay and
20/20 CTest: TG64/TG128 improve by 9.58%/9.99% to 8.41092/8.32747 tok/s.
EXP-0105 tests the analogous Q6_K × Q8_K packed-dot4 path. Native 16- and
64-token generation replay pass with zero decode allocations, but the P64
same-build A/B is neutral (9.44453 versus 9.44236 tok/s, +0.02%), so the
candidate is rejected and the scalar Q6_K path remains active. The next
optimization must come from a fresh post-EXP-0104 profile.
EXP-0106 tests a Q6_K×Q8_1 LM-head compatibility path matching the pinned
llama.cpp representation. Native replay passes, but P64 falls from 9.44236 to
8.37676 tok/s (-11.28%), so the scalar compatibility path is rejected and the
Q6_K×Q8_K production path remains active. The next LM-head attempt would need
the measured MMVQ access strategy rather than a representation-only port.
EXP-0107 implements that MMVQ-style candidate. After correcting its Q6 high
bit-plane shift, it passes 16- and 64-token replay, Release CTest 20/20, and
the complete temporary external observable fixture. Teacher-forced argmax
agreement is 62/64; the only differences are the already-adjudicated low-margin
P2/P12 choices, with late checkpoints and P64 completing successfully. Serial
P64 improves from 8.40767 to 8.68228 tok/s (+3.26%), while the final LM-head
profile falls from about 6.19 ms to 2.449 ms. The MMVQ path is now the
Qwen3.8 production default; `MIINFER_LM_Q8_1_MMVQ=0` retains the former
Q6_K×Q8_K control.
EXP-0108 adds a production-selected recurrent state-update path that avoids
the intermediate decayed-state global store. The complete external observable
run remains valid under the accepted A27 contract, Release CTest is 20/20, and
TG64/TG128 improve from 8.68093/8.60479 to 8.92048/8.82940 tok/s
(+2.76%/+2.61%). The representative state-update stage falls from 0.303039 to
0.211680 ms; `MIINFER_DELTA_NO_DECAY_STORE=0` retains the former kernel.
EXP-0109 adds a production-selected projection-input Q8_K reuse path. Recurrent
QKV/gate and FFN Gate/Up consumers, plus full-attention Q/K/V and FFN Gate/Up,
reuse one serialized quantized input where their normalized source is exactly
the same; `MIINFER_REUSE_PROJECTION_Q8=0` retains separate quantization. Native
replay, the 64-layer observable contract, and Release CTest remain valid. The
same-build stable-peak medians improve TG64 from 8.91219 to 8.98126 tok/s
(+0.78%) and TG128 from 8.82486 to 8.88730 tok/s (+0.71%).
EXP-0110 adds a production-selected gfx906 Q5_K×Q8_1 MMVQ-style path for the
recurrent `ssm_out` projection. It quantizes the FP32 input to canonical Q8_1
blocks and uses a 128-thread/output-row decomposition. Native replay and the
external observable contract remain valid, CTest is 20/20, and stable-peak
TG64/TG128 medians improve from 8.97493/8.89340 to 9.95072/9.80298 tok/s
(+10.87%/+10.23%). Set `MIINFER_Q5K_Q8_1_MMVQ=0` to select the former
Q5_K×Q8_K control.
EXP-0111 adds a production-selected gfx906 Q4_K×Q8_1 MMVQ-style path for
the Q4 FFN Down projection. It quantizes the FFN activation to canonical
Q8_1 blocks and uses a 128-thread/output-row decomposition. Native replay and
the external observable contract remain valid, CTest is 20/20, and stable-peak
TG64/TG128 medians improve from 9.94544/9.81133 to 10.1761/10.0237 tok/s
(+2.32%/+2.16%). Set `MIINFER_Q4K_Q8_1_MMVQ=0` to select the former
Q4_K×Q8_K control.
EXP-0112 adds a production-selected shared Q4_K×Q8_1 MMVQ-style path for
the Q4 FFN Gate/Up projections. It quantizes the normalized FFN input once
to Q8_1 and feeds both independent projections. Native replay and the
external observable contract remain valid, CTest is 20/20, and stable-peak
TG64/TG128 medians improve from 10.1707/10.0231 to 10.7750/10.6477 tok/s
(+5.94%/+6.23%). Set `MIINFER_Q4K_Q8_1_MMVQ_FFN_GATE_UP=0` to select the
former Q4_K×Q8_K control.
EXP-0113 tested the existing Q6_K×Q8_1 MMVQ primitive for recurrent QKV.
It failed native replay and produced immediate large external observable
errors, including approximately 96 hidden-state max error and a wrong P0
argmax. The candidate was removed; the B19 Q6_K×Q8_K QKV path remains
active. No performance result was accepted.
EXP-0114 adds a production-selected Q4_K×Q8_1 MMVQ path for the recurrent
`attn_gate` projection. Native replay and the external observable contract
remain valid, device usage is unchanged, and stable-peak TG64/TG128 medians
improve from 10.7745/10.6554 to 10.9673/10.8248 tok/s (+1.79%/+1.59%).
Set `MIINFER_Q4K_Q8_1_MMVQ_ATTN_GATE=0` to select the former Q4_K×Q8_K path.
EXP-0115 adds a production-selected packed-dot4 Q6_K×Q8_K path for recurrent
QKV. It keeps the existing Q6_K/Q8_K representation, passes native replay and
the external observable contract, and improves stable-peak TG64/TG128 medians
from 10.9603/10.8261 to 11.2805/11.1392 tok/s (+2.92%/+2.89%). Set
`MIINFER_Q6K_Q8K_DOT4_QKV=0` to select the scalar control.
EXP-0116 adds measurement-only stage attribution to a representative full-
attention layer at P64. The layer-3 profile reports 0.400319 ms for the
combined Q/K/V, head-normalization, RoPE, and KV-store region, 0.13840 ms for
cached attention, and about 0.852 ms for FFN work. No production behavior was
changed; the next experiment must split the combined preparation bucket before
selecting an optimization.
EXP-0117 splits that preparation bucket. At position 63, layer 3 Q projection
costs about 0.199 ms, Q split/head normalization about 0.085 ms, K projection
about 0.033 ms, V projection about 0.067 ms, and RoPE/KV store about 0.017 ms.
The next bounded candidate is an opt-in Q4_K×Q8_1 MMVQ Q projection using the
existing validated primitive; B23 remains the production default.
EXP-0118 rejects that Q-projection representation port. It failed native
replay and the external observable contract at P0, with final-hidden cosine
0.6435 and logits cosine 0.9465. The Q4_K×Q8_K production path is restored;
the next candidate should preserve Q8_K semantics or have a new external
contract justification.
EXP-0119 selects batched 256-thread head RMS normalization for full-attention
Q/K heads. Native replay, the external observable contract, poisoned reset,
and CTest 20/20 pass. Three-process stable-peak medians improve TG64/TG128
from 11.2861/11.1409 to 11.4420/11.2970 tok/s (+1.38%/+1.40%). Set
`MIINFER_BATCH_HEAD_RMS=0` to select the separate-launch control.
EXP-0120 refreshes the production profile after B27. Three instrumented
profiles measure 88.8735–89.0098 ms/token, with 48 recurrent layers
contributing 64.54208 ms aggregate and 16 full-attention layers contributing
21.01088 ms. Recurrent-layer support/projection work is the next measured
target; no production behavior changed.
EXP-0121 extends the existing profiler to recurrent layers 0, 1, and 2.
Their stage structure is stable: state update costs 0.21088–0.22960 ms and
FFN Gate/Up plus Down costs about 0.84–0.88 ms per recurrent layer. The next
uncleared recurrent-specific target is state-update attribution; no production
behavior changed.
EXP-0122 adds a transposed physical layout for the persistent DeltaNet state.
The logical state contract and recurrence remain unchanged; the isolated
state-update kernel improves from 202.979 to 118.153 us, and stable-peak native
TG64/TG128 improve from 11.4679/11.3067 to 11.8587/11.7159 tok/s. Native
generation through 128 tokens replays exactly with zero decode allocations,
the complete P64 external observable contract passes, and Release CTest is
20/20. `MIINFER_DELTA_TRANSPOSED_STATE=0` retains the row-major control.
See `experiments/EXP-0122-m6b30-transposed-deltanet-state.md`.
EXP-0123 tested two independent recurrent FFN Gate/Up output rows per
128-thread Q4_K×Q8_1 MMVQ workgroup. It improved sampled TG64/TG128 by
1.45%/1.49%, but introduced a new P6 teacher-forced decision change with only
4/5 top-5 overlap, outside the accepted A27 observable contract. The candidate
was removed; B30 remains the production default. See
`experiments/EXP-0123-m6b31-ffn-gate-up-two-row.md`.
EXP-0124 removes the intermediate decayed-state global store from the
production-selected transposed recurrent state-update path. Native generation
and the complete P64 external observable contract remain valid, Release CTest
is 20/20, and stable-peak TG64/TG128 improve from 11.8606/11.7073 to
12.0637/11.8983 tok/s (+1.71%/+1.63%). Set
`MIINFER_DELTA_TRANSPOSED_NO_DECAY_STORE=0` to select the B30 control. See
`experiments/EXP-0124-m6b32-transposed-no-decay-store.md`.
EXP-0125 refreshes the production profile after B32. Two position-63 profiles
measure 84.6009 and 84.6991 ms/token of GPU work, with 48 recurrent and 16
full-attention layers. Recurrent FFN Gate/Up plus Down remains the largest
repeated family at about 0.84–0.89 ms/layer; no production behavior changed.
See `experiments/EXP-0125-m6b33-post-b32-profile.md`.
EXP-0126 tested a fused SiLU-to-Q8_1 Down-input path that retained the FP16
rounding boundary. It passed native replay and the complete external P64
observable contract, but stable-peak TG64/TG128 improved only 0.09%/0.08%,
below the useful threshold. The candidate was removed; B32 remains the
production path. See `experiments/EXP-0126-m6b34-fused-silu-q8-1.md`.
EXP-0127 adds LDS activation reuse to the Q4_K×Q8_1 MMVQ path. Two
independent 128-thread output-row groups share one staged Q8_1 input tile in a
256-thread workgroup while retaining per-row arithmetic. Native 16/64/128-token
replay, the complete P64 external observable contract, and CTest 20/20 pass;
stable-peak TG64/TG128 improve from 12.0637/11.8983 to 12.4235/12.2563 tok/s
(+2.98%/+3.01%) with unchanged tracked device usage. Set
`MIINFER_Q4K_Q8_1_LDS_INPUT=0` for the B32 control. See
`experiments/EXP-0127-m6b35-q4k-q8-1-lds-input.md`.
EXP-0128 refreshes the profile after B35. Position-63 total GPU work is
82.1223 ms/token with a layer sum of 78.8699 ms, 48 recurrent layers, and 16
full-attention layers. LDS reuse reduces recurrent Gate/Up to about
0.36–0.38 ms/layer, while long-K Q4_K×Q8_1 FFN Down remains about
0.42–0.44 ms/layer. No production behavior changed. See
`experiments/EXP-0128-m6b36-post-b35-profile.md`.
M6-A8 provides a dedicated qwen35 model boundary and a real layer-0 RMSNorm
GPU fixture on gfx906. M6-A9 validates the real Q6_K output head through the
existing Q6_K×Q8_K GPU primitive, M6-A10 validates a real Q4_K×Q8_K attention
projection, M6-A11 composes RMSNorm, Q8_K quantization, and that projection,
and M6-A12 extends the path through real Q/K/V projections and K normalization,
and M6-A13 completes layer 3 through RoPE, persistent KV state, cached
attention, output projection, FFN, and residual output against external
checkpoints at positions 0–8. M6-A14 adds logical-byte fingerprints for L0–L2
recurrent state, L3 K/V state, and hidden outputs, plus poisoned reset/replay
checks. M6-A15 composes those layers through the complete layer-3 output
boundary and advances the stateful block through positions 0–16, with entry
and exit fingerprints for recurrent state, K/V cache, and hidden outputs and
external checks at positions 0, 1, 2, 4, 8, and 16. The host audit passes;
qwen35 recurrent HIP execution is now present for a complete layer-0 slice,
so M6-B1 remains deferred until full hybrid composition is complete.
M6-A16 validates a second independent hybrid block and M6-A17 composes the
full 64-layer host trunk with final logits. M6-A18 adds a persistent GPU
resident DeltaNet state update, M6-A19 adds GPU convolution history and Q/K
normalization, M6-A20 composes the complete layer-0 recurrent GPU path, and
M6-A21 composes recurrent GPU layers 0–2 with full-attention GPU layer 3.
M6-A22 extends the same GPU executor through positions 0, 1, 2, 4, 8, 16,
32, and 64, with bounded layer-output error, recurrent state-entry checks,
and logical recurrent/KV fingerprints at every checkpoint. The full qwen35 GPU
path remains absent, so M6-B1 remains deferred. M6-A23 composes layers 4–7
from the actual L0–L3 GPU output through P64 with independent recurrent state,
an independent L7 KV cache, and 48/48 recurrent state-entry checks. M6-A24
composes layers 0–7 through one common ordered executor through P64, with
bounded external-reference errors, entry/exit state fingerprints, poisoned
reset/replay, zero decode-loop allocations, and permanent cached-attention
determinism coverage. M6-A25 extends the same executor through layers 0–15,
including full-attention layers 11 and 15, through P64 with later recurrent and
K/V state validation and zero decode-loop allocations.
M6-A26 extends the same executor through layers 0–31 and eight full-attention
KV caches through P64. All layer outputs pass, but L30 recurrent-state
checkpoints remain in retest: the full P1–P64 scan reaches `0.110666` at P20
(index `86796`) and P64 reaches `0.0523846` at index `86909`, against the
current `0.05` state gate. A26 remains in retest rather than being marked
closed. EXP-0071 localizes the discrepancy as non-monotonic and already
present after P63; the strict gate remains unchanged.
EXP-0072 traces index `86909` through the recurrent update and shows the GPU
candidate equals the stored value at every sampled transition; the P63→P64
GPU value is `-0.300347` versus the external `-0.352732`. The moving maximum
is not a permanently corrupted cell, but the P20 outlier means the state
distribution is not cleared by the `0.05` gate. A26 remains in retest pending
an explicit external state-contract decision.
EXP-0073 replays the worst-case L30 P19→P20 update with external operands:
the external formula reaches max error `2.98e-8`, and the MIInfer GPU
recurrence reaches `5.96e-8` against the external P20 state. The recurrence
itself is therefore cleared; the remaining question is the upstream input or
checkpoint representation contract. Stop GPU state-mechanics debugging and
keep A26 in retest.
EXP-0074 captures the full L30 production operand set at P19→P20 and performs
one-at-a-time GPU substitutions. Replacing only `k_in` reduces state max error
from `0.0431299` to `0.00747788`; previous-state, q, value, beta, and decay
substitutions are largely neutral. The next trace target is the upstream L30
K-input boundary, not recurrence storage or update mechanics.
EXP-0075 traces that boundary through L30 input, attention RMSNorm, QKV
projection, convolution/SiLU, and K normalization. The first mismatch is
already at L30 input (`l_out-29`, max `0.0793692`); K-path error is `0.0197068`
after convolution/SiLU and `0.0180074` after K normalization. L30 K execution
is therefore not the first separation; the next target is L29 output provenance.
EXP-0076–EXP-0077 trace L29 through its recurrent output and gated path: the
recurrent output remains close, while head normalization amplifies small
differences and the gate projection also differs under the production input.
EXP-0078 replays that gate projection with the external normalized input and
reaches `1.90735e-6` max error against external `z-29`, clearing the gate
kernel. The remaining A26 issue is an upstream L29 hidden-state/contract
difference, not recurrent state storage or update mechanics.
EXP-0079 adjudicates the external state contract: internal recurrent-state
element identity remains diagnostic, while finite deterministic execution,
observable layer/output envelopes, and reset/replay behavior define M6
correctness. The strict `0.05` state check is unchanged for `--prefix32`; the
  new `--prefix32-external-contract` mode reports it diagnostically and permits
  composition to continue. M6-A27 adds layers 32–63 through the same executor,
  but the external-contract run first fails at L54/P1. EXP-0081 shows L54
  input already differs by `1.02168`, with recurrent output remaining close
  while its gated path and subsequent FFN amplify the difference. EXP-0082
  traces the matching L53 output and confirms that L54 inherits this discrepancy.
  A27 remains in RETEST pending the external-contract decision. EXP-0083
  replays L53 gating with external recurrent and `z-53` operands and matches
  the external output to `4.77e-7`; the L32–L53 scan shows gradual activation
  drift rather than a new gated-kernel or state-storage failure. EXP-0084
  runs the full 64-layer executor through final RMSNorm and the LM head
  without aborting on intermediate diagnostics. Final P64 logits remain
  highly aligned (cosine `0.999558`, relative RMS `0.029748`), but
  teacher-forced argmax agrees at only 63/64 positions: the deterministic P2
  decision is `1318` reference versus `1044` GPU. A27 therefore remains in
  RETEST; external observable correctness is not closed, and
  generation/performance benchmarking is still deferred.
  EXP-0085's P2 layer scan finds measurable drift at L0, growth through L1/L2,
  and bounded later relative RMS without a new L53/L54 discontinuity. The P2
  final-logit decision remains the only teacher-forced mismatch (63/64).
  EXP-0086 traces P2 operation boundaries through L0–L2: L0 recurrent output
  remains at roundoff while attention/residual reaches `0.00164509`; L1/L2
  QKV errors reach `0.0172018` and `0.0453629`. This confirms distributed
  early representation drift rather than a new late-layer defect. EXP-0087
  clears L0 gated output and residual arithmetic by shared-input replay, but
  isolates the same `0.00164509` error in the direct Q8_K → Q5_K `ssm_out`
  projection. A27 remains in RETEST pending this representation contract.

M0 is closed, M1 established the kernel laboratory, M2 passed its
gfx906-specific specialization gate with EXP-0009, and M3 is closed. The
minimal model/runtime scaffold is present; M4-C is complete and M5's measured
local optimization campaign is closed. M6-A has now been selected as the
reference-correct Qwen3.8-27B bring-up and performance phase.

---

# Current Hardware Target

```text id="f0sqdr"
AMD Instinct MI50 32GB
gfx906 / Vega20
Linux
single GPU
```

No other GPU architecture is currently supported.

---

# Current Project Status

## Completed

* project mission defined
* specialization strategy defined
* `AGENTS.md`
* `README.md`
* roadmap defined
* architecture boundaries defined
* benchmarking standard defined
* hardware target defined
* C++20/CMake gfx906 build presets
* host-only and GPU-required CTest infrastructure
* trivial HIP vector-add validation
* HIP-event microbenchmark and JSON output
* raw per-iteration benchmark samples
* environment capture and benchmark runner
* active-run GPU telemetry sampling
* EXP-0001 benchmark harness scaffold
* official Qwen3-8B dense-control revision and exact configuration record
* Qwen3-8B projection-shape record for EXP-0002
* EXP-0002 FP16 GEMV baseline scaffold
* EXP-0003 FP16 GEMV bottleneck characterization diagnostics

## Completed in Task 3

* physical MI50 execution with gfx802 isolated from KFD
* physical MI50 Debug and Release GPU validation
* EXP-0001 with five valid runs and active telemetry
* physical MI50 build and validation of the pinned external gfx906 reference
* reproducible Qwen3-8B F16 GGUF conversion and checksum
* initial llama.cpp PP/TG measurements
* EXP-0002 FP16 GEMV baseline implementation and five-run MI50 measurement
* EXP-0004 FP16 K-split K/V specialization, accepted after five-run measurement
* EXP-0005 Q4_0 × Q8_1 quantized GEMV baseline, accepted after five-run measurement
* EXP-0006 packed-dot ISA proof, correctness validation, and five-run comparison
* EXP-0007 zero-point-corrected Q4_0 × Q8_1 dot4 specialization, accepted after five-run measurement
* EXP-0008 direct comparison with the pinned gfx906 llama.cpp MMVQ path; the
  reference primitive is now measured
* EXP-0009 128-thread K/V geometry and Wave64 reduction comparison; accepted
  with `KEEP`, with M2 marked `GO`
* M3 minimal Qwen3-8B GGUF loader, GPU weight arena, and static plan; closed
  after pinned physical-MI50 acceptance
* M4-A5 independent four-position reference trace; host and MI50 Debug/Release
  stateful layer-0 comparisons pass
* M4-B initial full-depth host/GPU executor scaffold and independent
  single-token 36-layer reference fixture; acceptance remains open
* M4-B6 independent same-run external layer-6 trace and first-divergence
  host/MI50 diagnostic comparison; full-depth acceptance remains open
* M4-B8 canonical full-depth external oracle and exact-Q8 F16/F32 precision
  policy diagnostics
* M4-B9 terminal layer-35 external internal trace and first-divergence
  comparison; shared host/GPU FFN-tail numeric contract remains open;
  production precision remains unchanged
* M4-B10 host Gate/Up hybrid SwiGLU attribution; both projection errors are
  causal, with Gate the larger single-source contributor, while host SwiGLU
  arithmetic is exonerated
* M4-B11 Q8 contract and Gate/Up accumulation replay; Q8 lanes/scales and
  external-conditioned Gate/Up projection arithmetic match the pinned
  contract, shifting the remaining failure upstream to `ffn_norm`
* M4-B12 external-conditioned O/residual and RMSNorm replay; O/residual and
  all tested RMSNorm reductions are close to or exact with external inputs,
  shifting the remaining normal-path difference upstream to attention output
* M4-B13 attention RMSNorm/V/GQA replay; V and GQA are exonerated, while the
  external attention-output FP16 materialization reproduces the remaining
  layer-35 tail result
* M4-B14 production attention-output FP16 boundary; focused layer-35 host and
  full MI50 GPU gates pass, while host full-forward parity still first fails
  at layer 2
* M4-B15 sequential-composition diagnostic; full host forward is bitwise
  identical to its reconstructed layer chain, with the first inherited
  sequential divergence at layer 1 input and first strict failure at layer 2
* M4-B16 layer-0 28-checkpoint precision diagnostic; first mismatch is a
  small Q projection delta, the causal V/FFN drift is gradual, and an extra
  layer-output FP16 round-trip is rejected
* M4-B17 causal attention-output replay; external attention injection reduces
  host layer-0 error to `1.90735e-06` and MI50 error to `0.00282186`, proving
  the V/attention perturbation dominates while downstream GPU variance remains
* M4-B18 V precision/cause replay; F32->Q8Exact->F32 gives MI50 local V error
  `1.90735e-06`, but downstream materialization/quantization still leaves about
  `0.204956` layer-output error, so no production precision change is accepted
* M6-A0 Qwen3.8-27B Q4_K_M GGUF and hybrid architecture audit
* M6-A1 pinned llama.cpp external tensor/logit fixture
* M6-A2 Qwen3.8-27B projection compatibility audit
* M6-A3 single Gated DeltaNet layer bring-up
* M6-A4 single full-attention layer bring-up
* M6-A5 recurrent/full-attention four-layer hybrid block composition
* M6-A6 complete 64-layer host forward, final norm, and logits bring-up
* M6-A7 stateful 0–64 host generation/replay with recurrent-state and
  full-attention KV-history validation
* M6-B0 upstream llama.cpp Qwen3.8-27B Q4_K_M MI50 PP/TG and context baseline
* M6-B1 readiness audit; no MIInfer GPU profile claimed because the current
  HIP executor supports only the prior qwen3 model contract
* M6-A8 qwen35 model boundary and real MI50 layer-0 RMSNorm fixture; exact
  external fixture result with max absolute error `0`
* M6-A9 qwen35 Q6_K×Q8_K LM-head projection; max absolute logit error
  `3.8147e-06` and matching argmax
* M6-A10 qwen35 Q4_K×Q8_K attention projection; max absolute error
  `2.86102e-06`
* M6-A11 qwen35 layer-3 RMSNorm→Q8_K→Q4_K×Q8_K attention prefix; norm error
  `0` and projection error `2.86102e-06`
* M6-A12 qwen35 layer-3 Q/K/V projections with K normalization; Q error
  `2.86102e-06`, K error `9.53674e-07`, and V error `1.90735e-06`
* M6-A13 qwen35 layer-3 complete full-attention vertical slice through FFN
  and residual; positions 0–8 pass with max attention error `0.00292516`, max
  FFN error `0.00455475`, and max layer error `0.00516891`
* M6-A14 qwen35 state fingerprint/reset audit; 42 logical state records,
  clean/full/partial-poison/cross-sequence replay all pass exactly, with a
  diagnostic state plan of `9,830,400` bytes
* M6-A15 qwen35 layers 0–3 hybrid-block audit; stateful positions 0–16 pass
  through the complete layer-3 output boundary with entry/exit fingerprints
  for recurrent state, K/V cache, and hidden outputs; host-only, no GPU tok/s
  claim
* M6-A16 qwen35 layers 4–7 hybrid-block audit; the second block consumes the
  actual L0–L3 output and passes stateful positions 0–16 with recurrent/KV
  fingerprints; host-only, no GPU tok/s claim
* M6-A17 qwen35 composition ladder; 8/16/32/64-layer boundaries and final
  logits pass with approximately linear one-token host scaling; host-only, no
  GPU tok/s claim
* M6-A18 qwen35 DeltaNet state-update HIP primitive; persistent state and
  recurrent output pass positions 0→1 against the external fixture
* M6-A19 qwen35 recurrent convolution/SiLU/split HIP primitive; persistent
  circular history and Q/K normalization pass positions 0→1
* M6-A20 qwen35 recurrent projections, beta/alpha preparation, persistent
  state composition, recurrent output projection, and complete layer-0 FFN
  path pass positions 0→1; see
  `experiments/EXP-0064-m6a20-qwen35-recurrent-layer-gpu.md`
* M6-A21 qwen35 GPU layers 0–3 compose through the complete layer-3 boundary
  at positions 0→1; see
  `experiments/EXP-0065-m6a21-qwen35-gpu-hybrid-block.md`
* M6-A22 qwen35 GPU layers 0–3 remain externally bounded through P64 with
  recurrent state-entry checks and logical state/KV fingerprints; see
  `experiments/EXP-0066-m6a22-qwen35-gpu-hybrid-position-audit.md`
* M6-A23 qwen35 GPU layers 4–7 compose through the same executor from the
  actual L0–L3 output through P64; see
  `experiments/EXP-0067-m6a23-qwen35-gpu-hybrid-block-4-7.md`
* M6-A24 qwen35 GPU layers 0–7 compose through one common executor through
  P64 with poisoned reset/replay and cached-attention determinism coverage; see
  `experiments/EXP-0068-m6a24-qwen35-eight-layer-gpu-prefix.md`
* M6-A25 qwen35 GPU layers 0–15 compose through the same common executor
  through P64; see
  `experiments/EXP-0069-m6a25-qwen35-sixteen-layer-gpu-prefix.md`
* M4-B19 attention-to-Q8 boundary replay; external attention quantizes
  bitwise-identically to the host contract, while the external-attention GPU
  control itself retains `0.204956` layer-35 error, identifying a downstream
  GPU arithmetic floor rather than a V-induced Q8 threshold change
* M4-B20 identical-input arithmetic characterization; exact-Q8 metadata
  matches for O/Gate/Up/Down, and F32-output GPU projections remain within
  `0.000244141` while F16-output controls reach `1.79517` on Down; full-layer
  parity remains open
* M4-B21 full-model F32-output policy trial; output-only F32 fails through
  depth (`21.8325` at layer 35), while the combined diagnostic policy still
  reaches `12.5605` at layer 35 and `0.131546` logits error; no policy accepted
* M4-B22 independent CPU/offloaded-gfx906 trace comparison; external backends
  differ by `121.013` at layer 35 while both select argmax `8`; MIInfer remains
  closer to the CPU trace than the external GPU trace, so no new precision
  policy is accepted
* M4-B23 external backend contract characterization; the pinned CPU path uses
  Q4_0×Q8_0 AVX2/FMA accumulation while the single-token gfx906 path uses
  Q8_1/MMVQ/dp4a with F32 output, explaining the independent GPU trajectory
  without changing MIInfer production behavior
* M4-B24 MI50 correctness envelope; Debug and Release are deterministic and
  satisfy the independently measured external CPU↔gfx906 final-norm/logit
  envelope with matching argmax and top-5 behavior; M4-B is closed
* M4-C1 explicit-token stateful decode; persistent per-layer KV state produces
  first token `8`, consumes it at position 1, and passes Debug/Release physical
  acceptance with reset determinism
* M4-C2 short explicit-ID greedy decode; Release reproduces all eight pinned
  continuation IDs and unoptimized Debug remains a deterministic diagnostic
* M4-C3 model-backed Qwen2 byte-level BPE tokenizer and text-facing greedy
  Release CLI; prompt `hello` and the pinned continuation pass physical
  acceptance with exact IDs and generated text
* M5-A reproducible end-to-end MI50 baseline; the current C3 path measures
  sequential prompt ingestion, TTFT, and post-first-token decode separately
* M5-B decode profile; FFN projections are the largest named GPU event family,
  but the profile also records 1,588 dispatches and substantial instrumentation
  copy overhead
* M5-C0 trace-free decode control; short decode reaches 31.508 tok/s and the
  64-forward growing-context control reaches 12.724 tok/s
* EXP-0012 same-card pinned llama.cpp comparison; standard Q4_0 TG is about
  91 tok/s, while the raw `hello` controls expose a large context-scaling gap
* M5-C1 position-scaled execution audit; dispatches, copied bytes, temporary
  allocations, quantization, FFN, and KV-write copy cost remain flat from
  positions 1–64, while cached attention grows from 3.401 ms to 95.998 ms;
  cached-attention parallelism is the measured M5-C2 target
* M5-C2 cooperative cached attention; the 256-thread/head candidate passes
  the pinned greedy sequence and improves trace-free throughput from 14.430 to
  38.754 tok/s over 64 growing-context forwards; serial remains an explicit
  A/B control
* M5-C3 interleaved attention A/B harness; clean MI50 runs preserve identical
  64-token greedy IDs and measure a 3.073x cooperative-over-serial speedup;
  absolute rates are hardware-state-qualified because telemetry observed
  930/350 MHz auto-mode snapshots
* M5-C4 post-attention baseline; cooperative attention remains bounded through
  cache length 1024, reaching 75.489 ms in the intrusive audit while dispatches
  and copied bytes remain flat; absolute throughput awaits a validated clock
  state
* M5-C5a persistent decode workspace; full decode reuses one session-owned
  workspace across layers and tokens, reducing steady-state temporary
  allocations from 1,086 to zero and improving the low-clock short/growing
  controls by 32.3%/26.9%
* M5-C5b resident normalization weights; immutable F32 norm tensors are read
  directly from the GPU plan, reducing per-token copy bytes from 3,315,200 to
  2,082,304 and improving the low-clock short/growing controls by 9.9%/9.2%
* M5-C6a execution-overhead attribution; the remaining copy calls are 576 KV
  writes, 72 layer handoff copies, and one final logits copy, with dispatch
  topology unchanged at 1,588/token
* M5-C6b direct fast-path layer-output handoff; layer I/O copies fall from 72
  to 36 and copied bytes from 2,082,304 to 1,492,480 per token, with zero
  allocations and unchanged deterministic IDs; the interleaved timing result
  is neutral at the observed low clocks
* M5-C6c coalesced KV-cache writes; 576 tiny K/V memcpy operations become 36
  device-store launches, reducing synchronization sites from 614 to 38 and
  improving the interleaved 64-token control by 5.5% with identical IDs
* M5-C6d GPU-side greedy argmax; final logits transfer falls from 607,744 to 4
  bytes with first-index tie semantics, identical IDs, and a measured 0.37%
  interleaved 64-token gain
* M5-C7 post-copy-cleanup profile; lightweight whole-token GPU timing tracks
  clean wall time closely, so the next target is FFN kernel efficiency rather
  than HIP graphs
* M5-C8a FFN projection shape characterization; current production-like
  Q4_0 × Q8_1 controls measure about 50.24 µs Gate, 50.24 µs Up, and 57.28 µs
  Down with repeated oracle passes, while existing Wave64/alternate controls
  do not justify a production promotion
* M5-C8b Down four-Wave64 candidate; the expanded Q4/Q8 correctness matrix
  passes, but Down regresses from 57.28 µs to 83.36–84.80 µs and Gate/Up also
  regress, so the geometry family is rejected without production selection
* M5-C8c Down long-K attribution; the existing two-Wave64-per-row path is a
  direct split-K-style diagnostic but regresses Down from 57.28 to 73.12 µs
  (27.7%) with the same oracle result. Static code-object data shows equal
  45-VGPR use and no spills for Gate/Down, while hardware-counter profilers
  are unavailable; no split-K or new Down geometry is justified
* M5-C9c Gate/Up Q8 activation reuse is a production KEEP; 180 real-model
  Gate/Up block-stream checks across positions 1, 8, 16, 32, and 64 found zero
  mismatches, the 64-token trajectory stayed identical, and shared reuse
  improved the balanced low-clock A/B from 54.5501 to 55.1724 tok/s (+1.14%)
* M5-C10a refreshed the shared-reuse P64 profile; clean wall/whole-token GPU
  time is 20.213/20.121 ms, deferred attribution is 27.828 ms, and the
  remaining ranking is FFN projection 6.963 ms, attention 4.935 ms,
  normalization 2.978 ms, LM head 2.876 ms, and conversion 2.278 ms
* M5-C10b added real-model normalization/conversion boundary attribution at
  positions 1/8/16/32/64; clean-commit P64 is 19.602/19.733 ms clean wall/GPU
  event with 27.277 ms deferred attribution, 1553 dispatches, 38 sync sites, zero
  allocations, and zero new copy pathology. The selected next experiment is
  one bounded FFN RMSNorm + norm-scale + F32→F16 + shared-Q8 candidate with
  strict byte-identical Q8 and trajectory gates
* M5-C10c fused FFN RMSNorm + norm-scale + exact F32→F16 + shared-Q8 in one
  opt-in gfx906 dispatch. The real-model verifier passed 180/180 FP16 and Q8
  checks with zero mismatches, Release CTest remained 19/19, and all three
  serial 64-token A/B pairs were trajectory-identical. The candidate reduced
  P64 dispatches 1553→1445 but regressed throughput 54.125905→51.302002 tok/s
  (-5.217%), so it is rejected for production selection and the separate path
  remains the default
* M5-C11a refreshed the accepted shared-reuse production baseline at 55.0778
  tok/s over three serial 64-forward runs, with identical deterministic IDs and
  Release CTest 19/19. The P64 profile reports 19.797/19.924 ms clean
  wall/whole-token GPU event, 27.778 ms deferred attribution, 1553 dispatches,
  38 sync sites, and zero allocations; FFN projection is the largest family at
  6.956 ms, followed by attention at 4.937 ms
* M5-C11a also refreshed the pinned gfx906 llama.cpp control: PP512 984.552,
  TG128 91.875, and TG256 91.692 tok/s. The external run reached approximately
  1725/1000 MHz while MIInfer's measured phase was approximately 925/350 MHz,
  so the 1.67x standard-TG differential is directional, not clock-matched.
  The next bounded experiment is an exact-shape FFN GEMV differential against
  the available gfx906 MMVQ path
* M5-C12a refreshed the accepted shared-reuse path at stable peak: the current
  Release build measured 55.419 tok/s, while P64 measured 19.579 ms clean wall
  time and 19.605 ms whole-token GPU event. Dispatches, syncs, allocations,
  and residual copy bytes stayed flat through P64; attention alone grew with
  context from 0.451 ms at P1 to 4.928 ms at P64. The next target is bounded
  cooperative cached-attention differential profiling
* M5-C12b extended the cooperative attention scaling audit through P1024;
  attention measured 4.932/9.491/18.607/36.775/74.071 ms at P64/P128/P256/
  P512/P1024 with flat structural counters and no second collapse. Its
  four-Wave64 history-partition candidate changed the first token from 8 to
  8673 and was rejected; production attention remains unchanged
* M5-C13a measured the short-context fixed floor at P1/P2/P4/P8. Clean wall
  minus attention stayed near 14.7 ms/token, whole-token GPU events tracked
  wall time closely, and dispatches, syncs, allocations, and residual copies
  stayed flat. C13b then audited the proposed exact-shape LM-head differential
* M5-C13b audited that proposed LM-head differential and found the pinned
  llama.cpp gfx906 Q6_K path consumes Q8_1, while MIInfer's LM head consumes
  Q8_K. The contracts are not directly comparable; no LM-head replacement or
  Q8_1 compatibility path was added
* M5-C13c mapped the remaining fixed-floor contracts. Only the already-cleared
  Q4_0×Q8_1 projection rows have an evidence-backed direct llama.cpp
  comparison; LM head, norm, conversion, broad quantization, RoPE/KV, and
  attention remain non-comparable under the retained contracts. C13c made no
  production change and selected no implementation experiment
* M5-C14a mapped the complete layer/token execution path, including
  representations, materializations, quantizations, dispatches, and known
  llama.cpp differences. No uncleared same-contract differential or proven
  llama.cpp-only work elimination was identified; production is unchanged

EXP-0002 is accepted as `KEEP`. The seven real Qwen3-8B projection shapes are
correctness-valid for both the project-owned HIP baseline and the strongest
valid installed-library comparison (`hipblasGemmEx` with FP16 operands and
FP32 compute). The canonical streaming results and raw artifacts are recorded
in [`EXP-0002`](../experiments/EXP-0002-fp16-gemv-baseline.md).

ROCr/HSA initialization fails with both AMD GPUs exposed, but the confirmed
gfx802-isolation workaround leaves the MI50/gfx906 device usable. With that
configuration HIP, MIInfer, the pinned reference, and the Qwen3-8B smoke test
all work. The gfx802 isolation remains an operational platform prerequisite.

## Not implemented

* full production inference runtime and model-facing integration
* model loading outside the pinned Qwen3-8B contract
* general GGUF parsing
* custom tensor packing
* production execution planner
* general-purpose memory planner
* sampling
* MoE execution
* HIP graph capture
* HTTP server

The C++20/HIP infrastructure, model loader/planner, accepted single-token
full-model MI50 execution, persistent multi-token decode, text-facing Qwen3
tokenizer/generator, M5-A baseline, M5-B profile, M5-C0 trace-free control,
and M5-C1 position-scaled audit
are present. Sampling, serving, and generic runtime expansion remain outside
scope. The immediate performance question is context scaling and execution
overhead relative to the pinned gfx906 llama.cpp control.

---

# Immediate Objective

The immediate technical objective is:

> M6-A3: implement and externally validate one stateful Gated DeltaNet layer
> at positions 0, 1, 2, 4, and 8. Do not begin full-model execution or
> performance optimization before the single-layer state contract passes.

The initial eight-token fixture matches the independent MI50 reference through
position 2. Release passes the complete fixture and Debug remains a
finite/cache/determinism diagnostic that selects `419` instead of reference/
host token `470` at position 3. M4-C2 is closed under this build contract.
The fixed-prefix diagnostic localizes the first build-sensitive state to
position 1, where outputs first differ at layer 20; position-3 outputs first
differ at layer 21 and then grow gradually. Serialized Debug is unchanged,
while RelWithDebInfo follows Release, pointing to unoptimized HIP code
generation rather than a cache-ordering race.

M0 is closed under the documented gfx802-isolated configuration. The
repository-side infrastructure, physical MI50 validation, model artifact, and
reference baseline are recorded. The gfx802-isolation requirement remains a
documented platform prerequisite for M1 GPU execution.

The current C3 implementation owns a model-backed Qwen2 byte-level BPE
tokenizer for the embedded `gpt2`/`qwen2` GGUF contract. The Release CLI
acceptance uses prompt `hello`, which encodes to `14990`, runs the existing
persistent 36-layer MI50 decode for eight greedy steps, and detokenizes the
pinned IDs to `) {\n        return "Hello, "`. Sampling, chat templates, streaming,
and performance benchmarking is now recorded by M5-A; the cooperative
attention, workspace, residency, copy-cleanup, FFN characterization, and
normalization/conversion attribution slices are recorded by M5-C, with the
isolated C10c FFN normalization-to-shared-Q8 candidate next.

---

# Immediate Deliverables

The M1/M2 kernel-laboratory deliverables currently include:

* root CMake project
* canonical gfx906 build preset
* trivial HIP kernel
* device validation
* CPU-side correctness test infrastructure
* GPU test infrastructure
* microbenchmark harness
* machine-readable benchmark output
* hardware/environment capture
* experiment scaffold
* deterministic Q4_0/Q8_1 host quantization and CPU oracle
* project-owned Q4_0 × Q8_1 HIP baseline
* activation-quantization and fan-out measurements
* gfx906 `v_dot4_i32_i8` probe and Q4×Q8 packed-dot candidate
* EXP-0006 five-run scalar-versus-packed-dot evidence
* zero-point-corrected Q4×Q8 dot4 kernel using Q8_1 sum metadata
* EXP-0007 five-run scalar/control/candidate evidence and size-matched memory reference
* EXP-0008 five-run direct primitive comparison against pinned llama.cpp MMVQ
* EXP-0009 five-run 128-thread and Wave64 geometry comparison against MMVQ
* M3 pinned Qwen3-8B model recognition, 399-tensor validation, 4.77 GB GPU
  weight residency, and static kernel/buffer plan
* M4-A correctness foundation for Qwen3 host oracles and initial gfx906 GPU
  probes (RMSNorm, Q4_0 embedding lookup, and Q6_K GEMV)
* M4-A2 complete host-side layer-0 composition for the pinned single-token
  fixture, automatic comparison of all 28 reference checkpoints, and a
  comparator mutation test
* M4-A3 complete MI50 GPU layer-0 composition for the same position-zero
  fixture, with GPU-to-host and GPU-to-reference comparison in Debug and
  Release, plus a GPU composition mutation discriminator
* M4-A4 deterministic four-position host and MI50 layer-0 execution with an
  explicit post-RoPE KV-cache contract, reset/append/preservation checks,
  causal-prefix validation, and cache mutation discriminators
* M4-A5 independent external four-position reference trace, including
  post-RoPE K/V cache-write vectors, with Host/MI50 Debug/Release parity

The repository-side specialization and M3 runtime-scaffold deliverables are
complete. M4-A is complete: the host and MI50 layer-0 compositions match an
independent four-position trace from the pinned reference within the frozen
stage-specific tolerances in Debug and Release. The external trace also
directly pins post-RoPE K and unmodified V as the cache-write representation.
The four-position tests prove append, preservation, reset, causal extent,
host/GPU cache parity, and external-trace comparator mutation detection. No
token-generation or end-to-end performance claim is made here.

---

# Current Build Direction

The intended build stack is:

```text id="v2eslv"
CMake
C++20
HIP
gfx906
```

Canonical configurations include:

```text id="1kk4ya"
mi50-debug
mi50-release
```

Do not introduce additional build systems unless explicitly justified.

The release preset explicitly compiles for `gfx906`. A host-only preset is
available for environments without a HIP toolchain; the canonical MI50 presets
require HIP.

---

# Current Dependency Policy

Dependencies should remain minimal.

Do not add:

* llama.cpp
* GGML
* PyTorch
* vLLM
* Triton runtime dependency
* Boost
* large framework libraries

unless explicitly approved for a specific purpose.

Small test or utility dependencies may be considered if they reduce complexity without affecting runtime architecture.

---

# Current Runtime Policy

There is currently **no token-executing runtime**. M4 may implement only the
minimum execution path required for the selected Qwen3-8B target.

Do not prematurely create:

* generic graph abstractions
* scheduler frameworks
* backend interfaces
* plugin systems
* model registries
* generic tensor frameworks

The architecture should emerge from measured kernel and model requirements.

---

# Current Reference Strategy

MIInfer will maintain a separate external gfx906 reference implementation for:

* performance comparison
* correctness comparison
* model behavior reference
* research

The primary reference is pinned to `milpster/gfx906-llama-cpp` commit
`6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`. The dense control model is pinned
to `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`. The physical MI50 build,
conversion artifact, and measurements are recorded in
[`reference-baseline.md`](reference-baseline.md).

The reference implementation is not part of MIInfer's runtime architecture.

---

# Current Benchmark Priority

Initial benchmark work should focus on kernel-level infrastructure.

The first benchmark families should approximately be:

```text id="drjqfz"
1. trivial HIP launch/timing validation
2. memory bandwidth sanity
3. FP16 GEMV baseline
4. quantized GEMV baseline
5. gfx906-specialized GEMV experiments
```

The accepted EXP-0002 baseline has been characterized in EXP-0003, and
EXP-0004 established a K/V-specific FP16 K-split specialization with a 52%
latency reduction on the real `M=1024, K=4096` shapes. The
external llama.cpp kernel-share profiling remains unavailable because a
compatible ROCm profiler is not installed, but EXP-0008 completed the more
important direct primitive timing for the pinned MMVQ path. At that stage it
did not yet show a material MIInfer advantage across the major Q/O and FFN
regimes; EXP-0009 subsequently corrected the measured geometry gap.

The accepted EXP-0009 result produced a shape-specialized MIInfer family
competitive with or faster than the pinned MMVQ path on all seven projection
shapes.

Attention and MoE benchmarks should wait until representative target-model
shapes and actual bottlenecks are frozen. EXP-0005 accepted the Q4_0 × Q8_1
baseline on all seven real Qwen3-8B projection shapes. EXP-0006 proved that
the compiler emits `v_dot4_i32_i8` and the candidate is numerically correct,
but the current register unpack/pack implementation regresses Q/O and FFN
latency. The accepted K-split implementation is currently limited to the K/V
shape family; Q/O continues
to use the EXP-0002 baseline configuration.

---

# Current Experiment Queue

Provisional experiment sequence:

```text id="a3i4qd"
EXP-0001 — benchmark harness validation

EXP-0002 — FP16 GEMV baseline

EXP-0003 — FP16 GEMV bottleneck characterization (RETEST: external profiler gap)

EXP-0004 — FP16 K-split parallelism for K/V (KEEP)

EXP-0005 — quantized GEMV baseline (KEEP)

EXP-0006 — gfx906 Q4_0 × Q8_1 packed-dot specialization (REJECT)

EXP-0007 — zero-point-corrected Q4_0 × Q8_1 dot4 (KEEP)

EXP-0008 — direct MIInfer versus pinned gfx906 llama.cpp MMVQ (KEEP)

EXP-0009 — K/V workgroup and Wave64 reduction geometry (KEEP)

M3 — minimal Qwen3-8B runtime scaffold (CLOSED)

M5-C1 — trace-free dispatch/materialization and context-scaling characterization (CLOSED)

M5-C2 — cached-attention scaling optimization (CLOSED)

M5-C3 — repeat interleaved attention A/B and profile the new baseline (CLOSED)

M5-C4 — canonical post-attention baseline (RETEST: validated clock state unavailable)

M5-C5a — persistent decode workspace (CLOSED)

M5-C5b — resident normalization weights (CLOSED)

M5-C6a — execution-overhead attribution (CLOSED)

M5-C6b — direct layer-output handoff (CLOSED; structural KEEP, neutral timing)

M5-C6c — coalesced KV-cache writes (CLOSED; structural KEEP, +5.5% A/B)

M5-C6d — GPU-side greedy argmax (CLOSED; structural KEEP, +0.37% A/B)

M5-C7 — post-copy-cleanup bottleneck profile (CLOSED; KERNEL recommendation)

M5-C8 — FFN projection kernel experiment (C8a CLOSED; C8b REJECTED; C8c
CLOSED — no split-K promotion)

M5-C9a — production FFN/end-to-end attribution (CLOSED; measurement-only;
SwiGLU-to-Down-input quantization retained as isolated C9b candidate)

M5-C9b — fused SwiGLU-to-Down-input Q8 candidate (CLOSED; REJECTED for
production selection; long decode diverged at position 38)

M5-C9c — Gate/Up activation-Q8 reuse (CLOSED; KEEP; production-selected,
+1.14% balanced A/B)

M5-C10a — refreshed P64 production attribution (CLOSED; measurement-only;
next target not preselected)

M5-C10b — normalization/conversion boundary attribution (CLOSED;
measurement-only; C10c FFN normalization-to-shared-Q8 candidate selected)

M5-C10c — FFN normalization-to-shared-Q8 fusion (CLOSED; correctness PASS;
production REJECTED; 5.217% slower in clean serial A/B)

M5-C11a — production and llama.cpp differential baseline (CLOSED;
measurement-only; 55.0778 tok/s MIInfer shared-reuse baseline; fresh external
TG128/TG256 control 91.875/91.692 tok/s with clock-state qualification;
next target exact-shape FFN GEMV differential)

M5-C11b — exact-shape FFN GEMV differential (CLOSED; no FFN/MMVQ port
selected; MIInfer approximately tied/faster on Gate, Up, and Down under the
retained direct protocol; K/V advantage already addressed by EXP-0009;
clock-controlled end-to-end A/B complete: MIInfer 55.356 tok/s versus llama.cpp
90.566 TG128 / 90.389 TG256 under stable_peak)

M5-C12a — stable-peak non-FFN profile (CLOSED; current production 55.419 tok/s;
P64 19.579 ms clean wall and 19.605 ms whole-token GPU event; attention is the
only demonstrated context-growing family through P64; C12b selected for bounded
cooperative cached-attention differential/scaling)

M5-C12b — cooperative attention scaling (CLOSED; production cooperative path
KEEP; history-partition candidate REJECTED; linear P64-P1024 scaling recorded;
no C12c implementation preselected)

M5-C13a — fixed-cost floor profile (CLOSED; measurement-only; approximately
14.7 ms/token fixed wall-minus-attention floor at P1/P2/P4/P8; no idle or
structural-counter growth; C13b contract audit followed)

M5-C13b — LM-head contract audit (CLOSED; pinned external Q6_K path is
Q8_1, MIInfer path is Q8_K; no valid direct differential; no production change;
no production target selected)

M5-C13c — fixed-floor contract map (CLOSED; measurement-only; only the already
cleared Q4_0×Q8_1 projection rows have a valid direct differential; no
implementation selected)

M5-C14a — fixed-floor execution map (CLOSED; measurement-only; complete
layer/token representation and work map recorded; no uncleared same-contract
differential or implementation selected)

M5-C15 — optimization closure and parity decision gate (CLOSED; accepted and
rejected M5 record complete; current production path retained; Path A/Path B
architectural choice pending)

M6-A0 through M6-A7 — Qwen3.8-27B architecture, reference, hybrid bring-up,
full forward, and stateful host validation (CLOSED)

M6-B0 — upstream llama.cpp Qwen3.8-27B Q4_K_M MI50 baseline (CLOSED;
stable_peak policy; actual SCLK varied during capture)

M6-B1 — MIInfer Qwen3.8 GPU profile and native-generation baseline (CLOSED for
bring-up; performance parity remains the project objective). M6-A8 through
M6-A28 establish the qwen35 GPU executor, external observable contract, and
native generation through 128 tokens. M6-B29 attributes recurrent stages and
M6-B30 selects the transposed recurrent-state layout.
```

The exact ordering may change based on early measurements.

---

# First Major Gate

The first major project decision occurs at **M2 — Prove Specialization**.

Before significant runtime implementation begins, MIInfer must demonstrate credible evidence that gfx906-specific specialization can improve important target-model operations against the strongest relevant gfx906 implementation. A complete MIInfer runtime is not required for the M2 gate.

If M2 fails to show meaningful potential, the project should be reassessed rather than automatically continuing into a full runtime. EXP-0009 passed this gate with `M2 GO`.

---

# Current Correctness Policy

All candidate kernels must be validated against a trusted reference implementation.

Performance measurements from incorrect kernels are invalid.

Initial kernel tests should include:

* deterministic input generation
* CPU/reference output
* GPU output
* tolerance-based comparison
* explicit NaN/Inf detection

---

# Current Performance Policy

Do not accept performance claims from:

* one run
* unverified GPU clocks
* mismatched build flags
* mismatched tensor shapes
* mismatched quantization
* contaminated hardware state

Follow [`benchmarking.md`](benchmarking.md).

---

# Current Hardware Observation Requirements

Before meaningful GPU benchmarks are accepted, the project should be able to capture where available:

* GPU identity
* gfx architecture
* VRAM
* ROCm version
* HIP compiler version
* kernel version
* SCLK
* MCLK/HBM clock
* temperature
* power
* power limit

Unavailable metrics should be reported as unavailable, not guessed.

---

# Current Scope

## In scope now

* C++20
* HIP
* gfx906
* MI50
* benchmark infrastructure
* correctness infrastructure
* hardware-state capture
* low-level kernel experiments

## Not in scope now

* model serving
* OpenAI API compatibility
* speculative decoding
* MTP
* multimodal inference
* multi-GPU
* distributed inference
* generic model support
* Windows
* CUDA
* RDNA
* MI200/MI300
* training
* fine-tuning

---

# Do Not Start Yet

Until the roadmap explicitly advances, do not spend implementation effort on:

```text id="eofjf1"
HTTP server
OpenAI-compatible API
generic GGUF support
multi-model support
multi-GPU
speculative decoding
MTP
continuous batching
distributed scheduling
```

These do not help answer the current project question.

---

# Next Implementation Task

The current Codex task is:

> Characterize the trace-free decode path against the retained llama.cpp
> comparison, with particular attention to context-dependent attention/KV
> cost, dispatch count, and materialization/copy overhead. Then evaluate one
> measured performance hypothesis at a time. Keep prompt ingestion and decode
> separate; sampling, serving, batching, and unrelated runtime expansion
> remain out of scope.

The M0 evidence gates are complete under the documented gfx802-isolated
configuration. The M2 gate is satisfied by EXP-0009, M3 is closed by the
pinned real-model acceptance, M4-A is closed, and M4-B is closed under the
documented MI50 backend envelope. M4-C1 and M4-C2 prove persistent stateful
decode; M4-C3 now adds the model-backed tokenizer/detokenizer and text-facing
greedy CLI. Sampling remains out of scope initially.

---

# Definition of Current Success

The current phase succeeds when a contributor can:

```text id="f4dg36"
clone MIInfer
     ↓
configure canonical MI50 build
     ↓
compile gfx906 HIP code
     ↓
run correctness tests
     ↓
run a microbenchmark
     ↓
capture hardware state
     ↓
produce reproducible benchmark output
```

The M2 validation chain and M3 model-plan acceptance are complete. M4-C3
provides the correct deterministic text path; M5 now requires reproducible
MI50 performance evidence and measured improvements against the retained
gfx906 reference without broadening the project into a generic runtime.

---

# Last Updated

2026-09-05 — M6-B32 selects the transposed recurrent no-decay-store path,
improving stable-peak native TG64/TG128 by 1.71%/1.63% to 12.0637/11.8983
tok/s. Native generation and the complete P64 observable contract pass with
zero decode allocations, unchanged device usage, and CTest 20/20. See
`experiments/EXP-0124-m6b32-transposed-no-decay-store.md`.

2026-09-05 — M6-B30 selects a physical `[value_head][column][row]` DeltaNet
state layout. The logical state contract is unchanged; the isolated state
update falls from `202.979` to `118.153 us`, and stable-peak native TG64/TG128
improve from `11.4679/11.3067` to `11.8587/11.7159 tok/s`. Native 64/128-token
generation replays pass with zero decode allocations, the complete P64
observable contract passes with the complete `-p12` fixture, and CTest is
20/20. See `experiments/EXP-0122-m6b30-transposed-deltanet-state.md`.

2026-09-04 — M6-B1 recorded the first native Qwen3.8-27B MI50 generation
baseline. MIInfer measured 3.37 tok/s at TG64 and 3.36 tok/s at TG128, with
stable_peak telemetry reaching 1725/1000 MHz, deterministic replay, and zero
decode-loop allocations. Fresh pinned llama.cpp measured 22.57/22.58 tok/s
at TG64/TG128. This is a bring-up baseline, not a final workload-equivalent
parity claim. See
`experiments/EXP-0093-m6b1-qwen35-native-generation-baseline.md`.

2026-09-04 — M6-A28 completed native Qwen3.8-27B autoregressive GPU generation
for 16, 64, and 128 token runs. GPU Q4_K embedding, all 64 hybrid layers,
LM-head argmax, recurrent/KV state, and replay were exercised. Each run had
zero decode-loop allocations, exact replay, and stable device usage of
`17018706644` bytes. See
`experiments/EXP-0092-m6a28-native-qwen35-generation.md`. M6-B1 now owns the
performance-baseline work.

2026-09-04 — M6-A27-CLOSE adjudicated the two remaining teacher-forced argmax
flips under the external observable contract. P2 (`1318 → 1044`) has 5/5
top-5 overlap, cosine `0.999803`, reference margin `0.0349064`, and
`epsilon_top=0.127861`; P12 (`1044 → 1459`) has 5/5 overlap, cosine `0.999443`,
reference margin `0.0483742`, and `epsilon_top=0.0737877`. Both are
non-robust winners (`margin <= 2*epsilon_top`) with the alternate winner at
reference rank 2. Exact teacher-forced agreement remains `62/64` as a
diagnostic; no tolerance was changed. A27 is CLOSED under the
margin-aware external contract. Next is M6-A28 native autoregressive GPU
generation; M6-B1 performance remains pending. See
`experiments/EXP-0091-m6a27-close-observable-contract.md`.

2026-09-04 — M6-A27.9 fixed the L0 Q5_K × Q8_K arithmetic contract: per-block
contributions now match the scalar reference within `5.96e-8`, block-sum error
is `0`, and external-gated projection error dropped from `0.00164509` to
`9.53674e-7`. Release CTest remains 20/20. A27 still needs the existing
64-layer observable-contract and trajectory retest. That retest now shows L0–L2
at roundoff for P2, but P2 remains `1318 → 1044` and P12 is also `1044 → 1459`;
teacher-forced agreement is 62/64. The first visible P2 layer discrepancy is
now L3 (`0.00255775`). A27 remains in RETEST; no tolerance was changed. See
`experiments/EXP-0090-m6a279-qwen35-a27-observable-retest.md`. M6-A27.8 cleared the L0
Q8_K activation contract: the external
gated P2 replay produces a 7,008-byte Q8_K buffer byte-identical to the pinned
llama.cpp reference (`0` mismatches, fingerprint
`9331021456029706823`). The remaining L0 `ssm_out` projection discrepancy is
therefore in Q5_K × Q8_K projection/accumulation or dot-product ordering. A27
remains in RETEST; no tolerance or production behavior changed. M6-A27.7
isolated the L0 output-projection contract: external
gated input replay through MIInfer's direct Q8_K → Q5_K `ssm_out` path leaves
`0.00164509` error, while gated output and residual arithmetic are cleared.
A27 remains in RETEST; no tolerance or production behavior changed. M6-A27.6
traced P2 operation boundaries through L0–L2: L0
recurrent output remains at roundoff, while attention/residual and later QKV
boundaries introduce the first material discrepancies (`0.00164509`, `0.0172018`,
`0.0453629`). A27 remains in RETEST; no tolerance or production behavior
changed. M6-A27.5 localized the P2 mismatch to an early, distributed
activation drift: measurable at L0, growing through L1/L2, and with no new
L53/L54 discontinuity. A27 remains in RETEST; no tolerance or production
behavior changed. M6-A27.4 completed observable-contract adjudication: final P64
logits have cosine `0.999558` and 5/5 top-5 overlap, but teacher-forced
argmax agreement is 63/64 with a deterministic P2 mismatch (`1318` → `1044`).
A27 remains in RETEST; no tolerance or production behavior changed. M6-A27.3
cleared the L53 gated operation: external operands replay
matches the external gated output within `4.77e-7`. The L32–L53 P1 scan shows
gradual activation drift, reaching `1.02168` at L53, so A27 remains in RETEST
for external-contract adjudication. M6-A27.2 traced the L54/P1 input discrepancy to the preceding L53
layer output: L53 output and L54 input both reach `1.02168` max error, while
L53 recurrent output remains within `0.00111498`. A27 remains in RETEST.
M6-A27.1 was recorded after the 64-layer run localized its first observable
failure to the L54 gated path at P1. L54 recurrent output is within
`0.000902295`, while final layer output reaches `23.1531`; M6-A26.9 was recorded
after adjudicating the external recurrent
state contract. The strict `0.05` internal state diagnostic remains unchanged,
while the external-contract mode permits composition because recurrence,
observable layer outputs, and deterministic reset/replay are valid. M6-A26.8
was recorded after tracing the L29 gate projection with external normalized
input; the existing projection reaches `1.90735e-6` max error against external
`z-29`. The remaining discrepancy is upstream L29 hidden-state/contract
sensitivity. M6-A26.5 was recorded after tracing the L30 production K path
to its already-divergent L29 output input. The K-path error remains bounded
through convolution/SiLU and K normalization, so L29 output provenance is the
next target. M6-A21 was recorded after composing qwen35 recurrent GPU layers
0–2 with full-attention GPU layer 3 through positions 0 and 1. The upstream
Qwen3.8-27B Q4_K_M baseline is captured at stable_peak with PP512 median
196.585 tok/s, isolated TG64/TG128/TG256 medians 22.4888/22.2873/21.9009
tok/s, and combined P1+TG64/P1+TG256/P1+TG1024 medians
21.8576/21.9405/21.9107 tok/s. Actual telemetry observed 1000 MHz MCLK and
1282–1725 MHz SCLK, so the result is policy-controlled but clock-state
qualified. M6-B1 should profile the functioning MIInfer GPU path against this
external control. M5-C10b through M5-C15 were recorded after the C9c production
KEEP. C10c passed its strict real-model correctness gates but its one-workgroup
fused path regressed clean decode by 5.217%; the separate FFN normalization/Q8
path remains the production default. C11a refreshed the production baseline and
the external gfx906 control, with clock-state qualification, and selected an
exact-shape FFN GEMV differential as the next bounded experiment. C11b found no
new FFN/MMVQ candidate: MIInfer was approximately tied or faster on Gate, Up,
and Down in the retained direct comparison, while the historical K/V gap was
already closed by EXP-0009. The subsequent stable_peak run resolved the clock
qualification and measured 55.356 tok/s for MIInfer versus 90.566/90.389 tok/s
for llama.cpp TG128/TG256. C12a then measured the current production path at
stable_peak and selected cooperative cached-attention differential/scaling as
the next bounded target. C12b found linear production scaling through P1024,
but rejected its history-partition candidate because it changed the first
generated token. C13a then measured the P1/P2/P4/P8 fixed-cost floor at
approximately 14.7 ms/token, with whole-token GPU events tracking wall time
and no structural-counter growth. C13b then audited the proposed LM-head
differential and found an input-contract mismatch: the pinned external Q6_K
MMVQ path consumes Q8_1, while MIInfer consumes Q8_K. C13c then mapped the
fixed-floor contracts and found no uncleared same-contract external
differential. No valid direct differential or production LM-head change was
selected. C14a then mapped the complete layer/token execution path and
classified known differences as implementation, representation,
work-elimination, or scheduling/fusion; no new valid differential or
implementation target was selected. See
experiments/EXP-0040-m5c14a-fixed-floor-execution-map.md.
M5-C15 then closed the local optimization campaign: the stable-peak
 production path is approximately 55.419 tok/s, P64/P1024 attention scaling is
 predictable, and no further same-contract target is evidence-backed. The
 explicit next decision is Path A (preserve the current trajectory) or Path B
 (begin M6-A reference-correct execution-contract exploration). See
 experiments/EXP-0041-m5c15-optimization-closure-parity-gate.md.

2026-09-05 — M6-B42 refreshed the accepted B41 profile: position-63 total
GPU event `73.8215 ms/token`, layer sum `70.5777 ms`, final LM head `2.45504
ms`, argmax `0.49696 ms`, and zero allocations. M6-B43 tested staging Q6_K
LM-head `d`/scale metadata in LDS. Native replay passed, but the serial
same-build A/B was only `+0.13%` at TG64 and `+0.06%` at TG128, within noise;
the candidate was removed and rejected. See
 experiments/EXP-0134-m6b42-post-b41-profile.md and
 experiments/EXP-0135-m6b43-q6k-q8-1-lm-metadata.md.

M6-B44 tested the pinned llama.cpp-style two-Wave64 split-K mapping for the
Q4_K×Q8_1 path. Native replay and zero-allocation checks passed, but throughput
fell `16.25%` at TG64 and `16.20%` at TG128. The candidate was removed; the
two-independent-output-row B41 mapping remains selected. See
experiments/EXP-0136-m6b44-q4k-q8-1-split-k.md.

M6-B45 performed a source-level Q4_K×Q8_1 inner-loop differential after the
split-K rejection. A fresh position-63 profile measured `73.9278 ms` total GPU
event, `70.6851 ms` layer sum, `2.45568 ms` final LM head, and zero
allocations. The pinned llama.cpp wrapper packs Q8 operand words and scales
into local arrays before its dot helper, while MIInfer reads the same operands
from its LDS-resident Q8 blocks inside the two-part helper. This is a concrete
non-geometric hypothesis, but no performance result is inferred from source
inspection and no candidate was selected. See
experiments/EXP-0137-m6b45-q4k-q8-1-inner-loop-differential.md.

M6-B46 tested that inner-loop hypothesis as an opt-in diagnostic. Native and
64-layer external correctness checks passed, but serial whole-token A/B was
neutral: `+0.01%` at TG64 and `+0.08%` at TG128. The candidate was removed and
rejected; B41 remains selected. See
experiments/EXP-0138-m6b46-q4k-q8-1-packed-input.md.

2026-09-05 — M6-B47 tested an opt-in combined Gate/Up Q4_K×Q8_1 projection
kernel that staged the already-shared Q8_1 input once. Native generation,
the 64-layer external observable contract, poisoned reset/replay, and Release
CTest 20/20 passed. The candidate was nevertheless slightly slower in serial
same-build A/B: `-0.21%` at TG64 and `-0.04%` at TG128. It was removed and
rejected; the B41 separate-projection path remains selected. See
experiments/EXP-0139-m6b47-dual-gate-up.md.

2026-09-05 — M6-B48 refreshed the post-B47 profile and tested an opt-in
persistent decoded-metadata path for Q4_K FFN Down. The position-63 profile
measured `73.8226 ms` total GPU event and `70.5030 ms` layer sum, with FFN
Down still about `0.42–0.45 ms/layer`. The persistent candidate passed native
and 64-layer external correctness, poisoned reset/replay, and CTest 20/20,
but was neutral/slower in A/B: `+0.03%` TG64 and `-0.17%` TG128, while adding
`222,822,400` bytes of VRAM. It was removed and rejected; B41 remains
selected. See experiments/EXP-0140-m6b48-persistent-q4k-metadata.md.

2026-09-05 — M6-B49 tested an opt-in fusion of the transposed no-decay-store
DeltaNet state update with head RMS normalization. Native replay passed with
zero decode allocations and unchanged device usage, but fresh serial A/B was
noise-level: `-0.27%` at TG64 and `+0.16%` at TG128. The candidate was removed
and rejected; the existing transposed no-decay-store path remains selected. See
experiments/EXP-0141-m6b49-state-head-rms-fusion.md.

2026-09-05 — M6-B50 tested an opt-in fusion of the transposed no-decay-store
DeltaNet update through head normalization and recurrent gating. Native replay
passed with zero decode allocations and unchanged device usage, but fresh
serial A/B regressed `-0.32%` at TG64 and `-0.28%` at TG128. The candidate was
removed and rejected; the existing transposed no-decay-store path remains
selected. See experiments/EXP-0142-m6b50-fused-recurrent-output.md.

Update this document whenever:

* active milestone changes
* immediate technical objective changes
* supported hardware changes
* experiment priority changes
* a major architectural assumption changes
