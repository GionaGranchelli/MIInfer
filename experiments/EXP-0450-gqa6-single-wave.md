# EXP-0450 — GQA6 single-wave KV reuse

**Status:** REJECTED — exactness and performance gates failed
**Base:** `8b90661ea8ca0d1930749b761f0085efba5067f7`
**Device:** Machinist GPU0, BDF `0000:04:00.0` (MI50 / gfx906)
**Model:** `/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf`
**Container:** `localhost/miinfer-m31-rocm721-libdw:20261014T094012Z`

## Candidate

The opt-in candidate used one wave64 CTA per KV head, processing the three
GQA head pairs serially while retaining each head's online-softmax state. The
low half-wave loaded each packed Q8 K/V operand and the high half received the
values through wave shuffles. The intent was to reuse each KV load across all
six query heads while preserving the per-key update order.

The gfx906 object compiled with wave size 64, 133 VGPR, 42 SGPR, zero spills,
and zero LDS allocation. Source and ISA inspection confirmed the K/V loads
were outside the three-pair update sequence.

## Valid screen — pair2

One tokenizer-backed 1,024-token suffix was run from the restored 73,099-token
Q8/Q8 session. HIP Graph replay was enabled. The pair harness was launched
with `MIINFER_M31_Q8_GQA6_SINGLE_WAVE_PAIR=1`. Before candidate capture,
`MIINFER_M31_TRACE_ATTENTION_ROUTE=1` was set in-process with GDB; GDB detached
before the candidate timed prefill. The candidate emitted exactly 16
`miinfer_q8_gqa6_single_wave_dispatch=1 graph_state=1 token_count=1024` lines,
one per layer.

| Path | Restore | Suffix prefill | Final position | First / second token |
|---|---:|---:|---:|---|
| Direct-global control | 2.402 s | 19.031 s | 74,123 | 18,369 / 198 |
| Single-wave candidate | 2.439 s | 38.402 s | 74,123 | 18,369 / 198 |

The candidate was about 2.02× slower (`control / candidate = 0.496×`). Both
outputs were finite and the checked tokens matched, but exact logits and the
recurrent-state trace did not match. The candidate fails both required gates:
at least 10% faster and exact full-model state/logit agreement. No repeat or
cold-64K run was made.

Thermal telemetry ran throughout the pair. GPU0 peaked at 65°C junction,
below the 80°C warning threshold. GPU2 was not used.

## Invalid screen — pair1, excluded

The first harness launch omitted
`MIINFER_M31_Q8_GQA6_SINGLE_WAVE_PAIR=1`. It therefore compared
`q8_production_session` (68.871 s) with
`q8_gqa6_production_arith_session` (28.057 s). That comparison was exact but
did not invoke the direct-global control or the single-wave candidate and is
excluded from the result. The mistake was in launch configuration, not the
candidate result above.

## Decision

Reject the single-wave implementation. Revert its kernel, dispatch, graph
identity, and harness changes; retain this report as the negative result. The
single-wave KV reuse schedule did not preserve exact state/logits and did not
beat the direct-global GQA6 control on the restored suffix.
