# EXP-0329 — M25-E complete Mx attention Q/K contract screen

## Hypothesis

Using MIInfer's existing pinned-style `MxQ8_1MmqBlock`, Mx Q4_K repacking, and
Mx MMQ kernel for separate Q and K projections could provide the complete
external execution contract needed to improve the fused M23 Q/K path.

## Baseline

Current M23 Q/K prefill, which completed the B512 layer bakeoff at about
`63.36 ms` for layer 3.

## Candidate

An opt-in `MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_QK=1` path separately packed
and uploaded Q/K, quantized the normalized activation into Mx Q8 form once,
and dispatched the existing Mx Q4_K MMQ kernel for each projection.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M GGUF, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Release build; ROCm `7.1.52802-9999`; clang `20.0.0.rocm`
- layer 3, B512, hermetic bakeoff; Mx Q8 batch path otherwise disabled

## Benchmark

The candidate was run in three interleaved pairs with fresh control
processes. All control runs completed at `63.3630`, `63.3250`, and
`63.2649 ms`.

## Correctness

The candidate completed with finite values but failed the existing scalar
parity gate on all three attempts: `max_abs > 1.0`. Because correctness
failed, no candidate performance result is admissible.

## Results

No candidate timing is accepted. The candidate process exited with
`mx_qk wide attention candidate exceeds scalar control tolerance`.

## Interpretation

The already-present Mx kernel and format are not a drop-in replacement for
the M23 Q/K path at the required layer-output error budget. A full external
port would need independent layer-level numerical validation before any
throughput claim.

## Decision

**REJECT and remove.** No Q/K Mx selector is retained.

## Follow-up

The qualified H/I path remains unchanged. Further stretch work requires a
new measured production-shape target and a correctness contract before code.
