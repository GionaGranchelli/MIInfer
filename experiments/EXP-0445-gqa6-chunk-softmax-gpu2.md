# EXP-0445 — GQA6 UNROLL=4 block online-softmax

**Verdict: REJECT.** The candidate completed the restored suffix and was
slightly faster in one pair, but it failed the numerical gate by a wide margin.
The timing is a single screening pair and does not establish a repeatable gain.
The candidate kernel, route, and pair-selector changes were reverted; no
production default changed.

## Design

Base: `8b90661ea8ca0d1930749b761f0085efba5067f7`, the opt-in direct-global
Q8 GQA6 route. Keep its Q8 loads/scales, six-query-head mapping, 32-way
Split-K, QK dot order, graph-state bounds, and key order. The candidate computes
the maximum over each existing four-key UNROLL group, rescales the running
softmax sum and value accumulator once, then adds the four weighted V terms in
key order. This changes FP32 reduction/association order. The feature was
selected only by `MIINFER_M31_Q8_GQA6_CHUNK_SOFTMAX`; the HIP Graph cache key
included that flag.

Hypothesis: one rescale of the running value accumulator per four keys could
shorten its dependent update chain. Risk: the changed softmax/accumulator
association could alter full-model logits and recurrent state.

## Compiler gate

Built target `miinfer-m31-fp16-full-model` in the pinned ROCm 7.2.1 image
`f5f33d8e8efbe77f0a08335140f9c9a162202067189499be9dbddd2eb214afa4`.
The candidate gfx906 code object had 64 VGPR, 48 SGPR, zero VGPR/SGPR spills,
zero private bytes, zero group/LDS bytes, wave64, and workgroup size 192.

## One matched GPU2 pair

- Physical device: MI50 BDF `0000:88:00.0`, unique ID `0x2152414173497dd3`.
- Model: Qwen3.8-27B Q4_K_M; Q8/Q8 persistent session at 73,099 tokens.
- Same tokenizer-backed 1,024-token suffix and binary/image for both routes.
- Both restored 73,099 tokens, dispatched two 512-token HIP Graph replay tiles,
  and ended at position 74,123.
- Route traces: 16 control dispatches with `direct_global=1,
  chunk_softmax=0`; 16 candidate dispatches with `direct_global=1,
  chunk_softmax=1`; all had `graph_state=1` and `token_count=1024`.
- Control/candidate restore time: 2.451s / 2.492s. Suffix time: 20.879s /
  20.030s. The observed ratio was 1.042x (4.1% shorter for the candidate),
  but this is one pair only and is not a repeatable performance result.
- Both outputs were finite. First and second greedy tokens matched: `15`, `8`.
- Full-logit maximum absolute error: `0.868862152` (required `<= 0.01`).
- Full-logit cosine: `0.999290384` (required `>= 0.9999`).
- Recurrent-state maximum absolute error: `0.905632019`; mean absolute error:
  `0.001747033`.
- Harness exit code: `3` (`all_correct=0`). No repeat was run after the
  correctness failure.
- Thermal guard: maximum junction temperature `58 C`; no warning or stop.

**Decision:** reject on correctness. Matching two greedy tokens does not offset
the full-logit and recurrent-state errors. The one-shot 4.1% timing observation
is inconclusive and below the requested 10% improvement threshold.

## Raw evidence

Logs, telemetry, preflight identity, hashes, code object, and disassembly remain
on Machinist at:

`/home/machinist/m31-gqa6-longctx-results/EXP-0445-gqa6-chunk-softmax-gpu2/`

The run record is in `run/run.log`, `run/telemetry.log`,
`run/preflight-gpu2.txt`, `run/preflight-processes.txt`,
`run/input-sha256.txt`, `run/image-id.txt`, and `run/exit-code`. Compiler evidence
is in `build/code-object-notes.txt`, `build/disassembly.txt`, and `build/build.log`.
