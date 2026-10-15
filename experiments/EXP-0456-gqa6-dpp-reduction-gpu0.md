# EXP-0456 — GQA6 hybrid DPP reduction, GPU0

## Verdict

This is a positive but below-threshold screen. After fixing a DPP direction
bug, the reduction passed exactness checks and the restored-session result was
correct, but the suffix improved only 4.8%, below the predeclared 10% material
win threshold. Preserve the corrected implementation and benchmark selector
on this experimental branch as strictly opt-in; keep the production default
unchanged.

## Candidate and correction

The control was direct-global GQA6 Q8 attention from baseline
`8b90661ea8ca0d1930749b761f0085efba5067f7`. The candidate retained the
XOR-16 lane exchange and replaced reduction offsets 8, 4, 2, and 1 with DPP
row/quad operations, followed by the existing width-64 leader broadcast.

The first attempt mistakenly used DPP row-shift-right controls (`0x118`,
`0x114`). The GPU mapping probe showed these send lane 0 toward lanes 8 and 4:
lane 8 received lane 0 for row-shr:8, and lane 4 received lane 0 for row-shr:4.
The required lane-0 reduction instead needs lane 0 to read lane +8/+4. The
correct controls are row-shift-left (`0x108`, `0x104`). The two quad
permutations were correct: `0x04e` maps lane 0 from lane 2, and `0x0b1` maps
lane 0 from lane 1.

The original row-shift-right restored pair was wrong: logits, recurrent-state
trace and generated tokens differed. Its timing is retained only as diagnostic
history (18.802 s control, 17.935 s wrong-direction candidate) and is not a
valid candidate performance result. The corrected row-shift-left helper
matched the XOR control bit-for-bit over 128 deterministic subgroup reductions
on GPU0 (`mismatches=0`, `max_abs=0`).

## Isolated reduction microbenchmark

A single-wave, dependency-chained HIP microbenchmark ran five alternating
shuffle/hybrid pairs, 131,072 reductions per sample, with finite normalized
outputs. It ran on Machinist GPU0, BDF `0000:04:00.0`:

```text
shuffle cycles: 62,915,368  62,915,464  59,768,892  62,915,460  62,915,464
hybrid cycles:  31,458,128  31,458,136  30,408,752  31,457,328  31,457,328
```

Median dependent-reduction cycles were about 62.9M for shuffle and 31.46M for
the corrected hybrid, approximately 2.0x fewer cycles. This is a single-wave
latency microbenchmark, not a model or multi-wave throughput result.

## Corrected restored-session comparison

One matched pair used the pinned model `Qwen3.8-27B-Q4_K_M.gguf`, the same
73,099-token Q8 session, and the same 1,024 tokenizer-backed prompt tokens.
Both sides reported one HIP graph replay tile and final position 74,123. The
runtime route trace recorded exactly 16 GQA6 layer dispatches for each side:

- Control: `direct_global=1`, `dpp_reduce=0` for all 16 layers.
- Candidate: `direct_global=1`, `dpp_reduce=1` for all 16 layers.

| Route | Restore | Suffix | First token | Second token |
| --- | ---: | ---: | ---: | ---: |
| Direct-global control | 2.390 s | 18.796 s | 18,369 | 198 |
| Corrected DPP | 2.414 s | 17.938 s | 18,369 | 198 |

The harness reported exact session logits, exact recurrent-state trace,
matching generated tokens, finite values, and exit code 0. This was one sample
per route, so it does not establish repeatable performance. The suffix ratio
was `18.796 / 17.938 = 1.048x` (4.8% faster), below the 10% campaign gate.

## Build and runtime evidence

- ROCm image: `localhost/miinfer-dev`, digest
  `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`.
- Corrected full-model binary SHA-256:
  `17e2bb2d1a3e38ec9a6fa95e9b0b03a8e77054674d790ded29dd35a99df7345e`.
- Corrected kernel source SHA-256 during the run:
  `566fce053be6fc188de62bbe4629abeb4dda442aaace9dcb2865661606fef3c9`.
- Corrected target assembly metadata: control 5,524 bytes / 60 VGPR;
  hybrid 5,220 bytes / 55 VGPR; both 50 SGPR, occupancy 4, zero scratch.
- GPU: Machinist MI50 gfx906, BDF `0000:04:00.0`, unique ID
  `0x21678e17348c2f7`. Preflight was idle at 24°C. The 80/85°C thermal guard
  remained active and recorded a 66°C peak; it did not stop the run.
- Corrected run log, preflight, and telemetry:
  `/home/machinist/m31-gqa6-longctx-results/EXP-0456-gqa6-dpp-gpu0/corrected-row-shl/`.
- Initial wrong-direction run log and telemetry:
  `/home/machinist/m31-gqa6-longctx-results/EXP-0456-gqa6-dpp-gpu0/`.
- Tiny mapping/reduction probes and microbenchmark logs:
  `/tmp/m31-gqa6-dpp-probe/` on Machinist.

## Decision

The directional mistake explains the initial numerical failure. The corrected
reduction is exact and promising in isolation, but the valid end-to-end result
misses the campaign threshold. Preserve this candidate only as an opt-in
experimental route for future targeted work; this single pair does not qualify
it for the production default or establish repeatable performance.
