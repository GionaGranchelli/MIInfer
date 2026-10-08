# M31-0003 — Decode and MMQ investigation

## Current path

The M31 benchmark and production `run`/`chat`/`serve` use
`PrefillV2Model`. Each generated token traverses 16 topology blocks, comprising
48 recurrent/GDN layers and 16 GQA layers. Dense Q4_K projections use the Mx
repacked-MMQ interface; M=1 selects the `mx_repacked_mmv_kernel` by default.
The gate/up pair uses the separate fused `Q4KWaveSwigluFusedTile` kernel.
The production path is not the standalone `Q4KWaveTile` GEMV route found in
legacy/alternate code.

Ordinary `generate()` captures/replays a one-token HIP graph, then copies the
full logits vector to host and synchronizes for CPU sampling. The direct
`decode_step()` diagnostic instead uses device argmax and returns one token.
These are different routes and their timings must not be conflated.

## Findings

| Finding | Classification | Evidence and limitation |
|---|---|---|
| Recurrent/GDN layers dominate measured 8K decode profile (65.5%); GQA layers contribute 26.7%, logits 3.3%, residual 4.2%. | CONFIRMED at one diagnostic point | Machinist 8K synchronized profile, one sample; not decode throughput and not Z840 long-context attribution. |
| Context-dependent GQA KV scan grows with context. At 128K, reading all 16 layers' FP16 K/V once per token is about 8 GiB/token (about 1 GiB at 16K). | CONFIRMED algorithmically; traffic is a model | Counts 16 layers × 4 KV heads × 256 dimensions × K/V × 2 bytes × context. Cache reuse/transactions and actual DRAM traffic are not measured. |
| Direct `decode_step()` source contains approximately 933–997 kernel launches/token depending on projection branches. | STRONG source-derived estimate | Counted call sites, not observed dispatches; graph replay behavior differs. No hardware launch trace exists in this task. |
| Mx projection entrypoints re-read `MIINFER_MX_MMV` through environment lookup. | CONFIRMED source; impact unknown | Repeated in layer/token launch path; no measurement shows this is material. Do not optimize on sight. |
| Some activation quantizations repeat across stages. | CONFIRMED source; not shown redundant | QKV, gated output, and FFN-down inputs have differing consumers/formats. Removing one without proving identical format and numerical contract is unsafe. |

Historical decode evidence in EXP-0351 found context-dependent growth mainly in
attention/KV through 16K while the recurrent floor remained the larger fixed
cost. EXP-V2-0009 retained the fused gate/up route; EXP-0174's Q4_K native-down
candidate showed only a 3–4% candidate gain and was **RETEST / NO ROLLOUT**.
EXP-0358 closed as `ROUTES_NOT_COMPARABLE`; it is not evidence to swap current
decode backends. EXP-0252/0254 reject wider grouped-prefill MMQ64, not M=1
decode.

## Decision

No decode/MMQ production edit is justified by this investigation alone. The
large fixed GDN component and the 128K theoretical attention traffic identify
where future measurements should split; they do not establish which component
causes the current full-model gap at 32K–128K. Rejected routes stay rejected
unless materially new causal evidence appears.

The cheapest offline checks are existing host-side Q4 wave packing/dequant
reference tests. They can establish layout and scalar arithmetic invariants,
not HIP kernel behavior, resource occupancy, launch cost, or speed. Future GPU
validation must be explicitly scheduled outside this no-GPU goal.
