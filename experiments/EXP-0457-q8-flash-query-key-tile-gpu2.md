# EXP-0457 — Q8 FlashAttention query × key tile, GPU2

## Objective

Reduce late-context Q8 GQA6 Stage1 time on MI50 with a real multi-query ×
key-tile dataflow. Preserve Q8 KV and FP32 query/softmax/value accumulation.

## Baseline and measured bottleneck

- Frozen source control: `208a926d26aac126580bc26cd8c36ce0d2808762`.
- Direct-global Q8 GQA6 is the timing control, selected explicitly with the
  current route selector off for all experimental selectors.
- EXP-0421 attributed 17.993 s of a 22.384 s profiled restored suffix tile to
  Stage1. EXP-0443/0446 establish direct-global as the best exact arithmetic
  path, but its reported gain remains below the retain threshold.
- Test one restored 73,099-prefix + tokenizer-backed 1,024-token suffix on
  GPU2 only. GPU0 stays frozen.

## Reference and rejected work avoided

The Qwen grouped-query mapping, Q8 scales and online-softmax contract come
from the existing production GQA6 route. The dataflow uses the standard
streaming FlashAttention recurrence but keeps separate FP32 state per output
row and computes multiple rows against each staged Q8 K/V tile.

This is not the rejected 16-query fragment-reuse path (EXP-0417), the
tile-size/query-reuse screens (EXP-0425/0427), the broken four-key stable
reduction (EXP-0442/0445), the V2-0043 FP16 fragment route, or MFMA (no
gfx906 MFMA instruction was emitted in the pinned compiler screen).

## Fixed design

One CTA owns two adjacent suffix tokens, one KV head and one adjacent pair of
Q heads. Its four logical 32-lane rows are packed into two physical Wave64s;
each logical row owns one `(token, q_head)`. Each row maps lanes over D (eight
dimensions per lane) and computes four keys in ILP, then reduces each score
across lanes. Stage Q8 K/V once per CTA in row-major `[key][dimension]` LDS;
this makes QK dimension loads coalesced and lets each lane reuse its staged
dimension slice across four keys and four query rows. Compute a 4×32 score
tile with FP32 QK arithmetic;
perform a separate stable online-softmax update for each row; then reuse the
same K/V tile for each row's weighted-value update. Each row computes its own
causal length and original ceil-sized split interval. A short final query
tile masks invalid rows.

The one independent variable is the Stage1 Q×K execution/dataflow. Keep model,
Q8 cache, 32 splits, persistent state, graph replay, and output workspace
contract fixed. The feature stays opt-in and defaults off.

## Fixed-operand oracle

Before model execution, compare the implementation to a scalar FP32 reference
on a deterministic captured-operand fixture. Assert every row's token/head/KV
mapping, key scale, causal interval, split interval and partial-output offset.
Compare per-row max, sum, weighted accumulator and normalized output using
max-abs and cosine bounds; exact equality is diagnostic only because the
block-softmax association differs from the sequential production path. Include
the 73,099-prefix adjacent-token split-size transition (suffix tokens 20/21),
the 31/32 cold boundary, first/last split, empty early splits, and a partial
final BK tile. Stage the union of row intervals but mask every score/value by
that row's own original `[k_start,k_end)`.

## Gates

- **Compiler:** gfx906 object; report VGPR, SGPR, spills, scratch, LDS and
  workgroup. Stop on any spills or unexpected scratch.
- **Oracle:** all row-map assertions pass; state/output max-abs <= 1e-4 and
  cosine >= 0.99999 against scalar FP32 fixture. Any mask/head/scale/layout
  error stops the work.
- **Runtime correctness:** correct route observed in all 16 layers, finite
  full logits/state, logit max-abs <= 0.01 and cosine >= 0.9999, recurrent
  state max-abs <= 0.01 and cosine >= 0.9999, checked generated tokens
  recorded. Exact logit/state equality is diagnostic, since the tiled
  FP32-softmax association differs from the production sequence.
- **Performance:** one same-GPU2 direct-global/candidate pair on the restored
  suffix. Candidate must pass numerical gate and reduce suffix time by >=10%
  before any repeat or cold-64K run.
- **Safety:** verify BDF `0000:88:00.0`; thermal warning 80 C, hard stop 85 C.

## Result: rejected on GPU2

The fixed-operand oracle passed and the candidate compiled without spills, but
the restored full-model screen failed both runtime gates. Keep this result as a
negative experiment; remove the candidate from the active source tree.

The initial 512-token request was rejected by the harness before inference:
the restored graph path requires suffix lengths divisible by the 1,024-token
macro tile. The valid 1,024-token pair then ran once on GPU2 using the same
binary, image, model, restored session and tokenized prompt for both routes.

| Route | Restore | Suffix | Prefix + suffix | First / second generated token |
| --- | ---: | ---: | ---: | --- |
| Direct-global control | 2.401 s | 20.901 s | 73,099 + 1,024 = 74,123 | 18,369 / 198 |
| Flash-tile candidate | 2.415 s | 65.483 s | 73,099 + 1,024 = 74,123 | 18,369 / 198 |

The candidate took 3.13× as long as the control (0.319× control/candidate
speedup). Full outputs were finite and generated tokens matched, but numerical
correctness failed: logits max-abs 0.762503 and cosine 0.999659; recurrent
state max-abs 1.642116 and cosine 0.998932 across 48 observed recurrent
layers. The required bounds were max-abs <= 0.01 and cosine >= 0.9999 for both
logits and state. Route tracing confirmed direct-global control in all 16
layers and flash-tile dispatch in all 16 candidate layers, with graph state
active and `token_count=1024`.

The GPU2 process reported BDF `0000:88:00.0`, `gfx906`, and the exact pinned
image digest below. Junction temperature was 27 C at start and peaked at 55 C;
the 80 C warning and 85 C stop thresholds were not reached. The observed
restore times were 2.401 s and 2.415 s. Prompt SHA-256 was
`fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c` and the
model SHA-256, verified with the shared campaign artifact, was
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
Model path was `/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf`; session path
was `/home/machinist/m31-gqa6-longctx-results/EXP-0443-direct-global-gpu0/session.miinfer`.

The test used `localhost/miinfer-dev@sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`,
branch commit `2eff976d7f2722ee7a1d2ed80eadb2dc6579a1d2`, and binary SHA-256
`012863533a2e004b0f75bf8da727de8e89dd412d635168c9c32aa5d9639489c3`. The
command was the `miinfer-m31-fp16-full-model` binary with
`--q8-gqa6-session-pair --session /session.miinfer --prompt-file /prompt.txt
--tokens 1024`, under `ROCR_VISIBLE_DEVICES=2`,
`MIINFER_M31_Q8_GQA6_FLASH_TILE_PAIR=1`,
`MIINFER_M31_Q8_GQA6_PAIR_REPS=1`, and
`MIINFER_M31_TRACE_ATTENTION_ROUTE=1`.

Raw output, preflight, and thermal samples are preserved in
[`results/EXP-0457-gpu2-1024-run.log`](results/EXP-0457-gpu2-1024-run.log),
[`results/EXP-0457-gpu2-1024-preflight.log`](results/EXP-0457-gpu2-1024-preflight.log),
and [`results/EXP-0457-gpu2-1024-telemetry.log`](results/EXP-0457-gpu2-1024-telemetry.log).
The rejected 512-token precheck is in
[`results/EXP-0457-gpu2-512-precheck.log`](results/EXP-0457-gpu2-512-precheck.log).

## Harness

The restored-session pair mode adds `MIINFER_M31_Q8_GQA6_FLASH_TILE_PAIR=1`.
It explicitly uses direct-global GQA6 for the control and toggles only the
flash-tile selector for the candidate. It emits per-layer route traces,
requires a nonempty recurrent-state trace with matching, nonzero recurrent
layer counts, compares complete logits and recurrent state with the numeric
bounds above, and records exact equality and generated-token agreement as
diagnostics. This experiment rejected the candidate; no repeat or cold-64K run
was performed.
