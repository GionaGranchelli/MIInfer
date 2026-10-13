# EXP-0406 — Correct 1024-token PrefillV2 Macro Tile

**Status:** CORRECTNESS PASSED; 2K WIN; 73K FOLLOW-UP EFFECT NEGLIGIBLE

## Change

Raise the PrefillV2 macro tile and physical batch bound from 512 to 1024, and
extend the affected gfx906 launch guards and Q8 attention dispatch. Set recurrent
state positions from each model-forward call's `base_position` and advance them
after the batch. `RecurrentLayerStateStorage::view()` copies its position into a
temporary view, so the layer's outgoing-position update alone did not update the
model's stored position for the next call.

## Focused result

Machinist GPU2 (`0000:88:00.0`, gfx906), ROCm 7.14, Qwen3.8-27B Q4_K_M,
Q8/Q8 KV, capacity 4096. A 1024-token prefix was prepared before each suffix;
the same 1024 suffix tokens were run as two 512 calls and one 1024 call. Five
interleaved pairs were measured after warm-up.

| Path | Wall samples (ms) | Median (ms) |
|---|---|---:|
| 2 × 512 | 4893.213, 4896.406, 4896.359, 4899.724, 4899.999 | 4896.406 |
| 1 × 1024 | 4690.818, 4691.746, 4698.911, 4701.222, 4696.866 | 4696.866 |

The 1024 tile was 1.042× faster in this short run. Hidden states and logits
matched exactly (`max_abs=0`, cosine `1.0`); greedy first token and one
teacher-forced next token also matched. GPU2 peaked at 66°C. This is a focused
2K-token result, not evidence for a 64K cold-prefill improvement.

## Attention-side check

At a 64K cache endpoint (`prefix=64512`, `queries=1024`), direct Q8 attention
matched two sequential direct-Q8 512 calls exactly. The exact query-tiled Q8
candidate remained slower: median 1278.032 ms versus 958.901 ms direct Q8. It
is not selected for production dispatch.

## Restored long-context check

Machinist GPU2, ROCm 7.14, same model and Q8/Q8 KV. To avoid another cold
prefill, the benchmark restored the existing `sess_43c5693dd01fbab1_p73099`
checkpoint (73,099 cached tokens), then processed the same 1,024 synthetic
suffix tokens using five interleaved pairs. The 1024-token-capacity path is
correct, but its long-context speed advantage is negligible:

| Path | Wall samples (ms) | Median (ms) |
|---|---|---:|
| 2 × 512 | 21893.403, 21920.462, 21916.483, 21915.984, 21892.503 | 21915.984 |
| 1 × 1024 | 21754.684, 21773.613, 21824.815, 21802.999, 21763.410 | 21773.613 |

Speedup was 1.007×. Hidden states, logits, and the teacher-forced next token
matched exactly. The 85°C thermal guard remained clear; peak observed junction
temperature was 67°C, and the container exited 0. The suffix is synthetic, so
this isolates tile behavior on a real restored long-context state; it is not an
agent-workload result.

Profiling one 1,024-token suffix at that state reported 21,685.674 ms total
model GPU time, of which GQA blocks consumed 18,355.050 ms and causal attention
alone consumed 17,411.093 ms (80.3%). Session restore was 2,402.436 ms and is
outside the profiled suffix time. This makes long-context attention the next
optimization target; raising the macro tile is not a meaningful 64K remedy.

## Next

Keep the 1024 tile for correctness and small-prefill gains. Prioritize one
attention-path experiment against the restored long-context state. Do not
repeat the known 48-minute cold-prefill run just to retest this tile change.
