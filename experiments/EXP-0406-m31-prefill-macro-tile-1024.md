# EXP-0406 — Correct 1024-token PrefillV2 Macro Tile

**Status:** CORRECTNESS PASSED; SMALL PREFILL WIN; COLD 64K NOT YET MEASURED

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

## Next

Use the corrected recurrent-position behavior and 1024 tile for the next
representative long-context suffix workload. Reuse an existing 64K state if
available; do not repeat the known 48-minute cold-prefill run just to retest
this tile change.
