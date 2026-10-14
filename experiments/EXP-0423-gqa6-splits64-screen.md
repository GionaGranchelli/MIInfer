# EXP-0423 — Packed GQA6 with 64 Split-K partitions

**Status:** REJECTED — full logits/state differ and suffix is slightly slower
**Control:** packed-U4 GQA6 with 32 splits
**Candidate:** packed-U4 GQA6 with 64 splits
**Workload:** restored 73,099-token Q8/Q8 state, 1,024-token suffix

## Hypothesis

Stage1 still accounts for 80.5% of the packed-U4 late-context tile. Doubling
Split-K partitions should halve each CTA's serial key range while adding only a
small Stage2 cost. The workspace is sized for 64 splits. Since partitioning
changes reduction grouping, the candidate must pass exact full-logit and
recurrent-state checks before a speed result is accepted.

## Results

One sequential 32/64 pair ran on each MI50 with the same binary, image, model,
restored state, prompt and graph settings. Route traces confirmed 16 packed-U4
layers at 32 splits for each control graph and 16 at 64 splits for each
candidate graph, confirming graph recapture under the changed split setting.

| GPU / BDF | 32 splits | 64 splits | Relative time | Exact logits/state | Tokens equal | Peak junction |
| --- | ---: | ---: | ---: | :---: | :---: | ---: |
| GPU0 `0000:04:00.0` | 20.547 s | 20.584 s | 0.998x | no | yes | 68 C |
| GPU2 `0000:88:00.0` | 22.433 s | 22.498 s | 0.997x | no | yes | 59 C |

Both candidate runs exited 3 because complete logits and recurrent-state
traces did not match the 32-split control. Outputs were finite and the two
checked greedy tokens matched, which does not satisfy the exactness gate. The
candidate was also marginally slower on both cards. Reject without repeating.
Raw artifacts are retained on Machinist under
`/home/machinist/m31-gqa6-longctx-results/EXP-0423-splits64-screen-GPU0-20261014T1830Z/`
and
`/home/machinist/m31-gqa6-longctx-results/EXP-0423-splits64-screen-GPU2-20261014T1830Z/`.

## Decision

Revert the 64-split selector and workspace increase. Keep 32 splits and packed
U4 as the exact baseline. Do not revisit this split-count change without a
different exact reduction method.
