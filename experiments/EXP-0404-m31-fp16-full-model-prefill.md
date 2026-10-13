# EXP-0404 — M31 FP16 tiled full-model prefill

**Status:** IN PROGRESS

**Baseline:** `88fd63eb0cf8b765f37e5f52168c80783df62e7b`

## Objective

Determine whether the existing query-tiled attention improvement transfers to
full-model prefill with FP16 KV storage, and account for the 64K cache's VRAM
cost. Run matched deterministic 8,192-token prompts with Q8/Q8 control and
FP16/FP16 candidate, capacity 65,536. The candidate enables the existing
opt-in `MIINFER_V2_0043_GQA_ATTENTION` path.

## Gates

- Verify one MI50 BDF and record each mode's static VRAM allocation.
- Record cold prefill wall time and final-logit cosine/max error/top-1 match.
- Keep this as product progress only if FP16 fits with safe VRAM headroom and
  materially lowers full-model latency. Then run the established real coding
  task and the 64K workload before any production default change.

## Result

### Matched 8,192-token exploratory comparison

The benchmark accepts either its original deterministic token-ID input or a
text file encoded with the model's Qwen tokenizer. It records the selected GPU
BDF from inside the process. The 8K comparison used the original deterministic
input and alternated which cache mode ran first across five fresh processes.

| Run | Order | Q8/Q8 seconds | FP16/FP16 tiled seconds |
| ---: | --- | ---: | ---: |
| 1 | Q8, FP16 | 57.340 | 39.901 |
| 2 | FP16, Q8 | 57.860 | 39.922 |
| 3 | Q8, FP16 | 57.919 | 40.013 |
| 4 | FP16, Q8 | 58.120 | 39.956 |
| 5 | Q8, FP16 | 58.073 | 40.019 |

The medians are 57.919 s for Q8/Q8 and 39.956 s for FP16/FP16, a 31.0%
reduction in prefill wall time. Every pair selected the same first token. The
logits comparison was stable across pairs (`max_abs=0.190013766`,
`cosine=0.999931254`). Static allocation rose from 27,300,145,152 B to
29,430,851,584 B (+2,130,706,432 B). This is a five-pair 8K result; it does
not establish long-context correctness or product latency.

A separate one-pair reversed-order check ran one greedy decode step after the
prefill. Both modes generated token IDs `82, 484`. The observed decode times
were 42 ms for Q8 and 40 ms for FP16; one sample is insufficient to claim a
decode speedup. In that pair Q8 prefill was 49.949 s when it ran second, versus
57–58 s across the five-pair set, so order and warm-state sensitivity remains
visible despite the initial interleaving.

The correct GPU selection must be passed into the container with Podman's
`--env ROCR_VISIBLE_DEVICES=2`. Setting it only on the host-side `podman` CLI
left the container on GPU0 (`0000:04:00.0`), where the existing MIInfer service
was using about 29 GB and the benchmark exited on allocation failure. That
invalid run is excluded. The corrected run selected GPU2
(`0000:88:00.0`, UUID `0x2152414173497dd3`); its junction temperature peaked
at 63 C during the five-pair series, with no thermal warning.

### Next gate

Run the FP16 tiled route on exactly 65,536 tokenizer-produced tokens from a
deterministic, relevant MIInfer source-context corpus. Record processed and
available token counts, prompt-file hash, BDF, wall time, logits finiteness,
first two greedy tokens and VRAM. Then assess decode and task-level correctness;
the exploratory 8K logit similarity is not a qualification threshold.
