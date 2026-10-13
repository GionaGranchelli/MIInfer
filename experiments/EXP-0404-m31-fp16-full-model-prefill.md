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

### Long-context acceptance gate

The 65,536-token candidate-only execution is complete and recorded below. The
remaining correctness gate is a matched tokenizer-backed shorter prefix and a
late-tile stage profile; the 8K exploratory logit similarity is not a
qualification threshold.

### 65,536-token cold prefill

The pushed benchmark source `504675e` built with the pinned ROCm 7.2.1 image
and ran on Machinist GPU2 (`0000:88:00.0`, UUID `0x2152414173497dd3`). The
model SHA-256 was
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
The tokenizer encoded 198,637 tokens from 31 tracked project-context files;
the benchmark processed the first 65,536. Context file SHA-256:
`fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`.

| Measure | FP16 tiled candidate |
| --- | ---: |
| Processed prompt tokens | 65,536 |
| Capacity, including 4K suffix headroom | 69,632 |
| Prefill wall time | 501.057 s |
| Throughput | 130.8 tok/s |
| First / second greedy token | `17872 / 1420` |
| Logits | finite |
| Static VRAM allocation | 29,699,287,040 B |
| Weights / state / workspace | 24,068,487,168 / 4,722,262,016 / 727,711,744 B |
| Peak sampled junction temperature | 66 C (257 samples) |
| 80 C warning / 85 C stop | neither triggered |

This proves that the candidate completes a genuine tokenizer-backed 64K prefill
with decode headroom. It misses the 180-second target by 2.78x. The prior GPU2
Q8 product run processed 73,099 tokens in 2,298.230 s (31.8 tok/s); the current
run has about 4.1x that throughput, but the prompt and capacity differ, so this
is directional evidence rather than a matched A/B result. No same-prefix 64K
Q8 run was attempted because reproducing the multi-minute control is not needed
to establish the candidate's current latency.

The output's finite logits and two generated tokens establish execution, not
equivalence to the Q8 route. Before promotion, compare the candidate and control
on a tokenizer-backed shorter prefix, preserve decode behavior, and profile the
late 64K tile to identify the next limiting stage. EXP-0404 remains in progress.

### 65,024-position late-tile attribution

The existing profiler replayed a tokenizer-backed 65,536-token prompt on the
same GPU2 and timed the final 512-token tile at base position 65,024. The run
completed successfully with FP16 KV and the opt-in tiled route. On the verified
BDF `0000:88:00.0`, GPU event time for the tile was 5,518.430 ms; the GQA
blocks accounted for 3,733.061 ms, including 3,225.407 ms (58.448%) in causal
attention. All 16 GQA blocks spent about 200–208 ms in causal attention. The
final hidden state was finite. The thermal supervisor sampled 266 times, peaked
at 67 C, and reported `supervisor_failed=0 benchmark_exit=0`.

This is a late-tile FP16-KV profile, not a Q8 full-model attribution or a
512-versus-1024 tile comparison. It identifies attention as a large current
cost, so the next implementation targets Q8 K/V reuse inside the already
query-tiled attention schedule; it does not change the outer macro tile.
