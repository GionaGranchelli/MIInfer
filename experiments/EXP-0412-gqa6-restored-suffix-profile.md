# EXP-0412 — GQA6 on restored long-context suffixes

**Status:** SCREEN PASSED; NEXT OPTIMIZATION IN PROGRESS
**Control:** `c64e52d` production Q8 attention  
**Candidate:** `c64e52d` with `MIINFER_M31_Q8_GQA6_PROD_ARITH=1`

## Hypothesis

GQA6's shared Q8 K/V staging reduces the dominant long-context Q8 suffix
attention cost without changing the production arithmetic sequence. Cold 64K
prefill improved in a matched GPU2 pair; suffix behavior at a 73K restored
prefix is not yet measured for that implementation.

## Bottleneck evidence

The original profile path omitted production-arithmetic GQA6 dispatch, so its
first 93.3% attention attribution was the old route and is invalid for GQA6.
The profile dispatch was fixed and now fails unless all 16 attention layers
report GQA6.

On the restored 73,099-token prefix, one 1,024-token GQA6 tile took 30,164.812
ms model GPU time on BDF `0000:88:00.0`. GQA blocks account for 26,746.720 ms;
causal attention accounts for 25,780.040 ms (85.464%). Stage 1 accounts for
25,758.596 ms and Stage 2 for 20.269 ms. Other reported work: QKV projection
193.997 ms, RoPE/KV store 6.539 ms, output projection 79.310 ms, post norm
3.005 ms, FFN gate/up 434.009 ms, and SwiGLU/down/residual 247.108 ms. All 16
profiled layers used production-arithmetic GQA6. This makes Stage 1 the clear
next target; Stage 2 reduction is not the bottleneck.

The actual graph-replay suffix comparison used the same restored session,
tokenized text and MI50 for each control/candidate pair:

| Suffix | Production Q8 | GQA6 | Speedup | Graph replay and correctness |
| --- | ---: | ---: | ---: | --- |
| 1,024 | 65.923 s | 30.189 s | 2.184x | 1 tile; all recurrent state/history, logits, and checked greedy tokens exact |
| 4,096 | 269.408 s | 123.400 s | 2.183x | 4 tiles; same exact checks |

These are one sequential matched sample per path and suffix length, suitable
for screening rather than a repeated-sample performance claim. Guarded peak
temperature was 64 C for 1,024 and 67 C for 4,096; the profile run peaked at
61 C junction. All guards exited successfully. Results are on the remote host
under `/home/machinist/m31-gqa6-longctx-results/EXP-0412-20261014/`.

## Workload and one variable

- Restore the existing Q8/Q8 checkpoint `sess_43c5693dd01fbab1_p73099.miinfer`.
- Use the same tokenized real-text suffix, 1,024 tokens and 4,096 tokens, for
  control and candidate. No cold prefill is part of this experiment.
- Use one physical MI50, BDF `0000:88:00.0` (GPU2), with HIP Graph enabled.
- Keep model, checkpoint, suffix, capacity, binary, image and runtime options
  fixed. Change only `MIINFER_M31_Q8_GQA6_PROD_ARITH`.
- Measure restore separately from suffix time. For 4,096, record four
  sequential 1,024-token forwards and graph capture/replay evidence.

## Correctness gate

Both paths must restore all 73,099 prefix tokens, process the same suffix at
positions `73099..`, emit finite output, and agree exactly on each suffix
tile's final hidden state, all final logits, the first generated token and one
subsequent decode token. Any mismatch rejects the candidate for this route.

## Performance and resource gates

Record same-card suffix timings, restore duration, GPU BDF/clocks/power/VRAM,
image and binary identities, and guard samples. Keep the 80 C warning and 85 C
stop active. Stop a run at 85 C, on GPU/process loss, or if it exceeds 3x the
previous Q8 duration without progress. One matched pair per suffix length is
a screening result; repeat only if the comparison is close or noisy.

## Next decision

Keep bit-identical production-arithmetic GQA6 as the best-known path, with the
old Q8 path available for comparison and fallback. The measured GQA6 Stage 1
share justifies a bounded cross-token reuse candidate. Pair adjacent queries
inside one CTA, share only the Q8 LDS K/V tile, and preserve each query's own
Split-K bounds and ordered arithmetic updates. Reject it on any numerical
mismatch, register spill, or lack of material speedup. The candidate must
continue through the restored graph path before a broader context claim is
made.

This experiment does not establish a fast 64K cold prefill: the last known
result remains 1,023.715 s against the 180 s target. It also does not establish
coding-agent usability. A cold run should follow only after a Stage 1 change
shows meaningful end-to-end impact.

## Prior designs to avoid repeating

- EXP-V2-0023 register-resident cross-query tiles were 10.3x–11.9x slower
  from spills/instruction inflation.
- EXP-0360 six-wave cooperative GQA staging was slower at every tested size.
- EXP-0407 Q8-to-FP16 continuation was faster but failed its numerical gate.

These reject those exact schedules, not the new production-arithmetic GQA6
path or a measured, materially different cross-token schedule.
