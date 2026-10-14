# EXP-0412 — GQA6 on restored long-context suffixes

**Status:** IN PROGRESS  
**Control:** `c64e52d` production Q8 attention  
**Candidate:** `c64e52d` with `MIINFER_M31_Q8_GQA6_PROD_ARITH=1`

## Hypothesis

GQA6's shared Q8 K/V staging reduces the dominant long-context Q8 suffix
attention cost without changing the production arithmetic sequence. Cold 64K
prefill improved in a matched GPU2 pair; suffix behavior at a 73K restored
prefix is not yet measured for that implementation.

## Bottleneck evidence

EXP-0406 measured one restored 73,099-prefix / 1,024-token Q8 suffix. GQA
blocks took 18,355.050 ms, including 17,411.093 ms causal attention (80.3% of
profiled model GPU time). That is the previous production route; attribution
must be refreshed with GQA6 before choosing a second kernel change.

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

## Expected impact and next decision

The previous 73K profile places 80.3% of 1,024-suffix GPU time in attention;
that is the upper bound for an attention-only improvement on that path. The
new GQA6 contribution is unknown until measured. If exact and materially
faster, use a low-overhead GQA6 profile to select the next stage. If attention
still dominates, implement a bounded cross-token reuse schedule that avoids
the previously rejected register-resident Q2/Q4 design and preserves each
query's Split-K bounds and arithmetic. If it does not dominate, redirect the
next change to the measured bottleneck.

## Prior designs to avoid repeating

- EXP-V2-0023 register-resident cross-query tiles were 10.3x–11.9x slower
  from spills/instruction inflation.
- EXP-0360 six-wave cooperative GQA staging was slower at every tested size.
- EXP-0407 Q8-to-FP16 continuation was faster but failed its numerical gate.

These reject those exact schedules, not the new production-arithmetic GQA6
path or a measured, materially different cross-token schedule.
