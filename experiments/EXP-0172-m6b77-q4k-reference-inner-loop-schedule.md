# EXP-0172 — Reference Q4K inner-loop load schedule

## Hypothesis

Matching llama.cpp's Q4_K x Q8_1 MMVQ schedule—preloading Q8 words and scales
before the unrolled DP4A accumulation—would reduce instruction dependency and
register/l memory stalls in MIInfer's dominant Q4K matvec path.

## Baseline

Current MIInfer Q4K MMVQ dot, five-run TG64 median: 4481.76 ms / 14.280 tok/s.

## Candidate

Preload the two Q8 block word pairs and half scales into local arrays, then run
the same two-part DP4A/reduction loop.

## Correctness

Native replay: PASS. Decode allocations: 0.

## Results

Candidate five-run TG64 timings: 4472.96, 4479.31, 4485.25, 4487.79,
4487.85 ms. Median: 4485.25 ms / 14.269 tok/s.

Delta versus baseline: -0.08%.

## Decision

REJECT. The reference-like load schedule is not a material source of the
27 ms/token gap. Restore the baseline inner loop; do not revisit as a tuning
variant.

## Follow-up

The remaining redesign must change work sharing or the packed representation,
not instruction ordering within the existing one-row MMVQ kernel.
