# EXP-0335 — M25 pinned Mx single-token MMV

## Hypothesis

Decode currently invokes the 128-token Mx MMQ kernel with `token_count=1`.
Porting the pinned repacked single-token MMV contract should remove the
unused token-tile work and improve decode without changing P512 prefill.

## Baseline

The existing staged Mx MMQ launch is used for every token count, including the
resident-all decode projections.

## Candidate

An opt-in `MIINFER_MX_MMV=1` path launches a 1024-thread, 16-output-row
single-token kernel for the existing MI50 Mx Q4/Q5/Q6 repacked layouts. Each
Wave64 row shard walks its Q8 input groups, reduces in-wave, and writes one
output vector. The layout/arithmetic is MIInfer-owned and the shape is
adapted from the pinned mx q8_repack MMV path.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M GGUF, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Release build, exact 512-token repeated-fox prompt
- context capacity `1024`
- clean `env -i` processes with the explicit M25 H/I vector
- `MIINFER_MX_Q8_BATCH=0`; `MIINFER_MX_PIPELINE` unset
- five fresh interleaved `--repeat-p512-check` pairs
- continuous telemetry: 1588 samples, all SCLK/MCLK `1606/1000 MHz`
- maximum observed junction temperature `60 C`
- candidate and control allocation: `21,993,242,964 B`

## Correctness

All ten repeated checks returned:

```text
same_process_p512_check=PASS
first_token=13477(brown)
continuation_token=37550
```

The candidate also completed a 128-token no-stream generation with finite
output and no delayed state failure.

## Results

Five-pair P512 and one-token continuation samples:

| path | P512 ms | continuation ms | repeat P512 ms |
| --- | --- | --- | --- |
| MMQ control | 2479.23, 2507.85, 2488.44, 2510.91, 2505.06 | 516.723, 485.286, 505.314, 517.427, 514.626 | 2491.49, 2482.00, 2490.26, 2492.90, 2489.48 |
| MMV candidate | 2501.46, 2382.82, 2467.54, 2503.25, 2488.66 | 122.947, 122.915, 122.986, 123.003, 123.023 | 2463.02, 2505.13, 2454.85, 2454.42, 2494.26 |

Medians:

| path | P512 ms / tok/s | continuation ms | repeat P512 ms / tok/s |
| --- | ---: | ---: | ---: |
| MMQ control | 2505.06 / 204.38 | 514.626 | 2490.26 / 205.59 |
| MMV candidate | 2488.66 / 205.73 | 122.986 | 2463.02 / 207.88 |

The candidate lowers one-token continuation latency by `76.1%` (`4.19x`)
while P512 changes only `-0.65%`, within prefill run-to-run variation.

An additional 128-token generation measured:

| path | decode ms | decode ms/token | decode tok/s |
| --- | ---: | ---: | ---: |
| MMQ control | 74720.82 | 583.756 | 1.71 |
| MMV candidate | 24238.30 | 189.362 | 5.28 |

## Interpretation

The current wide MMQ kernel is the wrong shape for token-by-token decode. The
single-token MMV path gives a large, correctness-valid decode improvement and
does not materially move P512 prefill.

## Decision

**KEEP as opt-in.** Do not add `MIINFER_MX_MMV` to the qualified P512 preset;
the decode-only path needs longer-generation and broader context qualification
before becoming default.

## Follow-up

Run a longer decode qualification and compare power/tokens-per-joule before
promoting the MMV selector or changing the versioned preset.
