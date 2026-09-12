# EXP-0338 — Mx attention decode weight reuse

Status: KEEP opt-in; next validation required

## Question

Can attention decode consume the already packed M25-H/I Mx O and FFN weights,
eliminating the duplicate resident M23 attention-FFN representation without
changing the qualified H/I prefill contract?

## Baseline

The qualified H/I path uses Mx attention weights for P512 prefill but retains
resident M23 attention O and FFN copies for scalar decode. The exact qualified
baseline is the six-sample median from EXP-0322:

- P512: `2500.62 ms`, `204.75 tok/s`
- allocation: `21,993,242,964 B`
- Mx MMV: disabled

## Candidate

`MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_DECODE=1` routes attention O and FFN
decode through the existing Mx Q8/MMQ kernels and H/I packed weights. M23
Q/K/V decode remains unchanged. The candidate also skips packing and
allocating the redundant M23 attention FFN copies. Mx MMV remains a separate
opt-in selector.

The implementation is MIInfer-owned orchestration around the pinned
`mx-llama.cpp` Mx MMQ/MMV execution contract; it does not vendor the external
runtime.

## Environment and benchmark

Exact target model and prompt are the M25 qualification fixtures:

- Qwen3.8-27B-Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- MI50/gfx906, SCLK/MCLK `1606/1000 MHz`
- context capacity `1024`, P512 repeated-fox prompt
- full-layer-major B512 H/I preset, `MIINFER_MX_Q8_BATCH=0`
- fresh-process candidate screen, three samples

## Correctness

All candidate screen runs passed the first-token and continuation checks:
`13477` and `37550`. The 128-token generation with both attention decode Mx
reuse and `MIINFER_MX_MMV=1` produced the expected finite repeating sequence.
The focused host/GPU tests also passed:

```text
3/3 passed: kquant-wave-host, q4-q8-gemv-correctness, qwen3-primitives-gpu
```

## Results

### Mx attention decode with MMQ (`MIINFER_MX_MMV=0`)

| run | P512 ms | prompt tok/s | allocation |
| --- | ---: | ---: | ---: |
| 1 | 2485.77 | 205.97 | 18,472,649,044 B |
| 2 | 2503.89 | 204.49 | 18,472,649,044 B |
| 3 | 2407.58 | 212.66 | 18,472,649,044 B |
| median | **2485.77** | **205.97** | **18,472,649,044 B** |

The fresh control screen was noisy (`2423.43`, `2598.96`, and `2846.54 ms`),
so the existing six-sample EXP-0322 qualification remains the comparison
used for promotion decisions.

### Mx attention decode plus Mx MMV

One long-generation screen measured `2381.34 ms` P512 (`215.00 tok/s`) and
`7536.89 ms` for 128 decode tokens (`16.98 tok/s`, `58.882 ms/token`). This
should not be compared as a P512 qualification: Mx MMV is an independent
decode optimization. EXP-0335 previously measured `5.28 tok/s` for Mx MMV
without attention decode reuse, so the result supports the intended whole
attention-decode routing.

## Interpretation

The candidate removes exactly `3,520,753,920 B` (`~3.28 GiB`) of reported
allocation. Its P512 screen is within the existing H/I variation and does not
close the external `220.892 tok/s` stretch gap. It does, however, preserve
correct continuation and materially improves the measured long decode when
combined with Mx MMV.

## Decision

**KEEP as opt-in.** Do not add the selector to `m25_hi_qualified` or make it a
default until it survives a clock-qualified six-sample P512 interleaved A/B,
long-context testing, and a power/thermal check.

## Follow-up

Run the six-sample A/B with
`MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_DECODE=1`, then qualify the combined
Mx attention decode and Mx MMV path at the intended decode/context workload.
