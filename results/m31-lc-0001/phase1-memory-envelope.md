# M31-LC-0001 Phase 1 — Memory envelope estimate

This is a capacity estimate from the pushed E2-0003Q allocation measurements and current source allocation rules. It is not a context qualification or performance result.

## Inputs and method

- Z840 MI50 reports `34,342,961,152` total device bytes.
- E2-0003Q measured fixed model weights of `24,068,487,168` bytes for fused control and `16,938,170,368` bytes for MMQ-only.
- At 1,280 KV capacity, measured persistent state was `242,745,344` bytes: `158,859,264` bytes of recurrent GDN state plus `83,886,080` bytes of KV.
- The measured 512-token workspace was `727,711,744` bytes; activation buffers were `21,966,848` bytes.
- KV alone scales at the supplied historical rate of 65,536 bytes per capacity token. The model is allocated with `prompt + 256` capacity, matching the existing benchmark.
- Source changes the Split-K workspace from 32 to 3 splits above 66,000 capacity tokens. This reduces that workspace by `367,755,264` bytes; the estimate accounts for this instead of scaling all memory linearly.
- The largest measured post-run device residual above MIInfer's own allocation was `960,813,056` bytes (control). I conservatively add that fixed overhead to both routes.
- Reserve 10% of physical device memory (`3,434,296,115` bytes) for unmodeled allocation variance and other device use. Estimates at or above `30,908,665,037` bytes do not satisfy this margin.

The source-derived estimate is:

```text
route weight bytes + recurrent state bytes + (capacity tokens * 65,536)
+ workspace bytes + activation bytes + 960,813,056-byte observed residual
```

## Estimated usage

Decimal GB; headroom is against the measured `34.343 GB` device total and is before reserving the 10% margin.

| Prompt context | KV capacity | Fused peak estimate | Fused free headroom | MMQ-only peak estimate | MMQ-only free headroom |
|---:|---:|---:|---:|---:|---:|
| 2K | 2,304 | 26.089 GB | 8.254 GB | 18.959 GB | 15.384 GB |
| 4K | 4,352 | 26.223 GB | 8.120 GB | 19.093 GB | 15.250 GB |
| 8K | 8,448 | 26.491 GB | 7.851 GB | 19.361 GB | 14.982 GB |
| 16K | 16,640 | 27.028 GB | 7.315 GB | 19.898 GB | 14.445 GB |
| 32K | 33,024 | 28.102 GB | 6.241 GB | 20.972 GB | 13.371 GB |
| 64K | 65,792 | 30.250 GB | 4.093 GB | 23.119 GB | 11.224 GB |
| 128K | 131,328 | 34.177 GB | 0.166 GB | 27.046 GB | 7.296 GB |

## Capacity interpretation

- 2K through 32K are comfortably inside this estimate for both routes. The 32K route comparison remains conditional on earlier gates and stable thermal behavior.
- At 64K, MMQ-only has estimated room; fused control leaves only about `0.659 GB` beyond the safety reserve. The prior 64K/TG128 correctness failure remains open, so this is not a reason to run that full-prefill fixture automatically.
- At 128K, MMQ-only fits the estimate with about `3.862 GB` beyond the safety reserve. Fused control has only `0.166 GB` estimated free before the reserve and is unsafe under this margin. This does not establish 128K support: no 128K workload or correctness check is authorized by this estimate.
- Estimated memory ceiling among the listed capacities is 64K for fused and 128K for MMQ-only. Both are capacity-only estimates; correctness and runtime feasibility remain unproven at those lengths.

The immediate measured ladder remains 2K, 4K, then 8K. Do not infer throughput or correctness at 16K–128K from this table.
