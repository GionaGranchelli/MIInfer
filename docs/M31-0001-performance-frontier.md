# M31-0001 — Single-MI50 Performance Frontier

## Scope

Measure the frozen M31 N=1 runtime on the same Machinist host, model, image,
and GPU policy recorded by M31-0000. This is qualification only; it does not
change runtime code or tune kernels.

## Fixed matrix

| Context | Cold TTFT | Warm TTFT | Prefill | Decode | VRAM |
|---:|:---:|:---:|:---:|:---:|:---:|
| 8K | required | required | required | TG32/TG64/TG128 | required |
| 32K | required | required | required | TG32/TG64/TG128 | required |
| 64K | required | required | required | TG32/TG64/TG128 | required |
| 128K | required | required | required | TG32/TG64/TG128 | required |

The benchmark uses deterministic synthetic prompts and greedy generation. Cold
TTFT is the first generation after model construction. Warm TTFT is measured
after one same-prompt TG128 warmup. Warm means warmed model/graph execution,
not persistent prefix reuse; that architectural comparison belongs to
M31-0002. The benchmark emits one JSON row per context and decode length.

## Evidence

```text
M31_BASELINE_SHA=99840d10eabe5e64fe2564d34d3d3fcd42b43521
M31_0001_BENCHMARK_SHA=<filled after run>
M31_0001_RESULT=<PASS or FAIL>
```

Results must retain the complete benchmark output, model checksum, context
seeds, decode lengths, VRAM values, and output-parity status. A result with
failed parity is not a valid performance result.
