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

## Short-context reference points

The frontier also retains the existing V2-0045-compatible short-context
reference points, without changing that benchmark's output convention:

| Reference | Prefill | Decode |
|---:|:---:|:---:|
| P512 | required | TG128 required |
| P2K | required | TG128 required |
| P4K | required | TG128 required |

Run these with `miinfer-v2-0045-qual-bench` on the same frozen host, model,
image, GPU, and clock policy. Record the complete stdout, command line, and
model checksum alongside the frontier JSONL. These are comparison references,
not replacements for the 8K/32K/64K/128K frontier matrix.

## Evidence

```text
M31_BASELINE_SHA=99840d10eabe5e64fe2564d34d3d3fcd42b43521
M31_0001_BENCHMARK_SHA=d9648222ec0f8402876a4239b13a01c864d1015f
M31_0001_RESULT=FAIL
```

Results must retain the complete benchmark output, model checksum, context
seeds, decode lengths, VRAM values, and output-parity status. A result with
failed parity is not a valid performance result. The short-context evidence
must additionally retain one prefill and one TG128 decode result for each of
P512, P2K, and P4K.

The short-context reference run completed on Machinist card1 with all six
invocations returning `rc=0`; its complete stdout is retained in
`results-m31-0001-short-context-card1.txt`, with command and environment
metadata in `results-m31-0001-short-context-card1-evidence.txt`. The model
checksum in every row is
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
