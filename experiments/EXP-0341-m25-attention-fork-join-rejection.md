# EXP-0341 — M25 attention fork/join concurrency

**Status:** REJECT; removed after measurement  
**Milestone:** M25 stretch investigation  
**Date:** 2026-09-12  
**Baseline:** `23a89ca`  
**Reference:** `mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`

## Hypothesis

The pinned oracle uses independent nonblocking streams for attention branches.
Overlapping MIInfer's combined Q/K projection with V at B512 might reduce
P512 wall time without changing numerical kernels or layouts.

## Candidate

A temporary opt-in selector, `MIINFER_PREFILL_WIDE_ATTN_CONCURRENT_QKV=1`,
created two nonblocking streams, launched the existing combined Q/K and V M23
MMQ paths, and joined them before downstream attention work. This was the
smallest MIInfer-compatible test; it did not attempt the oracle's separate
Q/K/V graph.

## Environment

- AMD Instinct MI50, gfx906, Wave64
- Qwen3.8-27B-Q4_K_M, model SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- 512-token shape-control prompt, context capacity 1024, `--max-tokens 0`
- clean `env -i` processes; control was `MIINFER_PRESET=m25_hi_qualified`
- candidate enabled only the temporary selector
- clocks were spot checked at 1606/1000 MHz; this was not a qualification run

The prompt was generated from 51 repeated fox sentences plus ` a` to obtain
exactly 512 tokens. It controls the workload shape but is not the canonical
continuation prompt.

## Results

| sample | control ms | candidate ms |
| ---: | ---: | ---: |
| 1 | 2942.87 | 2486.44 |
| 2 | 2516.92 | 2496.81 |
| 3 | 2450.57 | 2504.35 |
| 4 | 2508.35 | 2476.30 |
| median | 2512.635 | 2491.625 |

The candidate median was 0.836% faster: 205.49 versus 203.77 tok/s. The
first control sample was a retained cold outlier; excluding it does not create
a material promotion case.

## Correctness

With two generated tokens, control and candidate produced byte-identical
stdout. Control prefill/decode measured 2497.39/487.88 ms; candidate measured
2417.39/487.48 ms. The focused GPU and primitive CTest subset passed 3/3.

## Stall observation

An earlier invalid 511-token test timed out while its child remained attached
to `/dev/kfd`, holding approximately 22.9 GB. After the orphan was terminated,
VRAM returned to 10.9 MB and the valid 512-token control completed. This is
direct evidence of a process/harness lifecycle failure, not a source regression
in H/I or J.

## Decision

**REJECT.** The temporary streams/events were removed. The result is below the
run-to-run spread and does not challenge the rejected static-graph result or
the qualified preset. The default and `m25_hi_qualified` paths are unchanged.

## Follow-up

Make timeout harnesses clean up children and check `/dev/kfd` ownership before
calling a run a code regression. Continue M25-L with matched recurrent-FFN
stage identity rather than another generic launch-concurrency candidate.
