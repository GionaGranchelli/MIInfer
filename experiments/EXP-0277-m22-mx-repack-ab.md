# EXP-0277 — M22 mx-llama native repack A/B

## Hypothesis

The compatible gfx906 reference's prompt advantage depends materially on its
native quantized weight layout.

## Baseline and candidate

* Reference checkout: `/home/fedora-workstation/Development/mx-llama.cpp`
* Commit: `2e9d29fe736969160f17476ec6f0a6298cee6966`
* Candidate: normal native repack path
* Control: `-nr 1` / `--no-repack`
* Full GPU offload, Flash Attention, Q8_0 K, F16 V, `-b 2048 -ub 2048`

Model SHA-256 is
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
GPU state was MI50/gfx906 at SCLK 1606 MHz and MCLK 1000 MHz.

## Benchmark

Raw outputs:

* `results/m22-reference-repack/20260909-2019/pp.json`
* `results/m22-reference-repack/20260909-2019/tg64-depth.json`

Both use three repetitions. PP uses P512/P2048/P8192. The decode run uses
`-p 0 -n 64 -d 512,2048,8192` so the depth is explicit.

## Results

### Prompt processing

| context | repack ON tok/s | `-nr` tok/s | no-repack delta |
|---:|---:|---:|---:|
| P512 | 222.55 | 199.09 | -10.5% |
| P2048 | 236.71 | 213.74 | -9.7% |
| P8192 | 229.39 | 207.87 | -9.4% |

### TG64 after explicit context depth

| depth | repack ON tok/s | `-nr` tok/s | no-repack delta |
|---:|---:|---:|---:|
| P512 | 25.38 | 22.77 | -10.3% |
| P2048 | 25.19 | 22.65 | -10.1% |
| P8192 | 23.82 | 21.51 | -9.7% |

All samples were stable across the three repetitions. This is a matched
cache/layout comparison, not a MIInfer comparison.

## Decision

KEEP native repacking as the primary reference configuration. REJECT treating
the canonical-layout `-nr` path as an equivalent competitor. MIInfer's first
parity experiment must account for layout and batch execution together.

## Follow-up

Use the repacked reference for all M22 ratios and retain `-nr` only as a
diagnostic control.
