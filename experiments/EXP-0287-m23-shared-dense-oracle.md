# EXP-0287 — M23 shared dense projection oracle

**Status:** RETEST
**Milestone:** M23
**Date:** 2026-09-10

## Hypothesis

One reusable FP16 weight destination plus one reusable quantized source scratch
can exercise all wide dense projections without retaining six private FP16
copies per recurrent layer.

## Candidate

`MIINFER_PREFILL_WIDE_DENSE_ALL=1` with the P512 layer-major schedule. Each
QKV, gate, SSM-out, FFN gate/up, and FFN down tensor is uploaded from its
memory-mapped GGUF view, expanded into the shared FP16 destination, and used by
one rocBLAS batched projection. Decode remains unchanged.

The rocBLAS batch wrapper was corrected to compute `weights^T * input` into
token-major output. The prior argument order produced transposed batched
outputs and non-finite logits.

## Environment

* AMD Instinct MI60 / MI50, gfx906, 32 GiB HBM
* ROCm/HIP 7.1.52802, clang 20.0.0.rocm
* `Qwen3.8-27B-Q4_K_M.gguf`
* context capacity 1024; exact 512-token repeated fox prompt
* `MIINFER_PREFILL_LAYER_MAJOR=1`
* `MIINFER_PREFILL_WIDE_CHUNK=1`
* `MIINFER_PREFILL_FULL_LAYER_MAJOR=1`
* `MIINFER_PREFILL_CHUNK=512`
* repacked MMQ attention enabled; `--max-tokens 0` for PP-only timing

## Correctness

The focused HIP test passes in debug and release, including repacked MMQ
token counts B128/B129/B256/B512. With `MIINFER_WIDE_VALIDATE=1`, the dense
wide recurrent layer reports `qkv_max_abs=0.0761`, `gate_max_abs=0.0665`,
`state_max_abs=0.0766`, and `output_max_abs=0.0495` against canonical layer
execution. A one-token continuation emits the valid token `brown`.
The final post-allocation-gating continuation also emitted `brown` (8873.05 ms
P512, 57.70 tok/s; 2.78 ms first token).

## Results

| path (before final scratch-allocation gating) | run 1 | run 2 | peak allocation |
|---|---:|---:|---:|
| repacked MMQ baseline | 9199.17 ms (55.66 tok/s) | 9339.36 ms (54.82 tok/s) | 19,468,038,484 B |
| shared dense oracle | 8832.01 ms (57.97 tok/s) | 8521.70 ms (60.08 tok/s) | 19,468,038,484 B |

A final post-allocation-gating pair measured baseline 8734.09 ms (58.62
tok/s) and oracle 9192.11 ms (55.70 tok/s). The earlier two pairs therefore
do not establish a stable oracle win; the matched final pair regressed 5.0%.

The best earlier pair was about 8% faster for the oracle, which adds 73,113,600
B of reusable source scratch. This is not a qualified performance win: it has
not reached the M23 P512 gate of 100 tok/s, and run-to-run variance reverses
the result.

The non-validation P512 profile identified the current whole-schedule buckets:
FFN gate/up 30.36%, FFN down 14.86%, deferred attention tail 12.00%, batch
prepare 10.97%, projection/norm 10.69%, KV/head norm 9.15%, and projection
attention output/rope 9.63% combined. These buckets are the next profiling
targets; no micro-tune is accepted from this experiment.

## Decision

RETEST. Keep the shared staging architecture as the correctness/performance
oracle; do not claim end-to-end acceptance. Profile the whole P512 schedule
before attempting micro-kernel tuning.
