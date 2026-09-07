# EXP-0206 — Layer-Major Chunked Prefill

## Hypothesis

Processing four prompt tokens at one layer before advancing to the next layer
will amortize native Q4_K projection weight traffic and improve production
prefill throughput.

## Baseline

Default M11-A production path on `Qwen3.8-27B-Q4_K_M.gguf`:

```text
P512: 15,728.95 ms, 32.49 tok/s
GPU: gfx906, SCLK 1606 MHz, MCLK 1000 MHz
ROCm: 6.4.0 / LLVM 20
```

## Candidate

Opt-in `MIINFER_PREFILL_LAYER_MAJOR=1` path:

- four-token bounded ping-pong activation buffers;
- native B=4 Q4_K/Q6_K projection kernels;
- deferred recurrent FFN tail with paired and separate SwiGLU variants;
- fused inter-layer normalization handoff buffers.

The default remains disabled.

## Results

| Candidate | P512 prefill | Throughput | Correctness |
| --- | ---: | ---: | --- |
| B=4 QKV/gate, scalar tail | 14,911.64 ms | 34.34 tok/s | short check passed |
| B=4 paired FFN tail | 12,965.37 ms | 39.57 tok/s | long check diverged |
| B=16 shared-LDS prototype | not retained | 33.06 tok/s at P17 | short check passed |

The B=16 kernel was slower than B=4 and was removed. It used a 1024-thread
workgroup with per-tile LDS barriers; synchronization and occupancy cost
outweighed additional weight reuse on gfx906.

## Correctness

Short prompts of one to four repeated `hello` tokens selected the same token
as the baseline. At 17 prompt tokens, the baseline selected `你好` while the
layer-major path selected `hello`. The divergence also reproduced with native
batch projections disabled, so it is not isolated to the B=4 Q4_K kernel.

## Decision

**REJECT as production execution.** The path is retained as an opt-in
diagnostic/profiling implementation, but it must not be enabled by default or
count as a performance win until long-generation token agreement is restored.
The ≥100 tok/s P512 target was not reached.

## Follow-up

Compare per-layer/per-position tensors between token-major and layer-major
execution to locate the first accumulated numerical divergence. Re-run the
performance comparison only after that mismatch is fixed.

## Re-evaluation — EXP-0208

EXP-0208 retained the B=4 layer-major path and batched the previously scalar
full-attention Q/K/V projections. This improved repeated P512 prefill from a
32.20 tok/s token-major median to 39.96 tok/s while matching the tested
generation output. The long-generation qualification and the ≥100 tok/s gate
remain open.
