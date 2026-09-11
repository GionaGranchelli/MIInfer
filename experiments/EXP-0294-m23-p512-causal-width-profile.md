# EXP-0294 — M23 P512 projection / B64 causal-width profile

## Hypothesis

Projection throughput and the causal WY scan should be independently sized: a
B512 MMQ projection batch must not imply a B512 recurrent-state scan.

## Candidate

Use the wide layer-major path with `MIINFER_PREFILL_CHUNK=512` and keep the
causal scan at the proven `kM23CausalChunk=64`. Profile the complete P512
schedule, including repacked-weight bytes and synchronous upload time.

## Environment

- AMD gfx906, Qwen3.8-27B-Q4_K_M, context capacity 1024
- full layer-major, wide/repacked MMQ, row-128 reader
- recurrent FFN residency enabled
- exact 512-token repeated fox prompt

## Correctness

The focused release convolution/MMQ test passes, including generic and
repacked Q4/Q6 MMQ at B128/B129/B256/B512. The P512 path also completed a
one-token continuation with `brown`.

## Results

The profiled run reported:

```text
projection_batch_width=B512 causal_chunk_width=B64
allocation=29,956,706,644 B
prefill=81.32 tok/s (6,296 ms)
repacked_weight_upload_bytes=9,463,398,400
repacked_weight_upload_ms=6,124.08
```

The largest remaining upload buckets were recurrent QKV (2,123,366,400 B),
recurrent gate and SSM-out (1,226,833,920 B each), and attention FFN
Gate/Up/Down (about 1.16/1.16/1.20 GB each). The profile therefore points to
weight residency/placement, not causal-scan micro-tuning, as the next measured
architecture question.

## Decision

**RETEST.** The width decoupling is implemented and observable, but the
M23 >=100 tok/s P512 gate remains open. Do not claim an end-to-end win.

## Follow-up

Only pursue replacing native decode buffers with opt-in resident projection
storage if a VRAM-safe restore path is designed and measured; otherwise retain
the current opt-in recurrent-FFN residency candidate.
