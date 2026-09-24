# EXP-0390 — P512 runtime-variance boundary

## Question

Attribute current MIInfer P512 run-to-run variance to host submission, GPU
timeline, terminal synchronization, or an insufficient boundary. No
optimization was implemented.

## Measurement

Added one diagnostic-only timer around the existing final
`hipStreamSynchronize(hipStreamPerThread)` under
`MIINFER_EXP0390_RUNTIME_BOUNDARY=1`. EXP-0376 chunk timing supplied the
existing host submission interval and HIP event GPU timeline. No extra sync,
stream change, kernel selector, profiler, or allocation behavior was added.

## P512 admitted samples

| Run | Prefill ms | Host submit ms | GPU timeline ms | Terminal sync ms | Unclassified ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 3183.41 | 46.8308 | 3183.27 | 3136.50 | 0.0792 |
| 2 | 3006.83 | 48.8241 | 3006.70 | 2957.93 | 0.0759 |
| 3 | 3138.06 | 67.1183 | 3137.91 | 3070.85 | 0.0917 |
| 4 | 4987.42 | 59.0083 | 4987.27 | 4928.33 | 0.0817 |
| 5 | 2513.00 | 47.3787 | 2512.86 | 2465.55 | 0.0713 |
| 6 | 4896.71 | 41.5141 | 4896.59 | 4855.12 | 0.0759 |

The accounting identity is `prefill - host_submit - terminal_sync`; it remains
approximately 0.08 ms. The EXP-0376 `gpu_ms` is a stream timeline, not pure
kernel-compute time.

## Fast/slow comparison

The observed clusters define fast as runs 1, 2, 3, and 5, and slow as runs 4
and 6. This split was derived after observing the distribution; no threshold
was selected in advance.

| Metric | Fast median | Slow median | Delta | Slow/Fast |
|---|---:|---:|---:|---:|
| Prefill | 3072.45 ms | 4942.07 ms | +1869.62 ms | 1.608× |
| Host submission | 57.97 ms | 50.26 ms | -7.71 ms | 0.867× |
| GPU timeline | 3072.31 ms | 4941.93 ms | +1869.63 ms | 1.608× |
| Terminal sync | 3014.39 ms | 4891.73 ms | +1877.34 ms | 1.623× |
| Unclassified | 0.0784 ms | 0.0788 ms | +0.0004 ms | 1.005× |

The fast-to-slow wall expansion is tracked almost exactly by the GPU timeline
and terminal wait. Host submission does not expand, and the unclassified
remainder is stable. Host and GPU deltas are overlapping measurements and are
not summed as independent causes.

## Sentinel and resource evidence

The periodic mx sentinel remained stable: four samples were approximately
`2308.99, 2310.72, 2310.93, 2310.93 ms`; no system-wide instability
reappeared. Continuous telemetry held SCLK 1606 MHz and MCLK 1000 MHz. The
junction temperature reached 61 C; no throttle flag, ROCm error, or stale
MIInfer/llama-bench process was observed. Each P512 timing run reported
`alloc_delta=0`, `total_bytes_delta=0`, and `live_bytes_delta=0`.

## Decision

**Classification: GPU_RUNTIME_VARIANCE.** Host submission is stable, while the
GPU stream timeline and its terminal completion wait expand together by about
1.87 s on slow runs. This localizes the variance to queued GPU/driver
execution, not host orchestration or an unclassified reporting interval. The
terminal sync is the observation point for completion, not evidence of a
second independent synchronization problem.

B128 attribution authorized: **NO**.

B64: **REJECTED — CURRENT M28 FRONTIER**.

B4: **BLOCKED**.

ONE next PRIMARY:

> Split the GPU timeline at the coarsest existing recurrent-versus-attention
> family boundaries, using the same fast/slow P512 contract, to determine
> which GPU phase expands. Do not inspect individual kernels until that split
> identifies a dominant family.

Explicitly not authorized: optimization, B128, residual architecture, B64,
B4, source bisect, synchronization changes, stream changes, precision changes,
launch tuning, or kernel tuning.

No optimization candidate was implemented. EXP-0390 only localized the P512
run-to-run variance to a coarse MIInfer GPU-runtime boundary.

## Provenance

Experiment commit and graph SHA are added after graph refresh. Final working
tree status is recorded in the provenance commit.
