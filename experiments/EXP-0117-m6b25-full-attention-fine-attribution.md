# EXP-0117 — M6-B25 fine full-attention attribution

## Question

Which separable operations make up the 0.400319 ms full-attention preparation
bucket identified by B24 at P64?

## Baseline

The B23 production path was profiled with finer event boundaries. No kernel
selection or numerical behavior was changed.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
ROCm: 7.1.52802-9999
Fixture: /tmp/m6a273-reference-p12
Command: miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --profile64
```

## Correctness and resources

* The instrumented executable built successfully.
* Release CTest passed 20/20 after the profiling change.
* The production path remained B23-selected and used zero profile
  allocations.
* Profile timing was repeated serially; one earlier concurrent run was not
  used for the reported range.

## Results

Across the clean serial profile runs, whole-token GPU time was approximately
90.28–90.60 ms at position 63. Representative layer-3 timings were:

| Stage | GPU ms |
| --- | ---: |
| Attention RMSNorm | 0.0234 |
| Q projection | 0.1986 |
| Q split and 24 head norms | 0.0849 |
| K projection | 0.0328 |
| K head norms | 0.0206 |
| V projection | 0.0666 |
| RoPE and KV store | 0.0165 |
| Cached attention | 0.1384 |
| Attention gate and O projection | 0.1040 |
| Attention residual | 0.0067 |
| FFN norm | 0.0234 |
| FFN Gate/Up | 0.4236 |
| FFN activation | 0.0073 |
| FFN Down | 0.4183 |
| FFN residual | 0.0089 |

Layer 3 measured 1.64–1.69 ms in these runs. The Q projection is therefore
the largest newly separable operation in the attention preparation region.
The Q head-normalization group is the next support-operation candidate.

The B23 recurrent layer-0 reference profile remained approximately:

```text
QKV projection       0.173–0.180 ms
state update         0.208–0.215 ms
FFN Gate/Up          0.424–0.446 ms
FFN Down             0.418–0.441 ms
```

## Interpretation

The former combined bucket was not a single KV or RoPE bottleneck. Q
projection accounts for roughly half of its separable cost; K, V, RoPE, and
KV storage are much smaller. A Q-projection kernel differential is a bounded
next experiment because the repository already has a validated Q4_K×Q8_1
MMVQ primitive used by other Q4 projections.

This result does not justify changing production selection yet. The candidate
must preserve the external observable contract and be judged by TG64/TG128.

## Decision

**MEASUREMENT-ONLY; B25 complete.** No production selection.

## Follow-up

Run one opt-in Q4_K×Q8_1 MMVQ Q-projection candidate for full-attention
layers. Keep the Q/K/V layer structure otherwise unchanged and retain the
current Q4_K×Q8_K path as the control.
