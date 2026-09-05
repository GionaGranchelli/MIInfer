# EXP-0116 — M6-B24 full-attention stage attribution

## Question

After B23, which stages dominate a representative Qwen3.8 full-attention
layer at P64, and what should be measured or optimized next?

## Baseline

The production-selected B23 path was profiled without changing execution,
quantization, or kernel selection. B23 uses the packed Q6_K×Q8_K dot4
recurrent-QKV projection.

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

* The default B23 path passed Release CTest 20/20 before this measurement.
* The profile completed successfully with zero profile allocations.
* No production behavior was changed by B24.

## Results

Whole-token profile at position 63:

| Metric | Result |
| --- | ---: |
| Total GPU event | 90.2061 ms |
| Layer-event sum | 86.9818 ms |
| Final norm | 0.02320 ms |
| Final Q8 | 0.00816 ms |
| Final LM head | 2.46784 ms |
| Final argmax | 0.46912 ms |
| Allocations | 0 |

Layer 3 full-attention stage profile:

| Stage | GPU ms |
| --- | ---: |
| Attention norm | 0.02480 |
| Q/K/V projections, head norms, RoPE, KV store | 0.400319 |
| Cached attention | 0.13840 |
| Attention gate and O projection | 0.10528 |
| Attention residual | 0.00672 |
| FFN norm | 0.02304 |
| FFN Gate/Up | 0.422719 |
| Activation, Down, residual | 0.42880 |
| Layer 3 event | 1.58544 |

The profile still has 1,553 dispatches/token and 38 synchronization sites in
the production harness. B23's representative recurrent layer 0 stages were:

```text
attn_norm                 0.02384 ms
qkv_projection            0.155999 ms
gate_projection           0.08400 ms
beta_alpha                0.02976 ms
conv_and_head_norm        0.01488 ms
state_update              0.22656 ms
recurrent_gate            0.01232 ms
ssm_output_projection     0.07152 ms
attention_residual        0.00656 ms
ffn_norm                  0.02336 ms
ffn_gate_up               0.423999 ms
ffn_activation            0.00704 ms
ffn_down                  0.41824 ms
ffn_residual              0.00912 ms
```

## Interpretation

B24 is measurement-only. Full-attention layer 3 has no single newly isolated
winner yet: the largest bucket combines Q/K/V projection, 28 per-head norm
operations, RoPE, and KV storage. The cached-attention kernel itself is only
0.1384 ms at P64 for this layer. FFN work remains substantial, but its
projection paths have already been improved and validated by the B18/B19
experiments.

The next useful step is to split the 0.400319 ms bucket into its existing
operation boundaries before implementing a candidate. The resulting choice
must be based on removable work or a demonstrated same-contract kernel
differential, not on dispatch count alone.

## Decision

**MEASUREMENT-ONLY; B24 complete.** No production selection.

## Follow-up

Make the next milestone a narrower full-attention Q/K/V preparation
attribution. Prioritize the largest separable operation among projection,
per-head normalization, RoPE, and KV store. Do not reopen B23 or start a new
fusion experiment until that attribution exists.
