# EXP-0119 — M6-B27 batched full-attention head RMS normalization

## Question

Can the 24 query-head and 4 key-head RMS-normalization loops in each
full-attention layer use one existing batched head-normalization launch per
group without changing the accepted external behavior?

## Candidate and control

The control launches `launch_qwen3_rms_normalize` once per 256-element head.
The candidate uses `launch_qwen3_head_rms_normalize` for 24 query heads and 4
key heads. The primitive was extended to use 256 threads and 256 shared
partials for the model's 256-dimensional heads; the per-head reduction
algorithm remains unchanged.

The candidate is selected by default. Set `MIINFER_BATCH_HEAD_RMS=0` to run
the separate-launch control.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
ROCm: 7.1.52802-9999
Fixture: /tmp/m6a273-reference-p12
```

## Correctness and resources

* Candidate native 16-token replay: **PASS**.
* Candidate 64-layer external observable contract: **PASS**.
* Candidate poisoned-reset replay: **PASS**.
* Release CTest: **20/20** after production selection.
* Candidate and control decode allocations: **0**.
* Candidate and control device bytes: `17019965780`.

## Results

Three serial five-sample process medians per workload:

| Workload | Control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| TG64 tok/s | 11.2861 | 11.4420 | +1.38% |
| TG128 tok/s | 11.1409 | 11.2970 | +1.40% |

Representative layer-3 profile comparison:

| Stage | Separate | Batched |
| --- | ---: | ---: |
| Q split and head normalization | 0.0849 ms | 0.0123 ms |
| K head normalization | 0.0206 ms | 0.0101 ms |

The candidate's representative full-token profile was 89.0983 ms versus
90.1724 ms for a control profile run. Profile timing is supportive; repeated
whole-token throughput is the selection authority.

## Interpretation

The existing primitive's original 128-thread limit prevented direct reuse for
Qwen3.8's 256-dimensional heads. Extending it to 256 threads and batching
independent heads removes the per-head launch repetition while retaining the
same per-head reduction structure. The change produces a repeatable ~1.4%
end-to-end gain on both required decode workloads.

## Decision

**KEEP; production-selected.** Batched full-attention head RMS normalization
is now the default. `MIINFER_BATCH_HEAD_RMS=0` retains the separate-launch
control.

## Follow-up

Refresh the production profile after B27. Do not infer that other groups of
small kernels should be fused without an equivalent correctness and
end-to-end A/B result.
