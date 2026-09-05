# EXP-0153 — M6-B61 fused beta/alpha preparation

## Question

Can recurrent `beta` sigmoid and `decay` preparation be computed inside the
transposed DeltaNet state-update/output kernel instead of a separate kernel?

## Candidate

The opt-in candidate computes beta and decay once per head in shared memory,
then uses the existing transposed state update, LDS query/key staging, and
output calculation. It removes the separate preparation dispatch but does not
change the production path. Enable with `MIINFER_FUSE_BETA_ALPHA=1`.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Clock:     profile_peak; 1725 MHz SCLK / 1000 MHz MCLK observed
Fixture:   /tmp/m6a273-reference-p12
Baseline:  B59 expanded Q4_K Down production path
```

## Correctness

The candidate built successfully, generated 16 tokens with the production
endpoint, passed deterministic replay, and retained zero decode allocations.
The candidate was not production-selected; the full external contract is not
needed to promote a performance-negative candidate.

## Results

| Workload | Production | Candidate | Change |
| --- | ---: | ---: | ---: |
| TG64 | 14.1819 tok/s | 14.1747 tok/s | -0.05% |
| TG128 | 13.9308 tok/s | 13.9277 tok/s | -0.02% |

Device usage remained `20,094,914,900` bytes and decode allocations remained
zero. The candidate's single preparation dispatch was not sufficient to
offset the added gate/decay arithmetic and shared-memory synchronization in
the recurrent kernel.

## Decision

**REJECT.** Keep the separate beta/alpha preparation kernel and the B59
production recurrent path. Retain the opt-in implementation only as evidence
of the tested llama.cpp-inspired boundary; do not select it without a
materially different resource or scheduling hypothesis.

## Follow-up

Return to whole-token differential profiling. Do not repeat this fusion or
generic dispatch-count reduction without evidence that the added kernel work
will be hidden or eliminated.
