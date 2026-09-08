# EXP-0242 — M11-B recurrent fused normalization/Q8 rejection

## Hypothesis

The recurrent layer-major tail currently launches a standalone Q8_1
quantization kernel after fused residual-add/RMS normalization before FFN
gate/up. The existing fused normalization primitive can emit the same Q8_1
format directly, potentially removing one launch and one normalized-buffer
read per token.

## Baseline and candidate

- Baseline: fused residual-add/RMS normalization, then standalone Q8_1
  quantization of the 5120-wide normalized output.
- Candidate: request Q8_1 emission from the fused normalization kernel and
  consume that buffer directly in FFN gate/up.

The candidate was production-path only for recurrent deferred tails and was
controlled by `MIINFER_PREFILL_FUSED_NORM_Q8=0` for A/B runs.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Build: `mi50-release`
- `MIINFER_PREFILL_LAYER_MAJOR=1`
- `MIINFER_HIP_GRAPH=0`
- Repetitions: one matched A/B pair at each prompt size

## Correctness

P17/TG1 fused and control runs produced the same continuation text hash:
`568e8341594b957c0a15c6a734a5896c3da3de82eb63512905f152701cc22185`.
Both runs completed with finite output.

## Results

| Prompt | Control | Candidate | Candidate speedup |
| ---: | ---: | ---: | ---: |
| P129 | 2709.06 ms | 2710.48 ms | 0.999x |
| P513 | 11080.57 ms | 11064.28 ms | 1.001x |

The P513 difference is `+0.15%`, within run-to-run variation. Removing the
standalone quantization launch does not change end-to-end production time;
the deferred FFN projections remain dominant.

## Decision

**REJECT.** The fused Q8 emission path was removed from the production
prefill change. The existing normalization primitive remains available for
other measured uses.

## Follow-up

Do not spend further M11-B effort on isolated normalization/quantization
launch removal unless a new profile shows those launches becoming material.
