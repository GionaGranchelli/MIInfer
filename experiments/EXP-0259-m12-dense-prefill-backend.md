# EXP-0259 — M12 dense FFN-down prefill backend

## Hypothesis

The B128 Q4_K/Q6_K staging result can survive runtime integration if the
canonical quantized matrix remains device-resident and only the currently
needed matrix is expanded into a shared FP16 workspace.

## Candidate

The opt-in backend is enabled with
MIINFER_PREFILL_DENSE_FFN_DOWN=1 and uses:

- one canonical Q4_K/Q6_K FFN-down source per recurrent layer;
- one shared FP16 weight workspace;
- one shared FP16 activation workspace;
- one persistent hipBLAS handle;
- FP32-to-FP16 activation conversion and FP16-to-FP32 GEMM output;
- existing B=4 upstream recurrent/FFN preparation;
- unchanged default and decode paths.

The FP16 matrix is not retained per layer. B256 was tested as a runtime
capacity extension and rejected because the current per-layer activation
allocation reaches HIP out-of-memory during initialization.

## Environment

- GPU: AMD Instinct MI50 / gfx906
- Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
- Build: mi50-release, Release
- ROCm: 6.4.0 / LLVM 20
- Prompt: exact 512 tokens
- Control: layer-major, HIP graphs off, chunk 64
- Candidate: same, dense FFN-down enabled, chunk 128

## Correctness

The backend completed exact P64 and P512 prompts without GPU faults. The
earlier P64 one-token greedy comparison produced the same continuation as the
control. Q4 and Q6 staging paths are selected from the model tensor type; the
full model run exercised the integrated path without numerical/runtime errors.

## Results

| Run | Time | Prefill |
| --- | ---: | ---: |
| Control A | 11001.31 ms | 46.54 tok/s |
| Dense B128 A | 9943.01 ms | 51.49 tok/s |
| Control B | 10972.19 ms | 46.66 tok/s |
| Dense B128 B | 9917.44 ms | 51.63 tok/s |

Aggregates: control 46.60 tok/s; dense B128 51.56 tok/s; improvement
approximately 10.6%.

The isolated exact FFN-down benchmark remains stronger: B128 repack-plus-GEMM
measured 2.736x over repeated B=4 launches. Runtime integration is lower
because Q/K/V, recurrent output, gate/up, attention, and ordered dependencies
remain on the existing path.

## Memory

The backend adds the current layer's canonical quantized FFN-down copy and
shared FP16 workspaces, but does not expand the whole model to FP16. The
worst-case canonical source bound for 48 recurrent layers is approximately
3.34 GiB; B256 activation allocation is not viable in the current layout.

## Decision

KEEP as an opt-in M12 prefill backend. It demonstrates a reproducible
production-shaped gain and preserves decode isolation, but it does not meet
the aspirational 60 tok/s P512 gate. Do not promote it as the default path or
expand all projections until the remaining tail is measured and independently
justified.

## Follow-up

The independent backend was later combined with the corrected chunkwise GDN
path in EXP-0260; the exact-P512 result was 52.99 tok/s.

## Rejected extension — 2026-09-08

A short-lived extension also staged FFN gate/up through the same GEMM path.
It was rejected before benchmarking: retaining one additional canonical Q4
gate and one additional canonical Q4 up source per recurrent layer caused
HIP out-of-memory during model initialization on the 32 GiB MI50. The
down-only backend is therefore the retained production-shaped experiment;
gate/up expansion needs a redesigned memory plan, not more persistent copies.
