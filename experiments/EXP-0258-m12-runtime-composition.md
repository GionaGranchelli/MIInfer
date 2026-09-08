# EXP-0258 — M12 runtime composition and 128-token schedule

## Hypothesis

The corrected chunkwise Gated DeltaNet path can be composed with the existing
layer-major prefill schedule without changing decode, and a 128-token schedule
can expose the dense B128 projection experiment.

## Candidate

- one shared five-buffer GDN workspace across all recurrent layers;
- one shared raw recurrent-output buffer;
- chunkwise GDN plus existing ordered B=4 attention/FFN tails;
- opt-in 128-token staging capacity, with 64-token GDN subchunks;
- no decode-path changes.

## Environment

- GPU: AMD Instinct MI50 / gfx906
- Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
- Build: mi50-release, Release
- ROCm: 6.4.0 / LLVM 20
- Baseline: M11-B frozen production path, commit 556a82a
- Runtime: MIINFER_PREFILL_LAYER_MAJOR=1, MIINFER_HIP_GRAPH=0

## Correctness

The standalone corrected 16-key-head/48-value-head/state-128 kernel reports
maximum output error 1.4e-8 and final-state error 1.5e-7. The integrated P64
one-token greedy check produced the same output as control. Both 64- and
128-token runtime schedules completed an exact 512-token prompt without GPU
faults.

## Results

| Schedule | Prompt | Time | Prefill |
| --- | ---: | ---: | ---: |
| GDN, chunk 64 | 512 | 10725.02 ms | 47.74 tok/s |
| GDN, chunk 128 | 512 | 10746.47 ms | 47.64 tok/s |

Earlier P512 integration runs measured 47.79 tok/s for the chunkwise path and
46.50 tok/s for the matched non-chunkwise control; those runs used the same
layer-major schedule but a 511/512-token prompt pair.

## Interpretation

The shared workspace fixes the memory scaling problem of per-layer allocation,
and the corrected runtime composition is operational. The 128-token schedule
does not produce a measurable gain at this workload. Dense staging now has a
separate opt-in production-shaped backend in EXP-0259; it remains out of the
default runtime.

## Decision

KEEP the 64-token GDN integration as an opt-in experiment; DO NOT PROMOTE
128-token scheduling. Decode remains unchanged. EXP-0259 records the separate
opt-in dense backend and its repeated A/B result; it is not promoted by
default.
