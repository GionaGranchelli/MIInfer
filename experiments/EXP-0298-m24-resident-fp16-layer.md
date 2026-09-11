# EXP-0298 — M24 direct resident-FP16 recurrent layer

## Hypothesis

The resident M23 MMQ tiles can execute one recurrent layer through reusable
FP16 scratch and rocBLAS without restoring canonical quantized device buffers.

## Candidate

`MIINFER_PREFILL_REPACKED_FP16=1` adds an opt-in branch to the existing wide
recurrent prefill path. It expands, in sequence, the resident QKV/gate Q4/Q6
tiles, Q5 SSM-out tile, and Q4/Q6 FFN Gate/Up/Down tiles into the existing
`m12_dense_weights` scratch buffer and calls the existing batched rocBLAS
helper. Decode and the default resident MMQ path are unchanged.

The branch is deliberately not enabled by `MIINFER_PREFILL_REPACKED_RESIDENT_ALL`;
it requires the separate environment variable so the qualified M23 baseline
remains reproducible.

## Environment

- AMD Instinct MI50 / gfx906, Qwen3.8-27B-Q4_K_M
- release `miinfer`, ROCm 6.4.0
- `MIINFER_PREFILL_LAYER_MAJOR=1`
- `MIINFER_PREFILL_WIDE_CHUNK=1`
- `MIINFER_PREFILL_FULL_LAYER_MAJOR=1`
- `MIINFER_PREFILL_CHUNK=128`
- `MIINFER_PREFILL_WIDE_REPACKED_MMQ=1`
- `MIINFER_PREFILL_REPACKED_RESIDENT_FFN=1`
- `MIINFER_PREFILL_REPACKED_RESIDENT_ALL=1`
- `MIINFER_PREFILL_REPACKED_FP16=1`
- `MIINFER_WIDE_VALIDATE=1`

## Correctness

The 129-token validation run completed successfully with exit status 0 and no
layer mismatch or non-finite report. It exercised the direct resident-FP16
branches while the existing wide validation replayed the canonical per-token
layer path. Peak allocation was `22,863,918,036` bytes. The validation run is
not a speed result: it took `149,341 ms` because it intentionally replays the
canonical layer for every token.

## Results

No qualified P512 timing is claimed yet. A non-validation P512 CLI run was
stopped after several minutes while the process remained GPU-busy; this is a
runtime scheduling investigation item, not a correctness failure. The
projection-only bake-off remains the current timing evidence.

## Decision

**KEEP as an integration prototype.** The layer correctness gate is passed,
but the timing harness needs to be isolated from full-model CLI overhead before
the branch can be compared against the 106.4 tok/s resident-all baseline.

## Follow-up

1. Add a layer-local event profile for the direct branches and MMQ control.
2. Run a short non-validation B128/B512 A/B with continuous clock telemetry.
3. Extend the same branch to FullAttention Q/K/V/O and FFN projections only
   after the recurrent measurements are reproducible.
