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

The layer-local harness uses the exact row-128 MMQ mapping and times one
recurrent layer without the 64-layer CLI scheduler or attention layers. Each
cell is the median of three interleaved control/candidate runs; values are
microseconds and the ratio is control/candidate.

| batch | resident MMQ | direct resident FP16 | ratio | resident allocation |
|---:|---:|---:|---:|---:|
| 128 | 21,585.7 | 21,935.8 | 0.984× | 391,781,904 B |
| 256 | 39,862.5 | 37,786.8 | 1.055× | 457,694,736 B |
| 512 | 73,521.2 | 68,755.6 | 1.069× | 589,520,400 B |

The direct path is therefore slightly slower at B128 and only about 1.07×
faster at B512 for this complete recurrent layer. A non-validation full-model
CLI run was stopped after several minutes while both the control and candidate
schedules remained GPU-busy; this is pre-existing scheduler/attention
overhead, not a projection correctness failure.

## Decision

**KEEP as a measured reference, not yet as the default P512 architecture.**
The direct path is correct and modestly faster at B256/B512, but the row-128
MMQ path is already close enough that migrating all projections is unlikely to
be a standalone 200 tok/s solution. Do not spend time on fusion until the
attention/output-projection coverage is measured.

## Follow-up

1. Extend the same harness to FullAttention Q/K/V/O and FFN projections.
2. Add continuous clock/power telemetry to the layer A/B runs.
3. Revisit full-model integration only if attention and output projection show
   a larger row-128 MMQ gap.
