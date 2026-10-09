# M31-0002T-0001 — Weight residency and VRAM ledger

Status: major weight-layout contributor identified; full tensor-type ledger and
some cross-runtime accounting remain unresolved. Values below distinguish
source-derived allocation sizes from observed device usage. No GPU workload or
code change was made for this investigation.

## MIInfer persistent weight allocation ledger

The harness sums bytes passed to model-weight device allocations; the model
keeps these allocations live for its lifetime. Decimal GB is 10^9 bytes; GiB
is 2^30 bytes.

| Category | Allocated bytes | GB | GiB | Notes |
|---|---:|---:|---:|---|
| 64 active layer weights excluding fused FFN | 15,110,514,688 | 15.111 | 14.073 | Remainder of layer/model allocations; exact GDN/GQA source split not isolated |
| Token embedding, Q4_K | 715,161,600 | 0.715 | 0.666 | MIInfer places it on GPU |
| Final norm, F32 | 20,480 | <0.001 | <0.001 | Direct upload |
| Output head, Q6_K Wave layout | 1,112,473,600 | 1.112 | 1.036 | Converted from 1,042,944,000 source bytes |
| Fused FFN Gate/Up Wave layout | 7,130,316,800 | 7.130 | 6.641 | 64 layers × 87,040 tiles/layer × 1,280 bytes |
| **Total MIInfer weight allocations** | **24,068,487,168** | **24.068** | **22.416** | Matches measured model ledger |

The largest identified contributor is a second resident layout for Gate/Up:
the layer constructors upload separate repacked MMQ Gate and Up tensors and
also upload a fused Wave layout. The fused allocation is derived from a
6,417,285,120-byte source Q4_K payload, plus 713,031,680 bytes of representation
overhead: 534,773,760 bytes of explicit tile padding and 178,257,920 bytes of
additional per-tile metadata space. The two layouts are consumed by different
prefill/decode paths; being duplicate representations does not prove either is
safe to remove.

The Q6_K output head adds 69,529,600 bytes over its source payload due to Wave
tile layout. Other MMQ repackers change row pitch/layout, but their total
padding contribution is not isolated by the saved per-tensor artifacts. No
additional persistent upload buffer is included in the weight ledger; packed
conversion vectors are host temporaries. Embedding and output head are
different source types and are not tied/shared in the inspected loader.

Source: `src/prefill_v2/recurrent_layer.cpp`,
`src/prefill_v2/attention_layer.cpp`, `src/kquant_wave_layout.cpp`,
`include/miinfer/kquant_wave_layout.hpp`, and `src/prefill_v2/model.cpp`.

## Model and observed VRAM reconciliation

The GGUF file is 17,106,775,008 bytes (15.932 GiB); file size is not an exact
active-weight sum because it includes metadata and tensors unused by the
64-layer runtime. The MIInfer run reports:

| MIInfer category | Bytes |
|---|---:|
| Weights | 24,068,487,168 |
| Persistent state | 242,745,344 |
| Workspace | 727,711,744 |
| Activations | 21,966,848 |
| Model-owned total | 25,060,911,104 |
| External sampled peak used VRAM | 26,000,453,632 |
| Post-call used VRAM | 26,021,724,160 |
| Post-call residual beyond model ledger | 960,813,056 |

The reported device high-water field is null. The external peak is therefore
sampling-based, not an allocator peak. Model-ledger and telemetry samples are
not necessarily simultaneous.

For the same model, the saved llama.cpp probe accounts 16,091,095,040 bytes in
ROCm model buffers and 715,161,600 bytes in CPU-mapped model buffers. The
embedding-sized CPU-mapped amount corresponds to the embedding that MIInfer
places on GPU. Thus:

- MIInfer weights exceed llama.cpp's GPU-only model buffers by
  **7,977,392,128 bytes**. This is composed of MIInfer's GPU embedding
  (715,161,600), fused FFN layout (7,130,316,800), and a 131,913,728-byte net
  residual in other model weights versus llama.cpp's GPU+CPU model-buffer
  accounting.
- MIInfer weight allocations exceed llama.cpp's GPU+CPU model-buffer total by
  **7,262,230,528 bytes**. This compares allocation ledgers, not exclusively
  GPU residency.
- The observed short-run total-VRAM peak difference is
  **8,759,263,232 bytes** (26,000,453,632 minus 17,241,190,400). After the
  7,977,392,128-byte GPU-weight delta, **781,871,104 bytes remain
  unclassified** across runtime/context/allocator sampling differences.

The llama.cpp ledger above comes from a separate 128K model/context probe, not
the same 1K call. The 1K llama run has no equivalent categorized allocation
ledger. The reference used the same model SHA, but this context/measurement
scope prevents treating the remaining differences as exact apples-to-apples
categories.

## Classification and recovery candidates

- `DUPLICATE_RESIDENCY`: 7,130,316,800 bytes of fused FFN layout co-resides
  with MMQ Gate/Up representations. This is an architectural dual-layout
  choice, not proven accidental duplication.
- `PADDING_OR_ALIGNMENT`: at least 713,031,680 bytes in the fused Gate/Up
  representation and 69,529,600 bytes in the output head layout, derived from
  format sizes. These are not certified safely reclaimable; the consuming
  kernels depend on the layouts.
- `TEMPORARY_LIFETIME`: host conversion vectors are temporary and not counted
  as GPU weight residency; no GPU upload staging duplication was found.
- `UNEXPLAINED`: 131,913,728 bytes in the net model-buffer residual, 781,871,104
  bytes in the total short-run VRAM delta after the comparable GPU-weight
  estimate, plus the 960,813,056-byte post-call MIInfer ledger residual.

**Bytes proven safe to recover now: 0.** Rank the next memory work as: (1)
determine whether MMQ and fused Gate/Up layouts must be live simultaneously;
(2) account tensor-level GDN/GQA and remaining MMQ row-pitch bytes; (3) quantify
the output-head layout tradeoff. Any future phase-specific release/reload or
single-layout path requires exact correctness and performance validation.
