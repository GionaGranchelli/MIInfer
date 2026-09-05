# EXP-0173 — Q4K native-layout laboratory

## Scope and acceptance

Baseline commit: 84e411e4a63a. Exact Down shape: 5120 output rows,
17408 input columns, 68 Q4_K blocks/row. The production-selected comparator
is B59 expanded Q4K, not B41 canonical Q4K. Require >=15% isolated throughput
improvement against B59 before executor integration. No candidate is integrated.

## Baseline harness

`miinfer-q4k-layout-bench MODEL --json-output FILE` loads the first Q4_K
Down tensor of that shape. On this artifact that is `blk.8.ffn_down.weight`;
layer 0 is not Q4_K. It quantizes a deterministic sinusoidal input once to
Q8_1 and passes the identical device buffer to both primitives. This input
is synthetic, not a recorded model activation. It checks finite outputs and
max error <=1e-4 before timing. It uses 50 warmups per path and 101 alternating
AB/BA batches, 10 launches per batch, with raw samples in JSON.

The repeated tensors exceed MI50 L2. This does not yet establish equivalence
to the cache state of a full model; a rotating-tensor check remains required
for a candidate close to the acceptance boundary.

## Preliminary run — NOT valid for stable-peak acceptance

Command:

```
scripts/run-bench.sh ./build/mi50-release/miinfer-q4k-layout-bench /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
```

Raw artifacts: `bench/results/20260905T172631Z-356588/`.

| Metric | B41 canonical | B59 production expanded |
| --- | ---: | ---: |
| Median microseconds | 203.456 | 179.744 |
| Allocated weight bytes | 50,135,040 | 96,092,160 |
| Nominal weight GB/s (allocated bytes/time) | 246.417 | 534.606 |
| Output max absolute difference | 0 | 0 |
| Workgroups | 2560 | 2560 |
| Waves/workgroup | 4 | 4 |
| Payload bytes/row | 8704 | 17408 |
| Metadata bytes/row | 1088 | 1360 |
| Static + dynamic LDS/workgroup | 1024 + 22304 | 1024 |

Bytes above describe unique tensor storage, not measured transactions.
Canonical B41 stages 19,584 Q8 bytes/workgroup, or 9792 amortized bytes/row.
B59 directly issues Q8 loads: 68 blocks * 16 lane contributions * 2 Q8 groups
* (8 quant bytes + 2 scale bytes) = 21,760 logical bytes/row before cache reuse.
Both paths perform 68*16*2*4 = 8704 dot4 operations/row, including input sums.
Both sum 128 thread partials per row: 127 additions, using LDS and barriers.
B41 decodes 544 scale/min pairs per row; B59 reads predecoded metadata but
reconstructs four DP4A words per lane contribution from interleaved bytes.
Compiler-level instruction counts, registers, and occupancy remain unmeasured.

Telemetry: profile_peak was selected, but SCLK dropped from 1725 to 1606 MHz
under load; MCLK stayed 1000 MHz. Power cap 225 W; sampled power reached 207 W;
edge temperature 52–55 C, junction 66–79 C. Consequently these timings are
diagnostic only and must not qualify any layout. HIP 7.1.52802-9999.

The earlier sandbox run failed GPU access and the next run failed the incorrect
layer-0 Q4_K assumption. Their result directories are retained; neither produced
performance evidence.

## Pinned reference representation

The external checkout HEAD is 6e4ef6c, not the specified c0bc859. Inspect source
with `git show c0bc859:ggml/src/ggml-cuda/vecdotq.cuh`. The pinned Q4K dot loads
two packed words 16 bytes apart for each of 16 lane contributions/block.
Each contribution covers two Q8 groups; it fetches four Q8 words plus two half
scales, decodes two scales and two minima, and masks/shifts nibbles into dot4
operands. This Q4K function has the same representation in the checked-out
revision, but other MMVQ code differs; checkout-wide conclusions cannot be
attributed to the pinned commit without checking the relevant source.

## First physical layout proposal (not implemented or tested yet)

Group four adjacent K blocks into a Wave64 tile. Transpose their two packed
word streams into two contiguous 64-word planes, so each wave issues contiguous
256-byte plane reads. Store four 20-byte decoded metadata records after the
planes. Align/pad the tile to 640 bytes (512 payload + 80 metadata + 48 padding).
Keep nibbles packed and preserve source half scales and integer metadata.

Storage: 17 tiles/row * 640 * 5120 = 55,705,600 bytes, 1.1111x canonical
and 0.5797x B59 expanded. Replacing each expanded Down allocation saves
40,386,560 bytes. At integration, count actual Q4K Down tensors before computing
the model-wide saving; do not assume all 64 Downs have the same quantization.

Expected benefit: contiguous lane reads and removal of B59 byte reconstruction,
with much smaller payload than B59. This changes the physical layout rather
than merely attaching a metadata sidecar. Nibble extraction remains necessary.
The storage upper bound cannot predict a speedup without a kernel measurement.

## Status

Baseline harness builds and its initial arithmetic comparison passes. No native
layout has been tested (0/3). Hardware-valid baseline, resource diagnostics,
native conversion validation, and candidate implementation remain outstanding.

## Layout 1 — implementation and initial primitive evidence

Implemented `bench/q4k_wave_layout.hpp` and `.hip`, linked only into the
laboratory benchmark. The host packs two word planes per tile and verifies
every source byte via an independent inverse index mapping. It reconstructs
the original 12 packed scale/min bytes from native metadata and compares
the half-scale bit patterns. There is no requantization.

The native kernel retains four waves/workgroup, two waves per row, 2560
workgroups, 1024 bytes static LDS, the same per-thread K partition and the
same LDS reduction order. Each Wave64 loads two contiguous word planes.
It performs four nibble masks/two high-nibble shifts per contribution instead
of expanded byte reconstruction; metadata is directly usable. Dot4 count and
floating-point arithmetic order are unchanged. Register and ISA analysis are
still outstanding; source operation counts are not ISA counts.

Two process runs, each with 101 timing batches per variant:

| Run under bench/results | B59 median us | Native median us | B59/native |
| --- | ---: | ---: | ---: |
| 20260905T173056Z-359637 | 179.216 | 93.8239 | 1.910 |
| 20260905T173216Z-360783 | 179.776 | 93.7759 | 1.917 |

All three sampled telemetry points in each run reported SCLK 1725 MHz and
MCLK 1000 MHz. Sampling is sparse; it does not rule out brief unsampled dips.
These runs provide substantially stronger evidence than the contaminated
baseline-only run, but longer confirmation remains useful before integration.

The second run tests zero input, alternating +/-12, deterministic modular
values, and the sinusoidal timing input. All native/production output max
errors were zero. Each path uses the same quantized input. These checks are
primitive correctness only, not generation or the A27 external contract.

There are 32 exact-shape Q4_K Down tensors. Replacing their expanded copies
would save 1,292,369,920 bytes. Using the previous production allocation
20,094,914,900 bytes as an estimate gives 18,802,544,980 bytes after replacement,
leaving about 15.54 GB of the reported 34,342,961,152-byte device before other
allocations/context changes. Actual integration allocation accounting is required.

The isolated saving projects to 32*(179.776-93.7759) us = 2.752 ms/token.
Against 71.913 ms/token this predicts roughly 3.98% throughput improvement,
not a proven >=5% end-to-end result. Dominant Down stage timings from the
earlier report cannot all be attributed to Q4_K: the model has mixed Down
types. Integration must measure this explicitly before expanding the scope.

Decision: retain layout 1 for confirmation and limited Down integration.
The isolated gate is provisionally exceeded by a large margin; full success
remains unproven. Layout count: 1/3. No executor integration yet.
