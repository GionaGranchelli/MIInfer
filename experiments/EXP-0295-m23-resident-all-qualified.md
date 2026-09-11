# EXP-0295 — M23 resident-all projection path

## Hypothesis

The repeated P512 repacked-weight transfers are the main architectural cost.
Keeping all recurrent and full-attention MMQ projection tiles resident, and
using those same tiles for decode, should remove the transfer loop without
requiring a native-weight restore before generation.

## Candidate

`MIINFER_PREFILL_REPACKED_RESIDENT_ALL=1` enables resident QKV/Z/SSM-out and
FFN Gate/Up/Down tiles on recurrent layers, plus QK/V/O and FFN Gate/Up/Down
tiles on attention layers. Native matrix buffers are omitted in this opt-in
mode. Decode uses the resident MMQ tiles with a reusable single-token MMQ
activation buffer.

## Environment

- AMD gfx906, Qwen3.8-27B-Q4_K_M, context capacity 1024
- full layer-major, `MIINFER_PREFILL_CHUNK=512`
- wide/repacked MMQ, row-128 reader, exact 512-token repeated fox prompt
- ROCm runtime state was sampled separately; the control series includes one
  low-clock contaminated run and is retained rather than discarded silently

## Correctness

- Focused release CTest passes, including generic and repacked Q4/Q6 MMQ at
  B128/B129/B256/B512.
- One-token continuation emits `brown`.
- Eight-token continuation emits `brown fox jumps over the lazy dog` without
  NaN/Inf or output collapse.

## Results

Resident allocations were stable at `22,801,772,884` bytes. Interleaved
P512 measurements:

| order | path | prefill | allocation |
|---|---|---:|---:|
| A1 | resident-all | 105.79 tok/s (4,839.68 ms) | 22,801,772,884 B |
| B1 | nonresident control | 72.60 tok/s (7,051.89 ms) | 19,394,924,884 B |
| A2 | resident-all | 106.24 tok/s (4,819.31 ms) | 22,801,772,884 B |
| B2 | nonresident control | 46.14 tok/s (11,096.87 ms) | 19,394,924,884 B |
| A3 | resident-all | 106.54 tok/s (4,805.77 ms) | 22,801,772,884 B |

An allocation smoke at context capacity 8192 completed at `23,271,563,604` B
and measured 105.70 tok/s, leaving substantial headroom below the 32-GB
device budget.

B2 is a documented low-clock/thermal-contaminated control, not silently
removed. The resident series is tightly grouped at 105.79–106.54 tok/s.

The profiled resident run measured 106.39 tok/s (4,812.45 ms), zero hot-path
repacked-weight uploads, and these dominant sampled buckets:

```text
FFN gate/up             31.32%
FFN down                17.45%
deferred attention tail 13.39%
KV/head norm            11.59%
projection/norm          9.84%
```

## Decision

**KEEP as the qualified opt-in M23 P512 path.** It clears the >=100 tok/s
performance gate with acceptable 22.8 GB allocation and preserves continuation
correctness. It is not the default path: resident packed weights trade roughly
3.4 GB for throughput and should remain opt-in until broader context and
hardware-state qualification are complete.

## Follow-up

Run the same resident path at larger context capacities and with captured
clock/power state. Do not resume local MMQ micro-tuning until those resource
and reproducibility checks are complete.
