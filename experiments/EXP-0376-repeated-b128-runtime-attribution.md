# EXP-0376 — Attribute the Repeated-B128 Runtime Collapse

## Question

Why does repeated B128 execution appear anomalously expensive after the
EXP-0375 dispatch correction: context scaling, sequential invocation state,
hidden route divergence, or host/runtime overhead?

## EXP-0375 basis

EXP-0375 corrected the repeated-B128 dispatch and requalified routing, base
advancement, final hidden, logits, top-10, continuation, finiteness, prefix
preservation, and later-base execution. This experiment adds only optional
chunk host/HIP-event timing and allocation counters. No GPU math kernel or
scheduler redesign was added.

## Environment

- Model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
- Preset: `m25_hi_qualified`
- GPU state: SCLK 1606 MHz, MCLK 1000 MHz, 225 W policy, physical fan full speed
- Temperature during the clean runs: approximately 37–39 C; no observed clock or thermal throttle
- Corrected current build; route/state diagnostics disabled

## Clean current-HEAD timing

The optional EXP-0376 mode records HIP events around each complete chunk,
host enqueue-call duration, and one terminal stream synchronization. It does
not synchronize at every chunk.

| Request | Prefill wall ms | Chunk GPU ms |
| --- | ---: | ---: |
| P512 | 2552.01 | 2551.90 |
| P640 | 15558.87 | 2401.56 + 13157.20 |
| P768 | 30227.73 | 3226.94 + 13391.90 + 13608.70 |
| P896 | 44484.35 | 2936.13 + 14850.50 + 13668.50 + 13029.00 |
| P1022 | 106849.68 | 2491.64 + 13333.50 + 13634.00 + 13664.70 |
| P1024 | 5348.83 | 2754.77 + 2593.88 |

P1024 remains a healthy complete-B512 control on this matched build.

## Critical per-chunk table

Values below are from P1022 for the repeated-B128 attribution and P640 for
the standalone first-B128 confirmation. Host time is the host duration of
the chunk call; GPU time is the stream event interval. Negative gaps on
complete chunks mean the host returned before the asynchronous GPU work
finished, as expected.

| Chunk | Base | Count | GPU ms | Host wall ms | Host/GPU gap | Increment vs first B128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B512 | 0 | 512 | 2491.64 | 37.38 | -2454.26 | — |
| B128 #1 | 512 | 128 | 13333.50 | 14473.20 | 1139.64 | baseline |
| B128 #2 | 640 | 128 | 13634.00 | 13859.00 | 225.03 | +300.50 GPU |
| B128 #3 | 768 | 128 | 13664.70 | 13596.80 | -67.84 | +331.20 GPU |

P640 independently measured B128 #1 at 13157.20 GPU ms and 14278.20 host
ms. The spread is run-to-run noise/clock-state variation, not a monotonic
invocation penalty.

## Base-position and invocation attribution

The required sequential matrix was measured directly:

- P640: one B128 at base 512
- P768: B128 at bases 512 and 640
- P896: B128 at bases 512, 640, and 768

The isolated restored-state matrix (`single B128 @512/@640/@768`) was not
run: the existing exact snapshot machinery restores recurrent/KV state but
does not provide an exact wide-prefill workspace checkpoint, and creating a
new checkpoint architecture would contaminate this attribution experiment.
The sequential matrix is sufficient to reject an invocation-number collapse:
B128 #2/#3 are within approximately 2.5% of B128 #1 on GPU.

For causal attention, the average prefix lengths for these B128 chunks are
approximately 576.5, 704.5, and 832.5 tokens, ratios 1.00, 1.22, and 1.44.
Observed total B128 GPU times are approximately 13.33, 13.63, and 13.66 s;
they do not exhibit a 5–10x sequential increase. The route contains substantial
non-attention recurrent work, so this aggregate is not expected to scale in
direct proportion to causal prefix length.

## Execution-mode counters

All complete chunks used `prefill_full_layer_major_chunk` with the corrected
qualified B512/B128 route. P640/P768/P896/P1022 report the expected complete
B512 and partial B128 counts. The B128 chunks report zero scalar layer-run
calls and zero scalar tokens. P1022 reports 64 scalar layer-run calls for its
126-token residual only; those calls are outside the zero-residual B128
attribution.

No internal B128 route difference was observed.

## Host/runtime and allocation audit

The host-minus-GPU gaps do not grow with B128 invocation number. P1022 B128
gaps are +1139.64, +225.03, and -67.84 ms. Allocation counters for B512 and
B128 chunks report:

```text
alloc_delta=0
total_bytes_delta=0
live_bytes_delta=0
```

No repeated model-sized allocation, upload, repack, or workspace allocation
was observed. The timing mode uses one terminal stream synchronization and no
per-chunk synchronization.

Coarse stage attribution was not started: the B128 GPU intervals do not show
a disproportionate repeated-chunk GPU outlier, so stage-level kernel
profiling would not answer the primary question.

## P1022 corrected residual attribution

```text
B512        2491.64 ms
B128 #1   13333.50 ms
B128 #2   13634.00 ms
B128 #3   13664.70 ms
subtotal  43123.84 ms
P1022    106849.68 ms
remainder-associated wall 63725.84 ms  (59.64%)
```

The 126-token residual is therefore still associated with the dominant
remaining wall time, but the repeated B128 chunks themselves are not
collapsing as invocation number increases. This does not prove all of the
remainder is caused by the residual; it identifies the residual-associated
contract as the next measurable frontier.

## Decision

**QUALIFY.** Clean current-HEAD timing shows no repeated-invocation B128
collapse. B128 cost is primarily GPU execution of the qualified route, with
small base-position variation and no repeated allocation/setup defect.

## Exact next PRIMARY

Existing B64/B4 residual contract attribution and optimization is now
authorized. The next experiment must isolate the 126-token residual from the
qualified B512/B128 subtotal above.

No new GPU math kernel or residual scheduler optimization was implemented.
