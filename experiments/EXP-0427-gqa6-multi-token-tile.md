# EXP-0427 — Packed GQA6 multi-token tiles

**Status:** REJECTED — both multi-token tiles are exact but slower

## Objective and current bottleneck

Reduce restored long-context Q8 suffix latency. EXP-0424's current exact
packed-U128 GQA6 path measured 80.5% of a late 1K tile in Stage1 attention.
EXP-0425 then added exact paired-token reuse but improved a 4K suffix by only
about 1%. The 64K cold-prefill target remains at most 180 seconds; no new cold
run is justified until this schedule shows a material real-model gain.

## Hypothesis and variable

With the updated packed-U128 GQA6 baseline, processing four or eight adjacent
query tokens in one CTA may amortize CTA setup and K/V history loads. The
candidate preserves each query's QK/V operations, online-softmax order,
causal interval, and Split-K boundaries. Compare token tile 4 and 8 against
tile 2 using the same model, restored state, tokenized 4K suffix, binary,
graph replay, and per-card sequential control/candidate pairs.

EXP-0413 rejected earlier exact cross-token schedules against the slower
pre-packed baseline. This is a bounded recheck against the new packed-U128
baseline, not evidence that the old schedules were successful. Reject any
tile that is slower or fails exactness; do not expand into another tile sweep.

## Gates

- **Build/resource:** compile the exact pushed source plus this candidate in
  the pinned ROCm 7.2.1 image. Record VGPR, SGPR, spills, scratch, LDS, and
  selected gfx906 code object before timing. Reject spills or invalid launch.
- **Correctness:** compare full logits, recurrent-state trace, checked greedy
  tokens, finite outputs, and final position after restore plus 4K suffix.
  Require exact equality.
- **Dispatch:** confirm the graph captures the requested tile and emits its
  route on all 16 attention layers.
- **Performance:** first run one paired sample per card. Continue only if
  exact and at least 10% faster on that card. If admitted, collect three
  interleaved pairs per card and require the median improvement to remain at
  least 10% with non-overlapping paired results.
- **Safety:** use the 80 C warning and 85 C hard stop; stop on HIP error,
  wrong route, non-finite values, or any correctness mismatch.

## Results

The candidate built with the pinned ROCm 7.2.1 image
`localhost/miinfer-m31-rocm721-libdw@sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`.
The gfx906 code object reports 83 VGPR, 49 SGPR, zero register spills, zero
private bytes, and wave64 for both specializations. The launch widths are 384
threads for tile4 and 768 for tile8; dynamic LDS is 2,064 bytes. Candidate
binary SHA-256: `427ee6d108418380b03a9530d013087de8ee0f9f3d2aef24abc18b6b54caa76a`.

Each card ran one sequential packed-U128 control / candidate pair using the
same restored Q8/Q8 prefix and tokenized 4K suffix. Both cards ran at the same
time; timings are compared only within each card. GPU0 tested tile4 and GPU2
tested tile8. The graph route trace showed the requested selector for all 16
attention layers on each card.

| MI50 / BDF | Packed-U128 control | Candidate | Speed ratio | Logits / state / tokens |
| --- | ---: | ---: | ---: | --- |
| GPU0 / `0000:04:00.0` | 83.052 s | tile4: 133.927 s | 0.620x | exact / exact / equal |
| GPU2 / `0000:88:00.0` | 90.759 s | tile8: 97.125 s | 0.934x | exact / exact / equal |

All outputs were finite and the restored position reached 77,195. The
temperature guards sampled 104 times, reported no warning or failure, and
recorded maximum junction temperatures of 76 C on GPU0 and 65 C on GPU2. Both
containers were stopped after completion; both cards returned to idle.

The screen fails the predeclared 10% gain gate on both cards. Tile4 is 61%
slower and tile8 is 6.6% slower. **Reject both specializations and do not run
more tile sizes.** Preserve packed-U128 GQA6 as the best-known path. The
earlier EXP-0413 cross-token result remains consistent with this outcome: more
queries per cooperative CTA do not turn the K/V reuse into faster suffix
execution, even with the newer packed staging baseline. This experiment does
not qualify cold 64K prefill.

Raw logs and guard files are on Machinist:

- `/home/machinist/m31-gqa6-longctx-results/EXP-0427-multitoken-20261014T2000Z/gpu0/`
- `/home/machinist/m31-gqa6-longctx-results/EXP-0427-multitoken-20261014T2000Z/gpu2/`

The first launch attempt lacked `/dev/kfd` and `/dev/dri` in the containers
and exited before inference. The measured runs used explicit device passthrough
and `--group-add keep-groups`; the device error is not part of the benchmark
result.

## Decision

Reject the cross-token tile4/tile8 CTA schedule. Keep paired-token tile2's
small exact gain and packed-U128 GQA6 as opt-in candidates; do not change the
production default. The next optimization must change Stage1's dataflow more
substantially than increasing cooperative query tile size.
