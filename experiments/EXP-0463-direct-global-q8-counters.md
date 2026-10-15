# EXP-0463 — Direct-global Q8 GQA6 counter profile, GPU2

## Question

Does the exact direct-global Q8 Stage1 route show a strong memory-wait signal
that would justify reopening load/compute pipelining, or should the next
rewrite target its arithmetic and recurrence schedule?

## Method and provenance

- GPU2 only, BDF `0000:88:00.0`; GPU0 was not used.
- Restored Q8/Q8 session at 73,099 tokens plus the same tokenized 1,024-token
  suffix; HIP Graph replay reached position 74,123.
- The paired harness ran packed-U128 control then direct-global candidate.
  Route traces show 16 Stage1 dispatches for each path. Full logits and
  recurrent state were exact; generated tokens matched; outputs were finite.
- Source checkout: `perf/m31-gqa6-attention-rethink-gpu2` at
  `8f0bce840a546ef0c3720480bd4adf2b24395f43`; binary SHA-256
  `bd08d7abb8ea4b1e3daafc680bcafc8302aa3aaa91efbc750b8741ce8f37e9c2`.
- Counter tool ran from ROCm 7.2.1 profiler image
  `localhost/miinfer-m31-rocm721-libdw:20261014T094012Z`, digest
  `sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`.
  This differs from the inference image used for the prior uninstrumented
  timing. Treat this as a kernel-counter diagnostic, not a qualified timing
  comparison.
- Raw CSV and agent metadata are on Machinist at
  `/home/machinist/m31-gqa6-longctx-results/EXP-0463-direct-global-profile/run2/rocprof/185610a8b92c/`.

The first capture attempt was stopped during the control half, before it had
collected candidate data. It is excluded. The second capture completed; the
80 C warning / 85 C stop guard remained satisfied (57 C peak).

## Results

Counter medians below are across the 16 Stage1 dispatches in this single
instrumented pair. Per-wave values divide each dispatch's counter by its
`SQ_WAVES` count. Raw counter units are retained; they are not wall-time stall
fractions.

| Counter | Packed-U128 control | Direct-global Q8 |
| --- | ---: | ---: |
| Stage1 dispatches | 16 | 16 |
| `SQ_WAVES` | 393,876 | 393,216 |
| `SQ_INSTS_VALU` / wave | 166,152 | 172,909 |
| `SQ_ACTIVE_INST_VALU` / wave | 173,666 | 179,860 |
| `TCC_HIT` | 2.440B | 7.502B |
| `TCC_MISS` | 28.340M | 27.890M |
| `TCC_HIT / (HIT + MISS)` | 98.85% | 99.63% |
| `TCP_TCP_TA_DATA_STALL_CYCLES` | 24.350M | 30.765M |

The app reported 22.412 s for packed-U128 and 20.982 s for direct-global in
this one profiled pair. Profiler timestamps sum to 17.885 s and 16.473 s across
the respective 16 Stage1 dispatches. Neither number is a stable performance
claim: the run is instrumented, one pair, and used the profiler image above.

## Interpretation and decision

The direct-global path issues more L2 hits, while L2 misses are nearly equal.
The raw TCP data-stall counter is higher on direct-global, but that counter is
not normalized to a wall-time stall fraction. These measurements do not show
that a next-tile load pipeline can produce a material win; the short ISA
load-to-use dependency from EXP-0462 remains unproven as a runtime limiter.

Both paths execute about 166K–173K VALU instructions per wave in this profile.
This is consistent with substantial per-key arithmetic work, but the counters
do not isolate the share spent in Q8 unpacking, QK reduction, softmax, or PV
updates. The next rewrite should target that inner attention schedule and
carry a per-operation attribution, rather than assume HBM latency is dominant.

**Disposition: LEARN.** No kernel source or runtime default changed. This
profile narrows the next hypothesis but does not complete the Stage1 rewrite,
prove a cold 64K gain, or demonstrate a real coding-agent task.
