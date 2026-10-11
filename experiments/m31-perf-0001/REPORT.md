# M31-PERF-0001: GQA performance comparison

Run on the Z840 MI50 on 2026-10-11 using the pinned ROCm 7.2.1 image. This compares the direct pre-change parent with the GQA candidate:

| Role | Commit | Binary SHA-256 |
| --- | --- | --- |
| Control | `016472160c295eb1927b1ed6e095719ada6c3d18` | `52dd0f09d0f1921c3556d0f08d300644dd3e7dc910bf7c3ef30ac3e34a0cc1c1` |
| Candidate | `8a9caec3366512e459a93bad68adde181225304b` | `2661d347ab12f5ab91d391ef119e69b61adb559210268938a549d2f684cfb5f9` |

The source commits are adjacent. Both builds used the same Release, gfx906, HIP, tests, and benchmark settings in `localhost/miinfer-dev:rocm-7.2.1`, digest `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`. The model SHA-256 was `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`; GPU BDF `0000:06:00.0`, unique ID `0x21678e17348c2f7`.

## Result

The candidate has a repeatable eager decode slowdown in both routes: median decode latency rises 3.82% on the default route and 3.56% on MMQ. Prefill is effectively unchanged. In graph mode, both routes are effectively neutral in this run. These are descriptive measurements, not production qualification.

Times are medians across measured samples; brackets give the observed minimum and maximum. Decode latency is `decode_ms / (generated_tokens - 1)` for five generated tokens. `control` is the default kernel route; `mmq_only` forces the MMQ Gate/Up route. The routes must be interpreted separately.

| Mode | Route | Samples per source | Source | Prefill ms median [min, max] | Decode ms/token median [min, max] | Decode tok/s median |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Eager | default | 5 | control | 8444.365 [8442.867, 8466.228] | 34.443 [34.379, 34.602] | 29.03 |
| Eager | default | 5 | candidate | 8467.500 [8444.217, 8481.241] | 35.759 [35.673, 36.998] | 27.97 |
| Eager | MMQ | 5 | control | 8465.911 [8461.997, 8480.035] | 38.912 [38.852, 38.967] | 25.70 |
| Eager | MMQ | 5 | candidate | 8467.633 [8465.764, 8485.751] | 40.299 [40.220, 41.428] | 24.82 |
| HIP graph | default | 6 | control | 8465.761 [8463.733, 8470.073] | 35.209 [35.034, 36.217] | 28.40 |
| HIP graph | default | 6 | candidate | 8465.186 [8445.044, 8471.636] | 35.196 [35.055, 35.258] | 28.41 |
| HIP graph | MMQ | 6 | control | 8470.803 [8453.022, 8479.879] | 39.420 [39.278, 40.845] | 25.37 |
| HIP graph | MMQ | 6 | candidate | 8469.304 [8464.668, 8483.730] | 39.286 [39.271, 39.461] | 25.45 |

Median candidate deltas versus control:

| Mode | Route | Prefill latency | Decode latency |
| --- | --- | ---: | ---: |
| Eager | default | +0.274% | +3.820% |
| Eager | MMQ | +0.020% | +3.564% |
| HIP graph | default | -0.0068% | -0.037% |
| HIP graph | MMQ | -0.018% | -0.341% |

## Correctness observations

Every measured sample repeated a stable token sequence within its source, mode, and route. The eager default route generated `[220, 248046, 198, 248045, 271]` for both sources. Eager MMQ generated that sequence on control and `[220, 248046, 198, 248045, 74455]` on candidate. Graph default generated the same sequence on both sources; graph MMQ generated `[220, 248046, 198, 248045, 74455]` on both.

The separate raw-logit captures contain five decisions of 248,320 FP32 values each. For eager default, all five argmax IDs match, although later logits differ numerically. For eager MMQ, the first four argmax IDs match and the fifth differs: control `271`, candidate `74455`. For both graph routes, control and candidate captures are bitwise identical across all five decisions. This characterizes the captured normal greedy trajectories; the benchmark has no teacher-forced history option, so these captures do not establish numerical equivalence on matched histories.

## Measurement and telemetry limits

The six graph samples per source followed three alternating two-iteration batches per revision: control, candidate, candidate, control, control, candidate. Eager has five samples per source, including one earlier control and candidate prelude sample from the initial runner, followed by two-iteration batches in candidate, control, control, candidate order. The eager prelude samples were retained as valid post-warmup observations, but they make the eager ordering less balanced. Treat the timing deltas as descriptive pending a more strictly interleaved repeat if a close causal estimate is required.

Each measured run call performs one unmeasured warmup. Raw-logit validation calls were separate and are not included in timing. All 16 records report `HARNESS_VALIDATED`; source identities match the intended commits. Telemetry had zero failed samples, zero missing junction samples, and no missing fields. The highest observed junction temperature was 80°C, exactly the warning threshold; no stop threshold or guard abort occurred.

Peak sampled physical HBM was approximately 26.132–26.133 GB on the default route and 18.883–18.884 GB on MMQ. The corresponding model allocation request was approximately 25.128 GB and 17.998 GB. These are different measurements; allocation request bytes are not physical HBM residency.

## Disposition

- The eager decode slowdown is the performance finding for this candidate; prefill did not materially change.
- Graph timing was neutral within the observed spread.
- Do not promote MMQ from this experiment. The MMQ route changes greedy output in eager mode, and this benchmark is not a correctness qualification.
- Correctness remains governed independently by LC-0006 and LC-0008. The exact-token production reference gate remains unresolved; this run does not relax it or qualify a production baseline.

## Evidence

Raw per-call records, metrics, environment captures, prompt IDs, manifests, and logit files are in [`results/m31-perf-0001/z840/results/`](../../results/m31-perf-0001/z840/results/). The two logit files for each mode/source validation run include SHA-256 metadata in their `benchmark.json`; all eight raw captures are retained.

Build flags and image/model/GPU details are in [`BUILD.md`](BUILD.md). Source manifests are in [`source-manifest-control.json`](source-manifest-control.json) and [`source-manifest-candidate.json`](source-manifest-candidate.json).
