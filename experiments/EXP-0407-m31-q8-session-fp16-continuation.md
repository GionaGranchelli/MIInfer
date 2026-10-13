# EXP-0407 — Q8 session restore into FP16 continuation

**Status: LEARN — faster suffix, correctness and latency gates fail**

## Objective

Test whether compact Q8/Q8 checkpoints can be restored into an FP16/FP16
runtime cache to accelerate persistent long-context continuation. The serving
goal remains a correct 4,096-token suffix in at most 10 seconds; a logit cosine
below 0.9999 is a correctness failure for this experiment.

## Hypothesis and method

The saved session uses about 2.4 GiB for a 73,099-token prefix. Convert each
Q8 KV layer to FP16 on GPU during restore, then compare continuation against
Q8/Q8 from the same checkpoint and deterministic suffix. The model, checkpoint,
suffix token IDs, context position and GPU remain fixed. The benchmark now
submits suffixes as sequential physical tiles no larger than
`kPrefillV2MacroTile` (1,024), advancing the recurrent and KV state after each
tile. `PrefillV2Model::forward()` intentionally accepts only one physical tile.

## Environment

- Machinist GPU2, BDF `0000:88:00.0`, UUID `0x2152414173497dd3`, gfx906.
- ROCm 7.14 image `docker.io/mixa3607/rocm-gfx906:7.14-complete`.
- Qwen3.8-27B Q4_K_M SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Existing Q8 session `sess_43c5693dd01fbab1_p73099`, 73,099 prefix tokens.
- Independent 80 C warning / 85 C stop supervisor; 26 telemetry samples,
  peak junction temperature 64 C, no warning or stop.

## Results

One matched run per cache mode, measured sequentially on the same restored
prefix and 4,096 deterministic synthetic suffix tokens:

| Mode | Restore | Suffix prefill | Decode | Total resume |
| --- | ---: | ---: | ---: | ---: |
| Q8/Q8 | 2.437 s | 88.551 s | 0.076 s | 91.064 s |
| Q8 session converted to FP16/FP16 | 2.458 s | 52.643 s | 0.072 s | 55.174 s |

FP16 reduced suffix time by 1.682x, but missed the 10-second target by 5.26x.
Final logits differed by `max_abs=0.417583942`, cosine `0.999666391`, below
the `0.9999` gate. The first and second greedy token IDs matched. The benchmark
exited 3 because the logit gate failed; the inference and decode completed.

At 1,024 suffix tokens, an earlier matched run measured 21.771 s Q8 versus
12.916 s FP16 (1.686x), with cosine `0.999893492`; it also missed the logit
gate. The FP32-accumulation FP16 attention variant passed cosine at 3 and 32
splits (`0.999958745` and `0.999919079`) but took 24.812 s and 24.519 s,
respectively, versus about 21.72 s for Q8. It is not a faster accurate route.

## Decision

Do not promote Q8-to-FP16 session conversion as a production continuation
path. The fast variant improves latency but does not meet the correctness
gate, while the tested accurate variant is slower. The improvement is also
too small for the 4K usability target. Keep the source available as an
experimental comparison and continue with a different attention/dataflow
approach. Do not tune split counts further from these results.

The 4K suffix is synthetic; this experiment does not establish real agent
behavior, checkpoint/restart workflow correctness, or end-to-end latency.
