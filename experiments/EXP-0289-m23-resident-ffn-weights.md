# EXP-0289 — Resident recurrent FFN MMQ weights

## Hypothesis

The P512 wide path is dominated by layer-scoped MMQ weight uploads. Keeping
the recurrent FFN Gate/Up/Down MMQ tiles resident should remove the largest
repeated transfer while retaining the shared scratch pool for QKV, SSM, and
attention.

## Candidate

Enable `MIINFER_PREFILL_REPACKED_RESIDENT_FFN=1` together with the accepted
M23 wide/repacked settings and row-128 reader. The candidate is opt-in; decode
still uses its existing native wave layout.

## Environment

- AMD Instinct MI50 / gfx906, ROCm 6.4.0
- Qwen3.8-27B-Q4_K_M
- exact P512 prompt used by EXP-0286
- context capacity 1024, full layer-major, projection width B512
- row-128 repacked MMQ, wide QKV/SSM/FFN/attention enabled

## Correctness

The model-sized validation run completed without NaN or invalid output:

```text
output_max_abs=0.000720978
qkv_max_abs=3.8147e-05
gate_max_abs=6.67572e-06
beta_max_abs=0
decay_max_abs=0
state_max_abs=2.28882e-05
history_max_abs=2.28882e-05
```

The focused Q4/Q6 varied-payload MMQ test also passes.

## Results

| case | prefill | allocation | P512 repacked uploads |
| --- | ---: | ---: | ---: |
| row-128 control | 68.52 tok/s (profile run) | 19,394,924,884 B | 20,025,180,160 B |
| resident FFN run 1 | 72.80 tok/s | 29,956,706,644 B | not profiled |
| resident FFN run 2 | 85.54 tok/s | 29,956,706,644 B | not profiled |
| resident FFN profile | 79.77 tok/s | 29,956,706,644 B | 9,463,398,400 B |
| interleaved control | 65.08 tok/s | 19,394,924,884 B | 20,025,180,160 B |
| interleaved resident | 81.44 tok/s | 29,956,706,644 B | 9,463,398,400 B |

The resident candidate removes 10.56 GB of P512 weight transfer and fits under
the 32-GB device allocation observed here. The interleaved pair improves
throughput by 25.1%, but leaves only about 2 GB below the device limit before
larger context or diagnostics.

At context capacity 8192, a one-token allocation smoke completed with
`30,426,497,364` bytes allocated. This remains within 32 GB, but the resident
candidate is not a safe default for the project's larger-context goals.

The resident profile's remaining hot-path upload buckets are:

```text
recurrent qkv       2,123,366,400 B
recurrent gate      1,226,833,920 B
recurrent ssm_out   1,226,833,920 B
attention qk          886,046,720 B
attention v            70,778,880 B
attention o           408,944,640 B
attention ffn_gate  1,158,676,480 B
attention ffn_up    1,158,676,480 B
attention ffn_down  1,203,240,960 B
```

## Decision

**KEEP as an opt-in memory/performance candidate.** Do not enable by default
and do not claim the >=100 tok/s gate. The candidate still needs a longer
interleaved series before any broader context recommendation.

## Follow-up

Run interleaved control/resident pairs with hardware-state capture, then test
resident attention projections only if the memory budget remains acceptable.
