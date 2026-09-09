# EXP-0275 — M21 prefill gap and M12 continuation correctness

## Question

Is the large prompt-processing gap caused by request overhead, or by the
runtime execution architecture, and can the existing layer-major path safely
serve as the prefill candidate?

## Configurations

MIInfer used the exact Qwen3.8-27B-Q4_K_M GGUF on the physical MI50, with the
default token-major path and the opt-in `MIINFER_PREFILL_LAYER_MAJOR=1` path.
The specialized reference was `mx-llama.cpp` commit `2e9d29fe`, full GPU
offload, Flash Attention, Q8_0 K/F16 V, `-b 2048 -ub 2048`, three repetitions.
The mandated `iacopPBK/llama.cpp-gfx906` pin remains separately rejected for
not supporting this model (EXP-0269).

## Correctness

After the layer-major first-token EOS fix, exact greedy token IDs matched the
default path at P8, P128, and P512 for an 8-token generation check. The P129
case also matched exactly: both paths returned `[248046]` and stopped. The
diagnostic captures are in `results/m21-prefill-profile/20260909-p513/`; the
token-ID capture uses `MIINFER_DUMP_TOKENS=1`.

## Results

| prompt | MIInfer default PP | MIInfer layer-major PP | mx-llama PP | default/ref | layer-major/ref |
|---:|---:|---:|---:|---:|---:|
| P8 | 31.80 | 32.32 | 55.89 | 0.569x | 0.578x |
| P128 | 33.12 | 47.05 | 180.45 | 0.184x | 0.261x |
| P512 | 32.49 | 45.85 | 222.64 | 0.146x | 0.206x |
| P8K | — | 38.33 | 229.20 | — | 0.167x |
| P16K | — | 36.18 | 219.91 | — | 0.165x |
| P32K | — | 34.08 | 202.84 | — | 0.168x |
| P64K | — | 28.37 | 175.75 | — | 0.161x |
| P128K | — | 20.55 | 138.39 | — | 0.149x |

The MIInfer short results are corrected PP-only or TG64 runs in
`results/m18-runtime-corrected/`. The reference raw ladder through P128K is in
`results/m21-reference/20260909-mx-2e9d29fe/`.

At P512, layer-major is about 4.86x slower than the specialized reference;
the default path is about 6.85x slower. At P128K, layer-major is about 6.73x
slower. At P8 the rates are close, so the gap appears when prompt parallelism
becomes useful, not in startup/dispatch alone.

The layer-major P513 profile accounts for 11,177.8 ms of an 11,189.0 ms
prefill wall time: 8.6 ms embedding, approximately 11,169.2 ms in layer
regions, and 11.1 ms unaccounted. Per-layer totals are approximately 170–181
ms, with the deferred tail dominating the existing three-stage profile.

The existing GDN chunkwise variant measured 46.83 PP tok/s at P513 and
selected the same EOS token, but it is not promoted independently of the
broader M12 qualification.

### Context-conditioned TG64 reference

The cached-depth reference run measures only the 64-token generation after
the depth prefill; the depth work is outside the timed interval.

| context depth | mx-llama TG64 tok/s | samples |
|---:|---:|---:|
| P128 | 25.54 | 25.484, 25.678, 25.461 |
| P512 | 25.48 | 25.370, 25.517, 25.551 |
| P2K | 25.13 | 24.771, 25.311, 25.316 |
| P8K | 23.82 | 23.689, 23.877, 23.897 |
| P16K | 22.40 | 22.295, 22.469, 22.450 |
| P32K | 19.72 | 19.403, 19.871, 19.876 |
| P64K | 14.78 | 14.063, 14.510, 15.781 |

Raw results, commands, compiler details, model hash, and hardware snapshots
are in `results/m21-reference/20260909-mx-tg64-p8k/` and
`results/m21-reference/20260909-mx-tg64-ladder/`. MIInfer has valid M12 TG64
measurements at P8K (24.65 tok/s) and P16K (24.65 tok/s); its P32K/P64K
one-token qualification runs stopped on EOS and therefore do not establish
steady-state TG64. A non-EOS long-context prompt is still required before
claiming those MIInfer decode points.

## Decision

**KEEP the EOS fix and exact-ID correctness evidence. KEEP M12 opt-in.** The
reference comparison localizes the dominant weakness to MIInfer prefill
architecture/dataflow rather than HTTP or tokenization. Do not add workers or
continuous batching; the next implementation experiment should target the
profiled layer-major deferred-tail dataflow and re-run end-to-end correctness.

The reference decode curve is healthy but declines from 25.54 tok/s at P128
to 14.78 tok/s at P64K. M12 is approximately equal at P8K, so the remaining
priority is prefill rather than decode. M19-C remains partially qualified:
the MIInfer P8K/P16K TG64 points pass, while P32K/P64K require a prompt that
does not terminate after the first generated token.
