# EXP-0276 — M22 P512 operator-family attribution

## Hypothesis

The M12 prompt gap is dominated by a specific deferred-tail or projection
family, rather than HTTP, tokenization, or request scheduling.

## Baseline

MIInfer commit `474db556c84975a5ba1f97f9feec592cbd8304d0`, release build,
`MIINFER_PREFILL_LAYER_MAJOR=1`, exact Qwen3.8-27B-Q4_K_M GGUF.

## Environment

* Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
* GPU: AMD Instinct MI50/gfx906, SCLK 1606 MHz, MCLK 1000 MHz
* ROCm: 7.1.52802-9999; Clang 20.0.0.rocm
* Profile artifact: `results/m22-prefill-profile/20260909-201924-1492471/`

## Benchmark

```bash
MIINFER_PREFILL_LAYER_MAJOR=1 \
MIINFER_PREFILL_PROFILE=1 \
MIINFER_PREFILL_PROFILE_POSITION=511 \
python3 scripts/bench-m18-runtime.py build/mi50-release/miinfer \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --output results/m22-prefill-profile --prompts 511 \
  --modes experimental --max-tokens 1
```

The 511-word prompt tokenized to P512. Profiling selects position 511 in the
last B64 chunk and records all 64 layers. The ordered interval covers the
whole layer loop for that chunk, including preparation and deferred tail;
component events are reported separately and are not summed with it.

## Results

| measurement | milliseconds | interpretation |
|---|---:|---|
| prefill wall | 11,148.96 | P512 qualification timing |
| embedding | 7.85 | negligible |
| deferred tail, sampled chunk | 971.28 | 78.9% of component sample |
| batch preparation, sampled chunk | 245.68 | 20.0% of component sample |
| selected-token stages | 6.35 | 0.5% of component sample |
| ordered layer path, sampled chunk | 1,452.61 | gross whole-layer-loop timing |

The ordered path averages about 21.5 ms for recurrent layers and 25.3 ms for
full-attention layers. Eight B64 chunks account for the prompt. The direct
component sample is not a full wall-time decomposition, but it identifies
the deferred tail and preparation path as the first implementation target;
the ordered interval confirms that the per-chunk layer loop, not tokenization,
is responsible for the wall time.

## Correctness

The run completed with return code 0 and the existing P512 deterministic
generation checks remain unchanged. No kernel behavior was modified by this
experiment; only host-side HIP event attribution was added.

## Decision

KEEP the profiler. The evidence supports a GPU prefill/dataflow bottleneck.
Do not pursue HTTP or tokenizer optimization for the M22 parity gate.

## Follow-up

Compare the native repacked mx-llama reference and its no-repack control, then
test a quantized matrix/batch alternative against the existing B4 path.
