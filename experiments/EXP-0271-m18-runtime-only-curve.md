# EXP-0271 — MIInfer runtime-only prefill/decode qualification

## Benchmark

The harness bypasses HTTP, JSON, ChatML, and Hermes and invokes `miinfer run`
directly. PP-only runs use one generated token and report prefill plus
first-token latency. TG runs use a multi-token generation and report steady
decode after removing the special first token from the decode rate.

```bash
python3 scripts/bench-m18-runtime.py \
  build/mi50-release/miinfer \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompts 8,128,512

python3 scripts/bench-m18-runtime.py \
  build/mi50-release/miinfer \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompts 8,128,512 --max-tokens 64
```

## Environment

MI50/gfx906, ROCm 7.1.52802-9999, Clang 20.0.0, model SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

## Results

The historical one-token run is valid for PP and first-token timing only:

| path | prompt tokens | prefill ms | PP tok/s | first decode ms | TTFT ms |
|---|---:|---:|---:|---:|---:|
| default | 9 | 247.24 | 32.36 | 35.61 | 282.85 |
| default | 129 | 3860.88 | 33.15 | 35.50 | 3896.38 |
| default | 513 | 15781.30 | 32.44 | 36.68 | 15817.98 |
| M12 experimental | 9 | 279.34 | 32.22 | 1.63 | 280.97 |
| M12 experimental | 129 | 2735.77 | 47.15 | 1.63 | 2737.40 |
| M12 experimental | 513 | 11167.91 | 45.94 | 1.57 | 11169.48 |

The old M12 `613–635 tok/s` values are not TG throughput; they measured only
the final-hidden-state LM-head path. A real 64-token generation must use
`--max-tokens 64` and report `steady_decode_tok_s` from the corrected harness.

The separate long-context TG64 HTTP measurements are recorded in EXP-0270:
MIInfer measured 25.00 tok/s after P8K and 24.65 tok/s after P16K.

Corrected short reruns are in `results/m18-runtime-corrected/20260909-144617-516076/`
(PP-only) and `results/m18-runtime-corrected/20260909-145121-682535/` (TG64):

| path | prompt tokens | PP tok/s | first decode ms | steady TG tok/s | generated |
|---|---:|---:|---:|---:|---:|
| default | 9 | 31.82 | 30.54 | 27.74 | 43 |
| default | 129 | 33.15 | 31.22 | unavailable (EOS) | 1 |
| M12 experimental | 9 | 32.47 | 1.60 | 32.39 | 43 |
| M12 experimental | 129 | 47.01 | 1.58 | 31.79 | 64 |

The TG64 rate is `(generated_tokens - 1) / (decode_ms - first_decode_ms)`;
the one-token PP run intentionally reports no TG rate.

The first TG64 attempt (`results/m18-runtime-corrected/20260909-144843-599298/`)
stopped in the harness while decoding non-UTF-8 generated output; it produced
no summary and is not used for results. The harness now decodes text with
replacement while retaining the captured raw streams.

The compatible supplemental llama.cpp reference measured PP8 33.48 tok/s,
PP128 151.27 tok/s, PP512 191.33 tok/s, and TG64 22.25 tok/s. This is not the
original pinned `125db33` checkout; it is recorded as the compatible R1
reference candidate in EXP-0269.

Historical raw results: `results/m18-runtime/20260909-105118-3881308/summary.json`.

## Decision

M12 is retained as an experimental path because it improves PP at useful
prompt lengths. The corrected evidence does not support a 600+ tok/s decode
claim. The compatible reference shows a preliminary PP gap of about 4.2x at
P512 (llama.cpp versus M12), making prefill architecture the next optimization
target rather than HTTP or tokenization overhead.

The original one-token record and its raw artifacts remain unchanged under
`results/m18-runtime/20260909-105118-3881308/`; this re-evaluation supersedes
only its decode-throughput interpretation.
