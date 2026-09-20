# EXP-0352 raw comparison artifacts — 2026-09-20

Model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`  
SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`  
GPU: MI50/MI60 gfx906, manual 1606/1000 MHz, 225 W cap  
ROCm: HIP `7.1.52802-9999`, clang `20.0.0.rocm`  
Current code: `7ef2a9263b84ebbaa24f7b63da1ddf7b037fed19`  
Historical code: `ff7aefffbe33ea386aeef0e12d1a8623fc6ff58d`

## Historical code + reconstructed fixture workload

Command:

```sh
MIINFER_DEVICE_TOKEN_CHAIN=0 \
MIINFER_Q4K_NATIVE_DOWN=1 MIINFER_Q4K_NATIVE_GATE_UP=1 MIINFER_Q4K_NATIVE_Q=1 \
MIINFER_Q4K_NATIVE_ATTN_GATE=1 MIINFER_Q4K_NATIVE_ATTN_OUT=1 MIINFER_Q4K_NATIVE_K=1 \
MIINFER_Q5K_NATIVE_SSM_OUT=1 MIINFER_KQUANT_NATIVE_QKV=1 MIINFER_KQUANT_NATIVE_V=1 \
MIINFER_Q6K_NATIVE_DOWN=1 MIINFER_Q6K_NATIVE_LM_HEAD=1 MIINFER_HIP_GRAPH=1 \
MIINFER_FUSED_GATE_UP_SWIGLU=1 MIINFER_FUSED_RECURRENT_CORE=1 MIINFER_FAST_ARGMAX=1 \
MIINFER_TILED_ONLINE_ATTENTION=1 MIINFER_FUSED_ROPE_NORM=1 \
MIINFER_FUSED_ADD_RMS_NORM=1 MIINFER_FUSED_INTERLAYER_NORM=1 \
MIINFER_WAVE64_Q8=1 MIINFER_FUSED_CORE_Q8=1 MIINFER_FUSED_NORM_Q8=0 \
MIINFER_SWIGLU_SHUFFLE=1 MIINFER_Q6K_SIMD_UNPACK=1 \
/tmp/miinfer-m26b-m8-build/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-reconstructed --bench64
```

Raw output: [historical-ff7aefff-reconstructed-bench64.log](historical-ff7aefff-reconstructed-bench64.log)

Median: `2132.74 ms / 64 = 33.324 ms/token` (`30.0083 tok/s`), replay PASS,
zero decode allocations. This matches the recorded `33.01 ms/token` claim within
1%; the original `-p12` fixture is absent, and this replacement fixture is not
claimed byte-identical.

## Current code + same reconstructed fixture workload

Use the identical command and environment above, replacing the executable with
`build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block`.

Raw output: [current-ff7aefff-workload-bench64.log](current-ff7aefff-workload-bench64.log)  
Telemetry: [current-legacy-historical-workload-telemetry.jsonl](current-legacy-historical-workload-telemetry.jsonl)

Observed median: `1999.83 ms / 64 = 31.2473 ms/token` (`32.0028 tok/s`), replay
PASS, zero decode allocations. The telemetry includes two isolated 1485 MHz
samples during the long setup/run interval; the five per-sample timings are not
timestamped, so do not treat this as a clock-qualified speedup. The earlier
recorded current-code result was `31.385 ms/token` with a contemporaneous
1606/1000 MHz spot check. Both measurements point away from a shared-pipeline
regression, but the historical non-comparability conclusion does not depend on
claiming this small A/B delta.

## Current interactive P512/TG128

Command:

```sh
MIINFER_PRESET=m25_interactive build/mi50-release/miinfer run MODEL \
  --context 16384 --decode-curve --curve-context 512 --curve-iterations 5 \
  --no-stream --max-tokens 128
```

Raw output: [current-interactive-p512-tg128.log](current-interactive-p512-tg128.log)

Median: `7321.7 ms / 128 = 57.2007 ms/token`. An in-run spot check showed
1606/1000 MHz and 100% GPU busy; this run did not have continuous telemetry.

## Current legacy P512/TG128 attempts

The first full run is retained in
[current-legacy-p512-tg128.log](current-legacy-p512-tg128.log): `32.2533
ms/token`. A spot check during that run caught SCLK at 1485 MHz before it later
returned to 1606 MHz, so do not qualify or use this result as the current legacy
control.

A second run enabled 250 ms ROCm SMI sampling. It was stopped when SCLK fell to
1485 MHz at 93°C junction; see
[current-legacy-clock-drop-telemetry.jsonl](current-legacy-clock-drop-telemetry.jsonl)
and [current-legacy-p512-tg128-aborted-clock-drop.log](current-legacy-p512-tg128-aborted-clock-drop.log).
No timing result was emitted. The previously recorded qualified current legacy
result (`32.2948 ms/token`, in-run 1606/1000 MHz) remains the control cited in
EXP-0352.

## Interpretation

The historical run and current interactive run use different starting state,
decode position range, execution route, and synchronization/timing boundary.
The legacy attempts show why a fresh paired route measurement needs continuous
clock telemetry. These artifacts establish historical reproducibility and
current interactive output; they do not create an apples-to-apples historical
versus interactive speed comparison.
