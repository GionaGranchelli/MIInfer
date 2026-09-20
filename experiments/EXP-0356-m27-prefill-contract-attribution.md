# EXP-0356 — M27 P512 semantic contract attribution and stop

**Status:** REJECT FP16 Q/K selector for cold P512; STOP cold optimization
**Date:** 2026-09-20
**MIInfer baseline tree:** `ab88603` plus profiling-report instrumentation only
**Pinned oracle:** `mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`
**Model:** Qwen3.8-27B-Q4_K_M, SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
**Hardware:** AMD Instinct MI50 / gfx906 / Wave64

## Question and baseline

Does an equivalent outer GPU span around the recurrent input contract explain
at least 40 ms of the P512 gap? If not, does another measured semantic family
meet that threshold?

The accepted cold P512 context is 1024 tokens. EXP-0355's separate
`m25_interactive` six-pair A/B remains the current experimental baseline at
2394.35 ms / 213.84 tok/s, 99.065 ms faster than its exact control. This
experiment did not retest that selector; it measured one existing FP16 Q/K
path against the same H/I plus decode-reuse selector vector.

## Outer contract boundaries

MIInfer recurrent span: the existing stage-0 normalization event through
stage-6 GDN output gate. MIInfer attention span: B512 batch preparation
(normalization and Q/K/V projection) through stage-7 attention output
projection. The attention preparation start is required because wide attention
produces normalized Q/K/V before the per-token stage events.

Mx span: `norm-N` node start through completion of `final_output-N` or
`attn_output-N`. The temporary source tracer records its end event after the
named boundary node executes. An earlier tracer version recorded it before
the node; those runs are superseded and are not used below. Mx graphs were
disabled. Per-node tracing additionally disabled fusion and is used only for
layer-63 decomposition.

All values below are one diagnostic P512 run per runtime, not qualification
timings. Selected-token spans are diagnostic and may overlap other queued GPU
work. Mx compute-graph node count is not HIP kernel-dispatch count. MIInfer's
M23 invocation counters are launcher-family counts, not a total HIP dispatch
count. Neither run collected ROCProfiler HBM/L2/occupancy counters.

## Contract results

| semantic contract | Mx measured | MIInfer measured | delta (MI − Mx) | dispatch evidence | materialization / confidence | possible P512 contribution |
|---|---:|---:|---:|---|---|---:|
| recurrent norm → GDN output gate, 47 non-first layers | 529.287 ms sum; 11.318 ms median/layer | 545.316 ms sum; 11.470 ms median/layer | +16.029 ms across 47 | Mx: 21 compute graph nodes/layer; MIInfer: dispatch total unavailable | Temporary tensors and actual HBM traffic not measured. Medium confidence for timing; one run. | 16 ms observed aggregate; below the 40 ms gate. No candidate. |
| attention norm/preparation → attention output projection, 16 layers | 177.109 ms sum; 11.062 ms median/layer | 279.088 ms sum; 17.446 ms median/layer | +101.979 ms across 16 | Mx: 16 compute graph nodes/layer with default fusion. MIInfer M23 counters: 16 Q/K and 16 V launches across 16 layers; total HIP dispatch count unavailable. | One layer-63 node trace: Mx node output sizes sum to 991,952,896 B, but this includes 822 MiB logical KV-cache view outputs and is **not** bytes written or HBM traffic. No reliable HBM/L2 estimate. Medium confidence for the equivalent outer timing; one run. | 102 ms aggregate span delta. This is a family attribution, not a predicted end-to-end saving. |

Layer 0 is retained as a first-layer anomaly: Mx recurrent span was 336.701 ms
with the corrected boundary, versus MIInfer 14.522 ms. The other 47 layers are
reported separately above; the anomaly's cause is unknown and it is not
discarded from the raw logs.

The representative layer-63 Mx node trace (fusion disabled) measured Q
projection 4.980 ms, K projection 0.811 ms, V projection 0.768 ms, online
attention 1.028 ms, and output projection 2.792 ms, plus normalization,
RoPE, gating, and cache-store nodes. MIInfer's corresponding selected-token
events measured Q/K preparation 10.365 ms, V projection 1.381 ms, attention
2.571 ms, and output projection 2.756 ms. The outer event is authoritative
for the contract comparison; individual event components are diagnostic and
should not be summed into a whole-model prediction.

## Candidate screen

The existing `MIINFER_PREFILL_REPACKED_FP16_ATTN_QK` path was screened at the
qualified 1024-token context, exact 512-token repeated-fox prompt, fresh
processes, no profiler, and fixed SCLK/MCLK `1606/1000 MHz`. The common
selectors were the `m25_interactive` H/I plus Mx decode-reuse vector; the Q/K
FP16 selector was the only A/B change. Each process allocated
18,472,649,108 B. Continuous telemetry captured 719 samples; every sample
reported 1606/1000 MHz, junction temperature 32–56 C, peak package power
183 W, and peak reported VRAM 19,365,388,288 B.

| pair | control ms | FP16 Q/K ms | control − candidate ms |
|---:|---:|---:|---:|
| 1 | 2433.25 | 2500.12 | -66.87 |
| 2 | 2534.44 | 2486.62 | +47.82 |
| 3 | 2502.18 | 2503.67 | -1.49 |
| **median** | **2502.18** | **2500.12** | **-1.49** |

No samples were removed. The selector does not produce a repeatable 40 ms
whole-P512 reduction; its paired median is a 1.49 ms regression. **REJECT** it
for cold P512. An earlier three-pair run at context 8192 had a 35.39 ms paired
median improvement; it is retained as a context-specific diagnostic only and
does not override the exact 1024-context result.

Raw logs:

- MIInfer profile: `/tmp/m27-context-smoke/mi-contract-fixed.log`
- Mx corrected outer spans: `/tmp/m27-context-smoke/mx-contract-after-node.log`
- Mx layer-63 node profile: `/tmp/m27-context-smoke/mx-attn63-nodes-final.log`
- 1024-context candidate A/B and telemetry: `/tmp/m27-context-smoke/qk-fp16-screen-1k/`
- 8192-context diagnostic A/B and telemetry: `/tmp/m27-context-smoke/qk-fp16-screen/`

The external Mx checkout's temporary diagnostic edit was restored after
collection; `git status --short` was clean at `2e9d29f`. No Mx source change is
part of this repository.

## Interpretation and decision

The recurrent input contract does not meet the 40 ms attribution gate. The
attention contract exceeds it by about 102 ms in the one-run outer-span
comparison, primarily around Q/K preparation and attention. However, the
existing FP16 Q/K route failed to produce a repeatable whole-P512 win at the
qualified context. No new kernel or runtime selector is retained. **STOP cold
P512 optimization and redirect to exact agent prefix/state reuse.** Keep the
decode-reuse cold-P512 result from EXP-0355 explicitly experimental; the
cross-runtime benchmark still has its synthetic-token prompt caveat.

## Environment and command shape

The P512 prompt file contains the exact text
`'The quick brown fox jumps over the lazy dog. ' * 51 + 'The quick'` with no
trailing newline. The current-build preset sanity sample was:

```bash
MIINFER_PRESET=m25_interactive build/mi50-release/miinfer run \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompt "$(cat /tmp/m27-context-smoke/p512.txt)" \
  --max-tokens 0 --no-stream
```

It measured 2404.47 ms / 212.94 tok/s and 18,472,649,108 B allocated. This is
one sanity sample, not a replacement for EXP-0355's six-pair qualification.

The selector-screen runs used this exact common environment for both A and B;
only `MIINFER_PREFILL_REPACKED_FP16_ATTN_QK` changed from 0 to 1:

```bash
MIINFER_CONTEXT_CAPACITY=1024
MIINFER_PREFILL_LAYER_MAJOR=1
MIINFER_PREFILL_WIDE_CHUNK=1
MIINFER_PREFILL_FULL_LAYER_MAJOR=1
MIINFER_PREFILL_WIDE_ATTN=1
MIINFER_PREFILL_WIDE_REPACKED_MMQ=1
MIINFER_PREFILL_WIDE_MX_REPACKED_MMQ=1
MIINFER_PREFILL_WIDE_MMQ_QKV=1
MIINFER_PREFILL_WIDE_MMQ_SSM_OUT=1
MIINFER_PREFILL_WIDE_MMQ_FFN=1
MIINFER_M23_REPACKED_ROW128=1
MIINFER_PREFILL_REPACKED_RESIDENT_ALL=1
MIINFER_PREFILL_CHUNK=512
MIINFER_PREFILL_MX_GDN=1
MIINFER_MX_Q8_BATCH=0
MIINFER_MX_PINNED_QKV=0
MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_FFN=1
MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_O=1
MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_DECODE=1
MIINFER_MX_MMV=1
MIINFER_PREFILL_REPACKED_FP16_ATTN_QK=0  # candidate: 1
```

The command was:

```bash
env <the assignments above> \
  build/mi50-release/miinfer run \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompt "$(cat /tmp/m27-context-smoke/p512.txt)" \
  --max-tokens 0 --no-stream --context 1024
```

The unabridged runner is retained at
`/tmp/m27-context-smoke/run-qk-screen-1k.sh`. Both runtime profiles used
context capacity 8192 to expose all layer events; those event runs are
diagnostic only. The Mx outer profile command used
`MIINFER_M27_CONTRACT_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1` with
`-ngl 99 -fa on -ctk q8_0 -ctv f16 -b 2048 -ub 2048 -n 1
--no-conversation -f /tmp/m27-context-smoke/p512.txt --perf`.
