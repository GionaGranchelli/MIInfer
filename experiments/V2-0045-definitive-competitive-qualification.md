# V2-0045 — Definitive competitive qualification

## State

PASS — competitive matrix and bounded serving checks complete; no production
kernel/runtime changes were made. The immutable MIInfer production baseline is `main` at
`81ce0e982220613453889a9a5178f11440ea6b22`. Qualification work is isolated on
`rewrite/v2-0045-competitive-qualification`.

## Pinned starting point

| Item | Pin / evidence |
| --- | --- |
| MIInfer production baseline | `81ce0e982220613453889a9a5178f11440ea6b22` |
| Production path | M28 Prefill V2, `MIINFER_PRESET=m25_hi_qualified`, iteration-26 attention (`MIINFER_V2_0043_GQA_ATTENTION=1`) |
| Rejected stretch | V2-0044 FP16 candidate absent from production dispatch; failed attribution preserved in `EXP-V2-0044` |
| Model | Qwen3.8-27B-Q4_K_M, `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf` |
| Model SHA-256 | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| mx-llama.cpp | `2e9d29fe736969160f17476ec6f0a6298cee6966` |
| upstream llama.cpp | `73a43d1f69345aee8bb186ef4b3172cef892f2e5` |
| GPU | AMD Instinct MI50 32 GB, gfx906, Wave64 |
| Qualification environment | `vllm-gfx906-7.2.1`; ROCm 7.2.1 / HIP 7.2.53211 / HIP Clang 22.0.0git |
| Working tree at start | tracked files clean; untracked `gpucore.3330886` preserved |

The configured 225 W cap was verified with `rocm-smi -a`. During sampled
benchmark blocks, SCLK/MCLK remained 1606/1000 MHz and PCIe remained 8.0 GT/s
×16. The highest observed junction temperature was 84°C and highest observed
socket power was 202 W, below cap. Each benchmark process was the sole MI50
KFD owner; no overlapping GPU workload was present. The host's low-power-state
warning was also emitted while active clocks were at their target values.

## Required scoreboard

Primary values are median prefill wall milliseconds and median decode milliseconds
per token. Each cell requires at least five valid samples and retained raw
results. A workload is invalid if evaluated prompt-token counts differ.

| Workload | MIInfer | mx-llama.cpp | upstream llama.cpp | Delta vs fastest | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| P512 prefill | 2179.94 ms | 2279.13 ms | 2566.41 ms | -4.35% | WIN |
| P1024 prefill | 4367.99 ms | 4575.84 ms | 5172.70 ms | -4.54% | WIN |
| P2048 prefill | 8838.83 ms | 9195.04 ms | 10415.89 ms | -3.87% | WIN |
| P4096 prefill | 18044.90 ms | 18576.33 ms | 21068.14 ms | -2.86% | WIN |
| P8192 prefill | 37633.89 ms | 37982.16 ms | 42978.20 ms | -0.92% | WIN (narrow) |
| P64 decode TG128 | 33.81 ms/token | 38.47 ms/token | 41.43 ms/token | -12.1% | WIN (forced model-forward replay) |
| P512 decode TG128 | 34.07 ms/token | 38.54 ms/token | 41.41 ms/token | -11.6% | WIN (forced model-forward replay) |
| P2048 decode TG128 | 34.50 ms/token | 39.31 ms/token | 41.99 ms/token | -12.2% | WIN (forced model-forward replay) |
| P8192 decode TG128 | 36.88 ms/token | 42.77 ms/token | 43.45 ms/token | -13.8% | WIN (forced model-forward replay) |

The previous P8192 serving medians (38.123261 s MIInfer / 38.936731 s mx /
44.838664 s upstream) remain historical until reproduced under this matrix.
The raw samples and deterministic prompt hashes are preserved in
[`results/v2-0045/prefill-p512.json`](../results/v2-0045/prefill-p512.json),
[`results/v2-0045/prefill-p1024.json`](../results/v2-0045/prefill-p1024.json),
[`results/v2-0045/prefill-p2048.json`](../results/v2-0045/prefill-p2048.json),
[`results/v2-0045/prefill-p4096.json`](../results/v2-0045/prefill-p4096.json),
and [`results/v2-0045/prefill-p8192.json`](../results/v2-0045/prefill-p8192.json).
All five prefill points favor MIInfer; P8192 is a narrow 0.92% lead over mx,
with non-overlapping five-sample ranges. All four forced-token model-forward
replay points also favor MIInfer. The P64 MIInfer harness smoke is excluded.
There is no losing cell in the measured matrix. The decode cells are not
production HTTP `generate()` timings: MIInfer uses its narrowest forced-input
`decode_step` model-forward route, while the references use `llama-bench
test_gen`. Preserve that scope when presenting the result; do not translate it
into a claim about server tokens/s.

## Attention frontier — deferred

MIInfer main attention was measured at approximately 1.91× mx. The FP16
reference-like path failed real-model parity; the numerical cause was partially
attributed in V2-0043/44. This is future research only. Do not revisit attention
unless the final competitive scoreboard or real multi-turn agent workload
shows it is materially limiting the product.

## Qualification protocol

Use one deterministic token corpus across runtimes, preserve its artifact and
hash, and record BOS handling and evaluated counts. The pinned GGUF uses the
`qwen35` pre-tokenizer and omits `tokenizer.ggml.add_bos_token`; both reference
builds and MIInfer therefore use no BOS, with vocabulary size 248320. The raw
token IDs are generated from the default glibc `rand()` seed and validated
against independent SHA-256 streams. Prefill is timed separately
from a minimal TG1 tail. Decode must replay the same predetermined 128 input
tokens after the same prompt/context; naturally sampled, different token
trajectories are not a definitive equivalent-work comparison. If a runtime
cannot replay through its public interface, use its narrowest internal
model-forward benchmark and document the exact scope. For decode, MIInfer uses
the internal forced-input `PrefillV2Model::decode_step` (including its device
argmax/token handoff), while references use `llama-bench test_gen`; this does
not time MIInfer's production HIP-graph `generate()` wrapper. The exact P64
prompt and each TG128 sequence hash are recorded in
[`results/v2-0045/decode-p64-tg128.json`](../results/v2-0045/decode-p64-tg128.json).
Raw P512, P2048, and P8192 replay distributions and input hashes are retained
in their correspondingly named `results/v2-0045/decode-p*-tg128.json` records.

For each workload, collect at least five samples, raw timing output, VRAM,
temperature, clocks, and process ownership. Interleave runtimes where possible.
Do not change production code, kernel schedules, precision, or hardware policy.

## Real agent workload and stability gate

The selected real serving run used the production HTTP route with
`m25_interactive`, session reuse enabled, 16K capacity, a 9,404-token initial
prompt, repository tools, and three sequential logical turns. Each turn
generated over 1,100 tokens across its tool request and visible answer. Its
tool-result continuation reused 9,404, 11,728, and 13,791 exact prefix tokens;
visible answers were 3,515, 985, and 1,842 characters. First-request TTFTs
were 44.1, 56.1, and 67.6 seconds; logical-turn wall times were 98.4, 114.0,
and 123.0 seconds. These include long-context prefill and substantial answers;
they are an MIInfer serving characterization, not a competitor comparison.
Per-request context, client timing, reused-prefix, suffix estimate, and VRAM
data are in [`agent-workload-final.json`](../results/v2-0045/agent-workload-final.json);
the runnable driver is [`v2_0045_agent_workload.py`](../bench/v2_0045_agent_workload.py).
Short-budget pilot outputs are retained separately and rejected because some
turns exhausted their budget in the reasoning channel before producing a
visible answer; they are not counted as the selected run. These pilots are
`agent-workload-attempt1.json`, `agent-workload.json`,
`agent-workload-visible.json`, and `agent-workload-qualified.json`.

The separate ten-inference-request sequence passed its count, request-success,
three-reuse-cycle, and P8192-class/TG128 gates. The long request used 7,886
prompt tokens (the predeclared P8192-class band) and generated exactly 128
tokens: 36.38 s prefill, 5.45 s decode, 41.83 s total. All three reuse cycles
hit, restoring 9,399, 9,405, and 9,405 tokens; two subsequent same-context
follow-ups reused 10,102 and 10,125 tokens. VRAM was 27,198,156,800 bytes at
both the start and end (zero observed growth), peak observed temperature was
68°C, and clocks remained 1606/1000 MHz. There were no inference errors or
GPU faults; the process exited cleanly and released its KFD ownership.
See [`stability-workload.json`](../results/v2-0045/stability-workload.json).
The driver is [`v2_0045_stability.py`](../bench/v2_0045_stability.py).

The P8192-class/TG128 stress request reached its token cap inside Qwen's
reasoning channel and returned no user-visible content. This is not an
output-quality sample; it does establish the requested decode load and clean
runtime completion. The visible agent run above used a larger output budget.
The server's raw `new_prefill_tokens` field reported the full prompt size even
on cache hits; the JSON's per-turn `new_prefill_tokens` is therefore derived as
prompt minus reused-prefix tokens for each hit (all tokens on a miss). The
separate server `steady_decode_tok_s` field was not used because subtracting
`first_token_ms` from `decode_ms` produced implausible outliers in this run.

## Decision

V2-0045 PASS: all nine matrix cells favor MIInfer within the declared benchmark
boundaries; the three-turn prefix-reuse flow and bounded stability gates
completed without runtime errors or VRAM growth. No next performance milestone
is justified by this scoreboard. Freeze `main` at the new baseline, stop
performance research for this generation, and move to release/serving polish.
Keep production decode and agent wall-time comparison caveats visible. The
approximately 1.91× slower isolated attention kernel remains deferred research,
not a goal by itself.
