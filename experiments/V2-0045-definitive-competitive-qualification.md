# V2-0045 — Definitive competitive qualification

## State

IN PROGRESS — qualification only; no production kernel/runtime changes are
authorized. The immutable MIInfer production baseline is `main` at
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

The configured 225 W cap and active-run telemetry still need verification.
Idle telemetry showed 1606/1000 MHz SCLK/MCLK, PCIe 8.0 GT/s ×16, 35°C
junction, ~10 MB VRAM use, and no `/dev/kfd` owners. Idle runtime state was
reported low-power; each accepted benchmark block must re-check clocks under
load.

## Required scoreboard

Primary values are median prefill wall milliseconds and median decode milliseconds
per token. Each cell requires at least five valid samples and retained raw
results. A workload is invalid if evaluated prompt-token counts differ.

| Workload | MIInfer | mx-llama.cpp | upstream llama.cpp | Delta vs fastest | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| P512 prefill | 2179.94 ms | 2279.13 ms | 2566.41 ms | -4.35% | provisional MIInfer win |
| P1024 prefill | 4367.99 ms | 4575.84 ms | 5172.70 ms | -4.54% | provisional MIInfer win |
| P2048 prefill | 8838.83 ms | 9195.04 ms | 10415.89 ms | -3.87% | provisional MIInfer win |
| P4096 prefill | 18044.90 ms | 18576.33 ms | 21068.14 ms | -2.86% | provisional MIInfer win |
| P8192 prefill | 37633.89 ms | 37982.16 ms | 42978.20 ms | -0.92% | narrow MIInfer win |
| P64 decode TG128 | 33.81 ms/token | 38.47 ms/token | 41.43 ms/token | -12.1% | provisional internal replay win |
| P512 decode TG128 | pending | pending | pending | pending | pending |
| P2048 decode TG128 | pending | pending | pending | pending | pending |
| P8192 decode TG128 | pending | pending | pending | pending | pending |

The previous P8192 serving medians (38.123261 s MIInfer / 38.936731 s mx /
44.838664 s upstream) remain historical until reproduced under this matrix.
The raw samples and deterministic prompt hashes are preserved in
[`results/v2-0045/prefill-p512.json`](../results/v2-0045/prefill-p512.json),
[`results/v2-0045/prefill-p1024.json`](../results/v2-0045/prefill-p1024.json),
[`results/v2-0045/prefill-p2048.json`](../results/v2-0045/prefill-p2048.json),
[`results/v2-0045/prefill-p4096.json`](../results/v2-0045/prefill-p4096.json),
and [`results/v2-0045/prefill-p8192.json`](../results/v2-0045/prefill-p8192.json).
All five prefill points favor MIInfer in this screen; P8192 is a narrow 0.92%
lead over mx and requires careful variance review. Decode validation,
agent-serving measurements, and stability remain pending.
decode validation, and agent-serving measurements remain pending. The P64
MIInfer harness smoke is excluded.

## Attention frontier — deferred

MIInfer main attention was measured at approximately 1.91× mx. The FP16
reference-like path failed real-model parity; the numerical cause was partially
attributed in V2-0043/44. This is future research only. Do not revisit attention
unless the final competitive scoreboard or real multi-turn agent workload
shows it is materially limiting the product.

## Qualification protocol

Use one deterministic token corpus across runtimes, preserve its artifact and
hash, and record BOS handling and evaluated counts. Prefill is timed separately
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

For each workload, collect at least five samples, raw timing output, VRAM,
temperature, clocks, and process ownership. Interleave runtimes where possible.
Do not change production code, kernel schedules, precision, or hardware policy.

## Real agent workload and stability gate

After the matrix, run one fixed three-turn coding-agent conversation (~8K-token
initial context, tool definitions and repository context, ≥128 generated tokens
per turn). Enable MIInfer's production prefix/session reuse. Record per turn the
context and reused/new token counts, TTFT, prefill, decode, wall time, and VRAM.
Then run the bounded ten-request MIInfer stability check including three reuse
cycles, a P8192-class request, and TG128.

## Decision

Pending matrix, agent workload, stability, and memory/context review. If every
important cell is WIN or defensible PARITY and serving is stable, close
performance research for this generation. If exactly one important cell loses,
make that one cell the next narrow milestone. Do not start a new attention goal
solely because its isolated kernel remains slower.
