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

Primary values are median prefill wall seconds and median decode milliseconds
per token. Each cell requires at least five valid samples and retained raw
results. A workload is invalid if evaluated prompt-token counts differ.

| Workload | MIInfer | mx-llama.cpp | upstream llama.cpp | Delta vs fastest | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| P512 prefill | pending | pending | pending | pending | pending |
| P1024 prefill | pending | pending | pending | pending | pending |
| P2048 prefill | pending | pending | pending | pending | pending |
| P4096 prefill | pending | pending | pending | pending | pending |
| P8192 prefill | pending | pending | pending | pending | pending |
| P64 decode TG128 | pending | pending | pending | pending | pending |
| P512 decode TG128 | pending | pending | pending | pending | pending |
| P2048 decode TG128 | pending | pending | pending | pending | pending |
| P8192 decode TG128 | pending | pending | pending | pending | pending |

The previous P8192 serving medians (38.123261 s MIInfer / 38.936731 s mx /
44.838664 s upstream) remain historical until reproduced under this matrix.

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
model-forward benchmark and document the exact scope.

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
