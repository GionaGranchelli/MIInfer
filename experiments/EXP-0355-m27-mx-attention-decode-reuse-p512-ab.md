# EXP-0355 — M27 attention decode-reuse P512 A/B and context smoke

**Status:** KEEP as experimental M27 cold-prefill baseline; RETEST before promotion
**Date:** 2026-09-20
**Control commit:** 448c84e (bench: requalify current P512 control)
**Candidate tree:** same executable/source tree; candidate selector only
**Model:** Qwen3.8-27B-Q4_K_M, SHA-256
7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
**Hardware:** AMD Instinct MI50 / gfx906 / Wave64, SCLK/MCLK 1606/1000 MHz
**Software:** HIP 7.1.52802-9999, HIP Clang 20.0.0.rocm

## Hypothesis

Reusing resident Mx attention O/FFN weights for decode removes their duplicate
M23 copies and lowers persistent VRAM. The earlier three-pair screen also
showed a possible cold P512 improvement; test that with six matched pairs.

## Benchmark

Same Release executable, model, exact prompt, no generated tokens, no profiler,
fresh process for each sample. Each pair ran control then candidate. The exact
prompt formula and command shape were:

```bash
prompt=$(python3 -c 'print("The quick brown fox jumps over the lazy dog. " * 51 + "The quick", end="")')
MIINFER_PRESET=m25_hi_qualified build/mi50-release/miinfer run \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompt "$prompt" --max-tokens 0 --no-stream
```

The candidate command uses `MIINFER_PRESET=m25_interactive` in the same
command. The prompt tokenized to exactly 512 MIInfer tokens.

Control was MIINFER_PRESET=m25_hi_qualified; candidate was
MIINFER_PRESET=m25_interactive, which applies the same H/I wide-prefill
selectors plus MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_DECODE=1 and the decode
MMV selector. The zero-token P512 timing does not execute MMV decode.

Raw logs and continuous telemetry: /tmp/m27-p512-attn-decode-reuse-ab-448c84e/.
The exact same-binary control requalification is in EXP-0354. The pinned mx
oracle is not repeated in these pairs; its P512 prompt is synthetic, so this
experiment makes no new cross-runtime claim.

## Results

| pair | control ms | candidate ms | control − candidate ms |
|---:|---:|---:|---:|
| 1 | 2468.30 | 2379.57 | 88.73 |
| 2 | 2455.42 | 2381.54 | 73.88 |
| 3 | 2524.46 | 2405.07 | 119.39 |
| 4 | 2498.79 | 2434.39 | 64.40 |
| 5 | 2493.03 | 2383.63 | 109.40 |
| 6 | 2726.80 | 2405.94 | 320.86 |
| **median** | **2495.91** | **2394.35** | **99.065** |

Median candidate throughput is 213.84 tok/s versus control 205.14 tok/s;
median paired reduction is 3.969%. All six pairs favor the candidate. No
sample was discarded; the 2726.80 ms control remains included. Every run
processed 512 tokens and returned finite output.

Control allocation was 21,993,243,028 B; candidate allocation was
18,472,649,108 B, a reduction of 3,520,753,920 B. Continuous 250 ms
telemetry recorded 1635 samples: clocks remained at 1606/1000 MHz, junction
temperature ranged 34–60 C, package power averaged 36.4 W and peaked at 200 W
under the 225 W cap, and peak reported VRAM use was 22,945,730,560 B. No
samples were removed.

## Correctness

- Candidate --repeat-p512-check: PASS; first token 13477 (brown),
  continuation token 37550; repeat prefill 2456.09 ms.
- Candidate same-process allocation: 18,472,649,108 B; reported free VRAM
  15,057,551,360 B.
- Release CTest: 24/24 passed (13 GPU-required, 11 host-only).
- Session invalidation checks and the real Pi 0.85.1 tool-call continuation
  remain documented in EXP-0350 and commit 7bfe0b3.

## Long-context sanity

One no-profiler zero-generation smoke per size used the candidate preset and
--context 8192. Repeated-fox prompts yielded the actual token counts below.
The context runs are sanity evidence, not repeated performance qualification.

| workload | tokens | prefill ms | tok/s | device peak allocation |
|---:|---:|---:|---:|---:|
| P512 matched A/B median | 512 | 2394.35 | 213.84 | 18,472,649,108 B |
| P2K smoke | 2052 | 10620.80 | 193.21 | 18,942,439,828 B* |
| P4K smoke | 4102 | 36067.49 | 113.73 | 18,942,439,828 B* |
| P8K smoke | 8002 | 71058.32 | 112.61 | 18,942,439,828 B* |

* The P2K/P4K/P8K runs shared an 8K context capacity, so allocation is the
same across these runs. Their combined 626-sample telemetry peaked at
19,900,211,200 B VRAM use; SCLK varied between 1485 and 1606 MHz, MCLK stayed
1000 MHz, and junction temperature ranged 33–82 C. This telemetry is not
clock-qualified, and the three context latencies are single-run diagnostics.

Separate event-profile runs used the same H/I plus candidate selector vector,
context capacity 8192, and the selected final prompt position. The reported
recurrent/attention values sum selected-token stage events across the 48/16
layers. At P512, deferred whole-B512 tails are also recorded. These are
diagnostic event sums, can overlap across streams, and must not be added to
infer total prefill time. Profiling materially increased wall latency.

| workload | recurrent event sum | attention event sum | profile peak allocation |
|---:|---:|---:|---:|
| P512 | 3141.79 ms* | 1073.33 ms* | 18,942,439,828 B |
| P2K | 32.22 ms | 29.09 ms | 18,942,439,828 B |
| P4K | 32.12 ms | 364.16 ms | 18,942,439,828 B |
| P8K | 32.25 ms | 36.09 ms | 18,942,439,828 B |

* Includes recorded deferred whole-B512 tail spans and overlapping events;
not comparable to the selected-token sums at larger prompt sizes.

## Interpretation and decision

The candidate clears the required six-pair gate, reduces P512 latency by
99.065 ms median, exceeds 210 tok/s, and saves 3.52 GB of persistent
allocation. **KEEP** it as the experimental cold-prefill baseline for the
remaining forensic work. Do not make it the default yet.

The source path skips duplicate resident M23 attention FFN copies when the Mx
O/FFN decode reuse selector is active. That explains the allocation reduction,
but not why the exact same wide-prefill candidate also improves P512: active
Mx prefill kernels are unchanged, and this A/B does not isolate resource/cache
effects. The causal mechanism remains unresolved. Do not attribute the full
99 ms to the decode-reuse branch.

The main M27 recurrent QKV→GDN contract attribution remains open. Continue
measurement before any further optimization. The long-context smoke shows no
capacity failure through 8002 tokens; the variable-clock telemetry and
single-run timing prevent a long-context performance claim.

## Follow-up

Compare equivalent recurrent QKV→GDN semantic boundaries against pinned mx at
B512. Keep the candidate opt-in while establishing whether that contract
accounts for at least 40 ms of the remaining differential.
