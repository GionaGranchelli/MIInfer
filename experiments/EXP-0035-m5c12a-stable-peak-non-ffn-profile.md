# EXP-0035 — M5-C12a stable-peak non-FFN profile

**Status:** CLOSED — C12b selected: attention differential/scaling  
**Milestone:** M5  
**Date:** 2026-09-01  
**Model:** Qwen3-8B Q4_0 (`Qwen3-8B-q4_0-b968826d.gguf`)

## 1. Question

With FFN GEMV and clock-state explanations excluded, which non-FFN family
accounts for the largest remaining real decode cost or context-dependent
differential?

## 2. Method and environment

The accepted MIInfer production path was measured with shared Gate/Up Q8 reuse,
GPU argmax, no rejected fusion candidates, and the current Release build
`134be1618364`. The MI50 was held at stock `stable_peak`:

```text
SCLK 1725 MHz
MCLK 1000 MHz
```

The model SHA-256 is
`458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628`.

The position audit used prompt token `14990`, positions `1,8,16,32,64`, and
GPU-side greedy argmax. Its raw artifact is retained at
`bench/results/20260901T-c12a-final/position-audit.json`.

The trace-free production benchmark used eight warmups, 64 measured decode
forwards, and five iterations. Its artifact is retained at
`bench/results/20260901T-c12a-final-fast/20260901T211214Z-433976/`.

The shape-matched external control used the pinned gfx906 llama.cpp build
`6e4ef6c1a`, the same model, `PP1/TG64`, five repetitions, and stable-peak
clocks. Because llama-bench uses its own synthetic tokens, this is a workload
shape control rather than a token-for-token correctness comparison.

## 3. Results

### MIInfer production path

The current trace-free benchmark measured:

| Metric | Result |
|---|---:|
| Decode | **55.419 tok/s** |
| Decode mean | 1154.844 ms / 64 tokens |
| TTFT | 15.908 ms |
| Deterministic IDs | PASS |
| Release correctness | PASS |

Position-scaled production timing and structural counters were:

| Position | Wall ms | Whole-token GPU ms | Deferred GPU ms | Attention ms | Dispatches | Syncs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.124 | 15.223 | 22.828 | 0.451 | 1553 | 38 |
| 8 | 15.767 | 15.724 | 23.289 | 0.958 | 1553 | 38 |
| 16 | 16.131 | 16.183 | 23.732 | 1.530 | 1553 | 38 |
| 32 | 17.214 | 17.504 | 24.965 | 2.655 | 1553 | 38 |
| 64 | **19.579** | **19.605** | **27.179** | **4.928** | **1553** | **38** |

All positions reported zero temporary allocations and `589,828` residual copy
bytes. At P64 the remaining category attribution was:

| Category | GPU ms | Dispatches |
|---|---:|---:|
| FFN projection | 6.819 | 108 |
| Attention | **4.928** | 36 |
| Quantization | 3.184 | 433 |
| Normalization | 2.835 | 289 |
| LM head | 2.975 | 1 |
| Conversion | 2.147 | 324 |
| Q/K/V preparation | 1.642 | 108 |
| RoPE | 0.535 | 72 |

Deferred category timers overlap and must not be summed as wall time. The
clean wall and whole-token GPU event are the authoritative end-to-end values.

### External shape control

Pinned llama.cpp measured **90.817 tok/s** for `PP1/TG64` at the same
stable-peak operating point. This is a controlled directional comparison; it
does not establish that llama.cpp's attention alone explains the full
end-to-end gap.

## 4. Interpretation

The structural counters are flat through P64:

```text
1553 dispatches/token
38 synchronization sites/token
0 temporary allocations/token
589828 residual copy bytes/token
```

The P1→P64 wall increase is `4.455 ms`, while attention increases by
`4.476 ms` (`0.451 → 4.928 ms`). Other major categories remain approximately
flat. Therefore the previous context-collapse defect is fixed, and attention
is now the only demonstrated context-growing cost in the P1–P64 range.

Normalization, quantization, and conversion remain substantial fixed-cost
families, but C10c already showed that blindly reducing their dispatch count
can regress the faster production kernels. No new fusion is selected from
C12a alone.

## 5. Decision

```text
C12a CLOSED — measurement-only
C12b selected: attention differential/scaling
```

C12b should isolate the cooperative cached-attention path against the
shape-matched external behavior at P64 and longer contexts, beginning with
position-scaled timing and memory/access attribution. It should be one bounded
attention experiment, not another generic dispatch or fusion sweep.

## 6. Follow-up

Do not reopen FFN GEMV or clock-state work. C12b should determine whether the
remaining attention cost is due to cache layout/access, reduction structure,
or another specific kernel-level mechanism. Test at least P64, P128, P256,
P512, and P1024 before selecting a production change.
