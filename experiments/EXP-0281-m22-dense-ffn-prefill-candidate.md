# EXP-0281 — M22 dense FFN prefill candidate

## Hypothesis

Replacing recurrent-layer FFN gate/up Q4_K B4 GEMV with shared FP16/rocBLAS
batch GEMM will reduce the deferred-tail cost without slowing decode.

## Baseline

MIInfer release build, commit `474db556c84975a5ba1f97f9feec592cbd8304d0`,
`MIINFER_PREFILL_LAYER_MAJOR=1`, exact Qwen3.8-27B-Q4_K_M model.

## Environment

* Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
* GPU: AMD Instinct MI50/gfx906, SCLK 1606 MHz, MCLK 1000 MHz
* ROCm: 7.1.52802-9999; Clang 20.0.0.rocm
* Reference: mx-llama commit `2e9d29fe736969160f17476ec6f0a6298cee6966`

## Benchmark

PP-only runs used `--max-tokens 1`; TG used direct CLI runs with a 721-token
prompt and eight generated tokens. The opt-in FFN candidate was enabled with:

```bash
MIINFER_PREFILL_LAYER_MAJOR=1
MIINFER_PREFILL_DENSE_PROJECTIONS=1
```

The QKV dense subpath additionally requires the unselected
`MIINFER_PREFILL_DENSE_QKV=1` switch.

## Results

| case | MIInfer baseline | FFN candidate | mx reference |
|---|---:|---:|---:|
| P512 PP tok/s | 45.92 | 55.86 | 222.55 |
| P2048 PP tok/s | 42.83 | 52.69 | 236.71 |
| P8192 PP tok/s | 28.45 | 44.89 | 229.39 |
| P512 peak VRAM bytes | 24,977,998,164 | 29,973,676,372 | n/a |
| P512 FFN + dense-down | 55.86 | OOM | n/a |

The P512 improvement is 21.6%; P8192 improvement is 57.8%. The candidate
still runs 4.1x slower than mx at P512 and 5.1x slower at P8192.

TG on the same 721-token prompt was 35.07 tok/s candidate versus 35.13 tok/s
baseline (-0.2%, within the 2% regression limit). Candidate and baseline
produced identical first eight greedy token IDs:

```text
561 3841 13477 37550 14330 42903 4906 13
```

The QKV+FFN combined variant reached 57.31 tok/s at P512 but was not accepted:
its continuation did not preserve the baseline text. Adding the existing
dense FFN-down source on top of FFN densification exceeded available VRAM.

Raw artifacts are under `results/m22-candidates/`:

* `20260909-210516-1686233/` — combined QKV+FFN P512
* `20260909-212514-1895267/` — FFN-only P512 with tracked environment metadata
* `20260909-211229-1769810/` — FFN-only P512 replay
* `20260909-210657-1707087/` — FFN-only P2048
* `20260909-211856-1874034/` — FFN-only P8192
* `20260909-211710-1853172/` — FFN+dense-down OOM

## Correctness

Build and deterministic greedy replay passed for the accepted FFN-only path.
The QKV-dense subpath is rejected pending numerical/state validation.

## Interpretation

FFN gate/up is a real deferred-tail bottleneck, but its removal does not close
the dominant prefill gap. The extra persistent canonical weights cost about
5.0 GB and make further dense staging difficult within 32 GB.

## Decision

KEEP the FFN candidate as an experimental, opt-in path. Do not promote it or
enable QKV/dense-down by default. REJECT the combined QKV and FFN+dense-down
variants for correctness and memory reasons.

## Follow-up

Profile the remaining recurrent-core, SSM-output, and B4 down-projection work;
any next candidate must preserve the FFN replay result and fit the existing
context/memory contract.
