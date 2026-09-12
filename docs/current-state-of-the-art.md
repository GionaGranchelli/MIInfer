# MIInfer Current State of the Art

Status: active M25 stretch investigation, 2026-09-12

## Target

The benchmark target is Qwen3.8-27B-Q4_K_M on one AMD Instinct MI50
(`gfx906`, Wave64), with SCLK/MCLK fixed at `1606/1000 MHz`:

- exact model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- P512: 512 repeated-fox prompt tokens, context capacity 1024
- command: `--max-tokens 0 --no-stream`
- primary gate: at least 200 prompt tokens/s, or at most 2560 ms
- external stretch oracle: `mx-llama.cpp` commit
  `2e9d29fe736969160f17476ec6f0a6298cee6966`

## Leaderboard

| implementation | status | P512 ms | prompt tok/s | allocation |
| --- | --- | ---: | ---: | ---: |
| `mx-llama.cpp` repacked | historical external oracle | 2317.872 | 220.892 | external |
| `mx-llama.cpp` `llama-bench` fresh screen | current external comparison | 2307.634 median | 221.872 | external |
| MIInfer H/I, pre-repair best | historical, not qualification | 2421.07 | 211.48 | 18,472,649,044 B |
| MIInfer H/I, repaired qualified | current accepted path, opt-in | 2500.62 median | 204.75 | 21,993,242,964 B |
| MIInfer H/I + Mx attention decode reuse | interleaved opt-in screen; promotion pending | 2412.24 median | 212.25 | 18,472,649,044 B |
| MIInfer matched control | current comparison | — | — | 19,108,282,708 B |

The H/I path clears the primary gate but is not the default. The external
stretch gap is `182.748 ms/P512` at the qualified medians.

## Retained MIInfer path

The current best accepted P512 configuration is the printed
`MIINFER_PRESET=m25_hi_qualified` vector. It combines:

- resident, gfx906-native Mx Q4/Q5/Q6 repacked weights;
- the measured staged MMQ kernel with interleaved affine DP4A ordering;
- Mx Q8 activation blocks, with the newer M25-J batch quantizer disabled;
- the state-correct Mx GDN scan with the Tc2 mapping;
- M25-H Mx attention Gate/Up/Down projections;
- M25-I Mx attention O projection;
- resident M23 copies retained where scalar decode requires them;
- static full-layer-major B512 prefill orchestration.

An additional opt-in candidate,
`MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_DECODE=1`, reuses the H/I Mx O and FFN
weights for attention decode and removes the duplicate M23 attention-FFN
representation. It saves `3,520,753,920 B` in the measured Q4_K_M layout. The
selector now fails closed unless resident-all attention weights are active. The
qualified preset still retains the M23 decode copies until the candidate has
completed a full promotion run.

H/I has passed finite scalar parity, the `00/10/01/11` configuration matrix,
real continuation, repeat-P512, CTest, and the long-generation gate. Its
allocation and live VRAM are now reported separately from cumulative
allocation, with `hipMemGetInfo` recorded at qualification boundaries.

## External-port results

The pinned source was used as an oracle and port source where the contract was
compatible. The following complete or isolated ports were measured on MI50:

| experiment | result | decision |
| --- | --- | --- |
| Mx compact Q4/Q5/Q6 MMQ | large primitive wins over M23 | keep |
| Mx GDN register scan, Tc2 | production-safe and faster than the old scan | keep |
| pinned large-batch MMQ contract | `+14.47%` P512 latency | reject |
| pinned MMQ on recurrent FFN only | `+11.98%` P512 latency | reject |
| pinned vectorized epilogue | `+8.4–10.4%` projection latency | reject |
| pinned two-GEMM beta/alpha | `+26.7%` at the exact shape | reject |
| pinned four-column GDN, Tc4 | slower in six-process qualification | reject for default |
| M25-J four-block Q8 quantizer | no qualified gain; kept disabled | reject |
| Mx attention Q/K | slower or outside parity tolerance | reject |
| Mx attention O/FFN decode reuse | `~3.28 GiB` less allocation; `16.98 tok/s` in a 128-token decode screen with Mx MMV | keep opt-in |

The evidence says the remaining stretch is not explained by a missing literal
Q4/Q5/Q6, Q8, or GDN source transplant. The static HIP graph was slower, so
generic host submission overhead is not currently supported as the explanation.
The unresolved question is the GPU execution contract around the primitives:
materialization boundaries, operation composition, and the exact recurrent FFN
schedule.

## Measured remaining work

The fresh diagnostic profile showed the resident H/I layer loop at roughly
`2769 ms` of summed ordered GPU time versus `2803 ms` wall time. This was a
diagnostic profile, not a throughput qualification, because profiling adds
events and changes timing. EXP-0337 tested the resulting hypothesis with an
opt-in static HIP graph. The graph passed continuation correctness but was
`0.44%` slower by three-process median and consumed about `8 MiB` of extra
reported VRAM, so it was removed.

The remaining stretch gap therefore still requires a different execution
contract. The new attention decode reuse candidate is not that stretch
contract: its three-pair interleaved screen is `2412.24 ms` / `212.25 tok/s`,
with the real continuation and repeat-P512 state gate passing. Its main
benefit remains the lower VRAM footprint; long-context and decode promotion
checks are still open. The `220.892 tok/s` stretch target remains open.

M25-L is now measurement-only. A fresh six-pair, no-profiler screen measured
the pinned `llama-bench` path at `2307.634 ms` / `221.872 tok/s` median and
MIInfer H/I at `2474.610 ms` / `206.901 tok/s` median, a `166.976 ms` or
`7.236%` latency gap. The oracle benchmark does not accept the exact text
prompt, so it uses the same model and P512 benchmark shape but synthetic
benchmark tokens; that limitation is recorded in EXP-0340. The current MIInfer
profile measures recurrent layer 60's whole-B512 FFN tail at `22.2694 ms`, but
there is not yet a like-for-like oracle stage boundary. No M25-L optimization
has been written.

## Reproducibility and promotion rules

Every performance claim must use clean processes, the exact model hash and
prompt, repeated measurements, continuous `1606/1000 MHz` telemetry, live
allocation plus `hipMemGetInfo`, and correctness evidence. A candidate is
promoted only when its absolute P512 result survives an interleaved A/B and
the continuation/repeat-P512 gates. The old P512 stall remains classified as
transient device/runtime or harness state: the exact pre-J/current-main A/B
did not reproduce it with M25-J disabled.

Detailed evidence is indexed in [`current-state.md`](current-state.md) and
the M25 records `EXP-0300` through `EXP-0340`.
