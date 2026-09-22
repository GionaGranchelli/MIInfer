# EXP-0364 — Full-model Qwen3.8-27B prefill bottleneck attribution

## Question

Which phases account for the current MIInfer-versus-mx Qwen3.8-27B-Q4_K_M
prefill gap at P512, P2K, P4K, and P8K, and is attention still the single
next optimization target?

This is a protocol-authorized measurement experiment. It does not implement
an optimization, change production behavior, or reopen EXP-0360, EXP-0362, or
EXP-0363.

## Environment

| Item | Value |
| --- | --- |
| MIInfer commit | `69b2a57952466091dff1daf2fdd64d5b3a996d39` |
| mx-llama.cpp commit | `2e9d29fe736969160f17476ec6f0a6298cee6966` |
| Model | `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf` |
| Model hash | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| Quantization | Q4_K_M |
| GPU | AMD Instinct MI50/MI60, gfx906, Wave64 |
| ROCm/HIP | ROCm 7.1.52802-9999; HIP Clang 20.0.0.rocm (existing qualified record) |
| SCLK/MCLK policy | 1606/1000 MHz |
| Power policy | existing qualified policy, 225 W cap in the existing contract |
| External fan | physically fixed at full speed; ROCm fan telemetry not used |

The current spot check showed 1606 MHz SCLK, 1000 MHz MCLK, 34 C junction,
and 31 W package power before measurement. The runs in this record did not
retain continuous telemetry, so temperature/power qualification is weaker than
EXP-0354 and the absolute results are diagnostic rather than a replacement for
that qualified P512 record.

## Canonical MIInfer route

The canonical control is the qualified wide/layer-major vector represented by
`MIINFER_PRESET=m25_hi_qualified`, not the experimental
`m25_interactive` lower-footprint decode-reuse route. For contexts above the
preset's 1024-token capacity, the same qualified selector vector was run
explicitly with `MIINFER_CONTEXT_CAPACITY` set to the requested context. No
kernel selector, layout, or production source was changed.

The MIInfer command shape was:

```text
build/mi50-release/miinfer run Qwen3.8-27B-Q4_K_M.gguf \
  --prompt <repeated exact-fox prompt> --max-tokens 0 --no-stream --context N
```

The P512 prompt was the existing exact repeated-fox prompt and tokenized to 512
tokens. Repeating that text produced 2042, 4082, and 8162 tokens at the larger
points because the repeated text boundary is not token-neutral; those points
are therefore reported as approximately P2K/P4K/P8K with exact token counts.

## mx reference

The reference was `/home/fedora-workstation/Development/mx-llama-build/bin/llama-bench`
at `2e9d29f`, using Q4_K_M, gfx906, `-ngl 99`, flash attention, F16 K/V,
`-b 2048 -ub 512 -t 24`, and three repetitions. mx phase decomposition was
not available from this `llama-bench` binary. Its source/runtime decomposition
and the earlier temporary outer-span trace remain separate evidence; no source
label is presented here as a fresh timing claim.

## Benchmark matrix

### Full-model wall-clock results

| Point | Exact tokens | MIInfer ms | MIInfer tok/s | mx ms (measured point) | mx tok/s | Approx. gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P512 | 512 | 2553.84 | 200.48 | 2300.22 | 222.59 | +253.62 ms |
| P2K | 2042 | 86753.70 | 23.54 | 9258.02 at 2048 | 221.21 | +77522.81 ms* |
| P4K | 4082 | 94521.30 | 43.19 | 18706.45 at 4096 | 218.96 | +75878.80 ms* |
| P8K | 8162 | 120281.76 | 67.86 | 38224.86 at 8192 | 214.31 | +82196.90 ms* |

`*` Larger-point gaps normalize the mx latency linearly to the exact MIInfer
token count before subtraction. The prompt tokenization is not identical
across runtimes, so these are context-curve comparisons, not exact semantic
paired prompts.

The fresh MIInfer P512 result is slower than the historical 2486.685 ms
qualified control in EXP-0354. It must not be silently replaced by the old
number. The mx curve is stable across repetitions, with sample standard
deviation below 0.05 tok/s at P2K–P8K.

## MIInfer phase attribution

The existing `MIINFER_PREFILL_PROFILE=1` diagnostic facility was used at P512
with `MIINFER_PREFILL_PROFILE_POSITION=511`. It records HIP events around the
existing stages and reports M23 family invocation counts. Profiling adds
synchronization and its wall time is not a throughput measurement; the normal
P512 wall result above is authoritative.

### P512 current diagnostic profile

| MIInfer phase/family | GPU ms | Share of profiled stage sum | Evidence |
| --- | ---: | ---: | --- |
| Deferred wide tails, all layers | 1777.18 | 40.17% | MEASURED, diagnostic GPU events |
| FFN gate/up projection | 898.70 | 20.31% | MEASURED, selected-token/whole-B512 aggregation |
| FFN down projection | 504.87 | 11.41% | MEASURED, selected-token/whole-B512 aggregation |
| Projection or norm family | 380.09 | 8.59% | MEASURED, aggregated family label |
| Q/K/RoPE-related projection family | 191.29 | 4.32% | MEASURED, aggregated family label |
| Batch preparation | 190.38 | 4.30% | MEASURED |
| Attention output projection family | 180.58 | 4.08% | MEASURED, aggregated family label |
| KV/head-normalization family | 126.68 | 2.86% | MEASURED, aggregated family label |
| V projection | 57.63 | 1.30% | MEASURED, aggregated family label |
| K projection | 49.14 | 1.11% | MEASURED, aggregated family label |

The profile's stage sum was 4424.43 ms while normal wall time was 2553.84 ms;
the difference is expected from overlapping/deferred event scopes and
profiling synchronization. It is not valid to sum these components as a wall
clock prediction.

The deferred-tail breakdown was:

```text
GDN core                         167.419 ms
SSM output and residual          141.886 ms
FFN gate/up and SwiGLU           689.549 ms
FFN down and residual            381.872 ms
```

This identifies wide recurrent/FFN tail execution as the largest measured
MIInfer work family at P512. It does not establish its mx differential.

The larger-context profile attempts did not produce a valid selected-position
profile because the repeated-text prompts tokenized to 2042/4082/8162 rather
than the requested 2048/4096/8192 positions. Their clean wall-clock runs are
valid, but no P2K/P4K/P8K phase table is claimed here. This is an explicit
missing-measurement result, not an inference from the P512 profile.

## Dispatch census

The P512 diagnostic profile recorded these M23 invocation lower bounds:

```text
recurrent: qkv, gate, beta_alpha, ssm_out, ffn_gate, ffn_up, ffn_down
           48 launches each
attention: qk, v, q_post, k_post, attention, o, ffn_gate, ffn_up, ffn_down
           16 launches each
```

These are launcher-family counts, not total HIP dispatch counts. They imply
480 counted family invocations across the 64-layer model, but additional
embedding, preparation, tail, normalization, copy, and orchestration launches
are not represented. No conclusion that launch count alone explains the gap is
justified.

No weight uploads occurred in the profiled prefill (`0` bytes across the
reported families), so repeated weight transfer is not the current measured
explanation.

## Context scaling

| Runtime | P512 | ~P2K | ~P4K | ~P8K | Observed behavior |
| --- | ---: | ---: | ---: | ---: | --- |
| MIInfer ms | 2553.84 | 86753.70 | 94521.30 | 120281.76 | strongly superlinear jump to ~2K, then sublinear over these wide points |
| mx ms | 2300.22 | 9258.02 | 18706.45 | 38224.86 | approximately linear, with increasing per-token cost at longer context |

The MIInfer curve is not consistent with a simple attention-only linear
scaling explanation. The ~P2K jump and the measured P512 tail dominance point
to a route/phase interaction or a measurement-contract issue that requires
further attribution before any kernel hypothesis.

## Gap accounting and Amdahl analysis

Fresh phase-to-phase mx numbers are unavailable, so an exact comparable gap
table cannot be manufactured. The wall-clock gap is measured; MIInfer phase
shares are measured diagnostically; mx sub-phases are `NOT COMPARABLE` in this
run. The only prior comparable outer-span evidence is EXP-0356's one-run P512
contract comparison: +16.029 ms recurrent norm-to-GDN across 47 warm layers
and +101.979 ms attention norm/preparation-to-output across 16 layers. That
evidence is historical, one-run, and from a different route/measurement pass;
it is not substituted for the current curve.

Using the fresh wall gaps only, the absolute upper bound for eliminating all
MIInfer-vs-mx difference is:

| Point | Gap | Maximum full-model reduction if entire gap vanished |
| --- | ---: | ---: |
| P512 | 253.62 ms | 9.93% of MIInfer wall time |
| ~P2K | 77495.68 ms | 89.33% |
| ~P4K | 75878.80 ms | 80.28% |
| ~P8K | 82196.90 ms | 68.33% |

These are competitor-gap ceilings, not ceilings for any particular operator.
The current evidence does not support assigning a valid Amdahl ceiling to
attention, projections, GDN, or runtime separately at P2K–P8K. Selecting one
of them would violate the protocol's bottleneck and reconciliation gates.

## Historical assumptions re-evaluated

### Is attention still dominant?

Not established. At P512, the current MIInfer profile's largest measured
family is deferred wide tails, not the attention core. EXP-0356's earlier
attention outer-span differential is material but not current P2K–P8K evidence.
The large MIInfer scaling discontinuity near P2K further prevents an
attention-only conclusion.

### If attention is dominant, which part?

Not determined by this run. The diagnostic profile separates Q/K preparation,
KV store, attention, and O/FFN families, but the mx counterparts are not
available from the current binary.

### Is recurrent/GDN material?

Yes as MIInfer work: the P512 wide-tail profile attributes 309.305 ms to GDN
core plus SSM output/residual, and 1071.421 ms to FFN gate/up/SwiGLU plus FFN
down/residual. Its competitor differential remains unmeasured here.

### Are projections close enough to mx to reject broad MMQ work?

Not answerable from fresh comparable evidence. Broad MMQ work is not
authorized; the exact projection shape and mx kernel must be measured first.

### Is host/runtime/dispatch tax significant?

Not established. The counted family invocations are only a lower bound and the
current run did not collect a total HIP dispatch census or host-gap trace.

### Does the dominant gap change with context?

The observed total gap changes dramatically: +254 ms at P512 versus roughly
75–82 seconds at ~P2K–P8K. Phase attribution at those larger points is
missing, so the changing dominant phase is unresolved.

## Evidence quality and stop condition

The wall-clock curves are fresh and repeated for mx, but MIInfer has one run
per context and no retained continuous telemetry. MIInfer/mx prompts are not
semantically identical at the larger points, and mx sub-phase timing is absent.
Therefore this record is sufficient to reject speculative attention tuning,
but insufficient to authorize a new optimization target.

## Decision

**LEARN.** The current data does not support promoting an optimization or
selecting attention as the next target. It establishes that the current
MIInfer route has a severe ~P2K–P8K scaling discrepancy and that P512 measured
work is dominated by wide recurrent/FFN tails, while current mx phase
decomposition and full-model MIInfer phase attribution at larger contexts are
missing.

No performance optimization or new production kernel was implemented.

## Next PRIMARY frontier

At most one next target is authorized, and it is measurement-only:

> Refresh the attribution capability for the current canonical route at ~P2K,
> ~P4K, and ~P8K, with continuous telemetry and per-phase timings that
> reconcile to wall clock; first isolate whether the large ~P2K jump is a
> route/dispatch/orchestration effect or a required operator phase.

This is not a kernel optimization. Once that measurement reconciles, select
one exact phase/shape and study the corresponding mx implementation. Do not
reopen the rejected attention families or write EXP-0365 as an optimization
candidate before that gate passes.
