# EXP-0375 — Correct repeated-B128 scheduler dispatch

## Question

Do authored B128 remainder chunks execute through the EXP-0373-qualified
`prefill_full_layer_major_chunk()` route?

## Source defect

EXP-0374 used `count % 128 == 0 && count % 512 == 0`. Therefore `count == 128`
failed the full-layer-major gate and fell through to the generic layer-major
path.

## Correction

The dispatcher now accepts only the explicitly authorized counts:

```cpp
count == kFullPrefillCapacity
    || (exp0374_scheduler && count == kM12PrefillBatch)
```

No 192/256/384/448 support was added. With
`MIINFER_EXP0369_TRACE_ROUTE=1`, each selected chunk emits its `base`, `count`,
and `prefill_full_layer_major_chunk` path.

## Route proof and timing results

The release `miinfer` HIP target builds successfully after the correction. Runs
used `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`,
`MIINFER_PRESET=m25_hi_qualified`, the existing 1606/1000 MHz policy, and
`MIINFER_EXP0374_REMAINDER_SCHED=1`.

The observed hardware state was gfx906 with SCLK 1606 MHz, MCLK 1000 MHz,
edge/junction/memory temperatures 37/39/37 C, and no competing KFD process.
ROCm reported a low-power-state warning despite the measured clocks; this is
retained as a run-quality caveat rather than silently discarded.

Runtime trace proof:

```text
P640:  base=0 count=512; base=512 count=128
P768:  base=0 count=512; base=512 count=128; base=640 count=128
P896:  base=0 count=512; base=512/640/768 count=128
P1022: base=0 count=512; base=512/640/768 count=128; residual=126
P1023: base=0 count=512; base=512/640/768 count=128; residual=127
P1024: base=0/512 count=512
P1788: base=0/512/1024 count=512; base=1536 count=128; residual=124
```

Every listed route trace named `prefill_full_layer_major_chunk`.

| Point | Prompt tokens | Prefill |
|---|---:|---:|
| P512 | 512 | 2,504.95 ms |
| P640 | 640 | 17,314.14 ms |
| P768 | 768 | 31,578.48 ms |
| P896 | 896 | 44,632.25 ms |
| P1022 | 1022 | 108,040.56 ms |
| P1023 | 1023 | 107,091.22 ms |
| P1024 | 1024 | 5,768.19 ms (`--max-tokens 0`, prefill-only) |
| P1788 | 1788 | 85,891.57 ms (`--max-tokens 0`, later-base proof) |

The matched prefill-only P1024 control (scheduler disabled) was 7,562.82 ms;
the corrected scheduler was 5,768.19 ms. This single pair shows no B512
regression, but is not a repeated qualification series.

Incremental B128 costs were `14,809.19 ms`, `14,264.34 ms`, and `13,053.77
ms` for P640−P512, P768−P640, and P896−P768 respectively. Repeated B128 is
not collapsing on the second or third invocation, but it is still approximately
14 seconds per chunk, not the EXP-0373 P640 qualified result.

The P1022−P896 difference is `63,408.31 ms`, while P1023−P896 is
`62,458.97 ms`. This is consistent with the final 126/127-token residual
region dominating P1022/P1023 wall time. This attribution is still based on
single clean runs rather than a repeated A/B timing series.

The exact P640 and P768 `The`-repeated prompts produced 16 tokens in both
control and scheduler runs. At both points, generated token text was identical
(`The` repeated 16 times), and exported final-hidden buffers were byte-identical
across all 5,120 FP32 values and finite. Imported-state probes at both points
produced next token ID `104980` in control and scheduler; logits were
byte-identical and finite with 10/10 top-10 overlap. A restore-only import left
the P768 source state SHA256 unchanged (`1ddf8c9d...` before and after), proving
no-prefix mutation for that check. The P1024 prefill-only run supplies a valid
timing/control result.

## Decision

**OUTCOME B — REPEATED-B128 PRIMARY.** Route selection is proven, but corrected
P768 remains 31.58 s and the B128 increments are ~13–15 s each. Do not
authorize B64/B4 residual work. Next work is repeated-B128 state/runtime
performance attribution; the route and current model-boundary semantic gates
are otherwise qualified.
