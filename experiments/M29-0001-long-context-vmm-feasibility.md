# M29-0001 — Long-Context Baseline and HIP-VMM Feasibility

**Status:** INCOMPLETE — hardware gate prevented required measurements  
**Starting commit:** `2a86acba3ea285d817fcdf0bcbed5c2b5301baeb`  
**Release source:** `v0.2.0`, `94fad71ee19f539ce2ec0c7e100ad97d031dbefa`  
**Branch:** `rewrite/m29-0001-long-context-vmm-feasibility`

## Objective

Requalify the released contiguous-KV path at 16K, 32K, 64K, and 128K-class
capacity, then directly evaluate HIP-VMM allocation, mapping, access,
incremental growth, remapping, graph behavior, and access cost on the MI50.
This milestone does not change production KV ownership or attention.

## Starting-state audit

The starting checkout was exactly the requested canonical `main` SHA. Every
commit after the v0.2.0 source through that SHA changes documentation only;
`src/`, `include/`, `tools/`, `gfx906/`, and production runtime sources have no
diff from the release source. The released runtime is therefore the control.
The working branch was created at the starting SHA.

The current performance protocol has no optimization `PRIMARY`; this goal is
measurement and physical-backing feasibility only. The relevant context
records are EXP-0270 (dynamic context and replay qualification) and
EXP-V2-0013 (historical long-context capacity). Attention-family rejections
EXP-0360, EXP-0362, and EXP-0363 are not applicable because this goal does not
change or optimize attention. No prior HIP-VMM MI50 experiment record was
found.

## Immutable production boundary

Allowed source changes are limited to the long-context benchmark, the isolated
`miinfer-m29-vmm-probe` benchmark, and its CMake target. The VMM target is in
the existing benchmarks-off-by-default group. Production KV, attention,
prefill, decode, session reuse, generation, CLI, and server sources remain
unchanged.

## Long-context control contract

The released `miinfer run` and `miinfer serve` entry points construct
`miinfer::prefill_v2::PrefillV2Model` directly (see `cmd_run` and `cmd_serve` in
`tools/miinfer_cli.cpp`). The benchmark uses that same production class and
generation API, while supplying token IDs directly to control exact length.
Each process constructs it with the exact requested capacity. The prompt has
`capacity - 128` deterministic synthetic token IDs;
the run requests exactly 128 generated tokens with stop tokens disabled. Thus
the intended final active position equals capacity, including the 131,072-token
128K-class point. Each process runs one unmeasured warm-up and two recorded
repetitions. The two recorded generated-token streams must match exactly by
hash, and all token IDs and timing fields must be valid.

The benchmark reports exact context counts and runtime allocation components:
KV, recurrent state/history, weights, workspace, activations, cached state,
total runtime VRAM, and HIP-reported free/total memory; device-used bytes are
derived from total minus free. Raw per-run output,
environment snapshots, and 250 ms GPU telemetry are retained under
`results/m29-0001/contiguous-control/`.

| Capacity | Prompt | Decode | Prefill tok/s | Decode ms/tok | Decode tok/s | KV bytes | Total VRAM | Free headroom | Correctness |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 16,384 | 16,256 | 128 | pending | pending | pending | pending | pending | pending | INVALID attempt; no result |
| 32,768 | 32,640 | 128 | pending | pending | pending | pending | pending | pending | not run |
| 65,536 | 65,408 | 128 | pending | pending | pending | pending | pending | pending | not run |
| 131,072 (128K-class) | 130,944 | 128 | pending | pending | pending | pending | pending | pending | not run |

### Invalid environment attempt

The first 16K control attempt used the operator-selected `high` performance
level. It was stopped before a benchmark result because active telemetry showed
thermal clock fallback: over 368 samples, junction temperature peaked at
109 C, SCLK varied from 925 to 1725 MHz, and MCLK fell as low as 350 MHz.
The maximum reported device allocation was 27,128,512,512 bytes. The attempt
does not count as a completed or qualified capacity point. Its complete
telemetry and pre-run environment snapshot are retained at
`results/m29-0001/contiguous-control/20261001T193602Z-889474/`.

## Hardware and toolchain gate

Observed device identity: AMD Instinct MI50, PCI `0000:06:00.0`, gfx906,
34,342,961,152 bytes VRAM. PCIe reports 8.0 GT/s ×16. Kernel is
`7.2.4-200.fc44.x86_64`; host ROCm is 7.1.5 with HIP Clang 20.0.0 and GCC
16.2.1. The qualified host compiler combination is used for this experiment.
The available ROCm 7.2.1 toolbox image exposes GCC 13 / HIP Clang 22 and is not
used to silently change the compile environment. Configured power cap is 225 W.

The operator-selected `high` performance level idles at SCLK/MCLK 1725/1000
MHz, but the first 16K workload reached 109 C junction and throttled down to
925 MHz SCLK and 350 MHz MCLK. The project's highest reproducible qualification
point is manual 1606/1000 MHz (EXP-0176). Host sysfs DPM writes require root;
manual DPM is therefore awaiting operator action. No long-context result is
qualified until its active telemetry meets the established clock/thermal
contamination checks. `rocm-smi` reports no GPU metric interface for this card;
use its supported clock, junction temperature, power, and process fields.

Model artifact: `Qwen3.8-27B-Q4_K_M.gguf`, SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

## Probe compiler-resource gate

The built gfx906 code object was inspected before any probe timing. The
sequential `copy_transform` kernel uses 4 VGPR and 11 SGPR, zero VGPR/SGPR
spills, zero private bytes, zero LDS, 256 work items, and Wave64. Its helper
kernels also have zero spills and zero private/LDS bytes (`fill_pattern`: 5
VGPR/12 SGPR; `check_pattern`: 4/12; `fill_float`: 3/12;
`check_transform`: 6/20). No unexpected compiler resource use is present.

## Regression checks

Clean Release production build from this branch passed with
`MIINFER_BUILD_BENCHMARKS=OFF` and `MIINFER_BUILD_RESEARCH_TOOLS=OFF`, using
GCC 16.2.1 and HIP Clang 20.0.0. Release CTest checks passed: 11/11 host tests
and 14/14 gfx906 GPU correctness tests. These checks do not qualify long-context
inference; they verify the isolated benchmark addition did not break the
production runtime.

## HIP-VMM probe design

The isolated probe derives KV bytes/token from the actual loaded Qwen model
layer count and MIInfer's FP16 `[head, token, dim]` KV constants. It records
minimum/recommended granularity and uses the recommended value for physical
and virtual alignment. The derived reservation is 256K logical tokens with no
physical backing; the incremental physical ladder is 16K, 32K, 64K, and 128K
equivalent KV bytes. It does not select a final page-token size.

The probe records every HIP status and duration for reserve, create, map,
access, unmap, release, and free. It checks basic GPU write/read, stable VA with
later adjacent mapping, same-VA remapping, graph replay before and after
growth/remap, and matched 64 MiB-per-buffer hipMalloc/VMM sequential access.
Each bandwidth side has nine raw samples with warm-up, mean, median, standard
deviation, relative delta, and correctness. `hipMemGetInfo` captures physical
VRAM changes after each commitment step and after cleanup.

## Predeclared decision rules

- **VMM_FEASIBLE:** required VMM lifecycle, logical reservation, incremental
  commitment, GPU access, remapping, stable-address graph replay, graph growth,
  and same-VA remap all work repeatedly; bandwidth shows no obvious stable
  access penalty; no unexplained allocation leak remains.
- **VMM_REJECT:** an unsupported required primitive, incorrect GPU access,
  incompatible graph behavior, unreliable mapping, or repeatable material
  steady-state penalty prevents the required semantics.
- **VMM_INCONCLUSIVE:** environmental/toolchain instability prevents a valid
  functional decision; list the missing evidence.

The global <=1% production decode regression limit is not tested here because
production decode will not be integrated with VMM in this milestone.

## VMM evidence

The probe builds, and its gfx906 resource usage passed the compiler gate, but it
was not executed. The required preceding contiguous control could not pass the
hardware gate: the `high` profile produced severe thermal clock fallback at
16K. Consequently, no VMM lifecycle, graph, incremental commitment, or
matched-bandwidth conclusion is available. No VMM run directory exists under
`results/m29-0001/vmm/`.

## Final decision and next recommendation

**VMM_INCONCLUSIVE — hardware qualification prevented a valid test.** The
exact software control is the released v0.2.0 contiguous KV implementation
through `miinfer::prefill_v2::PrefillV2Model`; the 16K/TG128 contract is
implemented in the research benchmark, but no capacity point has yet produced
a qualified result. The maximum functional and qualified context, available
VRAM headroom at that maximum, and VMM feasibility remain unknown. Keep HIP-VMM
as an unproven candidate and rerun the four controls first after the MI50 is
placed in the project's qualified manual 1606/1000 MHz state. Then run the VMM
probe only if the control telemetry passes. Do not infer a maximum context or
choose `pageTokens` from the failed attempt. No `ContextSpace`, logical pages,
device KV pool, production KV change, or attention change is part of this
milestone.
