# EXP-0174 — Wave-plane Q4K Down integration

## Hypothesis and scope

EXP-0173 cleared the isolated primitive gate (~1.91x B59 throughput).
Convert only the 32 Q4_K FFN Down tensors at setup, replacing expanded copies.
`MIINFER_Q4K_NATIVE_DOWN=1` enables the candidate; unset or `0` retains control.
Other values fail explicitly. Both recurrent and full-attention callers use
the same exact-shape packer and kernel. Other tensor families are unchanged.

Baseline source commit: 84e411e4a63a. Candidate is the uncommitted working diff.
Model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`.
Fixture: `/tmp/m6a273-reference-p12`.

## Correctness so far

Release build of benchmark and native generation harness passes.
Native `--generate16`: replay PASS, state fingerprint 7170666995386517766,
zero decode allocations. Device bytes 18,802,544,980 versus baseline
20,094,914,900: saving 1,292,369,920 bytes (~1.204 GiB).

`--prefix64-observable-contract` completed with final PASS and
`poisoned_reset_replay=PASS`. Its intermediate state mismatch diagnostics are
not exact-reference claims; acceptance is the existing observable contract.
Log: `/tmp/exp0173-native-contract.log`.

## Resources (compiler metadata, not hardware counters)

| Resource | B41 canonical | B59 expanded | Wave planes |
| --- | ---: | ---: | ---: |
| VGPR/thread | 25 | 28 | 34 |
| SGPR | 20 | 16 | 20 |
| Scratch bytes | 0 | 0 | 0 |
| Register spills | 0 | 0 | 0 |
| Static LDS bytes/workgroup | 1024 | 1024 | 1024 |
| Dynamic LDS bytes, this shape | 22304 | 0 | 0 |
| Waves/workgroup | 4 | 4 | 4 |

Obtained by `llvm-objdump --offloading` followed by `llvm-readelf --notes` on
the gfx906 code objects. Native has greater register demand, despite lower
weight traffic. Occupancy has not been measured; theoretical capacity alone
must not be reported as achieved occupancy.

## End-to-end diagnostic run (in progress)

Artifacts: `/tmp/exp0173-integration.afDDfY/`.
Same binary, control then candidate for TG64, followed by TG128. Each process
does warmup and five samples. `scripts/sample-gpu.sh` captures at 250 ms.

TG64: control 14.289 tok/s (4478.98 ms); candidate 14.8077 tok/s
(4322.09 ms). Both replay PASS, zero allocations. This is approximately
3.63% throughput improvement, below the required 5% rollout gate.

**Clock-contaminated:** telemetry records 1485/1606/1725 MHz SCLK under
profile_peak at the existing 225 W cap; MCLK is 1000 MHz. This run cannot
establish a stable-1725-MHz win or a definitive rejection at that clock.
No power-cap or clock-setting changes were made. Preserve these raw samples.

TG128, Release CTest, complete profile comparison, and final decision pending.
No additional Q4 family is enabled and the candidate is not production-selected.

## Completed integration checks and interpretation

Durable raw artifacts copied to `bench/results/exp0173-integration.afDDfY/`.
Native generation and contract logs are included there. The failed sandbox
attempt is retained separately from the `*-gpu` run files.

| Workload | Control tok/s | Native tok/s | Throughput change |
| --- | ---: | ---: | ---: |
| TG64, five-sample medians | 14.2890 | 14.8077 | +3.63% |
| TG128, five-sample medians | 14.0188 | 14.4730 | +3.24% |

TG128 medians: control 9130.57 ms, candidate 8844.08 ms. Both workloads
passed deterministic replay with zero decode-loop allocations. Release CTest
passed 20/20 after rebuilding all targets. CTest is the existing regression
suite; the new layout itself is exercised by the isolated benchmark and the
opt-in native generation/observable runs, not by an implicit new CTest case.

| P63 GPU profile | Control ms | Native ms | Saving ms |
| --- | ---: | ---: | ---: |
| Total token | 71.9130 | 69.2465 | 2.6665 |
| Layer sum | 68.6483 | 65.9990 | 2.6493 |

Profiles passed with zero profile allocations. These are single diagnostic
profiles, not repeated performance acceptance measurements. No dedicated
telemetry sampler accompanied the profile pair. End-to-end timing telemetry
has 260 samples at 1725 MHz, 258 at 1606 MHz, and 12 at 1485 MHz; it does not
support claiming fixed-1725-MHz throughput. The 225 W cap remained unchanged.

The measured layer saving closely matches the isolated estimate of 2.752 ms.
This is evidence that the faster primitive survives integration. The limiting
factor is its share of the workload: only 32 Down matrices are Q4_K. The
larger representative Down stage costs cited in the previous campaign include
other quantization types. Eliminating half of the actual Q4 Down time cannot
be assumed to eliminate half of all Down time.

## Decision

**RETEST / NO ROLLOUT.** Retain the candidate opt-in and preserve the validated
layout and negative gate evidence. It has not demonstrated the required >=5%
whole-token improvement. Do not enable additional families or declare the goal
complete. A stable-clock confirmation is still missing, and changing acceptance
to permit a 3–4% result would be a change to the stated goal.

If fixed-clock confirmation remains below 5%, report the first integration
gate as unmet and recommend the separately authorized coarser execution-plan
campaign. Do not silently redefine success as the isolated 1.91x result.

## Hardware revalidation — 2026-09-05

Read-only sysfs checks confirm `profile_peak` remains selected. At idle the
card returns to 1725/1000 MHz, edge 52 C and junction 66 C. Sustained-run
telemetry includes junction readings of 100 C; hwmon `temp2_label` is junction
and `temp2_crit` is 100000 millidegrees C. This is evidence of reaching the
reported thermal limit, not proof that every sampled frequency dip was thermal.
Some lower-clock samples also show elevated instantaneous power.

`power1_cap`, `power1_cap_default`, and `power1_cap_max` all report 225000000
microwatts. Raising the cap is not an available ordinary setting on this device.
Fan telemetry reports `fan1_input=0`, `fan1_target=0`, `fan1_enable=1`,
`pwm1=51`, and `pwm1_enable=1`; these do not establish control over a passive
MI50's external airflow. No fan, voltage, power, or clock controls were changed.

Further fixed-1725-MHz acceptance measurements require a hardware state that
maintains those clocks under sustained decode, including adequate cooling.
Blindly repeating the same hot run or filtering away low-clock samples would
not resolve the acceptance gap. The candidate remains opt-in and the >=5%
whole-token requirement remains unmet. No second layout or additional family
has been introduced to work around this measurement limitation.
