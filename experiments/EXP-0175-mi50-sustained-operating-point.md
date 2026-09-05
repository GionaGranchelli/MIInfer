# EXP-0175 — MI50 sustained operating-point qualification

## Hypothesis

The Q4K native layout should retain its measured whole-token benefit when the
MI50 is observed under a reproducible sustained clock condition.

## Environment

- GPU: AMD Instinct MI50, gfx906, 32 GB HBM2
- Model: Qwen3.8-27B-Q4_K_M.gguf
- Profile: `profile_peak`
- Reported DPM states: SCLK 925/930/1032/1143/1282/1386/1485/1606/1725 MHz;
  MCLK 350/800/1000 MHz
- Power cap: 225 W
- Telemetry interval: 200 ms
- Correctness: replay PASS; decode allocations: 0

## Results

| Run | tok/s | SCLK residency | MCLK | Junction | Power |
|---|---:|---|---|---:|---:|
| TG64 control | 14.3088 | 1725: 70/114; 1606: 42/114; 1485: 2/114 | 1000 | 65–95 C | 111–261 W |
| TG64 native | 14.8580 | 1725: 68/115; 1606: 46/115; 1485: 1/115 | 1000 | 69–96 C | 111–257 W |
| TG128 control | 14.0526 | 1725: 99/192; 1606: 88/192; 1485: 5/192 | 1000 | 66–99 C | 111–261 W |
| TG128 native | 14.5967 | 1725: 86/190; 1606: 97/190; 1485: 7/190 | 1000 | 73–100 C | 114–267 W |

The native layout improves throughput by 3.8–3.9% in these paired runs, while
reducing setup VRAM from 20,094,914,900 to 18,802,544,980 bytes.

## Interpretation

`profile_peak` is not a stable benchmark operating point. Both workloads leave
1725 MHz repeatedly, and TG128 reaches the 100 C junction limit. Reported
power also exceeds the configured cap, so the available telemetry does not
identify a single limiter by itself.

The driver exposes lower DPM states and a `COMPUTE` power profile, but their
sysfs controls are root-owned and denied to this session. No voltage, firmware,
power-cap, or undocumented control was changed.

## Decision

DEFER hardware qualification. The native layout result is consistent with the
expected Amdahl-level gain, but the required stable-clock and five-repeat
reproducibility evidence is still missing.

## Next step

With authorized access to the normal driver control, test the supported lower
DPM/compute profile after cooling, then run five serial TG64 and TG128
repetitions. If no supported profile holds clocks and low variance below the
thermal/power boundary, classify the hardware envelope as blocked and stop
software performance experiments.
