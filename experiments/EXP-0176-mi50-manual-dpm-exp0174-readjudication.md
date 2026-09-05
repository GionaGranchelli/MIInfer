# EXP-0176 — MI50 manual-DPM qualification and EXP-0174 re-adjudication

## Hypothesis

Fixing the MI50 at a supported lower SCLK will avoid thermal clock fallback
and make the EXP-0174 native Q4_K comparison reproducible.

## Configuration

- GPU: AMD Instinct MI50, gfx906, 32 GB HBM2
- Model: Qwen3.8-27B-Q4_K_M.gguf
- Fixture: `/tmp/m6a273-reference-p12`
- Driver state: `MANUAL`, SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz)
- Telemetry: `scripts/sample-gpu.sh`, 250 ms
- No voltage, firmware, power-cap, or thermal-control changes

## Sustained qualification

Three back-to-back TG128 control runs held SCLK at 1606 MHz for 505/505
samples. Throughput was 13.7518, 13.7624, and 13.7772 tok/s; junction peaked
at 88 C and VRAM at 67 C.

| Configuration | TG128 tok/s | SCLK residency | MCLK | Max junction | Max VRAM | Stable |
|---|---:|---|---|---:|---:|:---:|
| `profile_peak` / requested 1725 | 14.0526 prior | repeatedly 1725/1606/1485 | 1000 | 99 C | unavailable in run | No |
| manual 1606/1000 | 13.7518–13.7772 | 505/505 at 1606 | 1000 | 88 C | 67 C | Yes |

The manual 1606/1000 state is the highest tested reproducible operating point.
SCLK 1485 was not required.

## EXP-0174 A/B

Runs were interleaved control/native, five pairs per shape. Every run reported
replay PASS and zero decode allocations.

| Shape | Control median | Native median | Gain |
|---|---:|---:|---:|
| TG64 | 13.9682 tok/s | 14.5556 tok/s | +4.21% |
| TG128 | 13.7507 tok/s | 14.3171 tok/s | +4.12% |

TG64 telemetry held 1606/1000 for 989/989 samples, with junction/VRAM maxima
of 85/64 C. TG128 telemetry held 1606/1000 for 1671/1671 samples, with
junction/VRAM maxima of 89/67 C. Across the five TG64 runs, control ranged
13.9614–13.9785 and native 14.5373–14.5563 tok/s. Across TG128, control ranged
13.7469–13.7684 and native 14.3093–14.3234 tok/s.

## Amdahl reconciliation

At profile position 63 under manual 1606/1000:

| Profile metric | Control | Native | Delta |
|---|---:|---:|---:|
| Total GPU time | 73.9205 ms | 71.1741 ms | -2.7464 ms |
| Layer-sum GPU time | 70.6666 ms | 67.9199 ms | -2.7467 ms |

The family-level reduction is approximately 2.75 ms per profiled decode. The
observed 4.1% whole-token gain agrees with the exposed portion of the token
time; it does not need to reach an arbitrary 5% threshold.

VRAM after setup was 20,094,914,900 bytes control versus 18,802,544,980 bytes
native, a reduction of 1,292,369,920 bytes (about 1.20 GiB).

## Correctness

All ten A/B runs passed replay validation with zero decode-loop allocations.
The previously established native numerical and generation checks remain
valid.

## Decision

KEEP EXP-0174. The native Q4_K representation is a measured, reproducible
optimization at the qualified manual 1606/1000 operating point.

## Next action

A — extend the proven native format to the next largest compatible Q4
projection families. Do not begin that software work as part of this hardware
qualification experiment.
