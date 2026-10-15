# EXP-0455 — Four-wave GQA6 query/head tile

**Status: REJECTED — exact, but slower than direct-global GQA6.**

## Question

Can one 256-thread CTA share one staged Q8 K/V tile across four adjacent
queries and two query heads per KV group, while preserving each row's original
Split-K interval and ordered online-softmax updates?

## Resource screen

Built for gfx906 with the pinned ROCm 7.2.1 image
`localhost/miinfer-m31-rocm721-libdw@sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`.
Compiler metadata confirmed a 256-thread workgroup (four Wave64s), 53 VGPR,
57 SGPR, 2,080 bytes of LDS, zero register spills, zero private bytes, and
four static barriers across the full and tail loops. The direct-global control
reported 60 VGPR, 50 SGPR, no LDS, and zero spills/private bytes. This passed
the compile/resource gate.

## Restored suffix screen

One paired run used GPU2, BDF `0000:88:00.0`, on the pinned image and identical
Qwen3.8-27B Q4_K_M model, restored session, tokenized prompt, and 1,024-token
suffix. The session had 73,099 tokens before the suffix and 74,123 afterward.
The in-process identity was `gfx906:sramecc-:xnack-` at the expected BDF.

| Route | Restore | 1,024-token suffix |
|---|---:|---:|
| Direct-global control | 2.845 s | 20.905 s |
| Four-wave candidate | 2.938 s | 25.182 s |

The candidate took 1.205x the control time, a 20.5% regression (speed ratio
0.830x). It is rejected; no repeat or cold-prefill run is justified.

Correctness passed in the same pair: full session logits exact, recurrent-state
trace exact, generated tokens equal, output finite, and final position 74,123.
Explicit route tracing showed 16/16 layers on direct-global for control and
16/16 on four-wave for candidate, both with `graph_state=1` and
`token_count=1024`.

Temperature peaked at 57 C against the 80 C warning / 85 C stop guard. The
final GPU2 temperature was 33 C, and the KFD workload was idle after the run.

## Launcher correction

The first attempted launch exited before inference because it passed only
`/dev/dri/renderD130` to the container, so HIP found no GPU. This was a setup
failure, not a candidate result. The successful rerun used the known-good
campaign passthrough: `--device /dev/kfd --device /dev/dri --group-add
keep-groups --security-opt label=disable`, with `ROCR_VISIBLE_DEVICES=2`.
Container `rocminfo` exposed exactly one GPU agent (gfx906, chip `0x66a3`,
BDFID `0x8800`), and the inference process itself reported
`device_bdf=0000:88:00.0`.

Raw logs and thermal telemetry are retained on Machinist at
`/home/machinist/m31-attn-rewrite-gpu2-0455/results/EXP-0455-four-wave-gpu2-r2/`.

## Decision

The four-wave geometry and its source implementation were removed from this
branch after the measured regression. Keep direct-global GQA6 as the control
and production route. This experiment does not justify another wave-count or
tile-size variant; a future attention candidate must change the dataflow in a
meaningful way and must clear an initial resource/performance screen before
long-context work.
