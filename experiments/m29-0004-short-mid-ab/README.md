# M29-0004 short/mid same-host A/B gate

This record closes the required fresh 4K/64K performance evidence before
M29-0005. Both runs used HP Z840, `HIP_VISIBLE_DEVICES=0`, the same
Qwen3.8-27B-Q4_K_M model, MIINFER HIP graphs, 128 generated tokens, and the
same temporary two-case benchmark (`4K`, `64K`).

| Context | Baseline 4f predecessor | M29-0004 candidate | Decode delta |
|---|---:|---:|---:|
| 4K | 38.19 ms/token, 26.2 tok/s | 38.06 ms/token, 26.3 tok/s | -0.34% latency |
| 64K | 60.72 ms/token, 16.5 tok/s | 60.57 ms/token, 16.5 tok/s | -0.25% latency |

Both contexts reported `VALID (No NaN/Inf/Collapse)`. Static VRAM was
25.41 GiB at 4K and 27.44 GiB at 64K in both runs. No credible decode
regression greater than 1% was observed.

Raw logs:

- `raw-baseline.log` SHA-256:
  `7e023b919f8eb261a7ca040630e44577a10c13d74e14d6afbb29000f9241f59e`
- `raw-candidate.log` SHA-256:
  `d44ce11c763025df51765c3ee359d398ec95bd170f4eb5c1cac24ef400494392`

The benchmark narrowing was temporary and is not part of the production
source. This evidence is a performance gate only; it does not alter the
attention hot path.
