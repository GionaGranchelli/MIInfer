# EXP-0420 — Packed U4 GQA6 cold 64K prefill

**Status:** PASS for exactness; promising cross-card speedup; 64K target missed
**Source:** `5a9c115afdfa65e0a53941c78a43f21ab086f78c`
**Base optimization:** packed Q8 LDS staging, `4ab1d0c` / `d930ed1`
**Target:** Qwen3.8-27B Q4_K_M, 65,536 prompt tokens, Q8/Q8 KV

## Method

Built the benchmark from the pinned source commit in the pinned
`miinfer-m31-rocm721-libdw` image (`sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`).
Model SHA-256 was
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`;
binary SHA-256 was
`6dcad764a6581fbf90115f9025fe689e4a8a1ecaefb98781e098107c8c4084ff`.
The same 663,758-byte tokenized prompt file was used for both paths, with
capacity 69,632. Each MI50 ran one path at the same time, with the existing
80 C warning / 85 C stop guard attached:

| Role | MI50 / BDF | GQA6 arithmetic | Packed U4 |
| --- | --- | :---: | :---: |
| Control | GPU0 / `0000:04:00.0` | on | off |
| Candidate | GPU2 / `0000:88:00.0` | on | on |

This is a concurrent cross-card screen, not a same-card qualified timing pair.

## Results

Both runs exited 0 after processing exactly 65,536 prompt tokens. Both emitted
248,320 finite float32 logits, chose first/second tokens `271` / `248068`, and
produced the same complete-logit SHA-256:
`9a33d3f3b190126806dee63074189df0f83b78503eac9f2f80e881babb16f482`.
The packed candidate trace confirms 1,008 packed-U4 dispatches across the 16
attention layers; the first tile at position zero uses the existing fallback.

| Path | BDF | Prefill | Max junction |
| --- | --- | ---: | ---: |
| GQA6 control | `0000:04:00.0` | 948.970 s | 79 C |
| Packed U4 | `0000:88:00.0` | 802.708 s | 68 C |

The concurrent screen favors packed U4 by 1.182x (15.4% lower prefill time).
For context, the previous single-card GPU2 GQA6 result was 1,023.715 s; this
candidate is 21.6% lower than that historical result, but the runs had
different concurrency and thermal conditions. Do not present that as a
same-card matched gain.

The candidate remains 4.46x slower than the 180 s target. Packed U4 is worth
keeping as the current exact candidate, but it does not close the mission.
Both raw run directories are retained on Machinist:

- `/home/machinist/m31-gqa6-longctx-results/EXP-0420-gqa6-cold64k-control-GPU0-20261014T1800Z/`
- `/home/machinist/m31-gqa6-longctx-results/EXP-0420-packed-u4-cold64k-GPU2-20261014T1800Z/`

## Next action

Refresh the late-context stage attribution with packed U4 enabled, then target
the largest remaining measured stage. Do not repeat the cold 64K run merely to
increase sample count; use a same-card suffix pair if a stable packed-U4
timing claim is needed.
