# EXP-0442 — GQA6 key-block stable reduction (rejected)

**Base:** `c7ff28f` (exact packed-U128 GQA6 control)  
**Branch:** `perf/m31-gqa6-attention-rewrite-gpu2`  
**Target:** Machinist MI50 GPU2, BDF `0000:88:00.0`, ROCm device 2  
**Result:** Rejected; production default and source are unchanged.

## Candidate

The candidate kept the same 32 split/token/KV-head grid and Q8 K/V inputs, but
changed Stage1 from serial per-key online-softmax updates to a stable FP32
four-key block reduction and merge. This changed floating-point association.
The compiled candidate used 180 VGPR, 51 SGPR, wave64, and zero VGPR/SGPR
spills. The graph captured and dispatched the candidate route in all 16
attention layers (`graph_state=1`); the paired control dispatched packed-U128
in all 16 layers.

## Results

| Restored prefix + suffix | Path | Suffix time | Logit max abs / cosine | State max abs | First two tokens |
|---|---|---:|---:|---:|---|
| 73,099 + 1,024 | Control | 22.274 s | reference | reference | 15, 8 |
| 73,099 + 1,024 | Candidate | 46.403 s | 13.564 / 0.918 | 16.010 | 15, 8 |
| 73,099 + 1 | Control | 0.094 s | reference | reference | 361, 19086 |
| 73,099 + 1 | Candidate | 0.105 s | 4.124 / 0.968 | 3.703 | 361, 889 |

The 1,024-token run failed correctness (required max error <=0.01 and cosine
>=0.9999), so its timing is diagnostic only; it took 2.08x as long as control.
The bounded one-token follow-up captured the first divergent attention output
at layer 3. Across its 6,144 values, max absolute error was 0.363884, mean
absolute error 0.002286, and cosine 0.929888. Layers 0–2 were exact; this is a
mapping/logic error, not acceptable FP32 reassociation. No simple indexing or
scaling bug was established. Work on this candidate stopped without tuning.

## Evidence and disposition

Raw logs, telemetry, source snapshot/patch, binaries, and captures are on the
Machinist host at:

`/home/machinist/m31-gqa6-longctx-results/EXP-0442-gqa6-kblock-reduction-gpu2/`

The saved source patch SHA-256 is
`f6a8bb268fd2701d5c555a49365a6713e512101f9575b55558f0d7595e073bcd`.
Layer-3 captures: `l3-control.bin` SHA-256
`85152d2f9e167e4cfff4b0dd17afb4a46da4feb7a7c46570b34ee6e68630d85b`;
`l3-candidate.bin` SHA-256
`3d5240e5c36f3e714773b64a3aba059c46a8a573bee10014040048930099dd64`.
The GPU stayed below the 80 C warning threshold (peak 56 C). The candidate
kernel and temporary instrumentation were reverted; this report is the only
change intended for this branch.
