# M29-0003 Workstream A — 128K lifetime/allocation fix

Status: `128K_LIFETIME_FIX_QUALIFIED`

Base: `072072874b63344ed8884333fd135c52411038ae`

Host: HP Z840, AMD Instinct MI50/MI60 family, `gfx906`, 34,342,961,152-byte
VRAM. The run used `HIP_VISIBLE_DEVICES=0`. Model SHA-256:
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

## Change

The existing monolithic workspace reserved split-K scratch for 32 splits at
every KV capacity. At 128K this made the 727,711,744-byte workspace allocation
fail with 356,515,840 bytes free. The production path now preserves 32 splits
through the qualified 64K capacity (`66000`) and sizes the same scratch arena
to 3 splits above it. The selected split count is carried in the existing
workspace metadata and passed to the existing split-K launcher. KV precision,
layout, kernels, VMM, and context architecture are unchanged.

Workspace accounting:

| Case | Workspace bytes | Result |
|---|---:|---|
| Before, 128K | 727,711,744 | OOM; deficit 371,195,904 bytes |
| Candidate, 4 splits | 372,637,696 | OOM; deficit 16,121,856 bytes |
| Final, 3 splits | 359,956,480 | Constructs and executes |

## 128K end-to-end result

The focused qualification completed construction, 131,072-token prefill, 128
decode tokens, and numerical validity on the selected MI50:

- static accounting: 33,366,452,224 bytes / 31.07 GiB;
- observed free headroom: 0.05 GiB;
- prefill: 2,315,053 ms / 56.6 tok/s;
- decode: 87.76 ms/token / 11.4 tok/s;
- correctness: `VALID (No NaN/Inf/Collapse)`.

The observed free headroom is approximately 53 MiB. A 32 MiB operational
safety reserve is retained, leaving approximately 21 MiB measured margin.

The normal 4K–64K benchmark path was rebuilt after the focused run. The
production selection leaves the existing 32-split path unchanged for 64K
(`kv_capacity=66000`); prior same-host controls were 4K 26.6 tok/s decode and
64K 16.2 tok/s decode. The temporary one-context benchmark narrowing was not
committed.

## Verification

- `cmake --build --preset mi50-release --target miinfer-long-context-qualification-bench -j2`: PASS;
- focused 128K qualification: PASS;
- raw output: `raw-128k.log`;
- raw SHA-256: `a60281424931baaec5f839c2154b6241761c6493d7af0604f00433c4117297bd`;
- pre-existing `gpucore.3330886` and `graphify-out/reflections/` were preserved.
