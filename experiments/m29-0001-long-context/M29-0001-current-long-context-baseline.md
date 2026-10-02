# M29-0001 — Current MIInfer long-context baseline

Status: complete for Workstream A; 128K feasibility boundary established.

## Authority and scope

- Host: HP Z840, one AMD Instinct MI50 gfx906
- Branch: `m29/long-context-baseline`
- Branch HEAD: `853bb07ac349875127712a481d43a057b6ea24b7`
- Qualified source base: `e61c2488f790cc83029c16a35c1312701ddc3f04`
- OCI image ID: `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Instrumentation commit: `853bb07ac349875127712a481d43a057b6ea24b7`

This is a production MIInfer baseline. No persistent-context architecture, KV representation, attention, or model-semantics changes were made.

## Method

The existing `miinfer-long-context-qualification-bench` was built and run in the qualified ROCm 7.2.1 MIInfer container. It evaluated the existing sequence `4K, 8K, 16K, 32K, 64K, 128K`, with 128 decode tokens per point. The requested M29 points are 16K, 32K, 64K, and 128K.

The host telemetry sampler ran concurrently and captured ROCm SMI output. Raw files are preserved under `raw/`.

## Results

| Context | Initialization / allocation | Weights | Recurrent state | KV cache | Workspace | Activation | Cached state | Total reported | VRAM / headroom | Prefill | Decode | Correctness |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 16K | succeeded | 24,068,487,168 | 158,859,264 | 2,147,483,648 | 727,711,744 | 21,966,848 | 158,859,264 | 27,283,367,936 (25.41 GiB) | 25.41 / 31.98 GiB; 5.70 GiB free | 86,225.92 ms; 190.0 tok/s | 40.09 ms/token; 24.9 tok/s | VALID |
| 32K | succeeded | 24,068,487,168 | 158,859,264 | 2,162,688,000 | 727,711,744 | 21,966,848 | 158,859,264 | 27,298,572,288 (25.42 GiB) | 25.42 / 31.98 GiB; 5.63 GiB free | 216,201.80 ms; 151.6 tok/s | 46.57 ms/token; 21.5 tok/s | VALID |
| 64K | succeeded | 24,068,487,168 | 158,859,264 | 4,325,376,000 | 727,711,744 | 21,966,848 | 158,859,264 | 29,461,260,288 (27.44 GiB) | 27.44 / 31.98 GiB; 3.63 GiB free | 617,527.52 ms; 106.1 tok/s | 60.07 ms/token; 16.6 tok/s | VALID |
| 128K | failed before evaluation | not emitted | not emitted | not emitted | allocation failed | not emitted | not emitted | not emitted | final telemetry observed 99% VRAM allocation | not run | not run | not run |

The 128K failure is exact and fail-closed:

```text
MIInfer HIP failure: out of memory
expression: hipMalloc(&d_buffer_, total_bytes_)
location: /workspace/MIInfer/src/prefill_v2/workspace.cpp:79
```

The 128K evaluation banner and memory breakdown were not emitted, so no unobserved allocation values are inferred. The evidence establishes the current practical boundary at 64K feasible / 128K workspace allocation infeasible on this host and environment.

## Telemetry and raw evidence

- `raw/long-context.log`: 161 lines; SHA-256 `8567b05fbf82ec984c32edad0fa396328b3295cbbfa5a70a439ddcb0b7e758a9`
- `raw/telemetry.jsonl`: 3,115 samples/records from the existing ROCm SMI sampler; SHA-256 `4e5e890535214945ddf78ec69e07788d79cd459352cb61a1854a3d8ce722bf44`
- The telemetry file contains the sampler's ROCm SMI text blocks despite its historical `.jsonl` filename; it is retained byte-for-byte.
- The final sampled GPU identity is gfx906; final samples include edge/junction/memory temperatures of 33/34/33 C. Per-sample clocks, power, temperatures, and VRAM fields remain in the raw telemetry.

## Worktree boundary

The host repository had pre-existing untracked `gpucore.3330886` and `graphify-out/reflections/`. They were preserved and not included in this evidence commit. Only this record and `raw/` are added.
