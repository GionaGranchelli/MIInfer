# EXP-0473 — Query-dependent GQA6 block selection

## Goal

Test whether selecting older 512-token KV blocks from the current query can retain task-relevant context more reliably than EXP-0469's fixed anchors, while reducing Stage1 work. The path is opt-in and approximate; the production default remains the exact direct-global Q8 path.

## Candidate design

For every suffix query and query head, score 16 evenly spaced key rows in each complete 512-token block before the 2,048-token recent window. Keep the 16 highest-scoring older blocks, sort their block IDs into position order, and run the existing Q8/FP32 online-softmax attention over those blocks plus the complete recent window. Queries with at most 16 older blocks keep every older block, so this path is exact until more than 10,240 keys are available.

The selector reads the current Q, canonical Q8 K cache and existing key scales. It writes only a transient `[token, query_head, 17]` block list/count into the normal workspace. It does not change KV ownership, absolute positions, checkpoints, or persistent session format. Graph-cache identity includes the feature flag. Enable with `MIINFER_M31_Q8_GQA6_QUERY_BLOCK_SPARSE=1`.

At 65K context the Stage1 path visits up to 16 × 512 selected older keys plus the recent window (roughly 10K keys/query), with 16 sampled QK rows per older block for selection. This trades more work than fixed-anchor sparse attention for query-dependent recall. It is not exact attention and correctness must be judged on the coding task outcome.

## Static gates

- Mapping oracle: `tests/gqa6_query_block_sparse_oracle.cpp` compiled with host `g++` and passed its length/order/uniqueness checks for lengths 1 through 131,072.
- `git diff --check`: passed.
- Pinned build: `localhost/miinfer-dev:rocm-7.2.1`, image digest `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`.
- Built `miinfer-m31-fp16-full-model` and `miinfer` for `gfx906` successfully.
- Fresh HIP offload object SHA-256: `b42bb504e9562394e861807311ab01d995706d6a2e063345085056f2af205082`.
- Selector resources: 34 VGPR, 29 SGPR, 0 spills, 2,048 bytes LDS.
- Query-sparse Stage1 resources: 63 VGPR, 48 SGPR, 0 spills, 0 bytes LDS.

## GPU2 attempt — no valid inference result

Machinist GPU2 was the target (host `rocm-smi` index 2, BDF `0000:88:00.0`). The first container launch failed before allocation because `/dev/kfd` and `/dev/dri` were not passed through. A read-only `rocminfo` probe confirmed `ROCR_VISIBLE_DEVICES=2` sees `gfx906`; relaunching with both device mounts loaded the model and reported the server ready on port 8089.

The same EXP-0469 boundary-edit task was submitted using the exact stored context hash `98755b381bffd786553916aa419cdd06a8dd008cfd8710787364640ffb0e7c74` and copied exact checkpoint `sess_00d449dc68392a20_p66522.miinfer`. The request remained queued and produced no completed telemetry or tool response. At 15:48:53 +02:00, the owned candidate container exited with status 137; Podman reported `OOMKilled=false`. No stop command was issued by this run's thermal guard. The highest recorded GPU2 junction temperature was 67C, below the 85C hard-stop threshold. GPU2 returned to idle after exit.

The task runner JSON records `ConnectionRefusedError`, zero completed turns, and no source change. Therefore this run establishes neither prefix restore nor candidate correctness or performance; it is not a valid benchmark. The container log contains no MIInfer failure message. Kernel journal output also showed a `llama-server` general-protection fault in `libamdhip64.so` at 15:48:19, but there is no evidence that this caused the later container exit.

Raw logs, thermal samples, runner source/result, container inspect output, and checksums are in [`artifacts/EXP-0473-query-block-sparse/`](artifacts/EXP-0473-query-block-sparse/). The exact offload object remains on Machinist under `/tmp/MIInfer-query-block-sparse-build/`.

## Disposition

The implementation passes compile and static resource gates, but runtime acceptance is pending. Do not enable it by default or describe it as faster or correct. Before retrying GPU2, identify the cause of the unexpected status-137 termination; then verify exact checkpoint restoration and run the same tool-driven task to completion with route, latency, generated tokens, and tool/test outcome recorded.
