# LC-0009 QSUM decode reference handoff

Status: **OPEN**. This work did not produce a decode/QSUM reference, and LC-0009 remains **BLOCKED** under the registered qualification contract.

## Evidence checked

- Source branch starts at `a5140f8d2ce48a6aa0c783f0da2e0166065d1043` (`experiment/m31-lc-0009-quantized-correctness`). The frozen corpus and its manifest were not changed.
- `experiments/m31-lc-0009/gateup-prefill-ordinary512-position-sweep.json` reports a pass for three aligned-MMQ **prefill** positions. Its environment has `MIINFER_EXPERIMENTAL_MMQ_GATEUP_QSUM=0`; its limitations explicitly exclude decode/QSUM.
- `gpu-mmq-qsum-only-ordinary64-invalid.json` and its comparison are explicitly `INVALID_CONFIGURATION_NOT_COUNTED`: QSUM was set without `MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY=1`, so the default fused route ran.
- The older `results/m31-lc-0007/z840-gateup-graph-trace/mmq_only-pair-1.record.json` is not a substitute: it requests `MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY=1` but does not set QSUM, uses HIP graph mode, and identifies source `8a9caec3366512e459a93bad68adde181225304b` from before the selector was introduced.
- The committed 24-cell matrix has repeatable logits, IDs, and state captures, but the committed tree contains no raw one-token decode QSUM operand capture suitable for the Contract O scalar reference.
- The registered CPU oracle on Machinist uses llama.cpp `d1d3c3396aa13a5f239109a822666c4870490ad5` and model SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`. This is a full-model reference; it does not replace the missing same-input QSUM operation capture.

## Why it remains open

Contract O requires the actual represented operands and GPU result for the selected operation. The committed evidence still has no one-token decode QSUM capture, and the invalid QSUM-only matrix cell cannot be relabeled or reused as MMQ evidence. The eager `PrefillV2RecurrentLayer::decode` path already has a one-shot trace hook for the layer input, post-normalized Gate/Up input, captured Q8 blocks, and both projection outputs; this handoff did not run it. The scalar checker now supports QSUM captures with `--qsum`.

The initial two-attempt Machinist preflight found its checkout at `d8b833637f85537d6205c9480129d95c819e6d61`, branch `m30/exact-prefix-kv-reuse`, with dirty tracked edits in `CMakeLists.txt`, `include/miinfer/prefill_v2/model.hpp`, `src/prefill_v2/model.cpp`, and `tools/miinfer_cli.cpp`, plus untracked M30 files. It was left untouched. No active LC-0009 process or tmux session was visible. The remote login shell does not provide `rtk`; the first read-only command therefore failed to run its `rtk` subcommands. At that initial inspection, no clean isolated worktree was prepared or run. It did not establish that the remote model/oracle files are absent.

GPU work was out of scope for this CPU-only handoff, and the campaign report currently pauses sustained Z840 GPU work after its thermal warning. Therefore no missing decode capture could be generated here. Do not close LC-0009 based on this note.

## Machinist host-only checker verification (2026-10-12)

A clean detached worktree at checker source `c00136d00c4e5a00c4b5460a1e31f793d4fd435e` was created at `/home/machinist/Development/MIInfer-lc0009-qsum-checker`. The dirty `/home/machinist/Development/MIInfer` checkout was not touched, and no GPU device or GPU operation was used. The standalone checker is not registered as a CMake target, so it was compiled directly with the host compiler:

```text
cd /home/machinist/Development/MIInfer-lc0009-qsum-checker
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -I include tools/m31_lc9_gateup_contract_o.cpp src/gguf.cpp -o build/lc0009-host-check/m31-lc9-gateup-contract-o
build/lc0009-host-check/m31-lc9-gateup-contract-o --self-check
```

The command exited 0 and printed `sum_mode_self_check=PASS`. Checker binary SHA-256: `04a8da582cb50978c850ee3e2dab4032b272b1fbe8c6c846695bf2f7abd7c08c`; retained build/self-check log at `build/lc0009-host-check/build-self-check.log`, SHA-256 `c8a0ffafd5a16be15613106d77e98dc98111b4ab8b347ec737a1f55b1d450192`. This verifies compilation and the checker's small ordinary-versus-QSUM sum-mode self-check only. It does not execute the scalar projection reference, validate a raw QSUM capture, or change the OPEN/BLOCKED disposition.

## Machinist ROCm target build without GPU access (2026-10-12 07:50)

The missing capture's production-path executable and operation-reference target were built on Machinist from clean source `c00136d00c4e5a00c4b5460a1e31f793d4fd435e` in the pinned ROCm 7.2.1 image configuration. A first container preflight could not read the linked worktree's external `.git` path; it was preserved as a failed attempt. The successful build used a separate standalone clone at `/home/machinist/Development/MIInfer-lc0009-qsum-build-src` and an external build directory at `/home/machinist/experiments/m31-lc-0009-qsum/build-c00136d-rocm721-gfx906`.

Build options: Release, HIP enabled, `gfx906`, tests off, benchmarks on, explicit `/opt/rocm-7.2.1` CMake prefix. The container used `--network=none`, read-only source/dependency mounts, and asserted `/dev/kfd` and `/dev/dri` were absent. Build provenance says `c00136d00c4e`, `dirty=false`; exit 0. Retained build log is `/home/machinist/experiments/m31-lc-0009-qsum/build-c00136d-rocm721-gfx906.log`, SHA-256 `bb542960cefdd7e905c4e2693ae47cc1af9362111ad2f44d3241bd6a98a05526`. `e2-0003-checkpointed-call` SHA-256 `1046c8d5adee18e992d703c7fd5a1506f4a29493f2d830bbf26a3d624aad7410`; `e2-0003-gateup-operation-reference` SHA-256 `31f4ac45d95c5367ef8a374d279efe35776979a3fb6f69417bbe667457bb0f3e`; generated build header SHA-256 `09524dd9b0cc8c7c6916283281182703e6186d73e3ed5b799d01dc404e672340`.

The current Machinist model file hashes to the pinned SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`; frozen `ordinary-64-01.u32le` hashes to `c5c22b5353991655d59e599a074875541dbd8a51bb43a92098be2fdd0ccdc480`. The existing host checker self-check passes (`sum_mode_self_check=PASS`; checker SHA-256 `04a8da582cb50978c850ee3e2dab4032b272b1fbe8c6c846695bf2f7abd7c08c`).

This is compile and input-identity evidence only; no GPU operation or QSUM capture ran. Machinist reports GPU[0] as gfx1200 at `0000:04:00.0` and GPU[1] as gfx906 at `0000:85:00.0`; the `ROCR_VISIBLE_DEVICES=0` command below is Z840-specific and must not be reused there. Its container image ID matches the Z840 image ID `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`, but Podman's `.Digest` differs (`sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a` on Machinist versus `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e` on Z840). The 4 KiB copy probe does not qualify Machinist for production inference. No run is cleared until the applicable passive-heatsink airflow and exact device selection are confirmed.

## Required next run

Use the existing eager decode trace hook; no new capture code is needed. Set `MIINFER_LC_OP_TRACE_PREFIX` to a new Lake path and enable `MIINFER_LC_DECODE_TRACE=1`, with `MIINFER_LC_TRACE_LAYER=0`. It writes the exact post-normalized floats, Q8 codes/scales/sum, Gate and Up outputs, and decode position into separate files, refusing to overwrite them. After GPU operation is cleared to resume, execute the frozen `ordinary-64-01` prompt with the MMQ route and QSUM both selected, eager mode, and at least two generated tokens so the capture includes a one-token decode invocation. The required selectors are:

```text
ROCR_VISIBLE_DEVICES=0
MIINFER_HIP_GRAPH=0
MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY=1
MIINFER_EXPERIMENTAL_MMQ_GATEUP_QSUM=1
MIINFER_MX_Q8_BATCH=0
```

Use the pinned ROCm 7.2.1 image, pinned model and frozen prompt IDs. Run the scalar checker on the captured represented operands and pinned GGUF Q4_K weights; require exact quantizer representation, finite outputs, and a derived FP32 forward-error bound for every checked Gate/Up row. Preserve the command, environment, source/binary/model/image identities, raw files, exit status, and SHA-256 inventory. This operation result alone would close only the pending QSUM decode reference; it would not unblock LC-0009's broader Contract O, Contract X, or qualification decision.
