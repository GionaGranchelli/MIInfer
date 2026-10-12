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

The two-attempt Machinist preflight found its checkout at `d8b833637f85537d6205c9480129d95c819e6d61`, branch `m30/exact-prefix-kv-reuse`, with dirty tracked edits in `CMakeLists.txt`, `include/miinfer/prefill_v2/model.hpp`, `src/prefill_v2/model.cpp`, and `tools/miinfer_cli.cpp`, plus untracked M30 files. It was left untouched. No active LC-0009 process or tmux session was visible. The remote login shell does not provide `rtk`; the first read-only command therefore failed to run its `rtk` subcommands. No third SSH attempt was made. This inspection did not establish that the remote model/oracle files are absent; it established that no clean isolated worktree was prepared or run during this handoff.

GPU work was out of scope for this CPU-only handoff, and the campaign report currently pauses sustained Z840 GPU work after its thermal warning. Therefore no missing decode capture could be generated here. Do not close LC-0009 based on this note.

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
