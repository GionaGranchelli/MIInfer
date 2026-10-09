# E2-0003 — MMQ-only Gate/Up candidate

## Phase A checkpoint

**Status: FEASIBLE; GPU preflight unavailable at checkpoint.** No source code or
runtime behavior has changed in this campaign yet.

### Source and baseline identity

- Campaign start: `2026-10-09 22:50:25 UTC`; stop initiating experiments by
  `2026-10-10 06:50:25 UTC`.
- Candidate branch: `experiment/e2-0003-mmq-gateup-only`, created at current
  local M31 commit `99d40748bb4ae8921fbf3558b9f0216389ebfd8a`.
- The branch preserves the pre-existing dirty M31 source, docs, generated
  graph, and untracked evidence. That overlay remains uncommitted and is not
  part of this checkpoint's campaign change.
- E2-0002's source report identifies the actual code base as `99d4074` plus its
  staged M31 overlay. E3-0001 ran a separate source snapshot at
  `11228c0cab8d43267687d92d167373185f671d91`; its data is historical context,
  not a same-source control for this experiment.
- The E3-0001 two-pair raw records, corrected greedy IDs, and benchmark JSON
  are present in `/tmp/m31-e1-q1-gdn-gate`. JSON Schema validation passed, and
  the existing E3 unit tests passed (6 harness tests, 2 pair-merge tests).
- The existing thermal-guard self-test passed. It injected an 85°C sample,
  verified bounded worker-group termination, and confirmed stale/unavailable
  telemetry fails closed; campaign thresholds remain 80°C warning / 85°C stop.
- Historical model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`;
  prompt fingerprint `bce3932a8fc5d121`. Live remote verification remains
  required before any GPU run.
- The local ThinkStation P620 has no initialized AMD driver. SSH to the Z840
  at `192.168.68.54` timed out on a bounded retry. No GPU workload started.

### Implementation plan and feasibility

The model has 48 recurrent and 16 attention layers. Both layer types already
load separate MMQ Gate/Up weights for prefill and a fused Wave Gate/Up layout
for decode. The current MMQ launcher accepts `token_count=1` and is already
used for single-token recurrent projections. Shared workspace already has the
Q8 MMQ blocks and Gate, Up, and activation buffers. The candidate therefore
needs no new HIP kernel, conversion buffer, or host round trip.

1. Sample `MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY` once when constructing the
   model and carry that immutable choice through all 16 topology blocks and 64
   layers.
2. When enabled, omit both fused Gate/Up host packing and device allocation in
   recurrent and attention layers. Keep their existing MMQ weights and use
   Q8 quantization, the existing single-token MMQ projection calls, and the
   existing SwiGLU activation in eager and graph decode paths.
3. Leave the default route and prefill route unchanged. Verify cleanup and
   stable graph pointers for both modes.
4. Add a focused operation-level MMQ-versus-fused test and route tests. Build
   two same-source binaries/configurations and adapt the existing E3 guard and
   telemetry contract to compare default control against candidate without
   labeling either as llama.cpp.

The exact fused allocation-request upper bound is `7,130,316,800 B`; MMQ
Gate/Up requests of `6,452,936,704 B` remain. These are allocation requests,
not promised VRAM savings. The principal risk is decode throughput: the current
fused projection/activation kernel becomes separate MMQ projection and
activation launches.

### Phase result

- Source/baseline review: **PASS**, with preserved dirty overlay and current
  source rooted at `99d4074`.
- E3 artifact/schema and focused harness checks: **PASS**.
- MMQ single-token/workspace/graph feasibility: **PASS by source inspection**;
  GPU graph replay is unverified.
- Live Z840/model/toolchain/GPU availability: **NOT VERIFIED** due SSH timeout.
- Next authorized phase: implement the default-off MMQ-only candidate and
  offline correctness checks. Resume guarded GPU work only after live Z840
  identity, idle state, model SHA, and thermal telemetry pass preflight.
