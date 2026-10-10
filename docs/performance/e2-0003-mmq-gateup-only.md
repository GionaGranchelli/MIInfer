# E2-0003 — MMQ-only Gate/Up candidate

## Phase A checkpoint

**Phase A status: FEASIBLE; GPU preflight unavailable.** At this checkpoint,
source behavior had not changed.

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

## Phase B checkpoint

**Status: IMPLEMENTED; offline checks pass; GPU qualification blocked by host
unavailability.** Candidate implementation commit `fef7b6637d57e664a18c528245f0d558382842da`
and parser test commit `c2fd12bbd24b3ae75f34f6de2401fb00b0df785e` descend from
planning commit `dd35dab`. The in-place checkout still carries the preserved
M31 dirty overlay. No campaign GPU work started.

- Added default-off `MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY` parsing. The model
  reads it once and passes an immutable bool through 16 blocks to 48 recurrent
  and 16 attention layers. Invalid values fail model construction.
- In the opt-in route, both layer constructors skip fused Gate/Up packing and
  allocation. MMQ weights and workspace stay in place. Eager and profiled
  single-token decode use the existing MMQ quantizer, Gate and Up projections,
  and SwiGLU kernel when the fused pointer is null. Prefill code is unchanged.
- The default route retains a non-null fused pointer and its existing decode
  call. Model construction precedes graph capture, so the selected weight
  pointers remain fixed for a model's lifetime. Runtime graph replay is not
  verified.
- Structural request reduction remains the exact fused allocation-request
  upper bound: `7,130,316,800 B`; MMQ Gate/Up requests (`6,452,936,704 B`)
  remain. No external GPU-memory saving has been measured.
- Option parsing test passed with `-O2 -DNDEBUG`; `git diff --check` passed.
  HIP syntax-only checks passed for edited layer/model sources. Full CMake
  configuration failed before generation because ROCm 6.2 `lld` cannot load
  the host's missing `libxml2.so.2`.
- `graphify update .` completed in an isolated clean worktree at the candidate
  commit (11,207 nodes, 15,646 edges); existing generated graph files in the
  user's dirty checkout were preserved.
- Hash manifest: `results/e2-0003/source-manifest.json`. It records relevant
  source SHA-256 values, historical model/prompt identity with provenance,
  parser-test binary SHA-256, build flags, and explicitly absent runtime/GPU
  artifacts. The failed CMake output and parser-test exit codes are preserved
  under `results/e2-0003/`.
- Repeated bounded SSH preflight to Z840 `192.168.68.54` timed out. The local
  P620 has no initialized AMD driver. No control or candidate binary was
  built, no GPU experiment ran, and no memory, parity, throughput, or graph
  verdict exists.
- Constructor destructors release owned pointers after successful
  construction. Existing raw-pointer constructors do not provide RAII cleanup
  if an unrelated later allocation throws; this pre-existing limitation was
  not expanded into this campaign.
- Candidate A disposition: `BLOCKED / PARTIAL`, not KEEP or REJECT. Historical
  older-source fast-path measurements make decode throughput the principal
  risk but do not substitute for the required same-source gate.
- Candidate B: not triggered. Its only trigger is a measured Candidate A
  memory success with decode-throughput failure after correctness, graph, and
  thermal gates pass.
- Next authorized phase: wait for safe Z840 availability, then complete fresh
  identity/idle/model/telemetry preflight before building or running paired
  tests. If the host remains unavailable, finalize as blocked without using
  the unqualified Machinist.

## Interim campaign disposition

**M31-OVERNIGHT-01 checkpoint: PARTIAL / BLOCKED.** The implementation and
offline checks are complete. Fifty-six bounded Z840 SSH preflights timed out by
this checkpoint; no safe GPU experiment could begin. There are no
control or candidate runs and Candidate B was not triggered. The campaign
window remains open through the recorded hard stop below.

### Same-source harness checkpoint — 2026-10-09 23:16 UTC

- Added a guarded runner that uses one benchmark binary for both routes and
  changes only `MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY` between control and
  candidate. It rechecks the model and binary hashes before each call, records
  the requested environment, preserves per-call metrics and one-second GPU
  telemetry, and refuses to overwrite its output directory.
- The four focused harness tests pass, including the historical prompt
  fingerprint and the route selector check. Python compilation and
  `git diff --check` pass. Raw test output and the latest bounded SSH failure
  are preserved in `results/e2-0003/`.
- Review tightened both per-call and cooldown preflight to require an
  identifiable, idle GPU below 80°C; a fifth focused test covers rejection at
  the 80°C warning threshold.
- Following repository instructions, `graphify update .` ran in a detached
  worktree at `b461a2514726b36bf5b65ebf9e69aec2e08fb92c`; it produced 11,243
  nodes and 15,723 links. The user's dirty `graphify-out/` in the main checkout
  was preserved.
- The SSH retry at `2026-10-09T23:12:57Z` timed out after eight seconds.
  Host identity, model hash, image digest, and device state remain unverified;
  no GPU call was started. The harness is prepared but unqualified.
- A sixth bounded readiness attempt at `2026-10-09T23:15:50Z` also timed out;
  its raw output is `results/e2-0003/ssh-preflight-20261009-2315.log`.
- The local ROCm linker dependency check found only `libxml2.so.16` on the
  P620, not the requested `libxml2.so.2`, in the checked system and ROCm
  library directories. No compatible library was available to reuse; no
  soname alias or dependency change was made. Raw findings are in
  `results/e2-0003/libxml2-soname-check.txt`.
- A seventh bounded readiness attempt at `2026-10-09T23:17:59Z` timed out;
  its raw output is `results/e2-0003/ssh-preflight-20261009-2317.log`.
- An eighth bounded readiness attempt at `2026-10-09T23:19:20Z` timed out;
  its raw output is `results/e2-0003/ssh-preflight-20261009-2319.log`.
- A ninth bounded readiness attempt at `2026-10-09T23:21:16Z` timed out;
  its raw output is `results/e2-0003/ssh-preflight-20261009-2321.log`.
- A tenth bounded readiness attempt at `2026-10-09T23:23:36Z` timed out;
  its raw output is `results/e2-0003/ssh-preflight-20261009-2323.log`.
- The eleventh and twelfth readiness attempts at `2026-10-09T23:25:06Z` and
  `2026-10-09T23:27:22Z` timed out; raw logs are preserved in
  `results/e2-0003/ssh-preflight-20261009-2325.log` and
  `results/e2-0003/ssh-preflight-20261009-2327.log`.
- The thirteenth and fourteenth attempts at `2026-10-09T23:29:45Z` and
  `2026-10-09T23:31:58Z` also timed out; see their timestamped SSH preflight
  logs in `results/e2-0003/`.
- The fifteenth and sixteenth attempts at `2026-10-09T23:34:30Z` and
  `2026-10-09T23:37:45Z` timed out as well; both raw logs are preserved.
- The seventeenth and eighteenth readiness checks at `2026-10-09T23:39:25Z`
  and `2026-10-09T23:41:39Z` also timed out; raw logs are preserved.
- The nineteenth and twentieth checks at `2026-10-09T23:44:01Z` and
  `2026-10-09T23:48:21Z` timed out; raw logs are preserved.
- The twenty-first and twenty-second checks at `2026-10-09T23:53:00Z` and
  `2026-10-09T23:57:16Z` also timed out; raw logs are preserved.
- The twenty-third check at `2026-10-10T00:02:53Z` timed out; see
  `results/e2-0003/ssh-preflight-20261010-0002.log`.
- The twenty-fourth readiness check at `2026-10-10T00:08:24Z` timed out; see
  `results/e2-0003/ssh-preflight-20261010-0007.log`.
- The twenty-fifth and twenty-sixth checks at `2026-10-10T00:14:04Z` and
  `2026-10-10T00:18:24Z` also timed out; raw logs are preserved.
- The twenty-seventh and twenty-eighth checks at `2026-10-10T00:24:02Z` and
  `2026-10-10T00:28:19Z` timed out; raw logs are preserved.
- The twenty-ninth and thirtieth checks at `2026-10-10T00:32:55Z` and
  `2026-10-10T00:37:14Z` also timed out; raw logs are preserved.
- The thirty-first and thirty-second checks at `2026-10-10T00:41:48Z` and
  `2026-10-10T00:47:07Z` timed out; raw logs are preserved.
- The thirty-third and thirty-fourth checks at `2026-10-10T00:53:45Z` and
  `2026-10-10T00:59:04Z` timed out; raw logs are preserved.
- The thirty-fifth and thirty-sixth checks at `2026-10-10T01:04:43Z` and
  `2026-10-10T01:10:00Z` also timed out; raw logs are preserved.
- The thirty-seventh and thirty-eighth checks at `2026-10-10T01:15:37Z` and
  `2026-10-10T01:20:57Z` timed out; raw logs are preserved.
- The thirty-ninth and fortieth checks at `2026-10-10T01:26:33Z` and
  `2026-10-10T01:31:51Z` timed out; raw logs are preserved.
- The forty-first and forty-second checks at `2026-10-10T01:37:24Z` and
  `2026-10-10T01:42:41Z` also timed out; raw logs are preserved.
- The forty-third and forty-fourth checks at `2026-10-10T01:53:21Z` and
  `2026-10-10T02:03:49Z` timed out; raw logs are preserved.
- The forty-fifth and forty-sixth checks at `2026-10-10T02:14:38Z` and
  `2026-10-10T02:25:05Z` also timed out; raw logs are preserved.
- The forty-seventh and forty-eighth checks at `2026-10-10T02:35:50Z` and
  `2026-10-10T02:46:16Z` timed out; raw logs are preserved.
- The forty-ninth and fiftieth checks at `2026-10-10T02:56:56Z` and
  `2026-10-10T03:07:21Z` also timed out; raw logs are preserved.
- The fifty-first and fifty-second checks at `2026-10-10T03:18:03Z` and
  `2026-10-10T03:28:29Z` timed out; raw logs are preserved.
- The fifty-third and fifty-fourth checks at `2026-10-10T03:39:12Z` and
  `2026-10-10T03:49:51Z` timed out; raw logs are preserved.
- The fifty-fifth and fifty-sixth checks at `2026-10-10T04:01:40Z` and
  `2026-10-10T04:12:29Z` timed out; raw logs are preserved.
- The fifty-seventh check at `2026-10-10T04:22:47Z` timed out; the raw log is
  preserved.
- The fifty-eighth check at `2026-10-10T04:31:59Z` timed out; the raw log is
  preserved.
- The fifty-ninth check at `2026-10-10T04:47:26Z` timed out; the raw log is
  preserved.
- The sixtieth check at `2026-10-10T04:57:43Z` timed out; the raw log is
  preserved.
- The sixty-first check at `2026-10-10T05:06:56Z` timed out; the raw log is
  preserved.
- The sixty-second check at `2026-10-10T05:17:23Z` timed out; the raw log is
  preserved.
- The sixty-third check at `2026-10-10T05:28:06Z` timed out; the raw log is
  preserved.
- The sixty-fourth check at `2026-10-10T05:38:27Z` timed out; the raw log is
  preserved.
- The sixty-fifth check at `2026-10-10T05:49:57Z` timed out; the raw log is
  preserved.
- The sixty-sixth check at `2026-10-10T06:01:24Z` timed out; the raw log is
  preserved.
- The sixty-seventh check at `2026-10-10T06:11:19Z` timed out; the raw log is
  preserved.
- The sixty-eighth check at `2026-10-10T06:21:56Z` timed out; the raw log is
  preserved.
- The sixty-ninth check at `2026-10-10T06:32:29Z` timed out; the raw log is
  preserved.
- The seventieth check at `2026-10-10T06:42:59Z` timed out; the raw log is
  preserved.
- The seventy-first and final check at `2026-10-10T06:50:14Z` timed out; the
  raw log is preserved. Campaign ended at the authorized hard stop with no
  successful host preflight and zero GPU runs.

```text
START_TIME: 2026-10-09 22:50:25 UTC
CHECKPOINT_TIME: 2026-10-10 06:50:14 UTC
ELAPSED_AT_CHECKPOINT: 07:59:49
AUTHORIZED_WINDOW_END: 2026-10-10 06:50:25 UTC
BASE_SHA: 99d40748bb4ae8921fbf3558b9f0216389ebfd8a plus preserved M31 overlay
CAMPAIGN_SOURCE_SHA: c2fd12bbd24b3ae75f34f6de2401fb00b0df785e (report/evidence commits follow)
EXPERIMENT_BRANCHES: experiment/e2-0003-mmq-gateup-only
DEFAULT_PATH_UNCHANGED: source path retained; runtime not verified

CANDIDATE_A_IMPLEMENTED: YES, default off
BUILD_STATUS: HIP syntax-only PASS; full CMake/build BLOCKED by missing libxml2.so.2
CORRECTNESS: NOT_RUN (GPU route); parser test PASS
HIP_GRAPH: NOT_RUN
PREFILL_TOK_S: NOT_RUN
DECODE_TOK_S: NOT_RUN
SAMPLED_VRAM_SAVINGS: UNKNOWN
ALLOCATION_BYTES_SAVED: 7,130,316,800 B maximum fused request removal (static)
THERMAL_STATUS: no campaign GPU workload; guard self-test PASS
DECISION: BLOCKED / PARTIAL; no KEEP or REJECT verdict

CANDIDATE_B_TRIGGERED: NO
CANDIDATE_B_DECISION: NOT_RUN

E3_HARNESS: inspected and historical artifacts/schema validated; same-source runner added and offline-tested, no GPU validation
CONTROL_RUNS: 0
CANDIDATE_RUNS: 0
NUMERICAL_REGRESSION: NOT_RUN
PERSISTENT_CONTEXT: NOT_RUN
SNAPSHOT_ROLLBACK: NOT_RUN
TESTS_PASSED: option parser; gfx906 HIP syntax-only; same-source harness (4 tests); diff check; existing thermal-guard self-test
TESTS_FAILED: full CMake HIP compiler detection (host libxml2.so.2 unavailable)
TESTS_SKIPPED: operation reference, allocation/cleanup runtime, graph, A/B, sessions, memory, performance
KNOWN_LIMITATIONS: Z840 unreachable; constructor raw-pointer exception cleanup is pre-existing and not RAII
```

### Same-source scorecard

| Metric | Fused control | MMQ-only |
|---|---:|---:|
| Weight allocation bytes | NOT_RUN | NOT_RUN; static max request reduction 7,130,316,800 B |
| Sampled device memory | NOT_RUN | NOT_RUN |
| Memory saving | NOT_RUN | UNKNOWN |
| Prefill tok/s | NOT_RUN | NOT_RUN |
| Decode tok/s | NOT_RUN | NOT_RUN |
| Prefill change | NOT_RUN | NOT_RUN |
| Decode change | NOT_RUN | NOT_RUN |
| Eight-token parity | NOT_RUN | NOT_RUN |
| Numerical checks | NOT_RUN | Parser only; route reference not run |
| HIP Graph | NOT_RUN | NOT_RUN |
| Persistent-session checks | NOT_RUN | NOT_RUN |
| Snapshot/rollback checks | NOT_RUN | NOT_RUN |
| Maximum junction temperature | NOT_RUN | NOT_RUN |
| Decision | UNMEASURED | BLOCKED / PARTIAL |

### Interim conclusion and next goal

- Best candidate: none established. MMQ-only remains a default-off experiment.
- Physical memory recovered: unknown. Allocation requests are not residency.
- Performance retained: unknown. Older-source fast-path data supports treating
  decode speed as a risk only; it cannot substitute for this source's A/B.
- Correctness: unqualified. No model token or graph result was produced.
- Tracking: issue #7 and #9 received Phase B updates; #6 and PR #10 were not
  changed. No push or merge occurred. Dirty user M31 and graphify files remain.
- **Next goal:** continue offline evidence preparation and recheck Z840
  connectivity within the authorized window; after reachability, run a fresh
  safe preflight, then build same-source control/candidate binaries in the pinned runtime.
  Acceptance remains: sampled VRAM reduction >=6.0 GB, prefill >=95% and decode
  >=85% of control and above 20.56 tok/s, eight-token parity, operation checks,
  graph replay, and thermal guard all pass.
