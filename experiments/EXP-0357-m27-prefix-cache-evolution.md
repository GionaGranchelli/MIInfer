# EXP-0357 — M27 sparse exact prefix cache

Status: KEEP — experimental, opt-in single-process serving.

## Hypothesis

Retaining a small, bounded set of exact GPU state checkpoints lets a coding
agent reuse shared history after branching, without unbounded VRAM growth.

## Baseline

Commit `abc4158` held one latest B512 checkpoint. In a real Pi trace, a ~15K
request advanced the sole checkpoint beyond the shared prefix; a later ~7.5K
branch had 6051 common tokens but reused zero of its 7566 prompt tokens. One
16K state snapshot occupies about 2.1 GB.

## Candidate

Keep sparse B512 checkpoints, at most eight and at most 3 GiB of state payload.
An exact-token radix trie resolves the longest reusable prefix. The selected
restore point is touched before capturing branch checkpoints, making it recent
for eviction. Checkpoint compatibility includes execution-contract version,
model/config, quantization, and runtime identity. Reset, mismatch, cancellation,
or generation failure clears the portfolio.

## Environment

- AMD Instinct MI50 / gfx906, 32 GiB VRAM
- Qwen3.8-27B-Q4_K_M; SHA256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- ROCm 7.1.52802; HIP Clang 20
- `m25_interactive`, context 16384, Pi CLI 0.85.1
- Model allocation: 19,479,343,508 bytes; device free after load:
  14,050,918,400 bytes
- State bytes below come from request telemetry. Long timings are diagnostic;
  the full run was not clock-qualified.

## Correctness

Current-binary append/replay output-token equality passed at 512, 640, 3991,
8192 and 16000 prompt tokens for one- and four-token generated seeds. The same
run passed older-branch selection and return (`older_prefix_reuse=2048`,
`return_branch_reuse=3584`) and invalidation after cancellation, injected
generation failure, mismatch, and explicit reset. CTest passed all 24 tests,
including 13 GPU-required tests.

Raw correctness output: `/tmp/m27-check-multicache-pinned-16k.log`.

## Real Pi results

Using one local server process with session reuse enabled and the temporary
isolated Pi profile:

| request | prompt tokens | reused | prefilled | entries / state bytes |
|---|---:|---:|---:|---:|
| fixture tool turn | 6514 | 0 | 6514 | 1 / 1,051,154,432 |
| tool-result continuation | 6589 | 6144 | 445 | 1 / 1,051,154,432 |
| ~15K attached-context turn | 15030 | 5632 | 9398 | 2 / 3,002,073,088 |
| ~15K continuation | 15106 | 14848 | 258 | 2 / 3,002,073,088 |
| next ~15K continuation | 15311 | 14848 | 463 | 2 / 3,002,073,088 |
| ~7.7K divergent attached-context turn | 7605 | 5632 | 1973 | 2 / 1,995,440,128 |
| ~7.7K tool-result continuation | 7681 | 7168 | 513 | 3 / 3,160,932,352 |

Each Pi turn called the local read tool and returned the fixture values
`MI50` and `amber-cedar`. Raw server telemetry:
`/tmp/m27-pi-multicache-pinned-server.log`; temporary Pi session records are
under `/tmp/miinfer-pi-agent-m27-sessions/`.

Before the restore-point touch was added, the ~15K branch evicted its 6144
checkpoint, and the later ~7.5K branch replayed all 7566 tokens. After the
change, the long request retained the 6144 checkpoint alongside its latest
state under the 3 GiB cap; the divergent branch turn reused 5632 tokens, and
its tool-result continuation reused 7168.

## Decision

KEEP for opt-in single-process serving. Exact branch reuse is demonstrated;
request telemetry proves avoided prefill work, not a qualified latency win.
Process restart/resume, concurrent sessions, and broad coding-agent workloads
remain unqualified. The M27 cold-P512 stop remains as recorded in EXP-0356;
M26 decode work remains paused.
