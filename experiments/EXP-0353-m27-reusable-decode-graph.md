# EXP-0353 — M27 reusable device-state decode graph

## Token observer handoff microbenchmark

Rebuild the diagnostic and complete Release test suite with:

```bash
cmake -S . -B build-observer -DCMAKE_BUILD_TYPE=Release -DMIINFER_ENABLE_HIP=ON
cmake --build build-observer -j 8
ctest --test-dir build-observer --output-on-failure
```

Result: build PASS; CTest 24/24 PASS (11 host-only and 13 GPU-required).

The M27 observer comparison was measured in isolation with the same one-node
state-advance graph and per-token host synchronization for both output paths:

```bash
build-observer/miinfer-qwen3-token-observer-bench --tokens 128 --iterations 20
```

On the MI50/gfx906, 20 interleaved measured samples after three warmups gave:

```text
async D2H to pinned ring: 2795.517 us / 128 = 21.840 us/token
mapped host-visible ring: 2322.797 us / 128 = 18.147 us/token
```

Every iteration verified all observed values and the final device generated
count. This is a microbenchmark of graph launch, state-advance, observer
handoff, and synchronization; it is not full-model decode or streaming
performance. In this narrow setup mapped memory was 17% lower in median
elapsed time. End-to-end observer behavior still needs measurement in the
production decode engine before selecting a streaming strategy.

## Attention split-transition correctness

`--check-graph-state-context N` repeats the supplied prompt's token IDs to
construct an exact synthetic token prefix of length `N`, then compares a
graph decode with direct stepping. The tests use six generated tokens so the
dynamic attention length crosses each selected split boundary. Both token IDs
and all recurrent state, convolution history, and active K/V bytes matched:

| Starting prompt length | Attention lengths exercised | State bytes compared | Result |
| ---: | --- | ---: | --- |
| 508 | 508–513 (4→8 splits at 513) | 192,479,232 | PASS |
| 2,044 | 2,044–2,049 (8→16 splits at 2,049) | 293,142,528 | PASS |
| 8,188 | 8,188–8,193 (16→32 splits at 8,193) | 695,795,712 | PASS |

Reproduce with the real model artifact:

```bash
build-observer/miinfer run ~/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompt 'The AMD Instinct MI50 is a high-performance GPU featuring' \
  --context 520 --max-tokens 6 --check-graph-state \
  --check-graph-state-context 508 --no-stream

build-observer/miinfer run ~/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompt 'The AMD Instinct MI50 is a high-performance GPU featuring' \
  --context 2056 --max-tokens 6 --check-graph-state \
  --check-graph-state-context 2044 --no-stream

MIINFER_PRESET=m25_hi_qualified build-observer/miinfer run \
  ~/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompt 'The AMD Instinct MI50 is a high-performance GPU featuring' \
  --context 8196 --max-tokens 6 --check-graph-state \
  --check-graph-state-context 8188 --no-stream
```

The first two use the default prefill route; the 8K case uses the existing
qualified wide-prefill preset to avoid repeating the very slow token-at-a-time
prefill. These are correctness checks, not performance comparisons. The
16K-attention transition, full 128K context envelope, and M27 sustained
performance gate remain open.

## Hypothesis

One captured decode graph can be queued repeatedly while device-resident token
and position state advances between graph launches, preserving the existing
single-token output and removing per-position graph capture/selection.

## Implementation

`Qwen35RuntimeEngine` now owns one graph and one aligned `DeviceDecodeState`.
Dynamic position is consumed by token embedding, recurrent convolution history,
attention Q RoPE, fused K RoPE/KV append, and attention causal length. Attention
uses a fixed maximum captured grid; active split count is selected from the
device position so short contexts retain the existing split policy. Argmax
writes the next token back to device state, then a small state-advance kernel
updates position/generated/stop and appends the token to a device output ring.
The host uploads initial state once, queues the same graph, and reads the token
ring once. Streaming/cancellation remains on the direct synchronized path.
The reusable graph explicitly rejects non-fused RoPE, non-FP16 KV, or
non-tiled-attention configurations; those remain available through the direct
path with `MIINFER_HIP_GRAPH=0`.

## Correctness smoke

Artifact: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`.

Graph run:

```bash
MIINFER_DUMP_TOKENS=1 build/mi50-release/miinfer run \
  ~/models/Qwen3.8-27B-Q4_K_M.gguf --prompt 'Say hello.' \
  --max-tokens 4 --no-stream --context 64
```

Direct-step control used the same command without `--no-stream`.

| Route | Token IDs | Result |
| --- | --- | --- |
| Reusable graph | `271 9419 0 2500` | PASS |
| Direct synchronized step | `271 9419 0 2500` | PASS |

The graph contract reported four launches, one 64-byte initial state upload,
and one 16-byte output readback. There is no per-token H2D token feedback.

## Dynamic-position screen

```bash
build/mi50-release/miinfer run \
  ~/models/Qwen3.8-27B-Q4_K_M.gguf --context 16384 \
  --decode-curve --curve-context 512 --curve-iterations 1 \
  --no-stream --max-tokens 128
```

The reusable graph completed all 128 tokens across positions 512–639. The
single-run result was 4108.09 ms total / 32.0945 ms per token. Hardware clocks
and other state were not recorded for this run, so this is functional
smoke-test evidence only and must not be interpreted as a performance win or
qualification against the 32.2948 ms/token M26 baseline.

## 128-token graph/direct continuation

On the same 13-token prompt and 1024-token cache capacity, graph and direct
routes both generated 128 tokens with identical token IDs. The graph reported
31.17 tok/s (32.082 ms/token); direct stepping reported 30.75 tok/s
(32.525 ms/token). These are single diagnostic runs without matching
continuous telemetry, so the small apparent advantage is not a performance
claim. This extends end-to-end recurrent/KV parity evidence to 128 consecutive
decode updates.

The CLI now also provides a bytewise state check for the same full-window
comparison:

```bash
MIINFER_PRESET=m25_hi_qualified build/mi50-release/miinfer run \
  ~/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompt 'The AMD Instinct MI50 is a high-performance GPU featuring' \
  --context 1024 --max-tokens 128 --check-graph-state
```

Result: `graph_state_check=PASS tokens=128 positions=140
state_bytes=168034304`. The comparison includes every recurrent FP32 state,
recurrent convolution history, and active attention K/V byte through position
139. Thus both token parity and exact post-decode persistent-buffer parity
pass across 127 repeated launches; the prompt's first generated token comes
from prefill. This diagnostic rejects early-stop windows because the graph
currently queues all requested launches after EOS.

## Repeated P512 screen

A five-iteration P512/TG128 curve on the same build reported a median of
4136.77 ms total / 32.3185 ms per token. In-load spot checks showed
SCLK/MCLK 1606/1000 MHz, 223 W against the 225 W cap, and 94°C junction.
Clocks remained at the selected levels in those samples, but the high thermal
and power readings and absence of continuous hardware logging mean this is a
screen, not formal performance qualification. It is within 0.1% of the
32.2948 ms/token M26 baseline; no performance win is established.

## Verification

- Release build of `miinfer`: PASS.
- `mi50-release` CTest on the final dynamic-split tree: 24/24 passed.
- Short graph/direct exact token match: PASS.
- 128-token graph/direct stream: exact token-ID parity PASS; timing samples are
  diagnostic only.
- 128-token graph/direct recurrent/history/active-KV byte comparison:
  PASS, 168,034,304 bytes identical.
- P512/TG128 128-token completion: PASS; exact graph/direct token-stream
  parity is now checked, but no bytewise recurrent/KV snapshot comparison is
  available.
- Session restore/append token parity: PASS for 1- and 4-token graph seeds at
  the 512-token checkpoint.
- Long EOS screen: exact 27-token parity PASS; graph runs to queued window
  limit after EOS (62 launches), so early termination is not implemented.
- Five-iteration P512 timing screen: 32.3185 ms/token; no material delta from
  M26 baseline, hardware telemetry was spot-sampled.
- Follow-up five-iteration P512 curve: 32.2954 ms/token median, but invalid
  for qualification due to observed SCLK reduction and over-cap power samples;
  raw capture is in the ignored local `bench/results/m27-20260919/` directory.
- P2048/TG128 dynamic attention split transition (4→8): completed 128 tokens
  at 33.4689 ms/token in one run; functional-only, no matching direct-route or
  full hardware-state comparison.

## Session append / EOS overrun check

The existing `--check-session` diagnostic runs its seed and appended request
through the reusable graph, then compares the appended token sequence with a
fresh full-prefill replay using direct synchronized decoding. The four-token
seed exercises repeated launches of the same graph. Both 512-token checkpoint
cases passed on the real artifact:

```text
session_check=PASS prompt_tokens=512 seed_generated=1 reused_prefix_tokens=512 new_prefill_tokens=152
session_check=PASS prompt_tokens=512 seed_generated=4 reused_prefix_tokens=512 new_prefill_tokens=155
```

For each, graph-generated appended tokens exactly matched the direct full
replay. This verifies returned-token parity across checkpoint restore/append;
it does not compare every recurrent-state/KV byte.

A separate 62-step graph window in a 64-token context reached EOS at returned
token 27; the direct route produced the exact same 27 token IDs. The graph
still queued all 62 launches after EOS, so this validates output trimming, not
early GPU cancellation; the extra queued work is a latency cost for early-EOS
windows.

## Decision

**RETEST.** The reusable graph now has exact token and persistent-buffer
parity through 128 tokens and split transitions through 8K, but M27 is not
performance-qualified. Required next check: continue exactness through the
16K-attention split transition and obtain a repeated hardware-state-qualified
performance result. A
contiguous five-iteration P512 run gave 32.2954 ms/token but is contaminated:
only 261/294 telemetry samples held 1606 MHz SCLK, junction temperature
reached 96°C, and reported power reached 241 W against the 225 W cap. Raw logs
are retained under `bench/results/m27-20260919/`. Also decide whether the
measured early-EOS graph overrun warrants a safe smaller launch window.

The throttled samples correlate with heating, not startup: 1485 MHz first
appears at 79°C and persists through 96°C. Idle `rocm-smi` reports manual
performance level, 1606/1000 MHz SCLK/MCLK, 225 W cap, and fan at 15% (RPM
reported as 0). Raising fan speed requires sudo on this host; an attempted
`rocm-smi --setfan 80` was denied because the runner cannot authenticate, and
no hardware setting changed. Do not treat another thermally throttled curve
as qualification; run only after host-side cooling control is provided.

One isolated, cool-start no-preset P512/TG128 iteration then completed at
32.1368 ms/token. Its 250 ms telemetry had 101/102 samples at 1606/1000 MHz;
the single 1485 MHz sample occurred at 78°C. Junction temperature ranged
33–80°C and reported power peaked at 230 W against the 225 W cap. This is a
more useful screen but still not repeated qualification. Environment and raw
telemetry are retained in `bench/results/m27-20260919/cooled-01/`.

Five one-iteration runs were separated by cooldown to ≤40°C:

| Run | ms/token | Target SCLK samples | Junction °C | Peak reported W |
| --- | ---: | ---: | ---: | ---: |
| cooled-01 | 32.1368 | 101/102 | 33–80 | 230 |
| cooled-02 | 32.1209 | 97/100 | 34–81 | 231 |
| cooled-03 | 33.0750 | 99/101 | 35–81 | 232 |
| cooled-04 | 32.0947 | 99/101 | 37–83 | 231 |
| cooled-05 | 32.1488 | 97/100 | 37–83 | 231 |

The five-sample median is 32.1368 ms/token (range 32.0947–33.0750). Every
run had at least one sampled SCLK reduction and reported power above the
225 W cap, so this paced median remains diagnostic and does not satisfy the
strict locked-clock qualification. Per-run logs, telemetry, and environment
captures are retained under `bench/results/m27-20260919/cooled-*/`.
