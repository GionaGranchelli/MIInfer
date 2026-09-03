# EXP-0066 — M6-A22 Qwen3.8-27B GPU hybrid position audit

## Status

KEEP — the existing GPU executor composes layers 0–3 through position 64 with
bounded external-reference error, persistent state fingerprints, and clean
recurrent state-entry checks.

## Question

Does the existing layers 0–3 GPU hybrid executor remain correct across deeper
stateful positions P0, P1, P2, P4, P8, P16, P32, and P64 without adding a
second block-specific implementation?

## Artifact and reference

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
* fixture: `/tmp/m6a1-qwen38-reference`
* reference: llama.cpp commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`
* device: MI50-class gfx906

## Candidate

Extended `miinfer-m6a21-qwen35-gpu-hybrid-block` with `--deep`. The same
generic layer 0 → layer 1 → layer 2 → layer 3 executor is reused for every
position. Layer-3 diagnostic cache capacity is 128 positions so the audit can
reach P64. Production kernels and state ownership are unchanged; checkpoint
copies are diagnostic only.

For each selected position the harness compares layer outputs against the
external fixture and checks recurrent state at layer entry. It also records
logical-byte FNV-1a fingerprints for recurrent state after execution and for
the populated layer-3 K/V cache.

## Command

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release \
  --target miinfer-m6a21-qwen35-gpu-hybrid-block -j2
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a1-qwen38-reference --deep
```

## Results

Errors are `max_abs`, RMS, and RMS(error)/RMS(reference). `state_correct`
checks the three recurrent state-entry tensors against their external
`state_predelta` checkpoints; all checks passed.

| Position | L0 max / RMS / rel | L1 max / RMS / rel | L2 max / RMS / rel | L3 max / RMS / rel | State |
| -------: | -----------------: | -----------------: | -----------------: | -----------------: | :---- |
| P0  | 0.000973 / 0.000222 / 0.000832 | 0.005023 / 0.000910 / 0.002801 | 0.023933 / 0.001882 / 0.004103 | 0.032173 / 0.002543 / 0.004474 | PASS |
| P1  | 0.001511 / 0.000088 / 0.000461 | 0.006330 / 0.000798 / 0.003464 | 0.007359 / 0.002118 / 0.008612 | 0.073950 / 0.003056 / 0.008000 | PASS |
| P2  | 0.000526 / 0.000134 / 0.000573 | 0.025656 / 0.001004 / 0.003295 | 0.045748 / 0.001586 / 0.005008 | 0.019217 / 0.001973 / 0.004433 | PASS |
| P4  | 0.002347 / 0.000052 / 0.000319 | 0.002293 / 0.000485 / 0.002402 | 0.010408 / 0.000944 / 0.004696 | 0.029596 / 0.001568 / 0.004980 | PASS |
| P8  | 0.000206 / 0.000034 / 0.000214 | 0.001201 / 0.000327 / 0.001719 | 0.050670 / 0.001075 / 0.005921 | 0.062954 / 0.001627 / 0.005647 | PASS |
| P16 | 0.000729 / 0.000078 / 0.000353 | 0.001868 / 0.000444 / 0.001856 | 0.019795 / 0.000948 / 0.004001 | 0.032042 / 0.001781 / 0.004584 | PASS |
| P32 | 0.000823 / 0.000067 / 0.000318 | 0.007193 / 0.000397 / 0.001544 | 0.014580 / 0.000813 / 0.003122 | 0.014525 / 0.001079 / 0.002929 | PASS |
| P64 | 0.000853 / 0.000098 / 0.000523 | 0.004152 / 0.000338 / 0.001475 | 0.012252 / 0.000581 / 0.002658 | 0.011614 / 0.001096 / 0.003480 | PASS |

The maximum observed layer-output error is `0.0739498` at L3/P1. It is
bounded and does not grow monotonically with position.

Representative post-position fingerprints from the run were recorded for all
eight checkpoints by the harness. They cover the logical bytes only: three
recurrent `[48,128,128]` states and the populated L3 K/V cache through the
current position, excluding padding and allocator slack.

| Position | L0 state | L1 state | L2 state | L3 K | L3 V |
| -------: | -------: | -------: | -------: | ----: | ----: |
| P0  | 2451724452713627422 | 16100946777412942155 | 689492793206381606 | 16030679738720489512 | 39425660738274567 |
| P1  | 8918456245252580062 | 304148865373566752 | 8072574388220486157 | 9590143615054939663 | 11794599276045736033 |
| P2  | 3985104946628075402 | 10455551523943718013 | 16824913193211348513 | 10749924341784067565 | 10187679005748668280 |
| P4  | 13867228957992064674 | 2951962655791205847 | 7076563114778377901 | 14163139511240763244 | 16273778360926125935 |
| P8  | 12752520487470773291 | 6604948214140104265 | 6181467609250185764 | 6611998883016593203 | 10615119341344739775 |
| P16 | 3835682966140516700 | 7805026231362583818 | 13665270067888118022 | 3454383152374153314 | 14950440908617366063 |
| P32 | 4285384362259400531 | 7375965656806894406 | 14072243666591604932 | 6640144510259735580 | 14775097671863517002 |
| P64 | 3494274868604155650 | 14600517215752026630 | 15311287162012719556 | 12721657159595180664 | 4831602320261041139 |

## Checks

* Release HIP target build: PASS
* real MI50 execution through P64: PASS
* L0–L3 output checkpoints at P0/P1/P2/P4/P8/P16/P32/P64: PASS
* recurrent state-entry comparisons: 24/24 PASS
* recurrent and L3 K/V fingerprints: emitted at all eight checkpoints
* steady-state allocations in this executor: 0
* production state remains device-resident; diagnostic checkpoint copies only
* full Release CTest: 19/19 PASS

## Interpretation

The A21 executor is stable over the tested stateful depth. The non-monotonic
error profile and clean recurrent state-entry checks do not show a second
context-dependent correctness failure in the layers 0–3 composition. This is
an audit result, not an inference-throughput result; the full Qwen3.8 GPU
trunk, final norm, LM head, and generation path remain incomplete.

## Decision

**KEEP / M6-A22 complete.**

## Follow-up

Run A23: execute layers 4–7 through the same generic executor with independent
persistent state, then validate the second hybrid block before climbing the
composition ladder.
