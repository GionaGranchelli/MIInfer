# EXP-0067 — M6-A23 Qwen3.8-27B GPU hybrid block 4–7

## Status

KEEP — layers 4–7 execute through the same GPU layer machinery as layers 0–3,
using the actual L0–L3 output and independent persistent recurrent/KV state.

## Question

Can the generic GPU executor compose recurrent layers 4–6 and full-attention
layer 7 after layers 0–3, without hardcoded first-block state, tensor, or cache
assumptions?

## Artifact and reference

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
* fixture: `/tmp/m6a1-qwen38-reference`
* reference: llama.cpp commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`
* device: MI50-class gfx906

## Candidate

Added `--block4-7` to the existing `miinfer-m6a21-qwen35-gpu-hybrid-block`
tool. It runs:

```text
L0 recurrent → L1 recurrent → L2 recurrent → L3 full attention
  → L4 recurrent → L5 recurrent → L6 recurrent → L7 full attention
```

The second block receives the first block's actual GPU output. Layers 4–6 have
independent recurrent state and layer 7 has an independent KV cache; no state
or cache buffer is shared between the two blocks.

## Command

```bash
cmake --build --preset mi50-release \
  --target miinfer-m6a21-qwen35-gpu-hybrid-block -j2
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a1-qwen38-reference --block4-7
```

## Results

The same external envelope is used for all eight layer outputs. Errors are
`max_abs`, RMS, and RMS(error)/RMS(reference). `state_correct` covers the six
recurrent state-entry tensors for layers 0–2 and 4–6.

| Position | L4 max / RMS / rel | L5 max / RMS / rel | L6 max / RMS / rel | L7 max / RMS / rel | State |
| -------: | -----------------: | -----------------: | -----------------: | -----------------: | :---- |
| P0  | 0.071922 / 0.003250 / 0.005296 | 0.064835 / 0.003782 / 0.005953 | 0.022772 / 0.004276 / 0.005596 | 0.041119 / 0.004980 / 0.006369 | PASS |
| P1  | 0.119102 / 0.004013 / 0.009144 | 0.094368 / 0.004724 / 0.009981 | 0.050324 / 0.007798 / 0.014609 | 0.032604 / 0.008750 / 0.012759 | PASS |
| P2  | 0.016827 / 0.002492 / 0.005100 | 0.068169 / 0.003192 / 0.006248 | 0.060242 / 0.004355 / 0.007896 | 0.094055 / 0.004798 / 0.007926 | PASS |
| P4  | 0.006849 / 0.001815 / 0.004964 | 0.074122 / 0.002381 / 0.006121 | 0.011463 / 0.002801 / 0.006475 | 0.012151 / 0.003080 / 0.006383 | PASS |
| P8  | 0.049069 / 0.001839 / 0.005632 | 0.012291 / 0.002348 / 0.006400 | 0.013384 / 0.003510 / 0.008747 | 0.017520 / 0.004082 / 0.008358 | PASS |
| P16 | 0.017157 / 0.002609 / 0.006006 | 0.092300 / 0.003292 / 0.006463 | 0.162170 / 0.004451 / 0.007939 | 0.017953 / 0.004558 / 0.006914 | PASS |
| P32 | 0.015417 / 0.001284 / 0.003236 | 0.028046 / 0.001636 / 0.003871 | 0.009441 / 0.002154 / 0.004876 | 0.038525 / 0.002793 / 0.005538 | PASS |
| P64 | 0.024511 / 0.001372 / 0.003967 | 0.038591 / 0.001665 / 0.004498 | 0.047319 / 0.002112 / 0.005542 | 0.021269 / 0.002476 / 0.005569 | PASS |

Maximum observed error is `0.16217` at L6/P16. The error profile remains
bounded and non-monotonic through P64.

## State fingerprints

The harness emitted logical-byte FNV-1a fingerprints at every checkpoint for
L0/L1/L2 and L4/L5/L6 recurrent state, plus K/V caches for L3 and L7. The
fingerprints exclude padding and allocator slack. The executor uses separate
state/cache addresses for both blocks. Final fingerprints were:

| Position | L0 state | L1 state | L2 state | L4 state | L5 state | L6 state | K3 | V3 | K7 | V7 |
| -------: | -------: | -------: | -------: | -------: | -------: | -------: | --: | --: | --: | --: |
| P0  | 2451724452713627422 | 16100946777412942155 | 689492793206381606 | 282661175610317932 | 15224907680276593011 | 17079586940236964079 | 16030679738720489512 | 39425660738274567 | 1624542297103504268 | 17545717735304763783 |
| P1  | 8918456245252580062 | 304148865373566752 | 8072574388220486157 | 11248322196804456284 | 2961282135884722464 | 10878231710579082051 | 9590143615054939663 | 11794599276045736033 | 9325067286437769406 | 17662890690129796340 |
| P2  | 3985104946628075402 | 10455551523943718013 | 16824913193211348513 | 4333891549869241327 | 8727776586737094085 | 2628294253686017554 | 10749924341784067565 | 10187679005748668280 | 11441590600734364811 | 9863506681121141931 |
| P4  | 13867228957992064674 | 2951962655791205847 | 7076563114778377901 | 2434395089595806588 | 48078911901503166 | 16590438163949448572 | 14163139511240763244 | 16273778360926125935 | 7075958183140649254 | 18167361344080599478 |
| P8  | 12752520487470773291 | 6604948214140104265 | 6181467609250185764 | 3867124570546945270 | 2550270199873825521 | 3535244216237070261 | 6611998883016593203 | 10615119341344739775 | 13788468903945858 | 1106811327272249682 |
| P16 | 3835682966140516700 | 7805026231362583818 | 13665270067888118022 | 12652056960988837823 | 12274500480663072744 | 18059013720931969214 | 3454383152374153314 | 14950440908617366063 | 9026186233338756506 | 12079160118561295899 |
| P32 | 4285384362259400531 | 7375965656806894406 | 14072243666591604932 | 17303348931220987756 | 9286803431866441177 | 4443763481916440254 | 6640144510259735580 | 14775097671863517002 | 15522853115614516353 | 18205729754741314796 |
| P64 | 3494274868604155650 | 14600517215752026630 | 15311287162012719556 | 11261676379213826638 | 13683268155112180095 | 1712703147445879322 | 12721657159595180664 | 4831602320261041139 | 1825061489789440819 | 14599811060670407160 |

## Checks

* Release HIP target build: PASS
* real MI50 execution through P64: PASS
* all L0–L7 output checkpoints at P0/P1/P2/P4/P8/P16/P32/P64: PASS
* recurrent state-entry comparisons: 48/48 PASS
* logical state/KV fingerprints: emitted for both hybrid blocks at all checkpoints
* steady-state allocations in this executor: 0
* recurrent and KV state remain device-resident during execution
* fresh-process replay: two complete runs emitted identical audit output
* full Release CTest: 19/19 PASS

## Interpretation

The second hybrid block works through the same index-generic GPU layer classes.
The actual L3 output feeds L4, recurrent state remains independent across both
blocks, and L7 maintains its own full-attention history. No first-block
initialization or layer-index assumption was exposed through P64.

This is a correctness/composition result, not an inference-throughput result;
the full 65-layer GPU executor, final norm, LM head, and generation path remain
incomplete.

The first replay attempt exposed nondeterminism in the shared cached-attention
kernel: probability writeback could begin while another wave was still reading
the probabilities for its output. Adding the missing block barrier before
writeback made the complete audit deterministic without changing its numerical
contract.

## Decision

**KEEP / M6-A23 complete.**

## Follow-up

Compose the first eight GPU layers, then climb the 8 → 16 → 32 → 64 layer
ladder with selected boundary checkpoints and the VRAM/performance ledger.
