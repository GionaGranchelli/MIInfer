# M31-LC-0009 — Quantized Correctness Qualification Contract

## Status and frozen identity

This is the pre-registered contract and corpus. The measurements and final qualification assessment are still pending. The source base is LC-0008 commit `87454ba3f24df1a8207d6e95f422ee213b5d1132`, which descends from LC-0006 GQA fix `8a9caec3366512e459a93bad68adde181225304b` and LC-0007 selector `6c4dc6a1af2e27692eb6586ca4b5e9e2f340e917`. M31-PERF-0001 remains separate and supplies no correctness threshold.

| Input | Pinned identity |
| --- | --- |
| Model | `Qwen3.8-27B-Q4_K_M.gguf`, SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| Tokenizer source | `Qwen/Qwen3.8-27B@366a23fd246985933964e9d21ac63eb3a93190ab`, SHA-256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| GPU | One Z840 MI50, `gfx906`, BDF `0000:06:00.0`, unique ID `0x21678e17348c2f7` |
| GPU image | `localhost/miinfer-dev:rocm-7.2.1`, digest `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e` |
| CPU reference | llama.cpp commit `d1d3c3396aa13a5f239109a822666c4870490ad5`, CPU-only, zero offload |
| Corpus | [`corpus.jsonl`](corpus.jsonl), six prompts, input IDs frozen and hashed in [`corpus-manifest.json`](corpus-manifest.json) |

Before using the corpus, compare its tokenizer hash and special-token IDs with metadata read from the pinned GGUF. The corpus manifest currently marks that check pending. Any mismatch stops the run; do not retokenize silently.

`materialize_corpus.py corpus.jsonl corpus-manifest.json prompt_ids/` verifies the corpus and per-prompt hashes before writing runner-ready `.u32le` files. The GPU runner accepts `--prompt-ids FILE`; the CPU oracle accepts those files directly. Run the CPU oracle with `MODEL.gguf PROMPT.u32le - OUTPUT_PREFIX` to capture 64 greedy tokens twice, require bitwise-identical IDs and logits, and write the first four IDs as `OUTPUT_PREFIX.forced.ids`. The CPU oracle then captures matched-history logits in three fresh contexts. It uses the pinned llama.cpp revision with MMAP load mode, 48 CPU threads, zero GPU offload, 2,304-token context, and F16 K/V. For matched-history GPU logits, pass the same four IDs with `--forced-token-ids FILE`, capture five logits with `--capture-raw-logits FILE`, capture ordinary greedy IDs with `--capture-token-ids FILE`, and use `--capture-state-prefix PREFIX` for recurrent and KV snapshots at the prompt boundary and after forced-history tokens 1 and 4. Empty forced history preserves ordinary greedy generation. These GPU snapshots do not yet provide a full cross-engine CPU/GPU canonical-state comparison; LC-0009 stays blocked until that comparison or a justified substitute is established.

Use `compare_captures.py CPU_FORCED.f32 --route NAME=ROUTE.f32` for each GPU mode. Add two `--repeat NAME=FILE` arguments per route to compare all three captures, and `--greedy cpu=CPU.ids --greedy NAME=ROUTE.ids` to report one-based token mismatch positions. It reports numeric and top-k metrics only; it applies no acceptance threshold.

## Three independent contracts

### O — Operation agreement with quantization-aware references

For each operation used to justify a candidate route, capture its exact input bytes, quantized codes, scales, layout, and output. Compare against an independently implemented scalar reference that consumes those same represented operands and implements the documented quantization and accumulation semantics. Use FP64 or exact integer accumulation where the operation permits it.

Pass requires exact equality for integer/scale outputs and finite floating outputs within the forward-error bound of the specified FP32 arithmetic. For a length-`n` FP32 sum, use `gamma_n = n*u/(1-n*u)`, `u=2^-24`, applied to the sum of absolute products; account explicitly for additional rounding, conversion, and fused operations. For recurrences, propagate intervals through the stated recurrence or use an independently checked same-input reference. If a defensible bound cannot be computed, the operation is unqualified; do not substitute a universal relative-L2 tolerance.

LC-0008 is calibration evidence, not a corpus-wide limit: the captured Q6_K × MxQ8 GPU projection differed from its arithmetic reference by relative L2 `2.797e-7`, while the same-input GPU GDN scan differed from CPU by `2.171e-7`. These bound only those captured shapes and operands.

### D — Determinism within and between execution modes

For a fixed source, binary, route, mode, prompt IDs, forced history, and reset state, run three fresh contexts. Require bitwise-identical raw F32 logits, generated IDs, and captured state within each mode. For graph versus eager on the same route, require bitwise equality where both execute the same operation sequence; otherwise report the first differing state/logit and apply Contract O to the differing operation. Any nondeterminism without a documented source blocks qualification.

### X — Cross-engine and cross-route numerical and functional quality

Compare fused GPU, aligned-MMQ GPU, and pinned CPU llama.cpp on the same frozen IDs. Capture normal greedy generation separately from teacher-forced logits. Use the same CPU-generated history for teacher-forced comparisons at each decision so every engine sees identical prior IDs. Keep original, non-aligned MMQ diagnostic results labeled separately; do not let it substitute for aligned-MMQ evidence.

For each decision, report relative L2, RMSE, maximum absolute error, top-1 margin, top-5/top-10 overlap, each engine's top-1 ID and rank under the other logits, and the exact mismatch positions in normal generation. Compare recurrent/KV state after the same forced history in canonical layouts; report relative L2, RMSE, maximum error, and position. Reject non-finite values, invalid state positions, incorrect prompt consumption, or unexplained state resets.

There is no fixed full-model relative-L2 or top-k cutoff in the existing evidence. A cross-engine argmax mismatch is numerically explainable only when an independently established output-error interval overlaps the two competing top logits. It is a functional pass only if the prompt's prewritten acceptance check also passes on each required engine/route. The coding, reasoning, and ordinary-text checks are in each corpus record. Report all metrics even when checks pass.

## Run matrix

For all six prompts, run fused GPU, aligned-MMQ GPU (`MIINFER_EXPERIMENTAL_MMQ_GATEUP_QSUM=1`), and pinned CPU llama.cpp. Run each configuration three times from fresh contexts. On GPU, run eager and HIP-graph modes for both routes. For each prompt:

1. Record ordinary greedy output up to 64 tokens for all configurations and retain all output IDs.
2. Repeat CPU greedy twice first; require exact repeat output and logits before using its first four generated IDs as the teacher-forced history.
3. Capture five complete next-token logit vectors: after the prompt and after each of the four forced history tokens. Repeat the captures three times on CPU and both GPU routes/modes.
4. Capture recurrent and KV state at the prompt boundary and after forced-history positions 1 and 4, using a documented canonical layout mapping.
5. Preserve raw arrays, per-run metadata, source/binary/image/model/tokenizer hashes, full commands, and thermal/device telemetry. Run all GPU work serially on the selected MI50.

Use the same model, seed, greedy sampler, stop policy, context settings, and reset semantics. CPU llama.cpp must disable CUDA, HIP, Vulkan, GPU layers, KQV offload, and operation offload. Do not compare logits from divergent histories as if they were matched. The 64-token and 512-token prompts remain within the 512-token prefill tile; 2,048-token prompts use four tiles of 512, matching prior LC-0003 segmentation.

## Acceptance and disposition rules

- **Operation defect:** any output outside the independently derived arithmetic bound, wrong quantized bytes/layout, invalid state transition, or non-finite result. Fail the route and localize before whole-model judgment.
- **Expected quantization difference:** operation passes Contract O, difference is reproduced on the same inputs, and the propagated output uncertainty accounts for the measured logits/state difference. This is not by itself a functional pass.
- **Functional regression:** a prewritten corpus acceptance check fails, or a high-confidence top-1 disagreement falls outside the independent numerical envelope. Fail the route.
- **Repeat instability:** Contract D fails. Block qualification regardless of average metrics.
- **Qualification:** Contracts O, D, and X pass for all six prompts, both required GPU routes, and the pinned CPU reference. Missing reference bounds, unchecked tasks, or unverified tokenizer/model identity produce **BLOCKED**, not pass.

This protocol does not change the currently accepted exact-token production gate. Replacing it, widening its criteria, promoting aligned MMQ, or starting the LC-0010 context ladder requires a separate explicit approval after this decision record.
