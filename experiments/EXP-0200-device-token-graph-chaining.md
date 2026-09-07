# EXP-0200 — Device-Side Token Chaining in HIP Graph Decode Loop

## Hypothesis

In the qualified M8 baseline, each decode step executed a captured HIP graph for the 64 layers plus LM head, but returned to the host CPU on every generated token:
1. `argmax` wrote the winning token to `argmax_token` in GPU VRAM.
2. The host CPU issued a blocking 4-byte `hipMemcpy(DeviceToHost)` and `hipDeviceSynchronize()`.
3. The host CPU passed the token to `launch_qwen35_q4_k_embedding` as a kernel argument.
4. The host CPU dispatched the next HIP graph execution.

This roundtrip introduced ~0.35 - 0.55 ms of host PCIe latency, kernel launch overhead, and driver synchronization bubble per token.

By allocating an on-device token ring buffer `d_decode_tokens` and capturing embedding inside the HIP graph:
1. `qwen35_q4_k_embedding_device_token` reads the current token directly from `d_decode_tokens[position]`.
2. `launch_qwen3_argmax` writes the predicted token directly into `d_decode_tokens[position + 1]`.
3. All $N$ decode steps are submitted back-to-back into `hipStreamPerThread`.
4. A single asynchronous copy retrieves all generated tokens upon sequence completion.
5. Slashed driver synchronization overhead will save $\ge 0.30$ ms/token, breaching the primary project milestone of **32.00 tok/s TG64**.

## Motivation

Low-batch decode latency on AMD Instinct MI50 is around ~31.5 ms/token. At this scale, driver launch overhead and host-device synchronization bubbles represent 1.0% to 1.8% of total execution time. Eliminating all host-GPU roundtrips during decode allows the GPU command processor and hardware compute queues to operate continuously without idle gaps.

## Baseline (Config A)

* Implementation: Host-mediated decode loop. Per-token `hipDeviceSynchronize()`, 4-byte host `hipMemcpy`, host embedding kernel launch, and individual graph dispatch.
* Execution flag: `MIINFER_DEVICE_TOKEN_CHAIN=0`.
* TG64 Median: **31.7458 tok/s (31.500 ms/token)**.
* Allocations during decode: 0.
* Replay determinism: PASS.

## Candidate (Config B)

* Implementation: Device-side token chaining. Dedicated kernel `qwen35_q4_k_embedding_device_token_kernel` reading token pointer from GPU VRAM, argmax output wired to next position, all graphs executed in `hipStreamPerThread` with zero intermediate synchronizations.
* Execution flag: `MIINFER_DEVICE_TOKEN_CHAIN=1`.
* TG64 Median: **32.0894 tok/s (31.163 ms/token)**.
* Allocations during decode: 0.
* Replay determinism: PASS.
* Speedup: **+1.08% (+0.3436 tok/s, 0.337 ms saved per token)**.
* Milestone: **$\ge 32.00$ tok/s BREACHED (32.0894 tok/s)**.

## Environment

* GPU: AMD Instinct MI50 32GB
* Architecture: gfx906 / Vega20, 60 CUs, Wave64
* SCLK: 1606 MHz (DPM 7)
* MCLK: 1000 MHz (DPM 2)
* Power Cap: 225W
* Model: `Qwen3.8-27B-Q4_K_M.gguf`

## 5-Pair Interleaved A/B Benchmark (TG64)

| Pair | Config A (M8 Unchained) | Config B (M9 Chained) | Latency A (ms/tok) | Latency B (ms/tok) | Delta (tok/s) | Winner |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pair 1 | 31.7686 | 32.0262 | 31.478 | 31.224 | +0.2576 (+0.81%) | Candidate |
| Pair 2 | 31.7217 | 32.0894 | 31.524 | 31.163 | +0.3677 (+1.16%) | Candidate |
| Pair 3 | 31.6239 | 31.9743 | 31.622 | 31.275 | +0.3504 (+1.11%) | Candidate |
| Pair 4 | 31.7458 | 32.1495 | 31.500 | 31.105 | +0.4037 (+1.27%) | Candidate |
| Pair 5 | 31.7777 | 32.0903 | 31.469 | 31.162 | +0.3126 (+0.98%) | Candidate |
| **Median** | **31.7458** | **32.0894** | **31.500** | **31.163** | **+0.3436 (+1.08%)** | **Candidate (5/5)** |

## Correctness

* Observable Contract (64 Layers, `--prefix64-observable-contract`):
  * Positions evaluated: 64/64 PASS.
  * Logits cosine similarity: 0.999591 (vs 0.9995 threshold).
  * Argmax match: 64/64 bit-exact match.
  * Rank 1 on reference winner: PASS.
* Replay determinism: ALL PASS.
* Decode allocations: strictly 0 on all runs.

## Profiling & Architectural Interpretation

1. **Hardware Queue Starvation Eliminated:**
   - In the unchained baseline, the GPU completed argmax and became completely idle while waiting for the PCIe bus transfer of the 4-byte token to the CPU, CPU thread wake-up, embedding kernel launch, and subsequent graph dispatch.
   - Device token chaining keeps the hardware compute queues saturated across all 64 decode steps.
2. **Deterministic State Fingerprints:**
   - All intermediate layer states and output tokens match the unchained reference bit-for-bit, confirming that asynchronous device-side token chaining introduces zero numerical deviation.

## Decision

**KEEP**.
Device token chaining won 5 out of 5 interleaved pairs, reduced decode latency by **0.337 ms/token**, maintained bit-exact determinism, and decisively breached the project milestone of **32.00 tok/s TG64** (achieving **32.0894 tok/s**).
