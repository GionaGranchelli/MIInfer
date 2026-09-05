# Graph Report - mi50  (2026-09-05)

## Corpus Check
- 283 files · ~244,530 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4438 nodes · 6710 edges · 358 communities (331 shown, 27 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 232 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6ee887e0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- hardware.md
- architecture.md
- benchmarking.md
- EXP-0005 — Q4_0 × Q8_1 Quantized GEMV Baseline
- EXP-0001-benchmark-harness-validation.md
- MIInfer
- current-state.md
- AGENTS.md
- TEMPLATE.md
- Fp16GemvMetrics
- What You Must Do When Invoked
- context.md
- EXP-0002-fp16-gemv-baseline.md
- 5. ai-infos/vllm-gfx906-mobydick
- 3. iacopPBK/llama.cpp-gfx906
- 2. milpster/gfx906-llama-cpp
- Candidate research areas
- EXP-NNNN — Title
- references.md
- decisions.md
- 7. Neroued/ninfer
- M5 — Beat the Reference
- 4. mxxm-t/mx-llama.cpp
- roadmap.md
- Candidate work
- M0 — Baseline and Project Bootstrap
- 29. Development sequence
- fp16_gemv_bench.cpp
- graphify reference: extra exports and benchmark
- MI50 Platform Notes
- 8. anikifoss/llama.cpp-gfx906
- 2. Benchmark Targets
- Potential areas
- M3 — Minimal Runtime
- M1 — Kernel Laboratory
- 32. Codex task behavior
- Sha256
- graphify reference: query, path, explain
- 43. Initial MIInfer Benchmark Matrix
- Qwen3-8B Dense Control Model
- 1. ggml-org/llama.cpp
- qwen3_decode_sequence_gpu_test.cpp
- 15. Benchmark Protocol
- 10. Kernel experiments
- 40. Acceptance Categories
- 5. Required Environment Metadata
- D008 — Do Not Build a Generic Graph Runtime Initially
- 15. Research Classification
- 17. Highest-Priority Research Ideas
- 6. nlzy/vllm-gfx906
- .get
- 29. Decision
- 6. Baseline
- capture-env.sh
- 17. Correctness requirements
- 3. Fundamental engineering rules
- EXP-0004 — FP16 GEMV K-Split Parallelism
- qwen3_layer35_external_test.cpp
- qwen3_forward_gpu_test.cpp
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- 3. Correctness Is a Benchmark Prerequisite
- D001 — Build a New Runtime Instead of Forking llama.cpp
- D004 — Portability Is Not an Initial Goal
- D006 — Negative Experiments Are Retained
- D007 — M2 Is a Go/No-Go Gate
- D012 — FP32 Accumulation Is Allowed Where Correctness Requires It
- D014 — Kernel-Native Weight Layout Is Allowed
- D017 — Prefill and Decode May Use Different Kernels
- D018 — Attention Is Not Automatically the First Optimization Target
- D019 — Long Context Is a Distinct Performance Regime
- D021 — Graph Topology Is a Performance Parameter
- D002 — Maintain a Separate gfx906 Reference Runtime
- D028 — One Model First
- D029 — Batch 1 / Low Batch Is the Initial Optimization Target
- D003 — MI50 32GB / gfx906 Is the Initial Hardware Contract
- D005 — Performance Claims Require Controlled Measurement
- D009 — Static Decisions Belong Outside the Decode Hot Path
- D011 — Use Per-Operation Precision
- D013 — BF16 Is Not a Default Internal Format
- D016 — Static Activation Reuse Is Preferred Over Dynamic Cache Discovery
- D020 — HIP Graph Capture Comes After Correct Execution
- D022 — Dependencies Must Remain Minimal
- D025 — VRAM Is Part of Optimization Cost
- D030 — The Project May End After M2 or M5
- EXP-0067 — M6-A23 Qwen3.8-27B GPU hybrid block 4–7
- 18. Aggregated Results
- 7. Candidate
- run-bench.sh
- 2. Core hypothesis
- EXP-0006 — gfx906 Q4_0 × Q8_1 Packed-Dot Specialization
- M4-A — Qwen3 execution correctness foundation
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- D010 — No Silent CPU or Generic Fallback
- D015 — Repacking Should Not Occur in Hot Paths
- D026 — Strongest Available Relevant Baseline Wins
- EXP-0003 — FP16 GEMV Bottleneck Characterization
- 10. Software Environment
- 13. Correctness Method
- 17. Raw Results
- 8. Hardware
- sample-gpu.sh
- build_info.cpp
- 34. Guiding principle
- AGENTS.md
- Benchmarks
- EXP-0007 — gfx906 Zero-Point-Corrected Q4_0 × Q8_1 Dot4
- qwen3_tokenizer.cpp
- EXP-0009-kv-geometry.md
- extraction-spec.md
- diagnose-gfx802-isolation.sh
- tests/README.md
- 6. Baseline
- 35. Historical failed execution gate — superseded by Section 37
- fp16_gemv_k_split_bench.cpp
- m6a1_reference_fixture.cpp
- EXP-0008 — Direct MIInfer vs gfx906 llama.cpp MMVQ
- RecurrentLayer
- qwen3_inference_bench.cpp
- Q8_1Block
- External gfx906 Reference Baseline
- q4_q8_gemv_bench.cpp
- Qwen3GpuDecodeWorkspace
- EXP-0086 — M6-A27.6 Qwen3.8 P2 L0–L2 operation trace
- fp16_gemv_reduction_diag.cpp
- vector
- m6a15_qwen35_hybrid_block_audit.cpp
- ProfileScope
- qwen3_fast_decode_bench.cpp
- EXP-0021 — M5-C6c coalesced KV-cache writes
- Q6KHostBlock
- qwen3_primitives.cpp
- Qwen3GpuProfile
- m4b-layer6/README.md
- M3 Minimal Qwen3-8B Runtime Scaffold
- run_sequence
- Qwen3LayerTrace
- EXP-0074 — M6-A26.4 L30 production operand attribution
- EXP-0041 — M5-C15 optimization closure and parity decision gate
- m6a263_qwen35_recurrent_contract.cpp
- EXP-0100 — M6-B7 Q5_K paired-nibble decoding
- qwen3_layer_host_impl
- M4-C1 — Deterministic first generated token
- Metrics
- m4a4-four-position/README.md
- EXP-0110 — M6-B17 Q5_K×Q8_1 MMVQ recurrent projection
- M4-B — Full Qwen3 single-token forward
- m4b-single-token-legacy/README.md
- gguf.cpp
- m4b-single-token/README.md
- EXP-0026 — M5-C8c Down long-K bottleneck attribution
- run-m4c1-acceptance.sh
- run-m4b-acceptance.sh
- run-m4c2-acceptance.sh
- Q8BoundaryDiff
- m4b-layer35/README.md
- EXP-0046 — M6-A4 Qwen3.8-27B full-attention layer
- memory_stream_bench.cpp
- EXP-0042 — M6-A0 Qwen3.8-27B GGUF and architecture audit
- span
- qwen3_position_audit.cpp
- hip_smoke_bench.cpp
- model_plan.cpp
- path_
- size_t
- M5-A — Reproducible MI50 inference baseline
- M4-C3 — Text-facing greedy generation
- run-m4c3-acceptance.sh
- Qwen3GpuPlan
- qwen3_gpu_primitives.hpp
- qwen3_attention_ab_bench.cpp
- EXP-0010 — Qwen3-8B steady-state decode profile
- EXP-0017 — M5-C5a persistent Qwen3 decode workspace
- m4c3-text/README.md
- EXP-0016 — M5-C4 post-attention MI50 baseline
- run-m5a-baseline.sh
- EXP-0012 — Qwen3-8B Q4_0 MI50 comparison
- EXP-0018 — M5-C5b resident normalization weights
- EXP-0014 — Cooperative cached-attention execution
- EXP-0013 — Qwen3 position-scaled execution audit
- run-m5b-profile.sh
- GgufFile
- LayerPathCapture
- EXP-0011 — Trace-free Qwen3-8B decode control
- EXP-0099 — M6-B6 Q5_K subgroup-structured dot loop
- EXP-0022 — M5-C6d GPU-side greedy argmax
- FullAttentionLayer
- EXP-0020 — M5-C6b direct layer-output handoff
- EXP-0051 — M6-B1 Qwen3.8-27B MIInfer GPU profile readiness
- run-m5c6c-kv-cache-ab.sh
- run-m5c0-fast-decode.sh
- EXP-0015 — M5-C3 interleaved cached-attention A/B characterization
- run-m5c6d-argmax-ab.sh
- EXP-0107 — M6-B14 Q6_K MMVQ-style Q8_1 LM-head candidate
- EXP-0069 — M6-A25 Qwen3.8 sixteen-layer stateful GPU prefix
- EXP-0024 — M5-C8a FFN projection shape characterization
- run-m5c3-attention-ab.sh
- EXP-0034 — M5-C11b exact-shape FFN GEMV differential
- EXP-0019 — M5-C6a execution-overhead attribution
- EXP-0027 — M5-C9a production FFN attribution
- EXP-0037 — M5-C13a fixed-cost floor profile
- EXP-0023 — M5-C7 post-copy-cleanup decode profile
- EXP-0032 — M5-C10c FFN normalization-to-shared-Q8 fusion
- qwen3_layer6_external_test.cpp
- run-m5c6b-layer-output-ab.sh
- m6a14_qwen35_state_audit.cpp
- EXP-0028 — M5-C9b fused SwiGLU to Q8 quantization
- EXP-0071 — M6-A26.1 Qwen3.8 L30 state localization
- EXP-0031 — M5-C10b normalization/conversion boundary attribution
- EXP-0035 — M5-C12a stable-peak non-FFN profile
- EXP-0106 — M6-B13 Q6_K × Q8_1 LM-head compatibility path
- EXP-0025 — M5-C8b Down four-Wave64 GEMV candidate
- EXP-0057 — M6-A13 Qwen3.8-27B full-attention GPU layer
- m6a3_qwen35_layer.cpp
- EXP-0029 — M5-C9c Gate/Up activation-Q8 reuse
- run-m5c9c-gate-up-q8-ab.sh
- EXP-0119 — M6-B27 batched full-attention head RMS normalization
- EXP-0033 — M5-C11a production and llama.cpp differential baseline
- EXP-0030 — M5-C10a refreshed P64 production profile
- Q5K
- EXP-0039 — M5-C13c fixed-floor contract map
- EXP-0053 — M6-A9 Qwen3.8-27B LM-head GPU projection
- EXP-0036 — M5-C12b cooperative attention scaling
- EXP-0084 — M6-A27.4 Qwen3.8 full-model observable contract adjudication
- EXP-0038 — M5-C13b LM-head contract audit
- EXP-0054 — M6-A10 Qwen3.8-27B Q4_K projection
- EXP-0063 — M6-A19 Qwen3.8-27B convolution GPU path
- EXP-0040 — M5-C14a fixed-floor execution map
- EXP-0043 — M6-A1 Qwen3.8-27B external reference fixture
- run-m6a1-reference-fixture.sh
- 4. Difference inventory
- EXP-0044 — M6-A2 Qwen3.8 projection/kernel compatibility audit
- EXP-0062 — M6-A18 Qwen3.8-27B DeltaNet GPU state core
- EXP-0045 — M6-A3 Qwen3.8-27B single DeltaNet layer
- UpdateProvenance
- EXP-0047 — M6-A5 Qwen3.8-27B four-layer hybrid block
- EXP-0048 — M6-A6 Qwen3.8-27B full 64-layer forward
- string
- EXP-0050 — M6-B0 Qwen3.8-27B llama.cpp MI50 baseline
- array
- EXP-0056 — M6-A12 Qwen3.8-27B attention projections
- EXP-0049 — M6-A7 Qwen3.8-27B stateful generation
- Qwen35Config
- qwen3_layer0_gpu_test.cpp
- Metrics
- run-m6b0-llama-baseline.sh
- RmsVariant
- EXP-0058 — M6-A14 Qwen3.8-27B state fingerprints and reset audit
- EXP-0125 — M6-B33 post-B32 production profile
- EXP-0059 — M6-A15 Qwen3.8-27B layers 0–3 hybrid-block audit
- EXP-0093 — M6-B1 Qwen3.8-27B native GPU generation baseline
- EXP-0055 — M6-A11 Qwen3.8-27B composed attention prefix
- qwen3_trace_compare.cpp
- m6a13_qwen35_full_attention_layer.cpp
- EXP-0079 — M6-A26.9 Qwen3.8 external recurrent-state contract
- EXP-0052 — M6-A8 Qwen3.8-27B GPU foundation
- EXP-0089 — M6-A27.9 Qwen3.8 L0 Q5_K block contract
- EXP-0073 — M6-A26.3 Qwen3.8 recurrent-state contract adjudication
- EXP-0060 — M6-A16 Qwen3.8-27B layers 4–7 hybrid-block audit
- EXP-0090 — M6-A27.9 full observable-contract retest
- EXP-0061 — M6-A17 Qwen3.8-27B composition ladder
- AttentionPathReplay
- FfnTailReplay
- EXP-0096 — M6-B3 Q4_K metadata staging
- qwen3_gpu_layer.cpp
- m6a21_qwen35_gpu_hybrid_block.cpp
- EXP-0066 — M6-A22 Qwen3.8-27B GPU hybrid position audit
- m6a18_qwen35_deltanet_state_gpu.cpp
- qwen3_gpu_layer.hpp
- m6a10_qwen35_q4k_projection.cpp
- m6a8_qwen35_gpu_foundation.cpp
- EXP-0064 — M6-A20 Qwen3.8-27B recurrent layer on GPU
- EXP-0072 — M6-A26.2 Qwen3.8 L30 update provenance
- EXP-0103 — M6-B10 recurrent Q8_K input reuse
- qwen3_swiglu_q8_bench.cpp
- EXP-0065 — M6-A21 Qwen3.8-27B GPU hybrid block
- Qwen3Model
- EXP-0068 — M6-A24 Qwen3.8 eight-layer stateful GPU prefix
- 11. joe2gaan/localaiservers
- EXP-0080 — M6-A27 Qwen3.8 sixty-four-layer GPU composition
- EXP-0085 — M6-A27.5 Qwen3.8 P2 drift localization
- EXP-0070 — M6-A26 Qwen3.8 thirty-two-layer stateful GPU prefix
- m6a9_qwen35_lm_head.cpp
- EXP-0124 — M6-B32 transposed recurrent no-decay store
- model_loader_test.cpp
- qwen3_cached_attention_determinism_gpu_test.cpp
- EXP-0075 — M6-A26.5 L30 K-path provenance
- m6a12_qwen35_attention_projections.cpp
- EXP-0076 — M6-A26.6 L29 output provenance
- EXP-0077 — M6-A26.7 L29 gated-output provenance
- EXP-0078 — M6-A26.8 L29 gate-input provenance
- Group
- EXP-0087 — M6-A27.7 Qwen3.8 L0 output-projection contract adjudication
- EXP-0123 — M6-B31 recurrent FFN Gate/Up two-row MMVQ
- EXP-0081 — M6-A27.1 Qwen3.8 L54/P1 output attribution
- EXP-0109 — M6-B16 projection-input Q8_K reuse
- EXP-0095 — M6-B2 Q5_K scale/min unpack hoisting
- EXP-0091 — M6-A27 observable numerical-equivalence closure
- EXP-0088 — M6-A27.8 Qwen3.8 L0 Q8_K contract
- qwen3_primitives_test.cpp
- Qwen3LayerWeights
- Qwen3GpuDecodeCache
- qwen3_decode_profile.cpp
- EXP-0113 — M6-B20 Q6_K×Q8_1 MMVQ recurrent QKV
- EXP-0082 — M6-A27.2 Qwen3.8 L53/P1 output provenance
- M4 — First Correct End-to-End Generation
- EXP-0083 — M6-A27.3 Qwen3.8 L53 gated-path contract adjudication
- EXP-0101 — M6-B8 cached-attention Wave64-local reduction
- EXP-0105 — M6-B12 Q6_K packed dot4 projections
- EXP-0092 — M6-A28 native autoregressive GPU generation
- Qwen3GpuProfileEvent
- EXP-0102 — M6-B9 Q6_K LM-head index hoisting
- qwen3_generate.cpp
- EXP-0094 — M6-B2 direct layer-output handoff
- require_match
- Qwen3Layer0KvCache
- validate_position
- EXP-0116 — M6-B24 full-attention stage attribution
- EXP-0098 — M6-B5 Qwen3.8 Q8_K activation reuse
- EXP-0114 — M6-B21 Q4_K×Q8_1 MMVQ recurrent gate
- EXP-0097 — M6-B4 Q5_K four-row workgroup
- 5. Runtime Layers
- EXP-0104 — M6-B11 Q4_K packed dot4 projections
- EXP-0122 — M6-B30 transposed DeltaNet recurrent state
- GatePathCapture
- EXP-0111 — M6-B18 Q4_K×Q8_1 MMVQ FFN Down
- hip_check.hpp
- Roadmap Principles
- EXP-0108 — M6-B15 recurrent state-update no-decay-store candidate
- HostQ8Block
- EXP-0115 — M6-B22 Q6_K×Q8_K packed-dot4 recurrent QKV
- KeyPathCapture
- EXP-0120 — M6-B28 post-B27 production profile
- EXP-0112 — M6-B19 Q4_K×Q8_1 MMVQ FFN Gate/Up
- m6a19_qwen35_conv_gpu.cpp
- EXP-0121 — M6-B29 recurrent stage attribution
- m6a11_qwen35_attention_prefix.cpp
- RecurrentOperands
- run_ladder
- StageProfile
- EXP-0126 — M6-B34 fused SiLU to Q8_1
- EXP-0117 — M6-B25 fine full-attention attribution
- Metrics
- EXP-0118 — M6-B26 Q-projection Q8_1 MMVQ candidate
- Current Project Status
- 13. Prefill vs Decode
- Current Scope
- Metrics
- Q8KDeviceBlock
- Qwen3TensorView
- run_combined
- fp16_gemv_reference.cpp

## God Nodes (most connected - your core abstractions)
1. `RecurrentLayer` - 96 edges
2. `path_` - 78 edges
3. `FullAttentionLayer` - 58 edges
4. `Qwen3LayerTrace` - 55 edges
5. `Qwen3GpuPlan` - 51 edges
6. `Qwen3GpuProfile` - 48 edges
7. `Qwen3GpuDecodeWorkspace` - 46 edges
8. `Qwen3Model` - 45 edges
9. `main()` - 41 edges
10. `GgufFile` - 40 edges

## Surprising Connections (you probably didn't know these)
- `run_quantize()` --references--> `Q8_1Block`  [INFERRED]
  bench/q4_q8_gemv_bench.cpp → include/miinfer/q4_q8_gemv.hpp
- `run_fanout()` --references--> `GemvShape`  [INFERRED]
  bench/q4_q8_gemv_bench.cpp → include/miinfer/fp16_gemv.hpp
- `run_fanout()` --references--> `Q4_0Block`  [INFERRED]
  bench/q4_q8_gemv_bench.cpp → include/miinfer/q4_q8_gemv.hpp
- `run_fanout()` --references--> `Q8_1Block`  [INFERRED]
  bench/q4_q8_gemv_bench.cpp → include/miinfer/q4_q8_gemv.hpp
- `run_once()` --calls--> `length`  [INFERRED]
  bench/qwen3_attention_ab_bench.cpp → include/miinfer/qwen3_gpu_layer.hpp

## Import Cycles
- None detected.

## Communities (358 total, 27 thin omitted)

### Community 0 - "hardware.md"
Cohesion: 0.04
Nodes (46): 10. Candidate Quantized Execution Path, 11. FP16 Behavior, 12. BF16, 13. Memory Bandwidth, 14. HBM vs Cache, 15. Weight Compression, 16. HBM Clock, 17. GPU Clock (+38 more)

### Community 1 - "architecture.md"
Cohesion: 0.06
Nodes (33): 10. Memory Architecture, 11. Weight Residency, 12. Tensor Layout, 14. Context-Length Sensitivity, 15. Attention Architecture, 16. MoE Architecture, 17. Static Model Knowledge, 18. Kernel Configuration (+25 more)

### Community 2 - "benchmarking.md"
Cohesion: 0.05
Nodes (43): 10. Reversed Ordering, 11. Statistics, 12. Performance Delta, 13. Benchmark Stability, 14. Baseline Selection, 15. Baseline Pinning, 16. Model Equivalence, 17. Quantization Equivalence (+35 more)

### Community 3 - "EXP-0005 — Q4_0 × Q8_1 Quantized GEMV Baseline"
Cohesion: 0.11
Nodes (17): 10. Comparison with accepted FP16 controls, 11. Projection-only sanity check, 12. External Q4_0 reference, 13. Decision, 14. Follow-up, 1. Question, 2. Hypothesis, 3. Baseline and scope (+9 more)

### Community 4 - "EXP-0001-benchmark-harness-validation.md"
Cohesion: 0.06
Nodes (34): 10. Software Environment, 11. Model / Workload, 12. Test Matrix, 13. Correctness Method, 14. Correctness Results, 15. Benchmark Protocol, 16. Pre-Run Hardware State, 17. Raw Results (+26 more)

### Community 5 - "MIInfer"
Cohesion: 0.05
Nodes (38): Architecture direction, Benchmark philosophy, Building, Contributing, Core hypothesis, Correctness before performance, Design principles, Development roadmap (+30 more)

### Community 6 - "current-state.md"
Cohesion: 0.10
Nodes (19): Current Benchmark Priority, Current Build Direction, Current Correctness Policy, Current Dependency Policy, Current Experiment Queue, Current Hardware Observation Requirements, Current Hardware Target, Current Performance Policy (+11 more)

### Community 7 - "AGENTS.md"
Cohesion: 0.07
Nodes (27): 11. Static specialization, 12. Static kernel selection, 13. HIP graph strategy, 14. Benchmarking, 15. Hardware-state validation, 16. Benchmark methodology, 18. Experiment records, 19. Profiling (+19 more)

### Community 8 - "TEMPLATE.md"
Cohesion: 0.07
Nodes (27): 11. Model / Workload, 12. Test Matrix, 14. Correctness Results, 16. Pre-Run Hardware State, 19. Per-Shape Results, 1. Question, 20. Effective Bandwidth, 21. Resource Usage (+19 more)

### Community 9 - "Fp16GemvMetrics"
Cohesion: 0.25
Nodes (8): Fp16GemvMetrics, cosine_similarity, inf_detected, max_abs_error, max_relative_error, mean_abs_error, nan_detected, pass

### Community 10 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 11 - "context.md"
Cohesion: 0.09
Nodes (22): 1. Situation, 2. Context & Complication, 3. Question / Goal, 4. Answer / Recommendation, 5. Socratic Clause, 6. Summary / Key Takeaways, Experiment 001, **iacopPBK: NO** (+14 more)

### Community 12 - "EXP-0002-fp16-gemv-baseline.md"
Cohesion: 0.11
Nodes (18): 10. Cache regime and benchmark command, 11. Correctness results, 12. Canonical performance results, 13. Hardware state and contamination, 14. Kernel resources, 15. Interpretation, 16. Decision, 17. Follow-up recommendation (+10 more)

### Community 13 - "5. ai-infos/vllm-gfx906-mobydick"
Cohesion: 0.10
Nodes (20): 5. ai-infos/vllm-gfx906-mobydick, Accumulator precision, BF16, GPTQ / AWQ, Long-context correctness, MIInfer implication, MIInfer implication, MIInfer implication (+12 more)

### Community 14 - "3. iacopPBK/llama.cpp-gfx906"
Cohesion: 0.12
Nodes (17): 3. iacopPBK/llama.cpp-gfx906, FlashAttention and q8 attention, gfx906 primitive layer, Logical half-wave execution, MIInfer implication, MIInfer implication, MIInfer implication, MIInfer implication (+9 more)

### Community 15 - "2. milpster/gfx906-llama-cpp"
Cohesion: 0.13
Nodes (15): 2. milpster/gfx906-llama-cpp, Adaptive speculative/MTP work, Graph topology, Hardware-state contamination, Important lessons, MIInfer implication, MIInfer implication, MIInfer implication (+7 more)

### Community 16 - "Candidate research areas"
Cohesion: 0.13
Nodes (15): Candidate research areas, Core question, gfx906 primitives, GO, Go / no-go decision, Goal, Kernel configuration, M2 — Prove Specialization (+7 more)

### Community 17 - "EXP-NNNN — Title"
Cohesion: 0.13
Nodes (14): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-NNNN — Title, Follow-up (+6 more)

### Community 18 - "references.md"
Cohesion: 0.10
Nodes (18): 10. nick413-bit/gfx906-fa-vllm, 12. AMD gfx906 ISA Documentation, 13. AMD HIP / ROCm Documentation, 14. rocBLAS / hipBLAS, 16. Current Research Synthesis, 18. Research Intake Checklist, 19. External Code Policy, 20. Research Notes vs Decisions (+10 more)

### Community 19 - "decisions.md"
Cohesion: 0.15
Nodes (12): D023 — Triton Is a Research Tool, Not a Required Runtime, D024 — Benchmarkability Is an Architectural Requirement, D027 — Serving Does Not Define the Core Runtime, Decision, Decision, Decision, Decision Change Process, Guiding Rule (+4 more)

### Community 20 - "7. Neroued/ninfer"
Cohesion: 0.17
Nodes (12): 7. Neroued/ninfer, Fixed memory planning, Graph-based decode, MIInfer implication, MIInfer implication, MIInfer implication, MIInfer implication, Packed artifact (+4 more)

### Community 21 - "M5 — Beat the Reference"
Cohesion: 0.15
Nodes (13): Comparison dimensions, Comparison rules, Context regimes, Decode, Exit criteria, Goal, H0 supported, H1 supported (+5 more)

### Community 22 - "4. mxxm-t/mx-llama.cpp"
Cohesion: 0.20
Nodes (10): 4. mxxm-t/mx-llama.cpp, Activation reuse, MIInfer implication, MIInfer implication, MIInfer implication, MXFP4, Role, Weight repacking (+2 more)

### Community 23 - "roadmap.md"
Cohesion: 0.14
Nodes (12): Candidate areas, Contract, Current Execution Order, Current Status, Deferred / Explicitly Out of Scope, Goal, Immediate Next Milestone, M6-A — Reference-correct execution-contract exploration (+4 more)

### Community 24 - "Candidate work"
Cohesion: 0.20
Nodes (10): Activation reuse, Candidate work, Exit criteria, Goal, HIP graph capture, Kernel specialization, M6 — Runtime Specialization, Native weight packing (+2 more)

### Community 25 - "M0 — Baseline and Project Bootstrap"
Cohesion: 0.20
Nodes (10): Benchmark protocol, Deliverables, Exit criteria, Goal, Hardware environment, M0 — Baseline and Project Bootstrap, Non-goals, Questions (+2 more)

### Community 26 - "29. Development sequence"
Cohesion: 0.22
Nodes (9): 29. Development sequence, M0 — Baseline, M1 — Kernel laboratory, M2 — Prove specialization, M3 — Minimal runtime, M4 — First correct generation, M5 — Beat reference, M6 — Runtime specialization (+1 more)

### Community 27 - "fp16_gemv_bench.cpp"
Cohesion: 0.11
Nodes (32): ostream, string, uint32_t, vector, implementation_label(), json_escape(), main(), median_of() (+24 more)

### Community 28 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 29 - "MI50 Platform Notes"
Cohesion: 0.25
Nodes (7): Current gate status, Fedora development packages, MI50 Platform Notes, Recovery options, ROCr failure, Root-cause status, Target and topology

### Community 30 - "8. anikifoss/llama.cpp-gfx906"
Cohesion: 0.25
Nodes (8): 8. anikifoss/llama.cpp-gfx906, KV precision, MIInfer implication, MIInfer implication, MIInfer implication, Qwen3-30B-A3B, Relevant optimization areas, Role

### Community 31 - "2. Benchmark Targets"
Cohesion: 0.29
Nodes (7): 2.1 Primitive benchmarks, 2.2 Kernel benchmarks, 2.3 Model-component benchmarks, 2.4 End-to-end benchmarks, 2. Benchmark Targets, Prompt processing / prefill, Token generation / decode

### Community 32 - "Potential areas"
Cohesion: 0.29
Nodes (7): Additional quantization, Long-context specialization, M7 — Expansion, Potential areas, Second model, Serving, Speculative decoding / MTP

### Community 33 - "M3 — Minimal Runtime"
Cohesion: 0.29
Nodes (7): Architectural rule, Core runtime responsibilities, Exit criteria, Goal, M3 — Minimal Runtime, Model scope, Quantization scope

### Community 34 - "M1 — Kernel Laboratory"
Cohesion: 0.29
Nodes (7): Benchmark harness, Exit criteria, Goal, Initial kernel areas, Initial representative shapes, M1 — Kernel Laboratory, Questions

### Community 35 - "32. Codex task behavior"
Cohesion: 0.33
Nodes (6): 32. Codex task behavior, Before adding abstractions, Before adding fallback behavior, Before declaring a performance win, Before modifying code, Before removing apparently strange gfx906 code

### Community 36 - "Sha256"
Cohesion: 0.18
Nodes (15): array, byte, size_t, span, string, uint32_t, uint64_t, rotate_right() (+7 more)

### Community 37 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 38 - "43. Initial MIInfer Benchmark Matrix"
Cohesion: 0.33
Nodes (6): 43. Initial MIInfer Benchmark Matrix, Activation quantization, Cooperative execution, GEMV, Memory layout, Normalization

### Community 39 - "Qwen3-8B Dense Control Model"
Cohesion: 0.33
Nodes (5): Exact configuration, F16 artifact route, Qwen3-8B Dense Control Model, Real model projection shapes, Source

### Community 40 - "1. ggml-org/llama.cpp"
Cohesion: 0.33
Nodes (6): 1. ggml-org/llama.cpp, Do not inherit automatically, Important lessons, MIInfer question, Potentially reusable knowledge, Role

### Community 41 - "qwen3_decode_sequence_gpu_test.cpp"
Cohesion: 0.24
Nodes (22): argmax(), cache_lengths(), compare_dumps(), size_t, span, string, uint32_t, vector (+14 more)

### Community 42 - "15. Benchmark Protocol"
Cohesion: 0.33
Nodes (6): 15. Benchmark Protocol, Iterations per run, Measured runs, Ordering, Timing method, Warm-up

### Community 43 - "10. Kernel experiments"
Cohesion: 0.40
Nodes (5): 10. Kernel experiments, Activation reuse, Attention, Data layout, Matrix/vector execution

### Community 44 - "40. Acceptance Categories"
Cohesion: 0.40
Nodes (5): 40. Acceptance Categories, INVALID, KEEP, REJECT, RETEST

### Community 45 - "5. Required Environment Metadata"
Cohesion: 0.40
Nodes (5): 5. Required Environment Metadata, Hardware, Host, Software, Workload

### Community 46 - "D008 — Do Not Build a Generic Graph Runtime Initially"
Cohesion: 0.40
Nodes (5): Consequences, D008 — Do Not Build a Generic Graph Runtime Initially, Decision, Preferred direction, Reason

### Community 47 - "15. Research Classification"
Cohesion: 0.40
Nodes (5): 15. Research Classification, Category A — Architecture lessons, Category B — Kernel hypotheses, Category C — Failure lessons, Category D — Runtime features

### Community 48 - "17. Highest-Priority Research Ideas"
Cohesion: 0.40
Nodes (5): 17. Highest-Priority Research Ideas, Later, P0, P1, P2

### Community 49 - "6. nlzy/vllm-gfx906"
Cohesion: 0.40
Nodes (5): 6. nlzy/vllm-gfx906, Key historical observations, MIInfer implication, Role, Status

### Community 50 - ".get"
Cohesion: 0.23
Nodes (10): Buffer, allocate(), uint32_t, project(), project_q4_q8_1(), project_q4_q8_1_prequantized(), project_q5_q8_1(), project_q6_q8_k_dot4() (+2 more)

### Community 51 - "29. Decision"
Cohesion: 0.40
Nodes (5): 29. Decision, INVALID, KEEP, REJECT, RETEST

### Community 52 - "6. Baseline"
Cohesion: 0.40
Nodes (5): 6. Baseline, Commit, Implementation, Kernel, Relevant configuration

### Community 53 - "capture-env.sh"
Cohesion: 0.70
Nodes (4): first_line(), full_output(), json_quote(), capture-env.sh script

### Community 54 - "17. Correctness requirements"
Cohesion: 0.50
Nodes (4): 17. Correctness requirements, Level 1 — numerical, Level 2 — layer/model, Level 3 — generation

### Community 55 - "3. Fundamental engineering rules"
Cohesion: 0.50
Nodes (4): 3.1 Measure before optimizing, 3.2 Every optimization needs a baseline, 3.3 Negative results are valuable, 3. Fundamental engineering rules

### Community 56 - "EXP-0004 — FP16 GEMV K-Split Parallelism"
Cohesion: 0.11
Nodes (18): 10. Test Matrix, 11. Correctness Method, 12. Benchmark, 13. Acceptance, 14. Explicit Exclusions, 15. Results, 16. Decision, 17. Follow-up (+10 more)

### Community 57 - "qwen3_layer35_external_test.cpp"
Cohesion: 0.22
Nodes (32): AccumulationContract, RmsReduction, compare_q8_blocks(), size_t, span, vector, dequantize_q8_exact(), expand_gqa() (+24 more)

### Community 58 - "qwen3_forward_gpu_test.cpp"
Cohesion: 0.27
Nodes (22): argmax(), capture_q8_input(), compare_checkpoint(), size_t, vector, exact_equal(), main(), overlap_count() (+14 more)

### Community 59 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 60 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 61 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 62 - "3. Correctness Is a Benchmark Prerequisite"
Cohesion: 0.50
Nodes (4): 3. Correctness Is a Benchmark Prerequisite, End-to-end inference, Kernel, Model component

### Community 63 - "D001 — Build a New Runtime Instead of Forking llama.cpp"
Cohesion: 0.50
Nodes (4): Consequences, D001 — Build a New Runtime Instead of Forking llama.cpp, Decision, Reason

### Community 64 - "D004 — Portability Is Not an Initial Goal"
Cohesion: 0.50
Nodes (4): Consequences, D004 — Portability Is Not an Initial Goal, Decision, Reason

### Community 65 - "D006 — Negative Experiments Are Retained"
Cohesion: 0.50
Nodes (4): Consequences, D006 — Negative Experiments Are Retained, Decision, Reason

### Community 66 - "D007 — M2 Is a Go/No-Go Gate"
Cohesion: 0.50
Nodes (4): Consequences, D007 — M2 Is a Go/No-Go Gate, Decision, Reason

### Community 67 - "D012 — FP32 Accumulation Is Allowed Where Correctness Requires It"
Cohesion: 0.50
Nodes (4): Consequences, D012 — FP32 Accumulation Is Allowed Where Correctness Requires It, Decision, Reason

### Community 68 - "D014 — Kernel-Native Weight Layout Is Allowed"
Cohesion: 0.50
Nodes (4): Consequences, D014 — Kernel-Native Weight Layout Is Allowed, Decision, Reason

### Community 69 - "D017 — Prefill and Decode May Use Different Kernels"
Cohesion: 0.50
Nodes (4): Consequences, D017 — Prefill and Decode May Use Different Kernels, Decision, Reason

### Community 70 - "D018 — Attention Is Not Automatically the First Optimization Target"
Cohesion: 0.50
Nodes (4): Consequences, D018 — Attention Is Not Automatically the First Optimization Target, Decision, Reason

### Community 71 - "D019 — Long Context Is a Distinct Performance Regime"
Cohesion: 0.50
Nodes (4): Consequences, D019 — Long Context Is a Distinct Performance Regime, Decision, Reason

### Community 72 - "D021 — Graph Topology Is a Performance Parameter"
Cohesion: 0.50
Nodes (4): Consequences, D021 — Graph Topology Is a Performance Parameter, Decision, Reason

### Community 73 - "D002 — Maintain a Separate gfx906 Reference Runtime"
Cohesion: 0.50
Nodes (4): Consequences, D002 — Maintain a Separate gfx906 Reference Runtime, Decision, Reason

### Community 74 - "D028 — One Model First"
Cohesion: 0.50
Nodes (4): Consequences, D028 — One Model First, Decision, Reason

### Community 75 - "D029 — Batch 1 / Low Batch Is the Initial Optimization Target"
Cohesion: 0.50
Nodes (4): Consequences, D029 — Batch 1 / Low Batch Is the Initial Optimization Target, Decision, Reason

### Community 76 - "D003 — MI50 32GB / gfx906 Is the Initial Hardware Contract"
Cohesion: 0.50
Nodes (4): Consequences, D003 — MI50 32GB / gfx906 Is the Initial Hardware Contract, Decision, Reason

### Community 77 - "D005 — Performance Claims Require Controlled Measurement"
Cohesion: 0.50
Nodes (4): D005 — Performance Claims Require Controlled Measurement, Decision, Reason, Required evidence

### Community 78 - "D009 — Static Decisions Belong Outside the Decode Hot Path"
Cohesion: 0.50
Nodes (4): D009 — Static Decisions Belong Outside the Decode Hot Path, Decision, Examples, Reason

### Community 79 - "D011 — Use Per-Operation Precision"
Cohesion: 0.50
Nodes (4): D011 — Use Per-Operation Precision, Decision, Example, Reason

### Community 80 - "D013 — BF16 Is Not a Default Internal Format"
Cohesion: 0.50
Nodes (4): D013 — BF16 Is Not a Default Internal Format, Decision, Initial preference, Reason

### Community 81 - "D016 — Static Activation Reuse Is Preferred Over Dynamic Cache Discovery"
Cohesion: 0.50
Nodes (4): D016 — Static Activation Reuse Is Preferred Over Dynamic Cache Discovery, Decision, Example, Reason

### Community 82 - "D020 — HIP Graph Capture Comes After Correct Execution"
Cohesion: 0.50
Nodes (4): D020 — HIP Graph Capture Comes After Correct Execution, Decision, Order, Reason

### Community 83 - "D022 — Dependencies Must Remain Minimal"
Cohesion: 0.50
Nodes (4): D022 — Dependencies Must Remain Minimal, Decision, Initially disallowed by default, Reason

### Community 84 - "D025 — VRAM Is Part of Optimization Cost"
Cohesion: 0.50
Nodes (4): D025 — VRAM Is Part of Optimization Cost, Decision, Example, Reason

### Community 85 - "D030 — The Project May End After M2 or M5"
Cohesion: 0.50
Nodes (4): D030 — The Project May End After M2 or M5, Decision, Possible valid outcomes, Reason

### Community 86 - "EXP-0067 — M6-A23 Qwen3.8-27B GPU hybrid block 4–7"
Cohesion: 0.15
Nodes (12): Artifact and reference, Candidate, Checks, Command, Decision, EXP-0067 — M6-A23 Qwen3.8-27B GPU hybrid block 4–7, Follow-up, Interpretation (+4 more)

### Community 87 - "18. Aggregated Results"
Cohesion: 0.50
Nodes (4): 18. Aggregated Results, Baseline, Candidate, Delta

### Community 88 - "7. Candidate"
Cohesion: 0.50
Nodes (4): 7. Candidate, Commit, Difference from baseline, Implementation

### Community 89 - "run-bench.sh"
Cohesion: 0.83
Nodes (3): cleanup(), run-bench.sh script, stop_telemetry()

### Community 90 - "2. Core hypothesis"
Cohesion: 0.67
Nodes (3): 2. Core hypothesis, H0, H1

### Community 91 - "EXP-0006 — gfx906 Q4_0 × Q8_1 Packed-Dot Specialization"
Cohesion: 0.11
Nodes (18): 10. FP16 control comparison, 11. Quantization-inclusive views, 12. Kernel resources and ISA observations, 13. External MMVQ comparison, 14. Projection-only sanity check, 15. Bottleneck interpretation, 16. Decision, 17. Next experiment (+10 more)

### Community 92 - "M4-A — Qwen3 execution correctness foundation"
Cohesion: 0.25
Nodes (7): Current gate, Host layer-0 acceptance run, Implemented foundation, M4-A3 GPU composition acceptance, M4-A4 KV-cache contract, M4-A — Qwen3 execution correctness foundation, Pinned reference fixture

### Community 95 - "D010 — No Silent CPU or Generic Fallback"
Cohesion: 0.67
Nodes (3): D010 — No Silent CPU or Generic Fallback, Decision, Reason

### Community 96 - "D015 — Repacking Should Not Occur in Hot Paths"
Cohesion: 0.67
Nodes (3): D015 — Repacking Should Not Occur in Hot Paths, Decision, Reason

### Community 97 - "D026 — Strongest Available Relevant Baseline Wins"
Cohesion: 0.67
Nodes (3): D026 — Strongest Available Relevant Baseline Wins, Decision, Reason

### Community 98 - "EXP-0003 — FP16 GEMV Bottleneck Characterization"
Cohesion: 0.12
Nodes (15): Baseline and controls, Bottleneck classification, Cache regime, Decision, EXP-0003 — FP16 GEMV Bottleneck Characterization, Hypothesis and scope, K-scaling diagnostic, M-scaling diagnostic (+7 more)

### Community 99 - "10. Software Environment"
Cohesion: 0.67
Nodes (3): 10. Software Environment, Compiler flags, Environment variables

### Community 100 - "13. Correctness Method"
Cohesion: 0.67
Nodes (3): 13. Correctness Method, Acceptance tolerance, Validation metrics

### Community 101 - "17. Raw Results"
Cohesion: 0.67
Nodes (3): 17. Raw Results, Baseline, Candidate

### Community 102 - "8. Hardware"
Cohesion: 0.67
Nodes (3): 8. Hardware, GPU, Runtime state

### Community 107 - "Benchmarks"
Cohesion: 0.06
Nodes (31): Benchmarks, End-to-end M5-A baseline, M5-B steady-state decode profile, M5-C0 trace-free decode benchmark, M5-C10a refreshed P64 production profile, M5-C10b normalization/conversion boundary attribution, M5-C10c FFN normalization-to-shared-Q8 fusion, M5-C11a production and llama.cpp differential baseline (+23 more)

### Community 108 - "EXP-0007 — gfx906 Zero-Point-Corrected Q4_0 × Q8_1 Dot4"
Cohesion: 0.12
Nodes (16): 10. Hardware validity, 11. External MMVQ comparison, 12. Projection-only sanity check, 13. Decision, 14. M2 status, 15. Next experiment, 1. Question, 2. Hypothesis and motivation (+8 more)

### Community 109 - "qwen3_tokenizer.cpp"
Cohesion: 0.09
Nodes (41): size_t, string, uint32_t, unordered_map, vector, Qwen3Tokenizer, decode, encode (+33 more)

### Community 110 - "EXP-0009-kv-geometry.md"
Cohesion: 0.14
Nodes (13): 10. Comparison with MMVQ, 11. Decision, 12. Next experiment, 1. Question, 2. Hypothesis, 3. Prior evidence, 4. Candidates, 5. Environment and method (+5 more)

### Community 112 - "diagnose-gfx802-isolation.sh"
Cohesion: 0.54
Nodes (7): capture_env(), capture_topology(), finish(), rebind_target(), run_capture(), diagnose-gfx802-isolation.sh script, usage()

### Community 116 - "6. Baseline"
Cohesion: 0.50
Nodes (4): 6. Baseline, Implementation, Kernel, Relevant configuration

### Community 117 - "35. Historical failed execution gate — superseded by Section 37"
Cohesion: 0.67
Nodes (3): 35. Historical failed execution gate — superseded by Section 37, 36. Task 3 Platform-Recovery Pilot, 37. Accepted MI50 execution

### Community 118 - "fp16_gemv_k_split_bench.cpp"
Cohesion: 0.15
Nodes (20): ostream, string, uint32_t, vector, escape(), main(), median(), nonnegative() (+12 more)

### Community 119 - "m6a1_reference_fixture.cpp"
Cohesion: 0.11
Nodes (34): ggml_tensor, llama_model, llama_token, llama_vocab, callback(), Capture, position, positions (+26 more)

### Community 120 - "EXP-0008 — Direct MIInfer vs gfx906 llama.cpp MMVQ"
Cohesion: 0.13
Nodes (15): 10. Reference ISA and resources, 11. Architectural differences relevant to K/V, 12. M2 decision, 13.1 Re-evaluation after EXP-0009, 13. Next experiment, 1. Question, 2. Hypothesis and motivation, 3. Reference path (+7 more)

### Community 121 - "RecurrentLayer"
Cohesion: 0.03
Nodes (67): RecurrentLayer, alpha_raw, beta, beta_raw, d_a, d_alpha, d_attn_norm, d_beta (+59 more)

### Community 122 - "qwen3_inference_bench.cpp"
Cohesion: 0.09
Nodes (48): argmax(), build_json(), ostream, size_t, string, timespec, uint32_t, vector (+40 more)

### Community 123 - "Q8_1Block"
Cohesion: 0.08
Nodes (32): DeviceShapeData, device_input_fp16, device_input_q8, device_output, device_weights, input_fp16, input_q8, oracle_fp16 (+24 more)

### Community 124 - "External gfx906 Reference Baseline"
Cohesion: 0.22
Nodes (8): Baseline status, Checkout, External gfx906 Reference Baseline, Initial baseline, MI50 build starting point, Option validation on the available host, Pin, Toolchain preflight record

### Community 125 - "q4_q8_gemv_bench.cpp"
Cohesion: 0.16
Nodes (29): allocate_shape(), ostream, string, vector, escape(), free_shape(), launch_selected_gemv(), main() (+21 more)

### Community 126 - "Qwen3GpuDecodeWorkspace"
Cohesion: 0.05
Nodes (38): Qwen3GpuDecodeWorkspace, argmax_token, attention, attention_projected, attn_norm, attn_rms, embedding, ffn_input (+30 more)

### Community 127 - "EXP-0086 — M6-A27.6 Qwen3.8 P2 L0–L2 operation trace"
Cohesion: 0.25
Nodes (7): Decision, EXP-0086 — M6-A27.6 Qwen3.8 P2 L0–L2 operation trace, Follow-up, Interpretation, Method, Question, Results

### Community 128 - "fp16_gemv_reduction_diag.cpp"
Cohesion: 0.18
Nodes (16): string, vector, main(), median(), nonnegative(), Options, device, iterations (+8 more)

### Community 129 - "vector"
Cohesion: 0.10
Nodes (18): GemvKernelResources, local_bytes, max_threads_per_block, registers, shared_bytes, GemvShape, id, k (+10 more)

### Community 130 - "m6a15_qwen35_hybrid_block_audit.cpp"
Cohesion: 0.17
Nodes (26): optional, cache_fingerprint(), History, initializer_list, size_t, span, State, string (+18 more)

### Community 131 - "ProfileScope"
Cohesion: 0.13
Nodes (15): hipEvent_t, Qwen3ProfileCategory, ProfileScope, boundary_stage_, bytes_, category_, copy_, dispatches_ (+7 more)

### Community 132 - "qwen3_fast_decode_bench.cpp"
Cohesion: 0.10
Nodes (43): argmax(), build_json(), ostream, size_t, string, timespec, uint32_t, vector (+35 more)

### Community 133 - "EXP-0021 — M5-C6c coalesced KV-cache writes"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment and workload, EXP-0021 — M5-C6c coalesced KV-cache writes, Follow-up, Hypothesis, Performance (+1 more)

### Community 134 - "Q6KHostBlock"
Cohesion: 0.14
Nodes (16): int16_t, int8_t, uint16_t, uint8_t, Q4_0HostBlock, d_bits, qs, Q6KHostBlock (+8 more)

### Community 135 - "qwen3_primitives.cpp"
Cohesion: 0.24
Nodes (21): byte, int8_t, size_t, span, uint16_t, vector, fp16_bits_to_float(), load_half() (+13 more)

### Community 136 - "Qwen3GpuProfile"
Cohesion: 0.07
Nodes (25): qwen3_profile_category_count, Qwen3GpuProfile, boundary_bytes, boundary_dispatches, boundary_gpu_ms, copy_bytes, copy_ms, deferred_timing (+17 more)

### Community 138 - "M3 Minimal Qwen3-8B Runtime Scaffold"
Cohesion: 0.25
Nodes (7): GPU ownership and plan, M3 Minimal Qwen3-8B Runtime Scaffold, Parser boundary, Static projection kernel selection, Supported artifact, Validated configuration, Validation command

### Community 139 - "run_sequence"
Cohesion: 0.24
Nodes (19): snapshot_keys, snapshot_values, attention_contract(), cache_corruption_test(), cache_slots_preserved(), checkpoints(), compare_trace(), size_t (+11 more)

### Community 140 - "Qwen3LayerTrace"
Cohesion: 0.06
Nodes (31): Qwen3LayerTrace, attention_output, attention_probabilities, attention_scores, attn_norm, attn_rms, embedding, ffn_input (+23 more)

### Community 141 - "EXP-0074 — M6-A26.4 L30 production operand attribution"
Cohesion: 0.18
Nodes (10): Baseline and method, Decision, Environment and command, EXP-0074 — M6-A26.4 L30 production operand attribution, Follow-up, Interpretation, One-at-a-time GPU substitutions, Production operand comparison (+2 more)

### Community 142 - "EXP-0041 — M5-C15 optimization closure and parity decision gate"
Cohesion: 0.15
Nodes (12): 1. Question, 2. Authoritative production result, 3. M5 optimization record, 4. Experimentally eliminated explanations, 5. M5 result, 6. Architectural decision gate, 7. Decision, Accepted (+4 more)

### Community 143 - "m6a263_qwen35_recurrent_contract.cpp"
Cohesion: 0.21
Nodes (15): apply_external(), checkpoint(), compare(), size_t, span, string_view, vector, DeviceBuffer (+7 more)

### Community 144 - "EXP-0100 — M6-B7 Q5_K paired-nibble decoding"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Commands, Correctness and resources, Decision, Environment, EXP-0100 — M6-B7 Q5_K paired-nibble decoding, Follow-up (+3 more)

### Community 145 - "qwen3_layer_host_impl"
Cohesion: 0.23
Nodes (23): add_in_place(), size_t, span, uint32_t, vector, execute_qwen3_decode_host(), execute_qwen3_forward_host(), execute_qwen3_layer0_host() (+15 more)

### Community 146 - "M4-C1 — Deterministic first generated token"
Cohesion: 0.12
Nodes (13): Acceptance, Decision, M4-C2 — Short deterministic greedy decode sequence, Next slice, Pinned sequence, Acceptance, Decode contract, Deterministic fixture (+5 more)

### Community 147 - "Metrics"
Cohesion: 0.13
Nodes (26): Checkpoint, abs_tolerance, actual, file_index, name, rel_tolerance, compare(), size_t (+18 more)

### Community 149 - "EXP-0110 — M6-B17 Q5_K×Q8_1 MMVQ recurrent projection"
Cohesion: 0.17
Nodes (11): Baseline and candidate, Commands, Correctness, Decision, Environment, EXP-0110 — M6-B17 Q5_K×Q8_1 MMVQ recurrent projection, Follow-up, Hypothesis (+3 more)

### Community 150 - "M4-B — Full Qwen3 single-token forward"
Cohesion: 0.07
Nodes (29): Acceptance target, Current evidence, Implemented slice, Independent reference, M4-B10 layer-35 Gate/Up projection isolation, M4-B11 Q8 identity and CPU accumulation contract, M4-B12 pre-FFN residual and RMSNorm isolation, M4-B13 attention RMSNorm, V, and position-zero GQA isolation (+21 more)

### Community 152 - "gguf.cpp"
Cohesion: 0.08
Nodes (56): GgufError, metadata, GgufValue, value, GgufScalar, unordered_map, runtime_error, align_up() (+48 more)

### Community 154 - "EXP-0026 — M5-C8c Down long-K bottleneck attribution"
Cohesion: 0.17
Nodes (11): 10. Follow-up, 1. Question, 2. Hypothesis, 3. Motivation and prior evidence, 4. Method, 5. Results, 6. Static gfx906 evidence, 7. Interpretation (+3 more)

### Community 158 - "Q8BoundaryDiff"
Cohesion: 0.13
Nodes (15): int16_t, int8_t, uint16_t, Q8BoundaryDiff, different_lane_values, different_scale_blocks, different_sum_blocks, first_block (+7 more)

### Community 160 - "EXP-0046 — M6-A4 Qwen3.8-27B full-attention layer"
Cohesion: 0.20
Nodes (9): Candidate, Commands, Correctness gates, Decision, EXP-0046 — M6-A4 Qwen3.8-27B full-attention layer, Follow-up, Question, Reference and baseline (+1 more)

### Community 161 - "memory_stream_bench.cpp"
Cohesion: 0.17
Nodes (16): size_t, string, vector, escape(), main(), median(), Options, bytes (+8 more)

### Community 162 - "EXP-0042 — M6-A0 Qwen3.8-27B GGUF and architecture audit"
Cohesion: 0.11
Nodes (18): 10. Files changed, 11. Checks run, 12. Conclusion, 1. Question, 2. Local artifacts, 3. GGUF metadata, 4. Layer pattern, 5. Tensor inventory (+10 more)

### Community 163 - "span"
Cohesion: 0.22
Nodes (16): span, reset, Qwen3DecodeCache::reset(), cache_contract_test(), checkpoint_tolerance(), checkpoints(), size_t, span (+8 more)

### Community 164 - "qwen3_position_audit.cpp"
Cohesion: 0.07
Nodes (46): category_index(), array, ostream, qwen3_profile_category_count, Qwen3ProfileCategory, size_t, string, timespec (+38 more)

### Community 165 - "hip_smoke_bench.cpp"
Cohesion: 0.20
Nodes (15): size_t, string, json_escape(), main(), Options, device, elements, iterations (+7 more)

### Community 166 - "model_plan.cpp"
Cohesion: 0.13
Nodes (21): GpuWeightArena, allocate, release, upload, Q4GemvKernel, align_up(), checked_add(), byte (+13 more)

### Community 167 - "path_"
Cohesion: 0.40
Nodes (15): path_, argmax(), compare(), compare_checkpoint(), vector, fp16_round_trip(), main(), print_composition_delta() (+7 more)

### Community 168 - "size_t"
Cohesion: 0.16
Nodes (9): size_t, string, DeviceBytes, bytes_, data_, RecurrentTrace, kStateElements, layer (+1 more)

### Community 169 - "M5-A — Reproducible MI50 inference baseline"
Cohesion: 0.20
Nodes (9): Decision, Environment, Interpretation, M5-A — Reproducible MI50 inference baseline, Question, Reproduction, Result, Scope (+1 more)

### Community 170 - "M4-C3 — Text-facing greedy generation"
Cohesion: 0.33
Nodes (5): CLI, Decision, M4-C3 — Text-facing greedy generation, Pinned physical acceptance, Tokenizer contract

### Community 172 - "Qwen3GpuPlan"
Cohesion: 0.09
Nodes (34): GgufTensorType, size_t, string, uint64_t, vector, PlannedTensor, bytes, dimensions (+26 more)

### Community 173 - "qwen3_gpu_primitives.hpp"
Cohesion: 0.12
Nodes (17): uint8_t, Q4KDeviceBlock, d, dmin, qs, scales, Q5KDeviceBlock, d (+9 more)

### Community 174 - "qwen3_attention_ab_bench.cpp"
Cohesion: 0.08
Nodes (47): argmax(), ostream, size_t, string, timespec, uint32_t, vector, elapsed_ms() (+39 more)

### Community 175 - "EXP-0010 — Qwen3-8B steady-state decode profile"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0010 — Qwen3-8B steady-state decode profile, Follow-up (+4 more)

### Community 176 - "EXP-0017 — M5-C5a persistent Qwen3 decode workspace"
Cohesion: 0.18
Nodes (10): Candidate, Correctness, Decision, Environment and workload, EXP-0017 — M5-C5a persistent Qwen3 decode workspace, Follow-up, Hypothesis, Interpretation (+2 more)

### Community 178 - "EXP-0016 — M5-C4 post-attention MI50 baseline"
Cohesion: 0.20
Nodes (9): Decision, EXP-0016 — M5-C4 post-attention MI50 baseline, Follow-up, Hardware validity, Interpretation, Question, Results, Scope (+1 more)

### Community 180 - "EXP-0012 — Qwen3-8B Q4_0 MI50 comparison"
Cohesion: 0.17
Nodes (11): Closest raw-token continuation controls, Decision, Environment, EXP-0012 — Qwen3-8B Q4_0 MI50 comparison, Follow-up, Growing-context continuation, Hypothesis, Interpretation (+3 more)

### Community 181 - "EXP-0018 — M5-C5b resident normalization weights"
Cohesion: 0.17
Nodes (11): Candidate, Correctness, Decision, Environment and workload, EXP-0018 — M5-C5b resident normalization weights, Follow-up, Hypothesis, Interpretation (+3 more)

### Community 182 - "EXP-0014 — Cooperative cached-attention execution"
Cohesion: 0.15
Nodes (12): Baseline, Candidate, Correctness, Decision, End-to-end trace-free A/B, Environment, EXP-0014 — Cooperative cached-attention execution, Follow-up (+4 more)

### Community 183 - "EXP-0013 — Qwen3 position-scaled execution audit"
Cohesion: 0.20
Nodes (9): Decision, Environment, EXP-0013 — Qwen3 position-scaled execution audit, Follow-up, Hypothesis, Implementation, Interpretation, Results (+1 more)

### Community 185 - "GgufFile"
Cohesion: 0.09
Nodes (23): GgufFile, file_descriptor_, mapping_, metadata_array_is_string, metadata_array_size, metadata_float, metadata_string, metadata_unsigned (+15 more)

### Community 186 - "LayerPathCapture"
Cohesion: 0.20
Nodes (10): LayerPathCapture, attention_residual, ffn_output, gated, input, layer_output, normalized, post_normalized (+2 more)

### Community 187 - "EXP-0011 — Trace-free Qwen3-8B decode control"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0011 — Trace-free Qwen3-8B decode control, Follow-up (+4 more)

### Community 188 - "EXP-0099 — M6-B6 Q5_K subgroup-structured dot loop"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Commands, Correctness and resources, Decision, Environment, EXP-0099 — M6-B6 Q5_K subgroup-structured dot loop, Follow-up (+3 more)

### Community 189 - "EXP-0022 — M5-C6d GPU-side greedy argmax"
Cohesion: 0.18
Nodes (10): Candidate, Correctness, Decision, Environment and workload, EXP-0022 — M5-C6d GPU-side greedy argmax, Follow-up, Hypothesis, Performance (+2 more)

### Community 190 - "FullAttentionLayer"
Cohesion: 0.04
Nodes (43): FullAttentionLayer, attention, batch_head_rms, d_attn_norm, d_ffn_down, d_ffn_gate, d_ffn_up, d_k (+35 more)

### Community 191 - "EXP-0020 — M5-C6b direct layer-output handoff"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment and workload, EXP-0020 — M5-C6b direct layer-output handoff, Follow-up, Hypothesis, Performance (+1 more)

### Community 192 - "EXP-0051 — M6-B1 Qwen3.8-27B MIInfer GPU profile readiness"
Cohesion: 0.18
Nodes (10): 1. Question, 2. Hypothesis, 3. Artifact, 4. Checks, 5. Results, 6. Interpretation, 7. Decision, 8. Next task (+2 more)

### Community 195 - "EXP-0015 — M5-C3 interleaved cached-attention A/B characterization"
Cohesion: 0.18
Nodes (10): Correctness, Decision, EXP-0015 — M5-C3 interleaved cached-attention A/B characterization, Follow-up, Hypothesis, Implementation, Interpretation, Question (+2 more)

### Community 197 - "EXP-0107 — M6-B14 Q6_K MMVQ-style Q8_1 LM-head candidate"
Cohesion: 0.18
Nodes (10): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0107 — M6-B14 Q6_K MMVQ-style Q8_1 LM-head candidate, Follow-up (+2 more)

### Community 198 - "EXP-0069 — M6-A25 Qwen3.8 sixteen-layer stateful GPU prefix"
Cohesion: 0.18
Nodes (10): Candidate, Checks, Decision, EXP-0069 — M6-A25 Qwen3.8 sixteen-layer stateful GPU prefix, Follow-up, Performance and memory accounting, Question, Reference and workload (+2 more)

### Community 199 - "EXP-0024 — M5-C8a FFN projection shape characterization"
Cohesion: 0.20
Nodes (9): Available geometry controls, Correctness, Current production-like geometry, Decision, Environment and workload, EXP-0024 — M5-C8a FFN projection shape characterization, Follow-up, Hypothesis (+1 more)

### Community 201 - "EXP-0034 — M5-C11b exact-shape FFN GEMV differential"
Cohesion: 0.29
Nodes (7): 1. Question, 2. Method and clock qualification, 3. Exact-shape direct comparison, 4. Kernel-structure findings, 5. Decision, 6. Follow-up, EXP-0034 — M5-C11b exact-shape FFN GEMV differential

### Community 202 - "EXP-0019 — M5-C6a execution-overhead attribution"
Cohesion: 0.22
Nodes (8): Decision, EXP-0019 — M5-C6a execution-overhead attribution, Follow-up, Interpretation, Other fixed dispatch families, Per-token attribution, Question, Workload and source

### Community 203 - "EXP-0027 — M5-C9a production FFN attribution"
Cohesion: 0.20
Nodes (9): 1. Question, 2. Method, 3. Full-token attribution, 4. FFN stage attribution, 5. Interpretation, 6. Correctness, 7. Decision, 8. Follow-up (+1 more)

### Community 204 - "EXP-0037 — M5-C13a fixed-cost floor profile"
Cohesion: 0.25
Nodes (8): 1. Question, 2. Method and environment, 3. Position results, 4. P1 production attribution, 5. Interpretation, 6. Decision, 7. Follow-up — M5-C13b, EXP-0037 — M5-C13a fixed-cost floor profile

### Community 205 - "EXP-0023 — M5-C7 post-copy-cleanup decode profile"
Cohesion: 0.22
Nodes (8): Correctness, Decision, Environment and workload, EXP-0023 — M5-C7 post-copy-cleanup decode profile, Goal, Interpretation, Operation-family profile, Results

### Community 206 - "EXP-0032 — M5-C10c FFN normalization-to-shared-Q8 fusion"
Cohesion: 0.22
Nodes (8): 1. Hypothesis, 2. Candidate, 3. Environment and benchmark, 4. Correctness, 5. Results, 6. Decision, 7. Artifacts, EXP-0032 — M5-C10c FFN normalization-to-shared-Q8 fusion

### Community 207 - "qwen3_layer6_external_test.cpp"
Cohesion: 0.12
Nodes (22): Checkpoint, file, miinfer, name, tolerance, compare_authority(), compare_host_gpu(), size_t (+14 more)

### Community 209 - "m6a14_qwen35_state_audit.cpp"
Cohesion: 0.13
Nodes (34): kRecurrentLayers, align_up(), cache_fingerprint(), array, History, size_t, span, State (+26 more)

### Community 210 - "EXP-0028 — M5-C9b fused SwiGLU to Q8 quantization"
Cohesion: 0.22
Nodes (8): 1. Question, 2. Candidate, 3. Correctness results, 4. Performance results, 5. Interpretation, 6. Decision, 7. Follow-up, EXP-0028 — M5-C9b fused SwiGLU to Q8 quantization

### Community 211 - "EXP-0071 — M6-A26.1 Qwen3.8 L30 state localization"
Cohesion: 0.17
Nodes (11): Adjacent-layer P64 state entries, Baseline and change, Decision, Environment and command, EXP-0071 — M6-A26.1 Qwen3.8 L30 state localization, Follow-up, L30/P64 boundary trace, L30 state-entry results (+3 more)

### Community 212 - "EXP-0031 — M5-C10b normalization/conversion boundary attribution"
Cohesion: 0.22
Nodes (8): 1. Question, 2. Method, 3. P64 production result, 4. Boundary map, 5. Interpretation, 6. Decision, 7. Follow-up, EXP-0031 — M5-C10b normalization/conversion boundary attribution

### Community 213 - "EXP-0035 — M5-C12a stable-peak non-FFN profile"
Cohesion: 0.20
Nodes (9): 1. Question, 2. Method and environment, 3. Results, 4. Interpretation, 5. Decision, 6. Follow-up, EXP-0035 — M5-C12a stable-peak non-FFN profile, External shape control (+1 more)

### Community 214 - "EXP-0106 — M6-B13 Q6_K × Q8_1 LM-head compatibility path"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Decision, Environment, EXP-0106 — M6-B13 Q6_K × Q8_1 LM-head compatibility path, Follow-up, Interpretation, Question (+2 more)

### Community 215 - "EXP-0025 — M5-C8b Down four-Wave64 GEMV candidate"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment and method, EXP-0025 — M5-C8b Down four-Wave64 GEMV candidate, Follow-up, Hypothesis, Interpretation (+1 more)

### Community 216 - "EXP-0057 — M6-A13 Qwen3.8-27B full-attention GPU layer"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Checks, Correctness, Decision, EXP-0057 — M6-A13 Qwen3.8-27B full-attention GPU layer, Follow-up, Question (+2 more)

### Community 217 - "m6a3_qwen35_layer.cpp"
Cohesion: 0.20
Nodes (29): checkpoint(), compare(), conv_output(), array, kChannels, size_t, span, string_view (+21 more)

### Community 218 - "EXP-0029 — M5-C9c Gate/Up activation-Q8 reuse"
Cohesion: 0.20
Nodes (9): 1. Question, 2. Candidate, 3. Verification design, 4. Acceptance gates, 5. Benchmark commands, 6. Performance results, 7. Decision, 8. Follow-up (+1 more)

### Community 220 - "EXP-0119 — M6-B27 batched full-attention head RMS normalization"
Cohesion: 0.20
Nodes (9): Candidate and control, Correctness and resources, Decision, Environment, EXP-0119 — M6-B27 batched full-attention head RMS normalization, Follow-up, Interpretation, Question (+1 more)

### Community 221 - "EXP-0033 — M5-C11a production and llama.cpp differential baseline"
Cohesion: 0.20
Nodes (9): 1. Hypothesis, 2. Scope, 3. Environment, 4. MIInfer production baseline, 5. MIInfer position audit, 6. Fresh llama.cpp control, 7. Interpretation and decision, 8. Artifacts and follow-up (+1 more)

### Community 222 - "EXP-0030 — M5-C10a refreshed P64 production profile"
Cohesion: 0.25
Nodes (7): 1. Question, 2. Method, 3. P64 result, 4. Interpretation, 5. Decision, 6. Follow-up, EXP-0030 — M5-C10a refreshed P64 production profile

### Community 223 - "Q5K"
Cohesion: 0.09
Nodes (24): int16_t, int8_t, uint16_t, uint8_t, Q4K, d, dmin, qs (+16 more)

### Community 224 - "EXP-0039 — M5-C13c fixed-floor contract map"
Cohesion: 0.29
Nodes (7): 1. Question, 2. Method, 3. Fixed-floor contract map, 4. Eligible rows and ranking, 5. Decision, 6. Follow-up, EXP-0039 — M5-C13c fixed-floor contract map

### Community 225 - "EXP-0053 — M6-A9 Qwen3.8-27B LM-head GPU projection"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Checks, Decision, EXP-0053 — M6-A9 Qwen3.8-27B LM-head GPU projection, Follow-up, Question, Result (+1 more)

### Community 226 - "EXP-0036 — M5-C12b cooperative attention scaling"
Cohesion: 0.25
Nodes (7): 1. Question, 2. Baseline and method, 3. Production scaling results, 4. Candidate correctness result, 5. Interpretation, 6. Decision, EXP-0036 — M5-C12b cooperative attention scaling

### Community 227 - "EXP-0084 — M6-A27.4 Qwen3.8 full-model observable contract adjudication"
Cohesion: 0.18
Nodes (10): Correctness, Decision, EXP-0084 — M6-A27.4 Qwen3.8 full-model observable contract adjudication, Follow-up, Hypothesis, Method, Observable checkpoints, Question (+2 more)

### Community 229 - "EXP-0038 — M5-C13b LM-head contract audit"
Cohesion: 0.25
Nodes (7): 1. Question, 2. MIInfer production path, 3. External contract audit, 4. Whole-token context control, 5. Decision, 6. Follow-up, EXP-0038 — M5-C13b LM-head contract audit

### Community 230 - "EXP-0054 — M6-A10 Qwen3.8-27B Q4_K projection"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Checks, Decision, EXP-0054 — M6-A10 Qwen3.8-27B Q4_K projection, Follow-up, Question, Result (+1 more)

### Community 232 - "EXP-0063 — M6-A19 Qwen3.8-27B convolution GPU path"
Cohesion: 0.18
Nodes (10): Artifact and reference, Candidate, Checks, Command, Decision, EXP-0063 — M6-A19 Qwen3.8-27B convolution GPU path, Follow-up, Question (+2 more)

### Community 233 - "EXP-0040 — M5-C14a fixed-floor execution map"
Cohesion: 0.25
Nodes (6): 1. Question, 2. Method and limits, 3. Layer/token execution map, 5. Fixed-floor budget, 6. Decision, EXP-0040 — M5-C14a fixed-floor execution map

### Community 234 - "EXP-0043 — M6-A1 Qwen3.8-27B external reference fixture"
Cohesion: 0.22
Nodes (8): Correctness/checks, Decision, EXP-0043 — M6-A1 Qwen3.8-27B external reference fixture, Fixture contents, M6-A2 next task, Model selected, Question, Reference contract

### Community 236 - "4. Difference inventory"
Cohesion: 0.40
Nodes (5): 4. Difference inventory, A — implementation difference, same contract, B — representation difference, C — work-elimination difference, D — scheduling/fusion difference

### Community 237 - "EXP-0044 — M6-A2 Qwen3.8 projection/kernel compatibility audit"
Cohesion: 0.22
Nodes (8): Artifact and method, Compatibility map, Decision, Existing MIInfer contracts, EXP-0044 — M6-A2 Qwen3.8 projection/kernel compatibility audit, Important findings, M6-A3 next task, Question

### Community 238 - "EXP-0062 — M6-A18 Qwen3.8-27B DeltaNet GPU state core"
Cohesion: 0.18
Nodes (10): Baseline and artifact, Candidate, Checks, Command, Decision, EXP-0062 — M6-A18 Qwen3.8-27B DeltaNet GPU state core, Follow-up, Question (+2 more)

### Community 239 - "EXP-0045 — M6-A3 Qwen3.8-27B single DeltaNet layer"
Cohesion: 0.18
Nodes (10): Candidate, Commands, Decision, EXP-0045 — M6-A3 Qwen3.8-27B single DeltaNet layer, Follow-up, Hypothesis, Interpretation, Question (+2 more)

### Community 240 - "UpdateProvenance"
Cohesion: 0.13
Nodes (15): UpdateProvenance, beta, candidate, column, decay, decayed, delta, head (+7 more)

### Community 241 - "EXP-0047 — M6-A5 Qwen3.8-27B four-layer hybrid block"
Cohesion: 0.20
Nodes (9): Candidate, Command, Correctness gates, Decision, EXP-0047 — M6-A5 Qwen3.8-27B four-layer hybrid block, Follow-up, Question, Reference and baseline (+1 more)

### Community 242 - "EXP-0048 — M6-A6 Qwen3.8-27B full 64-layer forward"
Cohesion: 0.22
Nodes (8): Baseline and candidate, Command, Correctness contract, Decision, EXP-0048 — M6-A6 Qwen3.8-27B full 64-layer forward, Follow-up, Question, Result

### Community 243 - "string"
Cohesion: 0.12
Nodes (13): DeviceInfo, architecture, index, name, total_vram_bytes, size_t, string, ostream (+5 more)

### Community 244 - "EXP-0050 — M6-B0 Qwen3.8-27B llama.cpp MI50 baseline"
Cohesion: 0.17
Nodes (11): Baseline, Combined context controls, Commands, Decision, EXP-0050 — M6-B0 Qwen3.8-27B llama.cpp MI50 baseline, Follow-up, Interpretation, Model and environment (+3 more)

### Community 245 - "array"
Cohesion: 0.16
Nodes (22): tensor, array, rms_rows(), array, kChannels, size_t, span, vector (+14 more)

### Community 246 - "EXP-0056 — M6-A12 Qwen3.8-27B attention projections"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Checks, Decision, EXP-0056 — M6-A12 Qwen3.8-27B attention projections, Follow-up, Question, Result (+1 more)

### Community 247 - "EXP-0049 — M6-A7 Qwen3.8-27B stateful generation"
Cohesion: 0.18
Nodes (10): Baseline and candidate, Command, Correctness contract, Decision, Environment, EXP-0049 — M6-A7 Qwen3.8-27B stateful generation, Follow-up, Hypothesis (+2 more)

### Community 248 - "Qwen35Config"
Cohesion: 0.06
Nodes (42): byte, GgufTensorType, shared_ptr, size_t, string, uint32_t, Qwen35Config, attention_heads (+34 more)

### Community 249 - "qwen3_layer0_gpu_test.cpp"
Cohesion: 0.26
Nodes (16): abs_tolerance(), checkpoints(), compare(), size_t, string, vector, main(), Metrics (+8 more)

### Community 250 - "Metrics"
Cohesion: 0.22
Nodes (9): Metrics, actual_at_max, expected_at_max, finite, max_abs, max_index, max_rel, mean_abs (+1 more)

### Community 251 - "run-m6b0-llama-baseline.sh"
Cohesion: 0.83
Nodes (3): cleanup(), run-m6b0-llama-baseline.sh script, stop_telemetry()

### Community 252 - "RmsVariant"
Cohesion: 0.15
Nodes (12): Checkpoint, file, miinfer, name, tolerance, RmsVariant, inverse, mean (+4 more)

### Community 253 - "EXP-0058 — M6-A14 Qwen3.8-27B state fingerprints and reset audit"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Checks, Decision, EXP-0058 — M6-A14 Qwen3.8-27B state fingerprints and reset audit, Follow-up, Question, Results (+1 more)

### Community 254 - "EXP-0125 — M6-B33 post-B32 production profile"
Cohesion: 0.22
Nodes (8): Decision, Environment, EXP-0125 — M6-B33 post-B32 production profile, Follow-up, Interpretation, Method, Question, Results

### Community 255 - "EXP-0059 — M6-A15 Qwen3.8-27B layers 0–3 hybrid-block audit"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Checks, Command, Correctness contract, Decision, EXP-0059 — M6-A15 Qwen3.8-27B layers 0–3 hybrid-block audit, Follow-up (+3 more)

### Community 256 - "EXP-0093 — M6-B1 Qwen3.8-27B native GPU generation baseline"
Cohesion: 0.18
Nodes (10): Baseline and candidate, Commands, Correctness and resource checks, Decision, Environment, EXP-0093 — M6-B1 Qwen3.8-27B native GPU generation baseline, Follow-up, Interpretation (+2 more)

### Community 257 - "EXP-0055 — M6-A11 Qwen3.8-27B composed attention prefix"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Checks, Decision, EXP-0055 — M6-A11 Qwen3.8-27B composed attention prefix, Follow-up, Question, Result (+1 more)

### Community 258 - "qwen3_trace_compare.cpp"
Cohesion: 0.20
Nodes (17): argmax(), compare(), size_t, vector, main(), Metrics, first_value, max_abs (+9 more)

### Community 259 - "m6a13_qwen35_full_attention_layer.cpp"
Cohesion: 0.36
Nodes (10): check(), checkpoint(), copy_to_host(), size_t, string_view, vector, DeviceBuffer, main() (+2 more)

### Community 260 - "EXP-0079 — M6-A26.9 Qwen3.8 external recurrent-state contract"
Cohesion: 0.20
Nodes (9): Contract decision, Decision, Evidence, EXP-0079 — M6-A26.9 Qwen3.8 external recurrent-state contract, Follow-up, Harness change, Question, Status (+1 more)

### Community 261 - "EXP-0052 — M6-A8 Qwen3.8-27B GPU foundation"
Cohesion: 0.20
Nodes (9): 1. Question, 2. Hypothesis, 3. Change, 4. Artifact and fixture, 5. Result, 6. Checks, 7. Decision, 8. Next task (+1 more)

### Community 262 - "EXP-0089 — M6-A27.9 Qwen3.8 L0 Q5_K block contract"
Cohesion: 0.20
Nodes (9): Baseline, Change, Decision, EXP-0089 — M6-A27.9 Qwen3.8 L0 Q5_K block contract, Follow-up, Method, Question, Results after fix (+1 more)

### Community 263 - "EXP-0073 — M6-A26.3 Qwen3.8 recurrent-state contract adjudication"
Cohesion: 0.20
Nodes (9): Decision, Environment and command, EXP-0073 — M6-A26.3 Qwen3.8 recurrent-state contract adjudication, Follow-up, Interpretation, Question, Reference capture, Results (+1 more)

### Community 264 - "EXP-0060 — M6-A16 Qwen3.8-27B layers 4–7 hybrid-block audit"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Checks, Command, Correctness contract, Decision, EXP-0060 — M6-A16 Qwen3.8-27B layers 4–7 hybrid-block audit, Follow-up (+3 more)

### Community 265 - "EXP-0090 — M6-A27.9 full observable-contract retest"
Cohesion: 0.22
Nodes (8): Decision, EXP-0090 — M6-A27.9 full observable-contract retest, Follow-up, Key before/after results, Method, Other checks, Question, Teacher-forced trajectory

### Community 266 - "EXP-0061 — M6-A17 Qwen3.8-27B composition ladder"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Checks, Command, Correctness contract, Decision, EXP-0061 — M6-A17 Qwen3.8-27B composition ladder, Follow-up (+3 more)

### Community 267 - "AttentionPathReplay"
Cohesion: 0.33
Nodes (6): AttentionPathReplay, attention_output, ffn_input, ffn_norm, layer_output, v

### Community 268 - "FfnTailReplay"
Cohesion: 0.33
Nodes (6): FfnTailReplay, down, gate, layer_output, swiglu, up

### Community 269 - "EXP-0096 — M6-B3 Q4_K metadata staging"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Commands, Correctness and resources, Decision, Environment, EXP-0096 — M6-B3 Q4_K metadata staging, Follow-up (+3 more)

### Community 270 - "qwen3_gpu_layer.cpp"
Cohesion: 0.08
Nodes (46): AttentionKernel, Function, capture(), capture_qwen3_head_norm(), copy_to_host(), Qwen3BoundaryProfileStage, Qwen3FfnProfileStage, size_t (+38 more)

### Community 271 - "m6a21_qwen35_gpu_hybrid_block.cpp"
Cohesion: 0.10
Nodes (39): ByteMismatch, count, first, compare_bytes(), cosine_similarity(), byte, GgufTensorType, initializer_list (+31 more)

### Community 272 - "EXP-0066 — M6-A22 Qwen3.8-27B GPU hybrid position audit"
Cohesion: 0.17
Nodes (11): Artifact and reference, Candidate, Checks, Command, Decision, EXP-0066 — M6-A22 Qwen3.8-27B GPU hybrid position audit, Follow-up, Interpretation (+3 more)

### Community 273 - "m6a18_qwen35_deltanet_state_gpu.cpp"
Cohesion: 0.26
Nodes (12): size_t, span, T, vector, DeviceBuffer, data_, logical_state(), main() (+4 more)

### Community 274 - "qwen3_gpu_layer.hpp"
Cohesion: 0.12
Nodes (21): vector, Qwen3DownProjectionContractTrace, current_s_correction, direct_signed_oracle, exact_sum_correction, Qwen3FfnProbeTrace, ffn_output, gate (+13 more)

### Community 275 - "m6a10_qwen35_q4k_projection.cpp"
Cohesion: 0.39
Nodes (6): size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 276 - "m6a8_qwen35_gpu_foundation.cpp"
Cohesion: 0.39
Nodes (6): size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 277 - "EXP-0064 — M6-A20 Qwen3.8-27B recurrent layer on GPU"
Cohesion: 0.18
Nodes (10): Artifact and reference, Candidate, Checks, Command, Decision, EXP-0064 — M6-A20 Qwen3.8-27B recurrent layer on GPU, Follow-up, Question (+2 more)

### Community 278 - "EXP-0072 — M6-A26.2 Qwen3.8 L30 update provenance"
Cohesion: 0.18
Nodes (10): Decision, Diagnostic, Environment and command, EXP-0072 — M6-A26.2 Qwen3.8 L30 update provenance, Follow-up, Interpretation, Moving maximum and tracked-index results, Question (+2 more)

### Community 279 - "EXP-0103 — M6-B10 recurrent Q8_K input reuse"
Cohesion: 0.25
Nodes (7): Candidate, Decision, Environment, EXP-0103 — M6-B10 recurrent Q8_K input reuse, Follow-up, Question, Results

### Community 280 - "qwen3_swiglu_q8_bench.cpp"
Cohesion: 0.39
Nodes (7): vector, main(), measure(), Stats, mean_us, median_us, summarize()

### Community 281 - "EXP-0065 — M6-A21 Qwen3.8-27B GPU hybrid block"
Cohesion: 0.18
Nodes (10): Artifact and reference, Candidate, Checks, Command, Decision, EXP-0065 — M6-A21 Qwen3.8-27B GPU hybrid block, Follow-up, Question (+2 more)

### Community 282 - "Qwen3Model"
Cohesion: 0.14
Nodes (13): shared_ptr, size_t, string, vector, Qwen3Model, artifact_path_, config_, final_norm_ (+5 more)

### Community 283 - "EXP-0068 — M6-A24 Qwen3.8 eight-layer stateful GPU prefix"
Cohesion: 0.18
Nodes (10): Candidate, Checks, Correctness contract, Decision, EXP-0068 — M6-A24 Qwen3.8 eight-layer stateful GPU prefix, Follow-up, Performance and memory accounting, Question (+2 more)

### Community 284 - "11. joe2gaan/localaiservers"
Cohesion: 0.50
Nodes (4): 11. joe2gaan/localaiservers, Important dot-product lesson, MIInfer implication, Role

### Community 285 - "EXP-0080 — M6-A27 Qwen3.8 sixty-four-layer GPU composition"
Cohesion: 0.25
Nodes (7): Candidate, Decision, Environment and command, EXP-0080 — M6-A27 Qwen3.8 sixty-four-layer GPU composition, Follow-up, Question, Results

### Community 286 - "EXP-0085 — M6-A27.5 Qwen3.8 P2 drift localization"
Cohesion: 0.18
Nodes (10): Decision, EXP-0085 — M6-A27.5 Qwen3.8 P2 drift localization, Follow-up, Hypothesis, Interpretation, Method, Observable consequence, P2 layer-output error scan (+2 more)

### Community 287 - "EXP-0070 — M6-A26 Qwen3.8 thirty-two-layer stateful GPU prefix"
Cohesion: 0.20
Nodes (9): Candidate, Decision, Environment and command, EXP-0070 — M6-A26 Qwen3.8 thirty-two-layer stateful GPU prefix, Follow-up, Interpretation, Question, Results (+1 more)

### Community 288 - "m6a9_qwen35_lm_head.cpp"
Cohesion: 0.40
Nodes (7): argmax(), size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 289 - "EXP-0124 — M6-B32 transposed recurrent no-decay store"
Cohesion: 0.17
Nodes (11): Baseline and candidate, Benchmark, Commands, Correctness, Decision, Environment, EXP-0124 — M6-B32 transposed recurrent no-decay store, Follow-up (+3 more)

### Community 290 - "model_loader_test.cpp"
Cohesion: 0.35
Nodes (11): append_string(), append_u32(), append_u64(), string, uint32_t, uint64_t, uint8_t, vector (+3 more)

### Community 291 - "qwen3_cached_attention_determinism_gpu_test.cpp"
Cohesion: 0.33
Nodes (9): size_t, T, vector, DeviceBuffer, data_, download(), main(), same_bytes() (+1 more)

### Community 292 - "EXP-0075 — M6-A26.5 L30 K-path provenance"
Cohesion: 0.20
Nodes (9): Decision, Environment and command, EXP-0075 — M6-A26.5 L30 K-path provenance, Follow-up, Interpretation, Method, Question, Results — L30 P19 (+1 more)

### Community 293 - "m6a12_qwen35_attention_projections.cpp"
Cohesion: 0.39
Nodes (6): size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 294 - "EXP-0076 — M6-A26.6 L29 output provenance"
Cohesion: 0.20
Nodes (9): Decision, Environment and command, EXP-0076 — M6-A26.6 L29 output provenance, Follow-up, Interpretation, Method, Question, Results — L29 P19 (+1 more)

### Community 295 - "EXP-0077 — M6-A26.7 L29 gated-output provenance"
Cohesion: 0.20
Nodes (9): Decision, Environment and command, EXP-0077 — M6-A26.7 L29 gated-output provenance, Follow-up, Interpretation, Method, Question, Results — L29 P19 (+1 more)

### Community 296 - "EXP-0078 — M6-A26.8 L29 gate-input provenance"
Cohesion: 0.20
Nodes (9): Decision, Environment and command, EXP-0078 — M6-A26.8 L29 gate-input provenance, Follow-up, Interpretation, Method, Question, Results (+1 more)

### Community 297 - "Group"
Cohesion: 0.20
Nodes (13): set, size_t, string, uint64_t, vector, Group, bytes, count (+5 more)

### Community 298 - "EXP-0087 — M6-A27.7 Qwen3.8 L0 output-projection contract adjudication"
Cohesion: 0.22
Nodes (8): Contract clarification, Decision, EXP-0087 — M6-A27.7 Qwen3.8 L0 output-projection contract adjudication, Follow-up, Interpretation, Method, Question, Results

### Community 299 - "EXP-0123 — M6-B31 recurrent FFN Gate/Up two-row MMVQ"
Cohesion: 0.25
Nodes (7): Baseline and candidate, Decision, Environment, EXP-0123 — M6-B31 recurrent FFN Gate/Up two-row MMVQ, Follow-up, Question, Results

### Community 300 - "EXP-0081 — M6-A27.1 Qwen3.8 L54/P1 output attribution"
Cohesion: 0.29
Nodes (6): Decision, EXP-0081 — M6-A27.1 Qwen3.8 L54/P1 output attribution, Follow-up, Method, Question, Results

### Community 301 - "EXP-0109 — M6-B16 projection-input Q8_K reuse"
Cohesion: 0.17
Nodes (11): Baseline and candidate, Commands, Correctness, Decision, Environment, EXP-0109 — M6-B16 projection-input Q8_K reuse, Follow-up, Hypothesis (+3 more)

### Community 302 - "EXP-0095 — M6-B2 Q5_K scale/min unpack hoisting"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Commands, Correctness and resources, Decision, Environment, EXP-0095 — M6-B2 Q5_K scale/min unpack hoisting, Follow-up (+3 more)

### Community 303 - "EXP-0091 — M6-A27 observable numerical-equivalence closure"
Cohesion: 0.22
Nodes (8): Correctness decision, Decision, Environment, EXP-0091 — M6-A27 observable numerical-equivalence closure, Follow-up, Method, Question, Results

### Community 304 - "EXP-0088 — M6-A27.8 Qwen3.8 L0 Q8_K contract"
Cohesion: 0.18
Nodes (10): Baseline, Decision, Environment, EXP-0088 — M6-A27.8 Qwen3.8 L0 Q8_K contract, Follow-up, Hypothesis, Interpretation, Method (+2 more)

### Community 305 - "qwen3_primitives_test.cpp"
Cohesion: 0.43
Nodes (7): close_enough(), T, vector, device_copy(), gpu_tests(), host_tests(), main()

### Community 306 - "Qwen3LayerWeights"
Cohesion: 0.08
Nodes (24): uint32_t, Qwen3Config, attention_heads, context_length, head_dim, hidden_size, intermediate_size, kv_heads (+16 more)

### Community 307 - "Qwen3GpuDecodeCache"
Cohesion: 0.09
Nodes (26): size_t, Qwen3GpuDecodeCache, caches_, prepare, workspace_, Qwen3Layer0GpuKvCache, append, keys_ (+18 more)

### Community 308 - "qwen3_decode_profile.cpp"
Cohesion: 0.17
Nodes (19): build_json(), string, timespec, uint32_t, elapsed_ms(), json_escape(), main(), now() (+11 more)

### Community 309 - "EXP-0113 — M6-B20 Q6_K×Q8_1 MMVQ recurrent QKV"
Cohesion: 0.25
Nodes (7): Candidate, Checks, Decision, Environment, EXP-0113 — M6-B20 Q6_K×Q8_1 MMVQ recurrent QKV, Question, Result

### Community 310 - "EXP-0082 — M6-A27.2 Qwen3.8 L53/P1 output provenance"
Cohesion: 0.29
Nodes (6): Decision, EXP-0082 — M6-A27.2 Qwen3.8 L53/P1 output provenance, Follow-up, Method, Question, Results

### Community 311 - "M4 — First Correct End-to-End Generation"
Cohesion: 0.33
Nodes (6): Correctness validation, Exit criteria, Goal, Initial inference mode, M4 — First Correct End-to-End Generation, Required execution pieces

### Community 312 - "EXP-0083 — M6-A27.3 Qwen3.8 L53 gated-path contract adjudication"
Cohesion: 0.29
Nodes (6): Decision, EXP-0083 — M6-A27.3 Qwen3.8 L53 gated-path contract adjudication, Follow-up, Method, Question, Results

### Community 313 - "EXP-0101 — M6-B8 cached-attention Wave64-local reduction"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Commands, Correctness, Decision, Environment, EXP-0101 — M6-B8 cached-attention Wave64-local reduction, Follow-up (+3 more)

### Community 314 - "EXP-0105 — M6-B12 Q6_K packed dot4 projections"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness and resources, Decision, Environment, EXP-0105 — M6-B12 Q6_K packed dot4 projections, Follow-up, Interpretation (+2 more)

### Community 315 - "EXP-0092 — M6-A28 native autoregressive GPU generation"
Cohesion: 0.25
Nodes (7): Change, Decision, Environment, EXP-0092 — M6-A28 native autoregressive GPU generation, Follow-up, Question, Results

### Community 316 - "Qwen3GpuProfileEvent"
Cohesion: 0.14
Nodes (13): hipEvent_t, Qwen3BoundaryProfileStage, Qwen3FfnProfileStage, Qwen3ProfileCategory, Qwen3GpuProfileEvent, boundary_stage, bytes, category (+5 more)

### Community 317 - "EXP-0102 — M6-B9 Q6_K LM-head index hoisting"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness, Decision, Environment, EXP-0102 — M6-B9 Q6_K LM-head index hoisting, Follow-up, Interpretation (+2 more)

### Community 318 - "qwen3_generate.cpp"
Cohesion: 0.40
Nodes (10): argmax(), size_t, string, uint32_t, vector, main(), parse_count(), parse_id() (+2 more)

### Community 319 - "EXP-0094 — M6-B2 direct layer-output handoff"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Decision, Environment and workload, EXP-0094 — M6-B2 direct layer-output handoff, Follow-up, Hypothesis, Interpretation (+2 more)

### Community 320 - "require_match"
Cohesion: 0.21
Nodes (14): check_device(), check_values(), Metrics, size_t, span, T, uint32_t, device_error() (+6 more)

### Community 321 - "Qwen3Layer0KvCache"
Cohesion: 0.24
Nodes (7): size_t, vector, Qwen3DecodeCache, caches_, length, Qwen3Layer0KvCache, append

### Community 322 - "validate_position"
Cohesion: 0.33
Nodes (10): cached_attention(), Metrics, size_t, span, string_view, vector, main(), report() (+2 more)

### Community 323 - "EXP-0116 — M6-B24 full-attention stage attribution"
Cohesion: 0.20
Nodes (9): Baseline, Correctness and resources, Decision, Environment, EXP-0116 — M6-B24 full-attention stage attribution, Follow-up, Interpretation, Question (+1 more)

### Community 324 - "EXP-0098 — M6-B5 Qwen3.8 Q8_K activation reuse"
Cohesion: 0.25
Nodes (7): Baseline and candidate, Decision, Environment, EXP-0098 — M6-B5 Qwen3.8 Q8_K activation reuse, Follow-up, Question, Results

### Community 325 - "EXP-0114 — M6-B21 Q4_K×Q8_1 MMVQ recurrent gate"
Cohesion: 0.20
Nodes (9): Candidate and control, Correctness and resources, Decision, Environment, EXP-0114 — M6-B21 Q4_K×Q8_1 MMVQ recurrent gate, Follow-up, Interpretation, Question (+1 more)

### Community 326 - "EXP-0097 — M6-B4 Q5_K four-row workgroup"
Cohesion: 0.29
Nodes (6): Baseline and candidate, Decision, EXP-0097 — M6-B4 Q5_K four-row workgroup, Follow-up, Question, Result

### Community 327 - "5. Runtime Layers"
Cohesion: 0.29
Nodes (7): 5.1 Model Layer, 5.2 Packing / Representation Layer, 5.3 Memory Planner, 5.4 Kernel Planner, 5.5 Execution Plan, 5.6 Kernel Layer, 5. Runtime Layers

### Community 328 - "EXP-0104 — M6-B11 Q4_K packed dot4 projections"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness and resources, Decision, Environment, EXP-0104 — M6-B11 Q4_K packed dot4 projections, Follow-up, Interpretation (+2 more)

### Community 329 - "EXP-0122 — M6-B30 transposed DeltaNet recurrent state"
Cohesion: 0.17
Nodes (11): Baseline and candidate, Benchmark, Commands, Correctness, Decision, Environment, EXP-0122 — M6-B30 transposed DeltaNet recurrent state, Follow-up (+3 more)

### Community 330 - "GatePathCapture"
Cohesion: 0.29
Nodes (7): GatePathCapture, gate, gated, head_norm, head_scaled, normalized, recurrent_output

### Community 331 - "EXP-0111 — M6-B18 Q4_K×Q8_1 MMVQ FFN Down"
Cohesion: 0.17
Nodes (11): Baseline and candidate, Commands, Correctness, Decision, Environment, EXP-0111 — M6-B18 Q4_K×Q8_1 MMVQ FFN Down, Follow-up, Hypothesis (+3 more)

### Community 332 - "hip_check.hpp"
Cohesion: 0.23
Nodes (8): hip_check(), hip_check_failed(), hipError_t, string, main(), run_shape(), run_silu_q8_fusion_identity_test(), run_zero_point_identity_tests()

### Community 333 - "Roadmap Principles"
Cohesion: 0.40
Nodes (5): Benchmark before claim, Correctness before speed, Evidence before architecture, Narrow before broad, Roadmap Principles

### Community 334 - "EXP-0108 — M6-B15 recurrent state-update no-decay-store candidate"
Cohesion: 0.20
Nodes (9): Baseline and candidate, Benchmark, Correctness, Decision, Environment and workload, EXP-0108 — M6-B15 recurrent state-update no-decay-store candidate, Follow-up, Profile (+1 more)

### Community 335 - "HostQ8Block"
Cohesion: 0.33
Nodes (6): int8_t, uint16_t, HostQ8Block, d_bits, qs, s_bits

### Community 336 - "EXP-0115 — M6-B22 Q6_K×Q8_K packed-dot4 recurrent QKV"
Cohesion: 0.20
Nodes (9): Candidate and control, Correctness and resources, Decision, Environment, EXP-0115 — M6-B22 Q6_K×Q8_K packed-dot4 recurrent QKV, Follow-up, Interpretation, Question (+1 more)

### Community 337 - "KeyPathCapture"
Cohesion: 0.33
Nodes (6): KeyPathCapture, input, key, key_norm, normalized, qkv

### Community 338 - "EXP-0120 — M6-B28 post-B27 production profile"
Cohesion: 0.20
Nodes (9): Decision, Environment, EXP-0120 — M6-B28 post-B27 production profile, Follow-up, Interpretation, Question, Ranking, Results (+1 more)

### Community 339 - "EXP-0112 — M6-B19 Q4_K×Q8_1 MMVQ FFN Gate/Up"
Cohesion: 0.18
Nodes (10): Candidate and control, Correctness and resource checks, Decision, Environment, EXP-0112 — M6-B19 Q4_K×Q8_1 MMVQ FFN Gate/Up, Follow-up, Hypothesis, Interpretation (+2 more)

### Community 340 - "m6a19_qwen35_conv_gpu.cpp"
Cohesion: 0.33
Nodes (6): size_t, T, DeviceBuffer, data_, main(), upload()

### Community 341 - "EXP-0121 — M6-B29 recurrent stage attribution"
Cohesion: 0.22
Nodes (8): Decision, Environment, EXP-0121 — M6-B29 recurrent stage attribution, Follow-up, Question, Ranking, Results, Scope

### Community 342 - "m6a11_qwen35_attention_prefix.cpp"
Cohesion: 0.39
Nodes (6): size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 343 - "RecurrentOperands"
Cohesion: 0.14
Nodes (16): vector, download(), OutputProjectionPathCapture, gated, input, projected, q8_input, residual (+8 more)

### Community 344 - "run_ladder"
Cohesion: 0.30
Nodes (11): argmax(), size_t, span, uint32_t, is_boundary(), is_full_attention_layer(), main(), print_ledger() (+3 more)

### Community 345 - "StageProfile"
Cohesion: 0.50
Nodes (4): array, hipEvent_t, StageProfile, start

### Community 346 - "EXP-0126 — M6-B34 fused SiLU to Q8_1"
Cohesion: 0.18
Nodes (10): Baseline and candidate, Benchmark, Correctness, Decision, Environment, EXP-0126 — M6-B34 fused SiLU to Q8_1, Follow-up, Hypothesis (+2 more)

### Community 347 - "EXP-0117 — M6-B25 fine full-attention attribution"
Cohesion: 0.20
Nodes (9): Baseline, Correctness and resources, Decision, Environment, EXP-0117 — M6-B25 fine full-attention attribution, Follow-up, Interpretation, Question (+1 more)

### Community 348 - "Metrics"
Cohesion: 0.18
Nodes (11): size_t, Metrics, actual_at_max, expected_at_max, finite, max_abs, max_index, max_rel (+3 more)

### Community 349 - "EXP-0118 — M6-B26 Q-projection Q8_1 MMVQ candidate"
Cohesion: 0.22
Nodes (8): Baseline, Correctness, Decision, Environment, EXP-0118 — M6-B26 Q-projection Q8_1 MMVQ candidate, Follow-up, Interpretation, Question

### Community 350 - "Current Project Status"
Cohesion: 0.50
Nodes (4): Completed, Completed in Task 3, Current Project Status, Not implemented

### Community 351 - "13. Prefill vs Decode"
Cohesion: 0.67
Nodes (3): 13. Prefill vs Decode, Decode, Prefill

### Community 352 - "Current Scope"
Cohesion: 0.67
Nodes (3): Current Scope, In scope now, Not in scope now

### Community 353 - "Metrics"
Cohesion: 0.25
Nodes (8): Metrics, actual_at_max, expected_at_max, max_abs, max_index, max_rel, mean_abs, rmse

### Community 354 - "Q8KDeviceBlock"
Cohesion: 0.33
Nodes (6): int16_t, int8_t, Q8KDeviceBlock, bsums, d, qs

### Community 355 - "Qwen3TensorView"
Cohesion: 0.33
Nodes (4): byte, GgufTensorType, Qwen3TensorView, source

### Community 356 - "run_combined"
Cohesion: 0.40
Nodes (5): RuntimeState, uint32_t, vector, main(), run_combined()

### Community 357 - "fp16_gemv_reference.cpp"
Cohesion: 0.47
Nodes (5): uint32_t, vector, evaluate_fp16_gemv(), fp16_gemv_cpu_reference(), generate_fp16_gemv_data()

## Knowledge Gaps
- **2600 isolated node(s):** `experiment`, `shape`, `implementation`, `cache_regime`, `custom_label` (+2595 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **27 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RecurrentLayer` connect `RecurrentLayer` to `size_t`, `StageProfile`, `GatePathCapture`, `m6a21_qwen35_gpu_hybrid_block.cpp`, `UpdateProvenance`, `KeyPathCapture`, `.get`, `RecurrentOperands`, `Qwen35Config`, `GgufFile`, `LayerPathCapture`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Why does `GgufFile` connect `GgufFile` to `Qwen35Config`, `vector`, `m6a15_qwen35_hybrid_block_audit.cpp`, `run_ladder`, `run_combined`, `validate_position`, `path_`, `m6a14_qwen35_state_audit.cpp`, `m6a18_qwen35_deltanet_state_gpu.cpp`, `array`, `gguf.cpp`, `m6a3_qwen35_layer.cpp`, `Qwen3Model`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Why does `Qwen3Model` connect `Qwen3Model` to `Qwen3TensorView`, `qwen3_fast_decode_bench.cpp`, `model_plan.cpp`, `Qwen3GpuPlan`, `qwen3_tokenizer.cpp`, `qwen3_layer_host_impl`, `Qwen3LayerWeights`, `qwen3_decode_profile.cpp`, `gguf.cpp`, `GgufFile`, `qwen3_inference_bench.cpp`, `qwen3_layer35_external_test.cpp`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **What connects `experiment`, `shape`, `implementation` to the rest of the system?**
  _2600 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `hardware.md` be split into smaller, more focused modules?**
  _Cohesion score 0.0425531914893617 - nodes in this community are weakly interconnected._
- **Should `architecture.md` be split into smaller, more focused modules?**
  _Cohesion score 0.058823529411764705 - nodes in this community are weakly interconnected._
- **Should `benchmarking.md` be split into smaller, more focused modules?**
  _Cohesion score 0.045454545454545456 - nodes in this community are weakly interconnected._