# Graph Report - mi50  (2026-09-08)

## Corpus Check
- 478 files · ~372,549 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 6614 nodes · 9081 edges · 562 communities (506 shown, 56 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 282 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `cb7b6f33`
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
- EXP-0161 — M6-B65 post-B62 native benchmark baseline
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
- qwen3_gpu_primitives.hpp
- 4. Candidate Optimization Roadmap for M8
- 32. Codex task behavior
- Sha256
- graphify reference: query, path, explain
- 43. Initial MIInfer Benchmark Matrix
- Qwen3-8B Dense Control Model
- EXP-0146 — M6-B54 post-B53 production profile
- qwen3_decode_sequence_gpu_test.cpp
- 15. Benchmark Protocol
- 10. Kernel experiments
- 40. Acceptance Categories
- 5. Required Environment Metadata
- D008 — Do Not Build a Generic Graph Runtime Initially
- RecurrentLayer
- FullAttentionLayer
- 6. nlzy/vllm-gfx906
- EXP-0179 — Native Q6_K Wave64 Rollout & Activation Reuse
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
- .run
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
- qwen35_gpu_pipeline.hpp
- Q4_0Block
- DeviceShapeData
- External gfx906 Reference Baseline
- q4_q8_gemv_bench.cpp
- Qwen3GpuDecodeWorkspace
- EXP-0086 — M6-A27.6 Qwen3.8 P2 L0–L2 operation trace
- fp16_gemv_reduction_diag.cpp
- main
- m6a15_qwen35_hybrid_block_audit.cpp
- EXP-0184 — Fast 2-Stage Parallel Argmax Reduction
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
- qwen3_gpu_layer.cpp
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
- validate_position
- qwen3_position_audit.cpp
- EXP-0209 — B=8 LDS Shared-Weight Shape Check
- EXP-0153 — M6-B61 fused beta/alpha preparation
- qwen3_forward_test.cpp
- EXP-0178 — Native Q5_K Wave64 Rollout (SSM Out Projections)
- M5-A — Reproducible MI50 inference baseline
- M4-C3 — Text-facing greedy generation
- run-m4c3-acceptance.sh
- m12_gdn_chunk_bench.cpp
- qwen3_trace_compare.cpp
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
- EXP-0183 — Fused DeltaNet Recurrent Core in LDS
- m6a263_qwen35_recurrent_contract.cpp
- EXP-0011 — Trace-free Qwen3-8B decode control
- EXP-0099 — M6-B6 Q5_K subgroup-structured dot loop
- EXP-0022 — M5-C6d GPU-side greedy argmax
- span
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
- M7 — Establish and Beat the gfx906 Performance Frontier
- EXP-0047 — M6-A5 Qwen3.8-27B four-layer hybrid block
- EXP-0048 — M6-A6 Qwen3.8-27B full 64-layer forward
- DeviceInfo
- EXP-0050 — M6-B0 Qwen3.8-27B llama.cpp MI50 baseline
- EXP-0188 — Inter-Layer Norm Fusion, Native Q6_K LM Head & Wave64 Single-Wave GEMV (Stretch Gate Closure)
- EXP-0056 — M6-A12 Qwen3.8-27B attention projections
- EXP-0049 — M6-A7 Qwen3.8-27B stateful generation
- Qwen35Config
- EXP-0252 — Repacked MMQ64 retest on the production Q4_K down shape
- EXP-0254 — Repacked MMQ64 exact-sum activation layout
- run-m6b0-llama-baseline.sh
- qwen3_inference_bench.cpp
- EXP-0058 — M6-A14 Qwen3.8-27B state fingerprints and reset audit
- EXP-0125 — M6-B33 post-B32 production profile
- EXP-0059 — M6-A15 Qwen3.8-27B layers 0–3 hybrid-block audit
- EXP-0093 — M6-B1 Qwen3.8-27B native GPU generation baseline
- EXP-0055 — M6-A11 Qwen3.8-27B composed attention prefix
- kquant_wave_layout.cpp
- m6a13_qwen35_full_attention_layer.cpp
- EXP-0079 — M6-A26.9 Qwen3.8 external recurrent-state contract
- EXP-0052 — M6-A8 Qwen3.8-27B GPU foundation
- EXP-0089 — M6-A27.9 Qwen3.8 L0 Q5_K block contract
- EXP-0073 — M6-A26.3 Qwen3.8 recurrent-state contract adjudication
- EXP-0060 — M6-A16 Qwen3.8-27B layers 4–7 hybrid-block audit
- EXP-0090 — M6-A27.9 full observable-contract retest
- EXP-0061 — M6-A17 Qwen3.8-27B composition ladder
- AttentionPathReplay
- GgufFile
- EXP-0096 — M6-B3 Q4_K metadata staging
- EXP-0182 — Fused Gate+Up SwiGLU Wave64 GEMV
- Qwen35RuntimeEngine
- EXP-0066 — M6-A22 Qwen3.8-27B GPU hybrid position audit
- m6a18_qwen35_deltanet_state_gpu.cpp
- EXP-0141 — M6-B49 recurrent state-update/head-RMS fusion
- Metrics
- EXP-0226 — M11-B Q4_K chunked row-tile projection
- EXP-0064 — M6-A20 Qwen3.8-27B recurrent layer on GPU
- EXP-0072 — M6-A26.2 Qwen3.8 L30 update provenance
- EXP-0103 — M6-B10 recurrent Q8_K input reuse
- EXP-0204 — Production Context Scaling After M11-A Integration
- EXP-0065 — M6-A21 Qwen3.8-27B GPU hybrid block
- run-exp0183-fused-recurrent-ab.sh
- EXP-0068 — M6-A24 Qwen3.8 eight-layer stateful GPU prefix
- Qwen3Model
- EXP-0080 — M6-A27 Qwen3.8 sixty-four-layer GPU composition
- EXP-0085 — M6-A27.5 Qwen3.8 P2 drift localization
- EXP-0070 — M6-A26 Qwen3.8 thirty-two-layer stateful GPU prefix
- run_case
- EXP-0124 — M6-B32 transposed recurrent no-decay store
- EXP-0156 — M6-B64 post-B62 stage profile
- EXP-0137 — M6-B45 Q4_K×Q8_1 inner-loop differential
- EXP-0075 — M6-A26.5 L30 K-path provenance
- qwen3_gpu_layer.hpp
- EXP-0076 — M6-A26.6 L29 output provenance
- EXP-0077 — M6-A26.7 L29 gated-output provenance
- EXP-0078 — M6-A26.8 L29 gate-input provenance
- EXP-0230 — M11-B raw int8 GEMM ceiling
- EXP-0087 — M6-A27.7 Qwen3.8 L0 output-projection contract adjudication
- EXP-0123 — M6-B31 recurrent FFN Gate/Up two-row MMVQ
- EXP-0081 — M6-A27.1 Qwen3.8 L54/P1 output attribution
- EXP-0109 — M6-B16 projection-input Q8_K reuse
- EXP-0095 — M6-B2 Q5_K scale/min unpack hoisting
- EXP-0091 — M6-A27 observable numerical-equivalence closure
- EXP-0088 — M6-A27.8 Qwen3.8 L0 Q8_K contract
- EXP-0154 — M6-B62 DeltaNet row-wave state mapping
- EXP-0128 — M6-B36 post-B35 production profile
- EXP-0136 — M6-B44 Q4_K×Q8_1 split-K MMVQ mapping
- EXP-0168 — post-B1 whole-token profile
- EXP-0113 — M6-B20 Q6_K×Q8_1 MMVQ recurrent QKV
- EXP-0082 — M6-A27.2 Qwen3.8 L53/P1 output provenance
- EXP-0163 — M6-B67 post-B66 recurrent profile
- EXP-0083 — M6-A27.3 Qwen3.8 L53 gated-path contract adjudication
- EXP-0101 — M6-B8 cached-attention Wave64-local reduction
- EXP-0105 — M6-B12 Q6_K packed dot4 projections
- EXP-0092 — M6-A28 native autoregressive GPU generation
- EXP-0140 — M6-B48 persistent Q4_K FFN Down metadata
- EXP-0102 — M6-B9 Q6_K LM-head index hoisting
- Qwen3GpuDecodeCache
- EXP-0094 — M6-B2 direct layer-output handoff
- require_match
- EXP-0129 — M6-B37 Q4_K×Q8_1 Down weight staging
- EXP-0149 — M6-B57 direct layer output
- EXP-0116 — M6-B24 full-attention stage attribution
- EXP-0098 — M6-B5 Qwen3.8 Q8_K activation reuse
- EXP-0114 — M6-B21 Q4_K×Q8_1 MMVQ recurrent gate
- EXP-0097 — M6-B4 Q5_K four-row workgroup
- EXP-0127 — M6-B35 Q4_K×Q8_1 LDS activation reuse
- EXP-0104 — M6-B11 Q4_K packed dot4 projections
- EXP-0122 — M6-B30 transposed DeltaNet recurrent state
- Qwen3GpuPlan
- EXP-0111 — M6-B18 Q4_K×Q8_1 MMVQ FFN Down
- EXP-0147 — M6-B55 DeltaNet LDS input reuse
- EXP-0175 — MI50 sustained operating-point qualification
- EXP-0108 — M6-B15 recurrent state-update no-decay-store candidate
- EXP-0131 — M6-B39 Q4_K×Q8_1 metadata staging
- EXP-0115 — M6-B22 Q6_K×Q8_K packed-dot4 recurrent QKV
- EXP-0159 — M6-B67 transposed versus non-transposed full-model control
- EXP-0120 — M6-B28 post-B27 production profile
- EXP-0112 — M6-B19 Q4_K×Q8_1 MMVQ FFN Gate/Up
- EXP-0246 — M11-B Q4_K MMQ64 split-4 rejection
- EXP-0121 — M6-B29 recurrent stage attribution
- kquant_layout_bench.cpp
- EXP-0180 — M7 gfx906 Performance Frontier Benchmark
- EXP-0216 — M11-B Recurrent Q5_K `ssm_out` B=4 Projection
- EXP-0135 — M6-B43 Q6_K×Q8_1 LM-head metadata staging
- EXP-0126 — M6-B34 fused SiLU to Q8_1
- EXP-0117 — M6-B25 fine full-attention attribution
- EXP-0130 — M6-B38 post-B37 production profile
- EXP-0118 — M6-B26 Q-projection Q8_1 MMVQ candidate
- EXP-0138 — M6-B46 Q4_K×Q8_1 packed-input diagnostic
- EXP-0133 — M6-B41 Q4_K×Q8_1 decoded metadata staging
- EXP-0134 — M6-B42 post-B41 production profile
- EXP-0139 — M6-B47 dual Gate/Up Q4_K×Q8_1 projection
- EXP-0151 — M6-B59 expanded Q4_K FFN Down weights
- EXP-0150 — M6-B58 Q6_K×Q8_K QKV LDS input
- EXP-0162 — M6-B66 dual beta/alpha FP32 GEMV
- EXP-0132 — M6-B40 post-B39 production profile
- EXP-0142 — M6-B50 fused recurrent output path
- EXP-0167 — post-A28 native generation baseline
- EXP-0262 — M12 production qualification gate
- EXP-0144 — M6-B52 Q4_K×Q8_1 one-Wave64 row mapping
- EXP-0189 — Wave64-Native Shuffle-Based Q8_1 Activation Quantizer (Lane 1)
- EXP-0172 — Reference Q4K inner-loop load schedule
- EXP-0232 — M11-B Q4_K 16-token MMQ tile
- EXP-0148 — M6-B56 post-B55 production profile
- EXP-0143 — M6-B51 post-B50 production profile
- EXP-0211 — M11-B Row-LDS B=8 Production-Shape Rejection
- run-exp0188-stretch-goal-ab.sh
- EXP-0164 — M6-B68 fused beta/alpha preparation
- EXP-0145 — M6-B53 Q4_K×Q8_K versus Q4_K×Q8_1 FFN differential
- EXP-0191: Fused Residual RMS Norm with Direct Q8_1 Handoff
- EXP-0171 — M6-B76 architectural blocker report
- EXP-0152 — M6-B60 post-B59 production profile
- PrefillProfile
- EXP-0237 — M11-B production layer-major prefill profile
- EXP-0165 — M6-B69 dual query/key head normalization
- EXP-0160 — M6-B68 DeltaNet ordered row-wave reduction
- UpdateProvenance
- EXP-0166 — M6-B70 column-tiled DeltaNet state update
- EXP-0192: Wave64 Fused Gate+Up SwiGLU Intra-Wave Shuffle Reduction
- EXP-0176 — MI50 manual-DPM qualification and EXP-0174 re-adjudication
- 2. Command Architecture
- EXP-0158 — M6-B66 expanded Q4_K FFN Gate/Up
- Checkpoint
- EXP-0155 — M6-B63 post-B62 production baseline
- EXP-0190 — Fused Recurrent & Attention Epilogue Q8_1 Activation Quantization (Lane 2)
- EXP-0231 — M11-B Q4_K repacked MMQ tile
- EXP-0173 — Q4K native-layout laboratory
- EXP-0196 — Combined Projections for Recurrent (QKV+Gate) and Attention (Q+K) Layers
- EXP-0181 — Static HIP Graph Capture for Autoregressive Decode
- Candidate 1: FFN Gate & Up Rollout
- EXP-0218 — M11-B Q4_K Word-Reuse B=8 Rejection
- EXP-0174 — Wave-plane Q4K Down integration
- m12_dense_stage_bench.cpp
- EXP-0169 — M6-B74 pinned llama.cpp architecture differential
- EXP-0170 — M6-B75 fused Gate/Up/SwiGLU prototype
- Candidate 2: Full-Attention Q Projection Rollout
- EXP-0193: Vectorized SIMD Q6_K Decoding for LM Head and Down Projections
- m6a8_qwen35_gpu_foundation.cpp
- EXP-0228 — M11-B Q4_K four-token subwave mapping
- 5. Runtime Layers
- EXP-0197 — Paired Fused Gate+Up SwiGLU with 64-Bit dwordx2 Coalesced Memory Access
- Candidate 3: Recurrent Attention Gate Rollout
- launch_projection
- Candidate 4: Full-Attention Output Projection Rollout
- Milestone M9 Primary Gate Qualification Report
- EXP-0194 — Vectorized Q4_K and Q5_K Fast Tile Arithmetic Optimization
- EXP-0185 — Tiled Online-Softmax Attention with Gate Sigmoid Fusion
- RecurrentOperands
- run-m7-frontier-benchmark.sh
- EXP-0222 — M11-B Final Chunk Pointer Correction
- Architectural Design: DeltaNet Recurrent SSM Core Fusion
- Architectural Design: Fused LM-Head GEMV and Argmax Reduction
- Candidate 5: Full-Attention K Projection Rollout
- Architectural Analysis: HIP Graph Capture and Kernel Dispatch Overhead
- EXP-0201 — Real-World Latency Curve & Hybrid Context Scaling Analysis
- EXP-0177 — Native Q4_K layout rollout across projection families
- run-exp0184-fast-argmax-ab.sh
- EXP-0177 Campaign Summary & Parity Reconciliation
- run-exp0177-attn-gate-ab.sh
- run-exp0177-attn-out-ab.sh
- run-exp0177-gate-up-ab.sh
- run-exp0177-k-ab.sh
- run-exp0177-q-ab.sh
- EXP-0187 — Vectorized Wave64 RMS Norm & Fused Residual Addition
- hip_smoke_bench.cpp
- EXP-0243 — M11-B SwiGLU/Q8 producer-consumer fusion rejection
- RuntimeGenerateStats
- run-exp0178-ssm-out-ab.sh
- run-exp0179-q6k-ab.sh
- EXP-0195: 1-Wave-Per-Row 0-LDS Fused SwiGLU Kernel Evaluation
- EXP-0200 — Device-Side Token Chaining in HIP Graph Decode Loop
- run-exp0189-wave64-q8-ab.sh
- EXP-0261 — M13 quantized matrix prefill
- run-exp0190-fused-core-q8-ab.sh
- run-exp0181-hip-graph-ab.sh
- run-exp0191-fused-norm-q8-ab.sh
- run-exp0192-swiglu-shuffle-ab.sh
- run-exp0193-q6k-simd-unpack-ab.sh
- EXP-0186 — Fused RoPE + Head Norm in Full Attention Layers
- run-exp0182-fused-gate-up-ab.sh
- run-exp0185-tiled-online-attention-ab.sh
- run-m8a-requal.sh
- EXP-0236 — M11-B direct attention Q8_1 emission
- M1 — Kernel Laboratory
- Milestone M8 Primary Gate Qualification Report
- EXP-0205 — Batched Q4_K Wave GEMV for Prefill
- Milestone M10 — Real-World Inference Performance & Hybrid Context Scaling
- run-exp0186-fused-rope-norm-ab.sh
- run-exp0187-fused-add-rms-norm-ab.sh
- .generate
- EXP-0253 — Repacked MMQ64 with exact Q8 side sums
- EXP-0210 — M11-B Amdahl Profile and Sequential-Floor Check
- qwen3_cached_attention_determinism_gpu_test.cpp
- EXP-0213 — M11-B FP16 GEMM with On-Device K-Quant Dequantization
- m6a10_qwen35_q4k_projection.cpp
- EXP-0202 — FP32 vs FP16 KV Cache Numerical & Performance Investigation
- EXP-0199 — Multi-Wave Cooperative Q6_K FFN Down GEMV
- EXP-0203 — Wave64 Barrier-Free Vectorized Split-K Attention
- EXP-0206 — Layer-Major Chunked Prefill
- EXP-0217 — M11-B Deferred Attention Tail and Shape-Specific Q5 Reuse
- 2. Autoregressive Test Results
- M9 — Numerical Error Budget & Attribution Report
- M9 — Post-M8 Latency Floor & Decomposition Report
- miinfer_cli.cpp
- Milestone M9 — Context Scaling Qualification (TG64 → TG1024)
- EXP-0227 — M11-B Q4_K token-parallel row tile
- EXP-0241 — M11-B Q4_K row-major 16-token tile rejection
- size_t
- EXP-0214 — M11-B Native Q4_K Split-K Batched GEMM
- run-exp0194-kquant-fast-arith-ab.sh
- run-exp0196-combined-projections-ab.sh
- run-exp0197-swiglu-paired-ab.sh
- run-exp0198-cached-metadata-ab.sh
- run-m9-gate-telemetry.sh
- run-m9-qualification.sh
- bench_m10_latency_curve.py
- m6a12_qwen35_attention_projections.cpp
- EXP-0212 — M11-B B=8 Accumulator GEMV Rejection
- EXP-0219 — M11-B Q4_K LDS Tile-Reuse Rejection
- EXP-0244 — M11-B final qualification and measured ceiling
- bench-streaming-modes.sh
- Options
- m12_gdn_oracle.cpp
- recurrent_block
- Qwen3ForwardTrace
- EXP-0256 — M12 chunkwise Gated DeltaNet oracle
- Qwen3Layer0KvCache
- EXP-0221 — M11-B Direct Consumption of Batched Prefill Workspace
- EXP-0208 — Full-Attention Projection Batching
- qwen3_layer0_gpu_test.cpp
- EXP-0207 — B=8 LDS Weight-Reuse Prefill
- EXP-0223 — M11-B Q4_K B=4 Two-Wave Row Rejection
- EXP-0220 — M11-B Q4_K B=4 Compact Arithmetic Rejection
- EXP-0215 — M11-B Larger Logical Chunk with B=4 Inner Microtiles
- EXP-0249 — M11-B native Q4_K MMQ64 rows64 rejection
- EXP-0240 — M11-B Q4_K wave-major 16-token tile rejection
- EXP-0250 — M11-B decoded Q4_K MMQ64 rejection
- EXP-0255 — M12 dense staging feasibility
- EXP-0239 — M11-B native Q4_K 16-token tile rejection
- EXP-0234 — M11-B causal recurrent-core B=4 batch
- EXP-0224 — M11-B Q4_K B=4 LDS Token Distribution Rejection
- EXP-0233 — M11-B quantized projection Amdahl ceiling
- EXP-0229 — M11-B Q4_K token-microtile wave decomposition
- EXP-0225 — M11-B Q4_K B=4 Full Row-Tile LDS Rejection
- EXP-0238 — M11-B causal 64-token prefill chunk
- ProfileScope
- EXP-0248 — M11-B operator-major recurrent tail rejection
- model_loader_test.cpp
- Qwen3GpuProfileEvent
- Group
- Qwen3Config
- EXP-0235 — M11-B fused recurrent Q8_1 output
- EXP-0257 — M12 gfx906 Gated DeltaNet chunk prototype
- EXP-0259 — M12 dense FFN-down prefill backend
- EXP-0247 — M11-B Q4_K MMQ64 split-2 rejection
- FfnTailReplay
- Qwen3LayerWeights
- EXP-0251 — M11-B native Q4_K token-reuse rejections
- half
- M3 — Minimal Runtime
- m6a9_qwen35_lm_head.cpp
- EXP-0260 — M12 combined dense and chunkwise prefill
- m6a11_qwen35_attention_prefix.cpp
- run_ladder
- qwen3_generate.cpp
- vector
- EXP-0242 — M11-B recurrent fused normalization/Q8 rejection
- 1. ggml-org/llama.cpp
- Roadmap Principles
- Fp16GemvMetrics
- LayerPathCapture
- EXP-0245 — M11-B Q4_K MMQ16 split-4 rejection
- Buffer
- DeviceBuffer
- Q8ExactBlock
- RmsVariant
- M12GdnChunkWorkspace
- D031 — Freeze M11-B and Isolate Matrix Prefill from Decode
- D024 — Benchmarkability Is an Architectural Requirement
- EXP-0263 — M12 GDN contract fix and dense-path qualification
- 11. joe2gaan/localaiservers
- qwen3_swiglu_q8_bench.cpp
- EXP-0258 — M12 runtime composition and 128-token schedule
- M4 — First Correct End-to-End Generation
- 13. Prefill vs Decode
- Metrics
- qwen3_primitives_test.cpp
- run_combined
- GemvKernelResources
- Current Project Status
- HostQ8Block
- Current Scope
- 15. Research Classification
- test-package.sh
- 17. Highest-Priority Research Ideas
- Buffer
- Event
- Event

## God Nodes (most connected - your core abstractions)
1. `RecurrentLayer` - 168 edges
2. `FullAttentionLayer` - 104 edges
3. `Qwen35RuntimeEngine` - 67 edges
4. `Qwen3LayerTrace` - 55 edges
5. `Qwen3GpuPlan` - 51 edges
6. `Qwen3GpuProfile` - 48 edges
7. `Qwen3GpuDecodeWorkspace` - 46 edges
8. `GgufFile` - 43 edges
9. `Qwen3Model` - 43 edges
10. `GgufTensor` - 42 edges

## Surprising Connections (you probably didn't know these)
- `run_sequence()` --calls--> `snapshot_keys`  [INFERRED]
  tests/qwen3_kv_cache_gpu_test.cpp → include/miinfer/qwen3_gpu_layer.hpp
- `run_sequence()` --calls--> `snapshot_values`  [INFERRED]
  tests/qwen3_kv_cache_gpu_test.cpp → include/miinfer/qwen3_gpu_layer.hpp
- `run_q6k_benchmark()` --calls--> `pack_q6k_wave_tensor()`  [INFERRED]
  bench/kquant_layout_bench.cpp → src/kquant_wave_layout.cpp
- `run_q5k_benchmark()` --calls--> `pack_q5k_wave_tensor()`  [INFERRED]
  bench/kquant_layout_bench.cpp → src/kquant_wave_layout.cpp
- `main()` --calls--> `pack_q4k_wave_down()`  [INFERRED]
  bench/m12_dense_stage_bench.cpp → src/kquant_wave_layout.cpp

## Import Cycles
- None detected.

## Communities (562 total, 56 thin omitted)

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

### Community 9 - "EXP-0161 — M6-B65 post-B62 native benchmark baseline"
Cohesion: 0.25
Nodes (7): Command, Decision, Environment, EXP-0161 — M6-B65 post-B62 native benchmark baseline, Follow-up, Question, Results

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
Nodes (12): D023 — Triton Is a Research Tool, Not a Required Runtime, D026 — Strongest Available Relevant Baseline Wins, D027 — Serving Does Not Define the Core Runtime, Decision, Decision, Decision, Decision Change Process, Guiding Rule (+4 more)

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
Cohesion: 0.24
Nodes (18): ostream, string, vector, implementation_label(), json_escape(), main(), median_of(), metrics_json() (+10 more)

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

### Community 33 - "qwen3_gpu_primitives.hpp"
Cohesion: 0.08
Nodes (25): int16_t, int8_t, uint8_t, Q4KExpandedDeviceBlock, d, dmin, minimums, qs (+17 more)

### Community 34 - "4. Candidate Optimization Roadmap for M8"
Cohesion: 0.11
Nodes (18): 1. Executive Summary, 2.1 Model Weight and Memory Inventory, 2.2 Execution Stage Latency Breakdown (Per-Token), 2. Phase M8-B: Fresh Post-M7 Latency Attribution, 3.1 Hardware Memory Bandwidth Upper Bounds, 3.2 Bandwidth-Bound Decode Floors for 15.932 GiB Compulsory Weight Transfer, 3.3 The "Latency Tax" Above the Memory Floor, 3.4 Target Budget for 30.00 tok/s (+10 more)

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

### Community 40 - "EXP-0146 — M6-B54 post-B53 production profile"
Cohesion: 0.20
Nodes (9): Baseline, Commands, Decision, Environment, EXP-0146 — M6-B54 post-B53 production profile, Follow-up, Interpretation, Question (+1 more)

### Community 41 - "qwen3_decode_sequence_gpu_test.cpp"
Cohesion: 0.10
Nodes (42): build_json(), string, timespec, uint32_t, elapsed_ms(), json_escape(), main(), now() (+34 more)

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

### Community 47 - "RecurrentLayer"
Cohesion: 0.02
Nodes (127): RocblasGemmHandle, opaque, RecurrentLayer, alpha_raw, beta, beta_raw, d_a, d_alpha (+119 more)

### Community 48 - "FullAttentionLayer"
Cohesion: 0.02
Nodes (80): FullAttentionLayer, attention, batch_head_rms, d_attn_norm, d_ffn_down, d_ffn_down_expanded, d_ffn_down_native, d_ffn_gate (+72 more)

### Community 49 - "6. nlzy/vllm-gfx906"
Cohesion: 0.40
Nodes (5): 6. nlzy/vllm-gfx906, Key historical observations, MIInfer implication, Role, Status

### Community 50 - "EXP-0179 — Native Q6_K Wave64 Rollout & Activation Reuse"
Cohesion: 0.12
Nodes (16): 1. TG64 (64 tokens), 2. TG128 (128 tokens), 3. Comparison with Pinned llama.cpp Baseline, 4. Hardware State & Telemetry, Baseline, Benchmark Methodology, Candidate, Correctness (+8 more)

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
Cohesion: 0.21
Nodes (33): AccumulationContract, RmsReduction, compare_q8_blocks(), path, size_t, span, vector, dequantize_q8_exact() (+25 more)

### Community 58 - "qwen3_forward_gpu_test.cpp"
Cohesion: 0.25
Nodes (24): argmax(), capture_q8_input(), compare_checkpoint(), path, Q8_1Block, size_t, vector, exact_equal() (+16 more)

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

### Community 97 - ".run"
Cohesion: 0.18
Nodes (14): Q8_1Block, string, uint32_t, project(), project_native_down(), project_native_q6k_down(), project_q4_q8_1(), project_q4_q8_1_expanded() (+6 more)

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
Nodes (35): ggml_tensor, llama_model, llama_token, llama_vocab, callback(), Capture, position, positions (+27 more)

### Community 120 - "EXP-0008 — Direct MIInfer vs gfx906 llama.cpp MMVQ"
Cohesion: 0.13
Nodes (15): 10. Reference ISA and resources, 11. Architectural differences relevant to K/V, 12. M2 decision, 13.1 Re-evaluation after EXP-0009, 13. Next experiment, 1. Question, 2. Hypothesis and motivation, 3. Reference path (+7 more)

### Community 121 - "qwen35_gpu_pipeline.hpp"
Cohesion: 0.12
Nodes (38): GgufTensor, byte_size, data, dimensions, name, offset, type, byte (+30 more)

### Community 122 - "Q4_0Block"
Cohesion: 0.23
Nodes (12): uint8_t, Q4_0Block, d, qs, Q8_1Block, size_t, vector, q4_q8_cpu_reference() (+4 more)

### Community 123 - "DeviceShapeData"
Cohesion: 0.12
Nodes (17): allocate_shape(), DeviceShapeData, device_input_fp16, device_input_q8, device_output, device_weights, input_fp16, input_q8 (+9 more)

### Community 124 - "External gfx906 Reference Baseline"
Cohesion: 0.22
Nodes (8): Baseline status, Checkout, External gfx906 Reference Baseline, Initial baseline, MI50 build starting point, Option validation on the available host, Pin, Toolchain preflight record

### Community 125 - "q4_q8_gemv_bench.cpp"
Cohesion: 0.16
Nodes (29): ostream, Q8_1Block, string, vector, escape(), free_shape(), launch_selected_gemv(), main() (+21 more)

### Community 126 - "Qwen3GpuDecodeWorkspace"
Cohesion: 0.05
Nodes (38): Qwen3GpuDecodeWorkspace, argmax_token, attention, attention_projected, attn_norm, attn_rms, embedding, ffn_input (+30 more)

### Community 127 - "EXP-0086 — M6-A27.6 Qwen3.8 P2 L0–L2 operation trace"
Cohesion: 0.25
Nodes (7): Decision, EXP-0086 — M6-A27.6 Qwen3.8 P2 L0–L2 operation trace, Follow-up, Interpretation, Method, Question, Results

### Community 128 - "fp16_gemv_reduction_diag.cpp"
Cohesion: 0.18
Nodes (16): string, vector, main(), median(), nonnegative(), Options, device, iterations (+8 more)

### Community 129 - "main"
Cohesion: 0.08
Nodes (37): main(), ByteMismatch, count, first, compare_bytes(), cosine_similarity(), detailed_compare(), detailed_device_error() (+29 more)

### Community 130 - "m6a15_qwen35_hybrid_block_audit.cpp"
Cohesion: 0.16
Nodes (27): cache_fingerprint(), History, initializer_list, optional, path, size_t, span, State (+19 more)

### Community 131 - "EXP-0184 — Fast 2-Stage Parallel Argmax Reduction"
Cohesion: 0.12
Nodes (15): 1. TG64 (64 tokens), 2. TG128 (128 tokens), 3. Hardware State & Telemetry, Baseline, Benchmark Methodology, Candidate, Correctness, Decision (+7 more)

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
Cohesion: 0.25
Nodes (18): attention_contract(), cache_corruption_test(), cache_slots_preserved(), checkpoints(), compare_trace(), path, size_t, span (+10 more)

### Community 140 - "Qwen3LayerTrace"
Cohesion: 0.06
Nodes (31): Qwen3LayerTrace, attention_output, attention_probabilities, attention_scores, attn_norm, attn_rms, embedding, ffn_input (+23 more)

### Community 141 - "EXP-0074 — M6-A26.4 L30 production operand attribution"
Cohesion: 0.18
Nodes (10): Baseline and method, Decision, Environment and command, EXP-0074 — M6-A26.4 L30 production operand attribution, Follow-up, Interpretation, One-at-a-time GPU substitutions, Production operand comparison (+2 more)

### Community 142 - "EXP-0041 — M5-C15 optimization closure and parity decision gate"
Cohesion: 0.15
Nodes (12): 1. Question, 2. Authoritative production result, 3. M5 optimization record, 4. Experimentally eliminated explanations, 5. M5 result, 6. Architectural decision gate, 7. Decision, Accepted (+4 more)

### Community 143 - "qwen3_gpu_layer.cpp"
Cohesion: 0.11
Nodes (34): AttentionKernel, Qwen3Projection, Qwen3ProjectionPrecision, capture(), capture_qwen3_head_norm(), copy_to_host(), size_t, span (+26 more)

### Community 144 - "EXP-0100 — M6-B7 Q5_K paired-nibble decoding"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Commands, Correctness and resources, Decision, Environment, EXP-0100 — M6-B7 Q5_K paired-nibble decoding, Follow-up (+3 more)

### Community 145 - "qwen3_layer_host_impl"
Cohesion: 0.16
Nodes (27): byte, GgufTensorType, Qwen3TensorView, source, add_in_place(), size_t, span, uint32_t (+19 more)

### Community 146 - "M4-C1 — Deterministic first generated token"
Cohesion: 0.12
Nodes (13): Acceptance, Decision, M4-C2 — Short deterministic greedy decode sequence, Next slice, Pinned sequence, Acceptance, Decode contract, Deterministic fixture (+5 more)

### Community 147 - "Metrics"
Cohesion: 0.13
Nodes (27): Checkpoint, abs_tolerance, actual, file_index, name, rel_tolerance, compare(), path (+19 more)

### Community 149 - "EXP-0110 — M6-B17 Q5_K×Q8_1 MMVQ recurrent projection"
Cohesion: 0.17
Nodes (11): Baseline and candidate, Commands, Correctness, Decision, Environment, EXP-0110 — M6-B17 Q5_K×Q8_1 MMVQ recurrent projection, Follow-up, Hypothesis (+3 more)

### Community 150 - "M4-B — Full Qwen3 single-token forward"
Cohesion: 0.07
Nodes (29): Acceptance target, Current evidence, Implemented slice, Independent reference, M4-B10 layer-35 Gate/Up projection isolation, M4-B11 Q8 identity and CPU accumulation contract, M4-B12 pre-FFN residual and RMSNorm isolation, M4-B13 attention RMSNorm, V, and position-zero GQA isolation (+21 more)

### Community 152 - "gguf.cpp"
Cohesion: 0.06
Nodes (70): GgufError, GgufValue, value, GgufScalar, unordered_map, Q4GemvKernel, runtime_error, align_up() (+62 more)

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

### Community 163 - "validate_position"
Cohesion: 0.29
Nodes (11): cached_attention(), Metrics, path, size_t, span, string_view, vector, main() (+3 more)

### Community 164 - "qwen3_position_audit.cpp"
Cohesion: 0.10
Nodes (35): category_index(), array, ostream, qwen3_profile_category_count, Qwen3ProfileCategory, size_t, string, timespec (+27 more)

### Community 165 - "EXP-0209 — B=8 LDS Shared-Weight Shape Check"
Cohesion: 0.25
Nodes (7): Candidate, Correctness, Decision, EXP-0209 — B=8 LDS Shared-Weight Shape Check, Follow-up, Hypothesis, Results

### Community 166 - "EXP-0153 — M6-B61 fused beta/alpha preparation"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0153 — M6-B61 fused beta/alpha preparation, Follow-up, Question, Results

### Community 167 - "qwen3_forward_test.cpp"
Cohesion: 0.18
Nodes (26): argmax(), compare(), compare_checkpoint(), path, size_t, vector, fp16_round_trip(), main() (+18 more)

### Community 168 - "EXP-0178 — Native Q5_K Wave64 Rollout (SSM Out Projections)"
Cohesion: 0.12
Nodes (15): 1. TG64 (64 tokens), 2. TG128 (128 tokens), 3. Hardware State & Telemetry, Baseline, Benchmark Methodology, Candidate, Correctness, Decision (+7 more)

### Community 169 - "M5-A — Reproducible MI50 inference baseline"
Cohesion: 0.20
Nodes (9): Decision, Environment, Interpretation, M5-A — Reproducible MI50 inference baseline, Question, Reproduction, Result, Scope (+1 more)

### Community 170 - "M4-C3 — Text-facing greedy generation"
Cohesion: 0.33
Nodes (5): CLI, Decision, M4-C3 — Text-facing greedy generation, Pinned physical acceptance, Tokenizer contract

### Community 172 - "m12_gdn_chunk_bench.cpp"
Cohesion: 0.17
Nodes (17): as(), at(), Buffer, pointer, Fn, hipEvent_t, size_t, T (+9 more)

### Community 173 - "qwen3_trace_compare.cpp"
Cohesion: 0.19
Nodes (18): argmax(), compare(), path, size_t, vector, main(), Metrics, first_value (+10 more)

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

### Community 185 - "EXP-0183 — Fused DeltaNet Recurrent Core in LDS"
Cohesion: 0.12
Nodes (15): 1. TG64 (64 tokens), 2. TG128 (128 tokens), 3. Hardware State & Telemetry, Baseline, Benchmark Methodology, Candidate, Correctness, Decision (+7 more)

### Community 186 - "m6a263_qwen35_recurrent_contract.cpp"
Cohesion: 0.20
Nodes (16): apply_external(), checkpoint(), compare(), path, size_t, span, string_view, vector (+8 more)

### Community 187 - "EXP-0011 — Trace-free Qwen3-8B decode control"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0011 — Trace-free Qwen3-8B decode control, Follow-up (+4 more)

### Community 188 - "EXP-0099 — M6-B6 Q5_K subgroup-structured dot loop"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Commands, Correctness and resources, Decision, Environment, EXP-0099 — M6-B6 Q5_K subgroup-structured dot loop, Follow-up (+3 more)

### Community 189 - "EXP-0022 — M5-C6d GPU-side greedy argmax"
Cohesion: 0.18
Nodes (10): Candidate, Correctness, Decision, Environment and workload, EXP-0022 — M5-C6d GPU-side greedy argmax, Follow-up, Hypothesis, Performance (+2 more)

### Community 190 - "span"
Cohesion: 0.20
Nodes (17): span, reset, Qwen3DecodeCache::reset(), cache_contract_test(), checkpoint_tolerance(), checkpoints(), path, size_t (+9 more)

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
Nodes (23): Checkpoint, file, miinfer, name, tolerance, compare_authority(), compare_host_gpu(), path (+15 more)

### Community 209 - "m6a14_qwen35_state_audit.cpp"
Cohesion: 0.13
Nodes (35): kRecurrentLayers, align_up(), cache_fingerprint(), array, History, path, size_t, span (+27 more)

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
Cohesion: 0.18
Nodes (31): main(), checkpoint(), compare(), conv_output(), array, kChannels, path, size_t (+23 more)

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

### Community 240 - "M7 — Establish and Beat the gfx906 Performance Frontier"
Cohesion: 0.06
Nodes (30): 1.1 Throughput Summary (tokens/second), 1.2 Latency Breakdown (ms/token), 1.3 Telemetry Validation, 1. Competitive Frontier Benchmark Results, 2.1 The Repacking Frontend (+12.5% Throughput Win), 2.2 Dual-Accumulator FFN Fusion (`HAS_FUSION=true`), 2.3 LDS-Fused Recurrent DeltaNet, 2.4 HIP Graph Execution (+22 more)

### Community 241 - "EXP-0047 — M6-A5 Qwen3.8-27B four-layer hybrid block"
Cohesion: 0.20
Nodes (9): Candidate, Command, Correctness gates, Decision, EXP-0047 — M6-A5 Qwen3.8-27B four-layer hybrid block, Follow-up, Question, Reference and baseline (+1 more)

### Community 242 - "EXP-0048 — M6-A6 Qwen3.8-27B full 64-layer forward"
Cohesion: 0.22
Nodes (8): Baseline and candidate, Command, Correctness contract, Decision, EXP-0048 — M6-A6 Qwen3.8-27B full 64-layer forward, Follow-up, Question, Result

### Community 243 - "DeviceInfo"
Cohesion: 0.22
Nodes (12): DeviceInfo, architecture, index, name, total_vram_bytes, size_t, ostream, string (+4 more)

### Community 244 - "EXP-0050 — M6-B0 Qwen3.8-27B llama.cpp MI50 baseline"
Cohesion: 0.17
Nodes (11): Baseline, Combined context controls, Commands, Decision, EXP-0050 — M6-B0 Qwen3.8-27B llama.cpp MI50 baseline, Follow-up, Interpretation, Model and environment (+3 more)

### Community 245 - "EXP-0188 — Inter-Layer Norm Fusion, Native Q6_K LM Head & Wave64 Single-Wave GEMV (Stretch Gate Closure)"
Cohesion: 0.14
Nodes (13): 1. TG64 (64 tokens), 2. TG128 (128 tokens), Baseline, Benchmark Methodology, Candidate, Correctness, Decision, Environment (+5 more)

### Community 246 - "EXP-0056 — M6-A12 Qwen3.8-27B attention projections"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Checks, Decision, EXP-0056 — M6-A12 Qwen3.8-27B attention projections, Follow-up, Question, Result (+1 more)

### Community 247 - "EXP-0049 — M6-A7 Qwen3.8-27B stateful generation"
Cohesion: 0.18
Nodes (10): Baseline and candidate, Command, Correctness contract, Decision, Environment, EXP-0049 — M6-A7 Qwen3.8-27B stateful generation, Follow-up, Hypothesis (+2 more)

### Community 248 - "Qwen35Config"
Cohesion: 0.05
Nodes (45): GgufTensorType, uint64_t, find_tensor(), byte, GgufTensorType, shared_ptr, size_t, string (+37 more)

### Community 249 - "EXP-0252 — Repacked MMQ64 retest on the production Q4_K down shape"
Cohesion: 0.14
Nodes (13): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0252 — Repacked MMQ64 retest on the production Q4_K down shape, Follow-up (+5 more)

### Community 250 - "EXP-0254 — Repacked MMQ64 exact-sum activation layout"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness, Decision, Environment, EXP-0254 — Repacked MMQ64 exact-sum activation layout, Follow-up, Hypothesis (+2 more)

### Community 251 - "run-m6b0-llama-baseline.sh"
Cohesion: 0.83
Nodes (3): cleanup(), run-m6b0-llama-baseline.sh script, stop_telemetry()

### Community 252 - "qwen3_inference_bench.cpp"
Cohesion: 0.09
Nodes (48): argmax(), build_json(), ostream, size_t, string, timespec, uint32_t, vector (+40 more)

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

### Community 258 - "kquant_wave_layout.cpp"
Cohesion: 0.07
Nodes (51): main(), main(), int8_t, uint32_t, uint64_t, uint8_t, Metadata, d (+43 more)

### Community 259 - "m6a13_qwen35_full_attention_layer.cpp"
Cohesion: 0.34
Nodes (11): check(), checkpoint(), copy_to_host(), path, size_t, string_view, vector, DeviceBuffer (+3 more)

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

### Community 268 - "GgufFile"
Cohesion: 0.13
Nodes (14): GgufFile, file_descriptor_, mapping_, metadata, metadata_array_is_string, metadata_array_size, metadata_float, metadata_string (+6 more)

### Community 269 - "EXP-0096 — M6-B3 Q4_K metadata staging"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Commands, Correctness and resources, Decision, Environment, EXP-0096 — M6-B3 Q4_K metadata staging, Follow-up (+3 more)

### Community 270 - "EXP-0182 — Fused Gate+Up SwiGLU Wave64 GEMV"
Cohesion: 0.12
Nodes (15): 1. TG64 (64 tokens), 2. TG128 (128 tokens), 3. Hardware State & Telemetry, Baseline, Benchmark Methodology, Candidate, Correctness, Decision (+7 more)

### Community 271 - "Qwen35RuntimeEngine"
Cohesion: 0.05
Nodes (40): hipGraphExec_t, Buffer, unique_ptr, Qwen35RuntimeEngine, argmax_token_, attention_layers_, d_decode_tokens_, d_embedding_ (+32 more)

### Community 272 - "EXP-0066 — M6-A22 Qwen3.8-27B GPU hybrid position audit"
Cohesion: 0.17
Nodes (11): Artifact and reference, Candidate, Checks, Command, Decision, EXP-0066 — M6-A22 Qwen3.8-27B GPU hybrid position audit, Follow-up, Interpretation (+3 more)

### Community 273 - "m6a18_qwen35_deltanet_state_gpu.cpp"
Cohesion: 0.23
Nodes (13): path, size_t, span, T, vector, DeviceBuffer, data_, logical_state() (+5 more)

### Community 274 - "EXP-0141 — M6-B49 recurrent state-update/head-RMS fusion"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment, EXP-0141 — M6-B49 recurrent state-update/head-RMS fusion, Follow-up, Hypothesis, Question (+1 more)

### Community 275 - "Metrics"
Cohesion: 0.22
Nodes (9): Metrics, actual_at_max, expected_at_max, finite, max_abs, max_index, max_rel, mean_abs (+1 more)

### Community 276 - "EXP-0226 — M11-B Q4_K chunked row-tile projection"
Cohesion: 0.15
Nodes (12): Baseline, Candidate, Correctness, Decision, Environment, EXP-0226 — M11-B Q4_K chunked row-tile projection, Follow-up, Hypothesis (+4 more)

### Community 277 - "EXP-0064 — M6-A20 Qwen3.8-27B recurrent layer on GPU"
Cohesion: 0.18
Nodes (10): Artifact and reference, Candidate, Checks, Command, Decision, EXP-0064 — M6-A20 Qwen3.8-27B recurrent layer on GPU, Follow-up, Question (+2 more)

### Community 278 - "EXP-0072 — M6-A26.2 Qwen3.8 L30 update provenance"
Cohesion: 0.18
Nodes (10): Decision, Diagnostic, Environment and command, EXP-0072 — M6-A26.2 Qwen3.8 L30 update provenance, Follow-up, Interpretation, Moving maximum and tracked-index results, Question (+2 more)

### Community 279 - "EXP-0103 — M6-B10 recurrent Q8_K input reuse"
Cohesion: 0.25
Nodes (7): Candidate, Decision, Environment, EXP-0103 — M6-B10 recurrent Q8_K input reuse, Follow-up, Question, Results

### Community 280 - "EXP-0204 — Production Context Scaling After M11-A Integration"
Cohesion: 0.12
Nodes (15): Attention Scaling Analysis, Benchmark, Comparison with M10 Baseline, Decision, End-to-End Production Inference (Real Model), Environment, EXP-0204 — Production Context Scaling After M11-A Integration, Follow-up (+7 more)

### Community 281 - "EXP-0065 — M6-A21 Qwen3.8-27B GPU hybrid block"
Cohesion: 0.18
Nodes (10): Artifact and reference, Candidate, Checks, Command, Decision, EXP-0065 — M6-A21 Qwen3.8-27B GPU hybrid block, Follow-up, Question (+2 more)

### Community 283 - "EXP-0068 — M6-A24 Qwen3.8 eight-layer stateful GPU prefix"
Cohesion: 0.18
Nodes (10): Candidate, Checks, Correctness contract, Decision, EXP-0068 — M6-A24 Qwen3.8 eight-layer stateful GPU prefix, Follow-up, Performance and memory accounting, Question (+2 more)

### Community 284 - "Qwen3Model"
Cohesion: 0.14
Nodes (13): shared_ptr, size_t, string, vector, Qwen3Model, artifact_path_, config_, final_norm_ (+5 more)

### Community 285 - "EXP-0080 — M6-A27 Qwen3.8 sixty-four-layer GPU composition"
Cohesion: 0.25
Nodes (7): Candidate, Decision, Environment and command, EXP-0080 — M6-A27 Qwen3.8 sixty-four-layer GPU composition, Follow-up, Question, Results

### Community 286 - "EXP-0085 — M6-A27.5 Qwen3.8 P2 drift localization"
Cohesion: 0.18
Nodes (10): Decision, EXP-0085 — M6-A27.5 Qwen3.8 P2 drift localization, Follow-up, Hypothesis, Interpretation, Method, Observable consequence, P2 layer-output error scan (+2 more)

### Community 287 - "EXP-0070 — M6-A26 Qwen3.8 thirty-two-layer stateful GPU prefix"
Cohesion: 0.20
Nodes (9): Candidate, Decision, Environment and command, EXP-0070 — M6-A26 Qwen3.8 thirty-two-layer stateful GPU prefix, Follow-up, Interpretation, Question, Results (+1 more)

### Community 288 - "run_case"
Cohesion: 0.25
Nodes (8): Fn, string, vector, measure(), run_case(), Launch4, LaunchMm, Pack

### Community 289 - "EXP-0124 — M6-B32 transposed recurrent no-decay store"
Cohesion: 0.17
Nodes (11): Baseline and candidate, Benchmark, Commands, Correctness, Decision, Environment, EXP-0124 — M6-B32 transposed recurrent no-decay store, Follow-up (+3 more)

### Community 290 - "EXP-0156 — M6-B64 post-B62 stage profile"
Cohesion: 0.25
Nodes (7): Decision, Environment, EXP-0156 — M6-B64 post-B62 stage profile, Follow-up, Interpretation, Question, Results

### Community 291 - "EXP-0137 — M6-B45 Q4_K×Q8_1 inner-loop differential"
Cohesion: 0.22
Nodes (8): Baseline, Controls already run, Decision, EXP-0137 — M6-B45 Q4_K×Q8_1 inner-loop differential, External comparison, Interpretation, Next experiment, Question

### Community 292 - "EXP-0075 — M6-A26.5 L30 K-path provenance"
Cohesion: 0.20
Nodes (9): Decision, Environment and command, EXP-0075 — M6-A26.5 L30 K-path provenance, Follow-up, Interpretation, Method, Question, Results — L30 P19 (+1 more)

### Community 293 - "qwen3_gpu_layer.hpp"
Cohesion: 0.14
Nodes (15): vector, Qwen3DownProjectionContractTrace, current_s_correction, direct_signed_oracle, exact_sum_correction, Qwen3FfnProbeTrace, ffn_output, gate (+7 more)

### Community 294 - "EXP-0076 — M6-A26.6 L29 output provenance"
Cohesion: 0.20
Nodes (9): Decision, Environment and command, EXP-0076 — M6-A26.6 L29 output provenance, Follow-up, Interpretation, Method, Question, Results — L29 P19 (+1 more)

### Community 295 - "EXP-0077 — M6-A26.7 L29 gated-output provenance"
Cohesion: 0.20
Nodes (9): Decision, Environment and command, EXP-0077 — M6-A26.7 L29 gated-output provenance, Follow-up, Interpretation, Method, Question, Results — L29 P19 (+1 more)

### Community 296 - "EXP-0078 — M6-A26.8 L29 gate-input provenance"
Cohesion: 0.20
Nodes (9): Decision, Environment and command, EXP-0078 — M6-A26.8 L29 gate-input provenance, Follow-up, Interpretation, Method, Question, Results (+1 more)

### Community 297 - "EXP-0230 — M11-B raw int8 GEMM ceiling"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Decision, Environment, EXP-0230 — M11-B raw int8 GEMM ceiling, Follow-up, Interpretation, Question (+1 more)

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

### Community 305 - "EXP-0154 — M6-B62 DeltaNet row-wave state mapping"
Cohesion: 0.25
Nodes (7): Candidate, Correctness, Decision, Environment, EXP-0154 — M6-B62 DeltaNet row-wave state mapping, Follow-up, Question

### Community 306 - "EXP-0128 — M6-B36 post-B35 production profile"
Cohesion: 0.25
Nodes (7): Decision, EXP-0128 — M6-B36 post-B35 production profile, Follow-up, Interpretation, Method, Question, Results

### Community 307 - "EXP-0136 — M6-B44 Q4_K×Q8_1 split-K MMVQ mapping"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment, EXP-0136 — M6-B44 Q4_K×Q8_1 split-K MMVQ mapping, Follow-up, Interpretation, Question (+1 more)

### Community 308 - "EXP-0168 — post-B1 whole-token profile"
Cohesion: 0.29
Nodes (6): Decision, Environment, EXP-0168 — post-B1 whole-token profile, Interpretation, Question, Results

### Community 309 - "EXP-0113 — M6-B20 Q6_K×Q8_1 MMVQ recurrent QKV"
Cohesion: 0.25
Nodes (7): Candidate, Checks, Decision, Environment, EXP-0113 — M6-B20 Q6_K×Q8_1 MMVQ recurrent QKV, Question, Result

### Community 310 - "EXP-0082 — M6-A27.2 Qwen3.8 L53/P1 output provenance"
Cohesion: 0.29
Nodes (6): Decision, EXP-0082 — M6-A27.2 Qwen3.8 L53/P1 output provenance, Follow-up, Method, Question, Results

### Community 311 - "EXP-0163 — M6-B67 post-B66 recurrent profile"
Cohesion: 0.29
Nodes (6): Decision, Environment, EXP-0163 — M6-B67 post-B66 recurrent profile, Interpretation, Question, Results

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

### Community 316 - "EXP-0140 — M6-B48 persistent Q4_K FFN Down metadata"
Cohesion: 0.20
Nodes (9): Baseline and candidate, Baseline profile, Correctness, Decision, Environment, EXP-0140 — M6-B48 persistent Q4_K FFN Down metadata, Follow-up, Question (+1 more)

### Community 317 - "EXP-0102 — M6-B9 Q6_K LM-head index hoisting"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness, Decision, Environment, EXP-0102 — M6-B9 Q6_K LM-head index hoisting, Follow-up, Interpretation (+2 more)

### Community 318 - "Qwen3GpuDecodeCache"
Cohesion: 0.10
Nodes (21): size_t, unique_ptr, Qwen3GpuDecodeCache, caches_, prepare, workspace_, Qwen3Layer0GpuKvCache, append (+13 more)

### Community 319 - "EXP-0094 — M6-B2 direct layer-output handoff"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Decision, Environment and workload, EXP-0094 — M6-B2 direct layer-output handoff, Follow-up, Hypothesis, Interpretation (+2 more)

### Community 320 - "require_match"
Cohesion: 0.18
Nodes (16): check_device(), check_values(), Metrics, path, size_t, span, T, uint32_t (+8 more)

### Community 321 - "EXP-0129 — M6-B37 Q4_K×Q8_1 Down weight staging"
Cohesion: 0.18
Nodes (10): Baseline and candidate, Benchmark, Correctness, Decision, Environment, EXP-0129 — M6-B37 Q4_K×Q8_1 Down weight staging, Follow-up, Hypothesis (+2 more)

### Community 322 - "EXP-0149 — M6-B57 direct layer output"
Cohesion: 0.18
Nodes (10): Candidate, Correctness, Decision, Environment, EXP-0149 — M6-B57 direct layer output, Follow-up, Hypothesis, Interpretation (+2 more)

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

### Community 327 - "EXP-0127 — M6-B35 Q4_K×Q8_1 LDS activation reuse"
Cohesion: 0.18
Nodes (10): Baseline and candidate, Benchmark, Correctness, Decision, Environment, EXP-0127 — M6-B35 Q4_K×Q8_1 LDS activation reuse, Follow-up, Hypothesis (+2 more)

### Community 328 - "EXP-0104 — M6-B11 Q4_K packed dot4 projections"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness and resources, Decision, Environment, EXP-0104 — M6-B11 Q4_K packed dot4 projections, Follow-up, Interpretation (+2 more)

### Community 329 - "EXP-0122 — M6-B30 transposed DeltaNet recurrent state"
Cohesion: 0.17
Nodes (11): Baseline and candidate, Benchmark, Commands, Correctness, Decision, Environment, EXP-0122 — M6-B30 transposed DeltaNet recurrent state, Follow-up (+3 more)

### Community 330 - "Qwen3GpuPlan"
Cohesion: 0.07
Nodes (41): GpuWeightArena, allocate, release, upload, GgufTensorType, size_t, string, uint64_t (+33 more)

### Community 331 - "EXP-0111 — M6-B18 Q4_K×Q8_1 MMVQ FFN Down"
Cohesion: 0.17
Nodes (11): Baseline and candidate, Commands, Correctness, Decision, Environment, EXP-0111 — M6-B18 Q4_K×Q8_1 MMVQ FFN Down, Follow-up, Hypothesis (+3 more)

### Community 332 - "EXP-0147 — M6-B55 DeltaNet LDS input reuse"
Cohesion: 0.25
Nodes (7): Candidate, Decision, Environment, EXP-0147 — M6-B55 DeltaNet LDS input reuse, Follow-up, Question, Results

### Community 333 - "EXP-0175 — MI50 sustained operating-point qualification"
Cohesion: 0.25
Nodes (7): Decision, Environment, EXP-0175 — MI50 sustained operating-point qualification, Hypothesis, Interpretation, Next step, Results

### Community 334 - "EXP-0108 — M6-B15 recurrent state-update no-decay-store candidate"
Cohesion: 0.20
Nodes (9): Baseline and candidate, Benchmark, Correctness, Decision, Environment and workload, EXP-0108 — M6-B15 recurrent state-update no-decay-store candidate, Follow-up, Profile (+1 more)

### Community 335 - "EXP-0131 — M6-B39 Q4_K×Q8_1 metadata staging"
Cohesion: 0.18
Nodes (10): Baseline and candidate, Benchmark, Correctness, Decision, Environment, EXP-0131 — M6-B39 Q4_K×Q8_1 metadata staging, Follow-up, Hypothesis (+2 more)

### Community 336 - "EXP-0115 — M6-B22 Q6_K×Q8_K packed-dot4 recurrent QKV"
Cohesion: 0.20
Nodes (9): Candidate and control, Correctness and resources, Decision, Environment, EXP-0115 — M6-B22 Q6_K×Q8_K packed-dot4 recurrent QKV, Follow-up, Interpretation, Question (+1 more)

### Community 337 - "EXP-0159 — M6-B67 transposed versus non-transposed full-model control"
Cohesion: 0.33
Nodes (5): Decision, Environment, EXP-0159 — M6-B67 transposed versus non-transposed full-model control, Question, Results

### Community 338 - "EXP-0120 — M6-B28 post-B27 production profile"
Cohesion: 0.20
Nodes (9): Decision, Environment, EXP-0120 — M6-B28 post-B27 production profile, Follow-up, Interpretation, Question, Ranking, Results (+1 more)

### Community 339 - "EXP-0112 — M6-B19 Q4_K×Q8_1 MMVQ FFN Gate/Up"
Cohesion: 0.18
Nodes (10): Candidate and control, Correctness and resource checks, Decision, Environment, EXP-0112 — M6-B19 Q4_K×Q8_1 MMVQ FFN Gate/Up, Follow-up, Hypothesis, Interpretation (+2 more)

### Community 340 - "EXP-0246 — M11-B Q4_K MMQ64 split-4 rejection"
Cohesion: 0.17
Nodes (11): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0246 — M11-B Q4_K MMQ64 split-4 rejection, Follow-up (+3 more)

### Community 341 - "EXP-0121 — M6-B29 recurrent stage attribution"
Cohesion: 0.22
Nodes (8): Decision, Environment, EXP-0121 — M6-B29 recurrent stage attribution, Follow-up, Question, Ranking, Results, Scope

### Community 342 - "kquant_layout_bench.cpp"
Cohesion: 0.16
Nodes (12): as(), Buffer, p, hipEvent_t, size_t, string, T, Event (+4 more)

### Community 343 - "EXP-0180 — M7 gfx906 Performance Frontier Benchmark"
Cohesion: 0.17
Nodes (11): Benchmark Methodology, Competitive Candidates Evaluated, Decision, Decode Throughput (tokens/second), Environment & Hardware Qualification, EXP-0180 — M7 gfx906 Performance Frontier Benchmark, Hypothesis, Key Findings (+3 more)

### Community 344 - "EXP-0216 — M11-B Recurrent Q5_K `ssm_out` B=4 Projection"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness, Decision, Environment, EXP-0216 — M11-B Recurrent Q5_K `ssm_out` B=4 Projection, Follow-up, Hypothesis (+2 more)

### Community 345 - "EXP-0135 — M6-B43 Q6_K×Q8_1 LM-head metadata staging"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0135 — M6-B43 Q6_K×Q8_1 LM-head metadata staging, Follow-up, Question, Results

### Community 346 - "EXP-0126 — M6-B34 fused SiLU to Q8_1"
Cohesion: 0.18
Nodes (10): Baseline and candidate, Benchmark, Correctness, Decision, Environment, EXP-0126 — M6-B34 fused SiLU to Q8_1, Follow-up, Hypothesis (+2 more)

### Community 347 - "EXP-0117 — M6-B25 fine full-attention attribution"
Cohesion: 0.20
Nodes (9): Baseline, Correctness and resources, Decision, Environment, EXP-0117 — M6-B25 fine full-attention attribution, Follow-up, Interpretation, Question (+1 more)

### Community 348 - "EXP-0130 — M6-B38 post-B37 production profile"
Cohesion: 0.25
Nodes (7): Decision, Environment, EXP-0130 — M6-B38 post-B37 production profile, Interpretation, Method, Question, Results

### Community 349 - "EXP-0118 — M6-B26 Q-projection Q8_1 MMVQ candidate"
Cohesion: 0.22
Nodes (8): Baseline, Correctness, Decision, Environment, EXP-0118 — M6-B26 Q-projection Q8_1 MMVQ candidate, Follow-up, Interpretation, Question

### Community 350 - "EXP-0138 — M6-B46 Q4_K×Q8_1 packed-input diagnostic"
Cohesion: 0.22
Nodes (8): Baseline and candidate, Correctness, Decision, Environment, EXP-0138 — M6-B46 Q4_K×Q8_1 packed-input diagnostic, Follow-up, Question, Results

### Community 351 - "EXP-0133 — M6-B41 Q4_K×Q8_1 decoded metadata staging"
Cohesion: 0.20
Nodes (9): Baseline and candidate, Benchmark, Correctness, Decision, Environment, EXP-0133 — M6-B41 Q4_K×Q8_1 decoded metadata staging, Follow-up, Interpretation (+1 more)

### Community 352 - "EXP-0134 — M6-B42 post-B41 production profile"
Cohesion: 0.29
Nodes (6): Decision, Environment, EXP-0134 — M6-B42 post-B41 production profile, Follow-up, Question, Results

### Community 353 - "EXP-0139 — M6-B47 dual Gate/Up Q4_K×Q8_1 projection"
Cohesion: 0.22
Nodes (8): Baseline and candidate, Correctness, Decision, Environment, EXP-0139 — M6-B47 dual Gate/Up Q4_K×Q8_1 projection, Follow-up, Question, Results

### Community 354 - "EXP-0151 — M6-B59 expanded Q4_K FFN Down weights"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment, EXP-0151 — M6-B59 expanded Q4_K FFN Down weights, Follow-up, Interpretation, Question (+1 more)

### Community 355 - "EXP-0150 — M6-B58 Q6_K×Q8_K QKV LDS input"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment, EXP-0150 — M6-B58 Q6_K×Q8_K QKV LDS input, Follow-up, Interpretation, Question (+1 more)

### Community 356 - "EXP-0162 — M6-B66 dual beta/alpha FP32 GEMV"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0162 — M6-B66 dual beta/alpha FP32 GEMV, Follow-up, Question, Results

### Community 357 - "EXP-0132 — M6-B40 post-B39 production profile"
Cohesion: 0.29
Nodes (6): Decision, EXP-0132 — M6-B40 post-B39 production profile, Interpretation, Method and environment, Question, Results

### Community 358 - "EXP-0142 — M6-B50 fused recurrent output path"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0142 — M6-B50 fused recurrent output path, Follow-up, Question, Results

### Community 359 - "EXP-0167 — post-A28 native generation baseline"
Cohesion: 0.25
Nodes (7): Commands, Decision, Environment, EXP-0167 — post-A28 native generation baseline, Follow-up, Question, Results

### Community 360 - "EXP-0262 — M12 production qualification gate"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Decision, Environment, EXP-0262 — M12 production qualification gate, Follow-up, Hypothesis, Interpretation (+1 more)

### Community 361 - "EXP-0144 — M6-B52 Q4_K×Q8_1 one-Wave64 row mapping"
Cohesion: 0.25
Nodes (7): Candidate, Decision, Environment, EXP-0144 — M6-B52 Q4_K×Q8_1 one-Wave64 row mapping, Follow-up, Question, Results

### Community 362 - "EXP-0189 — Wave64-Native Shuffle-Based Q8_1 Activation Quantizer (Lane 1)"
Cohesion: 0.14
Nodes (13): Baseline, Benchmark Results, Candidate, Correctness, Decision, Environment, EXP-0189 — Wave64-Native Shuffle-Based Q8_1 Activation Quantizer (Lane 1), Follow-up (+5 more)

### Community 363 - "EXP-0172 — Reference Q4K inner-loop load schedule"
Cohesion: 0.22
Nodes (8): Baseline, Candidate, Correctness, Decision, EXP-0172 — Reference Q4K inner-loop load schedule, Follow-up, Hypothesis, Results

### Community 364 - "EXP-0232 — M11-B Q4_K 16-token MMQ tile"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Correctness, Decision, Environment, EXP-0232 — M11-B Q4_K 16-token MMQ tile, Follow-up, Hypothesis (+3 more)

### Community 365 - "EXP-0148 — M6-B56 post-B55 production profile"
Cohesion: 0.25
Nodes (7): Decision, Environment, EXP-0148 — M6-B56 post-B55 production profile, Follow-up, Interpretation, Question, Results

### Community 366 - "EXP-0143 — M6-B51 post-B50 production profile"
Cohesion: 0.20
Nodes (9): Baseline, Correctness, Decision, Environment, EXP-0143 — M6-B51 post-B50 production profile, Follow-up, Interpretation, Question (+1 more)

### Community 367 - "EXP-0211 — M11-B Row-LDS B=8 Production-Shape Rejection"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Correctness, Decision, Environment, EXP-0211 — M11-B Row-LDS B=8 Production-Shape Rejection, Follow-up, Hypothesis (+1 more)

### Community 369 - "EXP-0164 — M6-B68 fused beta/alpha preparation"
Cohesion: 0.25
Nodes (7): Candidate, Correctness, Decision, Environment, EXP-0164 — M6-B68 fused beta/alpha preparation, Question, Results

### Community 370 - "EXP-0145 — M6-B53 Q4_K×Q8_K versus Q4_K×Q8_1 FFN differential"
Cohesion: 0.25
Nodes (7): Candidate, Decision, Environment, EXP-0145 — M6-B53 Q4_K×Q8_K versus Q4_K×Q8_1 FFN differential, Follow-up, Question, Results

### Community 371 - "EXP-0191: Fused Residual RMS Norm with Direct Q8_1 Handoff"
Cohesion: 0.14
Nodes (13): Baseline (Control), Candidate, Correctness, Decision, Environment, EXP-0191: Fused Residual RMS Norm with Direct Q8_1 Handoff, Follow-up, Hypothesis (+5 more)

### Community 372 - "EXP-0171 — M6-B76 architectural blocker report"
Cohesion: 0.29
Nodes (6): Conclusion, Current differential, EXP-0171 — M6-B76 architectural blocker report, Gate, Re-evaluation — EXP-0172, Required redesign

### Community 373 - "EXP-0152 — M6-B60 post-B59 production profile"
Cohesion: 0.22
Nodes (8): Baselines, Decision, Environment, EXP-0152 — M6-B60 post-B59 production profile, Follow-up, Interpretation, Question, Results

### Community 374 - "PrefillProfile"
Cohesion: 0.18
Nodes (10): array, hipEvent_t, PrefillProfile, chunks, embedding_end, embedding_ms, embedding_start, enabled (+2 more)

### Community 375 - "EXP-0237 — M11-B production layer-major prefill profile"
Cohesion: 0.22
Nodes (8): Amdahl interpretation, Decision, Environment, EXP-0237 — M11-B production layer-major prefill profile, Follow-up, Method, Question, Results

### Community 376 - "EXP-0165 — M6-B69 dual query/key head normalization"
Cohesion: 0.25
Nodes (7): Candidate, Correctness, Decision, Environment, EXP-0165 — M6-B69 dual query/key head normalization, Question, Results

### Community 377 - "EXP-0160 — M6-B68 DeltaNet ordered row-wave reduction"
Cohesion: 0.33
Nodes (5): Candidate, Decision, EXP-0160 — M6-B68 DeltaNet ordered row-wave reduction, Question, Results

### Community 378 - "UpdateProvenance"
Cohesion: 0.12
Nodes (15): UpdateProvenance, beta, candidate, column, decay, decayed, delta, head (+7 more)

### Community 379 - "EXP-0166 — M6-B70 column-tiled DeltaNet state update"
Cohesion: 0.33
Nodes (5): Candidate, Decision, EXP-0166 — M6-B70 column-tiled DeltaNet state update, Question, Result

### Community 380 - "EXP-0192: Wave64 Fused Gate+Up SwiGLU Intra-Wave Shuffle Reduction"
Cohesion: 0.14
Nodes (13): Baseline (Control), Candidate, Correctness, Decision, Environment, EXP-0192: Wave64 Fused Gate+Up SwiGLU Intra-Wave Shuffle Reduction, Follow-up, Hypothesis (+5 more)

### Community 381 - "EXP-0176 — MI50 manual-DPM qualification and EXP-0174 re-adjudication"
Cohesion: 0.20
Nodes (9): Amdahl reconciliation, Configuration, Correctness, Decision, EXP-0174 A/B, EXP-0176 — MI50 manual-DPM qualification and EXP-0174 re-adjudication, Hypothesis, Next action (+1 more)

### Community 382 - "2. Command Architecture"
Cohesion: 0.12
Nodes (16): 1. Executive Summary, 2.1 `miinfer inspect`, 2.1a `miinfer config`, 2.1b `miinfer models`, 2.2 `miinfer run`, 2.3 `miinfer chat`, 2.4 `miinfer serve`, 2. Command Architecture (+8 more)

### Community 383 - "EXP-0158 — M6-B66 expanded Q4_K FFN Gate/Up"
Cohesion: 0.29
Nodes (6): Candidate, Correctness, Decision, Environment, EXP-0158 — M6-B66 expanded Q4_K FFN Gate/Up, Question

### Community 384 - "Checkpoint"
Cohesion: 0.40
Nodes (5): Checkpoint, file, miinfer, name, tolerance

### Community 385 - "EXP-0155 — M6-B63 post-B62 production baseline"
Cohesion: 0.25
Nodes (7): Baseline, Decision, Environment, EXP-0155 — M6-B63 post-B62 production baseline, Follow-up, Question, Results

### Community 386 - "EXP-0190 — Fused Recurrent & Attention Epilogue Q8_1 Activation Quantization (Lane 2)"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark Results, Candidate, Correctness, Decision, Environment, EXP-0190 — Fused Recurrent & Attention Epilogue Q8_1 Activation Quantization (Lane 2), Hypothesis (+4 more)

### Community 387 - "EXP-0231 — M11-B Q4_K repacked MMQ tile"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Correctness, Decision, Environment, EXP-0231 — M11-B Q4_K repacked MMQ tile, Follow-up, Hypothesis (+3 more)

### Community 388 - "EXP-0173 — Q4K native-layout laboratory"
Cohesion: 0.22
Nodes (8): Baseline harness, EXP-0173 — Q4K native-layout laboratory, First physical layout proposal (not implemented or tested yet), Layout 1 — implementation and initial primitive evidence, Pinned reference representation, Preliminary run — NOT valid for stable-peak acceptance, Scope and acceptance, Status

### Community 389 - "EXP-0196 — Combined Projections for Recurrent (QKV+Gate) and Attention (Q+K) Layers"
Cohesion: 0.13
Nodes (14): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0196 — Combined Projections for Recurrent (QKV+Gate) and Attention (Q+K) Layers, Follow-up (+6 more)

### Community 390 - "EXP-0181 — Static HIP Graph Capture for Autoregressive Decode"
Cohesion: 0.12
Nodes (15): 1. TG64 (64 tokens), 2. TG128 (128 tokens), 3. Hardware State & Telemetry, Baseline, Benchmark Methodology, Candidate, Correctness, Decision (+7 more)

### Community 391 - "Candidate 1: FFN Gate & Up Rollout"
Cohesion: 0.20
Nodes (10): A/B Benchmark Results (Interleaved 5 Pairs), Candidate 1: FFN Gate & Up Rollout, Correctness, Decision, Gap to llama.cpp Status, Hardware State Validation, Implementation Scope, P63 GPU Profile Breakdown (+2 more)

### Community 392 - "EXP-0218 — M11-B Q4_K Word-Reuse B=8 Rejection"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Correctness, Decision, Environment, EXP-0218 — M11-B Q4_K Word-Reuse B=8 Rejection, Follow-up, Hypothesis (+1 more)

### Community 393 - "EXP-0174 — Wave-plane Q4K Down integration"
Cohesion: 0.22
Nodes (8): Completed integration checks and interpretation, Correctness so far, Decision, End-to-end diagnostic run (in progress), EXP-0174 — Wave-plane Q4K Down integration, Hardware revalidation — 2026-09-05, Hypothesis and scope, Resources (compiler metadata, not hardware counters)

### Community 394 - "m12_dense_stage_bench.cpp"
Cohesion: 0.10
Nodes (25): as(), Buffer, pointer, check_hipblas(), Fn, hipEvent_t, size_t, T (+17 more)

### Community 395 - "EXP-0169 — M6-B74 pinned llama.cpp architecture differential"
Cohesion: 0.22
Nodes (8): Amdahl filter, Decision, Evidence, EXP-0169 — M6-B74 pinned llama.cpp architecture differential, Prior-art guard, Question, Re-evaluation, Required differential report

### Community 396 - "EXP-0170 — M6-B75 fused Gate/Up/SwiGLU prototype"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, EXP-0170 — M6-B75 fused Gate/Up/SwiGLU prototype, Hypothesis, Interpretation, Results, Revised implementation

### Community 397 - "Candidate 2: Full-Attention Q Projection Rollout"
Cohesion: 0.20
Nodes (10): A/B Benchmark Results (Interleaved 5 Pairs), Candidate 2: Full-Attention Q Projection Rollout, Correctness, Cumulative Gap to llama.cpp Status, Decision, Hardware State Validation, Implementation Scope, P63 GPU Profile Breakdown (+2 more)

### Community 398 - "EXP-0193: Vectorized SIMD Q6_K Decoding for LM Head and Down Projections"
Cohesion: 0.17
Nodes (11): Baseline (Control), Candidate, Correctness, Decision, Environment, EXP-0193: Vectorized SIMD Q6_K Decoding for LM Head and Down Projections, Hypothesis, Mechanism (+3 more)

### Community 399 - "m6a8_qwen35_gpu_foundation.cpp"
Cohesion: 0.33
Nodes (7): path, size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 400 - "EXP-0228 — M11-B Q4_K four-token subwave mapping"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Correctness, Decision, Environment, EXP-0228 — M11-B Q4_K four-token subwave mapping, Follow-up, Hypothesis (+3 more)

### Community 401 - "5. Runtime Layers"
Cohesion: 0.29
Nodes (7): 5.1 Model Layer, 5.2 Packing / Representation Layer, 5.3 Memory Planner, 5.4 Kernel Planner, 5.5 Execution Plan, 5.6 Kernel Layer, 5. Runtime Layers

### Community 402 - "EXP-0197 — Paired Fused Gate+Up SwiGLU with 64-Bit dwordx2 Coalesced Memory Access"
Cohesion: 0.13
Nodes (14): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0197 — Paired Fused Gate+Up SwiGLU with 64-Bit dwordx2 Coalesced Memory Access, Follow-up (+6 more)

### Community 403 - "Candidate 3: Recurrent Attention Gate Rollout"
Cohesion: 0.20
Nodes (10): A/B Benchmark Results (Interleaved 5 Pairs), Candidate 3: Recurrent Attention Gate Rollout, Correctness, Cumulative Gap to llama.cpp Status, Decision, Hardware State Validation, Implementation Scope, P63 GPU Profile Breakdown (+2 more)

### Community 404 - "launch_projection"
Cohesion: 0.20
Nodes (15): Q8_1Block, Qwen3BoundaryProfileStage, Qwen3FfnProfileStage, uint32_t, launch_projection(), projection_ffn_quantization_stage(), projection_ffn_stage(), projection_input_f16_stage() (+7 more)

### Community 405 - "Candidate 4: Full-Attention Output Projection Rollout"
Cohesion: 0.20
Nodes (10): A/B Benchmark Results (Interleaved 5 Pairs), Candidate 4: Full-Attention Output Projection Rollout, Correctness, Cumulative Gap to llama.cpp Status, Decision, Hardware State Validation, Implementation Scope, P63 GPU Profile Breakdown (+2 more)

### Community 406 - "Milestone M9 Primary Gate Qualification Report"
Cohesion: 0.14
Nodes (13): 1. Gate Scorecard, 2. Optimization Trajectory & Latency Reduction, 3. Milestone M9 Experiment Ledger, 4. 5-Pair Interleaved Primary Gate Benchmark (TG64), 5. Secondary Context Scaling Benchmark (TG128 & TG256), 6. Numerical Correctness & Observable Contract, 7. Usable Standalone Runtime CLI (`miinfer`), 8. Telemetry & Hardware State Audit (+5 more)

### Community 407 - "EXP-0194 — Vectorized Q4_K and Q5_K Fast Tile Arithmetic Optimization"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0194 — Vectorized Q4_K and Q5_K Fast Tile Arithmetic Optimization, Follow-up (+4 more)

### Community 408 - "EXP-0185 — Tiled Online-Softmax Attention with Gate Sigmoid Fusion"
Cohesion: 0.12
Nodes (15): 1. TG64 (64 tokens), 2. TG128 (128 tokens), 3. Hardware State & Telemetry, Baseline, Benchmark Methodology, Candidate, Correctness, Decision (+7 more)

### Community 409 - "RecurrentOperands"
Cohesion: 0.08
Nodes (29): download(), GatePathCapture, gate, gated, head_norm, head_scaled, normalized, recurrent_output (+21 more)

### Community 410 - "run-m7-frontier-benchmark.sh"
Cohesion: 0.83
Nodes (3): run_llama_candidate(), run_miinfer(), run-m7-frontier-benchmark.sh script

### Community 411 - "EXP-0222 — M11-B Final Chunk Pointer Correction"
Cohesion: 0.33
Nodes (5): Decision, EXP-0222 — M11-B Final Chunk Pointer Correction, Finding, Question, Verification

### Community 412 - "Architectural Design: DeltaNet Recurrent SSM Core Fusion"
Cohesion: 0.22
Nodes (8): 1. Context and Problem Statement, 2.1 Workgroup and Wave Mapping, 2.2 Execution Plan within a Workgroup, 2. Proposed Architecture: Fused Recurrent Core, 3. Amdahl Impact & Expected Savings, 4. Implementation & Validation Sequence, Architectural Design: DeltaNet Recurrent SSM Core Fusion, Overhead Identified:

### Community 413 - "Architectural Design: Fused LM-Head GEMV and Argmax Reduction"
Cohesion: 0.22
Nodes (8): 1. Context and Current Bottleneck, 2.1 Two-Stage Hierarchical Reduction, 2.2 Numerical Invariance Guarantee, 2. Proposed Architecture: Fused Q6_K LM-Head + Argmax, 3. Projected Gains, 4. Rollout Preconditions, Architectural Design: Fused LM-Head GEMV and Argmax Reduction, Architectural Inefficiency:

### Community 414 - "Candidate 5: Full-Attention K Projection Rollout"
Cohesion: 0.22
Nodes (9): A/B Benchmark Results (Interleaved 5 Pairs), Candidate 5: Full-Attention K Projection Rollout, Correctness, Decision, Hardware State Validation, Implementation Scope, P63 GPU Profile Breakdown, TG128 (+1 more)

### Community 415 - "Architectural Analysis: HIP Graph Capture and Kernel Dispatch Overhead"
Cohesion: 0.25
Nodes (7): 1.1 Host-Side Launch Profiling, 1. Motivation and Dispatch Accounting, 2.1 Topology Constraints and Specializations, 2. HIP Graph Capture Strategy, 3. Measured & Projected Impact, 4. Integration Roadmap, Architectural Analysis: HIP Graph Capture and Kernel Dispatch Overhead

### Community 416 - "EXP-0201 — Real-World Latency Curve & Hybrid Context Scaling Analysis"
Cohesion: 0.12
Nodes (16): 1. Real-World Latency Curve Scorecard, 2. Second Chat Turn Continuation, 3. Context Scaling Microbenchmark & Memory Curve (128 -> 65,536 tokens), A. The Invariant Recurrent Foundation ($O(1)$), B. The 16 Attention Layers & Bandwidth Roofline, Baseline, Benchmark Methodology, C. The Prefill Latency Bottleneck (+8 more)

### Community 417 - "EXP-0177 — Native Q4_K layout rollout across projection families"
Cohesion: 0.29
Nodes (6): EXP-0177 — Native Q4_K layout rollout across projection families, Hypothesis, Motivation, Parity Target, Phase 0: External llama.cpp baseline refresh, Phase 1: Workload analysis & Amdahl ceiling

### Community 419 - "EXP-0177 Campaign Summary & Parity Reconciliation"
Cohesion: 0.50
Nodes (4): Architectural Invariants Preserved, EXP-0177 Campaign Summary & Parity Reconciliation, Full Rollout Scorecard, Parity Gap Reconciliation vs Pinned llama.cpp

### Community 425 - "EXP-0187 — Vectorized Wave64 RMS Norm & Fused Residual Addition"
Cohesion: 0.12
Nodes (15): 1. TG64 (64 tokens), 2. TG128 (128 tokens), 3. Hardware State & Telemetry, Baseline, Benchmark Methodology, Candidate, Correctness, Decision (+7 more)

### Community 428 - "hip_smoke_bench.cpp"
Cohesion: 0.16
Nodes (15): size_t, string, json_escape(), main(), Options, device, elements, iterations (+7 more)

### Community 429 - "EXP-0243 — M11-B SwiGLU/Q8 producer-consumer fusion rejection"
Cohesion: 0.20
Nodes (9): Baseline and candidate, Correctness, Decision, Environment, EXP-0243 — M11-B SwiGLU/Q8 producer-consumer fusion rejection, Follow-up, Hypothesis, Interpretation (+1 more)

### Community 430 - "RuntimeGenerateStats"
Cohesion: 0.17
Nodes (12): vector, RuntimeGenerateStats, decode_ms, decode_tok_s, first_token_ms, generated_tokens, prefill_ms, prefill_tok_s (+4 more)

### Community 433 - "EXP-0195: 1-Wave-Per-Row 0-LDS Fused SwiGLU Kernel Evaluation"
Cohesion: 0.18
Nodes (10): Analysis & Root Cause, Baseline (Control), Benchmark Results (TG64), Candidate, Decision, EXP-0195: 1-Wave-Per-Row 0-LDS Fused SwiGLU Kernel Evaluation, Follow-up, Hypothesis (+2 more)

### Community 434 - "EXP-0200 — Device-Side Token Chaining in HIP Graph Decode Loop"
Cohesion: 0.18
Nodes (10): 5-Pair Interleaved A/B Benchmark (TG64), Baseline (Config A), Candidate (Config B), Correctness, Decision, Environment, EXP-0200 — Device-Side Token Chaining in HIP Graph Decode Loop, Hypothesis (+2 more)

### Community 436 - "EXP-0261 — M13 quantized matrix prefill"
Cohesion: 0.17
Nodes (11): A0 profile, Baseline, Candidate, Correctness, Decision, Environment, EXP-0261 — M13 quantized matrix prefill, Follow-up (+3 more)

### Community 442 - "EXP-0186 — Fused RoPE + Head Norm in Full Attention Layers"
Cohesion: 0.13
Nodes (14): 1. TG64 (64 tokens), 2. TG128 (128 tokens), 3. Hardware State & Telemetry, Baseline, Benchmark Methodology, Candidate, Correctness, Decision (+6 more)

### Community 446 - "EXP-0236 — M11-B direct attention Q8_1 emission"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Correctness, Decision, Environment, EXP-0236 — M11-B direct attention Q8_1 emission, Follow-up, Hypothesis (+1 more)

### Community 447 - "M1 — Kernel Laboratory"
Cohesion: 0.29
Nodes (7): Benchmark harness, Exit criteria, Goal, Initial kernel areas, Initial representative shapes, M1 — Kernel Laboratory, Questions

### Community 448 - "Milestone M8 Primary Gate Qualification Report"
Cohesion: 0.33
Nodes (5): 1. Gate Scorecard, 2. Optimization Trajectory & Latency Reduction, 3. Experiment Ledger, 4. Telemetry Validation, Milestone M8 Primary Gate Qualification Report

### Community 449 - "EXP-0205 — Batched Q4_K Wave GEMV for Prefill"
Cohesion: 0.15
Nodes (12): B=4 Sweet Spot Analysis, B=8+ Regression, Benchmark, Decision, Environment, EXP-0205 — Batched Q4_K Wave GEMV for Prefill, Follow-up, Hypothesis (+4 more)

### Community 450 - "Milestone M10 — Real-World Inference Performance & Hybrid Context Scaling"
Cohesion: 0.14
Nodes (13): 1. Executive Summary, 2. Real-World Latency Curve Across Production Modes, 3. Hybrid Architecture Context Scaling Analysis (128 -> 65,536 tokens), 4. Key Architectural Findings & Bottlenecks, 5. M10 Implementation Roadmap, Finding 1: The DeltaNet $O(1)$ Recurrent State Advantage, Finding 2: The Attention Bandwidth Roofline at 64K Context, Finding 3: The Current Attention Kernel Bottleneck at Long Context (+5 more)

### Community 453 - ".generate"
Cohesion: 0.14
Nodes (14): GenerateOptions, GenerateStats, time_point, size_t, span, uint32_t, RuntimeGenerateOptions, max_new_tokens (+6 more)

### Community 454 - "EXP-0253 — Repacked MMQ64 with exact Q8 side sums"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness, Decision, Environment, EXP-0253 — Repacked MMQ64 with exact Q8 side sums, Follow-up, Hypothesis (+2 more)

### Community 455 - "EXP-0210 — M11-B Amdahl Profile and Sequential-Floor Check"
Cohesion: 0.33
Nodes (5): Amdahl implication, Decision, EXP-0210 — M11-B Amdahl Profile and Sequential-Floor Check, Hypothesis, Measurement

### Community 456 - "qwen3_cached_attention_determinism_gpu_test.cpp"
Cohesion: 0.33
Nodes (9): size_t, T, vector, DeviceBuffer, data_, download(), main(), same_bytes() (+1 more)

### Community 457 - "EXP-0213 — M11-B FP16 GEMM with On-Device K-Quant Dequantization"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Correctness, Decision, Environment, EXP-0213 — M11-B FP16 GEMM with On-Device K-Quant Dequantization, Follow-up, Hypothesis (+3 more)

### Community 458 - "m6a10_qwen35_q4k_projection.cpp"
Cohesion: 0.33
Nodes (7): path, size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 459 - "EXP-0202 — FP32 vs FP16 KV Cache Numerical & Performance Investigation"
Cohesion: 0.20
Nodes (9): Benchmark, Decision, Environment, EXP-0202 — FP32 vs FP16 KV Cache Numerical & Performance Investigation, Follow-up, Hypothesis, Motivation, Profiling & Architectural Interpretation (+1 more)

### Community 460 - "EXP-0199 — Multi-Wave Cooperative Q6_K FFN Down GEMV"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Decision, Environment, EXP-0199 — Multi-Wave Cooperative Q6_K FFN Down GEMV, Hypothesis, Microbenchmark & Correctness, Motivation (+1 more)

### Community 461 - "EXP-0203 — Wave64 Barrier-Free Vectorized Split-K Attention"
Cohesion: 0.20
Nodes (9): Benchmark, Decision, Environment, EXP-0203 — Wave64 Barrier-Free Vectorized Split-K Attention, Follow-up, Hypothesis, Motivation, Profiling & Gate Evaluation (+1 more)

### Community 462 - "EXP-0206 — Layer-Major Chunked Prefill"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Correctness, Decision, EXP-0206 — Layer-Major Chunked Prefill, Follow-up, Hypothesis, Re-evaluation — EXP-0208 (+1 more)

### Community 463 - "EXP-0217 — M11-B Deferred Attention Tail and Shape-Specific Q5 Reuse"
Cohesion: 0.20
Nodes (9): Candidate, Causal correctness, Decision, Decode check, Environment, EXP-0217 — M11-B Deferred Attention Tail and Shape-Specific Q5 Reuse, Follow-up, Hypothesis (+1 more)

### Community 464 - "2. Autoregressive Test Results"
Cohesion: 0.25
Nodes (7): 1. Scope & Objective, 256 Tokens Generation (`--generate256`), 2. Autoregressive Test Results, 3. Stability & Determinism Verdict, 512 Tokens Generation (`--generate512`), 64 Tokens Generation (`--generate64`), Milestone M9 — Real Autoregressive Generation Qualification

### Community 465 - "M9 — Numerical Error Budget & Attribution Report"
Cohesion: 0.25
Nodes (7): 1. Executive Summary, 2. 64-Layer Cumulative Error Progression, 3. Observable Contract & Top-K Ranking Stability, 4.1 Quantization Noise vs Compute Precision, 4. Kernel-Level Error Attribution, 5. M9 Numerical Budget Rules for Lane B Optimizations, M9 — Numerical Error Budget & Attribution Report

### Community 466 - "M9 — Post-M8 Latency Floor & Decomposition Report"
Cohesion: 0.25
Nodes (7): 1. Executive Summary, 2. Model Weight & Compulsory Memory Inventory, 3. Bandwidth Analysis & Physical Floors, 4.1 Structural Inefficiencies in the Current Execution Path, 4. Post-M8 Execution Stage Latency Attribution, 5. Ranked M9 Optimization Opportunities, M9 — Post-M8 Latency Floor & Decomposition Report

### Community 467 - "miinfer_cli.cpp"
Cohesion: 0.15
Nodes (29): cmd_chat(), cmd_config(), cmd_inspect(), cmd_models(), cmd_run(), cmd_serve(), optional, string (+21 more)

### Community 468 - "Milestone M9 — Context Scaling Qualification (TG64 → TG1024)"
Cohesion: 0.33
Nodes (5): 1. Executive Summary, 2. Context Scaling Results Table, 3. Hardware Telemetry & Thermal Behavior, 4. Architectural Analysis, Milestone M9 — Context Scaling Qualification (TG64 → TG1024)

### Community 469 - "EXP-0227 — M11-B Q4_K token-parallel row tile"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Correctness, Decision, Environment, EXP-0227 — M11-B Q4_K token-parallel row tile, Follow-up, Hypothesis (+3 more)

### Community 470 - "EXP-0241 — M11-B Q4_K row-major 16-token tile rejection"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Correctness, Decision, Environment, EXP-0241 — M11-B Q4_K row-major 16-token tile rejection, Follow-up, Hypothesis (+1 more)

### Community 471 - "size_t"
Cohesion: 0.09
Nodes (7): DeviceBytes, bytes_, data_, GpuLayerRef, attention, recurrent, size_t

### Community 472 - "EXP-0214 — M11-B Native Q4_K Split-K Batched GEMM"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment, EXP-0214 — M11-B Native Q4_K Split-K Batched GEMM, Follow-up, Hypothesis, Re-evaluation — release build (+1 more)

### Community 479 - "bench_m10_latency_curve.py"
Cohesion: 0.42
Nodes (8): get_vram_mib(), main(), make_prompt(), parse_cli_stderr(), run_cold_request(), run_http_second_turn(), run_http_warm_request(), wait_for_server()

### Community 480 - "m6a12_qwen35_attention_projections.cpp"
Cohesion: 0.33
Nodes (7): path, size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 481 - "EXP-0212 — M11-B B=8 Accumulator GEMV Rejection"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0212 — M11-B B=8 Accumulator GEMV Rejection, Follow-up, Hypothesis, Results

### Community 482 - "EXP-0219 — M11-B Q4_K LDS Tile-Reuse Rejection"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0219 — M11-B Q4_K LDS Tile-Reuse Rejection, Follow-up, Hypothesis, Results

### Community 483 - "EXP-0244 — M11-B final qualification and measured ceiling"
Cohesion: 0.22
Nodes (8): Ceiling and blocker, Current production evidence, Decision, EXP-0244 — M11-B final qualification and measured ceiling, P512 Amdahl profile, Required next experiment, Result, Subsequent re-evaluation

### Community 485 - "Options"
Cohesion: 0.14
Nodes (14): uint32_t, Options, cache_regime, custom_k, custom_label, custom_m, device, experiment (+6 more)

### Community 486 - "m12_gdn_oracle.cpp"
Cohesion: 0.57
Nodes (7): at(), chunkwise(), size_t, main(), normalize_rows(), recurrent(), Matrix

### Community 487 - "recurrent_block"
Cohesion: 0.16
Nodes (22): tensor, rms_rows(), array, kChannels, path, size_t, span, vector (+14 more)

### Community 488 - "Qwen3ForwardTrace"
Cohesion: 0.17
Nodes (17): reset, reset, Qwen3ForwardTrace, embedding, final_norm, layer_outputs, logits, execute_qwen3_forward_gpu() (+9 more)

### Community 489 - "EXP-0256 — M12 chunkwise Gated DeltaNet oracle"
Cohesion: 0.20
Nodes (9): Baseline / oracle, Candidate, Correctness, Decision, Environment, EXP-0256 — M12 chunkwise Gated DeltaNet oracle, Follow-up, Hypothesis (+1 more)

### Community 490 - "Qwen3Layer0KvCache"
Cohesion: 0.24
Nodes (7): size_t, vector, Qwen3DecodeCache, caches_, length, Qwen3Layer0KvCache, append

### Community 491 - "EXP-0221 — M11-B Direct Consumption of Batched Prefill Workspace"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0221 — M11-B Direct Consumption of Batched Prefill Workspace, Follow-up, Hypothesis, Results

### Community 492 - "EXP-0208 — Full-Attention Projection Batching"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Correctness, Decision, Environment, EXP-0208 — Full-Attention Projection Batching, Follow-up, Hypothesis (+1 more)

### Community 493 - "qwen3_layer0_gpu_test.cpp"
Cohesion: 0.25
Nodes (17): abs_tolerance(), checkpoints(), compare(), path, size_t, string, vector, main() (+9 more)

### Community 494 - "EXP-0207 — B=8 LDS Weight-Reuse Prefill"
Cohesion: 0.22
Nodes (8): Baseline, Candidate, Decision, Environment, EXP-0207 — B=8 LDS Weight-Reuse Prefill, Follow-up, Hypothesis, Results

### Community 495 - "EXP-0223 — M11-B Q4_K B=4 Two-Wave Row Rejection"
Cohesion: 0.29
Nodes (6): Candidate, Correctness, Decision, EXP-0223 — M11-B Q4_K B=4 Two-Wave Row Rejection, Hypothesis, Results

### Community 496 - "EXP-0220 — M11-B Q4_K B=4 Compact Arithmetic Rejection"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0220 — M11-B Q4_K B=4 Compact Arithmetic Rejection, Follow-up, Hypothesis, Results

### Community 497 - "EXP-0215 — M11-B Larger Logical Chunk with B=4 Inner Microtiles"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0215 — M11-B Larger Logical Chunk with B=4 Inner Microtiles, Follow-up, Hypothesis, Results

### Community 498 - "EXP-0249 — M11-B native Q4_K MMQ64 rows64 rejection"
Cohesion: 0.20
Nodes (9): Baseline and candidate, Correctness, Decision, Environment, EXP-0249 — M11-B native Q4_K MMQ64 rows64 rejection, Follow-up, Hypothesis, Interpretation (+1 more)

### Community 499 - "EXP-0240 — M11-B Q4_K wave-major 16-token tile rejection"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Correctness, Decision, Environment, EXP-0240 — M11-B Q4_K wave-major 16-token tile rejection, Follow-up, Hypothesis (+1 more)

### Community 500 - "EXP-0250 — M11-B decoded Q4_K MMQ64 rejection"
Cohesion: 0.20
Nodes (9): Baseline and candidate, Correctness, Decision, Environment, EXP-0250 — M11-B decoded Q4_K MMQ64 rejection, Follow-up, Hypothesis, Interpretation (+1 more)

### Community 501 - "EXP-0255 — M12 dense staging feasibility"
Cohesion: 0.15
Nodes (12): Baseline, Candidate, Correctness, Decision, Environment, EXP-0255 — M12 dense staging feasibility, Follow-up, Hypothesis (+4 more)

### Community 502 - "EXP-0239 — M11-B native Q4_K 16-token tile rejection"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Correctness, Decision, Environment, EXP-0239 — M11-B native Q4_K 16-token tile rejection, Follow-up, Hypothesis (+1 more)

### Community 503 - "EXP-0234 — M11-B causal recurrent-core B=4 batch"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment, EXP-0234 — M11-B causal recurrent-core B=4 batch, Follow-up, Hypothesis, Interpretation (+1 more)

### Community 504 - "EXP-0224 — M11-B Q4_K B=4 LDS Token Distribution Rejection"
Cohesion: 0.29
Nodes (6): Candidate, Correctness, Decision, EXP-0224 — M11-B Q4_K B=4 LDS Token Distribution Rejection, Hypothesis, Results

### Community 505 - "EXP-0233 — M11-B quantized projection Amdahl ceiling"
Cohesion: 0.22
Nodes (8): Amdahl calculation, Baseline, Decision, Evidence, EXP-0233 — M11-B quantized projection Amdahl ceiling, Follow-up, Interpretation, Question

### Community 506 - "EXP-0229 — M11-B Q4_K token-microtile wave decomposition"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment, EXP-0229 — M11-B Q4_K token-microtile wave decomposition, Follow-up, Hypothesis, Interpretation (+1 more)

### Community 507 - "EXP-0225 — M11-B Q4_K B=4 Full Row-Tile LDS Rejection"
Cohesion: 0.29
Nodes (6): Candidate, Correctness, Decision, EXP-0225 — M11-B Q4_K B=4 Full Row-Tile LDS Rejection, Hypothesis, Results

### Community 508 - "EXP-0238 — M11-B causal 64-token prefill chunk"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness, Decision, Environment, EXP-0238 — M11-B causal 64-token prefill chunk, Follow-up, Hypothesis (+2 more)

### Community 509 - "ProfileScope"
Cohesion: 0.11
Nodes (21): Function, hipEvent_t, Qwen3ProfileCategory, profile_copy_call(), profile_gpu_call(), ProfileScope, boundary_stage_, bytes_ (+13 more)

### Community 510 - "EXP-0248 — M11-B operator-major recurrent tail rejection"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0248 — M11-B operator-major recurrent tail rejection, Follow-up, Hypothesis, Results

### Community 511 - "model_loader_test.cpp"
Cohesion: 0.32
Nodes (12): append_string(), append_u32(), append_u64(), path, string, uint32_t, uint64_t, uint8_t (+4 more)

### Community 512 - "Qwen3GpuProfileEvent"
Cohesion: 0.14
Nodes (13): hipEvent_t, Qwen3BoundaryProfileStage, Qwen3FfnProfileStage, Qwen3ProfileCategory, Qwen3GpuProfileEvent, boundary_stage, bytes, category (+5 more)

### Community 513 - "Group"
Cohesion: 0.20
Nodes (13): set, size_t, string, uint64_t, vector, Group, bytes, count (+5 more)

### Community 514 - "Qwen3Config"
Cohesion: 0.17
Nodes (12): uint32_t, Qwen3Config, attention_heads, context_length, head_dim, hidden_size, intermediate_size, kv_heads (+4 more)

### Community 515 - "EXP-0235 — M11-B fused recurrent Q8_1 output"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Correctness, Decision, Environment, EXP-0235 — M11-B fused recurrent Q8_1 output, Follow-up, Hypothesis (+1 more)

### Community 516 - "EXP-0257 — M12 gfx906 Gated DeltaNet chunk prototype"
Cohesion: 0.15
Nodes (12): Baseline, Candidate, Corrected isolated result — 2026-09-08, Correctness, Decision, Environment, EXP-0257 — M12 gfx906 Gated DeltaNet chunk prototype, Follow-up (+4 more)

### Community 517 - "EXP-0259 — M12 dense FFN-down prefill backend"
Cohesion: 0.18
Nodes (10): Candidate, Correctness, Decision, Environment, EXP-0259 — M12 dense FFN-down prefill backend, Follow-up, Hypothesis, Memory (+2 more)

### Community 518 - "EXP-0247 — M11-B Q4_K MMQ64 split-2 rejection"
Cohesion: 0.20
Nodes (9): Baseline and candidate, Correctness, Decision, Environment, EXP-0247 — M11-B Q4_K MMQ64 split-2 rejection, Follow-up, Hypothesis, Interpretation (+1 more)

### Community 519 - "FfnTailReplay"
Cohesion: 0.33
Nodes (6): FfnTailReplay, down, gate, layer_output, swiglu, up

### Community 520 - "Qwen3LayerWeights"
Cohesion: 0.17
Nodes (12): Qwen3LayerWeights, attention_norm, down, ffn_norm, gate, k, k_norm, output (+4 more)

### Community 521 - "EXP-0251 — M11-B native Q4_K token-reuse rejections"
Cohesion: 0.20
Nodes (9): Baseline, Candidates, Correctness, Decision, Environment, EXP-0251 — M11-B native Q4_K token-reuse rejections, Hypothesis, Interpretation (+1 more)

### Community 522 - "half"
Cohesion: 0.29
Nodes (10): uint32_t, launch_qwen35_fused_k_norm_rope_kv_store(), launch_qwen35_tiled_online_attention(), launch_qwen3_kv_cache_store(), uint32_t, vector, evaluate_fp16_gemv(), fp16_gemv_cpu_reference() (+2 more)

### Community 523 - "M3 — Minimal Runtime"
Cohesion: 0.29
Nodes (7): Architectural rule, Core runtime responsibilities, Exit criteria, Goal, M3 — Minimal Runtime, Model scope, Quantization scope

### Community 524 - "m6a9_qwen35_lm_head.cpp"
Cohesion: 0.35
Nodes (8): argmax(), path, size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 525 - "EXP-0260 — M12 combined dense and chunkwise prefill"
Cohesion: 0.29
Nodes (6): Candidate, Correctness, Decision, EXP-0260 — M12 combined dense and chunkwise prefill, Hypothesis, Results

### Community 526 - "m6a11_qwen35_attention_prefix.cpp"
Cohesion: 0.33
Nodes (7): path, size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 527 - "run_ladder"
Cohesion: 0.35
Nodes (10): argmax(), path, size_t, span, uint32_t, is_boundary(), is_full_attention_layer(), main() (+2 more)

### Community 528 - "qwen3_generate.cpp"
Cohesion: 0.40
Nodes (10): argmax(), size_t, string, uint32_t, vector, main(), parse_count(), parse_id() (+2 more)

### Community 529 - "vector"
Cohesion: 0.07
Nodes (21): as(), T, as(), T, string, vector, hip_check(), hip_check_failed() (+13 more)

### Community 530 - "EXP-0242 — M11-B recurrent fused normalization/Q8 rejection"
Cohesion: 0.22
Nodes (8): Baseline and candidate, Correctness, Decision, Environment, EXP-0242 — M11-B recurrent fused normalization/Q8 rejection, Follow-up, Hypothesis, Results

### Community 531 - "1. ggml-org/llama.cpp"
Cohesion: 0.33
Nodes (6): 1. ggml-org/llama.cpp, Do not inherit automatically, Important lessons, MIInfer question, Potentially reusable knowledge, Role

### Community 532 - "Roadmap Principles"
Cohesion: 0.40
Nodes (5): Benchmark before claim, Correctness before speed, Evidence before architecture, Narrow before broad, Roadmap Principles

### Community 533 - "Fp16GemvMetrics"
Cohesion: 0.25
Nodes (8): Fp16GemvMetrics, cosine_similarity, inf_detected, max_abs_error, max_relative_error, mean_abs_error, nan_detected, pass

### Community 534 - "LayerPathCapture"
Cohesion: 0.20
Nodes (10): LayerPathCapture, attention_residual, ffn_output, gated, input, layer_output, normalized, post_normalized (+2 more)

### Community 535 - "EXP-0245 — M11-B Q4_K MMQ16 split-4 rejection"
Cohesion: 0.22
Nodes (8): Baseline and candidate, Decision, Environment, EXP-0245 — M11-B Q4_K MMQ16 split-4 rejection, Follow-up, Hypothesis, Interpretation, Results

### Community 536 - "Buffer"
Cohesion: 0.50
Nodes (3): Buffer, p, size_t

### Community 537 - "DeviceBuffer"
Cohesion: 0.33
Nodes (5): size_t, T, DeviceBuffer, data_, upload()

### Community 538 - "Q8ExactBlock"
Cohesion: 0.20
Nodes (10): int16_t, int8_t, Q8_1Block, d, qs, s, Q8ExactBlock, d (+2 more)

### Community 539 - "RmsVariant"
Cohesion: 0.29
Nodes (7): RmsVariant, inverse, mean, norm, rms, root, sum

### Community 540 - "M12GdnChunkWorkspace"
Cohesion: 0.25
Nodes (6): M12GdnChunkWorkspace, corrected_values, decayed_keys, new_values, solved_keys, solved_values

### Community 541 - "D031 — Freeze M11-B and Isolate Matrix Prefill from Decode"
Cohesion: 0.50
Nodes (4): Consequences, D031 — Freeze M11-B and Isolate Matrix Prefill from Decode, Decision, Reason

### Community 542 - "D024 — Benchmarkability Is an Architectural Requirement"
Cohesion: 0.67
Nodes (3): D024 — Benchmarkability Is an Architectural Requirement, Decision, Reason

### Community 543 - "EXP-0263 — M12 GDN contract fix and dense-path qualification"
Cohesion: 0.25
Nodes (7): Candidate, Correctness, Decision, Environment, EXP-0263 — M12 GDN contract fix and dense-path qualification, Follow-up, Hypothesis

### Community 544 - "11. joe2gaan/localaiservers"
Cohesion: 0.50
Nodes (4): 11. joe2gaan/localaiservers, Important dot-product lesson, MIInfer implication, Role

### Community 545 - "qwen3_swiglu_q8_bench.cpp"
Cohesion: 0.39
Nodes (7): vector, main(), measure(), Stats, mean_us, median_us, summarize()

### Community 546 - "EXP-0258 — M12 runtime composition and 128-token schedule"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0258 — M12 runtime composition and 128-token schedule, Hypothesis, Interpretation, Results

### Community 547 - "M4 — First Correct End-to-End Generation"
Cohesion: 0.33
Nodes (6): Correctness validation, Exit criteria, Goal, Initial inference mode, M4 — First Correct End-to-End Generation, Required execution pieces

### Community 548 - "13. Prefill vs Decode"
Cohesion: 0.67
Nodes (3): 13. Prefill vs Decode, Decode, Prefill

### Community 549 - "Metrics"
Cohesion: 0.25
Nodes (8): Metrics, actual_at_max, expected_at_max, max_abs, max_index, max_rel, mean_abs, rmse

### Community 550 - "qwen3_primitives_test.cpp"
Cohesion: 0.43
Nodes (7): close_enough(), T, vector, device_copy(), gpu_tests(), host_tests(), main()

### Community 551 - "run_combined"
Cohesion: 0.33
Nodes (6): RuntimeState, path, uint32_t, vector, main(), run_combined()

### Community 552 - "GemvKernelResources"
Cohesion: 0.33
Nodes (6): GemvKernelResources, local_bytes, max_threads_per_block, registers, shared_bytes, size_t

### Community 553 - "Current Project Status"
Cohesion: 0.50
Nodes (4): Completed, Completed in Task 3, Current Project Status, Not implemented

### Community 554 - "HostQ8Block"
Cohesion: 0.33
Nodes (6): int8_t, uint16_t, HostQ8Block, d_bits, qs, s_bits

### Community 555 - "Current Scope"
Cohesion: 0.67
Nodes (3): Current Scope, In scope now, Not in scope now

### Community 556 - "15. Research Classification"
Cohesion: 0.40
Nodes (5): 15. Research Classification, Category A — Architecture lessons, Category B — Kernel hypotheses, Category C — Failure lessons, Category D — Runtime features

### Community 558 - "17. Highest-Priority Research Ideas"
Cohesion: 0.40
Nodes (5): 17. Highest-Priority Research Ideas, Later, P0, P1, P2

### Community 559 - "Buffer"
Cohesion: 0.50
Nodes (3): Buffer, pointer, size_t

### Community 560 - "Event"
Cohesion: 0.50
Nodes (3): hipEvent_t, Event, value

### Community 561 - "Event"
Cohesion: 0.50
Nodes (3): hipEvent_t, Event, p

## Knowledge Gaps
- **4044 isolated node(s):** `experiment`, `shape`, `implementation`, `cache_regime`, `custom_label` (+4039 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **56 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Qwen35RuntimeEngine` connect `Qwen35RuntimeEngine` to `.generate`, `qwen3_tokenizer.cpp`, `RuntimeGenerateStats`, `RecurrentLayer`, `FullAttentionLayer`, `miinfer_cli.cpp`, `PrefillProfile`, `size_t`, `Qwen35Config`, `qwen35_gpu_pipeline.hpp`, `M12GdnChunkWorkspace`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Why does `RecurrentLayer` connect `RecurrentLayer` to `.run`, `main`, `half`, `RecurrentOperands`, `Qwen35RuntimeEngine`, `LayerPathCapture`, `size_t`, `Qwen35Config`, `qwen35_gpu_pipeline.hpp`, `UpdateProvenance`, `M12GdnChunkWorkspace`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Why does `GgufFile` connect `GgufFile` to `Qwen35Config`, `m6a15_qwen35_hybrid_block_audit.cpp`, `validate_position`, `m6a3_qwen35_layer.cpp`, `run_combined`, `recurrent_block`, `qwen3_tokenizer.cpp`, `run_ladder`, `vector`, `m6a14_qwen35_state_audit.cpp`, `m6a18_qwen35_deltanet_state_gpu.cpp`, `gguf.cpp`, `qwen35_gpu_pipeline.hpp`, `Qwen3Model`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **What connects `experiment`, `shape`, `implementation` to the rest of the system?**
  _4044 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `hardware.md` be split into smaller, more focused modules?**
  _Cohesion score 0.0425531914893617 - nodes in this community are weakly interconnected._
- **Should `architecture.md` be split into smaller, more focused modules?**
  _Cohesion score 0.058823529411764705 - nodes in this community are weakly interconnected._
- **Should `benchmarking.md` be split into smaller, more focused modules?**
  _Cohesion score 0.045454545454545456 - nodes in this community are weakly interconnected._