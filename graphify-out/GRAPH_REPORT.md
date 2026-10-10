# Graph Report - e2-0003q-final-clean-ed21f41  (2026-10-10)

## Corpus Check
- 1044 files · ~1,294,920 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 11348 nodes · 15800 edges · 908 communities (832 shown, 76 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 879 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `cb99829e`
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
- Fp16GemvMetrics
- M0 — Baseline and Project Bootstrap
- 29. Development sequence
- fp16_gemv_bench.cpp
- graphify reference: extra exports and benchmark
- MI50 Platform Notes
- 8. anikifoss/llama.cpp-gfx906
- 2. Benchmark Targets
- Decision / next work
- WidePrefillWorkspace
- 4. Candidate Optimization Roadmap for M8
- 32. Codex task behavior
- PrefillV2Workspace
- graphify reference: query, path, explain
- 43. Initial MIInfer Benchmark Matrix
- Qwen3-8B Dense Control Model
- EXP-0146 — M6-B54 post-B53 production profile
- DeviceKvShard
- 15. Benchmark Protocol
- 10. Kernel experiments
- 40. Acceptance Categories
- 5. Required Environment Metadata
- D008 — Do Not Build a Generic Graph Runtime Initially
- RecurrentLayer
- FullAttentionLayer
- LayerEvaluationResult
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
- v2_0020_pipeline_bench.cpp
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
- DeviceKvPool
- m31_0006_graph_attention_microbench.cpp
- ComponentTimingBreakdown
- EXP-0003 — FP16 GEMV Bottleneck Characterization
- 10. Software Environment
- 13. Correctness Method
- 17. Raw Results
- 8. Hardware
- sample-gpu.sh
- build_info.cpp
- EXP-0364 — Full-model Qwen3.8-27B prefill bottleneck attribution
- MIInfer Performance Research Protocol
- Benchmarks
- EXP-0007 — gfx906 Zero-Point-Corrected Q4_0 × Q8_1 Dot4
- qwen3_tokenizer.cpp
- EXP-0009-kv-geometry.md
- extraction-spec.md
- diagnose-gfx802-isolation.sh
- README.md
- 6. Baseline
- 35. Historical failed execution gate — superseded by Section 37
- fp16_gemv_k_split_bench.cpp
- m6a1_reference_fixture.cpp
- EXP-0008 — Direct MIInfer vs gfx906 llama.cpp MMVQ
- qwen3_inference_bench.cpp
- EXP-0382 — Revalidate the Actual Composed-B64 Runtime
- run_sequence
- External gfx906 Reference Baseline
- q4_q8_gemv_bench.cpp
- Qwen3GpuDecodeWorkspace
- EXP-0086 — M6-A27.6 Qwen3.8 P2 L0–L2 operation trace
- fp16_gemv_reduction_diag.cpp
- EXP-0300 — M25-A MIInfer versus mx-llama differential
- m6a15_qwen35_hybrid_block_audit.cpp
- EXP-0184 — Fast 2-Stage Parallel Argmax Reduction
- qwen3_fast_decode_bench.cpp
- EXP-0021 — M5-C6c coalesced KV-cache writes
- EXP-0331 — Pinned Mx MMQ register weight prefetch
- qwen3_primitives.cpp
- Qwen3GpuProfile
- README.md
- M3 Minimal Qwen3-8B Runtime Scaffold
- ContextSpace
- Qwen3LayerTrace
- EXP-0074 — M6-A26.4 L30 production operand attribution
- EXP-0041 — M5-C15 optimization closure and parity decision gate
- run_bakeoff_main
- EXP-0100 — M6-B7 Q5_K paired-nibble decoding
- qwen3_layer_host_impl
- M4-C1 — Deterministic first generated token
- Metrics
- README.md
- EXP-0110 — M6-B17 Q5_K×Q8_1 MMVQ recurrent projection
- M4-B — Full Qwen3 single-token forward
- README.md
- EXP-0379 — B64 whole-model composition
- README.md
- EXP-0026 — M5-C8c Down long-K bottleneck attribution
- run-m4c1-acceptance.sh
- run-m4b-acceptance.sh
- run-m4c2-acceptance.sh
- Q8BoundaryDiff
- README.md
- EXP-0046 — M6-A4 Qwen3.8-27B full-attention layer
- EXP-0305 — M25-D mx GDN register-resident scan
- EXP-0042 — M6-A0 Qwen3.8-27B GGUF and architecture audit
- m6a263_qwen35_recurrent_contract.cpp
- qwen3_position_audit.cpp
- EXP-0209 — B=8 LDS Shared-Weight Shape Check
- EXP-0153 — M6-B61 fused beta/alpha preparation
- qwen3_forward_test.cpp
- EXP-0178 — Native Q5_K Wave64 Rollout (SSM Out Projections)
- M5-A — Reproducible MI50 inference baseline
- M4-C3 — Text-facing greedy generation
- run-m4c3-acceptance.sh
- Qwen35Config
- hip_smoke_bench.cpp
- qwen3_attention_ab_bench.cpp
- EXP-0010 — Qwen3-8B steady-state decode profile
- EXP-0017 — M5-C5a persistent Qwen3 decode workspace
- README.md
- EXP-0016 — M5-C4 post-attention MI50 baseline
- run-m5a-baseline.sh
- EXP-0012 — Qwen3-8B Q4_0 MI50 comparison
- EXP-0018 — M5-C5b resident normalization weights
- EXP-0014 — Cooperative cached-attention execution
- EXP-0013 — Qwen3 position-scaled execution audit
- run-m5b-profile.sh
- EXP-0183 — Fused DeltaNet Recurrent Core in LDS
- Sha256
- EXP-0011 — Trace-free Qwen3-8B decode control
- EXP-0099 — M6-B6 Q5_K subgroup-structured dot loop
- EXP-0022 — M5-C6d GPU-side greedy argmax
- Buffer
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
- PersistentSession::load_from_file
- run-m5c6b-layer-output-ab.sh
- PrefillV2Model
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
- m12_gdn_chunk_bench.cpp
- EXP-0039 — M5-C13c fixed-floor contract map
- EXP-0053 — M6-A9 Qwen3.8-27B LM-head GPU projection
- EXP-0036 — M5-C12b cooperative attention scaling
- EXP-0084 — M6-A27.4 Qwen3.8 full-model observable contract adjudication
- gguf.cpp
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
- GgufTensor
- EXP-0050 — M6-B0 Qwen3.8-27B llama.cpp MI50 baseline
- EXP-0188 — Inter-Layer Norm Fusion, Native Q6_K LM Head & Wave64 Single-Wave GEMV (Stretch Gate Closure)
- EXP-0056 — M6-A12 Qwen3.8-27B attention projections
- EXP-0049 — M6-A7 Qwen3.8-27B stateful generation
- m6a14_qwen35_state_audit.cpp
- EXP-0252 — Repacked MMQ64 retest on the production Q4_K down shape
- EXP-0254 — Repacked MMQ64 exact-sum activation layout
- run-m6b0-llama-baseline.sh
- DeviceKvBlock
- EXP-0058 — M6-A14 Qwen3.8-27B state fingerprints and reset audit
- EXP-0125 — M6-B33 post-B32 production profile
- EXP-0059 — M6-A15 Qwen3.8-27B layers 0–3 hybrid-block audit
- EXP-0093 — M6-B1 Qwen3.8-27B native GPU generation baseline
- EXP-0055 — M6-A11 Qwen3.8-27B composed attention prefix
- EXP-0377 — Qualify Existing B64 Residual Contract
- m6a12_qwen35_attention_projections.cpp
- EXP-0079 — M6-A26.9 Qwen3.8 external recurrent-state contract
- EXP-0052 — M6-A8 Qwen3.8-27B GPU foundation
- EXP-0089 — M6-A27.9 Qwen3.8 L0 Q5_K block contract
- EXP-0073 — M6-A26.3 Qwen3.8 recurrent-state contract adjudication
- EXP-0060 — M6-A16 Qwen3.8-27B layers 4–7 hybrid-block audit
- EXP-0090 — M6-A27.9 full observable-contract retest
- EXP-0061 — M6-A17 Qwen3.8-27B composition ladder
- Qwen3FfnProbeTrace
- EXP-0380 — Localize the first B64 attention-composition hang
- EXP-0096 — M6-B3 Q4_K metadata staging
- EXP-0182 — Fused Gate+Up SwiGLU Wave64 GEMV
- Qwen35RuntimeEngine
- EXP-0066 — M6-A22 Qwen3.8-27B GPU hybrid position audit
- m6a18_qwen35_deltanet_state_gpu.cpp
- EXP-0141 — M6-B49 recurrent state-update/head-RMS fusion
- PrefillV2AttentionLayer
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
- Qwen3GpuDecodeCache
- EXP-0124 — M6-B32 transposed recurrent no-decay store
- EXP-0156 — M6-B64 post-B62 stage profile
- EXP-0137 — M6-B45 Q4_K×Q8_1 inner-loop differential
- EXP-0075 — M6-A26.5 L30 K-path provenance
- m24_projection_bakeoff.cpp
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
- qwen3_trace_compare.cpp
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
- EXP-0274 — M18–M20 qualification closure
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
- AggregatedStats
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
- RunResult
- EXP-0237 — M11-B production layer-major prefill profile
- EXP-0165 — M6-B69 dual query/key head normalization
- EXP-0160 — M6-B68 DeltaNet ordered row-wave reduction
- qwen3_decode_sequence_gpu_test.cpp
- EXP-0166 — M6-B70 column-tiled DeltaNet state update
- EXP-0192: Wave64 Fused Gate+Up SwiGLU Intra-Wave Shuffle Reduction
- EXP-0176 — MI50 manual-DPM qualification and EXP-0174 re-adjudication
- 2. Command Architecture
- EXP-0158 — M6-B66 expanded Q4_K FFN Gate/Up
- qwen3_layer0_gpu_test.cpp
- EXP-0155 — M6-B63 post-B62 production baseline
- EXP-0190 — Fused Recurrent & Attention Epilogue Q8_1 Activation Quantization (Lane 2)
- EXP-0231 — M11-B Q4_K repacked MMQ tile
- EXP-0173 — Q4K native-layout laboratory
- EXP-0196 — Combined Projections for Recurrent (QKV+Gate) and Attention (Q+K) Layers
- EXP-0181 — Static HIP Graph Capture for Autoregressive Decode
- Candidate 1: FFN Gate & Up Rollout
- EXP-0218 — M11-B Q4_K Word-Reuse B=8 Rejection
- EXP-0174 — Wave-plane Q4K Down integration
- Post-M26 Performance Roadmap
- EXP-0169 — M6-B74 pinned llama.cpp architecture differential
- EXP-0170 — M6-B75 fused Gate/Up/SwiGLU prototype
- Candidate 2: Full-Attention Q Projection Rollout
- EXP-0193: Vectorized SIMD Q6_K Decoding for LM Head and Down Projections
- EXP-0393 — Requalify the aligned P512 baseline
- EXP-0228 — M11-B Q4_K four-token subwave mapping
- EXP-0297 — M24 exact resident-MMQ projection bake-off
- EXP-0197 — Paired Fused Gate+Up SwiGLU with 64-Bit dwordx2 Coalesced Memory Access
- Candidate 3: Recurrent Attention Gate Rollout
- EXP-0304 — M25-C mx register-pipeline transplant
- Candidate 4: Full-Attention Output Projection Rollout
- Milestone M9 Primary Gate Qualification Report
- EXP-0194 — Vectorized Q4_K and Q5_K Fast Tile Arithmetic Optimization
- EXP-0185 — Tiled Online-Softmax Attention with Gate Sigmoid Fusion
- EXP-0367 — Exact partial-tail stage attribution and state oracle
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
- V2-0002 — Prefill V2: Mx Compact Projection Backend & FAST_V1 Confrontation
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
- Qwen3ForwardTrace
- Milestone M8 Primary Gate Qualification Report
- EXP-0205 — Batched Q4_K Wave GEMV for Prefill
- Milestone M10 — Real-World Inference Performance & Hybrid Context Scaling
- run-exp0186-fused-rope-norm-ab.sh
- run-exp0187-fused-add-rms-norm-ab.sh
- M31-0006 — Implementation and promotion decision
- EXP-0253 — Repacked MMQ64 with exact Q8 side sums
- EXP-0210 — M11-B Amdahl Profile and Sequential-Floor Check
- EXP-0303 — M25-B mx compact repacked MMQ
- EXP-0213 — M11-B FP16 GEMM with On-Device K-Quant Dequantization
- EXP-0308 — M25-G mx weight-row prefetch
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
- EXP-0338 — Mx attention decode weight reuse
- EXP-0214 — M11-B Native Q4_K Split-K Batched GEMM
- run-exp0194-kquant-fast-arith-ab.sh
- run-exp0196-combined-projections-ab.sh
- run-exp0197-swiglu-paired-ab.sh
- run-exp0198-cached-metadata-ab.sh
- run-m9-gate-telemetry.sh
- run-m9-qualification.sh
- PrefillV2RecurrentLayer
- e2_0003_checkpointed_call.cpp
- EXP-0212 — M11-B B=8 Accumulator GEMV Rejection
- EXP-0219 — M11-B Q4_K LDS Tile-Reuse Rejection
- EXP-0244 — M11-B final qualification and measured ceiling
- bench-streaming-modes.sh
- EXP-0284 — M22 recurrent B4 tail-family profile
- EXP-0282 — M22 SSM-output dense batch candidate
- EXP-0306 — M25-E sliced activation staging for mx MMQ
- qwen3_gpu_primitives.hpp
- EXP-0256 — M12 chunkwise Gated DeltaNet oracle
- MatrixScenarioResult
- EXP-0221 — M11-B Direct Consumption of Batched Prefill Workspace
- EXP-0208 — Full-Attention Projection Batching
- size_t
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
- EXP-0365 — Context scaling reconciliation
- EXP-0248 — M11-B operator-major recurrent tail rejection
- EXP-0313 — M25-H/I review hardening and P512 stall triage
- EXP-0275 — M21 prefill gap and M12 continuation correctness
- EXP-0344 — M25 oracle GDN launch-bounds contract
- EXP-0294 — M23 P512 projection / B64 causal-width profile
- EXP-0235 — M11-B fused recurrent Q8_1 output
- EXP-0257 — M12 gfx906 Gated DeltaNet chunk prototype
- EXP-0259 — M12 dense FFN-down prefill backend
- EXP-0247 — M11-B Q4_K MMQ64 split-2 rejection
- m6a9_qwen35_lm_head.cpp
- suffix_ttft_attribution_bench.cpp
- EXP-0251 — M11-B native Q4_K token-reuse rejections
- EXP-0264 — M16-B bounded serving queue qualification
- EXP-0289 — Resident recurrent FFN MMQ weights
- observe
- EXP-0260 — M12 combined dense and chunkwise prefill
- EXP-0287 — M23 shared dense projection oracle
- EXP-0328 — M25-E split M23 attention Q/K geometry
- EXP-0352 — M26-B historical decode-floor differential
- EXP-0293 — Async next-layer repacked-weight prefetch
- EXP-0242 — M11-B recurrent fused normalization/Q8 rejection
- EXP-0371 — Isolate L7 B128 projection contract
- EXP-0276 — M22 P512 operator-family attribution
- EXP-0277 — M22 mx-llama native repack A/B
- EXP-0311 — M25-J Mx batched Q8 quantizer
- EXP-0245 — M11-B Q4_K MMQ16 split-4 rejection
- EXP-0283 — M22 shared dense FFN-down source
- EXP-0350 — Interactive serving and live continuation
- v2_0017_hierarchical_gqa_bakeoff.cpp
- EXP-0319 — M25-H/I configuration matrix
- EXP-0333 — Q4/Q5-only Mx MMQ occupancy annotation
- D031 — Freeze M11-B and Isolate Matrix Prefill from Decode
- D024 — Benchmarkability Is an Architectural Requirement
- EXP-0263 — M12 GDN contract fix and dense-path qualification
- EXP-0268 — M18-A lifecycle, cancellation, and request telemetry
- EXP-0278 — M22 quantized batch-width study
- EXP-0258 — M12 runtime composition and 128-token schedule
- EXP-0296 — M24 resident MMQ versus GPU dequantized FP16 GEMM
- EXP-0290 — Persistent-token affine MMQ
- EXP-0265 — M16-D OpenAI request contract qualification
- EXP-0279 — M22 dense FFN-down candidate
- test-serve.sh
- EXP-0323 — M25 P512 exact pre-J/source-delta A/B
- EXP-0288 — M23 direct P512 GDN scan
- EXP-0285 — M22 prefill parity closure
- EXP-0355 — M27 attention decode-reuse P512 A/B and context smoke
- test-package.sh
- EXP-0269 — M18-B pinned llama.cpp-gfx906 qualification
- m13_quant_mm_bench.cpp
- EXP-0356 — M27 P512 semantic contract attribution and stop
- EXP-0374 — Qualify repeated-B128 remainder scheduler
- EXP-0320 — M25 P512 continuation crash and resident FFN repair
- EXP-0267 — M17 Local Appliance Qualification
- EXP-0270 — M19 dynamic context capacity qualification
- EXP-0266 — M16-C serving concurrency qualification
- EXP-0280 — M22 M13 B8 eight-wave mapping
- EXP-0271 — MIInfer runtime-only prefill/decode qualification
- EXP-0309 — M25-H Mx attention FFN projections
- EXP-0272 — M20-B API authentication and serving hardening
- EXP-0370 — Qualify L7 partial-tail K/V and causal contract
- install.sh
- EXP-0273 — M20-A constrained Hermes qualification
- EXP-0301 — M25-D mx-style GDN state tiling
- EXP-0383 — Localize the First Composed-B64 Semantic Divergence
- EXP-0281 — M22 dense FFN prefill candidate
- EXP-0368 — L3 partial-attention K/V and causal contract
- UpdateProvenance
- run_layer
- Re-evaluation — full layer-major scheduling — 2026-09-10
- EXP-V2-0022-suffix-mmq-critical-path.md
- EXP-0324 — M25 Mx vectorized MMQ epilogue
- EXP-0298 — M24 direct resident-FP16 recurrent layer
- EXP-0310 — M25-I Mx attention O projection
- EXP-0325 — M25 beta/alpha batched FP32 hipBLAS port
- EXP-0295 — M23 resident-all projection path
- EXP-0291 — Fused wide FFN Gate/Up MMQ
- EXP-0340 — M25-L recurrent FFN contract differential
- EXP-0312 — M25-K Mx attention Q/K projection
- EXP-0299 — M24 full-attention attribution
- EXP-0318 — M25 attention bakeoff pipeline isolation
- EXP-0376 — Attribute the Repeated-B128 Runtime Collapse
- EXP-0292 — Resident attention FFN MMQ weights
- HttpRequest
- v2_0017_quad_head_subwave_bakeoff.cpp
- 1. ggml-org/llama.cpp
- EXP-0307 — M25-F mx affine DP4A ordering
- EXP-0354 — Current HEAD P512 control requalification
- EXP-0302 — M25-E wide SwiGLU to MMQ-Q8 producer
- EXP-0327 — M25-D GDN Tc4 interleaved qualification retest
- EXP-0329 — M25-E complete Mx attention Q/K contract screen
- EXP-0315 — M25 MMQ occupancy annotation
- EXP-0316 — M25-J quantizer under qualified MMQ
- Qwen3LayerWeights
- EXP-0317 — M25 staged MMQ loop lowering
- EXP-0330 — Mx recurrent FFN Gate/Up pair kernel
- EXP-0366 — Partial-tail prefill execution contract
- validate_position
- m6a13_qwen35_full_attention_layer.cpp
- EXP-0343 — M25 faithful GDN state input/output contract
- EXP-0314 — M25 P512 source-delta retest after stall recovery
- EXP-0322 — M25-H/I post-repair qualification
- 15. Research Classification
- EXP-0321 — M25 pinned mx-llama contract audit
- EXP-0332 — Mx MMQ full-tile specialization
- EXP-0349 — M25 pinned Q6 FFN-down probe
- Q5K
- EXP-0326 — M25-D GDN four-column wave shard
- compare-m26cq-logits.py
- EXP-0345 — M25 parallel recurrent input branches
- BenchmarkResult
- EXP-0334 — M25 complete pinned Mx MMQ contract rejection
- EXP-0347 — M25 persistent oracle GDN state layout
- EXP-0353 — M27 reusable device-state decode graph
- EXP-0363 — Spill-free attention state schedule
- EXP-0362 — Source-shaped query-tiled GQA2 prefill
- EXP-0348 — M25 current oracle requalification
- m6a10_qwen35_q4k_projection.cpp
- compare-m26cq-layer-path.py
- vector
- EXP-0335 — M25 pinned Mx single-token MMV
- EXP-0342 — M25 QKV contract retest
- EXP-0337 — M25 P512 prefill HIP graph screen
- EXP-0336 — M25 pinned Mx MMQ recurrent-FFN isolation rejection
- Candidate work
- EXP-0339 — Mx attention decode reuse P512 qualification screen
- run_ladder
- EXP-0341 — M25 attention fork/join concurrency
- EXP-0346 — M25 parallel recurrent QKV/Gate composition
- v2_0017_pipelined_stream_bakeoff.cpp
- EXP-0385 — Validate the Exact B64 L0 Recurrent-Wide Contract
- EXP-V2-0020 — FP16 Suffix Attention Load/Compute Pipeline Qualification
- 6. nlzy/vllm-gfx906
- BlockEvaluationResult
- ReusableContext
- compare-m26c-state.py
- memory_stream_bench.cpp
- BenchmarkResult
- hipStreamSynchronize
- EXP-0361 — Wave64 attention architecture analysis
- AttentionLayerKvCacheStorage
- prepare-m26cq-teacher-inputs.py
- EXP-V2-0008 — Dedicated Single-Token Decode Execution Path and Reusable HIP Graph Replay
- e2_0003_same_source_ab.py
- EXP-0357 — M27 sparse exact prefix cache
- half
- EXP-0372 — Isolate L6 → L7 inter-layer B128 tail contract
- compare-m26c-layer-path.py
- 17. Highest-Priority Research Ideas
- EXP-0381 — Isolate the Historical B64 Stop Above the Proven Prefill Boundary
- 11. joe2gaan/localaiservers
- compare-m26c-logits.py
- summarize-m26c-profile.py
- AccuracyMetrics
- EXP-0360 — GQA-shared tiled full-attention prefill candidate
- Performance Research Strategy
- EXP-0378 — Determine Whether a Non-Scalar B64 Core Contract Exists
- EXP-0391 — P512 recurrent/attention variance split
- EXP-0369 — Explain the P1664 position-768 divergence
- EXP-0386 — Determine Whether Normal B64 Drift Causes L3 Semantic Amplification
- EXP-0388 — B128 shape-collapse attribution
- EXP-V2-0023-cross-query-attention.md
- m6a8_qwen35_gpu_foundation.cpp
- EXP-0373 — Refresh full-model partial-tail attribution
- Decision
- m6a11_qwen35_attention_prefix.cpp
- EXP-0384 — Locate the First Material B64 Recurrent-Wide Layer Divergence
- EXP-V2-0024-quantized-prefix-kv.md
- qwen3_generate.cpp
- PhysicalPageView
- V2-0003 — Prefill V2 Slice 2: 4-Layer Repeating Topology Block (3 × GDN + 1 × GQA Attention)
- EXP-V2-0026-production-pipeline-hardening.md
- suffix_attention_halfwave_multitoken.cpp
- V2-0001 — Clean-Sheet Single-MI50 Prefill V2: Recurrent-Layer Vertical Slice
- RecurrentLayerStateStorage
- V2-0004 — Register-Resident GDN Integration and Full 64-Layer Prefill Pipeline
- Metrics
- EXP-0375 — Correct repeated-B128 scheduler dispatch
- EXP-0394 — Physical-B512 recurrent projections for logical B128
- model_plan.cpp
- EXP-0390 — P512 runtime-variance boundary
- EXP-V2-0015 — Suffix TTFT Roofline & Kernel Attribution
- V2-0006 — GQA Attention Optimization Sprint & Architectural Bakeoff
- PersistentSessionHeader
- Prefill V2 Architecture Specification
- EXP-0387 — M28 full-model prefill frontier refresh
- V2-0005 — Native P512 Full-Model Macro Tiling & Baseline Qualification
- tensor
- MIInfer Current State of the Art
- EXP-0395 — Legacy B4 tail fallback for logical B128
- EXP-0358 — M26-C current decode-route differential
- EXP-0389 — P512 regression discriminator
- EXP-0392 — P512 runtime-state scope
- ShapeBenchmark
- PhysicalKvView
- Qwen3Tokenizer
- v2_0045_qualification_bench.cpp
- main
- EXP-0351 — M26 real-context decode floor and context penalty
- PrefillV2WorkspaceManager
- RecurrentLayerState
- EXP-V2-0028 — Persistent Prefix Cache & Session Restore Qualification
- EXP-V2-0010: DeltaNet Transposed Wave Acceleration & Defeating mx-llama.cpp in Decode
- EXP-V2-0011 — Recovering the Historical ~32 ms Decode Frontier in Prefill V2
- DeviceShapeData
- V2-0047 — Release packaging and distribution hardening
- openai_api.cpp
- GenerateStats
- LogicalRange
- EXP-V2-0013 — Long-Context Frontier Qualification (4K -> 8K -> 16K -> 32K -> 64K -> 128K)
- EXP-V2-0012 — Reclaiming Resident VRAM via High-Value Layout Pruning while Preserving mx-Beating Decode
- EXP-V2-0007 — Unified Prefill V2 to Static Decode Pipeline
- RecurrentLayerDecodePhaseTimings
- Options
- EXP-V2-0027R — Cold-Prefill Arithmetic & Roofline Reconciliation
- EXP-V2-0018-suffix-hip-graph.md
- EXP-V2-0009: Decode Fast-Path Recovery Inside Prefill V2
- prefill_v2_model_bench.cpp
- hip_check.hpp
- Qwen3Layer0KvCache
- EXP-V2-0021-production-suffix-hip-graph.md
- OpenAiChatRequest
- EXP-V2-0014 — Prefix & State Reuse with Suffix-Only Prefill
- m31_0003_canonical_agent_workload.cpp
- DevicePrefillState
- TerminalTextStream
- M23ProfileCounters
- v2_0023_cross_query_attention_bench.cpp
- Iteration log
- qwen3_swiglu_q8_bench.cpp
- Qwen3GpuPlan
- qwen35_gpu_pipeline.hpp
- original-m31_0002_checkpointed_call.cpp
- qwen3_decode_profile.cpp
- M31-LC-0001 — MMQ-Only Long-Context Advantage
- Group
- qwen3_cached_attention_determinism_gpu_test.cpp
- EXP-0359 — M26-CQ decode semantic equivalence
- V2-0045 — Definitive competitive qualification
- EXP-V2-0016-wave64-suffix-attention.md
- recurrent_block
- EXP-V2-0017-hierarchical-gqa-query-reuse.md
- v2_0024_quantized_prefix_kv_bench.cpp
- Potential areas
- AttentionLayerDecodePhaseTimings
- DeviceBuffer
- EXP-V2-0025-lds-staged-gqa-attention.md
- m12_dense_stage_bench.cpp
- bench-m18-runtime.py
- GenerateOptions
- ShapeBenchmark
- EXP-V2-0019-quantized-kv-cache.md
- PrefillV2Model::PrefillV2Model
- v2_0025_lds_staged_gqa_attention_bench.cpp
- bench_m10_latency_curve.py
- v2_0019_quantized_kv_bench.cpp
- m31_0001_performance_frontier.cpp
- V2-0043 — Reference-shaped GQA attention rewrite
- DeviceBuffer
- gfx906 Toolbox Reference and MIInfer Follow-up
- v2_0021_production_suffix_hip_graph_bench.cpp
- EXP-V2-0030 — Direct Global-to-VGPR MMQ Weight Streaming & Instruction-Amplification Reduction
- PrefillV2Model::snapshot
- AccuracyMetrics
- GemvShape
- qwen35_gqa_prefill_bench.cpp
- run-m31-exploratory.sh
- V2-0048B — Generation completion contract
- resolve_model
- hipMemcpyAsync
- bench-serve-concurrency.py
- M30-0000 — Reuse Opportunity Map (Workstream B)
- hipMemcpy
- 5. Runtime Layers
- E2-0003 — MMQ-only Gate/Up candidate
- ProfileScope
- EXP-V2-0029 — MMQ Useful-Compute Efficiency & Instruction-Amplification Analysis
- V2-0048A — Final artifact release gate
- run_case
- m12_gdn_oracle.cpp
- V2-0048A — Generation budget and exact-prefix fault
- model_loader_test.cpp
- m31_0002_persistent_context_qualification.cpp
- model.cpp
- M30-0002 production authority and residual agent cost
- Qwen35Model
- GgufFile
- v2_0045_agent_workload.py
- Event
- Parallel MI50 Development
- M31-0006 — Build and stop-guard status
- download
- attention_layer.cpp
- q4k_layout_bench.cpp
- RecurrentLayerPhaseTimings
- Q8ExactBlock
- qwen3_gpu_layer.cpp
- test-cli-first-run.sh
- DeviceDecodeState
- AttentionLayerProfileBreakdown
- NumericalMetrics
- developer-guide.md
- M30-0001 final evidence
- PhysicalKvShardView
- m24_recurrent_layer_bakeoff.cpp
- 3. Correctness Is a Benchmark Prerequisite
- E2-0003Q qualification report
- Buffer
- MIInfer Current State
- M31-0002T-0001 — First-divergence and weight-residency attribution
- M31-0003 — Consolidated findings and bounded next steps
- Canonical Forward Roadmap
- Q4KWaveSwigluFusedTile
- FfnTailReplay
- test_m31_exploratory_thermal_guard.sh
- decode_fast_path_bakeoff.cpp
- M30-0005 Shared Snapshot State and Copy-on-Write Contract
- M31-0004 — GDN computation findings
- M31-0003 — Production execution path
- M31-0005 — Implementation results
- M1 — Kernel Laboratory
- V2-0044 — Precision-preserving KQ-to-V path
- DeviceInfo
- E2-0003 recovery results
- make_prompt
- CompareMetrics
- m31_0003_canonical_agent_workload.py
- DeviceBuffer
- M30-0000 — Canonical Agent Baseline
- M30-0000 — Integrated Evidence
- M30-0001 Exact-Prefix Continuation Contract
- M30-0003 Multi-Checkpoint and Longest-Prefix Reuse
- M30-0004 Snapshot, Fork, and Rollback Evidence
- M31-0001 — Single-MI50 Performance Frontier
- M31-0002 — Persistent-Context Qualification
- M31-0003 — Canonical Agent Workload
- M31-0004 — GDN checkpoint lifecycle
- M31-0004 — Dispatch and memory findings
- M31-0004 — Optimization results
- M31-0004 — Stability and Release Gate
- M31-0003 — Attention and KV investigation
- M31-0003 — VRAM ownership and accounting
- M31-0005 — Correctness and regression protection
- M31-0005 — Decode attribution
- M31-0005 — KV storage and memory ledger
- M29-0002 Workstream A — HIP VMM feasibility
- M29-0003 Workstream B — ContextSpace + logical pages
- GemvKernelResources
- Q4KMmqTile
- fp16_gemv_reference.cpp
- prefill_v2_agent_branch_workload_test.cpp
- EventPair
- StageEvents
- D032 — Persistent Context and Device-Local KV Ownership
- M29-0006 physical KV view
- M30-0000 canonical agent workload
- M30-0004 Snapshot, Fork, and Rollback Contract
- M30-0005 Shared Snapshot State and COW Evidence
- M31-0004 — Correctness and validation
- M31-0002T-0001 — First-divergence investigation
- M31-0002T-0001 — Numerical and state correctness audit
- M31-0002T-0001 — Reference comparison and regression tests
- M31-0002T-0001 — Weight residency and VRAM ledger
- M31-0003 — Decode and MMQ investigation
- M31-0003 — Host dispatch and synchronization
- M31-0003 — llama.cpp comparison limits
- M31-0003 — Prefill and GDN investigation
- M31-0005 — Attention kernel audit
- M31-0005 — Dispatch and synchronization audit
- Post-M28 production baseline
- M29-0002 Workstream B — 128K memory budget
- M29-0003 Workstream A — 128K lifetime/allocation fix
- prefill_v2_exact_prefix_reuse_test.cpp
- Buffer
- RawBuffer
- Buffer
- GpuBuffer
- GpuBuffer
- M26-D — One evidence-backed floor reduction
- E2-0003Q source and sampler audit
- ExtractedAssistantOutput
- m31_0004_stability_release_gate.py
- D026 — Strongest Available Relevant Baseline Wins
- D027 — Serving Does Not Define the Core Runtime
- M29-0004 Workstream A — DeviceKvPool
- M29-0004 Workstream B — DeviceKvShard + PlacementPlan
- run-m31-0006-attention-microbench.sh
- PresetFlag
- 35. Parallel MI50 development
- AGENTS.md
- as
- m31-0005-llama-comparison.md
- README.md
- README.md
- README.md
- phase0-preflight.md
- fake-rocm-smi.sh
- check-qualified-host.sh

## God Nodes (most connected - your core abstractions)
1. `RecurrentLayer` - 259 edges
2. `FullAttentionLayer` - 183 edges
3. `hipMemcpy()` - 136 edges
4. `Qwen35RuntimeEngine` - 121 edges
5. `PrefillV2Model` - 111 edges
6. `hipFree()` - 87 edges
7. `half()` - 69 edges
8. `Iteration log` - 61 edges
9. `GgufTensor` - 59 edges
10. `Qwen3LayerTrace` - 55 edges

## Surprising Connections (you probably didn't know these)
- `evaluate_token_count()` --calls--> `download`  [INFERRED]
  bench/prefill_v2_recurrent_layer_bakeoff.cpp → include/miinfer/prefill_v2/state.hpp
- `evaluate_token_count()` --calls--> `upload`  [INFERRED]
  bench/prefill_v2_recurrent_layer_bakeoff.cpp → include/miinfer/prefill_v2/state.hpp
- `run_bakeoff_main()` --calls--> `forward`  [INFERRED]
  bench/prefill_v2_recurrent_layer_bakeoff.cpp → include/miinfer/prefill_v2/recurrent_layer.hpp
- `PrefillV2Model::capture_snapshot_backing()` --calls--> `find_longest_snapshot_prefix`  [INFERRED]
  src/prefill_v2/model.cpp → include/miinfer/prefill_v2/model.hpp
- `GpuWeightArena::release()` --calls--> `hipFree()`  [INFERRED]
  src/model_plan.cpp → tests/reusable_context_allocation_test.cpp

## Import Cycles
- None detected.

## Communities (908 total, 76 thin omitted)

### Community 0 - "hardware.md"
Cohesion: 0.04
Nodes (46): 10. Candidate Quantized Execution Path, 11. FP16 Behavior, 12. BF16, 13. Memory Bandwidth, 14. HBM vs Cache, 15. Weight Compression, 16. HBM Clock, 17. GPU Clock (+38 more)

### Community 1 - "architecture.md"
Cohesion: 0.05
Nodes (40): 10. Memory Architecture, 11. Weight Residency, 12. Tensor Layout, 13. Prefill vs Decode, 14. Context-Length Sensitivity, 15. Attention Architecture, 16. MoE Architecture, 17. Static Model Knowledge (+32 more)

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
Nodes (40): Architecture direction, Benchmark philosophy, Building, Contributing, Core hypothesis, Correctness before performance, Current CMake build graph, Design principles (+32 more)

### Community 6 - "current-state.md"
Cohesion: 0.06
Nodes (31): Completed, Completed in Task 3, Current Benchmark Priority, Current Build Direction, Current Correctness Policy, Current Dependency Policy, Current Experiment Queue, Current Hardware Observation Requirements (+23 more)

### Community 7 - "AGENTS.md"
Cohesion: 0.07
Nodes (28): 11. Static specialization, 12. Static kernel selection, 13. HIP graph strategy, 14. Benchmarking, 15. Hardware-state validation, 16. Benchmark methodology, 18. Experiment records, 19. Profiling (+20 more)

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
Cohesion: 0.11
Nodes (18): 10. nick413-bit/gfx906-fa-vllm, 12. AMD gfx906 ISA Documentation, 13. AMD HIP / ROCm Documentation, 14. rocBLAS / hipBLAS, 16. Current Research Synthesis, 18. Research Intake Checklist, 19. External Code Policy, 20. Research Notes vs Decisions (+10 more)

### Community 19 - "decisions.md"
Cohesion: 0.15
Nodes (12): D010 — No Silent CPU or Generic Fallback, D015 — Repacking Should Not Occur in Hot Paths, D023 — Triton Is a Research Tool, Not a Required Runtime, Decision, Decision, Decision, Decision Change Process, Guiding Rule (+4 more)

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
Cohesion: 0.06
Nodes (31): Architectural rule, Benchmark before claim, Candidate areas, Contract, Core runtime responsibilities, Correctness before speed, Correctness validation, Current Status (+23 more)

### Community 24 - "Fp16GemvMetrics"
Cohesion: 0.25
Nodes (8): Fp16GemvMetrics, cosine_similarity, inf_detected, max_abs_error, max_relative_error, mean_abs_error, nan_detected, pass

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

### Community 32 - "Decision / next work"
Cohesion: 0.13
Nodes (15): Corrected selected reference schedule, Decision / next work, Decode regression A/B, Dispatch geometry validity caveat, Durable narrow serving requalification — 2026-09-28, Exact P8192 dispatch reconciliation (2026-09-28), EXP-V2-0042 — P8192 trace attribution and sampler-change decode A/B, Exploratory serving comparison (not qualification) (+7 more)

### Community 33 - "WidePrefillWorkspace"
Cohesion: 0.08
Nodes (25): WidePrefillWorkspace, attention_gate, core_beta, core_decay, core_gate, core_key, core_query, core_value (+17 more)

### Community 34 - "4. Candidate Optimization Roadmap for M8"
Cohesion: 0.11
Nodes (18): 1. Executive Summary, 2.1 Model Weight and Memory Inventory, 2.2 Execution Stage Latency Breakdown (Per-Token), 2. Phase M8-B: Fresh Post-M7 Latency Attribution, 3.1 Hardware Memory Bandwidth Upper Bounds, 3.2 Bandwidth-Bound Decode Floors for 15.932 GiB Compulsory Weight Transfer, 3.3 The "Latency Tax" Above the Memory Floor, 3.4 Target Budget for 30.00 tok/s (+10 more)

### Community 35 - "32. Codex task behavior"
Cohesion: 0.29
Nodes (7): 32. Codex task behavior, Before adding abstractions, Before adding fallback behavior, Before declaring a performance win, Before modifying code, Before removing apparently strange gfx906 code, Mandatory performance research protocol

### Community 36 - "PrefillV2Workspace"
Cohesion: 0.06
Nodes (32): Q8_1Block, PrefillV2Workspace, attn_gated_output, attn_k, attn_q_rope, attn_qfull, attn_v, beta (+24 more)

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

### Community 41 - "DeviceKvShard"
Cohesion: 0.07
Nodes (26): DeviceKvShard, device_, heads_, logical_, page_, physical_, LogicalPageId, size_t (+18 more)

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
Cohesion: 0.01
Nodes (183): Q5KMmqTile, RecurrentLayer, alpha_raw, beta, beta_raw, d_a, d_alpha, d_attn_gate_native (+175 more)

### Community 48 - "FullAttentionLayer"
Cohesion: 0.01
Nodes (139): RocblasGemmHandle, opaque, FullAttentionLayer, attention, batch_head_rms, d_attn_norm, d_ffn_down, d_ffn_down_expanded (+131 more)

### Community 49 - "LayerEvaluationResult"
Cohesion: 0.08
Nodes (37): compute_metrics(), size_t, span, uint32_t, vector, evaluate_token_count(), generate_initial_history(), generate_initial_state() (+29 more)

### Community 50 - "EXP-0179 — Native Q6_K Wave64 Rollout & Activation Reuse"
Cohesion: 0.12
Nodes (16): 1. TG64 (64 tokens), 2. TG128 (128 tokens), 3. Comparison with Pinned llama.cpp Baseline, 4. Hardware State & Telemetry, Baseline, Benchmark Methodology, Candidate, Correctness (+8 more)

### Community 51 - "29. Decision"
Cohesion: 0.40
Nodes (5): 29. Decision, INVALID, KEEP, REJECT, RETEST

### Community 52 - "6. Baseline"
Cohesion: 0.40
Nodes (5): 6. Baseline, Commit, Implementation, Kernel, Relevant configuration

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
Cohesion: 0.08
Nodes (63): AccumulationContract, byte, GgufTensorType, Qwen3TensorView, source, RmsReduction, AttentionPathReplay, attention_output (+55 more)

### Community 58 - "qwen3_forward_gpu_test.cpp"
Cohesion: 0.16
Nodes (30): argmax(), compare_checkpoint(), path, Q8_1Block, size_t, vector, exact_equal(), main() (+22 more)

### Community 59 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 60 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 61 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 62 - "v2_0020_pipeline_bench.cpp"
Cohesion: 0.29
Nodes (9): compute_diff(), hipEvent_t, uint32_t, vector, elapsed_ms(), main(), splitk_suffix_attn_pipe1_double_buf_u1_kernel(), splitk_suffix_attn_pipe2_double_buf_u2_kernel() (+1 more)

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

### Community 95 - "DeviceKvPool"
Cohesion: 0.13
Nodes (21): FreeRange, DeviceKvPool, active_, allocate, backing_, capacity_, committed_, free_ (+13 more)

### Community 96 - "m31_0006_graph_attention_microbench.cpp"
Cohesion: 0.17
Nodes (23): capture_attention_graph(), compare(), copy_output(), hipEvent_t, hipGraphExec_t, hipStream_t, size_t, uint32_t (+15 more)

### Community 97 - "ComponentTimingBreakdown"
Cohesion: 0.05
Nodes (39): ComponentTimingBreakdown, act_swiglu_ms, aggregate_gpu_kernel_ms, argmax_ms, gdn_beta_alpha_gemm_ms, gdn_beta_decay_prep_ms, gdn_chunk_scan_ms, gdn_conv1d_silu_split_ms (+31 more)

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

### Community 105 - "EXP-0364 — Full-model Qwen3.8-27B prefill bottleneck attribution"
Cohesion: 0.09
Nodes (22): Are projections close enough to mx to reject broad MMQ work?, Benchmark matrix, Canonical MIInfer route, Context scaling, Decision, Dispatch census, Does the dominant gap change with context?, Environment (+14 more)

### Community 106 - "MIInfer Performance Research Protocol"
Cohesion: 0.11
Nodes (19): Benchmark ladder, Bottleneck-first and Amdahl gates, Compiler-resource gate, Correctness gate, Current frontier status — 2026-09-29, DEFERRED, Environment, thermal, and commit hygiene, Historical next-frontier research plan (superseded by V2-0044 closeout above) (+11 more)

### Community 107 - "Benchmarks"
Cohesion: 0.06
Nodes (31): Benchmarks, End-to-end M5-A baseline, M5-B steady-state decode profile, M5-C0 trace-free decode benchmark, M5-C10a refreshed P64 production profile, M5-C10b normalization/conversion boundary attribution, M5-C10c FFN normalization-to-shared-Q8 fusion, M5-C11a production and llama.cpp differential baseline (+23 more)

### Community 108 - "EXP-0007 — gfx906 Zero-Point-Corrected Q4_0 × Q8_1 Dot4"
Cohesion: 0.12
Nodes (16): 10. Hardware validity, 11. External MMVQ comparison, 12. Projection-only sanity check, 13. Decision, 14. M2 status, 15. Next experiment, 1. Question, 2. Hypothesis and motivation (+8 more)

### Community 109 - "qwen3_tokenizer.cpp"
Cohesion: 0.19
Nodes (29): byte_encode(), byte_to_codepoint(), Codepoint, begin, end, codepoint_to_byte(), value, codepoints() (+21 more)

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

### Community 121 - "qwen3_inference_bench.cpp"
Cohesion: 0.09
Nodes (47): argmax(), build_json(), ostream, size_t, string, timespec, uint32_t, vector (+39 more)

### Community 122 - "EXP-0382 — Revalidate the Actual Composed-B64 Runtime"
Cohesion: 0.09
Nodes (22): A — zero generated tokens, Allocation and regression gates, Allocation/setup audit, B — first token, C — one direct decode step, Composed-B64 wrapper ladder, Decision, Exact next PRIMARY (+14 more)

### Community 123 - "run_sequence"
Cohesion: 0.22
Nodes (20): snapshot_keys, snapshot_values, attention_contract(), cache_corruption_test(), cache_slots_preserved(), checkpoints(), compare_trace(), path (+12 more)

### Community 124 - "External gfx906 Reference Baseline"
Cohesion: 0.22
Nodes (8): Baseline status, Checkout, External gfx906 Reference Baseline, Initial baseline, MI50 build starting point, Option validation on the available host, Pin, Toolchain preflight record

### Community 125 - "q4_q8_gemv_bench.cpp"
Cohesion: 0.16
Nodes (28): allocate_shape(), ostream, string, vector, escape(), free_shape(), main(), median() (+20 more)

### Community 126 - "Qwen3GpuDecodeWorkspace"
Cohesion: 0.05
Nodes (41): DeviceBuffer, DeviceBytes, bytes_, Qwen3GpuDecodeWorkspace, argmax_token, attention, attention_projected, attn_norm (+33 more)

### Community 127 - "EXP-0086 — M6-A27.6 Qwen3.8 P2 L0–L2 operation trace"
Cohesion: 0.25
Nodes (7): Decision, EXP-0086 — M6-A27.6 Qwen3.8 P2 L0–L2 operation trace, Follow-up, Interpretation, Method, Question, Results

### Community 128 - "fp16_gemv_reduction_diag.cpp"
Cohesion: 0.18
Nodes (16): string, vector, main(), median(), nonnegative(), Options, device, iterations (+8 more)

### Community 129 - "EXP-0300 — M25-A MIInfer versus mx-llama differential"
Cohesion: 0.18
Nodes (10): Baseline and environment, Correctness, Decision, End-to-end P512, EXP-0300 — M25-A MIInfer versus mx-llama differential, Follow-up, Hypothesis, Interpretation (+2 more)

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

### Community 134 - "EXP-0331 — Pinned Mx MMQ register weight prefetch"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0331 — Pinned Mx MMQ register weight prefetch, Follow-up (+4 more)

### Community 135 - "qwen3_primitives.cpp"
Cohesion: 0.11
Nodes (37): int16_t, int8_t, uint16_t, uint8_t, Q4_0HostBlock, d_bits, qs, Q6KHostBlock (+29 more)

### Community 136 - "Qwen3GpuProfile"
Cohesion: 0.05
Nodes (38): array, hipEvent_t, qwen3_profile_category_count, Qwen3BoundaryProfileStage, Qwen3FfnProfileStage, Qwen3ProfileCategory, Qwen3GpuProfile, boundary_bytes (+30 more)

### Community 138 - "M3 Minimal Qwen3-8B Runtime Scaffold"
Cohesion: 0.25
Nodes (7): GPU ownership and plan, M3 Minimal Qwen3-8B Runtime Scaffold, Parser boundary, Static projection kernel selection, Supported artifact, Validated configuration, Validation command

### Community 139 - "ContextSpace"
Cohesion: 0.15
Nodes (18): ContextSpace, append, committed_, next_id_, page_id_at, page_size_, pages_, remap (+10 more)

### Community 140 - "Qwen3LayerTrace"
Cohesion: 0.04
Nodes (54): Qwen3LayerTrace, attention_output, attention_probabilities, attention_scores, attn_norm, attn_rms, embedding, ffn_input (+46 more)

### Community 141 - "EXP-0074 — M6-A26.4 L30 production operand attribution"
Cohesion: 0.18
Nodes (10): Baseline and method, Decision, Environment and command, EXP-0074 — M6-A26.4 L30 production operand attribution, Follow-up, Interpretation, One-at-a-time GPU substitutions, Production operand comparison (+2 more)

### Community 142 - "EXP-0041 — M5-C15 optimization closure and parity decision gate"
Cohesion: 0.15
Nodes (12): 1. Question, 2. Authoritative production result, 3. M5 optimization record, 4. Experimentally eliminated explanations, 5. M5 result, 6. Architectural decision gate, 7. Decision, Accepted (+4 more)

### Community 143 - "run_bakeoff_main"
Cohesion: 0.14
Nodes (17): compute_metrics(), span, vector, generate_initial_history(), generate_initial_state(), generate_realistic_input(), run_bakeoff_main(), size_t (+9 more)

### Community 144 - "EXP-0100 — M6-B7 Q5_K paired-nibble decoding"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Commands, Correctness and resources, Decision, Environment, EXP-0100 — M6-B7 Q5_K paired-nibble decoding, Follow-up (+3 more)

### Community 145 - "qwen3_layer_host_impl"
Cohesion: 0.16
Nodes (29): add_in_place(), int8_t, size_t, span, uint16_t, uint32_t, vector, execute_qwen3_decode_host() (+21 more)

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

### Community 152 - "EXP-0379 — B64 whole-model composition"
Cohesion: 0.12
Nodes (16): Allocation/setup audit, Candidate, Clean current-HEAD timing, Coarse stage attribution, Decision, Exact next PRIMARY, Execution-mode counters, EXP-0375 basis (+8 more)

### Community 154 - "EXP-0026 — M5-C8c Down long-K bottleneck attribution"
Cohesion: 0.17
Nodes (11): 10. Follow-up, 1. Question, 2. Hypothesis, 3. Motivation and prior evidence, 4. Method, 5. Results, 6. Static gfx906 evidence, 7. Interpretation (+3 more)

### Community 158 - "Q8BoundaryDiff"
Cohesion: 0.25
Nodes (8): Q8BoundaryDiff, different_lane_values, different_scale_blocks, different_sum_blocks, first_block, first_lane, max_scale_bits_delta, max_sum_delta

### Community 160 - "EXP-0046 — M6-A4 Qwen3.8-27B full-attention layer"
Cohesion: 0.20
Nodes (9): Candidate, Commands, Correctness gates, Decision, EXP-0046 — M6-A4 Qwen3.8-27B full-attention layer, Follow-up, Question, Reference and baseline (+1 more)

### Community 161 - "EXP-0305 — M25-D mx GDN register-resident scan"
Cohesion: 0.14
Nodes (13): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0305 — M25-D mx GDN register-resident scan, Follow-up (+5 more)

### Community 162 - "EXP-0042 — M6-A0 Qwen3.8-27B GGUF and architecture audit"
Cohesion: 0.11
Nodes (18): 10. Files changed, 11. Checks run, 12. Conclusion, 1. Question, 2. Local artifacts, 3. GGUF metadata, 4. Layer pattern, 5. Tensor inventory (+10 more)

### Community 163 - "m6a263_qwen35_recurrent_contract.cpp"
Cohesion: 0.20
Nodes (16): apply_external(), checkpoint(), compare(), path, size_t, span, string_view, vector (+8 more)

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

### Community 172 - "Qwen35Config"
Cohesion: 0.10
Nodes (20): uint32_t, Qwen35Config, attention_heads, block_count, context_length, full_attention_interval, head_dim, hidden_size (+12 more)

### Community 173 - "hip_smoke_bench.cpp"
Cohesion: 0.16
Nodes (15): size_t, string, json_escape(), main(), Options, device, elements, iterations (+7 more)

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

### Community 186 - "Sha256"
Cohesion: 0.18
Nodes (15): array, byte, size_t, span, string, uint32_t, uint64_t, rotate_right() (+7 more)

### Community 187 - "EXP-0011 — Trace-free Qwen3-8B decode control"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0011 — Trace-free Qwen3-8B decode control, Follow-up (+4 more)

### Community 188 - "EXP-0099 — M6-B6 Q5_K subgroup-structured dot loop"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Commands, Correctness and resources, Decision, Environment, EXP-0099 — M6-B6 Q5_K subgroup-structured dot loop, Follow-up (+3 more)

### Community 189 - "EXP-0022 — M5-C6d GPU-side greedy argmax"
Cohesion: 0.18
Nodes (10): Candidate, Correctness, Decision, Environment and workload, EXP-0022 — M5-C6d GPU-side greedy argmax, Follow-up, Hypothesis, Performance (+2 more)

### Community 190 - "Buffer"
Cohesion: 0.50
Nodes (3): Buffer, p, size_t

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

### Community 207 - "PersistentSession::load_from_file"
Cohesion: 0.31
Nodes (12): hipStream_t, PrefillV2Model, span, string, uint32_t, uint64_t, vector, PersistentSession::find_matching_session() (+4 more)

### Community 209 - "PrefillV2Model"
Cohesion: 0.03
Nodes (68): hipGraphExec_t, KvCacheQuantMode, mt19937, pair, size_t, SnapshotBacking, SnapshotId, SnapshotRecord (+60 more)

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
Cohesion: 0.19
Nodes (30): checkpoint(), compare(), conv_output(), array, kChannels, path, size_t, span (+22 more)

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

### Community 223 - "m12_gdn_chunk_bench.cpp"
Cohesion: 0.17
Nodes (17): as(), at(), Buffer, pointer, Fn, hipEvent_t, size_t, T (+9 more)

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

### Community 228 - "gguf.cpp"
Cohesion: 0.06
Nodes (70): GgufError, GgufValue, value, GgufScalar, vector, byte, GgufTensorType, size_t (+62 more)

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

### Community 243 - "GgufTensor"
Cohesion: 0.06
Nodes (76): main(), main(), string, find_tensor(), main(), main(), GgufTensorType, uint64_t (+68 more)

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

### Community 248 - "m6a14_qwen35_state_audit.cpp"
Cohesion: 0.13
Nodes (35): kRecurrentLayers, align_up(), cache_fingerprint(), array, History, path, size_t, span (+27 more)

### Community 249 - "EXP-0252 — Repacked MMQ64 retest on the production Q4_K down shape"
Cohesion: 0.14
Nodes (13): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0252 — Repacked MMQ64 retest on the production Q4_K down shape, Follow-up (+5 more)

### Community 250 - "EXP-0254 — Repacked MMQ64 exact-sum activation layout"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness, Decision, Environment, EXP-0254 — Repacked MMQ64 exact-sum activation layout, Follow-up, Hypothesis (+2 more)

### Community 251 - "run-m6b0-llama-baseline.sh"
Cohesion: 0.83
Nodes (3): cleanup(), run-m6b0-llama-baseline.sh script, stop_telemetry()

### Community 252 - "DeviceKvBlock"
Cohesion: 0.12
Nodes (19): DeviceKvBlock, bytes, data, device, id, offset, coalesce, owns (+11 more)

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

### Community 258 - "EXP-0377 — Qualify Existing B64 Residual Contract"
Cohesion: 0.20
Nodes (9): Decision, Exact next PRIMARY, Existing B64 contract audit, EXP-0376 authorization, EXP-0377 — Qualify Existing B64 Residual Contract, Protocol gate, Question, Results (+1 more)

### Community 259 - "m6a12_qwen35_attention_projections.cpp"
Cohesion: 0.33
Nodes (7): path, size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

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

### Community 267 - "Qwen3FfnProbeTrace"
Cohesion: 0.15
Nodes (13): vector, Qwen3DownProjectionContractTrace, current_s_correction, direct_signed_oracle, exact_sum_correction, Qwen3FfnProbeTrace, ffn_output, gate (+5 more)

### Community 268 - "EXP-0380 — Localize the first B64 attention-composition hang"
Cohesion: 0.12
Nodes (16): B128 comparison, Clean exact-P960 sparse-ladder reruns, Complete bounded B64 chunk, Decision, Exact next PRIMARY, EXP-0379 evidence, EXP-0380 — Localize the first B64 attention-composition hang, Extended sparse-ladder evidence (+8 more)

### Community 269 - "EXP-0096 — M6-B3 Q4_K metadata staging"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Commands, Correctness and resources, Decision, Environment, EXP-0096 — M6-B3 Q4_K metadata staging, Follow-up (+3 more)

### Community 270 - "EXP-0182 — Fused Gate+Up SwiGLU Wave64 GEMV"
Cohesion: 0.12
Nodes (15): 1. TG64 (64 tokens), 2. TG128 (128 tokens), 3. Hardware State & Telemetry, Baseline, Benchmark Methodology, Candidate, Correctness, Decision (+7 more)

### Community 271 - "Qwen35RuntimeEngine"
Cohesion: 0.03
Nodes (63): Exp0376ChunkTiming, PrefillProfile, RemainderSchedulerCounters, array, Buffer, hipGraphExec_t, unique_ptr, Qwen35RuntimeEngine (+55 more)

### Community 272 - "EXP-0066 — M6-A22 Qwen3.8-27B GPU hybrid position audit"
Cohesion: 0.17
Nodes (11): Artifact and reference, Candidate, Checks, Command, Decision, EXP-0066 — M6-A22 Qwen3.8-27B GPU hybrid position audit, Follow-up, Interpretation (+3 more)

### Community 273 - "m6a18_qwen35_deltanet_state_gpu.cpp"
Cohesion: 0.23
Nodes (13): path, size_t, span, T, vector, DeviceBuffer, data_, logical_state() (+5 more)

### Community 274 - "EXP-0141 — M6-B49 recurrent state-update/head-RMS fusion"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment, EXP-0141 — M6-B49 recurrent state-update/head-RMS fusion, Follow-up, Hypothesis, Question (+1 more)

### Community 275 - "PrefillV2AttentionLayer"
Cohesion: 0.09
Nodes (19): size_t, uint8_t, vector, PrefillV2AttentionLayer, d_attn_norm_, d_ffn_down_mmq_, d_ffn_gate_mmq_, d_ffn_swiglu_fused_ (+11 more)

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
Cohesion: 0.08
Nodes (25): shared_ptr, size_t, string, uint32_t, vector, Qwen3Config, attention_heads, context_length (+17 more)

### Community 285 - "EXP-0080 — M6-A27 Qwen3.8 sixty-four-layer GPU composition"
Cohesion: 0.25
Nodes (7): Candidate, Decision, Environment and command, EXP-0080 — M6-A27 Qwen3.8 sixty-four-layer GPU composition, Follow-up, Question, Results

### Community 286 - "EXP-0085 — M6-A27.5 Qwen3.8 P2 drift localization"
Cohesion: 0.18
Nodes (10): Decision, EXP-0085 — M6-A27.5 Qwen3.8 P2 drift localization, Follow-up, Hypothesis, Interpretation, Method, Observable consequence, P2 layer-output error scan (+2 more)

### Community 287 - "EXP-0070 — M6-A26 Qwen3.8 thirty-two-layer stateful GPU prefix"
Cohesion: 0.20
Nodes (9): Candidate, Decision, Environment and command, EXP-0070 — M6-A26 Qwen3.8 thirty-two-layer stateful GPU prefix, Follow-up, Interpretation, Question, Results (+1 more)

### Community 288 - "Qwen3GpuDecodeCache"
Cohesion: 0.10
Nodes (22): validate_cache(), size_t, unique_ptr, Qwen3GpuDecodeCache, caches_, length, prepare, workspace_ (+14 more)

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

### Community 293 - "m24_projection_bakeoff.cpp"
Cohesion: 0.19
Nodes (15): as(), Buffer, pointer, Block, GgufTensorType, size_t, string, T (+7 more)

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

### Community 318 - "qwen3_trace_compare.cpp"
Cohesion: 0.19
Nodes (18): argmax(), compare(), path, size_t, vector, main(), Metrics, first_value (+10 more)

### Community 319 - "EXP-0094 — M6-B2 direct layer-output handoff"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Decision, Environment and workload, EXP-0094 — M6-B2 direct layer-output handoff, Follow-up, Hypothesis, Interpretation (+2 more)

### Community 320 - "require_match"
Cohesion: 0.20
Nodes (15): check_device(), check_values(), Metrics, path, size_t, span, T, uint32_t (+7 more)

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

### Community 330 - "EXP-0274 — M18–M20 qualification closure"
Cohesion: 0.25
Nodes (7): Architectural conclusion, Environment, EXP-0274 — M18–M20 qualification closure, Final decision, Gate summary, Runtime-only result, Scope

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

### Community 342 - "AggregatedStats"
Cohesion: 0.11
Nodes (22): aggregate(), AggregatedStats, generated_tokens, max_decode_tok_s, mean_decode_tok_s, mean_step_ms, mean_total_ms, mean_ttft_ms (+14 more)

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

### Community 374 - "RunResult"
Cohesion: 0.13
Nodes (21): size_t, uint32_t, vector, main(), now_ms(), options(), require(), run_snapshot_workload() (+13 more)

### Community 375 - "EXP-0237 — M11-B production layer-major prefill profile"
Cohesion: 0.22
Nodes (8): Amdahl interpretation, Decision, Environment, EXP-0237 — M11-B production layer-major prefill profile, Follow-up, Method, Question, Results

### Community 376 - "EXP-0165 — M6-B69 dual query/key head normalization"
Cohesion: 0.25
Nodes (7): Candidate, Correctness, Decision, Environment, EXP-0165 — M6-B69 dual query/key head normalization, Question, Results

### Community 377 - "EXP-0160 — M6-B68 DeltaNet ordered row-wave reduction"
Cohesion: 0.33
Nodes (5): Candidate, Decision, EXP-0160 — M6-B68 DeltaNet ordered row-wave reduction, Question, Results

### Community 378 - "qwen3_decode_sequence_gpu_test.cpp"
Cohesion: 0.24
Nodes (23): argmax(), cache_lengths(), compare_dumps(), path, size_t, span, string, uint32_t (+15 more)

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
Cohesion: 0.11
Nodes (17): 1. Executive Summary, 2.1 `miinfer inspect`, 2.1a `miinfer config`, 2.1b `miinfer models`, 2.2 `miinfer run`, 2.3 `miinfer chat`, 2.4 `miinfer serve`, 2. Command Architecture (+9 more)

### Community 383 - "EXP-0158 — M6-B66 expanded Q4_K FFN Gate/Up"
Cohesion: 0.29
Nodes (6): Candidate, Correctness, Decision, Environment, EXP-0158 — M6-B66 expanded Q4_K FFN Gate/Up, Question

### Community 384 - "qwen3_layer0_gpu_test.cpp"
Cohesion: 0.25
Nodes (17): abs_tolerance(), checkpoints(), compare(), path, size_t, string, vector, main() (+9 more)

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

### Community 394 - "Post-M26 Performance Roadmap"
Cohesion: 0.09
Nodes (22): Candidate architecture, Decision gate after M26-F, Exit, Exit, Exit artifact, Goal, Goal, M26-C — Close current route attribution (+14 more)

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

### Community 399 - "EXP-0393 — Requalify the aligned P512 baseline"
Cohesion: 0.18
Nodes (10): Contract, EXP-0393 — Requalify the aligned P512 baseline, Gate state, Hardware and runtime audit, MIInfer P512 samples, mx sentinels, Provenance, Qualification decision (+2 more)

### Community 400 - "EXP-0228 — M11-B Q4_K four-token subwave mapping"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Correctness, Decision, Environment, EXP-0228 — M11-B Q4_K four-token subwave mapping, Follow-up, Hypothesis (+3 more)

### Community 401 - "EXP-0297 — M24 exact resident-MMQ projection bake-off"
Cohesion: 0.20
Nodes (9): Baseline and candidate, Correctness, Decision, Environment, EXP-0297 — M24 exact resident-MMQ projection bake-off, Follow-up, Hypothesis, Interpretation (+1 more)

### Community 402 - "EXP-0197 — Paired Fused Gate+Up SwiGLU with 64-Bit dwordx2 Coalesced Memory Access"
Cohesion: 0.13
Nodes (14): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0197 — Paired Fused Gate+Up SwiGLU with 64-Bit dwordx2 Coalesced Memory Access, Follow-up (+6 more)

### Community 403 - "Candidate 3: Recurrent Attention Gate Rollout"
Cohesion: 0.20
Nodes (10): A/B Benchmark Results (Interleaved 5 Pairs), Candidate 3: Recurrent Attention Gate Rollout, Correctness, Cumulative Gap to llama.cpp Status, Decision, Hardware State Validation, Implementation Scope, P63 GPU Profile Breakdown (+2 more)

### Community 404 - "EXP-0304 — M25-C mx register-pipeline transplant"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0304 — M25-C mx register-pipeline transplant, Follow-up (+4 more)

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

### Community 409 - "EXP-0367 — Exact partial-tail stage attribution and state oracle"
Cohesion: 0.18
Nodes (10): Aligned-tail timing and route evidence, Amdahl ceiling, Decision, Execution-mode and reconciliation status, EXP-0365/0366 evidence, EXP-0367 — Exact partial-tail stage attribution and state oracle, Next PRIMARY frontier, Oracle validation (+2 more)

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

### Community 428 - "V2-0002 — Prefill V2: Mx Compact Projection Backend & FAST_V1 Confrontation"
Cohesion: 0.12
Nodes (15): 1. Mandatory $N=512$ Baseline Comparison Table, 2. Multi-Length Scaling ($N=64, 128, 512$), 3. Execution Phase Attribution ($N=512$, Layer 0), 4. Three-Layer Chain Semantic & Contiguous Timing ($L0 \rightarrow L1 \rightarrow L2$), 5. Stateful Split-Call Equivalence Test ($512$ vs $256 + 256$), Baseline vs Candidate, Decision, Economic & Full-Model P512 Budget (+7 more)

### Community 429 - "EXP-0243 — M11-B SwiGLU/Q8 producer-consumer fusion rejection"
Cohesion: 0.20
Nodes (9): Baseline and candidate, Correctness, Decision, Environment, EXP-0243 — M11-B SwiGLU/Q8 producer-consumer fusion rejection, Follow-up, Hypothesis, Interpretation (+1 more)

### Community 430 - "RuntimeGenerateStats"
Cohesion: 0.06
Nodes (32): GenerationStopReason, RuntimeGenerateStats, cancelled, checkpoint_position, common_prefix_tokens, decode_d2h_bytes, decode_graph_launches, decode_h2d_bytes (+24 more)

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

### Community 447 - "Qwen3ForwardTrace"
Cohesion: 0.13
Nodes (20): reset, vector, Qwen3DecodeCache, caches_, length, reset, Qwen3ForwardTrace, embedding (+12 more)

### Community 448 - "Milestone M8 Primary Gate Qualification Report"
Cohesion: 0.33
Nodes (5): 1. Gate Scorecard, 2. Optimization Trajectory & Latency Reduction, 3. Experiment Ledger, 4. Telemetry Validation, Milestone M8 Primary Gate Qualification Report

### Community 449 - "EXP-0205 — Batched Q4_K Wave GEMV for Prefill"
Cohesion: 0.15
Nodes (12): B=4 Sweet Spot Analysis, B=8+ Regression, Benchmark, Decision, Environment, EXP-0205 — Batched Q4_K Wave GEMV for Prefill, Follow-up, Hypothesis (+4 more)

### Community 450 - "Milestone M10 — Real-World Inference Performance & Hybrid Context Scaling"
Cohesion: 0.14
Nodes (13): 1. Executive Summary, 2. Real-World Latency Curve Across Production Modes, 3. Hybrid Architecture Context Scaling Analysis (128 -> 65,536 tokens), 4. Key Architectural Findings & Bottlenecks, 5. M10 Implementation Roadmap, Finding 1: The DeltaNet $O(1)$ Recurrent State Advantage, Finding 2: The Attention Bandwidth Roofline at 64K Context, Finding 3: The Current Attention Kernel Bottleneck at Long Context (+5 more)

### Community 453 - "M31-0006 — Implementation and promotion decision"
Cohesion: 0.10
Nodes (17): Default schedule, M31-0006 — Production graph-decode baseline, Raw evidence, Real-model smoke, Evidence-based ranking, M31-0006 — Implementation and promotion decision, Next engineering step, Outcome (+9 more)

### Community 454 - "EXP-0253 — Repacked MMQ64 with exact Q8 side sums"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness, Decision, Environment, EXP-0253 — Repacked MMQ64 with exact Q8 side sums, Follow-up, Hypothesis (+2 more)

### Community 455 - "EXP-0210 — M11-B Amdahl Profile and Sequential-Floor Check"
Cohesion: 0.33
Nodes (5): Amdahl implication, Decision, EXP-0210 — M11-B Amdahl Profile and Sequential-Floor Check, Hypothesis, Measurement

### Community 456 - "EXP-0303 — M25-B mx compact repacked MMQ"
Cohesion: 0.14
Nodes (13): Baseline, Candidate, Correctness, Decision, Environment, EXP-0303 — M25-B mx compact repacked MMQ, Follow-up, Hypothesis (+5 more)

### Community 457 - "EXP-0213 — M11-B FP16 GEMM with On-Device K-Quant Dequantization"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Correctness, Decision, Environment, EXP-0213 — M11-B FP16 GEMM with On-Device K-Quant Dequantization, Follow-up, Hypothesis (+3 more)

### Community 458 - "EXP-0308 — M25-G mx weight-row prefetch"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Decision, Environment, EXP-0308 — M25-G mx weight-row prefetch, Follow-up, Hypothesis (+4 more)

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
Cohesion: 0.17
Nodes (33): apply_runtime_preset(), cmd_config(), cmd_doctor(), cmd_inspect(), cmd_models(), cmd_serve(), constant_time_equal(), ostream (+25 more)

### Community 468 - "Milestone M9 — Context Scaling Qualification (TG64 → TG1024)"
Cohesion: 0.33
Nodes (5): 1. Executive Summary, 2. Context Scaling Results Table, 3. Hardware Telemetry & Thermal Behavior, 4. Architectural Analysis, Milestone M9 — Context Scaling Qualification (TG64 → TG1024)

### Community 469 - "EXP-0227 — M11-B Q4_K token-parallel row tile"
Cohesion: 0.17
Nodes (11): Baseline, Candidate, Correctness, Decision, Environment, EXP-0227 — M11-B Q4_K token-parallel row tile, Follow-up, Hypothesis (+3 more)

### Community 470 - "EXP-0241 — M11-B Q4_K row-major 16-token tile rejection"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Correctness, Decision, Environment, EXP-0241 — M11-B Q4_K row-major 16-token tile rejection, Follow-up, Hypothesis (+1 more)

### Community 471 - "EXP-0338 — Mx attention decode weight reuse"
Cohesion: 0.15
Nodes (12): Baseline, Candidate, Correctness, Decision, Environment and benchmark, EXP-0338 — Mx attention decode weight reuse, Follow-up, Interpretation (+4 more)

### Community 472 - "EXP-0214 — M11-B Native Q4_K Split-K Batched GEMM"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment, EXP-0214 — M11-B Native Q4_K Split-K Batched GEMM, Follow-up, Hypothesis, Re-evaluation — release build (+1 more)

### Community 479 - "PrefillV2RecurrentLayer"
Cohesion: 0.08
Nodes (23): GgufTensorType, size_t, uint8_t, vector, PrefillV2RecurrentLayer, d_attn_norm_, d_ffn_down_mmq_, d_ffn_gate_mmq_ (+15 more)

### Community 480 - "e2_0003_checkpointed_call.cpp"
Cohesion: 0.14
Nodes (18): Context, name, seed, tokens, ostream, size_t, string, uint32_t (+10 more)

### Community 481 - "EXP-0212 — M11-B B=8 Accumulator GEMV Rejection"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0212 — M11-B B=8 Accumulator GEMV Rejection, Follow-up, Hypothesis, Results

### Community 482 - "EXP-0219 — M11-B Q4_K LDS Tile-Reuse Rejection"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0219 — M11-B Q4_K LDS Tile-Reuse Rejection, Follow-up, Hypothesis, Results

### Community 483 - "EXP-0244 — M11-B final qualification and measured ceiling"
Cohesion: 0.22
Nodes (8): Ceiling and blocker, Current production evidence, Decision, EXP-0244 — M11-B final qualification and measured ceiling, P512 Amdahl profile, Required next experiment, Result, Subsequent re-evaluation

### Community 485 - "EXP-0284 — M22 recurrent B4 tail-family profile"
Cohesion: 0.22
Nodes (8): Baseline and candidate, Decision, Environment, EXP-0284 — M22 recurrent B4 tail-family profile, Follow-up, Hypothesis, Interpretation, Results

### Community 486 - "EXP-0282 — M22 SSM-output dense batch candidate"
Cohesion: 0.29
Nodes (6): Baseline and candidate, Decision, EXP-0282 — M22 SSM-output dense batch candidate, Follow-up, Hypothesis, Results

### Community 487 - "EXP-0306 — M25-E sliced activation staging for mx MMQ"
Cohesion: 0.14
Nodes (13): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0306 — M25-E sliced activation staging for mx MMQ, Follow-up (+5 more)

### Community 488 - "qwen3_gpu_primitives.hpp"
Cohesion: 0.05
Nodes (42): int16_t, int8_t, uint8_t, M23Q8_1MmqBlock, d, qs, qsum_scaled, s (+34 more)

### Community 489 - "EXP-0256 — M12 chunkwise Gated DeltaNet oracle"
Cohesion: 0.20
Nodes (9): Baseline / oracle, Candidate, Correctness, Decision, Environment, EXP-0256 — M12 chunkwise Gated DeltaNet oracle, Follow-up, Hypothesis (+1 more)

### Community 490 - "MatrixScenarioResult"
Cohesion: 0.07
Nodes (31): size_t, string, uint32_t, vector, main(), make_synthetic_prompt(), MatrixScenarioResult, cold_decode_step_ms (+23 more)

### Community 491 - "EXP-0221 — M11-B Direct Consumption of Batched Prefill Workspace"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0221 — M11-B Direct Consumption of Batched Prefill Workspace, Follow-up, Hypothesis, Results

### Community 492 - "EXP-0208 — Full-Attention Projection Batching"
Cohesion: 0.20
Nodes (9): Baseline, Candidate, Correctness, Decision, Environment, EXP-0208 — Full-Attention Projection Batching, Follow-up, Hypothesis (+1 more)

### Community 493 - "size_t"
Cohesion: 0.05
Nodes (25): b64_composition_enabled(), DeviceBytes, bytes_, data_, exp0385_target_layer(), GpuLayerRef, attention, recurrent (+17 more)

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

### Community 509 - "EXP-0365 — Context scaling reconciliation"
Cohesion: 0.12
Nodes (16): Amdahl ceiling, Capacity-dependent source audit, Capacity-only matrix, Decision, Environment, Execution-contract fingerprint, EXP-0364 anomaly, EXP-0365 — Context scaling reconciliation (+8 more)

### Community 510 - "EXP-0248 — M11-B operator-major recurrent tail rejection"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0248 — M11-B operator-major recurrent tail rejection, Follow-up, Hypothesis, Results

### Community 511 - "EXP-0313 — M25-H/I review hardening and P512 stall triage"
Cohesion: 0.25
Nodes (7): Changes, Correctness, Decision, EXP-0313 — M25-H/I review hardening and P512 stall triage, Follow-up, P512 stall diagnostics, Question

### Community 512 - "EXP-0275 — M21 prefill gap and M12 continuation correctness"
Cohesion: 0.25
Nodes (7): Configurations, Context-conditioned TG64 reference, Correctness, Decision, EXP-0275 — M21 prefill gap and M12 continuation correctness, Question, Results

### Community 513 - "EXP-0344 — M25 oracle GDN launch-bounds contract"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0344 — M25 oracle GDN launch-bounds contract, Follow-up, Hypothesis, Measurements

### Community 514 - "EXP-0294 — M23 P512 projection / B64 causal-width profile"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0294 — M23 P512 projection / B64 causal-width profile, Follow-up, Hypothesis, Results

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

### Community 519 - "m6a9_qwen35_lm_head.cpp"
Cohesion: 0.35
Nodes (8): argmax(), path, size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 520 - "suffix_ttft_attribution_bench.cpp"
Cohesion: 0.18
Nodes (11): size_t, uint32_t, vector, LayerProfileData, core_ms, is_gqa, layer_idx, mmq_ms (+3 more)

### Community 521 - "EXP-0251 — M11-B native Q4_K token-reuse rejections"
Cohesion: 0.20
Nodes (9): Baseline, Candidates, Correctness, Decision, Environment, EXP-0251 — M11-B native Q4_K token-reuse rejections, Hypothesis, Interpretation (+1 more)

### Community 522 - "EXP-0264 — M16-B bounded serving queue qualification"
Cohesion: 0.33
Nodes (5): Candidate, Correctness, Decision, EXP-0264 — M16-B bounded serving queue qualification, Hypothesis

### Community 523 - "EXP-0289 — Resident recurrent FFN MMQ weights"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0289 — Resident recurrent FFN MMQ weights, Follow-up, Hypothesis, Results

### Community 524 - "observe"
Cohesion: 0.36
Nodes (9): capture_observer_graph(), hipGraphExec_t, hipStream_t, uint32_t, vector, main(), median(), observe() (+1 more)

### Community 525 - "EXP-0260 — M12 combined dense and chunkwise prefill"
Cohesion: 0.29
Nodes (6): Candidate, Correctness, Decision, EXP-0260 — M12 combined dense and chunkwise prefill, Hypothesis, Results

### Community 526 - "EXP-0287 — M23 shared dense projection oracle"
Cohesion: 0.25
Nodes (7): Candidate, Correctness, Decision, Environment, EXP-0287 — M23 shared dense projection oracle, Hypothesis, Results

### Community 527 - "EXP-0328 — M25-E split M23 attention Q/K geometry"
Cohesion: 0.17
Nodes (11): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0328 — M25-E split M23 attention Q/K geometry, Follow-up (+3 more)

### Community 528 - "EXP-0352 — M26-B historical decode-floor differential"
Cohesion: 0.06
Nodes (31): Audit execution — 2026-09-15, B1–B5 closure audit, B1 — historical contract reconstruction, B2 — current contract, B3 — required comparison matrix, B4 — differential attribution, B5 — fast-path audit, Corrected legacy comparison — 2026-09-16 (+23 more)

### Community 529 - "EXP-0293 — Async next-layer repacked-weight prefetch"
Cohesion: 0.29
Nodes (6): Candidate, Correctness and result, Decision, Environment, EXP-0293 — Async next-layer repacked-weight prefetch, Hypothesis

### Community 530 - "EXP-0242 — M11-B recurrent fused normalization/Q8 rejection"
Cohesion: 0.22
Nodes (8): Baseline and candidate, Correctness, Decision, Environment, EXP-0242 — M11-B recurrent fused normalization/Q8 rejection, Follow-up, Hypothesis, Results

### Community 531 - "EXP-0371 — Isolate L7 B128 projection contract"
Cohesion: 0.20
Nodes (9): Buffer/index audit status, Decision, Downstream captures, Exact routes, EXP-0370 evidence, EXP-0371 — Isolate L7 B128 projection contract, L7 input-hidden comparison, Next PRIMARY frontier (+1 more)

### Community 532 - "EXP-0276 — M22 P512 operator-family attribution"
Cohesion: 0.20
Nodes (9): Baseline, Benchmark, Correctness, Decision, Environment, EXP-0276 — M22 P512 operator-family attribution, Follow-up, Hypothesis (+1 more)

### Community 533 - "EXP-0277 — M22 mx-llama native repack A/B"
Cohesion: 0.20
Nodes (9): Baseline and candidate, Benchmark, Decision, EXP-0277 — M22 mx-llama native repack A/B, Follow-up, Hypothesis, Prompt processing, Results (+1 more)

### Community 534 - "EXP-0311 — M25-J Mx batched Q8 quantizer"
Cohesion: 0.14
Nodes (13): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0311 — M25-J Mx batched Q8 quantizer, Follow-up (+5 more)

### Community 535 - "EXP-0245 — M11-B Q4_K MMQ16 split-4 rejection"
Cohesion: 0.22
Nodes (8): Baseline and candidate, Decision, Environment, EXP-0245 — M11-B Q4_K MMQ16 split-4 rejection, Follow-up, Hypothesis, Interpretation, Results

### Community 536 - "EXP-0283 — M22 shared dense FFN-down source"
Cohesion: 0.33
Nodes (5): Candidate, Decision, EXP-0283 — M22 shared dense FFN-down source, Hypothesis, Result

### Community 537 - "EXP-0350 — Interactive serving and live continuation"
Cohesion: 0.10
Nodes (18): Interactive serving, Baseline, Cancellation/failure invalidation and tool-turn reuse — 2026-09-19, Candidate, Correctness and results, Current build exact checkpoint recheck — 2026-09-20, Current HEAD 4K append and real Pi tool loop — 2026-09-19, Current HEAD recheck — 2026-09-19 (+10 more)

### Community 538 - "v2_0017_hierarchical_gqa_bakeoff.cpp"
Cohesion: 0.14
Nodes (19): AccuracyResult, cosine, is_finite, max_abs_error, compute_accuracy(), __global__, size_t, T (+11 more)

### Community 539 - "EXP-0319 — M25-H/I configuration matrix"
Cohesion: 0.22
Nodes (8): Decision, Environment and benchmark, EXP-0319 — M25-H/I configuration matrix, Matrix, Question, Re-evaluation — post-repair runtime gate — 2026-09-12, Results — B128, Results — B512

### Community 540 - "EXP-0333 — Q4/Q5-only Mx MMQ occupancy annotation"
Cohesion: 0.17
Nodes (11): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0333 — Q4/Q5-only Mx MMQ occupancy annotation, Follow-up (+3 more)

### Community 541 - "D031 — Freeze M11-B and Isolate Matrix Prefill from Decode"
Cohesion: 0.50
Nodes (4): Consequences, D031 — Freeze M11-B and Isolate Matrix Prefill from Decode, Decision, Reason

### Community 542 - "D024 — Benchmarkability Is an Architectural Requirement"
Cohesion: 0.67
Nodes (3): D024 — Benchmarkability Is an Architectural Requirement, Decision, Reason

### Community 543 - "EXP-0263 — M12 GDN contract fix and dense-path qualification"
Cohesion: 0.25
Nodes (7): Candidate, Correctness, Decision, Environment, EXP-0263 — M12 GDN contract fix and dense-path qualification, Follow-up, Hypothesis

### Community 544 - "EXP-0268 — M18-A lifecycle, cancellation, and request telemetry"
Cohesion: 0.29
Nodes (6): Candidate, Decision, Environment, EXP-0268 — M18-A lifecycle, cancellation, and request telemetry, Hypothesis, Verification

### Community 545 - "EXP-0278 — M22 quantized batch-width study"
Cohesion: 0.22
Nodes (8): Baseline and candidate, Decision, Environment and benchmark, EXP-0278 — M22 quantized batch-width study, Follow-up, Hypothesis, Interpretation, Results

### Community 546 - "EXP-0258 — M12 runtime composition and 128-token schedule"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0258 — M12 runtime composition and 128-token schedule, Hypothesis, Interpretation, Results

### Community 547 - "EXP-0296 — M24 resident MMQ versus GPU dequantized FP16 GEMM"
Cohesion: 0.15
Nodes (12): Candidate (initial bake-off), Correctness, Decision, Environment, EXP-0296 — M24 resident MMQ versus GPU dequantized FP16 GEMM, Follow-up, Hypothesis, Interpretation (+4 more)

### Community 548 - "EXP-0290 — Persistent-token affine MMQ"
Cohesion: 0.29
Nodes (6): Candidate, Correctness, Decision, EXP-0290 — Persistent-token affine MMQ, Hypothesis, Result

### Community 549 - "EXP-0265 — M16-D OpenAI request contract qualification"
Cohesion: 0.33
Nodes (5): Candidate, Correctness, Decision, EXP-0265 — M16-D OpenAI request contract qualification, Hypothesis

### Community 550 - "EXP-0279 — M22 dense FFN-down candidate"
Cohesion: 0.29
Nodes (6): Configuration, Correctness, Decision, EXP-0279 — M22 dense FFN-down candidate, Hypothesis, Result

### Community 552 - "EXP-0323 — M25 P512 exact pre-J/source-delta A/B"
Cohesion: 0.22
Nodes (8): Correctness, Decision, Environment, EXP-0323 — M25 P512 exact pre-J/source-delta A/B, Follow-up, Interpretation, Question, Results

### Community 553 - "EXP-0288 — M23 direct P512 GDN scan"
Cohesion: 0.33
Nodes (5): Candidate and correctness, Decision, EXP-0288 — M23 direct P512 GDN scan, Hypothesis, Result

### Community 555 - "EXP-0285 — M22 prefill parity closure"
Cohesion: 0.20
Nodes (9): Candidates, Correctness, capacity, and serving, Decision, EXP-0285 — M22 prefill parity closure, Final best MIInfer PP ladder, Fixed comparison, Original P512 attribution and Amdahl evidence, Reference repack A/B (+1 more)

### Community 556 - "EXP-0355 — M27 attention decode-reuse P512 A/B and context smoke"
Cohesion: 0.22
Nodes (8): Benchmark, Correctness, EXP-0355 — M27 attention decode-reuse P512 A/B and context smoke, Follow-up, Hypothesis, Interpretation and decision, Long-context sanity, Results

### Community 558 - "EXP-0269 — M18-B pinned llama.cpp-gfx906 qualification"
Cohesion: 0.29
Nodes (6): Baseline, Decision, EXP-0269 — M18-B pinned llama.cpp-gfx906 qualification, Hypothesis, Model and hardware, Verification

### Community 559 - "m13_quant_mm_bench.cpp"
Cohesion: 0.13
Nodes (13): as(), Fn, hipEvent_t, string, T, vector, Event, value (+5 more)

### Community 560 - "EXP-0356 — M27 P512 semantic contract attribution and stop"
Cohesion: 0.25
Nodes (7): Candidate screen, Contract results, Environment and command shape, EXP-0356 — M27 P512 semantic contract attribution and stop, Interpretation and decision, Outer contract boundaries, Question and baseline

### Community 561 - "EXP-0374 — Qualify repeated-B128 remainder scheduler"
Cohesion: 0.22
Nodes (8): Decision, EXP-0374 — Qualify repeated-B128 remainder scheduler, Next PRIMARY, Question, Scheduler, Semantic qualification, Structural results, Timing (pre-correction; repeated-B128 route not qualified)

### Community 562 - "EXP-0320 — M25 P512 continuation crash and resident FFN repair"
Cohesion: 0.25
Nodes (7): Decision, EXP-0320 — M25 P512 continuation crash and resident FFN repair, Failure, Question, Repair, Root cause, Verification

### Community 563 - "EXP-0267 — M17 Local Appliance Qualification"
Cohesion: 0.33
Nodes (5): Candidate, Decision, EXP-0267 — M17 Local Appliance Qualification, Hypothesis, Qualification

### Community 564 - "EXP-0270 — M19 dynamic context capacity qualification"
Cohesion: 0.25
Nodes (7): Contract, Decision, EXP-0270 — M19 dynamic context capacity qualification, Hypothesis, Long-context inference evidence, Physical verification, Qualification re-evaluation

### Community 565 - "EXP-0266 — M16-C serving concurrency qualification"
Cohesion: 0.25
Nodes (7): Decision, Environment and workload, EXP-0266 — M16-C serving concurrency qualification, Harness repair, Question, Results, Stream A/B

### Community 566 - "EXP-0280 — M22 M13 B8 eight-wave mapping"
Cohesion: 0.33
Nodes (5): Configuration, Decision, EXP-0280 — M22 M13 B8 eight-wave mapping, Hypothesis, Result

### Community 567 - "EXP-0271 — MIInfer runtime-only prefill/decode qualification"
Cohesion: 0.33
Nodes (5): Benchmark, Decision, Environment, EXP-0271 — MIInfer runtime-only prefill/decode qualification, Results

### Community 568 - "EXP-0309 — M25-H Mx attention FFN projections"
Cohesion: 0.17
Nodes (11): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0309 — M25-H Mx attention FFN projections, Follow-up (+3 more)

### Community 569 - "EXP-0272 — M20-B API authentication and serving hardening"
Cohesion: 0.40
Nodes (4): Contract, Decision, EXP-0272 — M20-B API authentication and serving hardening, Verification

### Community 570 - "EXP-0370 — Qualify L7 partial-tail K/V and causal contract"
Cohesion: 0.20
Nodes (9): Actual causal-read bounds, Attention output and continuation, Decision, EXP-0368/0369 evidence, EXP-0370 — Qualify L7 partial-tail K/V and causal contract, K/V pre-write captures, Next PRIMARY frontier, Question (+1 more)

### Community 572 - "EXP-0273 — M20-A constrained Hermes qualification"
Cohesion: 0.40
Nodes (4): Decision, EXP-0273 — M20-A constrained Hermes qualification, Results, Setup

### Community 573 - "EXP-0301 — M25-D mx-style GDN state tiling"
Cohesion: 0.22
Nodes (8): Baseline and candidate, Decision, Environment, EXP-0301 — M25-D mx-style GDN state tiling, Follow-up, Hypothesis, Interpretation, Results

### Community 574 - "EXP-0383 — Localize the First Composed-B64 Semantic Divergence"
Cohesion: 0.13
Nodes (14): Decision, Determinism check, Exact next PRIMARY, EXP-0382 failure basis, EXP-0383 — Localize the First Composed-B64 Semantic Divergence, Family classification, Final hidden, Final norm (+6 more)

### Community 575 - "EXP-0281 — M22 dense FFN prefill candidate"
Cohesion: 0.18
Nodes (10): Baseline, Benchmark, Correctness, Decision, Environment, EXP-0281 — M22 dense FFN prefill candidate, Follow-up, Hypothesis (+2 more)

### Community 576 - "EXP-0368 — L3 partial-attention K/V and causal contract"
Cohesion: 0.15
Nodes (12): Decision, EXP-0367 evidence, EXP-0368 — L3 partial-attention K/V and causal contract, Later-base qualification attempt, Next PRIMARY frontier, P640 controlled result, Per-stage and K/V localization, Position, causal, and prefix audit (+4 more)

### Community 577 - "UpdateProvenance"
Cohesion: 0.12
Nodes (15): UpdateProvenance, beta, candidate, column, decay, decayed, delta, head (+7 more)

### Community 578 - "run_layer"
Cohesion: 0.19
Nodes (17): array, path, size_t, string, string_view, uint32_t, fixture_checkpoint(), main() (+9 more)

### Community 579 - "Re-evaluation — full layer-major scheduling — 2026-09-10"
Cohesion: 0.04
Nodes (44): B128 convolution contract, Baseline, Candidate — B128 attention Q/K/V projection — 2026-09-10, Candidate — batched causal attention launch — 2026-09-10, Candidate — cached Q8_1 affine sum — 2026-09-10, Candidate — gfx906 affine MMQ launch bounds — 2026-09-10, Candidate — gfx906 Q6 MMQ launch bounds — 2026-09-10, Candidate — MMQ activation-scale hoist — 2026-09-10 (+36 more)

### Community 580 - "EXP-V2-0022-suffix-mmq-critical-path.md"
Cohesion: 0.12
Nodes (16): 1. Executive Summary & Core Engineering Conclusions, 2. Phase 1: GPU Critical-Path Operator Attribution Table, 3. Phase 2: Roofline & Optimization Ceiling Analysis, 4. Phase 3 & 4: Candidate Evaluations & Fusion Audits, 5. Phase 5: Production Interleaved A/B Benchmarks ($P=65,536, S=512$), 6. Milestone Qualification Gates, 7. Decision & Next Milestone: V2-0023, Candidate A — Isolated M=512 MMQ Specialization Bakeoff (+8 more)

### Community 581 - "EXP-0324 — M25 Mx vectorized MMQ epilogue"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment and benchmark, EXP-0324 — M25 Mx vectorized MMQ epilogue, Follow-up, Hypothesis, Results

### Community 582 - "EXP-0298 — M24 direct resident-FP16 recurrent layer"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0298 — M24 direct resident-FP16 recurrent layer, Follow-up, Hypothesis, Results

### Community 583 - "EXP-0310 — M25-I Mx attention O projection"
Cohesion: 0.17
Nodes (11): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0310 — M25-I Mx attention O projection, Follow-up (+3 more)

### Community 584 - "EXP-0325 — M25 beta/alpha batched FP32 hipBLAS port"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness, Decision, Environment, EXP-0325 — M25 beta/alpha batched FP32 hipBLAS port, Follow-up, Hypothesis (+2 more)

### Community 585 - "EXP-0295 — M23 resident-all projection path"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0295 — M23 resident-all projection path, Follow-up, Hypothesis, Results

### Community 586 - "EXP-0291 — Fused wide FFN Gate/Up MMQ"
Cohesion: 0.33
Nodes (5): Candidate, Correctness, Decision, EXP-0291 — Fused wide FFN Gate/Up MMQ, Hypothesis

### Community 587 - "EXP-0340 — M25-L recurrent FFN contract differential"
Cohesion: 0.17
Nodes (11): Current MIInfer recurrent-layer measurement, Current oracle kernel trace, Decision, Environment, EXP-0340 — M25-L recurrent FFN contract differential, Follow-up, Interpretation, Primary benchmark (+3 more)

### Community 588 - "EXP-0312 — M25-K Mx attention Q/K projection"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0312 — M25-K Mx attention Q/K projection, Follow-up (+4 more)

### Community 589 - "EXP-0299 — M24 full-attention attribution"
Cohesion: 0.17
Nodes (11): Baseline and environment, Correctness, D0 — full-model reproduction, D1/D2 — layer-3 bakeoff, D3 — isolated resident-FP16 projection A/B, D3 re-evaluation — canonical fixture gate, Decision, EXP-0299 — M24 full-attention attribution (+3 more)

### Community 590 - "EXP-0318 — M25 attention bakeoff pipeline isolation"
Cohesion: 0.33
Nodes (5): Decision, EXP-0318 — M25 attention bakeoff pipeline isolation, Finding, Fix, Verification

### Community 591 - "EXP-0376 — Attribute the Repeated-B128 Runtime Collapse"
Cohesion: 0.15
Nodes (12): Base-position and invocation attribution, Clean current-HEAD timing, Critical per-chunk table, Decision, Environment, Exact next PRIMARY, Execution-mode counters, EXP-0375 basis (+4 more)

### Community 592 - "EXP-0292 — Resident attention FFN MMQ weights"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0292 — Resident attention FFN MMQ weights, Follow-up, Hypothesis, Results

### Community 593 - "HttpRequest"
Cohesion: 0.15
Nodes (13): time_point, uint64_t, HttpReadResult, reason, request, status, HttpRequest, client_fd (+5 more)

### Community 594 - "v2_0017_quad_head_subwave_bakeoff.cpp"
Cohesion: 0.14
Nodes (18): AccuracyResult, cosine, is_finite, max_abs_error, compute_accuracy(), __global__, size_t, T (+10 more)

### Community 595 - "1. ggml-org/llama.cpp"
Cohesion: 0.33
Nodes (6): 1. ggml-org/llama.cpp, Do not inherit automatically, Important lessons, MIInfer question, Potentially reusable knowledge, Role

### Community 596 - "EXP-0307 — M25-F mx affine DP4A ordering"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0307 — M25-F mx affine DP4A ordering, Follow-up (+4 more)

### Community 597 - "EXP-0354 — Current HEAD P512 control requalification"
Cohesion: 0.29
Nodes (6): Environment and workload, EXP-0354 — Current HEAD P512 control requalification, Hardware audit, Interpretation, Question, Results

### Community 598 - "EXP-0302 — M25-E wide SwiGLU to MMQ-Q8 producer"
Cohesion: 0.22
Nodes (8): Baseline and candidate, Correctness, Decision, Environment, EXP-0302 — M25-E wide SwiGLU to MMQ-Q8 producer, Follow-up, Hypothesis, Measurements

### Community 599 - "EXP-0327 — M25-D GDN Tc4 interleaved qualification retest"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0327 — M25-D GDN Tc4 interleaved qualification retest, Follow-up (+4 more)

### Community 600 - "EXP-0329 — M25-E complete Mx attention Q/K contract screen"
Cohesion: 0.17
Nodes (11): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0329 — M25-E complete Mx attention Q/K contract screen, Follow-up (+3 more)

### Community 601 - "EXP-0315 — M25 MMQ occupancy annotation"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment and benchmark, EXP-0315 — M25 MMQ occupancy annotation, Hypothesis, Question, Results

### Community 602 - "EXP-0316 — M25-J quantizer under qualified MMQ"
Cohesion: 0.22
Nodes (8): Benchmark, Candidate and provenance, Correctness, Decision, EXP-0316 — M25-J quantizer under qualified MMQ, Hypothesis, Question, Results

### Community 603 - "Qwen3LayerWeights"
Cohesion: 0.17
Nodes (12): Qwen3LayerWeights, attention_norm, down, ffn_norm, gate, k, k_norm, output (+4 more)

### Community 604 - "EXP-0317 — M25 staged MMQ loop lowering"
Cohesion: 0.25
Nodes (7): Candidates, Correctness, Decision, Environment and benchmark, EXP-0317 — M25 staged MMQ loop lowering, Question, Results

### Community 605 - "EXP-0330 — Mx recurrent FFN Gate/Up pair kernel"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0330 — Mx recurrent FFN Gate/Up pair kernel, Follow-up (+4 more)

### Community 606 - "EXP-0366 — Partial-tail prefill execution contract"
Cohesion: 0.17
Nodes (11): Candidate contract, Decision, Existing primitive support matrix, EXP-0365 evidence and baseline, EXP-0365 slow-path execution map for a 510-token tail, EXP-0366 — Partial-tail prefill execution contract, Next PRIMARY frontier, Oracle and correctness/state matrix (+3 more)

### Community 607 - "validate_position"
Cohesion: 0.29
Nodes (11): cached_attention(), Metrics, path, size_t, span, string_view, vector, main() (+3 more)

### Community 608 - "m6a13_qwen35_full_attention_layer.cpp"
Cohesion: 0.34
Nodes (11): check(), checkpoint(), copy_to_host(), path, size_t, string_view, vector, DeviceBuffer (+3 more)

### Community 609 - "EXP-0343 — M25 faithful GDN state input/output contract"
Cohesion: 0.22
Nodes (8): Candidate, Correctness, Decision, Environment, EXP-0343 — M25 faithful GDN state input/output contract, Follow-up, Hypothesis, Measurements

### Community 610 - "EXP-0314 — M25 P512 source-delta retest after stall recovery"
Cohesion: 0.18
Nodes (10): Correctness, Decision, Diagnostic note, Environment, EXP-0314 — M25 P512 source-delta retest after stall recovery, Follow-up, Interpretation, Question (+2 more)

### Community 611 - "EXP-0322 — M25-H/I post-repair qualification"
Cohesion: 0.18
Nodes (10): Correctness, Decision, Environment, EXP-0322 — M25-H/I post-repair qualification, Follow-up, Hardware and memory, Hypothesis, Preset verification (+2 more)

### Community 612 - "15. Research Classification"
Cohesion: 0.40
Nodes (5): 15. Research Classification, Category A — Architecture lessons, Category B — Kernel hypotheses, Category C — Failure lessons, Category D — Runtime features

### Community 613 - "EXP-0321 — M25 pinned mx-llama contract audit"
Cohesion: 0.22
Nodes (8): Decision, Existing primitive evidence, EXP-0321 — M25 pinned mx-llama contract audit, Follow-up, Hypothesis, Interpretation, Motivation, Source comparison

### Community 614 - "EXP-0332 — Mx MMQ full-tile specialization"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0332 — Mx MMQ full-tile specialization, Follow-up (+4 more)

### Community 615 - "EXP-0349 — M25 pinned Q6 FFN-down probe"
Cohesion: 0.20
Nodes (9): Candidate, Decision, End-to-end result, Environment, EXP-0349 — M25 pinned Q6 FFN-down probe, Follow-up, Hypothesis, Interpretation (+1 more)

### Community 616 - "Q5K"
Cohesion: 0.09
Nodes (24): int16_t, int8_t, uint16_t, uint8_t, Q4K, d, dmin, qs (+16 more)

### Community 617 - "EXP-0326 — M25-D GDN four-column wave shard"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0326 — M25-D GDN four-column wave shard, Follow-up (+4 more)

### Community 618 - "compare-m26cq-logits.py"
Cohesion: 0.70
Nodes (4): error_metrics(), main(), read(), top()

### Community 619 - "EXP-0345 — M25 parallel recurrent input branches"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment, EXP-0345 — M25 parallel recurrent input branches, Follow-up, Hypothesis, Interpretation (+1 more)

### Community 620 - "BenchmarkResult"
Cohesion: 0.14
Nodes (17): BenchmarkResult, decode_step_ms, decode_tok_s, gen_tokens, generated_tokens, numerical_ok, observed_free_gib, prefill_tok_s (+9 more)

### Community 621 - "EXP-0334 — M25 complete pinned Mx MMQ contract rejection"
Cohesion: 0.13
Nodes (14): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0334 — M25 complete pinned Mx MMQ contract rejection, Follow-up (+6 more)

### Community 622 - "EXP-0347 — M25 persistent oracle GDN state layout"
Cohesion: 0.18
Nodes (10): Candidate, Correctness, Decision, Environment, EXP-0347 — M25 persistent oracle GDN state layout, Follow-up, Hypothesis, Interpretation (+2 more)

### Community 623 - "EXP-0353 — M27 reusable device-state decode graph"
Cohesion: 0.15
Nodes (12): 128-token graph/direct continuation, Attention split-transition correctness, Correctness smoke, Decision, Dynamic-position screen, EXP-0353 — M27 reusable device-state decode graph, Hypothesis, Implementation (+4 more)

### Community 624 - "EXP-0363 — Spill-free attention state schedule"
Cohesion: 0.29
Nodes (7): Candidate A compile gate, Decision, EXP-0363 — Spill-free attention state schedule, Hypothesis, Required correction, Runtime status, Schedule correction attempt

### Community 625 - "EXP-0362 — Source-shaped query-tiled GQA2 prefill"
Cohesion: 0.20
Nodes (10): Baseline, Candidate, Correctness, Decision, Environment, EXP-0362 — Source-shaped query-tiled GQA2 prefill, Follow-up, Hypothesis (+2 more)

### Community 626 - "EXP-0348 — M25 current oracle requalification"
Cohesion: 0.22
Nodes (8): Correctness and resource checks, Decision, Environment, EXP-0348 — M25 current oracle requalification, Follow-up, Interpretation, Question, Results

### Community 627 - "m6a10_qwen35_q4k_projection.cpp"
Cohesion: 0.33
Nodes (7): path, size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 629 - "vector"
Cohesion: 0.06
Nodes (23): vector, TimingResult, mean_us, min_us, size_t, uint32_t, vector, make_synthetic_prompt() (+15 more)

### Community 630 - "EXP-0335 — M25 pinned Mx single-token MMV"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness, Decision, Environment, EXP-0335 — M25 pinned Mx single-token MMV, Follow-up, Hypothesis (+2 more)

### Community 631 - "EXP-0342 — M25 QKV contract retest"
Cohesion: 0.20
Nodes (9): Candidate, Decision, End-to-end screen, Environment, EXP-0342 — M25 QKV contract retest, Follow-up, Hypothesis, Isolated result (+1 more)

### Community 632 - "EXP-0337 — M25 P512 prefill HIP graph screen"
Cohesion: 0.25
Nodes (7): Candidate, Correctness, Decision, Environment, EXP-0337 — M25 P512 prefill HIP graph screen, Hypothesis, Results

### Community 633 - "EXP-0336 — M25 pinned Mx MMQ recurrent-FFN isolation rejection"
Cohesion: 0.18
Nodes (10): Baseline, Candidate, Correctness, Decision, Environment, EXP-0336 — M25 pinned Mx MMQ recurrent-FFN isolation rejection, Follow-up, Hypothesis (+2 more)

### Community 634 - "Candidate work"
Cohesion: 0.20
Nodes (10): Activation reuse, Candidate work, Exit criteria, Goal, HIP graph capture, Kernel specialization, M6 — Runtime Specialization, Native weight packing (+2 more)

### Community 635 - "EXP-0339 — Mx attention decode reuse P512 qualification screen"
Cohesion: 0.22
Nodes (8): Correctness, Decision, Environment, EXP-0339 — Mx attention decode reuse P512 qualification screen, Follow-up, Interpretation, Question, Results

### Community 636 - "run_ladder"
Cohesion: 0.16
Nodes (18): RuntimeState, path, uint32_t, vector, main(), run_combined(), argmax(), path (+10 more)

### Community 637 - "EXP-0341 — M25 attention fork/join concurrency"
Cohesion: 0.20
Nodes (9): Candidate, Correctness, Decision, Environment, EXP-0341 — M25 attention fork/join concurrency, Follow-up, Hypothesis, Results (+1 more)

### Community 638 - "EXP-0346 — M25 parallel recurrent QKV/Gate composition"
Cohesion: 0.25
Nodes (7): Candidate, Decision, Environment, EXP-0346 — M25 parallel recurrent QKV/Gate composition, Follow-up, Hypothesis, Trial and failure

### Community 639 - "v2_0017_pipelined_stream_bakeoff.cpp"
Cohesion: 0.14
Nodes (17): AccuracyResult, cosine, is_finite, max_abs_error, compute_accuracy(), __global__, size_t, T (+9 more)

### Community 640 - "EXP-0385 — Validate the Exact B64 L0 Recurrent-Wide Contract"
Cohesion: 0.13
Nodes (14): B128 headline, B64/B128 comparison, B64 headline, Classification, Determinism and route, Exact candidate-resource configuration, Exact next PRIMARY, EXP-0384 limitation (+6 more)

### Community 641 - "EXP-V2-0020 — FP16 Suffix Attention Load/Compute Pipeline Qualification"
Cohesion: 0.14
Nodes (13): 10. Follow-up & Next Directions, 1. Hypothesis, 2. Motivation & Background, 3. Microarchitectural ISA Audit (gfx906), 4. Logical vs Physical Bandwidth Attribution, 5. Candidate Implementations, 6. Experimental Benchmark Results, 7. Numerical Parity Verification (+5 more)

### Community 642 - "6. nlzy/vllm-gfx906"
Cohesion: 0.40
Nodes (5): 6. nlzy/vllm-gfx906, Key historical observations, MIInfer implication, Role, Status

### Community 643 - "BlockEvaluationResult"
Cohesion: 0.10
Nodes (20): BlockEvaluationResult, k3_metrics, output_metrics, s0_metrics, s1_metrics, s2_metrics, speedup_vs_fast_v1, speedup_vs_token_oracle (+12 more)

### Community 644 - "ReusableContext"
Cohesion: 0.06
Nodes (36): test_failure_cases(), compute_token_sequence_hash(), extend_token_sequence_hash(), GdnCheckpointStorage, capture, d_checkpoint_conv_history_, d_checkpoint_states_, restore (+28 more)

### Community 645 - "compare-m26c-state.py"
Cohesion: 0.87
Nodes (5): compare_bytes(), main(), read_exact(), scalar(), string()

### Community 646 - "memory_stream_bench.cpp"
Cohesion: 0.17
Nodes (16): size_t, string, vector, escape(), main(), median(), Options, bytes (+8 more)

### Community 647 - "BenchmarkResult"
Cohesion: 0.13
Nodes (18): BenchmarkResult, estimated_host_overhead_ms, first_token, max_ttft_ms, median_restore_ms, median_suffix_ms, median_ttft_ms, min_ttft_ms (+10 more)

### Community 648 - "hipStreamSynchronize"
Cohesion: 0.08
Nodes (30): M26CDecodeResult, SessionCheckpoint, SessionPrefixNode, StepResult, hipStreamSynchronize(), cmd_run_debug(), byte, function (+22 more)

### Community 649 - "EXP-0361 — Wave64 attention architecture analysis"
Cohesion: 0.20
Nodes (10): Architectural difference, Decision, Evidence status, EXP-0361 — Wave64 attention architecture analysis, MIInfer compiler resource snapshot, MIInfer control, mx-llama.cpp path, mx runtime confirmation (+2 more)

### Community 650 - "AttentionLayerKvCacheStorage"
Cohesion: 0.07
Nodes (25): AttentionLayerKvCacheStorage, d_key_cache_, d_key_cache_q8_, d_key_scales_, d_value_cache_, d_value_cache_q8_, d_value_scales_, download_key (+17 more)

### Community 652 - "EXP-V2-0008 — Dedicated Single-Token Decode Execution Path and Reusable HIP Graph Replay"
Cohesion: 0.13
Nodes (14): 1. Specialized Direct Decode Methods (`decode()`), 2. Reusable HIP Graph Capture & Zero-Host Replay (`capture_decode_graph`), Architecture & Implementation, Benchmark Results Comparison, Context Scaling Comparison (V2-0007 vs V2-0008), Decision, End-to-End Performance Matrix, EXP-V2-0008 — Dedicated Single-Token Decode Execution Path and Reusable HIP Graph Replay (+6 more)

### Community 653 - "e2_0003_same_source_ab.py"
Cohesion: 0.11
Nodes (25): Event, main(), call(), capture_state(), container_command(), interval(), main(), make_prompt() (+17 more)

### Community 654 - "EXP-0357 — M27 sparse exact prefix cache"
Cohesion: 0.22
Nodes (8): Baseline, Candidate, Correctness, Decision, Environment, EXP-0357 — M27 sparse exact prefix cache, Hypothesis, Real Pi results

### Community 655 - "half"
Cohesion: 0.36
Nodes (18): __global__, uint32_t, d_wave_max(), d_wave_sum(), qwen35_splitk_suffix_attn_stage1_1w_kernel(), qwen35_splitk_suffix_attn_stage1_bqtiled_kernel(), qwen35_splitk_suffix_attn_stage1_fast_kernel(), qwen35_splitk_suffix_attn_stage1_gqa6_lds_kernel() (+10 more)

### Community 656 - "EXP-0372 — Isolate L6 → L7 inter-layer B128 tail contract"
Cohesion: 0.25
Nodes (7): Decision, EXP-0372 — Isolate L6 → L7 inter-layer B128 tail contract, Method, Next PRIMARY, Question, Result, State and history

### Community 658 - "17. Highest-Priority Research Ideas"
Cohesion: 0.40
Nodes (5): 17. Highest-Priority Research Ideas, Later, P0, P1, P2

### Community 659 - "EXP-0381 — Isolate the Historical B64 Stop Above the Proven Prefill Boundary"
Cohesion: 0.12
Nodes (15): D-state evidence, Decision, Exact next PRIMARY, EXP-0379 historical stop, EXP-0380 proven boundary, EXP-0381 — Isolate the Historical B64 Stop Above the Proven Prefill Boundary, First-token ladder, Initialization contamination (+7 more)

### Community 660 - "11. joe2gaan/localaiservers"
Cohesion: 0.50
Nodes (4): 11. joe2gaan/localaiservers, Important dot-product lesson, MIInfer implication, Role

### Community 661 - "compare-m26c-logits.py"
Cohesion: 0.83
Nodes (3): main(), metrics(), read_logits()

### Community 663 - "AccuracyMetrics"
Cohesion: 0.22
Nodes (9): AccuracyMetrics, cosine, is_finite, max_abs_error, mean_abs_error, rel_rms_error, rms_error, compute_accuracy() (+1 more)

### Community 664 - "EXP-0360 — GQA-shared tiled full-attention prefill candidate"
Cohesion: 0.22
Nodes (9): Baseline, Build, Candidate, Correctness, Decision, EXP-0360 — GQA-shared tiled full-attention prefill candidate, Forensic P4K profile, Hypothesis (+1 more)

### Community 665 - "Performance Research Strategy"
Cohesion: 0.13
Nodes (15): 1. Mental model, 2. Headroom before implementation, 3. Default authorization thresholds, 4. Decode ceiling study, 5. Freeze rule, 6. Compiler/runtime direction, 7. Success metric, 8. Research principle (+7 more)

### Community 666 - "EXP-0378 — Determine Whether a Non-Scalar B64 Core Contract Exists"
Cohesion: 0.17
Nodes (11): Attention audit details, B64 capability matrix, Decision, Exact next PRIMARY, EXP-0377 rejection basis, EXP-0378 — Determine Whether a Non-Scalar B64 Core Contract Exists, Focused probe results, Method and gates (+3 more)

### Community 667 - "EXP-0391 — P512 recurrent/attention variance split"
Cohesion: 0.29
Nodes (6): Decision, EXP-0390 basis, EXP-0391 — P512 recurrent/attention variance split, Instrumentation contract and perturbation gate, Provenance, Question

### Community 668 - "EXP-0369 — Explain the P1664 position-768 divergence"
Cohesion: 0.20
Nodes (9): Corrected state comparison, Decision, EXP-0369 — Explain the P1664 position-768 divergence, L7 intermediate-stage status, Next PRIMARY frontier, Position and causal audit, Question, Route trace (+1 more)

### Community 669 - "EXP-0386 — Determine Whether Normal B64 Drift Causes L3 Semantic Amplification"
Cohesion: 0.15
Nodes (12): B128 L2, B64 L2, Decision, Exact next PRIMARY, EXP-0384/0385 basis, EXP-0386 — Determine Whether Normal B64 Drift Causes L3 Semantic Amplification, First-group scalar route, Interpretation (+4 more)

### Community 670 - "EXP-0388 — B128 shape-collapse attribution"
Cohesion: 0.18
Nodes (10): B512 versus B128 family attribution, Baseline integrity gate, Decision, EXP-0388 — B128 shape-collapse attribution, Initial regression observations, mx sanity, P512/P640/P896 clean matrix, Provenance (+2 more)

### Community 671 - "EXP-V2-0023-cross-query-attention.md"
Cohesion: 0.11
Nodes (18): 1. Executive Summary & Core Engineering Conclusions, 2. Phase 0: MMQ Accounting Reconciliation, 3. Phase 1: Baseline Traffic Accounting ($Q_{\text{tile}} = 1$), 4. Phase 2 & 3: Cross-Query Kernel Implementation & Empirical Bakeoff, 5. Microarchitectural Root Cause Analysis, 6. Milestone Decision & Definitive Verdict, 7. Next Architectural Frontiers, Baseline Measurement ($Q_{\text{tile}} = 1$ Control): (+10 more)

### Community 672 - "m6a8_qwen35_gpu_foundation.cpp"
Cohesion: 0.33
Nodes (7): path, size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 673 - "EXP-0373 — Refresh full-model partial-tail attribution"
Cohesion: 0.20
Nodes (9): Clean prefill timing, Current evidence and routes, Decision, EXP-0373 — Refresh full-model partial-tail attribution, Full-model semantic comparison, Hardware, Next PRIMARY, Question (+1 more)

### Community 674 - "Decision"
Cohesion: 0.18
Nodes (11): Canonical snapshot field classification, Combined recurrent selector control, Combined-selector TG128 correctness result, Current-binary principal-route TG128 gate, Current-build graph/direct boundary checks, Decision, Final re-evaluation — M26-CQ semantic-equivalence investigation, Layer-stage capture — first hidden-state divergence (+3 more)

### Community 675 - "m6a11_qwen35_attention_prefix.cpp"
Cohesion: 0.33
Nodes (7): path, size_t, vector, DeviceBuffer, main(), max_abs_error(), read_f32()

### Community 676 - "EXP-0384 — Locate the First Material B64 Recurrent-Wide Layer Divergence"
Cohesion: 0.17
Nodes (11): C/R route proofs, Decision, Determinism, Exact next PRIMARY, EXP-0383 basis, EXP-0384 — Locate the First Material B64 Recurrent-Wide Layer Divergence, First difference versus first material amplification, Full C-vs-R layer table (+3 more)

### Community 677 - "EXP-V2-0024-quantized-prefix-kv.md"
Cohesion: 0.10
Nodes (20): 1. Executive Summary & Core Engineering Conclusions, 2. Phase 0: Physical Traffic & Bandwidth Reconciliation, 3. Phase 1 & 2: Q8 Representation & Hybrid Architecture, 4. Phase 3, 4, 5: Bakeoff, Resource Qualification & ISA Root Cause, 5. Phase 6: Numerical Qualification, 6. Phase 8: Q4 Feasibility & Context Capacity Evaluation, 7. Milestone Qualification Gates & Definitive Decision, 8. Updated Roadmap (+12 more)

### Community 678 - "qwen3_generate.cpp"
Cohesion: 0.40
Nodes (10): argmax(), size_t, string, uint32_t, vector, main(), parse_count(), parse_id() (+2 more)

### Community 679 - "PhysicalPageView"
Cohesion: 0.13
Nodes (18): check_range, LogicalPageId, PhysicalPageView, bytes, data, ResolvedPageView, logical_id, logical_range (+10 more)

### Community 680 - "V2-0003 — Prefill V2 Slice 2: 4-Layer Repeating Topology Block (3 × GDN + 1 × GQA Attention)"
Cohesion: 0.14
Nodes (13): 1. Mandatory $N=512$ Topology Block Baseline Comparison, 2. Multi-Length Scaling ($N=64, 128, 512$), 3. V2 Block 0 Layer Breakdown ($N=512$, Total = 180.88 ms), 4. Stateful Split-Call Invariant Test ($512$ vs $256 + 256$), 5. Full 64-Layer Model Latency Projection, Baseline vs Candidate, Decision, Empirical Results (+5 more)

### Community 681 - "EXP-V2-0026-production-pipeline-hardening.md"
Cohesion: 0.25
Nodes (7): 1. Executive Summary & Qualification Overview, 2. End-to-End Context Scaling Ladder (Suffix $S=512$), 3. Multi-Turn Conversation Serving Benchmark, 4. Final Subsystem Roofline Summary, 5. Production Release Verdict, EXP-V2-0026 — Production Suffix Prefill & Decode Pipeline Hardening, Key Production Metrics ($P = 65,536, S = 512$, 64 Layers):

### Community 682 - "suffix_attention_halfwave_multitoken.cpp"
Cohesion: 0.14
Nodes (17): AccuracyResult, cosine, is_finite, max_abs_error, compute_accuracy(), __global__, size_t, T (+9 more)

### Community 683 - "V2-0001 — Clean-Sheet Single-MI50 Prefill V2: Recurrent-Layer Vertical Slice"
Cohesion: 0.12
Nodes (15): 1. Numerical Equivalence Against Canonical V1 Oracle, 2. Performance Bakeoff (Single Recurrent Layer on MI50), 3. Execution Phase Breakdown ($N = 512$), Baseline vs Candidate, Compiler Resource & Occupancy Analysis, Decision, Empirical Results, Environment (+7 more)

### Community 684 - "RecurrentLayerStateStorage"
Cohesion: 0.12
Nodes (22): uint32_t, RecurrentLayerStateStorage, d_conv_history_, d_state_, download, reset, upload, MatchResult (+14 more)

### Community 685 - "V2-0004 — Register-Resident GDN Integration and Full 64-Layer Prefill Pipeline"
Cohesion: 0.14
Nodes (13): 1. Stage A: Recurrent Layer & Topology Block Bakeoff (N=512), 2. Stage B: Full 64-Layer Model Multi-Length Performance, 3. Numerical & Stateful Segmentation Invariants (Full 64-Layer Model), Baseline, Candidate Implementation, Decision, Environment, Full Model P512 Phase Breakdown (Total = 2295.86 ms): (+5 more)

### Community 686 - "Metrics"
Cohesion: 0.22
Nodes (9): Metrics, actual_at_max, expected_at_max, finite, max_abs, max_index, max_rel, mean_abs (+1 more)

### Community 687 - "EXP-0375 — Correct repeated-B128 scheduler dispatch"
Cohesion: 0.29
Nodes (6): Correction, Decision, EXP-0375 — Correct repeated-B128 scheduler dispatch, Question, Route proof and timing results, Source defect

### Community 688 - "EXP-0394 — Physical-B512 recurrent projections for logical B128"
Cohesion: 0.22
Nodes (8): Candidate, Correctness gate, Decision, EXP-0394 — Physical-B512 recurrent projections for logical B128, Hypothesis, P640 end-to-end timing observed during the correctness matrix, Provenance, Workspace and route proof

### Community 689 - "model_plan.cpp"
Cohesion: 0.20
Nodes (17): Q4GemvKernel, align_up(), checked_add(), byte, hipError_t, size_t, string, GpuWeightArena::allocate() (+9 more)

### Community 690 - "EXP-0390 — P512 runtime-variance boundary"
Cohesion: 0.22
Nodes (8): Decision, EXP-0390 — P512 runtime-variance boundary, Fast/slow comparison, Measurement, P512 admitted samples, Provenance, Question, Sentinel and resource evidence

### Community 691 - "EXP-V2-0015 — Suffix TTFT Roofline & Kernel Attribution"
Cohesion: 0.13
Nodes (14): 1. Executive Summary & Core Finding, 2. Environment & Hardware State, 3. Clean Baseline Reproducibility (Phase A), 4. Comprehensive Kernel & Runtime Attribution Table, 5. Per-Layer Attribution Analysis, 6. Attention Root-Cause Diagnosis, 7. Mathematical Roofline Comparison (64K Prefix + 512 Suffix), 8. V2-0016 Candidate Analysis (+6 more)

### Community 692 - "V2-0006 — GQA Attention Optimization Sprint & Architectural Bakeoff"
Cohesion: 0.17
Nodes (11): 1. Goal, 2. Environment & Hardware State, 3. Kernel Configurations, 4. Part 1: Numerical Correctness Gate, 5. Part 2: Standalone Attention Performance Benchmark (1 Layer), 6.1 Why Candidate A (Query-Tiled) Is Slower than Control, 6.2 Why Candidate B (Split-KV) Fails Severely, 6. Architectural Analysis & Root Cause (+3 more)

### Community 693 - "PersistentSessionHeader"
Cohesion: 0.12
Nodes (17): uint32_t, uint64_t, uint8_t, PersistentSessionHeader, created_timestamp, gdn_layers, gdn_state_bytes, gqa_layers (+9 more)

### Community 694 - "Prefill V2 Architecture Specification"
Cohesion: 0.13
Nodes (14): 1. Executive Summary, 2.1 Hardware Contract, 2.2 Model Contract: Qwen3.8-27B-Q4_K_M, 2. Hardware & Model Contract, 3.1 Invariant Semantic Contract, 3.2 Fixed Internal GDN Chunk ($C = 64$), 3.3 Explicit State Flow, 3.4 Zero Hot-Path Memory Allocation (+6 more)

### Community 695 - "EXP-0387 — M28 full-model prefill frontier refresh"
Cohesion: 0.25
Nodes (7): Attribution, B64 direct-wide disposition, Basis and environment, Clean timing matrix, Decision, EXP-0387 — M28 full-model prefill frontier refresh, Question

### Community 696 - "V2-0005 — Native P512 Full-Model Macro Tiling & Baseline Qualification"
Cohesion: 0.12
Nodes (16): 1. Goal, 2. Hypothesis, 3. Environment & Hardware State, 4.1 Native Macro-512 Sequence Tiling, 4.2 VRAM Footprint & Monolithic Workspace Sizing, 4.3 Resident LM Head & Logit Generation, 4. Architectural Implementation, 5. Correctness & Drift Audit (+8 more)

### Community 697 - "tensor"
Cohesion: 0.29
Nodes (7): tensor, size_t, T, DeviceBuffer, data_, main(), upload()

### Community 698 - "MIInfer Current State of the Art"
Cohesion: 0.12
Nodes (8): External-port results, Historical accepted result — V2-0045 / M28, Leaderboard, Measured remaining work, MIInfer Current State of the Art, Reproducibility and promotion rules, Retained MIInfer path, Target

### Community 699 - "EXP-0395 — Legacy B4 tail fallback for logical B128"
Cohesion: 0.25
Nodes (7): Candidate, Correctness and performance gate, Decision, EXP-0395 — Legacy B4 tail fallback for logical B128, Provenance, Question, Route accounting

### Community 700 - "EXP-0358 — M26-C current decode-route differential"
Cohesion: 0.15
Nodes (13): Contract mapping, CPU-oracle logits at the common frontier, Diagnostic result (not timing qualification), Dispatch and memory audit, Existing selector-control probe (diagnostic only), EXP-0358 — M26-C current decode-route differential, Follow-up: copy/synchronization wall timers, Initial contract matrix (+5 more)

### Community 701 - "EXP-0389 — P512 regression discriminator"
Cohesion: 0.25
Nodes (7): Contract comparison, Decision, EXP-0389 — P512 regression discriminator, Provenance, Question, Results, Workloads and builds

### Community 702 - "EXP-0392 — P512 runtime-state scope"
Cohesion: 0.20
Nodes (9): Decision, EXP-0392 — P512 runtime-state scope, mx sentinel and telemetry, Process × iteration matrix (ms), Provenance, Question, Repeated-P512 harness contract, Sequence-pattern classification (+1 more)

### Community 703 - "ShapeBenchmark"
Cohesion: 0.12
Nodes (19): size_t, string, uint32_t, vector, find_tensor(), make_synthetic_prompt(), median(), ShapeBenchmark (+11 more)

### Community 704 - "PhysicalKvView"
Cohesion: 0.12
Nodes (14): uint32_t, PhysicalKvView, capacity, head_count_kv, head_dim, key_cache, key_cache_q8, key_scales (+6 more)

### Community 705 - "Qwen3Tokenizer"
Cohesion: 0.13
Nodes (12): size_t, string, uint32_t, unordered_map, vector, Qwen3Tokenizer, decode, encode (+4 more)

### Community 706 - "v2_0045_qualification_bench.cpp"
Cohesion: 0.23
Nodes (14): PrefillV2Model, size_t, span, string, uint32_t, vector, DeviceFloats, data (+6 more)

### Community 707 - "main"
Cohesion: 0.27
Nodes (12): size_t, string, uint32_t, vector, find_tensor(), main(), make_synthetic_prompt(), mean() (+4 more)

### Community 708 - "EXP-0351 — M26 real-context decode floor and context penalty"
Cohesion: 0.15
Nodes (9): Attribution decision boundary, Benchmark, EXP-0351 — M26 real-context decode floor and context penalty, Five-run baseline, Follow-up, Hypothesis, Interpretation, Revised gates (+1 more)

### Community 709 - "PrefillV2WorkspaceManager"
Cohesion: 0.18
Nodes (10): size_t, uint32_t, PrefillV2WorkspaceManager, d_buffer_, total_bytes_, workspace_, align128(), size_t (+2 more)

### Community 710 - "RecurrentLayerState"
Cohesion: 0.13
Nodes (21): size_t, RecurrentLayerState, d_conv_history, d_state, kConvHistoryBytes, kConvHistoryElements, kStateBytes, kStateElements (+13 more)

### Community 711 - "EXP-V2-0028 — Persistent Prefix Cache & Session Restore Qualification"
Cohesion: 0.13
Nodes (14): 1. Executive Summary, 2. Hypothesis & Architectural Motivation, 3.1 Serialization Format (`.miinfer`), 3.2 KV Cache Buffer Raw Access, 3.3 Engine & Server Integration, 3. Implementation Details, 4.1 Measurement Results, 4. Benchmark & Parity Qualification (+6 more)

### Community 712 - "EXP-V2-0010: DeltaNet Transposed Wave Acceleration & Defeating mx-llama.cpp in Decode"
Cohesion: 0.18
Nodes (10): Baseline & Competitor Context, Context Scaling Invariance, Correctness Qualification, Decision, End-to-End Performance Results, EXP-V2-0010: DeltaNet Transposed Wave Acceleration & Defeating mx-llama.cpp in Decode, GDN Kernel Bakeoff, Question (+2 more)

### Community 713 - "EXP-V2-0011 — Recovering the Historical ~32 ms Decode Frontier in Prefill V2"
Cohesion: 0.18
Nodes (10): 1. Hypothesis, 2. Motivation, 3. Implementation Details, 4. Hardware Environment, 5. Correctness & Determinism Verification, 6. End-to-End Benchmark Results, 7. Decision, A/B Comparison: V2-0008 vs V2-0010 vs V2-0011 (This Work) (+2 more)

### Community 714 - "DeviceShapeData"
Cohesion: 0.12
Nodes (17): Q8_1Block, DeviceShapeData, device_input_fp16, device_input_q8, device_output, device_weights, input_fp16, input_q8 (+9 more)

### Community 715 - "V2-0047 — Release packaging and distribution hardening"
Cohesion: 0.25
Nodes (7): Baseline and environment, Build-graph correction, CPack failure reproduced, Final release artifact, Objective, V2-0047 — Release packaging and distribution hardening, Verification transcript

### Community 716 - "openai_api.cpp"
Cohesion: 0.20
Nodes (20): OpenAiGeneratedToolCalls, calls, append_assistant_tool_calls(), append_generated_json_calls(), append_generated_xml_calls(), append_tool_result_messages(), build_chatml(), GenerationStopReason (+12 more)

### Community 717 - "GenerateStats"
Cohesion: 0.07
Nodes (26): GenerateStats, avg_decode_latency_ms, checkpoint_position, decode_ms, decode_tok_per_sec, gdn_checkpoint_position, generated_tokens, gqa_kv_reused_tokens (+18 more)

### Community 718 - "LogicalRange"
Cohesion: 0.20
Nodes (13): LogicalRange, begin, length, overlaps, check_end(), LogicalPageId, size_t, vector (+5 more)

### Community 719 - "EXP-V2-0013 — Long-Context Frontier Qualification (4K -> 8K -> 16K -> 32K -> 64K -> 128K)"
Cohesion: 0.12
Nodes (15): 1. Hypothesis, 1. KV Cache Footprint Economics, 2. Exact Reconciled VRAM Breakdown at 128K Context ($131,200$ tokens), 2. Motivation & Architectural Questions, 3. Allocation Curve Behavior ($\le 32\text{K}$ vs $> 32\text{K}$), 3. Environment & Hardware State, 4. Experimental Results, 5. Memory Architecture & Reconciled Footprint Accounting (+7 more)

### Community 720 - "EXP-V2-0012 — Reclaiming Resident VRAM via High-Value Layout Pruning while Preserving mx-Beating Decode"
Cohesion: 0.18
Nodes (10): 1. Hypothesis, 2. Motivation, 3. Baseline & Environment, 4. Layout Economy Audit, 5. Experimental Results, 6. Context Window Feasibility on 1 × MI50 32GB, 7. Decision, End-to-End Latency & Throughput (TG = 128 tokens) (+2 more)

### Community 721 - "EXP-V2-0007 — Unified Prefill V2 to Static Decode Pipeline"
Cohesion: 0.14
Nodes (13): 1. Unified State & Model Representation (`PrefillV2Model`), 2. Physical Batch Geometry Support, Architecture & Implementation, Decision, End-to-End Benchmark Comparison, End-to-End Performance vs `mx-llama.cpp`, EXP-V2-0007 — Unified Prefill V2 to Static Decode Pipeline, Hardware State (+5 more)

### Community 722 - "RecurrentLayerDecodePhaseTimings"
Cohesion: 0.20
Nodes (10): RecurrentLayerDecodePhaseTimings, conv_l2_norm_ms, ffn_down_residual_ms, ffn_gate_up_swiglu_ms, gdn_step_ms, norm_beta_alpha_ms, qkv_gate_proj_ms, residual_norm_ms (+2 more)

### Community 723 - "Options"
Cohesion: 0.14
Nodes (14): uint32_t, Options, cache_regime, custom_k, custom_label, custom_m, device, experiment (+6 more)

### Community 724 - "EXP-V2-0027R — Cold-Prefill Arithmetic & Roofline Reconciliation"
Cohesion: 0.13
Nodes (14): 1. Executive Summary & Problem Statement, 2.1 Layer-by-Layer Useful Matrix Multiply Work (Qwen3.8-27B), 2. Rigorous Model Arithmetic Derivation ($M \times N \times K$), 3. Authoritative Reconciled Performance Metrics ($M=512$ Macro-Tile), 4. Context Scaling: P512 vs. Long-Context Cold Ingestion, 5. Where the Missing ~76% Compute Headroom Lies, 6. Scientific Verdict & Updated Roadmap, A. Recurrent GDN Layer (48 Layers) (+6 more)

### Community 725 - "EXP-V2-0018-suffix-hip-graph.md"
Cohesion: 0.18
Nodes (10): 1. Executive Summary & Objective, 2. Phase 0: Suffix Prefill Captureability Audit, 3. Primary Benchmark Results ($P=65,536, S=512$), 4. Context Length Scaling Matrix, 5. Multi-Turn Sequential Context Advance Simulation, 6. Milestone Qualification Gates, 7. Architectural Decisions & Future Directions, EXP-V2-0018 — Suffix Prefill HIP Graph Replay & Dispatch Elimination (+2 more)

### Community 726 - "EXP-V2-0009: Decode Fast-Path Recovery Inside Prefill V2"
Cohesion: 0.20
Nodes (9): Baseline (V2-0008 Control), Context Scaling Verification, Correctness Qualification, Cumulative Performance Progress, Decision, EXP-V2-0009: Decode Fast-Path Recovery Inside Prefill V2, Fast-Path Differential Analysis, Micro-Bakeoff Results (+1 more)

### Community 727 - "prefill_v2_model_bench.cpp"
Cohesion: 0.16
Nodes (13): AccuracyMetrics, cosine, is_finite, max_abs_err, mean_abs_err, rel_rms_err, rms_err, compute_accuracy() (+5 more)

### Community 728 - "hip_check.hpp"
Cohesion: 0.06
Nodes (30): string, TimingResult, delta_us, extra_mib_per_layer, mmq_us, total_mib, total_model_ms, type_str (+22 more)

### Community 729 - "Qwen3Layer0KvCache"
Cohesion: 0.16
Nodes (20): size_t, span, Qwen3Layer0KvCache, append, reset, Qwen3DecodeCache::reset(), cache_contract_test(), checkpoint_tolerance() (+12 more)

### Community 730 - "EXP-V2-0021-production-suffix-hip-graph.md"
Cohesion: 0.12
Nodes (15): 1. Executive Summary & Objective, 2. Experimental Setup & Workload Specification, 3. Experiment A: Correctness & Parity Across Context Regimes, 4. Experiment B: Interleaved Performance & Launch Attribution ($P=65,536, S=512$), 5. Experiment C: Multi-Turn Context Advancement, 6. Experiment D: Stress Replay & VRAM Leak Audit, 7. Experiment E: Non-Standard Suffix Fallback, 8. Milestone Qualification Gate Evaluation (+7 more)

### Community 731 - "OpenAiChatRequest"
Cohesion: 0.06
Nodes (39): optional, size_t, resolve_output_token_limit(), ChatMessage, content, role, tool_call_id, tool_calls (+31 more)

### Community 732 - "EXP-V2-0014 — Prefix & State Reuse with Suffix-Only Prefill"
Cohesion: 0.15
Nodes (12): 1. Hypothesis, 1. VRAM Breakdown with Prefix Caching ($64\text{K} + 512$ Context), 2. Architecture & Implementation, 2. Suffix-Only Execution Proof, 3. Environment & Hardware State, 4. Experimental Results, 5. Architectural & Memory Accounting, 6. Success Gates Evaluation (+4 more)

### Community 733 - "m31_0003_canonical_agent_workload.cpp"
Cohesion: 0.25
Nodes (13): main(), size_t, time_point, uint32_t, vector, elapsed_ms(), event_tokens(), options() (+5 more)

### Community 734 - "DevicePrefillState"
Cohesion: 0.13
Nodes (19): TopologyBlockProfileBreakdown, gdn0_ms, gdn1_ms, gdn2_ms, gqa3_ms, total_block_ms, DevicePrefillState, base_position (+11 more)

### Community 735 - "TerminalTextStream"
Cohesion: 0.18
Nodes (14): Mode, cmd_chat(), cmd_run(), configured(), initializer_list, extract_reasoning_and_content(), parse_cli_size(), terminal_output() (+6 more)

### Community 736 - "M23ProfileCounters"
Cohesion: 0.20
Nodes (10): array, M23ProfileCounters, attention_dispatches, attention_weight_upload_bytes, attention_weight_upload_ms, recurrent_dispatches, recurrent_weight_upload_bytes, recurrent_weight_upload_ms (+2 more)

### Community 737 - "v2_0023_cross_query_attention_bench.cpp"
Cohesion: 0.43
Nodes (7): hipEvent_t, vector, elapsed_ms(), main(), mean(), median(), stddev()

### Community 738 - "Iteration log"
Cohesion: 0.03
Nodes (61): All-call real-input attention comparison (2026-09-29), Current-build oracle and P8216 check (2026-09-29), Exact trace-shaped P8192/TG1 check (2026-09-29), Final real-input attribution and disposition (2026-09-29), First-chunk legacy substitution — no parity recovery (2026-09-29), Iteration 11 hypothesis / gates, Iteration 12 hypothesis / gates (archived), Iteration 13 hypothesis / gates (archived) (+53 more)

### Community 739 - "qwen3_swiglu_q8_bench.cpp"
Cohesion: 0.39
Nodes (7): vector, main(), measure(), Stats, mean_us, median_us, summarize()

### Community 740 - "Qwen3GpuPlan"
Cohesion: 0.07
Nodes (40): GpuWeightArena, allocate, release, upload, GgufTensorType, size_t, string, uint64_t (+32 more)

### Community 741 - "qwen35_gpu_pipeline.hpp"
Cohesion: 0.07
Nodes (63): pack_q4k_mmq_tensor(), main(), allocate(), ByteMismatch, count, first, combined_attn_qk_enabled(), combined_qkv_gate_enabled() (+55 more)

### Community 742 - "original-m31_0002_checkpointed_call.cpp"
Cohesion: 0.22
Nodes (13): Context, name, seed, tokens, ostream, size_t, string, uint32_t (+5 more)

### Community 743 - "qwen3_decode_profile.cpp"
Cohesion: 0.18
Nodes (18): build_json(), string, timespec, uint32_t, elapsed_ms(), json_escape(), main(), now() (+10 more)

### Community 744 - "M31-LC-0001 — MMQ-Only Long-Context Advantage"
Cohesion: 0.14
Nodes (12): Capacity interpretation, Estimated usage, Inputs and method, M31-LC-0001 Phase 1 — Memory envelope estimate, Decision, Evidence map, Final fields, M31-LC-0001 — MMQ-Only Long-Context Advantage (+4 more)

### Community 745 - "Group"
Cohesion: 0.21
Nodes (13): set, size_t, string, uint64_t, vector, Group, bytes, count (+5 more)

### Community 746 - "qwen3_cached_attention_determinism_gpu_test.cpp"
Cohesion: 0.33
Nodes (9): size_t, T, vector, DeviceBuffer, data_, download(), main(), same_bytes() (+1 more)

### Community 747 - "EXP-0359 — M26-CQ decode semantic equivalence"
Cohesion: 0.25
Nodes (8): Commands, Decision, EXP-0359 — M26-CQ decode semantic equivalence, Layer/operator localization, Method and environment, Numerical and output results, Question, Structural contract

### Community 748 - "V2-0045 — Definitive competitive qualification"
Cohesion: 0.25
Nodes (8): Attention frontier — deferred, Decision, Pinned starting point, Qualification protocol, Real agent workload and stability gate, Required scoreboard, State, V2-0045 — Definitive competitive qualification

### Community 749 - "EXP-V2-0016-wave64-suffix-attention.md"
Cohesion: 0.18
Nodes (10): 1. Executive Summary & Core Finding, 2. Candidate Architecture Bakeoff (11 Candidates Evaluated), 3. Architectural Discoveries & Profiling, 4. End-to-End Suffix Attribution Table ($64\text{K} + 512$), 5. Decision & Next Steps, EXP-V2-0016 — Specialized Wave64 Split-K Suffix Attention, Follow-up (V2-0017 Frontier):, Key Results ($P = 65,536, S = 512$): (+2 more)

### Community 750 - "recurrent_block"
Cohesion: 0.16
Nodes (20): array, kChannels, path, size_t, span, vector, full_block(), main() (+12 more)

### Community 751 - "EXP-V2-0017-hierarchical-gqa-query-reuse.md"
Cohesion: 0.17
Nodes (11): 1. Executive Summary & Objective, 1. The LDS vs HBM Trade-off on Vega20, 2. Architectural Bakeoff & Microbenchmark Results, 2. Register Budget & Wave Occupancy, 3. Microarchitectural Analysis & Roofline Limits, 4. Full Model Suffix TTFT Attribution ($64\text{K} + 512$), 5. Multi-Scenario Prefix State Reuse Qualification, 6. Decision & Recommendations (+3 more)

### Community 752 - "v2_0024_quantized_prefix_kv_bench.cpp"
Cohesion: 0.24
Nodes (10): compute_accuracy(), hipEvent_t, size_t, uint32_t, vector, elapsed_ms(), GpuBuffer, ptr (+2 more)

### Community 753 - "Potential areas"
Cohesion: 0.29
Nodes (7): Additional quantization, Long-context specialization, M7 — Expansion, Potential areas, Second model, Serving, Speculative decoding / MTP

### Community 754 - "AttentionLayerDecodePhaseTimings"
Cohesion: 0.20
Nodes (10): AttentionLayerDecodePhaseTimings, ffn_down_residual_ms, ffn_gate_up_swiglu_ms, norm_ms, o_proj_ms, qk_rope_kv_store_ms, qkv_proj_ms, residual_norm_ms (+2 more)

### Community 755 - "DeviceBuffer"
Cohesion: 0.36
Nodes (5): size_t, T, DeviceBuffer, count_, data_

### Community 756 - "EXP-V2-0025-lds-staged-gqa-attention.md"
Cohesion: 0.17
Nodes (11): 1. Executive Summary & Core Engineering Conclusions, 2. Detailed Microarchitectural Breakdown & Roofline Equilibrium, 3. Microarchitectural Analysis of Evaluated Candidates, 4. Final Verdict & Milestone Disposition, 5. Updated Roadmap, Candidate A: LDS KV Tile Tiled (192-Thread Workgroup), Candidate B: LDS Query Staging (64-Thread Wave64, 0 In-Loop Barriers), Candidate C: Co-scheduled Multi-Wave Workgroup (192-Thread Workgroup, L1 Cache Sharing) (+3 more)

### Community 757 - "m12_dense_stage_bench.cpp"
Cohesion: 0.12
Nodes (22): as(), check_hipblas(), Fn, hipblasHandle_t, hipblasStatus_t, hipEvent_t, T, vector (+14 more)

### Community 758 - "bench-m18-runtime.py"
Cohesion: 0.60
Nodes (5): add_steady_rate(), command_output(), main(), run_case(), sha256()

### Community 759 - "GenerateOptions"
Cohesion: 0.07
Nodes (30): GenerateOptions, cache_prefix_after, cache_prefix_len, enable_prefix_reuse, frequency_penalty, max_new_tokens, on_token, persistent_session_dir (+22 more)

### Community 760 - "ShapeBenchmark"
Cohesion: 0.12
Nodes (18): size_t, string, uint32_t, vector, find_tensor(), make_synthetic_prompt(), median(), ShapeBenchmark (+10 more)

### Community 761 - "EXP-V2-0019-quantized-kv-cache.md"
Cohesion: 0.14
Nodes (13): 1. Absence of INT8 Mixed-Precision Tensor Instructions, 1. Executive Summary & Objective, 2. Candidate Evaluation Matrix ($P=65,536, S=512$), 2. Instruction Issue Expansion in the Inner Loop, 3. Context Length Scaling Ladder ($S=512$, 16 GQA Layers), 3. Shift from Memory Bandwidth Bound to ALU Issue Bound, 4. Memory Footprint & Context Capacity, 5. Single-Token Decode Latency Check (+5 more)

### Community 762 - "PrefillV2Model::PrefillV2Model"
Cohesion: 0.29
Nodes (7): allocate_resources, clear_snapshots, free_resources, KvCacheQuantMode, PrefillV2Model, PrefillV2Model::PrefillV2Model(), PrefillV2Model::reset_state()

### Community 763 - "v2_0025_lds_staged_gqa_attention_bench.cpp"
Cohesion: 0.32
Nodes (7): hipEvent_t, uint32_t, vector, elapsed_ms(), main(), median(), qwen35_splitk_suffix_reduction_kernel()

### Community 764 - "bench_m10_latency_curve.py"
Cohesion: 0.42
Nodes (8): get_vram_mib(), main(), make_prompt(), parse_cli_stderr(), run_cold_request(), run_http_second_turn(), run_http_warm_request(), wait_for_server()

### Community 765 - "v2_0019_quantized_kv_bench.cpp"
Cohesion: 0.24
Nodes (10): compute_accuracy(), hipEvent_t, size_t, uint32_t, vector, elapsed_ms(), GpuBuffer, ptr (+2 more)

### Community 766 - "m31_0001_performance_frontier.cpp"
Cohesion: 0.29
Nodes (12): ContextConfig, name, seed, tokens, size_t, uint32_t, vector, main() (+4 more)

### Community 767 - "V2-0043 — Reference-shaped GQA attention rewrite"
Cohesion: 0.22
Nodes (9): Arithmetic boundary: reference-shaped candidate vs qualified MIInfer control, Baseline and bottleneck evidence, Candidate invariants, Compiler-schedule evidence, Correctness and performance gates, Objective, Reference mechanism reconstructed, Resource / ISA gate (+1 more)

### Community 768 - "DeviceBuffer"
Cohesion: 0.40
Nodes (5): size_t, T, DeviceBuffer, count, ptr

### Community 769 - "gfx906 Toolbox Reference and MIInfer Follow-up"
Cohesion: 0.15
Nodes (12): Candidate A — M29.0A external-reference calibration, Candidate B — gfx906 reference environment, Candidate C — MIInfer OCI/Podman distribution, Contract, Explicit non-goals, External source pin, gfx906 Toolbox Reference and MIInfer Follow-up, Question (+4 more)

### Community 770 - "v2_0021_production_suffix_hip_graph_bench.cpp"
Cohesion: 0.44
Nodes (8): size_t, uint32_t, vector, main(), make_synthetic_prompt(), mean(), median(), stddev()

### Community 771 - "EXP-V2-0030 — Direct Global-to-VGPR MMQ Weight Streaming & Instruction-Amplification Reduction"
Cohesion: 0.15
Nodes (12): 1. Executive Summary, 2. Hypothesis & Architectural Theory, 3. Resource & ISA Comparison, 4. Benchmark Results (Interleaved A/B Pairs, M=512), 5. Phase 7 — Gate + Up Projection Fusion Opportunity Analysis, 6. Qualification Gates Evaluation, 7. Decision & Roadmap, Architectural Reality on gfx906 (+4 more)

### Community 772 - "PrefillV2Model::snapshot"
Cohesion: 0.24
Nodes (13): capture_snapshot_backing, evict_snapshots_except, refresh_snapshot_bytes, release_snapshot, release_snapshot_record, restore_snapshot, SnapshotId, full_copy_snapshot_mode() (+5 more)

### Community 773 - "AccuracyMetrics"
Cohesion: 0.20
Nodes (10): AccuracyMetrics, cosine, is_finite, max_abs_error, mean_abs_error, rel_rms_error, rms_error, compute_accuracy() (+2 more)

### Community 774 - "GemvShape"
Cohesion: 0.29
Nodes (7): GemvShape, id, k, m, projection, check_output(), vector

### Community 775 - "qwen35_gqa_prefill_bench.cpp"
Cohesion: 0.22
Nodes (9): hipEvent_t, size_t, T, uint32_t, DeviceBuffer, ptr, elapsed(), main() (+1 more)

### Community 776 - "run-m31-exploratory.sh"
Cohesion: 0.37
Nodes (10): event(), group_has_live_members(), on_exit(), on_signal(), reap_worker(), run-m31-exploratory.sh script, stop_worker_group(), usage() (+2 more)

### Community 777 - "V2-0048B — Generation completion contract"
Cohesion: 0.25
Nodes (7): Current-source behavior checks, Final canonical artifact gate, Implementation, Natural-completion measurements before changing the default, Pending release gates, Status, V2-0048B — Generation completion contract

### Community 778 - "resolve_model"
Cohesion: 0.26
Nodes (13): configured_model_directories(), map, optional, path, vector, expand_user_path(), find_models(), parse_cli_float() (+5 more)

### Community 779 - "hipMemcpyAsync"
Cohesion: 0.16
Nodes (22): hipMemcpyKind, copy, AttentionLayerKvCacheStorage::AttentionLayerKvCacheStorage(), AttentionLayerKvCacheStorage::download_key(), AttentionLayerKvCacheStorage::download_raw(), AttentionLayerKvCacheStorage::download_raw_range(), AttentionLayerKvCacheStorage::download_value(), AttentionLayerKvCacheStorage::raw_tokens_bytes() (+14 more)

### Community 780 - "bench-serve-concurrency.py"
Cohesion: 0.57
Nodes (6): atomic_json(), chat(), command(), control_probe(), http_get(), main()

### Community 781 - "M30-0000 — Reuse Opportunity Map (Workstream B)"
Cohesion: 0.17
Nodes (11): Blockers and boundaries, Exact status, GDN/recurrent state, GQA KV state, Historical evidence (not current qualification), Invalidation and ownership conditions, M30-0000 — Reuse Opportunity Map (Workstream B), Opportunity map (+3 more)

### Community 782 - "hipMemcpy"
Cohesion: 0.07
Nodes (50): main(), main(), F, main(), measure(), main(), main(), main() (+42 more)

### Community 783 - "5. Runtime Layers"
Cohesion: 0.29
Nodes (7): 5.1 Model Layer, 5.2 Packing / Representation Layer, 5.3 Memory Planner, 5.4 Kernel Planner, 5.5 Execution Plan, 5.6 Kernel Layer, 5. Runtime Layers

### Community 784 - "E2-0003 — MMQ-only Gate/Up candidate"
Cohesion: 0.17
Nodes (11): E2-0003 — MMQ-only Gate/Up candidate, Implementation plan and feasibility, Interim campaign disposition, Interim conclusion and next goal, Phase A checkpoint, Phase B checkpoint, Phase result, Reopened recovery checkpoint — 2026-10-10 (+3 more)

### Community 785 - "ProfileScope"
Cohesion: 0.17
Nodes (12): hipEvent_t, ProfileScope, boundary_stage_, bytes_, category_, copy_, dispatches_, ffn_stage_ (+4 more)

### Community 786 - "EXP-V2-0029 — MMQ Useful-Compute Efficiency & Instruction-Amplification Analysis"
Cohesion: 0.18
Nodes (10): 1. Executive Summary & Objective, 2. Methodology & Experimental Environment, 3. Disassembly & Instruction Profile Analysis, 4. Benchmark Results Across Qwen3.8-27B Projection Shapes ($M=512$), 5. Architectural Findings & The Remaining Efficiency Ceiling, 6. Decision & Commit, EXP-V2-0029 — MMQ Useful-Compute Efficiency & Instruction-Amplification Analysis, Instruction Breakdown Table: (+2 more)

### Community 787 - "V2-0048A — Final artifact release gate"
Cohesion: 0.33
Nodes (5): Artifact and build identity, Decision, Interpretation and next gate, Qualification results, V2-0048A — Final artifact release gate

### Community 788 - "run_case"
Cohesion: 0.20
Nodes (11): check_hipblas(), Fn, hipblasHandle_t, hipblasStatus_t, Tile, uint32_t, uint8_t, vector (+3 more)

### Community 789 - "m12_gdn_oracle.cpp"
Cohesion: 0.57
Nodes (7): at(), chunkwise(), size_t, main(), normalize_rows(), recurrent(), Matrix

### Community 790 - "V2-0048A — Generation budget and exact-prefix fault"
Cohesion: 0.33
Nodes (5): Candidate and qualification, Evidence on the preserved V2-0048 package, Minimal correction, Status, V2-0048A — Generation budget and exact-prefix fault

### Community 791 - "model_loader_test.cpp"
Cohesion: 0.32
Nodes (12): append_string(), append_u32(), append_u64(), path, string, uint32_t, uint64_t, uint8_t (+4 more)

### Community 792 - "m31_0002_persistent_context_qualification.cpp"
Cohesion: 0.38
Nodes (9): size_t, uint32_t, vector, main(), now_ms(), options(), prompt(), report_mismatch() (+1 more)

### Community 793 - "model.cpp"
Cohesion: 0.07
Nodes (54): capture_decode_graph, capture_suffix_graph, cleanup_decode_graph, cleanup_suffix_graph, compute_logits, decode_graph_weight_pointers, load_session, reset_state (+46 more)

### Community 794 - "M30-0002 production authority and residual agent cost"
Cohesion: 0.20
Nodes (9): Completion fields, Completion gates, Current production path, Decision, Final capability recommendation, Historical boundary, M30-0002 production authority and residual agent cost, Residual canonical workload (+1 more)

### Community 795 - "Qwen35Model"
Cohesion: 0.24
Nodes (10): string, main(), run_q5k_benchmark(), run_q6k_benchmark(), shared_ptr, string, Qwen35Model, artifact_path_ (+2 more)

### Community 796 - "GgufFile"
Cohesion: 0.12
Nodes (16): GgufFile, file_descriptor_, mapping_, metadata, metadata_array_is_string, metadata_array_size, metadata_float, metadata_string (+8 more)

### Community 797 - "v2_0045_agent_workload.py"
Cohesion: 0.22
Nodes (15): main(), append_tool_exchange(), attach_server_latency(), build_context(), execute_tool(), gpu_snapshot(), http_text(), latest_server_latency() (+7 more)

### Community 798 - "Event"
Cohesion: 0.50
Nodes (3): hipEvent_t, Event, value

### Community 799 - "Parallel MI50 Development"
Cohesion: 0.20
Nodes (8): Qualified MI50 development environment, End of the two-host phase, Environment-parity exit gate, First prerequisite — reproducible environment, Initial parallel split, Parallel MI50 Development, Parallel work rules, Purpose

### Community 800 - "M31-0006 — Build and stop-guard status"
Cohesion: 0.20
Nodes (8): GPU evidence and remaining boundary, M31-0006 — Build and stop-guard status, Reproducible build, Stop guard, M31-0004, M31-0005, M31-0006 — GPU regression status, Raw evidence

### Community 801 - "download"
Cohesion: 0.05
Nodes (44): DecodeLayerCapture, inputs, outputs, download(), download_bytes(), GatePathCapture, gate, gated (+36 more)

### Community 802 - "attention_layer.cpp"
Cohesion: 0.33
Nodes (13): compare_iteration26_real_operands(), hipStream_t, size_t, uint32_t, PrefillV2AttentionLayer::decode(), PrefillV2AttentionLayer::decode_profiled(), PrefillV2AttentionLayer::forward(), PrefillV2AttentionLayer::forward_profiled() (+5 more)

### Community 803 - "q4k_layout_bench.cpp"
Cohesion: 0.29
Nodes (5): as(), hipEvent_t, T, Event, p

### Community 804 - "RecurrentLayerPhaseTimings"
Cohesion: 0.20
Nodes (10): RecurrentLayerPhaseTimings, conv_l2_norm_ms, ffn_gate_up_ms, gdn_chunkwise_ms, norm_beta_alpha_ms, qkv_gate_proj_ms, residual_norm_ms, ssm_post_out_ms (+2 more)

### Community 805 - "Q8ExactBlock"
Cohesion: 0.13
Nodes (18): int16_t, int8_t, Q8_1Block, d, qs, s, Q8ExactBlock, d (+10 more)

### Community 806 - "qwen3_gpu_layer.cpp"
Cohesion: 0.08
Nodes (57): AttentionKernel, Qwen3Projection, Qwen3ProjectionPrecision, capture(), capture_qwen3_head_norm(), copy_to_host(), Function, Q8_1Block (+49 more)

### Community 808 - "DeviceDecodeState"
Cohesion: 0.20
Nodes (10): DeviceDecodeState, current_token, generated, max_generated, position, stop, uint32_t, launch_qwen35_fused_k_norm_rope_kv_store() (+2 more)

### Community 809 - "AttentionLayerProfileBreakdown"
Cohesion: 0.20
Nodes (10): AttentionLayerProfileBreakdown, causal_attn_ms, ffn_gate_up_ms, norm_ms, o_proj_ms, post_norm_ms, qkv_proj_ms, rope_kv_store_ms (+2 more)

### Community 810 - "NumericalMetrics"
Cohesion: 0.22
Nodes (9): size_t, NumericalMetrics, all_finite, cosine_similarity, max_abs_err, mean_abs_err, rel_rms, rmse (+1 more)

### Community 811 - "developer-guide.md"
Cohesion: 0.12
Nodes (11): MIInfer CLI product contract, V2-0046 verification — 2026-09-30, Build a release archive from source, Install a release archive, Qualified performance scope, Release and installation, Linux / gfx906 artifact, MIInfer v0.2.0 (+3 more)

### Community 812 - "M30-0001 final evidence"
Cohesion: 0.22
Nodes (8): Baseline qualification boundary, Bounded no-reuse attribution probe, Canonical run result, Fixed-transcript 86-request replay, Frozen-context rerun, Implementation evidence, M30-0001 final evidence, Qualification boundary

### Community 813 - "PhysicalKvShardView"
Cohesion: 0.22
Nodes (9): PhysicalKvShardView, head_begin, head_count, key_cache, key_cache_q8, key_scales, value_cache, value_cache_q8 (+1 more)

### Community 814 - "m24_recurrent_layer_bakeoff.cpp"
Cohesion: 0.16
Nodes (12): as(), T, uint32_t, main(), run_case(), set_common_environment(), M12GdnChunkWorkspace, corrected_values (+4 more)

### Community 815 - "3. Correctness Is a Benchmark Prerequisite"
Cohesion: 0.50
Nodes (4): 3. Correctness Is a Benchmark Prerequisite, End-to-end inference, Kernel, Model component

### Community 816 - "E2-0003Q qualification report"
Cohesion: 0.22
Nodes (8): Context lifecycle gates, E2-0003Q qualification report, Evidence map, Limits and skipped work, Required final fields, Runtime results, Source and build, Verdict

### Community 817 - "Buffer"
Cohesion: 0.21
Nodes (6): Buffer, pointer, size_t, size_t, GpuBuffer, ptr

### Community 818 - "MIInfer Current State"
Cohesion: 0.25
Nodes (8): Closed attention stretch, Current production implementation, Current qualification phase — M31, Current released baseline — v0.2.0 (2026-10-01), External gfx906 reference discovery — 2026-10-02, Historical checkpoint — M28 / V2-0043 (2026-09-29), Historical milestone record — V2-0008 (Dedicated Single-Token Decode Execution & Reusable HIP Graph Replay), MIInfer Current State

### Community 819 - "M31-0002T-0001 — First-divergence and weight-residency attribution"
Cohesion: 0.25
Nodes (7): Baseline and provenance, Correctness, M31-0002T-0001 — First-divergence and weight-residency attribution, Memory, Ranked next work, Required final fields, Validation and blockers

### Community 820 - "M31-0003 — Consolidated findings and bounded next steps"
Cohesion: 0.25
Nodes (7): Consolidated ranking, Dependency-aware next optimization sequence, Implemented fix, M31-0003 — Consolidated findings and bounded next steps, Offline validation and outstanding GPU work, Scope and identity, What is established—and what is not

### Community 821 - "Canonical Forward Roadmap"
Cohesion: 0.25
Nodes (8): Canonical Forward Roadmap, Cross-cutting — gfx906 reproducibility and distribution, M29.0A — Current gfx906 external-reference calibration, M29 — Persistent Context Architecture, M30 — Agent Runtime Advantage, M31 — Single-MI50 Agent Frontier Qualification, Second-model / generalisation work, V3 — Dual-MI50 Agent Engine

### Community 822 - "Q4KWaveSwigluFusedTile"
Cohesion: 0.25
Nodes (8): uint64_t, Q4KWaveSwigluFusedTile, metadata_gate, metadata_up, padding, words_p0, words_p1, Metadata

### Community 823 - "FfnTailReplay"
Cohesion: 0.33
Nodes (6): FfnTailReplay, down, gate, layer_output, swiglu, up

### Community 824 - "test_m31_exploratory_thermal_guard.sh"
Cohesion: 0.46
Nodes (6): assert_bounded_shutdown(), assert_dead(), assert_event_order(), is_live(), test_m31_exploratory_thermal_guard.sh script, sleep()

### Community 825 - "decode_fast_path_bakeoff.cpp"
Cohesion: 0.33
Nodes (6): F, measure(), TimingResult, max_us, mean_us, min_us

### Community 826 - "M30-0005 Shared Snapshot State and Copy-on-Write Contract"
Cohesion: 0.29
Nodes (6): Accounting, API and limits, M30-0005 Shared Snapshot State and Copy-on-Write Contract, Ownership model, Required invariants, Scope

### Community 827 - "M31-0004 — GDN computation findings"
Cohesion: 0.29
Nodes (6): Confirmed opportunity and decision, Hardware validation, M31-0004 — GDN computation findings, Production hot path, Remaining computational opportunities, Scope and evidence

### Community 828 - "M31-0003 — Production execution path"
Cohesion: 0.29
Nodes (6): Cold prefill, Existing long-context evidence, M31-0003 — Production execution path, One-token decode, Persistent-context suffix prefill, Scope and source identity

### Community 829 - "M31-0005 — Implementation results"
Cohesion: 0.29
Nodes (6): Candidate decision and next bottleneck, Implemented, Investigated production paths, M31-0005 — Implementation results, Validation and blockers, Work and memory model

### Community 830 - "M1 — Kernel Laboratory"
Cohesion: 0.29
Nodes (7): Benchmark harness, Exit criteria, Goal, Initial kernel areas, Initial representative shapes, M1 — Kernel Laboratory, Questions

### Community 831 - "V2-0044 — Precision-preserving KQ-to-V path"
Cohesion: 0.29
Nodes (7): Decision, Gates, Objective and measured motivation, Production-route parity result (2026-09-29), Reference and fixed architecture, State, V2-0044 — Precision-preserving KQ-to-V path

### Community 832 - "DeviceInfo"
Cohesion: 0.29
Nodes (6): DeviceInfo, architecture, index, name, total_vram_bytes, size_t

### Community 833 - "E2-0003 recovery results"
Cohesion: 0.29
Nodes (6): Disposition and limits, E2-0003 recovery results, Operation reference, Separate Machinist qualification, Two matched 1K/eight-token pairs, Z840 build and identity

### Community 834 - "make_prompt"
Cohesion: 0.38
Nodes (6): size_t, uint32_t, uint64_t, vector, make_prompt(), prompt_fingerprint()

### Community 835 - "CompareMetrics"
Cohesion: 0.33
Nodes (6): compare_outputs(), CompareMetrics, finite, max_abs, rmse, vector

### Community 836 - "m31_0003_canonical_agent_workload.py"
Cohesion: 0.67
Nodes (5): manifest(), main(), run_path(), sha256(), transcript()

### Community 837 - "DeviceBuffer"
Cohesion: 0.40
Nodes (5): size_t, T, DeviceBuffer, count, ptr

### Community 838 - "M30-0000 — Canonical Agent Baseline"
Cohesion: 0.33
Nodes (5): Context regimes and micro-metrics, M30-0000 — Canonical Agent Baseline, Provenance, Session totals, Status

### Community 839 - "M30-0000 — Integrated Evidence"
Cohesion: 0.33
Nodes (5): Exactly one next capability, M30-0000 — Integrated Evidence, Opportunity synthesis, Required completion report, Reusable-state accounting

### Community 840 - "M30-0001 Exact-Prefix Continuation Contract"
Cohesion: 0.33
Nodes (5): Base, Explicit non-goals, M30-0001 Exact-Prefix Continuation Contract, Required observability, Scope

### Community 841 - "M30-0003 Multi-Checkpoint and Longest-Prefix Reuse"
Cohesion: 0.33
Nodes (5): Evidence boundary, Exit criteria for implementation, Feasibility boundary, M30-0003 Multi-Checkpoint and Longest-Prefix Reuse, Verified current state

### Community 842 - "M30-0004 Snapshot, Fork, and Rollback Evidence"
Cohesion: 0.33
Nodes (5): Delivered, M30-0004 Snapshot, Fork, and Rollback Evidence, Qualification boundary, Workstream A — lifecycle gate, Workstream B — branch/retry workload

### Community 843 - "M31-0001 — Single-MI50 Performance Frontier"
Cohesion: 0.33
Nodes (5): Evidence, Fixed matrix, M31-0001 — Single-MI50 Performance Frontier, Scope, Short-context reference points

### Community 844 - "M31-0002 — Persistent-Context Qualification"
Cohesion: 0.33
Nodes (5): Evidence, M31-0002 — Persistent-Context Qualification, Required flow, Required metrics, Scope

### Community 845 - "M31-0003 — Canonical Agent Workload"
Cohesion: 0.33
Nodes (5): Evidence, Headline metrics, M31-0003 — Canonical Agent Workload, Required sequence, Scope

### Community 846 - "M31-0004 — GDN checkpoint lifecycle"
Cohesion: 0.33
Nodes (5): Change made, Copy ledger, M31-0004 — GDN checkpoint lifecycle, Ownership and purpose, Semantics and risks

### Community 847 - "M31-0004 — Dispatch and memory findings"
Cohesion: 0.33
Nodes (5): Changes not made, Host bookkeeping optimization, M31-0004 — Dispatch and memory findings, Per-tile execution model, Synchronization and transfers

### Community 848 - "M31-0004 — Optimization results"
Cohesion: 0.33
Nodes (5): Implemented changes, M31-0004 — Optimization results, Remaining work and validation, Status, Work reduction ledger

### Community 849 - "M31-0004 — Stability and Release Gate"
Cohesion: 0.33
Nodes (5): Evidence, M31-0004 — Stability and Release Gate, Required repetitions, Required statistics and failure accounting, Scope

### Community 850 - "M31-0003 — Attention and KV investigation"
Cohesion: 0.33
Nodes (5): Decision, Dispatch and historical experiments, Long-context traffic model, M31-0003 — Attention and KV investigation, Scope and evidence

### Community 851 - "M31-0003 — VRAM ownership and accounting"
Cohesion: 0.33
Nodes (5): Implemented safe reclamation, M31-0003 — VRAM ownership and accounting, Offline validation and limit, Remaining footprint gap, Source-derived model

### Community 852 - "M31-0005 — Correctness and regression protection"
Cohesion: 0.33
Nodes (5): Change and adversarial case, Invariants retained, M31-0005 — Correctness and regression protection, Minimal hardware test when safe, Validation

### Community 853 - "M31-0005 — Decode attribution"
Cohesion: 0.33
Nodes (5): Candidate ranking, M31-0005 — Decode attribution, Main deductions, Production path and measured frontier, Source-derived work model

### Community 854 - "M31-0005 — KV storage and memory ledger"
Cohesion: 0.33
Nodes (5): Attention scratch, M31-0005 — KV storage and memory ledger, Persistent state and snapshots, Precision decision, Serving KV layout

### Community 855 - "M29-0002 Workstream A — HIP VMM feasibility"
Cohesion: 0.33
Nodes (5): Decision, M29-0002 Workstream A — HIP VMM feasibility, Reproduction, Tested, Timings

### Community 856 - "M29-0003 Workstream B — ContextSpace + logical pages"
Cohesion: 0.33
Nodes (5): Authority, Deliberate non-goals, Evidence, Implemented substrate, M29-0003 Workstream B — ContextSpace + logical pages

### Community 857 - "GemvKernelResources"
Cohesion: 0.33
Nodes (6): GemvKernelResources, local_bytes, max_threads_per_block, registers, shared_bytes, size_t

### Community 858 - "Q4KMmqTile"
Cohesion: 0.33
Nodes (6): uint16_t, Q4KMmqTile, dmin, high, scale_d, values

### Community 859 - "fp16_gemv_reference.cpp"
Cohesion: 0.47
Nodes (5): uint32_t, vector, evaluate_fp16_gemv(), fp16_gemv_cpu_reference(), generate_fp16_gemv_data()

### Community 860 - "prefill_v2_agent_branch_workload_test.cpp"
Cohesion: 0.53
Nodes (5): size_t, main(), now_ms(), options(), require()

### Community 861 - "EventPair"
Cohesion: 0.40
Nodes (4): hipEvent_t, EventPair, start, stop

### Community 862 - "StageEvents"
Cohesion: 0.40
Nodes (3): StageProfile, StageEvents, profile

### Community 863 - "D032 — Persistent Context and Device-Local KV Ownership"
Cohesion: 0.40
Nodes (5): Consequences, D032 — Persistent Context and Device-Local KV Ownership, Decision, Reason, Revisit when

### Community 864 - "M29-0006 physical KV view"
Cohesion: 0.40
Nodes (4): M29-0006 physical KV view, N=1, N=2 same-MI50, Scope boundary

### Community 865 - "M30-0000 canonical agent workload"
Cohesion: 0.40
Nodes (4): Execution entry point, Frozen shape, M30-0000 canonical agent workload, Required per-turn evidence

### Community 866 - "M30-0004 Snapshot, Fork, and Rollback Contract"
Cohesion: 0.40
Nodes (4): API, M30-0004 Snapshot, Fork, and Rollback Contract, Required invariants, State contract

### Community 867 - "M30-0005 Shared Snapshot State and COW Evidence"
Cohesion: 0.40
Nodes (4): Contract and implementation, Gate result, M30-0005 Shared Snapshot State and COW Evidence, Workload

### Community 868 - "M31-0004 — Correctness and validation"
Cohesion: 0.40
Nodes (4): Correctness issue fixed, Fingerprint extension contract, M31-0004 — Correctness and validation, Test status

### Community 869 - "M31-0002T-0001 — First-divergence investigation"
Cohesion: 0.40
Nodes (4): Classification and evidence boundary, Input and generation equivalence, M31-0002T-0001 — First-divergence investigation, Smallest decisive follow-up

### Community 870 - "M31-0002T-0001 — Numerical and state correctness audit"
Cohesion: 0.40
Nodes (4): M31-0002T-0001 — Numerical and state correctness audit, Measured execution path, Recent M31 fixes and applicability, Required correctness evidence

### Community 871 - "M31-0002T-0001 — Reference comparison and regression tests"
Cohesion: 0.40
Nodes (4): M31-0002T-0001 — Reference comparison and regression tests, Minimum follow-up tests, Reference and model identity, Regression coverage and status

### Community 872 - "M31-0002T-0001 — Weight residency and VRAM ledger"
Cohesion: 0.40
Nodes (4): Classification and recovery candidates, M31-0002T-0001 — Weight residency and VRAM ledger, MIInfer persistent weight allocation ledger, Model and observed VRAM reconciliation

### Community 873 - "M31-0003 — Decode and MMQ investigation"
Cohesion: 0.40
Nodes (4): Current path, Decision, Findings, M31-0003 — Decode and MMQ investigation

### Community 874 - "M31-0003 — Host dispatch and synchronization"
Cohesion: 0.40
Nodes (4): Decision, M31-0003 — Host dispatch and synchronization, Production timelines, Repeated boundary checkpoint copies

### Community 875 - "M31-0003 — llama.cpp comparison limits"
Cohesion: 0.40
Nodes (4): Existing measurements are directional, not a controlled A/B, Implementation differences that are safe to state, M31-0003 — llama.cpp comparison limits, Minimum valid future comparison

### Community 876 - "M31-0003 — Prefill and GDN investigation"
Cohesion: 0.40
Nodes (4): Current path, Decision, Evidence and candidate inefficiencies, M31-0003 — Prefill and GDN investigation

### Community 877 - "M31-0005 — Attention kernel audit"
Cohesion: 0.40
Nodes (4): Graph decode kernel, M31-0005 — Attention kernel audit, Separate eager/suffix path, Unproven opportunities

### Community 878 - "M31-0005 — Dispatch and synchronization audit"
Cohesion: 0.40
Nodes (4): Attribution limits, Follow-up, M31-0005 — Dispatch and synchronization audit, Per-token graph sequence

### Community 879 - "Post-M28 production baseline"
Cohesion: 0.40
Nodes (5): Build and correctness checks, Decision, Next frontier, Performance evidence, Post-M28 production baseline

### Community 880 - "M29-0002 Workstream B — 128K memory budget"
Cohesion: 0.40
Nodes (4): Decision, Exact failure, M29-0002 Workstream B — 128K memory budget, Measured allocation families

### Community 881 - "M29-0003 Workstream A — 128K lifetime/allocation fix"
Cohesion: 0.40
Nodes (4): 128K end-to-end result, Change, M29-0003 Workstream A — 128K lifetime/allocation fix, Verification

### Community 882 - "prefill_v2_exact_prefix_reuse_test.cpp"
Cohesion: 0.80
Nodes (4): main(), options(), require(), require_same_tokens()

### Community 883 - "Buffer"
Cohesion: 0.50
Nodes (3): Buffer, pointer, size_t

### Community 884 - "RawBuffer"
Cohesion: 0.50
Nodes (3): size_t, RawBuffer, pointer

### Community 885 - "Buffer"
Cohesion: 0.50
Nodes (3): Buffer, p, size_t

### Community 886 - "GpuBuffer"
Cohesion: 0.50
Nodes (3): size_t, GpuBuffer, ptr

### Community 887 - "GpuBuffer"
Cohesion: 0.50
Nodes (3): size_t, GpuBuffer, ptr

### Community 888 - "M26-D — One evidence-backed floor reduction"
Cohesion: 0.50
Nodes (4): Authorization gate, M26-D — One evidence-backed floor reduction, Scope, Stop condition

### Community 889 - "E2-0003Q source and sampler audit"
Cohesion: 0.50
Nodes (3): E2-0003Q source and sampler audit, Recovered source, Sampler settings

### Community 890 - "ExtractedAssistantOutput"
Cohesion: 0.50
Nodes (4): ExtractedAssistantOutput, content, has_reasoning, reasoning_content

### Community 892 - "D026 — Strongest Available Relevant Baseline Wins"
Cohesion: 0.67
Nodes (3): D026 — Strongest Available Relevant Baseline Wins, Decision, Reason

### Community 893 - "D027 — Serving Does Not Define the Core Runtime"
Cohesion: 0.67
Nodes (3): D027 — Serving Does Not Define the Core Runtime, Decision, Reason

### Community 897 - "PresetFlag"
Cohesion: 0.67
Nodes (3): PresetFlag, name, value

## Knowledge Gaps
- **6792 isolated node(s):** `type_str`, `wave_us`, `mmq_us`, `delta_us`, `extra_mib_per_layer` (+6787 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **76 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RecurrentLayer` connect `RecurrentLayer` to `M23ProfileCounters`, `download`, `UpdateProvenance`, `qwen35_gpu_pipeline.hpp`, `size_t`, `m24_recurrent_layer_bakeoff.cpp`, `Qwen35RuntimeEngine`, `FullAttentionLayer`, `LayerEvaluationResult`, `half`, `GgufTensor`, `Q4KMmqTile`, `Qwen35Model`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Why does `FullAttentionLayer` connect `FullAttentionLayer` to `M23ProfileCounters`, `download`, `qwen35_gpu_pipeline.hpp`, `hipStreamSynchronize`, `size_t`, `Qwen35RuntimeEngine`, `half`, `GgufTensor`, `Q4KMmqTile`, `Qwen35Model`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **Why does `PrefillV2Model` connect `PrefillV2Model` to `PrefillV2Model::snapshot`, `ReusableContext`, `PrefillV2WorkspaceManager`, `AttentionLayerKvCacheStorage`, `RecurrentLayerStateStorage`, `GenerateStats`, `run_bakeoff_main`, `GgufTensor`, `vector`, `GenerateOptions`, `model.cpp`, `PrefillV2Model::PrefillV2Model`?**
  _High betweenness centrality (0.009) - this node is a cross-community bridge._
- **Are the 131 inferred relationships involving `hipMemcpy()` (e.g. with `main()` and `main()`) actually correct?**
  _`hipMemcpy()` has 131 INFERRED edges - model-reasoned connections that need verification._
- **What connects `type_str`, `wave_us`, `mmq_us` to the rest of the system?**
  _6792 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `hardware.md` be split into smaller, more focused modules?**
  _Cohesion score 0.0425531914893617 - nodes in this community are weakly interconnected._
- **Should `architecture.md` be split into smaller, more focused modules?**
  _Cohesion score 0.04878048780487805 - nodes in this community are weakly interconnected._