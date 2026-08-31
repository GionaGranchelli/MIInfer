# Graph Report - mi50  (2026-08-31)

## Corpus Check
- 135 files · ~135,956 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2421 nodes · 3850 edges · 193 communities (171 shown, 22 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 114 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `339ad3c9`
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
- model_plan.cpp
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
- Comparison dimensions
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
- memory_stream_bench.cpp
- graphify reference: query, path, explain
- 43. Initial MIInfer Benchmark Matrix
- Qwen3-8B Dense Control Model
- 1. ggml-org/llama.cpp
- M4 — First Correct End-to-End Generation
- 15. Benchmark Protocol
- 10. Kernel experiments
- 40. Acceptance Categories
- 5. Required Environment Metadata
- D008 — Do Not Build a Generic Graph Runtime Initially
- 15. Research Classification
- 17. Highest-Priority Research Ideas
- 6. nlzy/vllm-gfx906
- DeviceInfo
- 29. Decision
- 6. Baseline
- capture-env.sh
- 17. Correctness requirements
- 3. Fundamental engineering rules
- EXP-0004 — FP16 GEMV K-Split Parallelism
- qwen3_layer35_external_test.cpp
- EXP-0009-kv-geometry.md
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
- 11. joe2gaan/localaiservers
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
- qwen3_decode_sequence_gpu_test.cpp
- extraction-spec.md
- diagnose-gfx802-isolation.sh
- tests/README.md
- 6. Baseline
- 35. Historical failed execution gate — superseded by Section 37
- fp16_gemv_k_split_bench.cpp
- GgufFile
- EXP-0008 — Direct MIInfer vs gfx906 llama.cpp MMVQ
- path_
- qwen3_inference_bench.cpp
- Q8_1Block
- External gfx906 Reference Baseline
- q4_q8_gemv_bench.cpp
- Current Project Status
- Q8BoundaryDiff
- hip_smoke_bench.cpp
- fp16_gemv_reduction_diag.cpp
- qwen3_layer_gpu_impl
- GemvShape
- qwen3_fast_decode_bench.cpp
- Qwen3LayerWeights
- qwen3_gpu_layer.cpp
- qwen3_primitives.cpp
- Qwen3GpuProfile
- m4b-layer6/README.md
- M3 Minimal Qwen3-8B Runtime Scaffold
- run_sequence
- Qwen3LayerTrace
- Qwen3GpuPlan
- qwen3_layer6_external_test.cpp
- Metrics
- Qwen3Model
- qwen3_layer_host_impl
- M4-C1 — Deterministic first generated token
- Metrics
- m4a4-four-position/README.md
- RmsVariant
- M4-B — Full Qwen3 single-token forward
- m4b-single-token-legacy/README.md
- fail
- m4b-single-token/README.md
- Qwen3ForwardTrace
- run-m4c1-acceptance.sh
- run-m4b-acceptance.sh
- run-m4c2-acceptance.sh
- AttentionPathReplay
- m4b-layer35/README.md
- vector
- qwen3_decode_profile.cpp
- qwen3_gpu_layer.hpp
- Qwen3Layer0KvCache
- qwen3_position_audit.cpp
- Q6KDeviceBlock
- string
- qwen3_layer0_gpu_test.cpp
- DeviceShapeData
- M5-A — Reproducible MI50 inference baseline
- M4-C3 — Text-facing greedy generation
- run-m4c3-acceptance.sh
- FfnTailReplay
- Checkpoint
- gguf.cpp
- EXP-0010 — Qwen3-8B steady-state decode profile
- require_tensor
- m4c3-text/README.md
- GemvKernelResources
- run-m5a-baseline.sh
- EXP-0012 — Qwen3-8B Q4_0 MI50 comparison
- fp16_gemv.hpp
- Fp16GemvMetrics
- EXP-0013 — Qwen3 position-scaled execution audit
- run-m5b-profile.sh
- gguf_tensor_byte_size
- qwen3_primitives_test.cpp
- EXP-0011 — Trace-free Qwen3-8B decode control
- replay_projection
- Qwen3GpuProfileEvent
- Qwen3TensorView
- hip_check.hpp
- run-m5c0-fast-decode.sh

## God Nodes (most connected - your core abstractions)
1. `Qwen3LayerTrace` - 55 edges
2. `Qwen3GpuPlan` - 46 edges
3. `Qwen3Model` - 45 edges
4. `path_` - 38 edges
5. `Qwen3GpuProfile` - 31 edges
6. `M4-B — Full Qwen3 single-token forward` - 29 edges
7. `GgufFile` - 28 edges
8. `qwen3_layer_host_impl()` - 27 edges
9. `qwen3_layer_gpu_impl()` - 24 edges
10. `Qwen3Layer0GpuKvCache` - 20 edges

## Surprising Connections (you probably didn't know these)
- `run_sequence()` --calls--> `snapshot_keys`  [INFERRED]
  tests/qwen3_kv_cache_gpu_test.cpp → include/miinfer/qwen3_gpu_layer.hpp
- `run_sequence()` --calls--> `snapshot_values`  [INFERRED]
  tests/qwen3_kv_cache_gpu_test.cpp → include/miinfer/qwen3_gpu_layer.hpp
- `Qwen3DecodeCache::reset()` --calls--> `reset`  [INFERRED]
  src/qwen3_layer.cpp → include/miinfer/qwen3_layer.hpp
- `run_quantize()` --references--> `Q8_1Block`  [INFERRED]
  bench/q4_q8_gemv_bench.cpp → include/miinfer/q4_q8_gemv.hpp
- `run_fanout()` --references--> `GemvShape`  [INFERRED]
  bench/q4_q8_gemv_bench.cpp → include/miinfer/fp16_gemv.hpp

## Import Cycles
- None detected.

## Communities (193 total, 22 thin omitted)

### Community 0 - "hardware.md"
Cohesion: 0.04
Nodes (46): 10. Candidate Quantized Execution Path, 11. FP16 Behavior, 12. BF16, 13. Memory Bandwidth, 14. HBM vs Cache, 15. Weight Compression, 16. HBM Clock, 17. GPU Clock (+38 more)

### Community 1 - "architecture.md"
Cohesion: 0.05
Nodes (43): 10. Memory Architecture, 11. Weight Residency, 12. Tensor Layout, 13. Prefill vs Decode, 14. Context-Length Sensitivity, 15. Attention Architecture, 16. MoE Architecture, 17. Static Model Knowledge (+35 more)

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
Cohesion: 0.09
Nodes (22): Current Benchmark Priority, Current Build Direction, Current Correctness Policy, Current Dependency Policy, Current Experiment Queue, Current Hardware Observation Requirements, Current Hardware Target, Current Performance Policy (+14 more)

### Community 7 - "AGENTS.md"
Cohesion: 0.07
Nodes (27): 11. Static specialization, 12. Static kernel selection, 13. HIP graph strategy, 14. Benchmarking, 15. Hardware-state validation, 16. Benchmark methodology, 18. Experiment records, 19. Profiling (+19 more)

### Community 8 - "TEMPLATE.md"
Cohesion: 0.07
Nodes (27): 11. Model / Workload, 12. Test Matrix, 14. Correctness Results, 16. Pre-Run Hardware State, 19. Per-Shape Results, 1. Question, 20. Effective Bandwidth, 21. Resource Usage (+19 more)

### Community 9 - "model_plan.cpp"
Cohesion: 0.07
Nodes (36): GpuWeightArena, allocate, release, upload, GgufTensorType, size_t, string, uint64_t (+28 more)

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
Nodes (12): D023 — Triton Is a Research Tool, Not a Required Runtime, D024 — Benchmarkability Is an Architectural Requirement, D027 — Serving Does Not Define the Core Runtime, Decision, Decision, Decision, Decision Change Process, Guiding Rule (+4 more)

### Community 20 - "7. Neroued/ninfer"
Cohesion: 0.17
Nodes (12): 7. Neroued/ninfer, Fixed memory planning, Graph-based decode, MIInfer implication, MIInfer implication, MIInfer implication, MIInfer implication, Packed artifact (+4 more)

### Community 21 - "Comparison dimensions"
Cohesion: 0.17
Nodes (12): Comparison dimensions, Comparison rules, Context regimes, Decode, Exit criteria, Goal, H0 supported, H1 supported (+4 more)

### Community 22 - "4. mxxm-t/mx-llama.cpp"
Cohesion: 0.20
Nodes (10): 4. mxxm-t/mx-llama.cpp, Activation reuse, MIInfer implication, MIInfer implication, MIInfer implication, MXFP4, Role, Weight repacking (+2 more)

### Community 23 - "roadmap.md"
Cohesion: 0.12
Nodes (13): Benchmark before claim, Correctness before speed, Current Execution Order, Current Status, Deferred / Explicitly Out of Scope, Evidence before architecture, Immediate Next Milestone, MIInfer Roadmap (+5 more)

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

### Community 36 - "memory_stream_bench.cpp"
Cohesion: 0.17
Nodes (16): size_t, string, vector, escape(), main(), median(), Options, bytes (+8 more)

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

### Community 41 - "M4 — First Correct End-to-End Generation"
Cohesion: 0.33
Nodes (6): Correctness validation, Exit criteria, Goal, Initial inference mode, M4 — First Correct End-to-End Generation, Required execution pieces

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

### Community 50 - "DeviceInfo"
Cohesion: 0.24
Nodes (12): DeviceInfo, architecture, index, name, total_vram_bytes, size_t, ostream, string (+4 more)

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
Cohesion: 0.31
Nodes (20): compare_q8_blocks(), vector, dequantize_q8_exact(), expand_gqa(), fp16_roundtrip(), host_swiglu(), main(), read_f32() (+12 more)

### Community 58 - "EXP-0009-kv-geometry.md"
Cohesion: 0.14
Nodes (13): 10. Comparison with MMVQ, 11. Decision, 12. Next experiment, 1. Question, 2. Hypothesis, 3. Prior evidence, 4. Candidates, 5. Environment and method (+5 more)

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

### Community 86 - "11. joe2gaan/localaiservers"
Cohesion: 0.50
Nodes (4): 11. joe2gaan/localaiservers, Important dot-product lesson, MIInfer implication, Role

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
Cohesion: 0.33
Nodes (5): Benchmarks, End-to-end M5-A baseline, M5-B steady-state decode profile, M5-C0 trace-free decode benchmark, M5-C1 position-scaled execution audit

### Community 108 - "EXP-0007 — gfx906 Zero-Point-Corrected Q4_0 × Q8_1 Dot4"
Cohesion: 0.12
Nodes (16): 10. Hardware validity, 11. External MMVQ comparison, 12. Projection-only sanity check, 13. Decision, 14. M2 status, 15. Next experiment, 1. Question, 2. Hypothesis and motivation (+8 more)

### Community 109 - "qwen3_tokenizer.cpp"
Cohesion: 0.09
Nodes (41): size_t, string, uint32_t, unordered_map, vector, Qwen3Tokenizer, decode, encode (+33 more)

### Community 110 - "qwen3_decode_sequence_gpu_test.cpp"
Cohesion: 0.11
Nodes (37): Qwen3GpuDecodeCache, caches_, length, reset, reset, argmax(), cache_lengths(), size_t (+29 more)

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

### Community 119 - "GgufFile"
Cohesion: 0.08
Nodes (28): GgufFile, file_descriptor_, mapping_, metadata_array_is_string, metadata_array_size, metadata_float, metadata_string, metadata_unsigned (+20 more)

### Community 120 - "EXP-0008 — Direct MIInfer vs gfx906 llama.cpp MMVQ"
Cohesion: 0.12
Nodes (15): 10. Reference ISA and resources, 11. Architectural differences relevant to K/V, 12. M2 decision, 13.1 Re-evaluation after EXP-0009, 13. Next experiment, 1. Question, 2. Hypothesis and motivation, 3. Reference path (+7 more)

### Community 121 - "path_"
Cohesion: 0.09
Nodes (55): path_, argmax(), capture_q8_input(), compare_checkpoint(), size_t, vector, exact_equal(), main() (+47 more)

### Community 122 - "qwen3_inference_bench.cpp"
Cohesion: 0.09
Nodes (48): argmax(), build_json(), ostream, size_t, string, timespec, uint32_t, vector (+40 more)

### Community 123 - "Q8_1Block"
Cohesion: 0.12
Nodes (23): __half, int16_t, int8_t, uint8_t, Q4_0Block, d, qs, Q8_1Block (+15 more)

### Community 124 - "External gfx906 Reference Baseline"
Cohesion: 0.22
Nodes (8): Baseline status, Checkout, External gfx906 Reference Baseline, Initial baseline, MI50 build starting point, Option validation on the available host, Pin, Toolchain preflight record

### Community 125 - "q4_q8_gemv_bench.cpp"
Cohesion: 0.16
Nodes (28): allocate_shape(), ostream, string, vector, escape(), free_shape(), main(), median() (+20 more)

### Community 126 - "Current Project Status"
Cohesion: 0.50
Nodes (4): Completed, Completed in Task 3, Current Project Status, Not implemented

### Community 127 - "Q8BoundaryDiff"
Cohesion: 0.20
Nodes (10): uint16_t, half_bits(), Q8BoundaryDiff, different_lane_values, different_scale_blocks, different_sum_blocks, first_block, first_lane (+2 more)

### Community 128 - "hip_smoke_bench.cpp"
Cohesion: 0.20
Nodes (15): size_t, string, json_escape(), main(), Options, device, elements, iterations (+7 more)

### Community 129 - "fp16_gemv_reduction_diag.cpp"
Cohesion: 0.17
Nodes (17): __half, string, vector, main(), median(), nonnegative(), Options, device (+9 more)

### Community 130 - "qwen3_layer_gpu_impl"
Cohesion: 0.18
Nodes (16): size_t, Qwen3Layer0GpuKvCache, append, keys_, reset, snapshot_keys, snapshot_values, values_ (+8 more)

### Community 131 - "GemvShape"
Cohesion: 0.24
Nodes (10): GemvShape, id, k, m, projection, check_output(), __half, vector (+2 more)

### Community 132 - "qwen3_fast_decode_bench.cpp"
Cohesion: 0.10
Nodes (43): argmax(), build_json(), ostream, size_t, string, timespec, uint32_t, vector (+35 more)

### Community 133 - "Qwen3LayerWeights"
Cohesion: 0.08
Nodes (24): uint32_t, Qwen3Config, attention_heads, context_length, head_dim, hidden_size, intermediate_size, kv_heads (+16 more)

### Community 134 - "qwen3_gpu_layer.cpp"
Cohesion: 0.13
Nodes (26): Qwen3Projection, Qwen3ProjectionPrecision, capture(), copy_to_host(), __half, size_t, span, T (+18 more)

### Community 135 - "qwen3_primitives.cpp"
Cohesion: 0.11
Nodes (37): int16_t, int8_t, uint16_t, uint8_t, Q4_0HostBlock, d_bits, qs, Q6KHostBlock (+29 more)

### Community 136 - "Qwen3GpuProfile"
Cohesion: 0.07
Nodes (29): Function, qwen3_profile_category_count, Qwen3GpuProfile, copy_bytes, copy_ms, deferred_timing, dispatches, finalization_synchronizations (+21 more)

### Community 138 - "M3 Minimal Qwen3-8B Runtime Scaffold"
Cohesion: 0.25
Nodes (7): GPU ownership and plan, M3 Minimal Qwen3-8B Runtime Scaffold, Parser boundary, Static projection kernel selection, Supported artifact, Validated configuration, Validation command

### Community 139 - "run_sequence"
Cohesion: 0.28
Nodes (17): attention_contract(), cache_corruption_test(), cache_slots_preserved(), checkpoints(), compare_trace(), size_t, span, string (+9 more)

### Community 140 - "Qwen3LayerTrace"
Cohesion: 0.06
Nodes (31): Qwen3LayerTrace, attention_output, attention_probabilities, attention_scores, attn_norm, attn_rms, embedding, ffn_input (+23 more)

### Community 141 - "Qwen3GpuPlan"
Cohesion: 0.19
Nodes (18): Qwen3GpuPlan, buffers_, build, device_, device_tensor_data, kernel_for, model_, tensors_ (+10 more)

### Community 142 - "qwen3_layer6_external_test.cpp"
Cohesion: 0.12
Nodes (22): Checkpoint, file, miinfer, name, tolerance, compare_authority(), compare_host_gpu(), size_t (+14 more)

### Community 143 - "Metrics"
Cohesion: 0.22
Nodes (9): Metrics, actual_at_max, expected_at_max, finite, max_abs, max_index, max_rel, mean_abs (+1 more)

### Community 144 - "Qwen3Model"
Cohesion: 0.14
Nodes (13): shared_ptr, size_t, string, vector, Qwen3Model, artifact_path_, config_, final_norm_ (+5 more)

### Community 145 - "qwen3_layer_host_impl"
Cohesion: 0.16
Nodes (30): add_in_place(), int8_t, size_t, span, uint16_t, uint32_t, vector, execute_qwen3_decode_host() (+22 more)

### Community 146 - "M4-C1 — Deterministic first generated token"
Cohesion: 0.12
Nodes (13): Acceptance, Decision, M4-C2 — Short deterministic greedy decode sequence, Next slice, Pinned sequence, Acceptance, Decode contract, Deterministic fixture (+5 more)

### Community 147 - "Metrics"
Cohesion: 0.13
Nodes (26): Checkpoint, abs_tolerance, actual, file_index, name, rel_tolerance, compare(), size_t (+18 more)

### Community 149 - "RmsVariant"
Cohesion: 0.19
Nodes (13): RmsReduction, span, f32_tensor(), quantize_gpu_q8_exact(), replay_rms_norm(), report_pre_ffn_contract(), RmsVariant, inverse (+5 more)

### Community 150 - "M4-B — Full Qwen3 single-token forward"
Cohesion: 0.07
Nodes (29): Acceptance target, Current evidence, Implemented slice, Independent reference, M4-B10 layer-35 Gate/Up projection isolation, M4-B11 Q8 identity and CPU accumulation contract, M4-B12 pre-FFN residual and RMSNorm isolation, M4-B13 attention RMSNorm, V, and position-zero GQA isolation (+21 more)

### Community 152 - "fail"
Cohesion: 0.24
Nodes (16): align_up(), checked_add(), checked_size(), byte, GgufScalar, shared_ptr, size_t, uint32_t (+8 more)

### Community 154 - "Qwen3ForwardTrace"
Cohesion: 0.29
Nodes (6): vector, Qwen3ForwardTrace, embedding, final_norm, layer_outputs, logits

### Community 158 - "AttentionPathReplay"
Cohesion: 0.33
Nodes (6): AttentionPathReplay, attention_output, ffn_input, ffn_norm, layer_output, v

### Community 160 - "vector"
Cohesion: 0.06
Nodes (53): vector, array, array, byte, size_t, string, uint32_t, uint64_t (+45 more)

### Community 161 - "qwen3_decode_profile.cpp"
Cohesion: 0.19
Nodes (18): build_json(), string, timespec, uint32_t, elapsed_ms(), json_escape(), main(), now() (+10 more)

### Community 162 - "qwen3_gpu_layer.hpp"
Cohesion: 0.16
Nodes (13): vector, Qwen3DownProjectionContractTrace, current_s_correction, direct_signed_oracle, exact_sum_correction, Qwen3FfnProbeTrace, ffn_output, gate (+5 more)

### Community 163 - "Qwen3Layer0KvCache"
Cohesion: 0.14
Nodes (21): size_t, span, Qwen3DecodeCache, caches_, length, Qwen3Layer0KvCache, append, reset (+13 more)

### Community 164 - "qwen3_position_audit.cpp"
Cohesion: 0.11
Nodes (33): category_index(), array, ostream, qwen3_profile_category_count, Qwen3ProfileCategory, size_t, string, timespec (+25 more)

### Community 165 - "Q6KDeviceBlock"
Cohesion: 0.15
Nodes (13): __half, int16_t, int8_t, uint8_t, Q6KDeviceBlock, d, qh, ql (+5 more)

### Community 167 - "qwen3_layer0_gpu_test.cpp"
Cohesion: 0.26
Nodes (16): abs_tolerance(), checkpoints(), compare(), size_t, string, vector, main(), Metrics (+8 more)

### Community 168 - "DeviceShapeData"
Cohesion: 0.15
Nodes (13): __half, DeviceShapeData, device_input_fp16, device_input_q8, device_output, device_weights, input_fp16, input_q8 (+5 more)

### Community 169 - "M5-A — Reproducible MI50 inference baseline"
Cohesion: 0.20
Nodes (9): Decision, Environment, Interpretation, M5-A — Reproducible MI50 inference baseline, Question, Reproduction, Result, Scope (+1 more)

### Community 170 - "M4-C3 — Text-facing greedy generation"
Cohesion: 0.33
Nodes (5): CLI, Decision, M4-C3 — Text-facing greedy generation, Pinned physical acceptance, Tokenizer contract

### Community 172 - "FfnTailReplay"
Cohesion: 0.33
Nodes (6): FfnTailReplay, down, gate, layer_output, swiglu, up

### Community 173 - "Checkpoint"
Cohesion: 0.40
Nodes (5): Checkpoint, file, miinfer, name, tolerance

### Community 174 - "gguf.cpp"
Cohesion: 0.29
Nodes (12): GgufError, metadata, runtime_error, string, T, GgufFile::metadata(), GgufFile::metadata_array_is_string(), GgufFile::metadata_array_size() (+4 more)

### Community 175 - "EXP-0010 — Qwen3-8B steady-state decode profile"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0010 — Qwen3-8B steady-state decode profile, Follow-up (+4 more)

### Community 176 - "require_tensor"
Cohesion: 0.35
Nodes (12): initializer_list, add_bytes(), GgufTensorType, size_t, string, uint32_t, uint64_t, narrow_u32() (+4 more)

### Community 178 - "GemvKernelResources"
Cohesion: 0.33
Nodes (6): GemvKernelResources, local_bytes, max_threads_per_block, registers, shared_bytes, size_t

### Community 180 - "EXP-0012 — Qwen3-8B Q4_0 MI50 comparison"
Cohesion: 0.17
Nodes (11): Closest raw-token continuation controls, Decision, Environment, EXP-0012 — Qwen3-8B Q4_0 MI50 comparison, Follow-up, Growing-context continuation, Hypothesis, Interpretation (+3 more)

### Community 181 - "fp16_gemv.hpp"
Cohesion: 0.27
Nodes (6): RocblasGemmHandle, opaque, string, main(), run_shape(), run_zero_point_identity_tests()

### Community 182 - "Fp16GemvMetrics"
Cohesion: 0.17
Nodes (14): Fp16GemvMetrics, cosine_similarity, inf_detected, max_abs_error, max_relative_error, mean_abs_error, nan_detected, pass (+6 more)

### Community 183 - "EXP-0013 — Qwen3 position-scaled execution audit"
Cohesion: 0.20
Nodes (9): Decision, Environment, EXP-0013 — Qwen3 position-scaled execution audit, Follow-up, Hypothesis, Implementation, Interpretation, Results (+1 more)

### Community 185 - "gguf_tensor_byte_size"
Cohesion: 0.25
Nodes (11): checked_mul(), GgufTensorType, uint64_t, vector, elements(), gguf_tensor_byte_size(), gguf_tensor_type_name(), layout() (+3 more)

### Community 186 - "qwen3_primitives_test.cpp"
Cohesion: 0.43
Nodes (7): close_enough(), T, vector, device_copy(), gpu_tests(), host_tests(), main()

### Community 187 - "EXP-0011 — Trace-free Qwen3-8B decode control"
Cohesion: 0.15
Nodes (12): Baseline, Benchmark, Candidate, Correctness, Decision, Environment, EXP-0011 — Trace-free Qwen3-8B decode control, Follow-up (+4 more)

### Community 188 - "replay_projection"
Cohesion: 0.25
Nodes (11): AccumulationContract, int16_t, int8_t, size_t, Q8ReplayBlock, d_bits, qs, sum (+3 more)

### Community 189 - "Qwen3GpuProfileEvent"
Cohesion: 0.20
Nodes (9): hipEvent_t, Qwen3ProfileCategory, Qwen3GpuProfileEvent, bytes, category, copy, dispatches, start (+1 more)

### Community 190 - "Qwen3TensorView"
Cohesion: 0.33
Nodes (4): byte, GgufTensorType, Qwen3TensorView, source

### Community 191 - "hip_check.hpp"
Cohesion: 0.47
Nodes (3): hip_check(), hip_check_failed(), hipError_t

## Knowledge Gaps
- **1307 isolated node(s):** `experiment`, `shape`, `implementation`, `cache_regime`, `custom_label` (+1302 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Qwen3Model` connect `Qwen3Model` to `qwen3_decode_profile.cpp`, `qwen3_fast_decode_bench.cpp`, `Qwen3LayerWeights`, `model_plan.cpp`, `Qwen3GpuPlan`, `qwen3_tokenizer.cpp`, `require_tensor`, `qwen3_layer_host_impl`, `RmsVariant`, `GgufFile`, `qwen3_layer35_external_test.cpp`, `qwen3_inference_bench.cpp`, `replay_projection`, `Qwen3TensorView`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Why does `Qwen3GpuPlan` connect `Qwen3GpuPlan` to `qwen3_decode_profile.cpp`, `qwen3_layer_gpu_impl`, `qwen3_fast_decode_bench.cpp`, `qwen3_gpu_layer.cpp`, `model_plan.cpp`, `qwen3_layer6_external_test.cpp`, `Qwen3Model`, `DeviceInfo`, `path_`, `qwen3_inference_bench.cpp`, `qwen3_layer35_external_test.cpp`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Why does `GgufFile` connect `GgufFile` to `vector`, `gguf.cpp`, `Qwen3Model`, `require_tensor`, `fail`, `path_`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **What connects `experiment`, `shape`, `implementation` to the rest of the system?**
  _1307 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `hardware.md` be split into smaller, more focused modules?**
  _Cohesion score 0.0425531914893617 - nodes in this community are weakly interconnected._
- **Should `architecture.md` be split into smaller, more focused modules?**
  _Cohesion score 0.045454545454545456 - nodes in this community are weakly interconnected._
- **Should `benchmarking.md` be split into smaller, more focused modules?**
  _Cohesion score 0.045454545454545456 - nodes in this community are weakly interconnected._