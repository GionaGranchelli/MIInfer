# Graph Report - mi50  (2026-08-30)

## Corpus Check
- 83 files · ~80,260 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1495 nodes · 1895 edges · 142 communities (132 shown, 10 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 33 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f962defe`
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
- Roadmap Principles
- 29. Decision
- 6. Baseline
- capture-env.sh
- 17. Correctness requirements
- 3. Fundamental engineering rules
- EXP-0004 — FP16 GEMV K-Split Parallelism
- hip_smoke_bench.cpp
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
- fp16_gemv_reduction_diag.cpp
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
- bench/README.md
- EXP-0007 — gfx906 Zero-Point-Corrected Q4_0 × Q8_1 Dot4
- DeviceInfo
- Qwen3GpuPlan
- extraction-spec.md
- diagnose-gfx802-isolation.sh
- tests/README.md
- 6. Baseline
- 35. Historical failed execution gate — superseded by Section 37
- fp16_gemv_k_split_bench.cpp
- gguf.cpp
- EXP-0008 — Direct MIInfer vs gfx906 llama.cpp MMVQ
- Sha256
- GgufFile
- Q8_1Block
- External gfx906 Reference Baseline
- q4_q8_gemv_bench.cpp
- Current Project Status
- Current Scope
- Qwen3LayerWeights
- Qwen3Model
- require_tensor
- GemvShape
- vector
- Qwen3Config
- DeviceShapeData
- string
- Options
- Options
- M3 Minimal Qwen3-8B Runtime Scaffold
- GemvKernelResources
- gguf.hpp
- hip_check.hpp

## God Nodes (most connected - your core abstractions)
1. `GgufFile` - 27 edges
2. `Qwen3Model` - 25 edges
3. `Qwen3GpuPlan` - 22 edges
4. `fail()` - 19 edges
5. `MIInfer` - 19 edges
6. `Options` - 18 edges
7. `EXP-0004 — FP16 GEMV K-Split Parallelism` - 18 edges
8. `EXP-0006 — gfx906 Q4_0 × Q8_1 Packed-Dot Specialization` - 18 edges
9. `DeviceShapeData` - 17 edges
10. `DeviceInfo` - 17 edges

## Surprising Connections (you probably didn't know these)
- `run_quantize()` --references--> `Q8_1Block`  [INFERRED]
  bench/q4_q8_gemv_bench.cpp → include/miinfer/q4_q8_gemv.hpp
- `run_fanout()` --references--> `GemvShape`  [INFERRED]
  bench/q4_q8_gemv_bench.cpp → include/miinfer/fp16_gemv.hpp
- `run_fanout()` --references--> `Q4_0Block`  [INFERRED]
  bench/q4_q8_gemv_bench.cpp → include/miinfer/q4_q8_gemv.hpp
- `run_fanout()` --references--> `Q8_1Block`  [INFERRED]
  bench/q4_q8_gemv_bench.cpp → include/miinfer/q4_q8_gemv.hpp
- `add_bytes()` --calls--> `GgufError`  [INFERRED]
  src/qwen3_model.cpp → include/miinfer/gguf.hpp

## Import Cycles
- None detected.

## Communities (142 total, 10 thin omitted)

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
Cohesion: 0.10
Nodes (19): Current Benchmark Priority, Current Build Direction, Current Correctness Policy, Current Dependency Policy, Current Experiment Queue, Current Hardware Observation Requirements, Current Hardware Target, Current Performance Policy (+11 more)

### Community 7 - "AGENTS.md"
Cohesion: 0.07
Nodes (27): 11. Static specialization, 12. Static kernel selection, 13. HIP graph strategy, 14. Benchmarking, 15. Hardware-state validation, 16. Benchmark methodology, 18. Experiment records, 19. Profiling (+19 more)

### Community 8 - "TEMPLATE.md"
Cohesion: 0.07
Nodes (27): 11. Model / Workload, 12. Test Matrix, 14. Correctness Results, 16. Pre-Run Hardware State, 19. Per-Shape Results, 1. Question, 20. Effective Bandwidth, 21. Resource Usage (+19 more)

### Community 9 - "Fp16GemvMetrics"
Cohesion: 0.17
Nodes (14): Fp16GemvMetrics, cosine_similarity, inf_detected, max_abs_error, max_relative_error, mean_abs_error, nan_detected, pass (+6 more)

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
Cohesion: 0.20
Nodes (8): Current Execution Order, Current Status, Deferred / Explicitly Out of Scope, Immediate Next Milestone, MIInfer Roadmap, Milestone Dependencies, Roadmap Change Policy, Success Definition

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

### Community 50 - "Roadmap Principles"
Cohesion: 0.40
Nodes (5): Benchmark before claim, Correctness before speed, Evidence before architecture, Narrow before broad, Roadmap Principles

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

### Community 57 - "hip_smoke_bench.cpp"
Cohesion: 0.20
Nodes (15): size_t, string, json_escape(), main(), Options, device, elements, iterations (+7 more)

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

### Community 92 - "fp16_gemv_reduction_diag.cpp"
Cohesion: 0.17
Nodes (17): __half, string, vector, main(), median(), nonnegative(), Options, device (+9 more)

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

### Community 108 - "EXP-0007 — gfx906 Zero-Point-Corrected Q4_0 × Q8_1 Dot4"
Cohesion: 0.12
Nodes (16): 10. Hardware validity, 11. External MMVQ comparison, 12. Projection-only sanity check, 13. Decision, 14. M2 status, 15. Next experiment, 1. Question, 2. Hypothesis and motivation (+8 more)

### Community 109 - "DeviceInfo"
Cohesion: 0.24
Nodes (12): DeviceInfo, architecture, index, name, total_vram_bytes, size_t, ostream, string (+4 more)

### Community 110 - "Qwen3GpuPlan"
Cohesion: 0.06
Nodes (52): GpuWeightArena, allocate, release, upload, GgufTensorType, size_t, string, uint64_t (+44 more)

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
Cohesion: 0.29
Nodes (11): ostream, string, vector, escape(), main(), median(), nonnegative(), parse() (+3 more)

### Community 119 - "gguf.cpp"
Cohesion: 0.13
Nodes (38): GgufError, metadata, runtime_error, align_up(), checked_add(), checked_mul(), checked_size(), byte (+30 more)

### Community 120 - "EXP-0008 — Direct MIInfer vs gfx906 llama.cpp MMVQ"
Cohesion: 0.12
Nodes (15): 10. Reference ISA and resources, 11. Architectural differences relevant to K/V, 12. M2 decision, 13.1 Re-evaluation after EXP-0009, 13. Next experiment, 1. Question, 2. Hypothesis and motivation, 3. Reference path (+7 more)

### Community 121 - "Sha256"
Cohesion: 0.12
Nodes (25): array, path_, byte, size_t, string, uint32_t, uint64_t, rotate_right() (+17 more)

### Community 122 - "GgufFile"
Cohesion: 0.10
Nodes (22): GgufFile, file_descriptor_, mapping_, metadata_array_size, metadata_float, metadata_string, metadata_unsigned, open (+14 more)

### Community 123 - "Q8_1Block"
Cohesion: 0.14
Nodes (19): __half, launch_selected_gemv(), __half, uint8_t, Q4_0Block, d, qs, Q8_1Block (+11 more)

### Community 124 - "External gfx906 Reference Baseline"
Cohesion: 0.22
Nodes (8): Baseline status, Checkout, External gfx906 Reference Baseline, Initial baseline, MI50 build starting point, Option validation on the available host, Pin, Toolchain preflight record

### Community 125 - "q4_q8_gemv_bench.cpp"
Cohesion: 0.29
Nodes (18): ostream, string, vector, escape(), free_shape(), main(), median(), nonnegative() (+10 more)

### Community 126 - "Current Project Status"
Cohesion: 0.50
Nodes (4): Completed, Completed in Task 3, Current Project Status, Not implemented

### Community 127 - "Current Scope"
Cohesion: 0.67
Nodes (3): Current Scope, In scope now, Not in scope now

### Community 128 - "Qwen3LayerWeights"
Cohesion: 0.11
Nodes (16): byte, GgufTensorType, Qwen3LayerWeights, attention_norm, down, ffn_norm, gate, k (+8 more)

### Community 129 - "Qwen3Model"
Cohesion: 0.14
Nodes (13): shared_ptr, size_t, string, vector, Qwen3Model, artifact_path_, config_, final_norm_ (+5 more)

### Community 130 - "require_tensor"
Cohesion: 0.35
Nodes (12): initializer_list, add_bytes(), GgufTensorType, size_t, string, uint32_t, uint64_t, narrow_u32() (+4 more)

### Community 131 - "GemvShape"
Cohesion: 0.21
Nodes (11): allocate_shape(), GemvShape, id, k, m, projection, check_output(), __half (+3 more)

### Community 132 - "vector"
Cohesion: 0.27
Nodes (7): RocblasGemmHandle, opaque, vector, string, main(), run_shape(), run_zero_point_identity_tests()

### Community 133 - "Qwen3Config"
Cohesion: 0.17
Nodes (12): uint32_t, Qwen3Config, attention_heads, context_length, head_dim, hidden_size, intermediate_size, kv_heads (+4 more)

### Community 134 - "DeviceShapeData"
Cohesion: 0.18
Nodes (11): DeviceShapeData, device_input_fp16, device_input_q8, device_output, device_weights, input_fp16, input_q8, oracle_fp16 (+3 more)

### Community 136 - "Options"
Cohesion: 0.22
Nodes (9): uint32_t, Options, device, iterations, json_output, seed, shape, warmup (+1 more)

### Community 137 - "Options"
Cohesion: 0.22
Nodes (9): Options, device, implementation, iterations, json_output, length, mode, shape (+1 more)

### Community 138 - "M3 Minimal Qwen3-8B Runtime Scaffold"
Cohesion: 0.25
Nodes (7): GPU ownership and plan, M3 Minimal Qwen3-8B Runtime Scaffold, Parser boundary, Static projection kernel selection, Supported artifact, Validated configuration, Validation command

### Community 139 - "GemvKernelResources"
Cohesion: 0.33
Nodes (6): GemvKernelResources, local_bytes, max_threads_per_block, registers, shared_bytes, size_t

### Community 140 - "gguf.hpp"
Cohesion: 0.40
Nodes (5): GgufValue, value, GgufScalar, unordered_map, variant

### Community 141 - "hip_check.hpp"
Cohesion: 0.47
Nodes (3): hip_check(), hip_check_failed(), hipError_t

## Knowledge Gaps
- **952 isolated node(s):** `experiment`, `shape`, `implementation`, `cache_regime`, `custom_label` (+947 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GgufFile` connect `GgufFile` to `Qwen3Model`, `require_tensor`, `vector`, `gguf.hpp`, `gguf.cpp`, `Sha256`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Why does `MIInfer` connect `MIInfer` to `roadmap.md`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **What connects `experiment`, `shape`, `implementation` to the rest of the system?**
  _952 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `hardware.md` be split into smaller, more focused modules?**
  _Cohesion score 0.0425531914893617 - nodes in this community are weakly interconnected._
- **Should `architecture.md` be split into smaller, more focused modules?**
  _Cohesion score 0.045454545454545456 - nodes in this community are weakly interconnected._
- **Should `benchmarking.md` be split into smaller, more focused modules?**
  _Cohesion score 0.045454545454545456 - nodes in this community are weakly interconnected._
- **Should `EXP-0005 — Q4_0 × Q8_1 Quantized GEMV Baseline` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._