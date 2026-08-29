# MIInfer Hardware Target

This document defines the initial hardware contract for MIInfer.

MIInfer is not a generic AMD GPU inference runtime.

The first supported target is:

> **AMD Instinct MI50 32GB, gfx906 / Vega20**

All early architecture and kernel decisions may assume this target unless explicitly documented otherwise.

---

# 1. Initial Supported Device

Primary target:

```text
GPU: AMD Instinct MI50
Architecture: Vega20
ISA target: gfx906
VRAM: 32 GB HBM2
Execution model: Wave64
Initial topology: single GPU
Platform: Linux
Programming model: HIP
```

MIInfer should fail clearly on unsupported devices rather than silently selecting a generic fallback.

---

# 2. Hardware Scope

The following are considered in-scope for the initial project:

* AMD Instinct MI50 32GB
* gfx906
* Vega20
* single-device execution

The following are initially out of scope:

* Radeon VII
* MI50 16GB variants
* MI60
* gfx900
* gfx908
* gfx90a
* CDNA
* RDNA
* MI200
* MI300
* NVIDIA
* Intel GPU
* CPU execution as a performance backend

Other gfx906 devices may be useful for research, but they must not automatically be treated as equivalent to the initial MI50 target.

---

# 3. Architectural Characteristics

The initial target provides approximately:

| Property                  | MI50 target |
| ------------------------- | ----------: |
| Architecture              |      Vega20 |
| ISA                       |      gfx906 |
| Compute Units             |          60 |
| Stream processors         |        3840 |
| Wave size                 |          64 |
| VRAM                      |  32 GB HBM2 |
| Memory bus                |    4096-bit |
| Peak memory bandwidth     |     ~1 TB/s |
| FP32 peak                 |  ~13 TFLOPS |
| FP16 peak                 |  ~26 TFLOPS |
| Typical board power class |      ~300 W |

Exact frequency-dependent peak values are not treated as benchmark targets.

Measured application performance is authoritative.

---

# 4. Wave64

gfx906 executes 64 lanes per hardware wave.

MIInfer may assume:

```text
hardware_wave_size = 64
```

for architecture-specific kernels.

However, logical cooperative groups may still use subsets of the physical wave.

Examples:

```text
1 × 64-lane group
2 × 32-lane groups
4 × 16-lane groups
```

Do not assume that one Wave64 should always compute one output row.

Logical width is a kernel-design parameter and must be benchmarked against representative shapes.

---

# 5. Compute Units and Occupancy

MI50 exposes 60 Compute Units.

Kernel design should account for:

* workgroup distribution across CUs
* resident workgroups per CU
* VGPR pressure
* SGPR pressure
* LDS use
* wave occupancy

Occupancy should not be optimized in isolation.

A lower-occupancy kernel may still be faster if it:

* reduces memory traffic
* enables larger register tiles
* eliminates synchronization
* reduces instruction count

Likewise, theoretical high occupancy does not guarantee high throughput.

---

# 6. LDS

gfx906 provides local/shared memory resources per Compute Unit.

For MIInfer, LDS usage is an important constraint because large tiles may limit resident workgroups.

When evaluating a kernel, record:

* LDS bytes per workgroup
* maximum possible resident workgroups
* VGPR use
* achieved performance

Do not attempt to reduce VGPR use solely to increase occupancy if LDS already prevents additional residency.

This must be checked per kernel.

---

# 7. Register Pressure

VGPR pressure is expected to be a major kernel-design constraint.

Potential sources include:

* unpacked quantized values
* accumulation registers
* prefetched values
* multiple output rows
* larger tiles

When evaluating a kernel optimization, consider whether performance changes are caused by:

* extra instructions
* VGPR growth
* spilling
* occupancy changes

Explicit prefetch should never be assumed beneficial.

---

# 8. Matrix Instructions

gfx906 should not be treated as a modern CDNA tensor-core architecture.

Do not design MIInfer around:

* MFMA assumptions from newer CDNA
* rocWMMA
* WMMA-like matrix-core execution
* MI200/MI300 optimization strategies

Kernel work should instead focus on the execution capabilities actually present on gfx906.

---

# 9. Packed Dot Instructions

gfx906 exposes packed dot-product instructions that may be relevant to quantized inference.

Known useful instruction families include:

```text
v_dot2_f32_f16
v_dot2_i32_i16
v_dot4_i32_i8
v_dot4_u32_u8
v_dot8_i32_i4
v_dot8_u32_u4
```

These instructions are candidate building blocks, not guaranteed optimization wins.

Every use must be benchmarked.

In particular:

> Replacing ordinary arithmetic with a dot instruction does not automatically make a kernel faster.

Instruction count, packing overhead, register pressure, and memory behavior all matter.

---

# 10. Candidate Quantized Execution Path

One important initial research path is:

```text
Q4 weights
      +
Q8 activations
      ↓
register unpack / conversion
      ↓
packed integer dot product
      ↓
INT32 accumulation
      ↓
scale application
      ↓
FP32 / FP16 output
```

A likely candidate instruction is:

```text
v_dot4_i32_i8
```

The project must compare this against alternative paths.

Do not encode it as the only acceptable implementation before evidence exists.

---

# 11. FP16 Behavior

FP16 remains important for:

* activations
* attention
* KV cache
* unquantized paths
* reference kernels

Where needed, FP32 accumulation should be used to preserve numerical stability.

A useful candidate path is:

```text
FP16 input
FP16 weight
FP32 accumulation
```

when benchmark and correctness results justify it.

---

# 12. BF16

gfx906 does not provide the same native BF16 execution characteristics as modern accelerators.

Do not assume BF16 is a desirable internal dtype.

Model metadata requesting BF16 must not automatically cause MIInfer to execute everything as BF16.

If BF16 support is introduced, its behavior and generated instructions must be validated explicitly.

For the initial project, prefer:

```text
FP16
FP32 accumulation
```

where floating-point execution is required.

---

# 13. Memory Bandwidth

MI50 offers approximately 1 TB/s peak HBM2 bandwidth.

Low-batch LLM decode may frequently be bandwidth-sensitive.

This makes the following particularly important:

* compressed weight residency
* coalesced access
* minimal weight re-reading
* packed formats
* activation reuse
* avoiding unnecessary conversions
* context/KV traffic

Peak hardware bandwidth is not expected to be fully achieved by every kernel.

Use effective bandwidth measurements to understand how close a workload is to practical limits.

---

# 14. HBM vs Cache

Kernel experiments should distinguish between:

```text
HBM-bound behavior
```

and:

```text
cache-resident behavior
```

Repeated microbenchmarks can accidentally keep weights in cache.

Ensure working-set sizes reflect real model execution where possible.

A kernel that performs well only when its weights fit in cache may not improve real decode.

---

# 15. Weight Compression

For batch-1 decode, retaining compressed weights across HBM is usually desirable.

Conceptually:

```text
HBM
 ↓
packed Q4/Q6/Q8
 ↓
register unpack / decode
 ↓
compute
```

is generally preferable to:

```text
HBM
 ↓
expanded FP16 weights
```

when the expanded representation materially increases memory traffic.

However, exact trade-offs must be benchmarked.

---

# 16. HBM Clock

Memory-clock state is part of benchmark validity.

Where the platform exposes it, record HBM/MCLK during important performance runs.

Unexpectedly low memory clocks can invalidate bandwidth-sensitive measurements.

Environment capture should eventually include:

```text
sclk
mclk
```

before and after benchmark runs.

---

# 17. GPU Clock

Core clock must also be monitored.

Do not assume the GPU reaches expected clocks because the workload is active.

A benchmark should be marked contaminated if the MI50 remains stuck in an unexpectedly low performance state.

This is especially important after:

* abnormal process termination
* repeated benchmark crashes
* runtime failures
* system suspend/resume
* driver instability

---

# 18. Power

Power state must be considered when comparing implementations.

Record, where available:

* configured power limit
* observed power during workload

Do not compare two implementations at different power limits without documenting the difference.

Eventually consider reporting:

```text
tokens / joule
```

in addition to throughput.

---

# 19. Temperature

Temperature can affect clocks and therefore benchmark validity.

Record temperature during sustained benchmarks.

For controlled A/B comparisons, avoid:

```text
A = cold GPU
B = heat-soaked GPU
```

without compensating through interleaving or warm-up.

---

# 20. VRAM

The initial target has 32 GB HBM2.

VRAM should be treated as a budget.

Track:

```text
weights
KV cache
persistent packed weights
persistent activation buffers
temporary workspace
runtime metadata
graph resources
optional caches
```

A performance optimization that increases persistent VRAM must justify the reduced context capacity.

---

# 21. Single-GPU Assumption

The initial runtime assumes:

```text
one model
one MI50
one process
```

No initial architecture should be added for:

* tensor parallelism
* pipeline parallelism
* model sharding
* cross-GPU synchronization
* peer-to-peer transport

Multi-GPU work belongs to a later roadmap decision.

---

# 22. PCIe

The initial design assumes model execution remains primarily GPU-resident after loading.

PCIe should not be part of the per-token critical path.

Avoid architectures that require frequent:

* CPU↔GPU tensor movement
* weight swapping
* activation offload

unless future constraints force such behavior.

---

# 23. Weight Residency

The initial target model should fit within the MI50 32GB VRAM budget.

If the selected model does not fit:

* reconsider model/quantization choice
* reconsider representation
* document the decision

Do not introduce CPU offloading as an automatic solution.

---

# 24. KV Cache

KV cache becomes increasingly important as context grows.

Initial implementation should prioritize correctness.

Suggested progression:

```text
FP16 K
FP16 V
```

then, after profiling:

```text
Q8 K
FP16 V
```

Additional compression should be introduced only after:

* throughput measurement
* VRAM measurement
* model-quality validation

KV precision may differ from weight/activation precision.

---

# 25. Context Effects

Hardware bottlenecks may shift substantially with context length.

At short context:

```text
weight execution
GEMV/GEMM
MoE
```

may dominate.

At long context:

```text
attention
KV reads
memory traffic
```

may increasingly dominate.

This means a kernel optimization cannot be judged only at one context size.

---

# 26. HIP

HIP is the initial GPU programming model.

The project should compile directly for:

```text
gfx906
```

Avoid depending on runtime portability abstractions designed to target CUDA and multiple AMD generations unless they are demonstrated to have negligible cost and clear engineering value.

---

# 27. ROCm Support Reality

gfx906 is legacy hardware in modern ROCm ecosystems.

Do not assume:

```text
latest ROCm = best gfx906 environment
```

Different community projects have successfully used different ROCm generations and custom builds.

The validated MIInfer toolchain must therefore be treated as an explicit project artifact.

Record:

* ROCm version
* HIP version
* LLVM/clang version
* relevant runtime libraries
* gfx906 support method

---

# 28. Prebuilt Library Assumptions

Do not assume every prebuilt ROCm library contains usable gfx906 code.

Potentially affected libraries include:

* rocBLAS
* hipBLAS
* other ROCm math libraries

If a library is used, validate:

1. it contains gfx906 support
2. the relevant kernel actually executes on gfx906
3. it is correct
4. its performance is competitive

Do not treat successful linking as proof of usable device support.

---

# 29. rocBLAS / hipBLAS

rocBLAS/hipBLAS may be useful as:

* correctness reference
* performance baseline
* fallback for non-critical paths

They should not automatically define MIInfer's kernel strategy.

For representative shapes, compare:

```text
rocBLAS
vs
reference gfx906 implementation
vs
MIInfer
```

where practical.

---

# 30. Triton

Triton-gfx906 may be useful during research if a working compatible toolchain is available.

Potential role:

```text
idea
 ↓
rapid Triton prototype
 ↓
correctness
 ↓
benchmark
 ↓
profile
 ↓
HIP / ISA implementation if justified
```

Triton should not automatically become a production runtime dependency.

The final project should prefer a small deployable execution stack where practical.

---

# 31. ISA-Level Code

Inline assembly or compiler intrinsics may be appropriate for gfx906-specific primitives.

Use them only when:

* the compiler cannot reliably generate the desired instruction
* profiling shows the path matters
* correctness is covered
* the implementation remains inspectable

Do not use inline assembly merely to make code appear more specialized.

---

# 32. Device Validation

MIInfer should validate the target device during startup.

Conceptually:

```text
detect GPU
 ↓
confirm AMD
 ↓
confirm gfx906
 ↓
validate expected capabilities
 ↓
continue
```

Unsupported devices should result in an explicit error.

Do not silently continue with unvalidated assumptions.

---

# 33. Hardware Assertions

Architecture-specific code may use compile-time or runtime assertions.

Examples:

```cpp
static_assert(kWaveSize == 64);
```

or:

```text
require(device_arch == gfx906)
```

Assertions should protect assumptions that materially affect correctness.

---

# 34. Kernel Build Target

Canonical HIP builds should explicitly target:

```text
gfx906
```

Do not compile fat binaries for many GPU architectures during early development unless required by tooling.

The initial goal is to optimize the exact target.

---

# 35. Compiler Output Inspection

For important kernels, generated ISA should occasionally be inspected.

Questions include:

* Did the desired dot instruction appear?
* Did compiler-generated spills occur?
* Did vector operations become scalar?
* Did a code change unexpectedly increase VGPR use?
* Did the compiler introduce conversions?

Source-level appearance is not sufficient for low-level performance conclusions.

---

# 36. Performance Counters

When tooling permits, collect hardware counters for difficult performance questions.

Potential areas:

* HBM utilization
* cache hit rate
* wave occupancy
* memory stalls
* VALU utilization
* LDS behavior

Counter interpretation must remain subordinate to actual benchmark results.

---

# 37. Initial Hardware Validation Checklist

Before considering the MI50 environment ready for M1:

* [ ] GPU detected correctly
* [ ] gfx906 reported
* [ ] 32 GB VRAM visible
* [ ] trivial HIP kernel runs
* [ ] canonical CMake preset targets gfx906
* [ ] benchmark timer works
* [ ] GPU SCLK can be observed
* [ ] HBM/MCLK can be observed where available
* [ ] temperature can be observed
* [ ] power can be observed
* [ ] ROCm version captured
* [ ] compiler version captured
* [ ] no unexpected second GPU breaks ROCr initialization
* [ ] reference gfx906 runtime can execute
* [ ] environment capture produces machine-readable output

---

# 38. Multi-GPU Host Systems

A machine may contain additional GPUs even though MIInfer uses only one MI50.

This can affect ROCm/HSA initialization.

The development environment must verify that unsupported GPUs present in the system do not prevent ROCr from initializing the MI50.

If necessary, document a validated device-isolation strategy.

Do not assume all installed AMD GPUs can coexist safely in the same ROCm runtime configuration.

---

# 39. Known Development-System Risk

Older/non-compute AMD GPUs can cause compatibility problems with modern ROCm/HSA stacks.

If the host contains additional display adapters:

* verify HSA topology
* verify device enumeration
* validate runtime initialization
* isolate devices if required

This should be solved at environment level rather than worked around inside kernel code.

---

# 40. Hardware State Capture Format

The benchmark infrastructure should eventually generate something similar to:

```json
{
  "gpu": {
    "name": "AMD Instinct MI50",
    "gfx": "gfx906",
    "vram_bytes": 34359738368,
    "sclk_mhz": 0,
    "mclk_mhz": 0,
    "temperature_c": 0,
    "power_w": 0,
    "power_limit_w": 0
  },
  "software": {
    "rocm": "",
    "hip": "",
    "compiler": "",
    "kernel": ""
  }
}
```

Unavailable metrics should be represented explicitly rather than invented.

---

# 41. Benchmark Hardware Policy

Important performance results should state:

```text
Hardware:
AMD Instinct MI50 32GB
gfx906
single GPU
```

If another machine/device is used, label the result separately.

Do not merge heterogeneous hardware results into one performance series.

---

# 42. Primary Hardware Optimization Areas

Based on the initial hardware characteristics, likely high-value research areas include:

1. quantized GEMV
2. skinny GEMM
3. weight packing
4. Wave64 / half-wave cooperation
5. DPP reductions
6. activation quantization/reuse
7. launch overhead
8. static execution
9. long-context memory traffic
10. MoE expert execution if the target model is MoE

This list is a hypothesis.

Profiling decides the actual priority.

---

# 43. Hardware Non-Assumptions

Do not assume:

* FA is always the main bottleneck
* larger workgroups are faster
* maximum occupancy is optimal
* explicit prefetch helps
* latest ROCm is faster
* rocBLAS is slower
* custom kernels are faster
* packed dot instructions automatically win
* Q8 KV automatically improves throughput
* lower precision is always better
* a Radeon VII result directly predicts MI50 behavior

All of these require measurement.

---

# 44. Hardware Contract Philosophy

The initial hardware contract is deliberately narrow:

```text
MI50
+
gfx906
+
single GPU
+
known toolchain
```

This allows MIInfer to make choices that a portable runtime cannot.

If later hardware support requires weakening those assumptions, that must be an explicit architectural decision.

---

# 45. Guiding Principle

MIInfer should treat the MI50 as the platform, not as a compatibility target.

The design question is not:

> How do we make a generic inference framework continue to work on gfx906?

It is:

> What execution strategy best fits the actual strengths and limitations of gfx906?
