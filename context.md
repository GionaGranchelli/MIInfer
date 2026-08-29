## 1. Situation

Giona, I would **not make our main project a fork of any of them**.

I recommend a **two-repository strategy**:

```text
1. gfx906-reference      ← fork/clone of milpster
2. MIInfer               ← new project, our architecture
```

The first exists to give us the strongest reproducible llama.cpp/gfx906 baseline. The second is where we try to beat it.

---

## 2. Context & Complication

There are really four possible approaches:

| Approach                    | Initial speed | Freedom | Maintenance burden | Long-term differentiation |
| --------------------------- | ------------: | ------: | -----------------: | ------------------------: |
| Fork llama.cpp              |          ★★★★ |      ★★ |                  ★ |                        ★★ |
| Fork milpster               |         ★★★★★ |      ★★ |                 ★★ |                        ★★ |
| Fork iacopPBK/etc.          |           ★★★ |      ★★ |                  ★ |                        ★★ |
| **New specialized runtime** |            ★★ |   ★★★★★ |               ★★★★ |                     ★★★★★ |

`llama.cpp` is MIT licensed and extremely active, but it is a huge generic runtime with over 22,000 forks.

Milpster is also MIT licensed and currently very actively maintained, with its latest push today, August 28. It deliberately follows upstream while preserving its gfx906-specific changes.

That makes milpster an excellent **reference implementation**, but it doesn't remove the fundamental problem:

> If we fork it, our architecture remains llama.cpp's architecture.

And the interesting hypothesis we want to test is precisely whether we can outperform that architecture by specializing much more aggressively for MI50/gfx906.

---

## 3. Question / Goal

The objective shouldn't initially be:

> Build another general LLM runtime.

It should be much narrower:

> **Build the fastest practical inference path we can for one MI50 and one carefully selected modern model family, then expand only where evidence justifies it.**

That distinction determines the repo strategy.

---

# 4. Answer / Recommendation

## My choice: **new project from scratch, but not reinvent-everything-from-scratch**

This distinction matters.

I would **not** do:

```text
blank C++ repository
↓
write GGUF
↓
write tokenizer
↓
write every quant format
↓
write sampling
↓
write inference
↓
18 months later benchmark something
```

Nor would I do:

```text
fork llama.cpp
↓
delete things
↓
keep deleting things
↓
discover everything depends on GGML
↓
eventually still have llama.cpp
```

Instead:

```text
                    OUR WORKSPACE

      ┌──────────────────────────────┐
      │       gfx906-reference       │
      │                              │
      │ fork: milpster/gfx906...     │
      │                              │
      │ llama.cpp + best known       │
      │ gfx906 optimizations         │
      └──────────────┬───────────────┘
                     │
            correctness oracle
              + performance
                 baseline
                     │
                     ▼
      ┌──────────────────────────────┐
      │           MIInfer            │
      │                              │
      │       NEW repository         │
      │                              │
      │ gfx906-native runtime        │
      │ specialized for MI50         │
      └──────────────────────────────┘
```

### Why fork **milpster** for the reference rather than vanilla llama.cpp?

Because it already gives us:

* upstream llama.cpp compatibility
* dedicated Vega20 MMQ configuration
* gfx906-specific FlashAttention work
* known-good ROCm configuration
* benchmark methodology
* adaptive MTP work
* documented negative results
* real MI50/Radeon VII measurements
* recent upstream synchronization

We don't need to rediscover that baseline.

But **we don't develop MIInfer inside that fork**.

---

# What MIInfer should actually contain

I'd start with roughly:

```text
miinfer/
├── CMakeLists.txt
├── LICENSE
├── README.md
│
├── include/miinfer/
│   ├── runtime.hpp
│   ├── model.hpp
│   └── tensor.hpp
│
├── src/
│   ├── runtime/
│   ├── model/
│   ├── memory/
│   └── sampling/
│
├── gfx906/
│   ├── primitives/
│   ├── kernels/
│   │   ├── gemv/
│   │   ├── gemm/
│   │   ├── norm/
│   │   ├── rope/
│   │   ├── attention/
│   │   └── moe/
│   └── config/
│
├── bench/
│   ├── micro/
│   ├── model/
│   └── compare-llamacpp/
│
├── experiments/
│   ├── accepted/
│   └── rejected/
│
└── tests/
```

But initially most directories stay empty.

---

# Phase 0 — establish the enemy

Before implementing our runtime, freeze a baseline.

Take milpster and pin:

```text
gfx906-reference
└── baseline/2026-08-mi50
```

Choose **one model**.

Something like:

```text
Qwen3 / Qwen3.x
~30B class
quantized
single MI50 32GB
```

If MoE fits our target, even better because it gives more opportunities for specialization.

Establish:

```text
PP512
PP2048
PP8192

TG128
TG512
TG1024

ctx:
512
8k
32k

VRAM
power
GPU clocks
```

And output quality hashes/logits where practical.

That becomes:

> **MIInfer has to beat this.**

---

# Phase 1 — don't build an LLM runtime yet

This is where I'd begin coding.

Create **`gfx906-lab` either as part of MIInfer or as a separate repository**.

My preference now is actually:

```text
MIInfer
└── lab/
```

rather than three repositories.

Start with only kernel experiments.

For example:

### Experiment 001

```text
Q4_K × Q8 activation
matrix-vector multiplication
```

Implement:

```text
A. llama.cpp equivalent
B. straightforward HIP implementation
C. gfx906-specialized implementation
```

Test them against one another.

Then:

```text
Q6_K
RMSNorm
RoPE
```

etc.

We don't need a model loader to answer:

> Can our architecture-specific kernels beat llama.cpp?

That gives us an **early kill criterion**.

---

# Phase 2 — first end-to-end model

Only once we have evidence that the kernels are competitive do we build enough runtime to execute a model.

And here I would **reuse selectively**.

Because llama.cpp is MIT licensed, we can legally reuse specific pieces while retaining the required copyright/license notices.

For example, it would be perfectly sensible to initially reuse or adapt:

```text
GGUF parsing
tokenizer vocabulary handling
quantization definitions
sampling utilities
```

while **not reusing**:

```text
GGML execution graph
GGML scheduler
generic GPU backend
generic CUDA abstraction architecture
dynamic operator dispatch
generic multi-device machinery
```

That's the boundary I care about.

Think:

```text
        reusable commodity
               │
      ┌────────┴─────────┐
      │ GGUF / tokenizer │
      └────────┬─────────┘
               │
               ▼
       OUR representation
               │
               ▼
       static model planner
               │
               ▼
        gfx906 kernels
               │
               ▼
             MI50
```

Not:

```text
GGUF
 ↓
GGML
 ↓
GGML graph
 ↓
GGML scheduler
 ↓
CUDA abstraction
 ↓
HIP
```

---

# Phase 3 — the key architectural difference

Suppose our first target is exactly:

```text
Model: Qwen X
Quant: Q4_K_M
GPU: gfx906 / MI50 32GB
Batch: 1
```

Then at model load we can turn:

```text
model.gguf
```

into something conceptually like:

```text
MI50ExecutionPlan {
    layer[0] {
        q_proj  = kernel_17
        k_proj  = kernel_17
        v_proj  = kernel_17
        rope    = kernel_4
        attn    = kernel_9
        o_proj  = kernel_21
        ...
    }

    layer[1] ...
}
```

No operator lookup on every token.

No generic architecture decisions.

No "which GPU am I on?"

We already know:

```text
gfx906
Wave64
64 KB LDS
MI50
specific quant
specific shapes
```

So:

```text
load
 ↓
inspect model
 ↓
select/construct static execution plan
 ↓
allocate VRAM once
 ↓
capture HIP graph where possible
 ↓
run token
 ↓
replay
 ↓
replay
 ↓
replay
```

That's our differentiator.

---

# So which project do we fork?

### **Milpster: YES — but only as benchmark/reference.**

Call it something unmistakable:

```text
GionaGranchelli/gfx906-llama-reference
```

And don't heavily develop it.

Use branches such as:

```text
upstream
baseline-mi50
experiments/*
```

---

### **llama.cpp: NO direct product fork**

Use it as the upstream/reference source.

Milpster already does the painful work of integrating upstream changes. There's little value in maintaining another equivalent derivative.

---

### **iacopPBK: NO**

Treat it as a **research mine**.

When we identify an interesting kernel:

```text
study
↓
understand
↓
reimplement/port isolated concept
↓
benchmark
```

Don't inherit its whole tree.

---

### **Turbo fork: NO**

Same.

Specific ideas such as compressed KV can become experiments later.

Not architectural foundations.

---

### **NInfer / MobyDick: NO**

Same pattern:

```text
idea source
not codebase
```

Unless during implementation we find a beautifully isolated MIT-compatible component that's genuinely worth importing.

---

# The repository setup I would create

Ultimately:

```text
github.com/Constant-Labs/
│
├── MIInfer
│   │
│   ├── runtime
│   ├── gfx906 kernels
│   ├── benchmark suite
│   ├── experiments
│   └── docs/research
│
└── gfx906-llama-reference
    │
    └── fork/mirror of milpster
```

Potential naming deserves thought, but structurally this is what I'd do.

And I **would put this under Constant Labs rather than your personal account** if we see it as a serious open-source/product asset.

---

## Very concrete first roadmap

I would make the initial roadmap ridiculously focused:

```text
M0 — Reproducible baseline
 ├─ MI50 environment fingerprint
 ├─ build milpster
 ├─ choose one model
 ├─ benchmark
 └─ freeze results

M1 — Kernel laboratory
 ├─ benchmark harness
 ├─ correctness harness
 ├─ Q4/Q6 × Q8 matvec baseline
 └─ first gfx906 kernel

M2 — Prove specialization
 ├─ DPP/wave64 experiments
 ├─ LDS/register experiments
 ├─ packing experiments
 └─ KEEP/REJECT decisions

M3 — Minimal runtime
 ├─ GGUF metadata
 ├─ tensor loading
 ├─ static memory planner
 ├─ one model architecture
 └─ one quantization

M4 — First token
 ├─ embeddings
 ├─ norm
 ├─ attention
 ├─ FFN/MoE
 └─ sampler

M5 — Beat baseline
 ├─ PP
 ├─ TG
 ├─ VRAM
 ├─ power
 └─ correctness

M6 — Optimize runtime
 ├─ fixed execution plan
 ├─ HIP graphs
 ├─ kernel fusion
 ├─ packing
 └─ allocation elimination

M7 — Expansion
 ├─ additional quant
 ├─ second model
 ├─ long context
 └─ speculative/MTP
```

The critical milestone is **M2**, not M7.

At M2 we want to know:

> **Can we actually produce kernels that justify building MIInfer?**

If the answer is no, we've spent relatively little and learned a lot.

If the answer is yes, then building the runtime becomes justified.

---

## 5. Socratic Clause

The tempting move is to fork milpster because it gives us an instant working product.

That would also be the safest way to accidentally spend six months making **“milpster + 12%”** instead of building something architecturally interesting.

Conversely, "from scratch" can become an ego trap too. There is zero strategic value in writing our own tokenizer or GGUF parser merely because we can.

So I'd draw the intellectual-property/architecture boundary here:

```text
Commodity infrastructure
    → reuse where sensible

Execution architecture
    → ours

gfx906 kernel strategy
    → ours

benchmark methodology
    → ours

model specialization
    → ours
```

That's the balance between engineering pragmatism and actual differentiation.

---

## 6. Summary / Key Takeaways

* **Main project:** start **MIInfer from scratch**.
* **Do not** fork llama.cpp as MIInfer's foundation.
* **Fork milpster separately** as our reproducible best-known gfx906/llama.cpp baseline.
* Treat iacopPBK, NInfer, MobyDick and Turbo as **research/patch mines**, not foundations.
* Reuse commodity MIT components where useful, especially GGUF/tokenizer/format definitions.
* Do **not** inherit GGML's graph/scheduler/backend architecture.
* First implementation target should be **one GPU + one model + one quant + batch 1**.
* Build a kernel lab **before** spending serious effort on an end-to-end runtime.
* First existential milestone: prove at least one or two critical gfx906 kernels can meaningfully beat the strongest llama.cpp baseline.
* Only then build the static MI50 execution runtime.

**So if we start tomorrow, the first commit is not a llama.cpp fork. It is `MIInfer` with a benchmark/experiment harness, while a separate milpster fork becomes the opponent we measure against.**
