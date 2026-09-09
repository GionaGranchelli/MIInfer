# EXP-0274 — M18–M20 qualification closure

## Scope

This record closes the implementable portion of the lifecycle, context,
serving-hardening, benchmark, and Hermes campaign on the physical MI50. It
does not promote experimental long-context behavior or manufacture a reference
comparison when the pinned competitor cannot load the exact model.

## Environment

* GPU: AMD Instinct MI50, gfx906, Wave64, 32 GB HBM2
* ROCm/HIP: 7.1.52802-9999
* Compiler: Clang 20.0.0
* Kernel: Linux 7.1.10-200.fc44 x86_64
* Model: Qwen3.8-27B-Q4_K_M GGUF
* Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
* Reference pin: `125db33d6d352b5c65357eaf37e8f7ae2fe6fbd8`

## Gate summary

| gate | result | evidence |
|---|---|---|
| lifecycle/cancellation | PASS | EXP-0268; `results/m18-lifecycle/` |
| strict request limits | PASS | `scripts/test-serve.sh`; CTest |
| runtime-only MIInfer PP/TG | PASS | EXP-0271; `results/m18-runtime/` |
| pinned llama.cpp PP/TG | BLOCKED | EXP-0269: exact model unsupported at pin |
| dynamic allocation 1K–128K | PASS | EXP-0270; `results/m19-context/20260909-allocation-sweep/` |
| 1K inference qualification | PASS | EXP-0270 |
| 8K inference smoke | PASS | EXP-0270 |
| 16K–128K correctness/steady-state decode | NOT QUALIFIED | no promotion from allocation-only evidence |
| API authentication | PASS | EXP-0272; `results/m20-auth/` |
| constrained Hermes submission | PARTIAL | EXP-0273; `results/hermes/` |
| regression suite | PASS | 23/23 CTest tests |

## Runtime-only result

The retained short curve uses one generated token per case:

| path | P9 PP tok/s | P129 PP tok/s | P513 PP tok/s | TG tok/s |
|---|---:|---:|---:|---:|
| default | 32.36 | 33.15 | 32.44 | 27.26–28.17 |
| M12 experimental | 32.22 | 47.15 | 45.94 | 613.31–635.51 |

The comparison excludes HTTP, JSON, ChatML, and Hermes. Since the pinned
llama.cpp binary rejects the exact model, the campaign cannot answer the
requested MIInfer-vs-llama.cpp ratio or attribute the gap to HTTP versus GPU
prefill architecture. That is a reference compatibility blocker, not a
performance conclusion.

## Architectural conclusion

The observed evidence supports retaining one serialized GPU worker, explicit
host-boundary cancellation, and the opt-in M12 layer-major prefill path. It
does not justify multiple workers, continuous batching, or automatic promotion
of 128K. The next valid performance step is to obtain an approved llama.cpp
reference revision that supports the same GGUF, then repeat the pinned PP/TG
ladder before changing prefill kernels.

## Final decision

M18-A, dynamic serving controls, M20-B authentication, and the available
regression gates are KEEP/PASS. M18-B exact comparison and M20-A production
Hermes qualification remain explicitly BLOCKED/PARTIAL for the evidence stated
above. The repository is ready for that next external-state change without
silently claiming long-context production support.
