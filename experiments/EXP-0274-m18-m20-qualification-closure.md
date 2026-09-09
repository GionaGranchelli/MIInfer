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
| runtime-only MIInfer PP/first-token | PASS | EXP-0271; `results/m18-runtime/` |
| pinned llama.cpp PP/TG | BLOCKED | EXP-0269: exact model unsupported at pin; compatible supplemental curve retained |
| dynamic allocation 1K–128K | PASS | EXP-0270; `results/m19-context/20260909-allocation-sweep/` |
| 1K inference qualification | PASS | EXP-0270 |
| 8K inference smoke | PASS | EXP-0270 |
| 16K–128K correctness/steady-state decode | NOT QUALIFIED | 16K/32K/64K functional smoke and 8K/16K TG64 completed; replay/KV/deeper TG64 ladder remains open |
| API authentication | PASS | EXP-0272; `results/m20-auth/` |
| constrained Hermes submission | PARTIAL | EXP-0273; two completed ~8K requests plus controlled cancellation |
| regression suite | PASS | 23/23 CTest tests |

## Runtime-only result

The retained short curve uses one generated token per case:

| path | P9 PP tok/s | P129 PP tok/s | P513 PP tok/s |
|---|---:|---:|---:|
| default | 32.36 | 33.15 | 32.44 |
| M12 experimental | 32.22 | 47.15 | 45.94 |

The comparison excludes HTTP, JSON, ChatML, and Hermes. These historical
one-token runs qualify PP and first-token timing only; they do not qualify TG.
The mandated pinned llama.cpp binary rejects the exact model, so the requested
pinned ratio remains unavailable. A compatible, newer local llama.cpp build
measured PP8/128/512 at 33.4849/151.2700/191.3250 tok/s and TG64 at 22.2467
tok/s; it is explicitly supplemental, not the pinned claim.

## Architectural conclusion

The observed evidence supports retaining one serialized GPU worker, explicit
host-boundary cancellation, and the opt-in M12 layer-major prefill path. It
does not justify multiple workers, continuous batching, or automatic promotion
of 128K. M12 is numerically functional through the measured 64K smoke path but
its PP curve falls from 34.0826 tok/s at 32K to 28.3657 tok/s at 64K. The next
valid performance step is an approved compatible pinned reference and a
long-context correctness/replay campaign before changing prefill kernels.

## Final decision

M18-A, dynamic serving controls, M20-B authentication, the available regression
gates, and functional long-context smoke through 64K are KEEP/PASS. M18-B
exact comparison remains BLOCKED by the pinned checkout’s model support;
M19 correctness/TG64 promotion and M20-A production Hermes remain PARTIAL.
The repository records those limits without silently claiming long-context
production support.
