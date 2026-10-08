# M31-0005 — llama.cpp architectural comparison

Status: implementation comparison only; historical performance remains
directional.

The reference result artifact identifies llama.cpp commit
`9c2e0e491a822adae1f0b1c831adb4160057d24f`, HIP gfx906 build, ROCm 7.2.1,
same model SHA as the Z840 8K/32K MIInfer samples. At 8K, llama.cpp prefill was
227.3 tok/s and TG32 decode 25.2 tok/s; MIInfer was 215.6/26.1. At 32K,
llama.cpp was 200.7/24.5 and MIInfer was 149.6/21.0. The prompt/benchmark
semantics were not fully matched. At 64K/128K the historical MIInfer numbers
are from Machinist while llama.cpp ran on Z840, so the larger gaps are
cross-host and directional only.

At the source architecture level, llama.cpp's CUDA/HIP backend uses tiled
attention dispatch, online-softmax vector paths, occupancy-aware block
selection, and scratch pooling. The exact chosen HIP kernel for the reported
binary was not captured in a runtime dispatch trace; it is unsafe to claim that
one source kernel explains the measured gap. MIInfer's graph path is a custom
Wave64 two-stage Split-K kernel with fixed 64-grid and context-dependent active
splits. The most defensible transferable idea is to test dispatch/split policy
and memory reuse, not to transplant third-party code.

References at the measured llama.cpp commit:

- [`fattn-vec.cuh`](https://raw.githubusercontent.com/ggml-org/llama.cpp/9c2e0e491a822adae1f0b1c831adb4160057d24f/ggml/src/ggml-cuda/fattn-vec.cuh)
- [`fattn-common.cuh`](https://raw.githubusercontent.com/ggml-org/llama.cpp/9c2e0e491a822adae1f0b1c831adb4160057d24f/ggml/src/ggml-cuda/fattn-common.cuh)
- [`ggml-hip` source configuration](https://github.com/ggml-org/llama.cpp/tree/9c2e0e491a822adae1f0b1c831adb4160057d24f/ggml/src/ggml-hip)

No llama.cpp code was copied or adapted. Any future adaptation requires license
review and attribution. A useful next comparison is one host, one MI50, exact
model bytes and prompt/output counts, same initial thermal condition, and
separate prefill/decode/VRAM/output parity measurements. Real GPU execution
remains guarded by proof that the stop mechanism interrupts actual inference
in time.
