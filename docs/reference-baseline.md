# External gfx906 Reference Baseline

MIInfer's primary external comparison is the separate
[`milpster/gfx906-llama-cpp`](https://github.com/milpster/gfx906-llama-cpp)
repository. It is a reference and benchmark opponent, not a MIInfer runtime
dependency.

## Pin

```text
Repository: https://github.com/milpster/gfx906-llama-cpp.git
Branch: master
Commit: 6e4ef6c1a553b8f61ad77bba18e9ca05aa677295
Commit date: 2026-08-28T21:48:37+02:00
Pinned on: 2026-08-29
```

The commit was obtained from `git ls-remote` and verified by a shallow checkout.
Do not benchmark a moving branch and label it as this baseline.

## Checkout

Keep the reference checkout outside this repository:

```bash
git clone https://github.com/milpster/gfx906-llama-cpp.git gfx906-reference
cd gfx906-reference
git checkout 6e4ef6c1a553b8f61ad77bba18e9ca05aa677295
```

## MI50 build starting point

The pinned repository's `BUILD-VEGA20.md` is the source for its Vega20 build
settings. For a single MI50 baseline, begin with the following reduced
configuration and record any toolchain-specific changes in the experiment
record:

```bash
cmake -S . -B build-mi50 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx906 \
  -DGPU_TARGETS=gfx906 \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_LTO=OFF \
  -DGGML_VULKAN=OFF
cmake --build build-mi50 --target llama-cli llama-bench
```

The upstream guide also describes ROCm 6.1, custom rocBLAS, Vulkan, and a
multi-GPU configuration. Those are not silently assumed here. The physical
MI50 run must record the actual ROCm/HIP compiler, rocBLAS availability,
environment variables, GPU visibility, and final build command before results
are considered comparable.

## Baseline status

```text
Commit pinned: YES
Build on physical MI50: PENDING
Model selected: PENDING
Correctness comparison: PENDING
Performance measurements: PENDING
```

The first model and exact benchmark matrix remain open until the reference is
built and the target model is frozen. No external benchmark result is claimed
by this record.
