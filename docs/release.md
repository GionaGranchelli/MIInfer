# Release and installation

MIInfer releases target Linux on one AMD Instinct MI50 32 GB (`gfx906`) and
require a compatible ROCm/HIP runtime on the host. The archive contains the
MIInfer executables, installer, license, and user-facing documentation; it
does not contain the model or ROCm libraries.

## Install a release archive

Download `miinfer-<version>-gfx906-Linux.tar.gz` from the project release
distribution, then install it into the default user prefix:

```bash
tar -xzf miinfer-<version>-gfx906-Linux.tar.gz
./miinfer-<version>-gfx906-Linux/install.sh \
  miinfer-<version>-gfx906-Linux.tar.gz
export PATH="$HOME/.local/miinfer/bin:$PATH"
miinfer doctor --model ~/models/Qwen3.8-27B-Q4_K_M.gguf
miinfer chat ~/models/Qwen3.8-27B-Q4_K_M.gguf
```

The installer accepts an optional destination prefix. Re-running it overlays
the archive files at that prefix, which supports replacing an existing
installation. User configuration is stored separately under
`~/.config/miinfer`; model files are never copied into the installation.

## Build a release archive from source

With the supported HIP/ROCm development toolchain installed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --build build --target package
scripts/test-package.sh build/miinfer-0.2.0-gfx906-Linux.tar.gz
```

The package name is derived from the CMake project version, target
architecture, and operating system. `miinfer --version` reports that version,
the source commit, dirty state, compiler, and target architecture. Normal
release builds omit research benchmarks and forensic tools. Developers can
include them in the default build with
`-DMIINFER_BUILD_BENCHMARKS=ON -DMIINFER_BUILD_RESEARCH_TOOLS=ON`; release
tests are enabled by default and can be controlled with
`-DMIINFER_BUILD_TESTS=OFF`.

## Qualified performance scope

The qualified model is Qwen3.8-27B-Q4_K_M on one MI50. These are the measured
V2-0045 medians; prefill is prompt processing and decode is forced-token
model-forward replay, not HTTP generation throughput.

| Workload | MIInfer | mx-llama.cpp | upstream llama.cpp |
| --- | ---: | ---: | ---: |
| P512 prefill | 2.180 s | 2.279 s | 2.566 s |
| P1024 prefill | 4.368 s | 4.576 s | 5.173 s |
| P2048 prefill | 8.839 s | 9.195 s | 10.416 s |
| P4096 prefill | 18.045 s | 18.576 s | 21.068 s |
| P8192 prefill | 37.634 s | 37.982 s | 42.978 s |
| P64 decode, TG128 | 33.81 ms/token | 38.47 ms/token | 41.43 ms/token |
| P512 decode, TG128 | 34.07 ms/token | 38.54 ms/token | 41.41 ms/token |
| P2048 decode, TG128 | 34.50 ms/token | 39.31 ms/token | 41.99 ms/token |
| P8192 decode, TG128 | 36.88 ms/token | 42.77 ms/token | 43.45 ms/token |

This is a runtime benchmark claim for the tested setup, not a promise of
equivalent end-to-end HTTP throughput on other hardware, models, or software
versions. The real multi-turn MIInfer serving workload passed with prefix
reuse; it was not competitively measured against the other runtimes.
