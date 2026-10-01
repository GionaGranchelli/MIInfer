# V2-0047 — Release packaging and distribution hardening

## Objective

Produce the normal MIInfer release archive using CPack, exclude research-only
build prerequisites and package contents, and qualify installation and the
installed runtime without relying on the source/build tree. No inference source,
kernel, numerical behavior, or public CLI contract changed.

## Baseline and environment

- V2-0046 was fast-forwarded onto canonical `main` at
  `16f18b9f783f8f97c31df2196a3eddac57d76933` and pushed to `origin/main`.
- `rewrite/v2-0047-release-packaging` was created from that commit.
- Implementation commit: `6a53b4b89717` (`build: isolate release package from research targets`).
- Clean build: Release, GCC `16.2.1`, HIP Clang `20.0.0`, ROCm HIP `7.1.52802`, target `gfx906`.
- The model smoke used `Qwen3.8-27B-Q4_K_M.gguf` on the MI50; no performance benchmark was run.

## CPack failure reproduced

Commands:

```bash
cmake -S . -B /tmp/miinfer-v2-0047-repro -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/miinfer-v2-0047-repro --target package --verbose -j2
```

The second command failed with exit 2. The generated top-level Makefile has
`package: preinstall` and `preinstall: all`; the generated `Makefile2` default
`all` target included `miinfer-v2-0045-qual-bench`. Its object compile was:

```text
/usr/bin/c++ -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_AMD__=1 \
  -I/home/fedora-workstation/Development/mi50/include \
  -I/tmp/miinfer-v2-0047-repro/generated -O3 -DNDEBUG -std=c++20 -fPIE \
  -Wall -Wextra -Wpedantic -MD \
  -MT CMakeFiles/miinfer-v2-0045-qual-bench.dir/bench/v2_0045_qualification_bench.cpp.o \
  -MF CMakeFiles/miinfer-v2-0045-qual-bench.dir/bench/v2_0045_qualification_bench.cpp.o.d \
  -o CMakeFiles/miinfer-v2-0045-qual-bench.dir/bench/v2_0045_qualification_bench.cpp.o \
  -c /home/fedora-workstation/Development/mi50/bench/v2_0045_qualification_bench.cpp
```

GCC failed in the transitive libstdc++ include chain `bench source → <chrono>
→ bits/chrono_io.h → <format>`:

```text
/usr/include/c++/16/format:4550:30: error: expected an identifier for the attribute name [-Wtemplate-body]
 4550 |       [[__gnu__::__noinline__]]
```

This is a compiler/header parse failure in
`bench/v2_0045_qualification_bench.cpp`, not a CPack or MIInfer runtime source
failure. `install(TARGETS ...)` installs only `miinfer` and
`miinfer-device-info`; the qualification benchmark is not installed and is not
required by either executable.

## Build-graph correction

Added `MIINFER_BUILD_BENCHMARKS`, `MIINFER_BUILD_RESEARCH_TOOLS`, and
`MIINFER_BUILD_TESTS`. The standard build excludes benchmark and historical /
forensic executables from `all`; the release test executables remain enabled by
default. Developer opt-in configuration restores benchmark and research targets.
The GCC 16 failure remains reproducible when that benchmark is explicitly
included; it no longer blocks a normal release.

The install manifest is now curated to the two runtime executables, installer,
license, README, and the public CLI, hardware, and release documents. Historical
experiments, benchmark instructions, build-tree files, and developer-only docs
are not installed. `test-package.sh` enforces an exact file allowlist and is
registered as a host-only CTest package gate (skipped only before the archive
has been produced).

## Final release artifact

- Name: `miinfer-0.2.0-gfx906-Linux.tar.gz`
- Size: `958843` bytes (937 KiB)
- SHA-256: `84645a01880cbe4b5efb7e4b5371d565cd5e951f416d94e43fbff04aa51b44c3`
- Artifact build identity: version `0.2.0`, Git commit `6a53b4b89717`,
  `Git dirty: false`, Release, GNU 16.2.1, HIP Clang 20.0.0, `gfx906`.
- Manifest (8 files):

```text
miinfer-0.2.0-gfx906-Linux/bin/miinfer
miinfer-0.2.0-gfx906-Linux/bin/miinfer-device-info
miinfer-0.2.0-gfx906-Linux/install.sh
miinfer-0.2.0-gfx906-Linux/share/doc/MIInfer/LICENSE
miinfer-0.2.0-gfx906-Linux/share/doc/MIInfer/README.md
miinfer-0.2.0-gfx906-Linux/share/doc/MIInfer/docs/cli-product-contract.md
miinfer-0.2.0-gfx906-Linux/share/doc/MIInfer/docs/hardware.md
miinfer-0.2.0-gfx906-Linux/share/doc/MIInfer/docs/release.md
```

## Verification transcript

Clean source worktree at `6a53b4b89717`:

```text
cmake -S /tmp/miinfer-v2-0047-source -B /tmp/miinfer-v2-0047-final-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/tmp/miinfer-v2-0047-clean/_deps/nlohmann_json-src
cmake --build /tmp/miinfer-v2-0047-final-build -j2                   PASS
cmake --build /tmp/miinfer-v2-0047-final-build --target package -j2 PASS
```

CPack output:

```text
CPack: Create package using TGZ
CPack: Install projects
CPack: - Run preinstall target for: MIInfer
CPack: - Install project: MIInfer []
CPack: Create package
CPack: - package: /tmp/miinfer-v2-0047-final-build/miinfer-0.2.0-gfx906-Linux.tar.gz generated.
```

CTest: `26/26` passed, including 14 MI50/GPU-required tests and the package
archive gate. `test-package.sh` also passed with the model argument against the
clean commit-built archive. From the installed prefix it verified:

- default `$HOME/.local/miinfer` install and explicit-prefix installation;
- repeated install preserves unrelated files and overwrites packaged files;
- version, commit, target architecture, executable permissions, and dynamic
  dependencies;
- `doctor --model`, `models`, and `inspect`;
- `run` with `--prompt`, positional prompt, and stdin, plus production-profile
  selection even with a stale `MIINFER_PRESET` value;
- default-model multi-turn `chat` with nonzero prefix reuse;
- unauthenticated localhost serving, `/readyz`, `/v1/models`, and an
  OpenAI-compatible chat completion;
- unauthenticated non-loopback (`0.0.0.0`) serving remains rejected.

The exact real-model package smoke command and terminal result were:

```text
$ rtk bash scripts/test-package.sh \
    /tmp/miinfer-v2-0047-final-build/miinfer-0.2.0-gfx906-Linux.tar.gz \
    /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Testing run --prompt...
Testing automatic production profile with a stale selector set...
Profile: production
Testing positional prompt...
Profile: production
Testing piped prompt...
Profile: production
Testing configured default-model chat...
Testing configured default-model server and API completion...
first-run CLI test passed for Qwen3.8-27B-Q4_K_M.gguf
package smoke test passed: /tmp/miinfer-v2-0047-final-build/miinfer-0.2.0-gfx906-Linux.tar.gz
```

The retained installed prefix independently reported:

```text
MIInfer version: 0.2.0
Git commit: 6a53b4b89717
Git dirty: false
Build type: Release
Compiler: GNU 16.2.1
HIP compiler: Clang 20.0.0
HIP availability: true
Target architecture: gfx906
```

`doctor --model` returned PASS for GPU gfx906, MI50 identity, the Qwen3.8
production model, available port, ROCm, production runtime profile, and 32 GiB
free VRAM. `models` found the GGUF and `inspect` reported Qwen3.5, 65 layers,
262144 model context, 248320 vocabulary, 15.92 GiB weights, and supported
production compatibility.

No inference performance qualification was repeated; this goal changes only
build graph, packaging contents, docs, and package tests.
