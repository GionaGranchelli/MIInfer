# Tests

There are two intentionally distinct test categories:

* `host-only` checks basic C++20 build/test infrastructure and requires no GPU.
* `gpu-required` runs device validation and a deterministic vector-add HIP
  kernel. It requires a usable gfx906 HIP runtime and does not skip silently.

Canonical commands on a HIP development machine are:

```bash
cmake --preset mi50-debug
cmake --build --preset mi50-debug
ctest --preset mi50-debug
```

Run only the host test when GPU access is unavailable:

```bash
ctest --test-dir build/mi50-debug -L host-only --output-on-failure
```

The GPU test should fail clearly without gfx906 hardware or when ROCm cannot
access `/dev/kfd`; that failure is evidence that the required environment is
not available, not a passing CI skip. Device inspection is also available via:

```bash
./build/mi50-debug/miinfer-device-info
./build/mi50-debug/miinfer-device-info --device 0
```

The explicit device form is useful for proving that a non-gfx906 selection is
rejected. No CPU fallback is provided.
