# MI50 Platform Notes

This document records the reproducible platform diagnosis for the MIInfer
target host. It is intentionally separate from benchmark results.

## Target and topology

The host contains two AMD GPUs, both bound to `amdgpu`:

| PCI BDF | PCI ID | DRM | KFD node | KFD name | GFX | KFD GPU ID |
| --- | --- | --- | --- | --- | --- | ---: |
| `0000:03:00.0` | `1002:692b` | `card1`, `renderD128` | 2 | Tonga | `gfx802` | 38113 |
| `0000:06:00.0` | `1002:66a1` | `card2`, `renderD129` | 3 | Vega20 / MI50 | `gfx906` | 4670 |

The KFD capability words are `0x00404280` for Tonga and `0xac73a280` for
Vega20. Bits 12--13 decode to DoorbellType 0 and 2 respectively. The MI50
node reports wavefront size 64, 60 active CUs, and 32752 MiB of VRAM.

The kernel log shows successful `amdgpu` initialization and KFD device
registration for both GPUs, including:

```text
kfd kfd: added device 1002:692b
kfd kfd: added device 1002:66a1
amdgpu 0000:06:00.0: SE 4, SH per SE 1, CU per SH 16, active_cu_number 60
```

The user has `render` and `video` access, and `/dev/kfd`, `renderD128`, and
`renderD129` are present.

## ROCr failure

The installed stack is Fedora 44 ROCm 7.1.1 packages with HIP 7.1.52802-9999
and Clang 20.0.0.rocm. `rocminfo` fails during HSA initialization:

```text
hsa api call failure at: .../rocminfo.cc:1324
Call returned HSA_STATUS_ERROR: A generic error has occurred.
```

The failure is not a PCI visibility or permissions failure. `rocm-smi` sees
both GPUs, sysfs exposes both KFD nodes, and the kernel registers both devices.
`HSAKMT_DEBUG_LEVEL=7` additionally reports one remapped-MMIO mapping warning
for each GPU, but no kernel initialization failure is logged.

The local evidence matches the ROCr pre-Vega multi-GPU defect documented in
[ROCm issue #8801](https://github.com/ROCm/rocm-systems/issues/8801): the
unsupported DoorbellType 0 Tonga node throws a generic HSA error during agent
discovery, aborting `hsa_init()` before the supported DoorbellType 2 MI50 can
be exposed. The corresponding upstream fix is
[ROCm pull request #4936](https://github.com/ROCm/rocm-systems/pull/4936),
which makes the unsupported-device path skip Tonga instead of aborting the
whole enumeration.

`ROCR_VISIBLE_DEVICES` was tested with GPU indices, KFD GPU IDs, and node
indices. Every value produced the same HSA error. This is expected when the
failure occurs during agent discovery; these variables are not a valid
workaround for this installed runtime.

### Root-cause status

The mechanism is confirmed by the matching KFD capability data and the
upstream reproducer/fix. The local privileged A/B test that temporarily
unbinds `0000:03:00.0` has not been run: this Wayland session uses that display
GPU, and the current user has no non-interactive root authorization. No
persistent driver or boot configuration was changed.

## Recovery options

The preferred durable recovery is to install a ROCr package containing the
fix from PR #4936, while keeping the MI50 identified as real `gfx906`. Do not
use `HSA_OVERRIDE_GFX_VERSION`.

The controlled temporary A/B test should be performed from a console or
remote session where losing the display GPU is acceptable:

```bash
sudo sh -c 'printf %s 0000:03:00.0 > /sys/bus/pci/drivers/amdgpu/unbind'
rocminfo
sudo sh -c 'printf %s 0000:03:00.0 > /sys/bus/pci/drivers/amdgpu/bind'
```

The expected diagnostic result is that the gfx906-only configuration makes
`rocminfo` succeed. This command sequence is a controlled test, not yet a
validated permanent host configuration.

## Fedora development packages

The runtime packages do not include the CMake development metadata. Fedora
provides the required packages separately:

```text
hipblas-devel-7.1.0-6.fc44
hipblas-common-devel-7.1.0-3.fc44
rocblas-devel-7.1.1-7.fc44
rocsolver-devel-7.1.1-4.fc44
```

The primary hipBLAS config is named `hipblas-config.cmake` and is installed
under `/usr/lib64/cmake/hipblas`; `hipblas-common` is under
`/usr/share/cmake/hipblas-common`. The runtime-only installation currently
has none of these development packages. A temporary extracted prefix was
validated for reference configuration with:

```bash
cmake -S . -B build-mi50-localhipblas \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/rocm-development-prefix \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_LTO=OFF -DGGML_VULKAN=OFF
```

The intended system fix is to install these packages normally, not to retain
hardcoded library paths in the reference or MIInfer source trees.

## Current gate status

```text
ROCr/HSA: BLOCKED by installed ROCr multi-GPU initialization bug
gfx906 in rocminfo: BLOCKED
generic HIP device access: BLOCKED downstream of HSA
MIInfer GPU tests: BLOCKED downstream of HSA
privileged gfx802 isolation A/B: NOT RUN
hipBLAS CMake metadata: LOCATED; not installed system-wide
reference CMake configure: PASS with temporary development prefix
reference build (llama-cli, llama-bench): PASS with temporary development prefix
```
