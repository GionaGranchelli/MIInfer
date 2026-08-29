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
indices. Every value produced the same HSA error while both GPUs were
exposed. This is expected when the failure occurs during agent discovery;
these variables are not a valid workaround for this installed runtime.

### Root-cause status

The mechanism is confirmed on this host by the retained A/B run in
`bench/platform-results/gfx802-isolation/20260829T191952Z-14516/`:

* with both KFD GPU nodes present, `rocminfo` returned the generic HSA error;
* after unbinding `0000:03:00.0`, `rocminfo` exposed only `gfx906`;
* MIInfer device validation and Release/Debug GPU smoke tests passed;
* the post-cleanup KFD snapshot contained only `vega20`.

The gfx802 display GPU did not restore a usable GNOME session after rebinding,
so the validated operational procedure is to run the MI50 workload over SSH,
leave gfx802 unbound, and reboot afterward.

## Recovery options

The preferred durable recovery is to install a ROCr package containing the
fix from PR #4936, while keeping the MI50 identified as real `gfx906`. Until
that is available, use the validated gfx802 isolation procedure. Do not use
`HSA_OVERRIDE_GFX_VERSION`.

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

For a single-password, retained-results run that also tests MIInfer, use:

```bash
scripts/diagnose-gfx802-isolation.sh
```

The script requests `sudo` authentication once, stores raw command output and
environment captures under `bench/platform-results/gfx802-isolation/`, and
attempts to rebind the display GPU automatically on exit. Run it from a TTY
or remote session because unbinding `0000:03:00.0` may terminate the graphical
session.

The local keyboard/TTY may also become unusable because the same GPU owns the
kernel display framebuffer. SSH is therefore the preferred control channel.
To have the script restart Fedora's display manager after rebinding, use:

```bash
scripts/diagnose-gfx802-isolation.sh --restart-gdm
```

This ends the existing graphical session, so save any work first.

Because this host did not restore GNOME successfully after rebinding, the
preferred validation mode is to leave gfx802 detached and reboot afterward:

```bash
scripts/diagnose-gfx802-isolation.sh --leave-unbound
```

Run this over SSH. The script will retain the MI50-only results and will not
attempt a display recovery; reboot once the test has finished.

## Fedora development packages

The runtime packages do not include the CMake development metadata. Fedora
provides the required packages separately, and they are now installed on the
host:

```text
hipblas-devel-7.1.0-6.fc44
hipblas-common-devel-7.1.0-3.fc44
rocblas-devel-7.1.1-7.fc44
rocsolver-devel-7.1.1-4.fc44
```

The primary hipBLAS config is named `hipblas-config.cmake` and is installed
under `/usr/lib64/cmake/hipblas`; `hipblas-common` is under
`/usr/share/cmake/hipblas-common`. The system-wide installation was validated
for reference configuration with:

```bash
cmake -S . -B build-mi50-localhipblas \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_LTO=OFF -DGGML_VULKAN=OFF
```

The reference resolved `hipblas_DIR=/usr/lib64/cmake/hipblas` and
`hipblas-common_DIR=/usr/share/cmake/hipblas-common` from this installation;
no hardcoded library path or temporary prefix is required.

## Current gate status

```text
ROCr/HSA with both GPUs: BLOCKED by installed ROCr multi-GPU initialization bug
ROCr/HSA with gfx802 isolated: PASS
gfx906 in rocminfo: PASS with gfx802 isolated
generic HIP device access: PASS with gfx802 isolated
MIInfer GPU tests: PASS with gfx802 isolated
privileged gfx802 isolation A/B: CONFIRMED
hipBLAS CMake metadata: INSTALLED system-wide
reference CMake configure: PASS with system hipBLAS/rocBLAS development packages
reference build (llama-cli, llama-bench): PASS for gfx906
reference Qwen3-8B F16 smoke: PASS with 37/37 layers offloaded
reference PP512 / PP2048 / TG128 baseline: PASS; five repetitions each
```
