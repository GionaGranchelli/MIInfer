# Qualified MI50 development environment

The reproducible userspace is the OCI image built from
[`container/Containerfile`](../container/Containerfile). Its ROCm base is
`docker.io/rocm/dev-ubuntu-22.04@sha256:42851dac319afce41cf993e25f95005b7f2cd0a0f6abd32ad8f25cd876ec56df`.
The image contains only the compiler, CMake, build tools and Python needed by
MIInfer; the GPU, kernel and physical host remain outside the image.

On either host, from the MIInfer checkout:

```sh
podman pull docker.io/rocm/dev-ubuntu-22.04@sha256:42851dac319afce41cf993e25f95005b7f2cd0a0f6abd32ad8f25cd876ec56df
podman build --pull=never -f container/Containerfile -t localhost/miinfer-dev:rocm-7.2.1 .
MIINFER_IMAGE=localhost/miinfer-dev:rocm-7.2.1 tools/check-qualified-host.sh \
  "$HOME/models/Qwen3.8-27B-Q4_K_M.gguf" | tee "evidence-$(hostname).txt"
```

The qualification command mounts the checkout and model, exposes `/dev/kfd`
and `/dev/dri`, builds host and `gfx906` HIP configurations, runs the host test
label, runs bounded GPU correctness tests, and executes the 10-iteration
vector benchmark. The model must have SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

The two hosts intentionally retain different CPU, motherboard, PCIe, kernel,
clock, power and thermal values. Those values are reported by the command and
are evidence, not parity failures. The parity checks are the single `gfx906`
device, pinned image base digest, source SHA, model SHA, container HIP
toolchain, host tests, GPU correctness smoke and benchmark completion.
