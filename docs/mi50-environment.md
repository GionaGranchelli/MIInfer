# Qualified MI50 development environment

The reproducible userspace is the OCI image built from
[`container/Containerfile`](../container/Containerfile). Its immutable ROCm
foundation is
`docker.io/kyuz0/rocm-toolbox-gfx906@sha256:fb0de294a2919ff6d503b4845e055c7fb80db5861954694ee5228e48a14411ec`.
That foundation is the gfx906 ROCm 7.2.1 toolbox; the image adds only the
MIInfer build metadata and a pinned CMake 3.30.5. The GPU, kernel and physical
host remain outside the image.

On either host, from the MIInfer checkout:

```sh
podman pull docker.io/kyuz0/rocm-toolbox-gfx906@sha256:fb0de294a2919ff6d503b4845e055c7fb80db5861954694ee5228e48a14411ec
podman build --network=host --pull=never -f container/Containerfile -t localhost/miinfer-dev:rocm-7.2.1 .
MIINFER_IMAGE=localhost/miinfer-dev:rocm-7.2.1 tools/check-qualified-host.sh \
  "$HOME/models/Qwen3.8-27B-Q4_K_M.gguf" | tee "evidence-$(hostname).txt"
```

The qualification command mounts the checkout and model, exposes `/dev/kfd`
and `/dev/dri`, isolates one `gfx906` ROCr device, builds host and `gfx906`
HIP configurations, runs the host test label, runs bounded GPU correctness
tests, and executes the 10-iteration vector benchmark. The model must have
SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

The two hosts intentionally retain different CPU, motherboard, PCIe, kernel,
clock, power and thermal values. Those values are reported by the command and
are evidence, not parity failures. The parity checks are the single `gfx906`
device, pinned image base digest, source SHA, model SHA, container HIP
toolchain, host tests, GPU correctness smoke and benchmark completion.
