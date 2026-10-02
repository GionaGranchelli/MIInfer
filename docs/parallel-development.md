# Parallel MI50 Development

## Purpose

MIInfer currently has two independent single-MI50 development hosts:

- **mi50-a:** HP Z840 + AMD Instinct MI50 32 GB
- **mi50-b:** Machinist X99 + AMD Instinct MI50 32 GB

The purpose of this setup is to finish the single-MI50 roadmap faster before
both GPUs are moved into one host for V3. The two machines are not a distributed
inference target.

## First prerequisite — reproducible environment

Do not start parallel M29 implementation until both hosts can run the same
qualified MIInfer environment.

Use one OCI image that works with Podman or Docker. Prefer pinning by image
digest rather than relying only on a mutable tag. The image should contain the
userspace pieces that must remain identical between hosts, including:

- ROCm/HIP userspace and compiler/toolchain required by MIInfer;
- build dependencies and build configuration;
- MIInfer runtime dependencies;
- benchmark/test tooling;
- the expected environment variables and runtime configuration.

The model remains an external artifact but must have the same SHA-256 on both
hosts. Benchmark inputs must also be identical.

The container does **not** make the physical machines identical. Record
host-specific kernel, CPU, motherboard/PCIe, cooling, clocks, power and
temperature information with benchmark evidence.

### Environment-parity exit gate

Both hosts must independently:

1. identify one MI50 32 GB / gfx906 device;
2. build the same MIInfer source SHA from the same OCI image digest;
3. verify the exact model SHA-256;
4. pass the same host/GPU correctness smoke;
5. run a small known benchmark matrix without errors;
6. capture the environment and hardware state used for the run.

Absolute timings do not have to match. The purpose is to prove that both hosts
can execute the same experiment contract reproducibly.

The canonical implementation is `container/Containerfile` plus
`tools/check-qualified-host.sh`. Build the image with `--pull=never` after the
base image has been resolved, then run the qualification command documented in
[`mi50-environment.md`](mi50-environment.md) on each host. Record the complete
stdout as host evidence and retain the exact source SHA, model SHA and image
base digest with it.

## Parallel work rules

After the environment gate passes, both hosts are peers. Assign whichever
dependency-ready goal provides the most value.

Every Codex workstream receives a small goal packet:

```text
GOAL
HOST
BASE SHA
BRANCH
QUESTION
AUTHORIZED SCOPE
DO NOT TOUCH
REQUIRED EVIDENCE
EXIT CONDITION
```

Performance comparisons are always same-host A/B:

```text
mi50-a: baseline A -> candidate A -> delta A
mi50-b: baseline B -> candidate B -> delta B
```

Never use a baseline from one host to judge a candidate measured on the other.

Parallel implementation is allowed only when dependencies are clear. If two
tasks share an architectural interface, first land/freeze that contract, then
branch both tasks from the contract commit.

Agents stop at their assigned exit condition. They do not continue into the
next roadmap item without a new goal.

## Initial parallel split

Once environment parity passes:

```text
mi50-a
  M29.0 — current MIInfer long-context baseline

mi50-b
  M29.0A — current gfx906 external-reference calibration
```

When M29.0A closes, mi50-b can move to M29.1 HIP-VMM feasibility while mi50-a
continues any remaining long-context baseline work.

Later work should be parallelized only where the current roadmap dependencies
allow it. Important milestone-closing or surprising results should be reproduced
on the second host before final acceptance; routine exploratory work does not
need duplicate execution.

## End of the two-host phase

Keep both MI50s in separate machines through the single-MI50 roadmap:

```text
M29 -> M30 -> M31
```

After M31 freezes the qualified N=1 baseline and the critical gates have been
reproduced across the two hosts, move both MI50s into one machine and begin V3
dual-MI50 topology qualification.
