# M31-ACCEL-0001 — 64K persistent multi-turn harness

This bounded source-level workload starts at exactly 65,536 prompt tokens and
then exercises eight sequential turns, each appending exactly 16 tokens
including the previous turn's generated IDs and deterministic tool-result-like
IDs. The final prompt is 65,648 tokens. Generation is capped at two tokens per
turn. Turn four saves and restores the supported V2-0028 persistent session,
then continues from the restored prefix.

Each line of output is emitted directly from its corresponding native request,
so per-turn timings and state counts need no request-ID join against a server
log. The harness checks exact turn lengths, prefix-reuse counts,
generated-token bounds, vocabulary IDs, finite logits, four-token repetition
runs, and all 48 recurrent position counters. It fingerprints the active GDN
states and KV bytes before save and after restore and requires byte identity.
The KV API does not expose a logical length counter; continuity is checked
through the restored state fingerprint, exact token history, successful
prefix reuse, and continuation after restore.

The transcript uses deterministic synthetic token IDs, not decoded ChatML or
real repository text. It is a 64K persistence and state-continuity harness,
not an agent-quality or production qualification result. Snapshot/fork/COW
APIs are absent on this pinned source; the existing persistent-session
save/load API provides the restore checkpoint.

## Build and self-check

```sh
cmake -S . -B build -DMIINFER_ENABLE_HIP=ON -DMIINFER_BUILD_BENCHMARKS=ON
cmake --build build --target miinfer-m31-accel-0001-agent64k -j
./build/miinfer-m31-accel-0001-agent64k --self-check
```

The self-check does not load a model or initialize a GPU. It verifies the
initial 65,536-token prompt, eight-turn append plan, final prompt length,
KV headroom, bounded restore-turn generation, and repetition-loop detector.

## MI50 execution

After validating the host thermal and VRAM guards, run with the pinned model
and a writable session path:

```sh
mkdir -p /mnt/lc9-lake/m31-accel-0001
./build/miinfer-m31-accel-0001-agent64k \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /mnt/lc9-lake/m31-accel-0001/turn4.miinfer
```

The initial prompt is exactly 65,536 tokens; KV capacity is 65,792 to allow
eight small continuations and bounded decode. This command performs a GPU
workload and was not run as part of the source-only implementation.
