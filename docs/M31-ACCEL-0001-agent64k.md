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
After model allocation it fails closed unless at least 1 GiB of VRAM remains
free. It samples HIP free/total VRAM after each turn and reports the largest
sampled use; this is a discrete in-process sample, not an external telemetry
peak. The guard and each completed turn line flush immediately so an
externally stopped diagnostic preserves its last completed state.
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

## Production-path coding-agent workload

`bench/m31_accel_0001_agent_workload.py` reuses the M30 OpenAI streaming and
repository-tool flow for eight turns. A host-only helper counts the exact
`build_chatml()` output using MIInfer's Qwen tokenizer; the prompt builder
extends only with tracked repository excerpts and stops unless the count is
exactly 65,536. The first live request must report the same count. It never
constructs or pads token IDs.

Build and run the tokenizer-only self-check without initializing HIP:

```sh
cmake -S . -B build-agent64k-host \
  -DMIINFER_ENABLE_HIP=OFF -DMIINFER_BUILD_BENCHMARKS=ON \
  -DMIINFER_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-agent64k-host --target miinfer-m31-agent-token-count -j
python3 bench/m31_accel_0001_agent_workload.py \
  --model-path /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --token-counter build-agent64k-host/miinfer-m31-agent-token-count \
  --self-check
```

For the production-path run, use the already guarded 128K serving configuration
and point `--server-log` at that server's stderr log:

```sh
python3 bench/m31_accel_0001_agent_workload.py \
  --model-path /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --token-counter build-agent64k-host/miinfer-m31-agent-token-count \
  --server-log results/m31-accel-0001/server-128k.log \
  --output results/m31-accel-0001/agent-workload.json
```

The report captures per-turn client TTFT and wall time, server decode
tokens/second, reused-prefix and suffix counts, and the largest VRAM use seen
in `rocm-smi` samples taken before and after each request. Those are sampled
observations, not a continuous hardware peak. The serving workload was not run
for this source-only change; the current workspace has no matching Qwen3.8
GGUF, so exact 65,536-token prompt construction has not been runtime-verified
here.

The final exit gate requires each later turn's first request to have both a
nonzero reused prefix and a nonzero dispatched suffix, in addition to the
visible-answer, generation, and telemetry gates.
This operational gate does not establish cold-versus-reuse token parity; the
known M31-0002 divergence remains unresolved, so the run cannot qualify exact
reuse correctness by itself.

On Z840, launch the production-path workload only through the guarded runner:

```sh
experiments/m31-accel-0001/run-guarded.sh \
  /mnt/lc9-lake/m31-accel-0001-captures/agent64k-01 agent
```

The server allocates a 131,072-token capacity so the exact 65,536-token initial
prompt can generate and retain eight turns; the input, measured prompt, and
expected ending remain below 128K. This is a 64K-start workload, not a 128K
prompt or performance run. The runner records 85 C warnings, stops at 90 C or
the VRAM floor, and preserves the server/client output and actual container exit
status.
