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
runs, and all 48 recurrent position counters. The shared production sampler now
fails the request on any non-finite logit; the no-GPU self-check covers finite
values, NaN, and both infinities. The HTTP streaming workload does not receive
generated token IDs. The server checks the IDs it already holds and writes
only a repeated-run boolean to its private request telemetry; the production
runner fails closed if a loop is detected or that signal is missing. No token
IDs are exposed. The harness fingerprints active GDN states and KV bytes
before save and after restore and requires byte identity.
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
cmake -S . -B build/m31-accel-0001 \
  -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.1 -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 -DMIINFER_HIP_ARCHITECTURE=gfx906 \
  -DMIINFER_TARGET_ARCH=gfx906 -DMIINFER_ENABLE_HIP=ON \
  -DMIINFER_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/m31-accel-0001 \
  --target miinfer miinfer-m31-accel-0001-agent64k -j4
./build/m31-accel-0001/miinfer-m31-accel-0001-agent64k --self-check
```

Run these commands in the pinned ROCm 7.2.1 container. For an offline build,
mount a populated nlohmann/json source directory at `/deps/nlohmann_json` and
add `-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/deps/nlohmann_json` to configure.

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
pads token IDs. For each request sent, the tokenizer helper can also emit the
exact ChatML token IDs; the workload saves these beside the exact request body
in a `.tokens.json` artifact tied to the request SHA-256, pinned GGUF SHA-256,
workload UUID, request sequence, and canonical little-endian token-ID SHA-256.
This lets later analysis derive the true common
token prefix across consecutive requests, including tokenizer boundary effects.
The decode benchmark accepts one artifact with `--prompt-token-ids`; the prefix
reuse benchmark accepts two adjacent artifacts with `--request-pair`, derives
their exact token-ID LCP, and compares cold second-request prefill with reuse of
the first request's saved state. These modes validate artifact hashes, counts,
vocabulary bounds, and nonempty prefix/suffix constraints before benchmarking.
For guarded execution, stage the `.tokens.json` files in a new capture
directory. Set `M31_DECODE_TOKEN_IDS` to one filename for `decode`; set
`M31_PREFIX_FIRST_TOKEN_IDS` and `M31_PREFIX_SECOND_TOKEN_IDS` to two
consecutive filenames for `prefix64k`. The guard validates that both files are
inside the capture directory before starting the GPU container.

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

The self-check also accepts a complete SSE response and rejects one missing
`[DONE]`. During a live run the JSON artifact is atomically updated before each
request with its active turn, phase, request hash, and tokenizer count; the
exact serialized API body and corresponding token-ID artifact are saved under a sibling
`agent-workload.inputs/` directory before it is sent. Completed request records
keep the artifact path and SHA-256. The guarded launcher supplies the verified
model SHA; the workload records it with the source and runner-script hashes.
If a request fails, the JSON artifact is marked `failed` and retains the active
request identity and failure details.

Before a guarded agent run, build the tokenizer counter into the shared
production build directory used by the server:

```sh
cmake --build build/m31-accel-0001 --target miinfer-m31-agent-token-count -j
```

For the production-path run, use the already guarded 81,920-token serving configuration
and point `--server-log` at that server's stderr log:

```sh
python3 bench/m31_accel_0001_agent_workload.py \
  --model-path /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --token-counter build/m31-accel-0001/miinfer-m31-agent-token-count \
  --server-log results/m31-accel-0001/server.log \
  --guard-telemetry results/m31-accel-0001/telemetry.log \
  --output results/m31-accel-0001/agent-workload.json
```

The report captures per-turn client TTFT and wall time, server decode
tokens/second, reused-prefix and suffix counts, and the largest VRAM use seen
in request-boundary samples plus the guarded runner's timestamped `rocm-smi`
samples that fall within each request window. Each turn records its request
window timestamps, in-window guard sample count, boundary sample count, and
sample sources. This is a sampled maximum, not a continuous hardware peak; the
guard still retains its complete 2-second telemetry log separately. Live runs
fail closed if the structured guard telemetry is absent, malformed, or has no
sample inside a request window. The serving workload was not run
for this source-only change; the current workspace has no matching Qwen3.8
GGUF, so exact 65,536-token prompt construction has not been runtime-verified
here.

The final exit gate requires each later turn's first request to have both a
nonzero reused prefix and a nonzero dispatched suffix, in addition to the
visible-answer, generation, and telemetry gates.
This operational gate does not establish cold-versus-reuse token parity; the
known M31-0002 divergence remains unresolved, so the run cannot qualify exact
reuse correctness by itself.

The production server enables the supported persistent snapshot path for this
workload: `run-agent-server.sh` sets `MIINFER_SESSION_REUSE=1` and a writable
`--session-dir`; the streaming API handler enables prefix reuse and
`cache_prefix_after`. `PrefillV2Model::generate()` saves the processed prompt
checkpoint and restores a matching prefix for a later request. The live
request log exposes `restore_ms`, reused GQA tokens, and GDN checkpoint/final
positions. The separate native harness checks byte identity across an explicit
turn-four save/load, but with synthetic token IDs. Neither path has completed
the guarded 64K GPU correctness run yet.

On Z840, launch the production-path workload only through the guarded runner:

```sh
capture=/mnt/lc9-lake/m31-accel-0001-captures/agent64k-01
session=m31-accel-0001-agent64k-01
mkdir -p "$capture"
tmux new-session -d -s "$session" \
  "cd /mnt/lc9-lake/m31-accel-0001-cursor-build && exec ./experiments/m31-accel-0001/run-guarded.sh '$capture' agent"
tmux set-option -w -t "$session" remain-on-exit on
tmux list-panes -t "$session" -F '#{pane_pid}' > "$capture/guard.pid"
```

The server allocates an 81,920-token capacity so the exact 65,536-token initial
prompt can generate and retain eight bounded turns within the observed MI50
memory budget. This is a 64K-start workload, not a 128K prompt or performance
run. The tmux session keeps the thermal monitor attached to the detached
container if SSH disconnects; the pane PID, runner logs, telemetry, and container
exit status are retained in the capture directory. A signal to the monitor stops
the container. The guard stops at the 85 C warning threshold pending operator
inspection, at 90 C, or at the VRAM floor; use a new capture directory after
inspection/cooldown.
