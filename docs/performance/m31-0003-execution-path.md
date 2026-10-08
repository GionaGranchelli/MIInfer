# M31-0003 — Production execution path

## Scope and source identity

This trace follows the current working tree, not only its commit. `HEAD` is
`dc7370b76e7f3fdb01c2c810cd07181703bef2d2`; the repository already had
uncommitted edits in the production prefill/runtime files and M31 artifacts
before this investigation. Those edits are preserved. No GPU workload was run.

The current production CLI paths are `miinfer run`, `miinfer chat`, and
`miinfer serve`. Each loads `miinfer::prefill_v2::PrefillV2Model` with FP16 K/V
and routes generation through `PrefillV2Model::generate`:

| Entry | Evidence | Route |
|---|---|---|
| `run` | `tools/miinfer_cli.cpp:3964-4040` | ChatML tokenize → `PrefillV2Model` → `engine.generate`; HIP graph enabled |
| `chat` | `tools/miinfer_cli.cpp:4669-4725` | Persistent in-process engine → `engine.generate`; prefix reuse enabled |
| `serve` | `tools/miinfer_cli.cpp:4957-5048`, request handler below | One persistent `PrefillV2Model` → request generation |
| M31 checkpointed benchmark | `bench/m31_0002_checkpointed_call.cpp:197` | Direct `model.generate` using the same model class |

`Qwen35RuntimeEngine` in `tools/miinfer_cli.cpp` is a separate legacy and
diagnostic path; it is not the engine instantiated by these production
entrypoints. Future findings must not attribute its layer-major code to the
current `run`/`chat`/`serve` inference route.

## Cold prefill

`PrefillV2Model::generate` checks in-memory and optional disk prefix reuse. A
cold request resets recurrent state when `reset_state_before` is set (the
production default), then advances the prompt in chunks of at most 512 tokens.
For each chunk it enqueues an H2D token-ID copy and calls `forward` at that
chunk's base position. The forward path traverses the model's 16 topology
blocks: 48 recurrent/GDN layers and 16 full-attention/GQA layers. The model
profile and allocation evidence identify those as the relevant components.

After prefill, the LM head computes logits, copies the full logits vector to
host memory, synchronizes, and samples the first token on CPU. See
`src/prefill_v2/model.cpp:971-1215`, `:541-627`, and `:762-790`.

## Persistent-context suffix prefill

When an exact reusable prefix is present, the code reuses the prefix's GQA KV
and restores GDN state only when the saved state is not already GPU-resident.
It processes only the suffix in <=512-token chunks. Full 512-token chunks may
use a captured suffix HIP graph; residual chunks use the ordinary `forward`
path. It synchronizes the stream at the end of suffix prefill. This route is
implemented in `src/prefill_v2/model.cpp:987-1105` in the current dirty tree.

Disk session restore and in-memory `ReusableContext` are distinct mechanisms;
do not treat their restore/copy costs as one path. The former M31 persistent
context gate passed on its recorded workload, but that does not measure every
production suffix shape.

## One-token decode

Production `GenerateOptions` defaults to HIP graph and the CLI explicitly keeps
it enabled. After first-token sampling, generation captures/replays the decode
graph once per next-token iteration. Each iteration copies the full logits
vector D2H and synchronizes before CPU sampling, then uploads updated
`DeviceDecodeState` before the next replay. The no-graph fallback uploads one
input token, runs embedding plus all 16 blocks, computes logits/argmax, copies
one token D2H and synchronizes (`src/prefill_v2/model.cpp:792-872,1215-1432`).

These transfer/synchronization sites are confirmed by source. Their long-context
latency contribution is not established by source inspection. The available
8K synchronized decode profile attributed 65.5% to GDN layers, 26.7% to
attention layers, 3.3% to logits, and 4.2% residual; it is one Machinist sample,
not a throughput profile (`results-m31-0002-machinist-8k-profiled-decode.txt`).

## Existing long-context evidence

The Z840 checkpointed funnel records 8K/32K/64K prefill at 216.971/152.100/
107.254 tok/s and model-owned VRAM at 25.690/27.300/29.448 GB. Same-host
llama.cpp references are 227.319/200.746/173.832 tok/s, but prompts/run
conditions were not fully matched. The 8K component profile attributes about
66% to three GDN groups and 34% to GQA; the 32K profile aborted at a 100°C
guard before a component row. Thus the increasing long-context gap is real in
the recorded directional funnel, but its 32K/64K component cause remains
unattributed. See `results-m31-0002-z840-context-funnel.md`.

The protocol/worktree still records M31-0002 attribution as the active
performance PRIMARY, including the unresolved 32K/64K profile gate. This
M31-0003 task is user-directed source investigation using existing evidence;
it does not retroactively turn an 8K profile into long-context attribution.
