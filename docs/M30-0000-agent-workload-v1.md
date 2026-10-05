# M30-0000 canonical agent workload

```text
M30_AGENT_WORKLOAD_VERSION=m30-agent-v1
M30_0000_BASE_SHA=23deab2eda689fc0cfc21782d3f6df1303bee2ee
```

This is a measurement workload only. It does not add or enable a reuse
mechanism.

## Frozen shape

The workload is a 40-turn deterministic replay driven through the production
OpenAI-compatible serving path. Turns are grouped as:

| Turns | Activity |
|---:|---|
| 1–10 | repository orientation and architecture inspection |
| 11–20 | implementation planning and focused code reading |
| 21–30 | test interpretation and bounded failure diagnosis |
| 31–40 | regression review and release-readiness synthesis |

Each turn preserves the complete prior message history, adds a small user
task, performs one deterministic repository-tool exchange, and requests a
short assistant continuation. Tool results are bounded excerpts from the
candidate checkout, not synthetic token padding.

The fixed repository context is sourced from the candidate checkout with the
following byte limits:

```text
README.md                                      12000
docs/architecture.md                           10000	docs/interactive-serving.md                    9000
include/miinfer/prefill_v2/model.hpp           10000
src/prefill_v2/model.cpp                        14000
src/openai_api.cpp                               7000
tools/miinfer_cli.cpp                           12000
```

The harness rotates real line-range reads across those files for turns 1–40.
It uses the repository's `build_chatml()` and model tokenizer through the
production server; it does not implement a second tokenizer. Requests use
temperature 0, top-p 1, top-k 1, and bounded generation so the transcript is
replayable. The server runs with its current production session-reuse setting,
and the report distinguishes logical prompt tokens from authoritative
`suffix_tokens_dispatched`.

## Required per-turn evidence

The harness records prompt tokens, reused prefix, suffix tokens dispatched,
GDN restore time, suffix-prefill time, TTFT, prefill/decode/total wall time,
generated tokens, context size, client wall time, and VRAM before/after. The
server telemetry additionally records the model/configuration and checkpoint
state size.

Session totals must report logical prompt tokens, physically processed prompt
tokens, generated tokens, peak context, peak VRAM, and total wall-clock time.

The workload is allowed to end below 128K. Context checkpoints are reported
when naturally crossed; no turn is enlarged solely to hit a power-of-two
target.

## Execution entry point

```text
python3 bench/v2_0045_agent_workload.py --m30 \
  --url http://127.0.0.1:8087/v1/chat/completions \
  --output results/m30-0000/agent-workload.json
```

The existing three-turn V2-0045 mode remains available without `--m30`; the
M30 version is the frozen 40-turn workload and is the only one used for this
goal's baseline.
