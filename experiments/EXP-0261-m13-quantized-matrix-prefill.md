# EXP-0261 — M13 quantized matrix prefill

## Hypothesis

A gfx906-packed Q4_K/Q6_K projection can reuse each decoded weight tile across
eight prompt activations without materializing an FP16 matrix, preserving the
M12 matrix-parallelism benefit inside the MI50 VRAM budget.

## Baseline

The existing native packed-Q4/Q6 path executes repeated B=4 projections. The
M12 combined GDN+dense path remains the production-shaped reference at
52.99 tok/s P512, opt-in only.

## Candidate

The candidate keeps `Q4KWaveTile`/`Q6KWaveTile` in device memory and extends
the existing tile decoder to an eight-token batched dispatch. It accepts an
explicit Q8_1 input stride so the prefill workspace can retain its FFN-sized
allocation. No FP16 weight buffer or decode-path change is involved.

## Environment

- GPU: AMD Instinct MI50 / gfx906
- Model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
- Build: `mi50-release`, Release
- Prompt profile: exact P512, layer-major, GDN chunkwise + dense M12 paths
- Baseline commit: `27ebb03`

## Correctness

The isolated B64 candidate output matched the repeated-B4 output exactly for
both Q4_K and Q6_K cases. The first harness attempt passed a host pointer to a
device quantizer and faulted; that harness bug was fixed before collecting the
results below. The generated GPU core dumps were removed.

## A0 profile

The combined M12 path was re-profiled at exact P512 with graphs disabled and
128-token scheduling. It completed in `9897.51 ms` (`51.73 tok/s`). Recurrent
layers were approximately `31.3 ms` prepare, `15.8 ms` ordered, and `98.3 ms`
tail each; attention layers were approximately `28.3 ms` prepare, `41.4 ms`
ordered, and `104–115 ms` tail. This confirms that the deferred projection
tail remains the large target, while GDN is already sufficiently improved.

## Results

The metric is direct quantized multi-token time versus the existing repeated
B4 path; both include the same packed weights and Q8 inputs.

| Type | B64 | B128 | B256 | B512 | B2048 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K direct / B4 | 0.863x | 0.863x | 0.862x | 0.861x | 0.860x |
| Q6_K direct / B4 | 0.695x | 0.695x | 0.712x | 0.717x | 0.716x |

The Q4 case is the exact dominant `blk.8.ffn_down.weight` shape `[17408,
5120]`. The Q6 case is `blk.0.attn_qkv.weight` `[5120,10240]`, the largest
compatible Q6 projection available in this model. Raw measurements and the
benchmark executable are retained in `bench/m13_quant_mm_bench.cpp`.

## Interpretation

The eight-token version reduces dispatch count but increases register and
per-wave work enough to lose to the shape-specific B4 decoder. Keeping the
weights quantized is not sufficient by itself; the tile-to-token mapping must
also preserve the existing occupancy/arithmetic balance.

## Decision

**REJECT.** Do not integrate this candidate into prefill. Do not redesign GDN,
add FP16 gate/up copies, or spend another M13 branch on larger unmeasured
batch templates. The direct quantized MMQ promotion gate failed before
production integration, so M13 is the hard stop for single-MI50 prefill
kernel research.

## Follow-up

Keep M12's opt-in 52.99 tok/s composition as the final research reference.
Move the roadmap to packaging, serving/UI, and multi-MI50 work only after the
existing M12 path receives its final qualification and release hygiene.
