# EXP-0312 — M25-K Mx attention Q/K projection

## Hypothesis

The remaining attention Q/K projection tail should improve if the existing
Mx Q4 repacked MMQ kernel replaces the M23 Q4 repacked reader for the
combined Q+K output.

## Motivation

M25-H/I reduced the attention FFN and O tails, leaving Q/K as the largest
measured attention projection stage. The pinned mx implementation provides a
proven Q4 repacked layout and MMQ geometry, so this was tested as an isolated
port.

## Baseline

The existing combined M23 Q4 Q/K repacked path at layer 3, B512.

## Candidate

An opt-in path packed Q and K into one combined Mx Q4 layout, uploaded it
alongside the retained M23 decode layout, and used the existing Mx Q4
repacked MMQ kernel for wide prefill. The candidate was removed after the
measurement.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M
- layer 3, B512, synthetic attention-layer bakeoff input
- `MIINFER_MX_Q8_BATCH=0`, `MIINFER_MX_PIPELINE=1`
- pinned mx commit `2e9d29fe736969160f17476ec6f0a6298cee6966`

## Benchmark

```sh
build/mi50-release/miinfer-m24-attention-layer-bakeoff \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf 512 mx_qk
build/mi50-release/miinfer-m24-attention-layer-bakeoff \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf 512 control
```

## Correctness

The candidate completed with finite output and scalar parity of max absolute
error `0.263`, RMSE `0.010`. This passed the existing `1.0` max-error layer
tolerance, but was not sufficient to justify its extra memory and latency.

## Results

| path | Q/K stage | total layer | tracked layer allocation |
| --- | ---: | ---: | ---: |
| M23 control | `10.328 ms` | `63.340 ms` | `561,950,608 B` |
| Mx Q/K candidate | `13.785 ms` | `66.815 ms` | `610,529,168 B` |

The candidate was `33.5%` slower for Q/K and `5.5%` slower for the complete
attention layer, while adding `48,578,560 B` of layer allocation.

## Interpretation

The existing Mx MMQ geometry is not a drop-in win for this combined
attention shape on gfx906. The slower result outweighs the source-level
similarity to mx.

## Decision

**REJECT.** The candidate code was removed; M23 Q/K remains selected.

## Follow-up

Do not revisit combined Mx Q/K without a new measured bottleneck or a
different gfx906-specific kernel mapping that addresses the row/working-set
cost.
