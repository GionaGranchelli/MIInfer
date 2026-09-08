# EXP-0256 — M12 chunkwise Gated DeltaNet oracle

> **Superseded geometry note:** early surrounding M12 records described 32
> value heads. The qualified geometry is 16 key heads / 48 value heads /
> state 128; see the re-evaluation recorded in EXP-0258 and EXP-0261.

## Hypothesis

The FLA/Hugging Face chunkwise Gated DeltaNet formulation can reproduce the
current token-recurrent state transition and output while reducing the
sequence-dependent work to one sequential step per chunk.

## Candidate

Implemented a host-only mathematical oracle for the exact MIInfer head
geometry:

- 16 key heads
- 32 value heads, mapped by `value_head % key_heads`
- state size 128
- 128 tokens
- chunk size 64
- nonzero initial state

The candidate follows the current Qwen3.5 chunk equations: cumulative
decay, lower-triangular WY solve, intra-chunk causal attention, and one
state update per chunk. It does not implement a prefix scan or change runtime
code.

The formulation was cross-checked against the current Hugging Face
Qwen3.5 implementation and the FLA DeltaNet design. The upstream implementation
uses cumulative decay, a unit-lower-triangular solve, intra-chunk attention,
and a sequential scan over chunk boundary states.

## Baseline / oracle

The independent token reference applies, for every token:

```text
S <- g * S
delta <- beta * (v - k^T S)
S <- S + k outer delta
o <- q^T S / sqrt(128)
```

Both implementations use FP32 arithmetic and the same normalized Q/K inputs,
beta, multiplicative decay, and nonzero initial state. The runtime and oracle
store decay as a positive multiplicative factor; the GPU chunk kernel converts
it to log space only for the cumulative product calculation.

## Environment

- Host Release build: `mi50-release`
- Compiler: project C++20 toolchain
- No GPU or model/runtime integration used

## Correctness

The oracle executable completed successfully:

```json
{"tokens":128,"chunk":64,"key_heads":16,"value_heads":32,
 "state_size":128,"max_output_error":0.000000014,
 "max_state_error":0.000000149}
```

The final-state error is `1.49e-7`; output error is `1.40e-8`. The previous
version failed at `6.84e-3` until the missing cumulative query decay was found,
so the test has already caught a substantive algebra error.

## Decision

**KEEP AS THE M12 MATHEMATICAL ORACLE.** The chunkwise form is correct enough
to justify a gfx906 prototype. No production integration or performance claim
is made.

## Follow-up

Implement the smallest gfx906 chunk kernel for one recurrent layer and compare
its state/output against this oracle and the existing token kernel before
combining it with EXP-0255's B128+ matrix projection path.

References:

- [Hugging Face Qwen3.5 chunk implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L2674-L2892)
- [FLA DeltaNet implementation](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py)

## Re-evaluation — 2026-09-08

The initial record used an incorrect 32-value-head geometry. The Qwen3.8
runtime contract is 16 key heads, 48 value heads, and state size 128; the
host oracle and GPU prototype were corrected without changing the chunkwise
algebra. The corrected host run still reports 1.4e-8 maximum output error and
1.5e-7 maximum final-state error. The original 32-head measurements are
historical and are superseded by the corrected 48-head result.
