# M31-0004 — GDN computation findings

## Scope and evidence

This is a source-level trace of the production path, not a GPU profile. The
review covered `PrefillV2Model::generate` / `forward`,
`PrefillV2RecurrentLayer::forward`, and
`gfx906/kernels/m12_gdn_chunk.hip::launch_mx_gdn_chunk`. The CPU oracle is
`bench/m12_gdn_oracle.cpp`.

## Production hot path

For each 512-token macro tile, each of the 48 recurrent layers executes its
input normalization, Q/K/V and gate projections, convolution and normalization,
GDN recurrence, output projection, residual/FFN projections, SwiGLU and down
projection. The current path uses about 20 HIP launches per Q4 layer and 21 for
the Q6K QKV variant: approximately 960–1,008 recurrent-layer launches per
macro tile, before embedding, attention, and final normalization. The model
does not allocate temporary HIP buffers per tile; workspace is reused.

The GDN kernel processes up to 512 tokens as eight sequential 64-token scans
inside each CTA. Each CTA keeps its assigned recurrent-state shard in registers,
loads it once, applies the recurrence in token order, then writes it once. This
is the required state dependency for this formulation; no source evidence shows
that the recurrence itself can be skipped or reordered without changing
semantics. A state shard's update is about 114.8K floating-point operations per
value-head/token, or about 5.51M per recurrent layer/token for 48 value heads.

The CPU oracle implements the same recurrence. An alternative WY formulation
has prior CPU output/state evidence, but that does not establish that replacing
the production sequential recurrence is faster or safe on gfx906.

## Confirmed opportunity and decision

With the default TC2 layout, the source dispatches 32 state-column tiles and two
waves per value head. Each tile requests the same 128-element Q and K vectors;
the logical source-level repeated-read demand is about 3 MiB per token per
layer versus 16 KiB of distinct Q/K data. This is a logical-load estimate, not
measured HBM traffic; cache reuse may substantially reduce physical traffic.

TC4 halves the tile count and these repeated requests, but the existing
interleaved P512 experiment measured it 1.63% slower at the median. It is not a
defensible production default change. No recurrent arithmetic, projection,
normalization, or state transfer was removed in this milestone.

The implemented host-side optimization is checkpoint-prefix bookkeeping, not
GDN arithmetic: monotonic 512-token boundary saves now append the known token
suffix and extend the FNV-1a fingerprint rather than re-copying and re-hashing
the whole prefix. At 128K, the hash alone visits 16,842,752 tokens before and
131,072 after (16,711,680 fewer; about 99.2% fewer). Full-prefix vector
assignment is replaced with appending the 131,072 new token payload; occasional
capacity growth can relocate elements. GPU checkpoint captures remain
unchanged. No wall-time improvement is claimed without MI50 measurement.

## Remaining computational opportunities

The strongest kernel opportunity is to reduce repeated Q/K loads across state
column tiles, but any cooperative-sharing variant needs a new focused gfx906
measurement and parity test. Fusing projection/normalization stages is a
secondary candidate; source inspection alone does not establish that it
reduces total work or preserves the current quantized paths. Do not replace the
recurrence with a parallel scan based only on its theoretical parallelism.

## Hardware validation

No GPU work was run for M31-0004. Kernel parity, the TC2/TC4 choice, and all
performance implications remain pending the existing hardware-safety gate.
