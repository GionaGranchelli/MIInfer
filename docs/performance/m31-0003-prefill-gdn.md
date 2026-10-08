# M31-0003 — Prefill and GDN investigation

## Current path

Cold prefill dispatches at most 512 token IDs per `forward()` call through 16
topology blocks: 48 recurrent/GDN layers and 16 GQA layers. The selected GDN
implementation is the MX chunk scan in `RecurrentLayer::forward`; its temporal
scan is sequential within each workgroup and linear in chunk length. The M12
matrix-style chunk path is not selected by default; it is an explicit research
alternative. Historical EXP-0305 favored MX by 17.3% on P512, which is not
long-context evidence.

Suffix prefill processes only the new suffix after exact prefix matching. Its
kernel route differs: full 512-token suffix chunks can run the captured suffix
graph and pass `DevicePrefillState`; residual/eager chunks call ordinary
`forward()`. No source evidence shows a matched prefix being recomputed.

## Evidence and candidate inefficiencies

| Finding | Classification | Evidence / limit |
|---|---|---|
| GDN contributes about 66% of the 8K profiled prefill; GQA about 34%. | Confirmed at 8K only | Existing Z840 profile. The 32K guarded profile stopped at 100°C before a component row; 64K has no component attribution. |
| Q/K projections are produced for 16 key heads and consumed across 48 value heads (three value heads per key head). | Confirmed source structure | Repeated Q/K load/consumer work is visible; HBM savings from sharing are unmeasured. |
| A more time-parallel/shared GDN algorithm could reduce repeated Q/K work. | Strong hypothesis | State-update dependency and register/shared-memory pressure create correctness and occupancy risk; requires parity and real-device measurement. |
| Q6 normalized activation is re-quantized for Q4 gate projection. | Confirmed, not proven redundant | Formats differ by consumer; removing conversion without a numeric contract would be unsafe. |
| Cold prefill uses captured HIP graphs by default. | False | Graph capture applies to eligible full suffix chunks; cold chunks take ordinary forward. |

For suffix attention, V2-0043 is eligible only with a positive base position,
FP16 KV, no `DevicePrefillState`, and token count divisible by 16. Cold first
chunks and graph suffixes therefore bypass it. This supports documenting
separate cold and suffix routes, not broad claims that a particular optimized
kernel covers both.

Historical direct-suffix correctness evidence is mixed: an earlier integrated
pass is followed by a direct suffix mismatch and integrated parity failure in
later artifacts. Do not assign that mismatch to GDN based on source inspection.
The host-only M12 oracle validates its CPU reference calculation, not HIP
execution.

## Decision

Prioritize future 32K/64K component attribution, then investigate Q/K reuse
across GDN value heads with state/output parity checks. Do not switch to M12,
reopen the known physical-B512 projection-width failure, or optimize from the
single 8K profile. No GPU benchmark or cooling test was run for this report.
