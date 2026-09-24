# EXP-0391 — P512 recurrent/attention variance split

## Question

Determine whether the EXP-0390 GPU-runtime expansion is predominantly in the
48 recurrent layers, 16 attention layers, both, or outside those families.

## EXP-0390 basis

EXP-0390 measured fast/slow P512 GPU timeline medians of `3072.31/4941.93 ms`
with a `+1869.63 ms` expansion. Host submission stayed stable and the mx
sentinel stayed stable, so host submission attribution was closed. B128 remains
unauthorized; B64 is rejected and B4 is blocked.

## Instrumentation contract and perturbation gate

The attempted diagnostic added stream event boundaries for the 16 natural
three-recurrent-plus-one-attention groups and retained the EXP-0390/0376
measurements. It did not add synchronization or change routes. However, the
first run showed `prefill=4302.62 ms`, `host_submit=344.388 ms`, and
`gpu_timeline=4302.20 ms`, compared with the EXP-0390 host interval of roughly
50–68 ms. The family event recording itself therefore materially perturbed the
host submission interval. Although it produced provisional spans
`recurrent=2758.11 ms` and `attention=1213.32 ms`, those values are not
interpreted as evidence because the perturbation gate failed.

The family instrumentation was removed after this gate failure. No family
split is claimed, and no individual stage or kernel was inspected.

## Decision

**Classification: FAMILY_INSTRUMENTATION_PERTURBS_TARGET.** The requested
coarse family event design is too invasive on this P512 path when combined with
the existing timing boundary. It cannot distinguish recurrent from attention
without changing the target workload.

Family tracking: **INSUFFICIENT**.

B128 attribution authorized: **NO**.

B64: **REJECTED — CURRENT M28 FRONTIER**.

B4: **BLOCKED**.

ONE next PRIMARY:

> Develop a less-invasive coarse recurrent-versus-attention boundary
> measurement that does not add dozens of per-group stream event records to
> the timed host path. Until then, do not attribute the GPU variance to either
> family.

Explicitly not authorized: optimization, kernel tuning, B128, residual
architecture, B64, B4, source bisect, stream changes, or synchronization
changes.

No optimization candidate was implemented. EXP-0391 only determined that the
attempted recurrent/attention event split perturbs the P512 runtime target.

## Provenance

Experiment commit and graph SHA are added after graph refresh. Final working
tree status is recorded in the provenance commit.
