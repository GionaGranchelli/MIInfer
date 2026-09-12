# EXP-0316 — M25-J quantizer under qualified MMQ

**Status:** REJECT as a performance claim; RETEST only with clock-qualified A/B  
**Milestone:** M25  
**Date:** 2026-09-12  
**Baseline commit:** `e0bae93`  
**Candidate:** working tree runtime selector

## Question

Does the pinned four-block Mx Q8 activation quantizer help when paired with
MIInfer's qualified staged MMQ kernel?

## Hypothesis

The earlier J screen also enabled the already rejected
`MIINFER_MX_PIPELINE=1` path. Removing that confounder may reveal a quantizer
win from the retained J source port.

## Candidate and provenance

The candidate enabled `MIINFER_MX_Q8_BATCH=1`; the control used `0`. The
candidate is the four-block workgroup port in
`gfx906/kernels/kquant_wave_layout.hip`, based on the pinned mx-llama.cpp
`quantize_mmq_q8_1` contract at commit
`2e9d29fe736969160f17476ec6f0a6298cee6966` (MIT, ggml authors; attribution is
recorded in the source and `docs/references.md`).

## Benchmark

Exact P512 H/I vector from EXP-0314, with `MIINFER_MX_PIPELINE` unset and
`MIINFER_MX_Q8_BATCH` changed only for the candidate/control. Device allocation
was `18,472,649,044` bytes in every run.

## Results

Candidate samples were `216.05`, `203.20`, `213.63`, and `207.52` tok/s; the
three-run candidate median was `207.52` tok/s. Fresh control samples were
`210.20`, `187.66`, `215.05`, and `212.93` tok/s; the three-run control median
was `212.93` tok/s. The runs were not interleaved with concurrent clock
telemetry, so these are screening data rather than a qualification claim.

## Correctness

All P512 processes completed with finite output and unchanged allocation.
Model-generation correctness remains covered by the established H/I checks.

## Decision

**REJECT as a measured win.** Keep J opt-in and disabled in the qualified
preset. Revisit only with interleaved clock-qualified A/B data or a materially
different quantizer implementation.
