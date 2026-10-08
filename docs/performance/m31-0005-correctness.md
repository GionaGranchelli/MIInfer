# M31-0005 — Correctness and regression protection

Status: host regression suite passed; GPU attention regression pending.

## Invariants retained

- Exact causal attention includes the current token's stored K/V and no later
  KV position.
- Graph decode derives inclusive cache length as
  `DeviceDecodeState::position + 1`.
- M29 device KV ownership/views and M30 snapshot/fork/rollback ownership are
  unchanged.
- M31-0004 prefix residency invalidation and incremental fingerprinting are
  untouched by the production attention patch.
- No approximate/sliding-window behavior or precision change was introduced.

## Change and adversarial case

The eager one-token suffix caller previously added one before passing the base
position to a kernel that computes `base_position + token + 1`. At position P,
this admitted P+1 (the first future slot) into attention. It now passes P. The
new GPU test gives positions 0 and 1 values 1 and 3, poisons position 2 with
1000, uses zero query/gate, and expects gated output 1.0. It exercises the
suffix kernel boundary directly; it does not instantiate the full transformer
or prove GPU behavior until run on hardware.

## Validation

- 13 host-only CTest tests passed (7.64 seconds in the previous focused run).
- C++20 syntax checks passed for the changed attention source and new HIP test.
- The new GPU test was not run.
- M31-0004's full GPU exact-prefix and snapshot/fork/rollback paths were not
  rerun; no corresponding source was changed.
- HIP configure/build did not reach compilation because the installed ROCm
  6.2.41133 linker cannot load `libxml2.so.2`. Host-only full application build
  also remains blocked by the CLI's existing `hip/hip_fp16.h` dependency.
- A complete release gate was not run.

## Minimal hardware test when safe

1. Build and run `miinfer-m31-0005-decode-attention-bound-test` on one MI50;
   confirm exact PASS and clean GPU memory afterward.
2. Run eager non-graph one-token decode with a short prompt and output parity
   against graph decode, including near-capacity and minimum prompt boundaries.
3. Run the existing exact-prefix boundaries (511/512/513/1024/1025) and
   snapshot/fork/rollback parity suite only if shared runtime state is touched.
4. For any graph kernel/split candidate: first short 8K parity, then 32K timing;
   only promote after a controlled repeat, and use 64K/128K only for a winner.

All real inference tests are deferred: the prior real benchmark guard returned
after its second threshold sample rather than demonstrating timely process
termination. Synthetic watchdog tests are not equivalent proof.
