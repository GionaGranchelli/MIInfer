# EXP-0068 — M6-A24 Qwen3.8 eight-layer stateful GPU prefix

## Question

Can the common GPU executor compose the first eight Qwen3.8 layers statefully,
including both recurrent and full-attention layers, through P64 without state
or cache contamination?

## Candidate

The `--prefix8` mode uses one ordered `GpuLayerRef` array and `run_prefix()`:

```text
L0 Delta → L1 Delta → L2 Delta → L3 full attention
→ L4 Delta → L5 Delta → L6 Delta → L7 full attention
```

Each recurrent layer owns persistent state and each full-attention layer owns an
independent persistent K/V cache. Existing kernels and numerical boundaries
were unchanged. The mode uses one preallocated output buffer per layer and one
input buffer.

## Correctness contract

Reference: the pinned M6-A1 llama.cpp Qwen3.8 fixture in
`/tmp/m6a1-qwen38-reference`, using
`/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`.

The model artifact is SHA256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

Checked positions were P0, P1, P2, P4, P8, P16, P32, and P64. Every L0–L7
output was compared using max absolute error, RMS error, reference-relative
RMS, and finite-value checks. Recurrent state entry and exit fingerprints were
checked for L0, L1, L2, L4, L5, and L6. K/V cache entry and exit fingerprints
were checked for L3 and L7 over logical cache bytes only.

## Results

All 64 stateful positions executed through the same eight-layer executor. The
checkpoint rows below are `max_abs / RMS / relative RMS` for each layer.

| Position | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000973 / .000222 / .000832 | .005023 / .000910 / .002801 | .023933 / .001882 / .004103 | .032173 / .002543 / .004474 | .071922 / .003250 / .005296 | .064835 / .003782 / .005953 | .022772 / .004276 / .005596 | .041119 / .004980 / .006369 |
| 1 | .001511 / .000088 / .000461 | .006330 / .000798 / .003464 | .007359 / .002118 / .008612 | .073950 / .003056 / .008000 | .119102 / .004013 / .009144 | .094368 / .004724 / .009981 | .050324 / .007798 / .014609 | .032604 / .008750 / .012759 |
| 2 | .000526 / .000134 / .000573 | .025656 / .001004 / .003295 | .045748 / .001586 / .005008 | .019217 / .001973 / .004433 | .016827 / .002493 / .005100 | .068169 / .003192 / .006248 | .060242 / .004355 / .007896 | .094055 / .004798 / .007926 |
| 4 | .002347 / .000052 / .000319 | .002293 / .000485 / .002402 | .010408 / .000944 / .004696 | .029596 / .001568 / .004980 | .006849 / .001815 / .004964 | .074122 / .002381 / .006121 | .011463 / .002801 / .006475 | .012151 / .003081 / .006383 |
| 8 | .000206 / .000034 / .000214 | .001201 / .000327 / .001719 | .050670 / .001075 / .005921 | .062954 / .001627 / .005647 | .049069 / .001839 / .005632 | .012292 / .002348 / .006400 | .013384 / .003510 / .008747 | .017520 / .004082 / .008358 |
| 16 | .000729 / .000078 / .000353 | .001868 / .000444 / .001856 | .019795 / .000948 / .004001 | .032042 / .001781 / .004584 | .017157 / .002609 / .006006 | .092300 / .003292 / .006464 | **.162170 / .004451 / .007939** | .017953 / .004558 / .006914 |
| 32 | .000823 / .000067 / .000318 | .007193 / .000397 / .001544 | .014580 / .000813 / .003122 | .014525 / .001079 / .002929 | .015417 / .001284 / .003236 | .028046 / .001636 / .003871 | .009441 / .002154 / .004876 | .038525 / .002793 / .005538 |
| 64 | .000853 / .000098 / .000523 | .004152 / .000338 / .001475 | .012252 / .000581 / .002658 | .011614 / .001096 / .003480 | .024511 / .001372 / .003967 | .038591 / .001665 / .004498 | .047319 / .002112 / .005542 | .021269 / .002476 / .005569 |

Maximum observed error was `0.16217` at L6/P16. Relative RMS remained below
`0.01461` for every reported layer checkpoint. All checkpoints passed the
existing external-reference envelope and all outputs were finite.

The post-run logical fingerprints were:

| P | hidden3 | hidden7 | state0 | state1 | state2 | state4 | state5 | state6 | K3 | V3 | K7 | V7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16789182811572405522 | 17721018176990300011 | 2451724452713627422 | 16100946777412942155 | 689492793206381606 | 282661175610317932 | 15224907680276593011 | 17079586940236964079 | 16030679738720489512 | 39425660738274567 | 1624542297103504268 | 17545717735304763783 |
| 1 | 17757679717823461043 | 2988657362706964906 | 8918456245252580062 | 304148865373566752 | 8072574388220486157 | 11248322196804456284 | 2961282135884722464 | 10878231710579082051 | 9590143615054939663 | 11794599276045736033 | 9325067286437769406 | 17662890690129796340 |
| 2 | 6537715723945590266 | 7445782956106923749 | 3985104946628075402 | 10455551523943718013 | 16824913193211348513 | 4333891549869241327 | 8727776586737094085 | 2628294253686017554 | 10749924341784067565 | 10187679005748668280 | 11441590600734364811 | 9863506681121141931 |
| 4 | 6531938485154443033 | 12444429570514645453 | 13867228957992064674 | 2951962655791205847 | 7076563114778377901 | 2434395089595806588 | 48078911901503166 | 16590438163949448572 | 14163139511240763244 | 16273778360926125935 | 7075958183140649254 | 18167361344080599478 |
| 8 | 6123567600958581867 | 3101768566055500188 | 12752520487470773291 | 6604948214140104265 | 6181467609250185764 | 3867124570546945270 | 2550270199873825521 | 3535244216237070261 | 6611998883016593203 | 10615119341344739775 | 13788468903945858 | 1106811327272249682 |
| 16 | 9981689178820893827 | 17854155028479021201 | 3835682966140516700 | 7805026231362583818 | 13665270067888118022 | 12652056960988837823 | 12274500480663072744 | 18059013720931969214 | 3454383152374153314 | 14950440908617366063 | 9026186233338756506 | 12079160118561295899 |
| 32 | 12907688065427662007 | 17200982187157509890 | 4285384362259400531 | 7375965656806894406 | 14072243666591604932 | 17303348931220987756 | 9286803431866441177 | 4443763481916440254 | 6640144510259735580 | 14775097671863517002 | 15522853115614516353 | 18205729754741314796 |
| 64 | 3786783812604125928 | 5790982511971742776 | 3494274868604155650 | 14600517215752026630 | 15311287162012719556 | 11261676379213826638 | 13683268155112180095 | 1712703147445879322 | 12721657159595180664 | 4831602320261041139 | 1825061489789440819 | 14599811060670407160 |

## Stateful replay and reset

The run recorded logical entry and exit fingerprints for all six recurrent
states, both attention K/V caches, and all eight layer outputs. Then every
recurrent state and cache was poisoned (`0xA5`; recurrent history additionally
`0xFF`), reset through the real initialization path, and replayed through P64.

Result: `poisoned_reset_replay=PASS`; every recorded entry/exit fingerprint was
identical. A separate fresh-process replay of the prefix command was also
identical.

The cached-attention barrier regression is now a permanent GPU test:
`qwen3-cached-attention-determinism-gpu`, which performs 16 exact replays of a
deterministic cache/query fixture.

## Performance and memory accounting

This is an engineering prefix measurement, not a full-model throughput claim.
The first pass reported:

| Metric | Result |
|---|---:|
| Prefix CPU elapsed around GPU launches/syncs | 2324.45 ms / 65 positions |
| Engineering average | 35.761 ms/position |
| Allocations during decode | 0 |
| Device bytes after eight-layer setup | 2,041,279,360 |
| Dispatch count | not instrumented in this tool |
| Copy bytes | not instrumented in this tool |

The timing includes synchronization and is not a publication-quality GPU
benchmark. The ledger nevertheless confirms the important invariant: no
allocation occurs in the stateful prefix loop. Persistent recurrent state alone
is 1,572,864 bytes, and the two 128-position full-attention K/V caches are
1,048,576 bytes; the reported device total also includes weights and all layer
workspaces.

## Checks

```text
Release CTest: 20/20
Eight-layer external checkpoints: PASS
Recurrent state-entry checks: 48/48
Poisoned reset/replay: PASS
Cached-attention determinism: 16/16 exact replays
```

## Decision

**KEEP / CLOSED.** M6-A24 proves the first eight real GPU layers compose through
one common stateful executor with bounded external error, deterministic state
replay, zero steady-state allocations, and both recurrent/full-attention state
types active in one prefix.

## Follow-up

Proceed to the 16-layer composition ladder, using boundary checkpoints and
binary localization rather than adding another block-specific executor.
