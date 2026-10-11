# M31-PERF-0001 build checkpoint

Both revisions built successfully from separate clean, detached source checkouts on the Z840 in `localhost/miinfer-dev:rocm-7.2.1` (digest `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`). The host model SHA-256 was `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`; GPU identity was gfx906 at BDF `0000:06:00.0`, unique ID `0x21678e17348c2f7`.

| Role | Source commit | Binary SHA-256 |
| --- | --- | --- |
| Control | `016472160c295eb1927b1ed6e095719ada6c3d18` | `52dd0f09d0f1921c3556d0f08d300644dd3e7dc910bf7c3ef30ac3e34a0cc1c1` |
| Candidate | `8a9caec3366512e459a93bad68adde181225304b` | `2661d347ab12f5ab91d391ef119e69b61adb559210268938a549d2f684cfb5f9` |

Both builds used identical CMake settings: Release, HIP enabled for gfx906, tests and benchmarks enabled, research tools disabled, ROCm 7.2.1 prefix paths, and the pinned image's nlohmann_json 3.11.3 source cache mounted read-only. The build target was `e2-0003-checkpointed-call`. The build logs reported two existing HIP compiler warnings and both targets completed successfully.

Per-revision source hashes and build provenance are in [`source-manifest-control.json`](source-manifest-control.json) and [`source-manifest-candidate.json`](source-manifest-candidate.json). Timing runs and the performance disposition are pending.
