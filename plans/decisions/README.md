# vLLM optimization decision catalog

**Status:** Maintained handoff index
**Last reviewed:** 2026-08-24

## Purpose

This directory records the vLLM changes and operating assumptions that survived the
DGX Spark/GB10 GLM-504B investigations and the RTX PRO 6000 Qwen3.8 investigation. It is
the starting point for the team maintaining this fork. The experiment repositories keep
the raw harnesses; these records keep the decisions, limits, source identities, and
measurements that should influence future vLLM work.

The central conclusion is:

> Treat startup as a bounded data pipeline followed by independently cacheable engine
> phases. Read only the bytes a rank owns, separate file residency from CUDA delivery,
> bound transient residency by bytes, and persist expensive derived work under exact
> fingerprints.

Prepared weight caches remain a supported research path, not the default. On GLM-504B,
the repaired direct loader was both faster and 318 GB smaller than the best prepared-cache
restore.

## Decisions

| Record | Decision |
| --- | --- |
| [001](001-bounded-weight-loading-pipeline.md) | Design loading as a bounded residency and transfer pipeline |
| [002](002-filter-unowned-checkpoint-ranges-before-materialization.md) | Filter rank/model-owned checkpoint ranges before materialization |
| [003](003-prepared-weight-cache-is-optional.md) | Use prepared weights only when semantic transforms justify a second representation |
| [004](004-persist-startup-work-by-phase.md) | Persist startup work by phase under exact, runtime-specific fingerprints |
| [005](005-isolate-model-and-platform-specific-workarounds.md) | Keep model and platform workarounds narrow, gated, and upstream-aware |
| [006](006-package-patches-as-immutable-release-contracts.md) | Package source patches as immutable release contracts |
| [007](007-lesson-measure-the-whole-startup-critical-path.md) | Measure the complete startup critical path and name every cache state |

## Maintained-changes ledger

**Re-audited 2026-08-31 against `upstream/main@5707355209`.** `main` was fast-forwarded to
upstream on this date (it was 2284 behind and 0 ahead — it carried nothing of ours, so the
sync was lossless). Every carried patch set below was re-verified with `git apply --check`
against that upstream and **applies clean**; we are not fighting drift.

| Patch set | Branch | Code carried | Upstream status |
| --- | --- | ---: | --- |
| Generic startup (loader geometry, adaptive O_DIRECT, startup-plan reuse, MM/AOT overlap) | `codex/fastsafetensors-parallel-mtp-share` | 702 lines / 21 files | Candidate. Reader controls + the 16 GiB `max_copy_block_size` default are a genuine upstream defect; adaptive O_DIRECT needs the `fastsafetensors` fork upstreamed first (two-repo chain) |
| GLM-5.3 NoPE sparse MLA on sm120 | `codex/glm53-sm120-nope-sparse-mla` | 34 lines / 2 files | **Check first** — upstream PR #53906 "add GLM-5.3-Flash support" is open (93 files, +14k, needs FlashInfer 0.6.18); may already cover this path |
| Reasoning output logging | `codex/dsv4-log-outputs-reasoning` | 26 lines / 2 files | **Send.** Upstream bug present in every build we run; previously reported as #24578 / #25918 / #19462 without diagnosis |

Serving trees are not this fork: DSv4 runs `jasl/vllm` PR#41834 (`20260809`) and GLM-5.3 runs a
local build (`487ecf187`, not on any remote). We author here and ship patches to those trees, so
upstreaming reduces our maintenance burden but does **not** propagate to the fleet on its own.

## Historical branch ledger

This table was audited against every local `refs/heads` entry on 2026-08-24 and traced back through
`dgx-spark`, `dgx-spark-v2`, `dgx-spark-v3`, `dgx-spark-v4`, and `dgx-spark-v5`.

| Branch | Tip | Experiment provenance | Standing |
| --- | --- | --- | --- |
| `main` | `5707355209` | Fast-forwarded to `upstream/main` on 2026-08-31; previously the frozen `aa8bb5562e` baseline | Now a clean upstream mirror — safe to branch from |
| `glm-504b-gb10-sm121` | [`8e9c7ae071`](https://github.com/mardelden/vllm-custom/commit/8e9c7ae071ea9c00eb070004b268b1242aad3c1c) | Original DGX Spark GLM enablement; inherited by v2-v5 | Historical validated sm_121/sparse-MLA enablement branch |
| `glm-504b-gb10-fast-loading` | [`f3871e287`](https://github.com/mardelden/vllm-custom/commit/f3871e28717dde9bb0b8cf1039aaab2968611d3e) | Canonical GLM loader/cache consolidation preserved by `dgx-spark-v5` | Retained historical implementation; all features off by default |
| `fastsafetensors-loader-controls` | [`a62c29bba`](https://github.com/mardelden/vllm-custom/commit/a62c29bba699d2f5ba8f81e2fa4bc31712f16e23) | Isolated RTX/Qwen loader-control experiment recorded by the v5 handover | Superseded as a release by the composite branch; commit retained there |
| `pp-weight-filter-before-read` | [`9cbeff386`](https://github.com/mardelden/vllm-custom/commit/9cbeff386f09259905ee740ad54dfe405fc83d78) | Isolated generic PP ownership experiment reviewed during the v5 RTX work | Candidate building block; not independently released |
| `bounded-pinned-weight-loader` | [`31e1f58938`](https://github.com/mardelden/vllm-custom/commit/31e1f58938363b121c7862626495c754b8dc2489) | Isolated generic pinned-reader experiment reviewed during the v5 RTX work | Valid fallback/candidate; not part of the maintained release |
| `codex/qwen3-vl-skip-zero-dummy-modalities` | `768c007dc6` | Preliminary v5 RTX/Qwen multimodal experiment: skip normal video dummy setup when video count is zero | Parked local-only branch; unmerged, no preserved performance measurement, and not a release input |
| `codex/renderer-minimal-mm-warmup-inputs` | [`809eb0bcb7`](https://github.com/mardelden/vllm-custom/commit/809eb0bcb7ec2a841adf6419a3372ff502f5dfdf) | Successor v5 RTX/Qwen experiment using lightweight warmup-specific media and overlap | Retained; first source commit in the composite branch |
| `codex/startup-plan-stable-fingerprint` | `2c49d626cd` | Isolated v5 RTX startup-plan correctness experiment | Superseded by `74cd835791`, which retains the boot-stable fingerprint and adds schema-2 exact KV reuse/MM-profile rules |
| `codex/fastsafetensors-parallel-mtp-share` | [`af7c0d47eb`](https://github.com/mardelden/vllm-custom/commit/af7c0d47eb02ed2779b9a19638592caa11a6c9d8) | Composite RTX/Qwen development and release-contract branch from the v5 handover | Current development pointer; immutable release commits/manifests, not this moving branch, are deployment identities |

The Codex `v2`-`v4` Spark work did not create additional vLLM git branches. It patched derived
images from archived build contexts:

- `dgx-spark-v2`: native mmap page-prefetch/CUDA-copy experiments under
  `experiments/cache-prefetch-restore/`;
- `dgx-spark-v3`: contiguous CUDA arena experiments under
  `experiments/cache-arena-v7-restore/`;
- `dgx-spark-v4`: Ray rank-swap/cache-locality experiments under
  `experiments/cache-ray-rank-swap-restore/`.

Their measurements and lessons remain in the GLM evidence ledger below. They are not missing fork
branches. Future local branches should be added here when created and marked retained, superseded,
parked, or rejected before their refs are removed.

## Durable source map

| Area | Source identity | What it contains | Standing |
| --- | --- | --- | --- |
| GLM-504B on GB10/sm_121 | [`glm-504b-gb10-sm121@8e9c7ae071`](https://github.com/mardelden/vllm-custom/commit/8e9c7ae071ea9c00eb070004b268b1242aad3c1c) | Triton sparse-MLA/DeepGEMM fallbacks and vLLM compatibility fixes | Historical validated enablement branch; re-check current upstream before porting |
| Consolidated GLM loading | [`glm-504b-gb10-fast-loading@f3871e287`](https://github.com/mardelden/vllm-custom/commit/f3871e28717dde9bb0b8cf1039aaab2968611d3e) | Bounded prepared-cache restore, bounded raw FastSafetensors, lifecycle, placement, and distributed warmup persistence | Canonical record of the Codex GLM implementation; off by default |
| GLM experiment/fleet evidence | [`dgx-spark@c24cda69`](https://github.com/mardelden/dgx-spark/commit/c24cda69d89d895e0c5d4a8235a5f4b6b7f60fdd) | Harnesses, measurements, checkpoints, and lifecycle decisions | Durable experiment archive |
| Isolated PP filter | [`pp-weight-filter-before-read@9cbeff386`](https://github.com/mardelden/vllm-custom/commit/9cbeff386f09259905ee740ad54dfe405fc83d78) | Generic opt-in PP ownership filter | Candidate building block; not an independent release |
| Isolated pinned loader | [`bounded-pinned-weight-loader@31e1f589`](https://github.com/mardelden/vllm-custom/commit/31e1f58938363b121c7862626495c754b8dc2489) | Coalesced ranges, parallel reads, rotating pinned buffers | Candidate/fallback; platform policy still required |
| Isolated FastSafetensors controls | [`fastsafetensors-loader-controls@a62c29bba`](https://github.com/mardelden/vllm-custom/commit/a62c29bba699d2f5ba8f81e2fa4bc31712f16e23) | Reader, queue, bounce-buffer, and copy-block controls | Composed into the generic startup line |
| Maintained RTX/Qwen loader | [`5684372cc9`](https://github.com/mardelden/vllm-custom/commit/5684372cc95f592142320ed50cbee88098365e84), [`53e1502cb3`](https://github.com/mardelden/vllm-custom/commit/53e1502cb315e41567ef9c3efb452afd71e666d2), [`3673260f2b`](https://github.com/mardelden/vllm-custom/commit/3673260f2b5f37c3d3effa9b28536ba7e6d89404) | FastSafetensors geometry, Qwen/MTP range ownership, adaptive direct I/O | Retained in the generic startup patch set |
| Maintained startup reuse | [`809eb0bcb7`](https://github.com/mardelden/vllm-custom/commit/809eb0bcb7ec2a841adf6419a3372ff502f5dfdf), [`74cd835791`](https://github.com/mardelden/vllm-custom/commit/74cd835791f19eb0abbcd843a4e307e5e993c519), [`4f1cd0856c`](https://github.com/mardelden/vllm-custom/commit/4f1cd0856c291c099c30f4afd5a8c7cbed3b8e70) | Multimodal overlap, exact startup plans, AOT/frontend overlap, early forkserver | Retained source; deployment status is release-specific |
| Release contracts | [release 1](../001-generic-startup-overlay-deployment-handover.md), [release 2](../002-generic-startup-release-2-deployment-handover.md), [release 3](../003-generic-startup-release-3-deployment-handover.md) | Immutable source/dependency/runtime contracts and promotion gates | r2 is the healthy rollback; r3 is failed/quarantined evidence |

## GLM weight-phase evidence ledger

The clocks below are vLLM per-rank weight-load intervals. PP ranks load concurrently, so
cluster weight wall time is approximately the slowest rank. Baselines and end-to-end startup
conditions varied; do not turn this table into an API-ready ranking.

| Experiment line | PP0 / PP1 / PP2 | Representation | Durable conclusion |
| --- | ---: | --- | --- |
| Stock direct load | 480 / 697 / 708 s | Original safetensors | Pageable/file-backed delivery plus reading all PP bytes was the main wall |
| Codex v2 | 41.42 / 58.50 / 54.71 s | Prepared cache, native page-in then native copies | Prefetch/residency must precede copy; native copy threads alone were insufficient |
| Codex v3 | No successful serving result | CUDA arena cache | A source arena that overlaps NVFP4 repack can OOM even when restore itself is contiguous |
| Codex v4 | 567.90 / 808.49 / 817.98 s | Prepared cache replicated for rank swaps | Cache locality was fixed; transport remained slow |
| Codex v5 | 22.77 / 29.35 / 29.57 s | One prepared shard staged on CUDA | Fastest prepared-cache transport; hardened lifecycle was 23.19 / 29.74 / 30.36 s |
| Codex placement/builder line | 24.98 / 27.48 / 26.24 s | Prepared cache with deterministic rank placement | Automated build/restart and rank-local cache identity worked |
| Grok v1 | 31 / 45 / 47 s | Blob-first prepared cache | Large sequential blobs can work, with a separate format/lifecycle burden |
| Grok v2 | 90.5 / 124 / 128 s | One device copy per existing shard | Synthetic device-copy wins did not predict live UMA throughput |
| Muse v1 | 30.3 / about 34 / about 34 s | Three or four large blobs | Simple parallel blob restore was competitive but remained a second model representation |
| Qwen v1 | 31.16 / 43.40 / 44.54 s | Muse v2 blobs plus legacy converter | Proved deployment/reproducibility; did not invent a different loader |
| Claude v1 | **11.95 / 14.50 / 14.44 s** | Direct O_DIRECT reads into bounded pinned buffers | Fastest measured path; no prepared cache |
| Claude v2 | 42.8 / 49.1 / 51.1 s | Direct bounded anonymous clones | Smallest successful expression of the file-backed-copy fix |

## Human-readable reports

These Vitrina pages are private and Cloudflare Access-gated:

- [Weights in Motion — GB10 Unified Memory & Loader Design](https://vitrina.vulcandom.com/w/dgx-spark-gb10-memory-loading/)
  explains the memory and transport mechanism.
- [Every GLM-504B Loading Attempt Compared](https://vitrina.vulcandom.com/w/dgx-spark-weight-loading/)
  is the detailed cross-team evidence ledger.
- [Operating a 3-Spark Cluster](https://vitrina.vulcandom.com/w/dgx-spark-operations/)
  contains the GLM field journal and operational context.
- [Serving Large Models on DGX Spark](https://vitrina.vulcandom.com/w/dgx-spark-serving/)
  provides the broader serving context.

## Interpretation rules

- Re-run upstream discovery before porting an old branch; upstream vLLM and FlashInfer moved
  materially after the GB10 work.
- Treat measurements as evidence for a mechanism, not a promise for another model or host.
- Preserve off-by-default gates until correctness, memory, startup, and serving-performance
  checks pass on the target release.
- A source checkout, image, and running service are three different deployment states. Verify
  the code inside the selected artifact.
- Do not edit immutable release manifests after a canary. A changed contract is a new release.
- Local-only commits are not durable handoff artifacts. Push/tag a retained branch or record that
  it is intentionally parked before pruning local refs.
