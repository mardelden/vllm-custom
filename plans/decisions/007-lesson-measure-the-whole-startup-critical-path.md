# Lesson 007: Measure the complete startup critical path

**Date:** 2026-08-24
**Area:** performance methodology / model loading / serving startup

## What we were trying to do

Reduce the interval users experience between starting a model service and receiving a healthy API,
while preserving correctness, memory headroom, and serving performance.

## What we initially got wrong

The first GLM analysis attributed the wall to roughly 50,000 Python/GIL-bound destination copies.
That count described raw source tensors, not the 1,012--1,164 fused tensors in each prepared-cache
manifest. More importantly, varying only source memory type reproduced a roughly 150x H2D rate
cliff. One busy core and negative thread scaling were symptoms of the file-backed delivery path,
not proof of a Python-dispatch root cause.

Later, a worker loader appeared 48 seconds slower than PP0 even after transport was fixed. The
extra time was Hugging Face file-list/DNS retry setup for an already local model, not streaming.
After weight load fell to seconds, FlashInfer autotuning, memory profiling, frontend construction,
AOT/import work, and packaging became the new critical path.

## Measurement decision

Every startup report must identify these clocks separately:

1. Process/container/service start.
2. Model/config/tokenizer/processor resolution.
3. Weight iterator setup and first byte.
4. Storage/page-in, staging, and CUDA delivery.
5. Model-specific transform/repack.
6. Compile/JIT/autotune.
7. CUDA graph capture.
8. Memory profile and KV allocation.
9. Engine warmup and API readiness.

If one log timer spans several stages, instrument the boundary or label the interval as combined.
Do not manufacture a more precise attribution.

For each run, name the state of at least:

- local model snapshot and offline resolution;
- OS page cache or targeted file eviction;
- prepared-weight cache, if any;
- Torch/vLLM/AOT cache;
- Triton/FlashInfer JIT cache;
- FlashInfer autotune tactics;
- startup-plan and multimodal-budget cache;
- process-local CUDA graphs.

Report population boot, page-cold restart, and resident restart separately. Use targeted file-page
eviction for controlled cold-I/O tests; do not globally clear unrelated host caches as a routine
service step.

## Diagnostic rules that survived

- Count bytes and operations, then check whether the implied per-operation time is plausible.
- Benchmark one hot operation while varying source memory kind before naming the GIL or disk.
- Threads that become slower suggest contention; threads with no gain do not identify the lock.
- Split fixed setup cost from byte-proportional streaming. A rank-time ratio that matches its byte
  ratio usually means the hot path is behaving normally.
- `mmap` provides an address, not residency. Warm page cache does not guarantee a fast CUDA source
  path on every platform.
- Low `MemAvailable` on UMA does not prove page cache is the problem. Inspect `Cached`, allocator
  reserves, page scans/stalls, and overlapping lifetimes.
- A cache file on disk is not proof of a hit. Require identity and hit logs plus phase-time evidence.
- Source present on a host is not proof that the running image contains it.

## Required result gates

| Gate | Requirement |
| --- | --- |
| Correctness | Fixed expected output plus coherent generation; destination equality/digests for risky filters/transforms |
| Stability | No restart, OOM, Xid, allocator failure, or hidden cold fallback |
| Memory | Peak host RAM and device/UMA residency for loading, repack, graphs, and KV allocation |
| Startup | Median/range over repeated controlled runs, including API-ready wall |
| Serving | Representative TTFT, ITL, throughput, context/concurrency, and MTP/multimodal behavior |
| Reproducibility | Exact source/dependency/image/model/config/cache identities |

## Outcome

The method changed the technical direction. GLM moved from a parked "C++/CUDA scatter is the only
remaining answer" conclusion to an 11.95 / 14.50 / 14.44-second direct load with no weight cache.
RTX/Qwen moved from an 8.535-second stock weight phase and 65.681-second API wall to about
3.2-second page-cold weight loading, then shifted attention to multimodal/profile/AOT/frontend work.

The best optimization is therefore not a particular loader. It is the discipline of finding the
current longest phase, changing one mechanism and one cache state at a time, and remeasuring the
whole critical path after every win.

## References

- [Cross-team GLM comparison](https://vitrina.vulcandom.com/w/dgx-spark-weight-loading/)
- [GB10 loader and memory mechanism](https://vitrina.vulcandom.com/w/dgx-spark-gb10-memory-loading/)
- [Generic startup release 3 handover](../003-generic-startup-release-3-deployment-handover.md)
