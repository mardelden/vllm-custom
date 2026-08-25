# Decision 003: Prepared weights are an optional semantic cache

**Status:** Accepted
**Date:** 2026-08-24

## Context

The GLM prepared cache captured each PP rank after checkpoint loading and before ModelOpt NVFP4
post-load processing. That was a useful semantic boundary: it skipped repeated model-specific
merging and conversion. It did not, by itself, fix physical transport. The original cache restore
still combined cold file-backed pages with scattered destination copies and took 567.90 / 808.49 /
817.98 seconds.

After the original checkpoint loader was repaired, direct loading took 11.95 / 14.50 /
14.44 seconds. The lifecycle-hardened prepared cache took 23.19 / 29.74 / 30.36 seconds and
occupied about 318 GB across the three ranks. For this model and software revision, the second
representation no longer paid for itself.

## Decision

Do not make a prepared-weight cache the default loader architecture. Add one only when profiling
shows that unavoidable semantic transforms or repacking dominate after direct transport and range
ownership are fixed.

When a prepared cache is justified, cache semantics and cache transport are separate designs. The
minimum contract is:

1. Fingerprint model revision, vLLM/loader/cache schema, quantization, parallel layout, rank, dtype,
   shape, and transformation boundary.
2. Validate every source range and destination before the first write.
3. Restore in place through a bounded transport: one prepared shard staged and scattered, or a
   similarly bounded large-blob reader.
4. Synchronize and release staging before opening the next resident window.
5. Write a unique staging generation, then publish the complete manifest atomically.
6. Reach save-success consensus across every distributed rank.
7. Exit the cache-building process before normal profiling/KV allocation, then strict-load in a
   fresh process.
8. Fail hard on an expected load miss or mismatch; do not silently enter a many-minute cold path.
9. Make rank placement deterministic or deliberately distribute rank caches. Never assume Ray will
   return the same worker host.

## Evidence

| Prepared-cache implementation | PP0 / PP1 / PP2 | Lesson |
| --- | ---: | --- |
| Original mmap/Python restore | 567.90 / 808.49 / 817.98 s | Semantic cache without transport design |
| Native page prefetch then four copy workers | 41.42 / 58.50 / 54.71 s | Separate page-in from copy |
| One CUDA staging shard, 64 MiB requests | 22.77 / 29.35 / 29.57 s | Fastest prepared transport |
| Lifecycle-hardened staged restore | 23.19 / 29.74 / 30.36 s | Small cost for strict validation/reclaim/lifecycle |
| Muse parallel blobs | 30.3 / about 34 / about 34 s | Fewer large files also work, with a new format |
| Qwen legacy-to-blob deployment | 31.16 / 43.40 / 44.54 s | Artifact verification and reversible conversion matter |
| CUDA destination arena | No successful serving run | Arena lifetime overlapped repack and OOMed |

## Alternatives considered

| Alternative | Standing |
| --- | --- |
| Keep prepared cache because it is already built | Rejected; sunk storage and build cost do not make a slower path desirable |
| Persistent mmap of the whole cache | Rejected on near-full UMA; virtual mappings, file pages, staging, and destination allocations overlap |
| Contiguous destination arena | Rejected for GLM's changing post-load representation; source and replacement lifetimes overlap |
| Large contiguous source blobs | Viable when transforms dominate, but still require versioning, atomic publication, placement, and bounded restore |

## Consequences

- Direct loading is the reference correctness path and default fallback.
- A cache hit is not considered successful merely because the key exists; logs must prove strict
  selection, validation, and the expected transport.
- Cache-building and normal serving are separate process lifecycles on tight-memory systems.
- Cache storage should be reclaimable derived state, never the only copy of model weights.
- Rank-local cache placement is part of correctness and reproducibility, not only performance.

## References

- [Canonical GLM cache/loader implementation](https://github.com/mardelden/vllm-custom/commit/f3871e28717dde9bb0b8cf1039aaab2968611d3e)
- [Canonical GLM experiment archive](https://github.com/mardelden/dgx-spark/commit/c24cda69d89d895e0c5d4a8235a5f4b6b7f60fdd)
- [Prepared-cache and direct-loader comparison](https://vitrina.vulcandom.com/w/dgx-spark-weight-loading/)
