# Decision 001: Design weight loading as a bounded pipeline

**Status:** Accepted design principle; implementation depends on platform
**Date:** 2026-08-24

## Context

The GLM-504B load initially looked like a Python/GIL problem because one core was busy and
threading did not help. The prepared-cache manifests, however, contained only 1,012--1,164
fused destination tensors. Microbenchmarks reproduced the production rate with far fewer
operations. The decisive variable on GB10 was the source memory type: CUDA copies from a
safetensors file-backed mmap were about 0.139 GiB/s, while the same data in anonymous memory
or bounded pinned buffers copied far faster.

Our measurements are consistent with a serialized pageable-host staging path in which file
faults and CUDA delivery are coupled. The exact driver implementation is platform/version
specific; the observed performance cliff is the fact the design must address.

FastSafetensors had the right transport idea, but early configurations used request and
resident-buffer geometry that could OOM a nearly full 128 GiB GB10 node. More threads were
not sufficient: the resident bytes and lifetime overlap had to be bounded.

## Decision

Model loading changes in this fork should express these stages explicitly:

1. Resolve an immutable local checkpoint and select only owned byte ranges.
2. Read or fault ranges into bounded anonymous or pinned storage.
3. Deliver resident bytes to the final CUDA parameters through a path that does not fault
   file-backed pages inside each destination copy.
4. Apply model-specific transforms and repacking.
5. Synchronize, release staging, and drain the relevant allocator before KV sizing.
6. Reclaim file pages only after their final consumer, and keep teardown off the critical
   path when possible.

Concurrency is bounded by bytes in flight, not only by thread or tensor count. Parallelize
chunks within a bounded file/shard window; do not make several model-sized or shard-sized
representations resident merely to keep workers busy.

For cold local NVMe, allow aligned direct I/O. Prefer an adaptive policy that uses O_DIRECT
for cold selected ranges and buffered reads for resident ranges. Alignment failures and
unsupported filesystems must have an explicit, measured fallback.

## Evidence

| Platform/model | Baseline | Candidate | Result |
| --- | ---: | --- | ---: |
| 3x GB10, GLM-504B PP=3 | Stock safetensors | Bounded anonymous clone | 42.8 / 49.1 / 51.1 s |
| 3x GB10, GLM-504B PP=3 | 480 / 697 / 708 s | PP filter + aligned O_DIRECT into rotating pinned buffers | **11.95 / 14.50 / 14.44 s** |
| 3x GB10, GLM-504B PP=3 | 567.90 / 808.49 / 817.98 s cache restore | One prepared shard staged then D2D-scattered | 22.77 / 29.35 / 29.57 s |
| RTX PRO 6000, Qwen3.8-27B-NVFP4 | 8.535 s total model load | Corrected 16-reader FastSafetensors geometry | 4.024 s |
| RTX PRO 6000, Qwen3.8-27B-NVFP4 | 8.535 s | Adaptive direct I/O | 3.18--3.22 s cold; 2.071 s resident |

The RTX result used 512 MiB copy blocks and a 16 MiB bounce pool; the strongest GLM path
used different geometry. Values are host/model measurements, not defaults for every system.

## Alternatives considered

| Alternative | Why it is not the default |
| --- | --- |
| Add Python threads around existing mmap-to-CUDA `copy_` | Threads contended on the same unfavorable path and could be slower than one thread |
| Prefetch the entire checkpoint before allocating the model | Later parameter allocation can evict the prefetched pages; prefetch must be adjacent to consumption |
| Eager safetensors loading | Zero-patch fallback, but single-file eager reads and whole-shard residency were slower/tighter on GB10 |
| Unbounded whole-shard FastSafetensors | Correct transport mechanism with unsafe transient residency for near-full UMA models |
| Always use O_DIRECT | Resident checkpoints were faster through buffered reads on the RTX host |

## Consequences

- Loader APIs need controls for reader count, queue depth, copy block size, bounce-buffer
  size, direct-I/O policy, and resident-byte limits.
- Every implementation must report peak host RAM and device/UMA memory as well as time.
- On GB10, pinned host allocations consume the same physical pool as CUDA weights and KV.
  Releasing device cache alone is insufficient; see decision 005.
- On discrete GPUs, host RAM and VRAM have separate budgets, but both peaks still need bounds.
- The simple anonymous-clone path remains a useful fallback and diagnostic even when a native
  pinned/direct-I/O implementation is faster.

## References

- [FastSafetensors controls](https://github.com/mardelden/vllm-custom/commit/5684372cc95f592142320ed50cbee88098365e84)
- [Adaptive direct I/O](https://github.com/mardelden/vllm-custom/commit/3673260f2b5f37c3d3effa9b28536ba7e6d89404)
- [Bounded pinned-loader branch](https://github.com/mardelden/vllm-custom/commit/31e1f58938363b121c7862626495c754b8dc2489)
- [GB10 loader design report](https://vitrina.vulcandom.com/w/dgx-spark-gb10-memory-loading/)
