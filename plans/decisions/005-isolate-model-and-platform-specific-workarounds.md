# Decision 005: Isolate model and platform-specific workarounds

**Status:** Accepted
**Date:** 2026-08-24

## Context

The experiments crossed two materially different memory systems and several model-specific
initialization paths:

- GB10 exposes CPU pages, page cache, pinned buffers, CUDA allocations, model weights, and KV
  through one 128 GiB coherent LPDDR pool.
- RTX PRO 6000 has separate host RAM and 96 GB device VRAM with real H2D DMA.
- GLM-5.2-504B on the tested sm_121 vLLM base lacked a usable DeepGEMM sparse-MLA path.
- Qwen3.8 target/MTP ownership and multimodal warmup have model-specific semantics.

A workaround that is essential on one combination can be pure overhead or incorrect on another.

## Decision

Keep generic transport and startup primitives generic, but place model/hardware policy behind
narrow capability checks or explicit opt-ins. Each workaround must state its removal condition and
must be re-evaluated against the target upstream release before porting.

### GB10/UMA rules

- Budget `final weights + resident file pages + staging + pinned buffers + allocator reserves +
  runtime margin + KV` against one physical pool.
- Use a fixed number of bounded staging buffers and destroy them before KV sizing.
- `torch.cuda.empty_cache()` does not drain PyTorch's pinned-host caching allocator. The tested
  Torch 2.11 runtime required the private `torch._C._host_emptyCache()` entry point; this is a
  version-specific workaround, not a stable public API.
- Prefer loader-owned, bounded page reclamation or O_DIRECT. A global 10 Hz `drop_caches` loop can
  evict pages between prefetch and copy and wastes CPU when model allocations permanently keep
  `MemAvailable` low.
- A no-GDS/bounce-buffer path may be safer for a near-full UMA model when a unified copier pins a
  whole source shard while allocating a whole CUDA destination.

### Discrete RTX rules

- Maintain separate host-RAM and VRAM peak equations. Pinned buffers consume host RAM and improve
  H2D, but do not consume the same physical pool as final device weights.
- Let FastSafetensors/platform detection report non-unified memory. Do not carry the GB10
  `FASTSAFETENSORS_UNIFIED_MEM=0` override as unexplained cargo cult configuration.
- Validate the executable backend closure. On the tested XQA image, an importable FlashInfer Python
  package was insufficient; `/usr/local/cuda/bin/nvcc` had to be mounted, executable, and on PATH,
  with a persistent JIT workspace.

### GLM sm_121 enablement

The historical [`glm-504b-gb10-sm121`](https://github.com/mardelden/vllm-custom/tree/glm-504b-gb10-sm121)
branch grafted Apache-2.0 CosmicRaisins Triton sparse-MLA/DeepGEMM fallbacks and added two compatibility
fixes for the pinned vLLM base:

1. Preserve the 656-byte `fp8_ds_mla` KV layout when a reshape caller forwards `auto`.
2. Return the shape-only paged-MQA metadata buffer expected by the stock indexer instead of `None`.

The branch also included UMA allocator-reserve handling and ModelOpt memory tracing. The validated
serve required the corrected 168-expert model configuration, sparse FlashMLA, `fp8_ds_mla`,
ModelOpt FP4, and the Triton sparse-MLA gate. It returned `391` coherently.

This is an enablement branch, not a generic loader patch. Current upstream includes later GLM,
SM120/121, sparse-MLA, and fused-MoE work; compare behavior and tests before carrying any part of
the historical graft forward.

### Qwen-specific rules

- Keep target/MTP byte-range filtering and shared embedding/head ownership conditional on the exact
  model contract.
- Keep lightweight multimodal warmup inputs and processor overlap in model-aware adapters; do not
  assume every multimodal processor can be constructed or warmed concurrently.

## Consequences

- Platform-specific private APIs and environment variables remain off by default and have explicit
  fallbacks.
- A performance result on GB10 does not select RTX memory policy, and an RTX GDS/direct-I/O result
  does not prove safety on UMA.
- Imported kernel code retains license/provenance and is removed when an upstream executable path
  satisfies the same correctness and performance gates.
- Validation includes fixed output, coherent free text, model-specific weight equality, peak memory,
  zero restarts/OOM/Xid, and representative serving performance.

## References

- [GLM sm_121 enablement commit](https://github.com/mardelden/vllm-custom/commit/8e9c7ae071ea9c00eb070004b268b1242aad3c1c)
- [Qwen multimodal overlap](https://github.com/mardelden/vllm-custom/commit/809eb0bcb7ec2a841adf6419a3372ff502f5dfdf)
- [Qwen target/MTP ownership](https://github.com/mardelden/vllm-custom/commit/53e1502cb315e41567ef9c3efb452afd71e666d2)
