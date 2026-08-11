# DGX Spark prepared-weight cache and bounded cold loading

This fork branch preserves the vLLM-side changes validated while serving
GLM-5.2-504B across three DGX Spark nodes. All behavior is disabled by default.

## Prepared-cache path

The cache captures each PP rank after checkpoint loading and before ModelOpt
NVFP4 post-load processing. Warm restore validates the full generation, reads
each 2–4 GiB cache shard through 16 no-GDS C++ readers into one CUDA staging
buffer, scatters typed views into final parameters, and releases staging before
opening the next shard.

The lifecycle supports strict load, save, auto, build-only, transactional
publication, PP-wide readiness/save consensus, and fail-hard operation. The
outer build-then-restart supervisor remains a deployment artifact in the
`mardelden/dgx-spark` repository rather than part of the vLLM CLI.

The validated warm GLM run restored PP0/1/2 in 23.19/29.74/30.36 seconds and
reached the API in 1 minute 51 seconds.

## Bounded raw fastsafetensors path

The vLLM iterator forwards configurable reader-count and bounce-buffer values.
The complete cold profile also requires a compatible fastsafetensors build
with these semantics:

- `VLLM_FASTSAFETENSORS_QUEUE_SIZE=-1` serializes shard lifetimes;
- `VLLM_FASTSAFETENSORS_MAX_COPY_BLOCK_MB=64` splits the shard into bounded
  read requests;
- the loader synchronizes and releases CUDA allocator reserves after closing
  each raw shard;
- `FASTSAFETENSORS_UNIFIED_MEM=0` selects the no-GDS bounce-buffer copier on
  GB10.

The last setting is essential for a near-full model. The unified copier pins a
complete mmap source while also allocating a complete CUDA destination. For
the 5.13 GB GLM raw shards, one logical shard therefore becomes roughly two
physical shard-sized residencies and OOMs PP2. The no-GDS copier retains one
CUDA shard plus bounded bounce storage.

With 16 readers, 64 MiB requests, and a 16 MiB bounce setting, all 22 raw
batches completed in about 197–198 seconds. Complete model loading took
205.96–206.78 seconds and the API was ready in 4 minutes 36 seconds with no
kernel process kill or service restart.

## Other startup controls

- `DGX_RAY_BUNDLE_NODE_IPS` pins placement-group bundles to ordered node IPs,
  keeping PP rank identity aligned with node-local caches.
- Distributed FlashInfer warmup uses each node's metadata-keyed persistent
  cache. The validated topology has one rank per node; do not share the same
  cache path among concurrently tuning ranks on one node.
- `VLLM_WEIGHT_CACHE_RECLAIM_WINDOW_SHARDS` controls exact loader-owned page
  reclaim. The balanced validated value is `2`; a global drop-cache daemon is
  not required.

See the `glm-504b-fast-loading` branch of `mardelden/dgx-spark` for Docker
artifacts, systemd experiment wrappers, bootstrap tooling, measurements, and
lessons learned.
