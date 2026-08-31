# Decision 002: Filter unowned checkpoint ranges before materialization

**Status:** Accepted, with model/rank-specific ownership adapters
**Date:** 2026-08-24

## Context

Under GLM pipeline parallelism, every rank originally traversed the full 296 GiB checkpoint
and discarded weights for layers it did not own. PP0 ultimately needed about 83.6 GiB and the
workers about 103.6 GiB each. Filtering after tensor materialization saves destination work but
not disk I/O, page faults, staging memory, or source-to-device traffic.

The same ownership problem appeared inside Qwen target/MTP loading on one GPU: the target did
not need `mtp.*` ranges, while the MTP model needed those ranges but could share the target's
embedding and `lm_head` rather than loading private copies.

## Decision

Apply ownership at the earliest layer that has both trustworthy model semantics and byte-range
metadata:

- For PP, derive owned layers or missing prefixes from the constructed destination model,
  then reject non-local checkpoint ranges before reading their payload.
- For target/draft pairs, give each model an explicit source-range predicate and an explicit
  sharing contract for identical destination parameters.
- Keep unscoped/shared tensors unless ownership can be proven. Unknown naming conventions must
  fail open for correctness or fail configuration when a strategy requires exact filtering.
- Reject loader strategies whose iterator cannot preserve the ownership contract instead of
  silently reading all data and claiming a filtered load.
- Validate the resulting parameter state against an unfiltered stock load by rank/model identity,
  not by physical host.

Filtering is independent of transport. It composes with buffered, O_DIRECT, pinned,
FastSafetensors, or prepared-cache readers.

## Evidence

The GLM filter reduced each rank from 296 GiB and 152,921 raw tensor entries to:

| Rank | Bytes read | Tensor entries handled |
| --- | ---: | ---: |
| PP0 | 83.6 GiB | 42,709 |
| PP1/PP2 | 103.6 GiB | 54,848 |

All ranks matched a stock-loader-written cache over 1,100 / 1,271 / 1,263 destination ranges
with zero mismatches, and generation returned `391`.

For Qwen3.8-27B-NVFP4, target/MTP range filtering plus sharing reduced MTP loading from about
0.57 seconds to about 0.27 seconds. It did not reduce reported post-load VRAM because the CUDA
caching allocator retained released blocks; no VRAM-saving claim is attached to that result.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Read every shard on every PP rank and discard downstream | Wastes storage bandwidth, page residency, staging, and setup time |
| Infer ownership only from host name | Ray can move PP ranks between hosts; host identity is not semantic rank identity |
| Hard-code GLM layer prefixes as a generic feature | Unsafe for other model naming and sharing conventions |
| Count returned raw names as proof of correctness | vLLM can report names that downstream PP logic skipped; destination digests/ranges are the useful gate |

## Consequences

- A single-GPU PP=1 model gets no PP byte reduction; do not carry a no-op filter and claim a win.
- Rank placement and rank-local caches are deployment concerns, but the loader must expose stable
  rank identity where cache keys depend on it.
- Shared target/draft parameters need lifetime and mutation rules. Sharing is valid only when
  later initialization does not independently mutate the two logical copies.
- Every new model adapter needs equality and completeness tests, including non-layer-scoped
  parameters, embeddings, heads, and tied weights.

## References

- [Generic opt-in PP filter](https://github.com/mardelden/vllm-custom/commit/9cbeff386f09259905ee740ad54dfe405fc83d78)
- [Qwen target/MTP filtering and sharing](https://github.com/mardelden/vllm-custom/commit/53e1502cb315e41567ef9c3efb452afd71e666d2)
- [Cross-team loader evidence](https://vitrina.vulcandom.com/w/dgx-spark-weight-loading/)
