# Decision 004: Persist startup work by phase and exact fingerprint

**Status:** Accepted; individual release promotion remains gate-controlled
**Date:** 2026-08-24

## Context

Once weight transport improved, later startup phases became dominant. GLM repeated about
116 seconds of distributed FlashInfer tuning on every restart. RTX/Qwen repeated multimodal
memory profiling, frontend/processor work, AOT artifact discovery, and conservative language-model
profiling even when an unchanged deployment already knew the exact KV allocation.

Calling all of this "the cache" obscures incompatible identities and lifetimes. Model files,
OS pages, prepared weights, compiled kernels, autotune tactics, startup plans, and CUDA graphs
do not mean the same thing and must not share one informal validity rule.

## Decision

Treat each reusable phase as a separate, fingerprinted artifact:

| Artifact | Persisted identity must cover | Reuse rule |
| --- | --- | --- |
| Local model snapshot | Repository/revision and complete config/tokenizer/processor files | Offline only after completeness is proven |
| vLLM/Torch/AOT artifacts | Runtime source/package, PyTorch/toolchain, model/config, GPU architecture | Accept only the exact resolved artifact/hash |
| Triton/FlashInfer JIT | Kernel source, toolchain, backend, GPU architecture | Persist workspace; executable backend preflight required |
| FlashInfer autotune tactics | Kernel metadata, shapes/dtypes, runtime config, rank/collective context | Metadata-validated per rank; preserve collective call order |
| Startup plan | Immutable model/config/GPU fingerprint and exact allocated KV bytes | Skip only redundant memory measurement; keep correctness warmups |
| Multimodal token budget | Processor/model revision and input constraints | Use fingerprinted value; recompute on mismatch |
| CUDA graphs | Process, addresses, graph configuration | Process-local only; capture again after restart |

Persisted state must live outside ephemeral containers and in release/runtime-specific namespaces.
Native and container runtimes must not write the same Torch/AOT/startup-plan/FlashInfer roots merely
because their source commit and nominal paths match.

On an exact startup-plan hit, vLLM may reuse the exact KV allocation and skip redundant multimodal
encoder memory profiling. It must retain language-model compilation/warmup, sampler profiling,
MTP setup, kernel warmup, and CUDA graph capture unless a separate validated artifact explicitly
covers that work.

## Evidence

- GLM distributed FlashInfer initialization fell from 116.48 seconds to 9.33 seconds, and later
  measured about 5.95--7.20 seconds, after per-rank metadata-validated tactics were persisted.
- RTX/Qwen stock readiness was 65.681 seconds. Corrected weight transport plus multimodal overlap
  reached 48.513 seconds; adaptive direct I/O reached 47.490 seconds; an exact schema-2 startup-plan
  hit reached 38.416 seconds while preserving the exact 1,144,164-token KV capacity.
- The complete research stack reached 14.914 seconds, but several sub-second changes were excluded
  from the maintained patch set because their complexity exceeded their individual value.
- Generic startup release 3 passed functional loader/XQA/MTP/correctness gates but its fleet canary
  reached health in 33.392 seconds and failed the immutable 20-second promotion gate. It remains
  failed/quarantined evidence, not a successful production release.
- Sharing native cache roots with Docker caused a 367.310-second Docker population run and later
  invalidated native FlashInfer artifacts. Runtime-specific namespaces are therefore required.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Keep a sleeping process with model state in memory | Does not support the operating model of stopping containers between model switches |
| One shared `cache/` directory for every runtime | Cross-runtime invalidation and rebuilds were measured even with nominally identical source |
| Skip all warmup/profile/graphs on a plan hit | Risks incorrect capacity, missing compilation, or serving regressions |
| Persist CUDA graph objects to disk | Addresses and process state are not portable across restart |
| Enable offline mode before artifacts are complete | Converts a resolvable dependency problem into a startup failure |

## Consequences

- A population boot and a steady-state restart are different test cases and must be reported
  separately.
- Cache files need atomic publication, schema versions, hit logs, and removal conditions.
- Every meaningful model, runtime, quantization, context, concurrency, graph, or GPU change should
  produce a miss/new generation rather than reuse a plausible-looking artifact.
- The deployment must persist the required roots and attest that the runtime can execute the chosen
  backend, not merely import its Python package.

## References

- [Exact startup-plan reuse](https://github.com/mardelden/vllm-custom/commit/74cd835791f19eb0abbcd843a4e307e5e993c519)
- [Persisted startup overlap](https://github.com/mardelden/vllm-custom/commit/4f1cd0856c291c099c30f4afd5a8c7cbed3b8e70)
- [Release 3 handover](../003-generic-startup-release-3-deployment-handover.md)
- [GB10 memory/loading report](https://vitrina.vulcandom.com/w/dgx-spark-gb10-memory-loading/)
