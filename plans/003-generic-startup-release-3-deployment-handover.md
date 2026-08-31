# Generic startup release 3 deployment handover

- **Status:** Source contract implemented; fleet canary required
- **Date:** 2026-08-22
- **Patch set:** `vllm-generic-startup` release 3
- **Code:** `mardelden/vllm-custom@4f1cd0856c291c099c30f4afd5a8c7cbed3b8e70`
- **Package tree:** `ca93282614ec21c41a5b67191048abd5ec8309d3`
- **Dependency:** `mardelden/fastsafetensors@b43888df0eac286849f1238b7e42e254ee1d285f`

## Outcome

Release 2 is healthy and immutable, but its approximately 57-second warm API readiness is expected:
it contains the optimized loader and exact KV startup plan, not the later combined startup stack that
measured approximately 15 seconds. Release 3 packages the material, generic parts of that stack. It
does not disable compilation, sampling, multimodal support, MTP, FlashInfer XQA, or CUDA graphs.

The authoritative contract is
[`overlays/generic-startup/manifest-r3.json`](../overlays/generic-startup/manifest-r3.json). Release 2
and [`manifest-r2.json`](../overlays/generic-startup/manifest-r2.json) remain unchanged and are the
rollback target.

## Immutable release identity

| Item | Release 3 value |
| --- | --- |
| Patch set | `vllm-generic-startup@3` |
| Suggested profile | `qwen38-nvfp4-mtp-generic-startup-r3` |
| Suggested image | `vulcandom/vllm:0.27.1-cu129-generic-startup-r3-4f1cd08-b43888d` |
| vLLM code commit | `4f1cd0856c291c099c30f4afd5a8c7cbed3b8e70` |
| vLLM Python package tree | `ca93282614ec21c41a5b67191048abd5ec8309d3` |
| FastSafetensors commit | `b43888df0eac286849f1238b7e42e254ee1d285f` |

Build a new derived image from the complete pinned `vllm` package. Do not retag r2, copy scratch
files into production, or copy only the r3 diff onto the r2 package.

## Deployment delta from release 2

Keep the r2 CUDA and persistent-cache mounts, FlashInfer runtime closure, FastSafetensors dependency,
loader geometry, and shared-role compatibility fixes. Add these environment variables:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
VLLM_ENABLE_AOT_LOAD_OVERLAP=1
VLLM_WORKER_MULTIPROC_METHOD=forkserver
```

Offline mode requires the complete pinned model snapshot, tokenizer, processor, configuration, and
remote code to be synchronized before activation. A missing artifact fails deployment; it is not a
reason to silently remove the flags.

Launch the same serve arguments through this module instead of the standard CLI entrypoint:

```text
python3 -m vllm.entrypoints.openai.api_server_bootstrap <existing r2 serve arguments>
```

The early module starts the AsyncLLM-preloaded forkserver before importing the large API parent
graph. The normal launcher then verifies the same start method and completes its setup idempotently.

The following r2 activation remains required:

```text
PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
FLASHINFER_WORKSPACE_BASE=/var/cache/vllm
VLLM_CACHE_ROOT=/var/cache/vllm
VLLM_ENABLE_STARTUP_PLAN=1
```

The mounts remain:

```text
/usr/local/cuda:/usr/local/cuda:ro
/var/cache/vllm:/var/cache/vllm
```

Container removal is supported. The speedup depends on preserving disk caches, not container RAM or
a sleeping process. Keep the startup-plan, multimodal-budget, Torch/AOT, AOT-alias, and FlashInfer
artifacts under `/var/cache/vllm` across container recreation and model switches.

## What r3 adds

- A fingerprinted cache for the multimodal maximum-token result. The earlier experiment removed
  about five seconds in each of the API and EngineCore processes.
- Renderer and input/output processor construction overlapped with EngineCore launch and loading.
- An early AsyncLLM-preloaded forkserver, with idempotent normal launcher setup.
- Language-model profiling bounded to the request/CUDA-graph requirement on an exact startup-plan
  hit instead of profiling all `max_num_batched_tokens`.
- Fingerprinted AOT aliases and background artifact deserialization during weight reads. A preloaded
  artifact is accepted only when its resolved path matches the current full AOT hash.

The direct Qwen tokenizer path, FlashInfer connectivity prefetch, lazy tokenizers, custom-op cache,
and other sub-second experiments remain excluded.

## Readiness definition

The historical approximately 15-second measurement was a fresh API/service process with Qwen
checkpoint pages explicitly evicted and persistent compilation/startup caches retained. It was not a
first-ever JIT/cache-population boot and did not retain model weights in process RAM.

For the fleet canary:

1. Allow one population boot and representative MTP request to create persistent artifacts.
2. Stop and remove the container without deleting `/var/cache/vllm`.
3. Evict only the Qwen checkpoint pages.
4. Create a new r3 container and measure process start to successful readiness.
5. Repeat three times; every page-cold run must be at most 20.0 seconds with zero restarts.

Required hit evidence is listed exactly in the manifest. In particular, the logs must show the exact
startup-plan hit, multimodal max-token cache hit, bounded profile size, and AOT artifact consumed from
the weight-load overlap.

## Adaptive loader and serving gates

The page-cold run must exercise aligned O_DIRECT; an immediate resident-checkpoint run must exercise
the buffered path. Both must keep total target-plus-MTP model loading at or below 5 seconds. Also
retain the r2 gates for executable XQA, persistent FlashInfer cache reuse, correctness, MTP depth 5,
TTFT, ITL, throughput, peak RAM/VRAM, API health, and zero restarts.

If r3 misses the 20-second ceiling, preserve its logs and stage timings and keep r2 selected. Do not
change the r3 image or manifest in place. A source or contract fix requires a new immutable release.

## Source validation completed

- All repository hooks passed for the r3 source and tests, including Ruff, mypy, SPDX, imports,
  environment-schema validation, and forbidden-API checks.
- Four targeted unit tests passed in a disposable container based on the exact live r2 image.
- An isolated AOT gate rejected a stale alias path and accepted the matching artifact.
- An executable disposable-container gate started the early forkserver and repeated the normal API
  forkserver setup without error.

The fleet repository still owns image construction, profile/service names, mounts, attestation,
cache population and page eviction, canary observation, promotion, and rollback.
