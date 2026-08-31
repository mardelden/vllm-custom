# Generic startup overlay deployment handover

**Status:** Ready for deployment-team packaging; not enabled on the production service
**Patch set:** `vllm-generic-startup` release 1
**Code:** `mardelden/vllm-custom@74cd835791f19eb0abbcd843a4e307e5e993c519`
**Dependency:** `mardelden/fastsafetensors@b43888df0eac286849f1238b7e42e254ee1d285f`

## Decision

Maintain one generic startup patch set in the `vllm-custom` fork. It contains the five substantial,
validated changes already on `codex/fastsafetensors-parallel-mtp-share`. The five later sub-second
experiments are excluded.

The deployment repository should consume an immutable code commit and bake it into a derived image.
The development branch is only a moving pointer for future experiments. Normal model profiles may
select an already-built patch-set image and supply runtime settings, but a config-only run must not
build or modify the image.

The authoritative machine-readable contract is
[`overlays/generic-startup/manifest.json`](../overlays/generic-startup/manifest.json).

## Ownership boundary

| Owner | Responsibility |
| --- | --- |
| `vllm-custom` | Patch source, dependency pin, compatibility statement, benchmarks, correctness evidence, and removal conditions |
| Deployment/fleet repository | Derived image, idempotent application, source/version assertions, service profiles, persistent volumes, secrets, rollout, and rollback |
| Shared vLLM role | Fleet-wide FlashInfer Python 3.11 shim and compatible FastAPI/Starlette/Prometheus-instrumentator pins |

The shared-role compatibility fixes are prerequisites, not part of this source patch set. Keeping them
separate avoids duplicating a fleet concern in every model/platform overlay.

## Why this is a full source overlay

The five retained commits are based on upstream commit
`ba07e4a48fc951300d97eb506217dd530583dea3`, not the `v0.27.1` release source tree. The experiment
used the complete Python package from the fork on top of the `0.27.1+cu129` binary installation so
the compiled extensions remained available.

Do not copy only the 14 files changed by the five commits into the stock 0.27.1 Python package. That
combination was not tested and several base-file hashes differ. Use one of these fail-closed paths:

1. Bake the complete `vllm/` Python package exported from the exact code commit into the derived
   image while preserving the base image's compiled extension files. This matches the tested
   prototype shape.
2. Build/package the exact fork revision as a coherent custom vLLM artifact.
3. For a future pinned vLLM release, port the five commits to that release and rerun all gates before
   updating the manifest.

The scratch directory on `vllm-code`,
`/opt/vllm-startup-exp/repo-loader-candidate-overlay`, is not a release input. It contains an entire
working package plus instrumentation and the five discarded experiments.

## Retained components

| Component | Commit | Scope | Effect |
| --- | --- | --- | --- |
| Multimodal warmup overlap | `809eb0bcb7` | Core with Qwen3-VL specialization | Moves processor warmup off the serial engine-start path |
| FastSafetensors controls | `5684372cc9` | Generic loader | Exposes reader, queue, bounce-buffer, copy-block, and direct-I/O controls |
| Qwen range filtering/MTP sharing | `53e1502cb3` | Qwen3.5/MTP only | Avoids unnecessary byte materialization and duplicate target weights |
| Adaptive direct I/O | `3673260f2b` | Generic loader | Selects aligned O_DIRECT for cold ranges and buffered reads for resident ranges |
| Exact startup-plan reuse | `74cd835791` | Generic engine | Reuses exact KV capacity and skips only redundant MM memory profiling |

This is one deployment patch set even though its internals are traceable as components. Generic code
paths remain generic; Qwen-specific behavior is conditional and does not claim a benefit for every
model.

## Required image contents

- Base runtime compatible with the tested Python 3.11, CUDA 12.9, and vLLM binary extensions.
- Complete `vllm/` Python source tree from `74cd835791f19eb0abbcd843a4e307e5e993c519`.
- FastSafetensors 0.3.3 from
  `b43888df0eac286849f1238b7e42e254ee1d285f`, installed during image construction.
- Existing FlashInfer 0.6.16 Python 3.11 shim, which inserts
  `from __future__ import annotations` in `fd_exchange.py`.
- Existing dependency pins: FastAPI 0.136.3, Starlette 0.52.1, and
  `prometheus-fastapi-instrumentator` 7.1.0.

Tag the image with the immutable vLLM and FastSafetensors short SHAs. Do not tag only as `latest` or
derive the deployed revision from a branch head.

## Runtime contract

Persist a writable cache directory across container deletion:

```bash
VLLM_CACHE_ROOT=/var/cache/vllm
VLLM_ENABLE_STARTUP_PLAN=1
```

Use the tested loader configuration for this RTX/Qwen profile:

```json
{
  "load_format": "fastsafetensors",
  "model_loader_extra_config": {
    "queue_size": -1,
    "max_threads": 16,
    "bbuf_size_kb": 16384,
    "max_copy_block_size": 536870912,
    "use_o_direct": "auto"
  }
}
```

The startup plan is regenerable derived state on disk. The first start for a new immutable
model/config/GPU fingerprint populates it; later containers can reuse it. This does not require a
sleeping process or a surviving container.

Enable `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` only after the model, configuration,
tokenizer, and processor artifacts are present locally. Persist model, vLLM, Triton/FlashInfer, and
startup-plan caches independently of the container filesystem.

## Measured result

The controlled Qwen3.8-27B-NVFP4 experiment used targeted cold checkpoint ranges and persistent warm
compiler/JIT caches:

| Metric | Stock | Patch set | Reduction |
| --- | ---: | ---: | ---: |
| Model load | 8.535 s | 3.186 s | 62.7% |
| Process start to API ready | 65.681 s | 38.416 s | 41.5% |

The fixed correctness response remained exact, KV capacity remained 1,144,164 tokens, and language
model warmup, compilation, sampler profiling, MTP, kernel warmup, FP8 KV, and CUDA graph capture
remained enabled. The one-second health polling interval limits API-ready precision.

Recorded focused validation:

- 76 FastSafetensors core/range tests passed.
- 17 vLLM loader/configuration tests passed.
- 14 worker/renderer-warmup tests passed.
- Focused ruff and formatting checks passed.

Before broader rollout, the final derived image still needs representative TTFT, ITL, throughput,
peak host RAM/VRAM, restart, and rollback checks.

## Deployment verification

The deployment implementation should fail before changing the running service unless all of these
are true:

1. The fetched vLLM commit is exactly the manifest's `code_commit` and its `vllm` Git tree is
   `1ebce48c69e06c6ee219a7e764f3e641cc317ede`.
2. FastSafetensors resolves to exactly the manifest commit.
3. The derived image reports the intended Python/CUDA/vLLM/FastSafetensors versions.
4. The FlashInfer Python 3.11 shim and web dependency pins are still effective.
5. The cache mount is writable by the service user and survives container replacement.
6. A first start saves a startup plan and the unchanged second start logs that it applies the plan.
7. The readiness probe, fixed correctness request, representative serving benchmark, and memory
   gates pass.

Useful source-side checks from the vLLM development environment are:

```bash
.venv/bin/python -m pytest \
  tests/model_executor/model_loader/fastsafetensors_loader/test_weight_utils.py \
  tests/model_executor/model_loader/test_registry.py \
  tests/models/test_qwen3_5_mtp_config.py

.venv/bin/python -m pytest \
  tests/models/multimodal/processing/test_qwen3_vl.py \
  tests/renderers/test_warmup.py \
  tests/v1/engine/test_engine_core_client.py \
  tests/v1/worker/test_gpu_worker.py
```

Run the equivalent FastSafetensors unit suite from its pinned source checkout. A source-test pass does
not replace the final image-level startup and serving checks.

## Rollback

Select the previous stock image and remove the patch-set-specific FastSafetensors configuration and
startup-plan opt-in. Do not mutate or uninstall files inside a running container. Existing startup
plan files may remain because they are cache data; incompatible fingerprints and schema versions are
ignored. Retain the failed image tag and logs for diagnosis.

## Excluded work

The following later experiments are deliberately absent because their measured improvements were
only about 0.09-0.87 seconds and did not justify their maintenance and invalidation surface:

- FlashInfer connectivity prefetch;
- multimodal budget cache plus lazy tokenizers;
- custom-op wrapper cache;
- lazy TorchInductor patch;
- direct Qwen tokenizer construction.

The production `vllm.service` was restored to the stock image after experimentation. This handover
does not authorize an in-place production mutation; deployment and rollout remain fleet-owned.
