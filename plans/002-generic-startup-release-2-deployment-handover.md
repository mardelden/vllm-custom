# Generic startup release 2 deployment handover

**Status:** Source contract implemented; fleet canary required
**Date:** 2026-08-22
**Patch set:** `vllm-generic-startup` release 2
**Code:** `mardelden/vllm-custom@74cd835791f19eb0abbcd843a4e307e5e993c519`
**Dependency:** `mardelden/fastsafetensors@b43888df0eac286849f1238b7e42e254ee1d285f`

## Context

The release 1 fleet canary proved the optimized loader path: target weights loaded in 1.22 seconds,
MTP weights in 0.11 seconds, and total model loading completed in 2.351709 seconds. Startup later
failed in `profile_cudagraph_memory`. The full source overlay selected FlashInfer XQA on SM120, but
`vllm.utils.flashinfer.has_flashinfer()` returned false because the image had no
`flashinfer-cubin` and the mounted `/usr/local/cuda/bin/nvcc` was not on `PATH`.

The Python 3.11 shim made `flashinfer-python` importable; it did not make a backend executable. The
release 1 attestation checked package identity and imports, so it could not detect this mismatch.

## Decision

Release 1 remains an immutable failed-canary artifact. Do not edit its manifest, retag its image, or
select it as a fallback. Release 2 uses the same pinned vLLM source and FastSafetensors dependency,
but adds the missing runtime closure and executable promotion gates. The authoritative contract is
[`overlays/generic-startup/manifest-r2.json`](../overlays/generic-startup/manifest-r2.json).

The deployment shape remains unchanged: ordinary model settings live in profiles, a profile selects
an already-built patch-set image, and the manifest owns patch-set activation. A configuration-only
operation must not build or mutate an image.

## Immutable release identity

| Item | Release 2 value |
| --- | --- |
| Patch set | `vllm-generic-startup@2` |
| Suggested profile | `qwen38-nvfp4-mtp-generic-startup-r2` |
| Suggested image tag | `vulcandom/vllm:0.27.1-cu129-generic-startup-r2-74cd835-b43888d` |
| vLLM code commit | `74cd835791f19eb0abbcd843a4e307e5e993c519` |
| vLLM Python package tree | `1ebce48c69e06c6ee219a7e764f3e641cc317ede` |
| FastSafetensors commit | `b43888df0eac286849f1238b7e42e254ee1d285f` |

The fleet owns its image registry and may adapt the suggested names, but the release number, source
commits, package tree, manifest contents, image digest, and selected profile must be asserted before
changing the running service.

## Deployment delta from release 1

Keep the release 1 loader and startup-plan configuration. Add this activation environment from the
release 2 manifest:

```text
PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
FLASHINFER_WORKSPACE_BASE=/var/cache/vllm
```

The production container requires these mounts:

```text
/usr/local/cuda:/usr/local/cuda:ro
/var/cache/vllm:/var/cache/vllm
```

`/usr/local/cuda/bin/nvcc` must be executable inside the container. The cache mount must be writable
and must survive container removal. With `FLASHINFER_WORKSPACE_BASE=/var/cache/vllm`, FlashInfer's
JIT cache is expected under `/var/cache/vllm/.cache/flashinfer`.

Do not add a runtime source bind mount. Do not disable XQA as the release fix. Installing large
precompiled FlashInfer packages is not required for this release because the production CUDA toolkit
is already mounted and the persistent cache amortizes JIT compilation across container lifetimes.

## Required executable preflight

Run this inside the candidate image with the production mounts and release 2 activation environment:

```bash
python3 -c "import shutil; from vllm.utils.flashinfer import has_flashinfer; assert shutil.which('nvcc') == '/usr/local/cuda/bin/nvcc'; assert has_flashinfer()"
```

An import-only FlashInfer check is insufficient. A nonzero result prevents changing the running
service.

## Promotion gates

### XQA, API readiness, and zero restarts

The same canary process must:

1. Log `decode_backend=xqa`.
2. Complete `profile_cudagraph_memory` and CUDA-graph setup without a backend error.
3. Reach the configured API readiness endpoint.
4. Keep the service and container restart counters at zero through readiness and the representative
   serving check.

### Persistent FlashInfer cache reuse

1. Complete a first start and representative MTP request.
2. Verify `/var/cache/vllm/.cache/flashinfer` is non-empty and record its artifact paths.
3. Remove and recreate the container with the same cache mount.
4. Verify the second start reaches API readiness with XQA and zero restarts without rebuilding the
   already-recorded XQA artifacts.
5. Verify those artifacts remain present and were reused from the persistent mount.

Before default promotion, also run fixed correctness, MTP depth-5 generation, representative TTFT,
ITL, throughput, and peak host RAM/VRAM checks. Prove rollback directly to stock; release 1 is not a
rollback target.

## Ownership boundary

The `vllm-custom` repository owns this manifest, the pinned source facts, and these acceptance gates.
The fleet repository owns image construction, actual service/profile names, mount implementation,
attestation, canary rollout, health observation, and rollback. If fleet validation finds another
contract defect, fix the source-owned release contract here and issue a new immutable release rather
than silently forking it in deployment code.
