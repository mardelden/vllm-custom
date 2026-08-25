# Decision 006: Package patches as immutable release contracts

**Status:** Accepted
**Date:** 2026-08-24

## Context

Several investigations found source code on a workstation that was not present in the selected
node image. The RTX release-1 canary also proved that Python importability is not an executable
runtime contract: the full overlay selected FlashInfer XQA, but the image could not build/execute
it because the mounted CUDA compiler was absent from PATH.

The maintained RTX patch set is based on upstream commit `ba07e4a48f`, not the stock v0.27.1
Python source tree. Copying only the changed files onto that release was never validated and can
mix incompatible base-file assumptions.

## Decision

`vllm-custom` owns source-level release facts:

- exact vLLM commit and full Python package tree;
- exact external dependency commits;
- component list, default gates, compatibility assumptions, and removal conditions;
- unit/correctness/performance evidence and immutable promotion thresholds;
- required runtime environment, mounts, and executable preflights.

The fleet/deployment repository owns image construction, registry identity/digest, service profiles,
mount implementation, secrets, attestation, canary rollout, promotion, and rollback.

Build a coherent artifact from the complete pinned fork package or port the changes to a pinned
release and repeat all gates. Do not mutate a running container, use a branch head as release
identity, or copy a subset of files across unmatched source bases.

Every changed contract receives a new immutable release number. Failed release images, manifests,
logs, and measurements are preserved and never silently retagged.

## Release history and standing

| Release | Outcome | Standing |
| --- | --- | --- |
| Generic startup r1 | Proved about 2.35-second model loading, then failed in XQA setup because the executable CUDA/FlashInfer closure was incomplete | Immutable failed canary; never a rollback target |
| Generic startup r2 | Added CUDA PATH/mount, executable preflight, and persistent FlashInfer workspace; measured about 56.2-second warm readiness | Healthy immutable container rollback |
| Generic startup r3 | Added the retained combined startup stack; functional gates passed, but page-cold fleet health was 33.392 seconds against a 20-second ceiling | Failed/quarantined; do not promote or edit in place |

## Required gates

Before a fleet changes the selected service image, require:

1. Exact source commit, full package tree, dependency commit, image digest, Python/CUDA/Torch
   versions, and model revision.
2. Assertions that the intended code markers exist inside the built artifact.
3. Executable backend preflights, not import-only checks.
4. Writable, persistent, release-specific cache roots and a deliberate population procedure.
5. Page-cold and resident model-load measurements, process-to-readiness wall time, and zero restarts.
6. Fixed correctness, model-specific MTP/multimodal checks, TTFT/ITL/throughput, and peak RAM/VRAM.
7. A proved rollback to the prior healthy immutable artifact.

Native and packaged runtimes use distinct cache namespaces. A source commit match does not make
their bytecode, Torch/AOT, startup-plan, or FlashInfer artifacts interchangeable.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Bind-mount a host checkout into production | Reproducibility and rollback depend on mutable workstation state |
| Build images during config-only deployment | Couples service configuration to source mutation/build failures |
| Copy only the apparent changed files | Base-file/version skew was not tested and can fail after the fast loader succeeds |
| Retag a failed canary after a small fix | Destroys the evidence and identity of what was actually tested |
| Share one cache root between native and Docker | Measured cross-runtime misses and invalidation made both paths slower |

## Consequences

- The manifests under `overlays/generic-startup/` are release artifacts, not editable examples.
- The release contract includes web/runtime dependency compatibility when required by the selected
  image, even if those pins are fleet-owned rather than vLLM optimizations.
- Build-time bytecode is reasonable packaging hygiene, but its 3.3--3.6 second improvement did not
  justify a release by itself and did not close the native/container gap.
- A deployment can be functionally correct and still fail its immutable latency gate.

## References

- [Release 1 handover](../001-generic-startup-overlay-deployment-handover.md)
- [Release 2 handover](../002-generic-startup-release-2-deployment-handover.md)
- [Release 3 handover](../003-generic-startup-release-3-deployment-handover.md)
- [`overlays/generic-startup`](../../overlays/generic-startup/)
