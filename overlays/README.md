# Internal overlay contract

This directory is the handoff boundary between the `vllm-custom` fork and the
fleet repository. The fork owns source changes, compatibility facts, and
validation evidence. The deployment repository owns image construction,
service configuration, secrets, rollout, and rollback.

There is one engine overlay with immutable release contracts:

- [`generic-startup/manifest.json`](generic-startup/manifest.json) is release 1.
  Its fleet canary failed after model loading because FlashInfer XQA was
  importable but not executable. It remains unchanged for provenance and must
  not be deployed or retagged.
- [`generic-startup/manifest-r2.json`](generic-startup/manifest-r2.json) is the
  healthy immutable release 2 contract and current rollback target. It added
  the CUDA compiler path, persistent FlashInfer workspace, required mounts,
  executable preflight, and fail-closed promotion gates.
- [`generic-startup/manifest-r3.json`](generic-startup/manifest-r3.json) is the
  release 3 deployment candidate. It retains the release 2 runtime closure and
  adds the material persisted-cache, overlap, early-forkserver, bounded-profile,
  and AOT preload stack.
- [`plans/003-generic-startup-release-3-deployment-handover.md`](../plans/003-generic-startup-release-3-deployment-handover.md)
  is the current deployment handoff. The
  [`release 2 handoff`](../plans/002-generic-startup-release-2-deployment-handover.md)
  documents the rollback release, and the
  [`release 1 handoff`](../plans/001-generic-startup-overlay-deployment-handover.md)
  remains historical evidence.

The branch named in the manifest is a moving development line. Deployments must
use the immutable `code_commit`, never the branch head. A model/profile switch
must select an already-built image; a configuration-only restart must not build
or mutate the overlay.

Fleet-wide compatibility work remains outside this patch set. In particular,
the FlashInfer Python 3.11 annotation shim and the compatible FastAPI,
Starlette, and Prometheus-instrumentator pins remain owned by the shared vLLM
deployment role. The shim proves Python import compatibility only; each release
that selects a native backend must also define and pass its executable runtime
gate.
