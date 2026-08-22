# Internal overlay contract

This directory is the handoff boundary between the `vllm-custom` fork and the
fleet repository. The fork owns source changes, compatibility facts, and
validation evidence. The deployment repository owns image construction,
service configuration, secrets, rollout, and rollback.

There is one active engine overlay:

- [`generic-startup/manifest.json`](generic-startup/manifest.json) is the
  machine-readable release contract.
- [`plans/001-generic-startup-overlay-deployment-handover.md`](../plans/001-generic-startup-overlay-deployment-handover.md)
  explains how to consume and verify it.

The branch named in the manifest is a moving development line. Deployments must
use the immutable `code_commit`, never the branch head. A model/profile switch
must select an already-built image; a configuration-only restart must not build
or mutate the overlay.

Fleet-wide compatibility work remains outside this patch set. In particular,
the FlashInfer Python 3.11 annotation shim and the compatible FastAPI,
Starlette, and Prometheus-instrumentator pins remain owned by the shared vLLM
deployment role.
