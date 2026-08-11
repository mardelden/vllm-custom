# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transactional, PP-safe lifecycle helpers for the prepared-weight cache."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable
from typing import TypeVar

_ModelT = TypeVar("_ModelT")

try:
    from vllm.logger import init_logger
except ModuleNotFoundError:  # Allow the filesystem lifecycle tests to run locally.
    logger = logging.getLogger(__name__)
else:
    logger = init_logger(__name__)


def cache_is_complete(cache_dir: str) -> bool:
    """Return whether *cache_dir* is a published cache generation."""
    manifest_path = os.path.join(cache_dir, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    files = manifest.get("files", [])
    return (
        bool(manifest.get("complete"))
        and isinstance(files, list)
        and all(
            isinstance(filename, str)
            and os.path.isfile(os.path.join(cache_dir, filename))
            for filename in files
        )
    )


def _pp_all_true(local_value: bool, operation: str) -> bool:
    """Return whether *local_value* is true on every PP rank."""
    import torch

    from vllm.distributed import get_pp_group

    pp_group = get_pp_group()
    if pp_group.world_size == 1:
        return local_value

    flag = torch.tensor([int(local_value)], dtype=torch.int32, device="cpu")
    torch.distributed.all_reduce(
        flag,
        op=torch.distributed.ReduceOp.MIN,
        group=pp_group.cpu_group,
    )
    result = bool(flag.item())
    logger.info(
        "weight-cache PP consensus for %s: local=%s, all=%s (%d ranks)",
        operation,
        local_value,
        result,
        pp_group.world_size,
    )
    return result


def all_ranks_cache_ready(local_ready: bool) -> bool:
    """Choose warm restore only when every PP rank has a complete cache."""
    return _pp_all_true(local_ready, "cache readiness")


def _save_barrier(local_success: bool) -> bool:
    if os.getenv("VLLM_WEIGHT_CACHE_SAVE_BARRIER", "0") != "1":
        return local_success
    return _pp_all_true(local_success, "transactional save")


def _write_build_sentinel(cache_dir: str) -> None:
    """Atomically tell the outer supervisor that a generation is complete."""
    sentinel_path = os.getenv("VLLM_WEIGHT_CACHE_BUILD_SENTINEL", "").strip()
    if not sentinel_path:
        return
    payload = {
        "v": 1,
        "cache_dir": os.path.realpath(cache_dir),
        "pid": os.getpid(),
        "published_at": int(time.time()),
    }
    sentinel_dir = os.path.dirname(sentinel_path) or "."
    os.makedirs(sentinel_dir, exist_ok=True)
    temporary_path = f"{sentinel_path}.{os.getpid()}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as sentinel_file:
        json.dump(payload, sentinel_file)
    os.replace(temporary_path, sentinel_path)
    logger.info("weight-cache build sentinel published: %s", sentinel_path)


def wrap_cache_save(
    save: Callable[[_ModelT, str], None],
) -> Callable[[_ModelT, str], None]:
    """Publish the base cache writer atomically and synchronize PP ranks."""

    def save_transactionally(model: _ModelT, cache_dir: str) -> None:
        save_error: BaseException | None = None
        all_saves_succeeded = False
        staging_dir = f"{cache_dir}.building-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            if cache_is_complete(cache_dir):
                logger.info(
                    "weight-cache already complete; preserving generation: %s",
                    cache_dir,
                )
            else:
                logger.info("weight-cache staging save: %s", staging_dir)
                save(model, staging_dir)
                if not cache_is_complete(staging_dir):
                    raise RuntimeError(
                        "weight-cache staging save did not publish a complete manifest"
                    )

                if os.path.exists(cache_dir):
                    abandoned_dir = (
                        f"{cache_dir}.abandoned-{int(time.time())}-"
                        f"{uuid.uuid4().hex[:8]}"
                    )
                    os.replace(cache_dir, abandoned_dir)
                    logger.warning(
                        "preserved incomplete weight-cache attempt at %s",
                        abandoned_dir,
                    )
                os.replace(staging_dir, cache_dir)
                logger.info("weight-cache staged generation published: %s", cache_dir)
        except BaseException as error:
            save_error = error
            if os.path.isdir(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)
        finally:
            # Every rank participates even after a local failure so peers cannot
            # deadlock in the PP consensus operation.
            all_saves_succeeded = _save_barrier(save_error is None)

        if save_error is not None:
            if os.getenv("VLLM_WEIGHT_CACHE_SAVE_FAIL_HARD", "0") == "1":
                raise SystemExit(
                    "transactional weight-cache save failed"
                ) from save_error
            raise save_error

        if not all_saves_succeeded:
            peer_error = RuntimeError(
                "a peer PP rank failed its transactional cache save"
            )
            if os.getenv("VLLM_WEIGHT_CACHE_SAVE_FAIL_HARD", "0") == "1":
                raise SystemExit(
                    "transactional weight-cache save failed"
                ) from peer_error
            raise peer_error

        if os.getenv("VLLM_WEIGHT_CACHE_BUILD_ONLY", "0") == "1":
            _write_build_sentinel(cache_dir)
            logger.info(
                "transactional weight-cache build complete; exiting before "
                "post-load profiling and KV-cache allocation"
            )
            raise SystemExit("transactional weight-cache build complete")

    return save_transactionally
