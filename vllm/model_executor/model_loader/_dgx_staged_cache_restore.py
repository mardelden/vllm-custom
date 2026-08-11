# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked fastsafetensors restore through one CUDA staging buffer per shard."""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch
from fastsafetensors import cpp as fstcpp
from fastsafetensors.copier.nogds import load_library_func
from fastsafetensors.frameworks import get_framework_op
from safetensors.torch import _TYPES as SAFETENSORS_DTYPES
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_CHUNK_MB = 64
_DEFAULT_BOUNCE_KB = 16 * 1024
_DEFAULT_THREADS = 16


def _positive_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive, found {value}")
    return value


def _nonnegative_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative, found {value}")
    return value


def _reclaim_and_close(fd: int) -> None:
    try:
        posix_fadvise = getattr(os, "posix_fadvise", None)
        dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
        if posix_fadvise is not None and dontneed is not None:
            posix_fadvise(fd, 0, 0, dontneed)
    finally:
        os.close(fd)


def _read_header(path: str) -> tuple[int, dict[str, Any]]:
    with open(path, "rb") as cache_file:
        raw_size = cache_file.read(8)
        if len(raw_size) != 8:
            raise RuntimeError(f"invalid safetensors header in {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size > os.path.getsize(path) - 8:
            raise RuntimeError(f"invalid safetensors header size in {path}")
        header = json.loads(cache_file.read(header_size))
    return header_size, header


def _device(targets: dict[str, torch.Tensor]) -> torch.device:
    devices = {tensor.device for tensor in targets.values()}
    if len(devices) != 1:
        raise RuntimeError(
            "staged weight-cache restore requires one target device; "
            f"found {sorted(map(str, devices))}"
        )
    device = next(iter(devices))
    if device.type != "cuda":
        raise RuntimeError(
            f"staged weight-cache restore requires CUDA targets, found {device}"
        )
    return device


def _restore_impl(model: nn.Module, cache_dir: str) -> None:
    with open(os.path.join(cache_dir, "manifest.json"), encoding="utf-8") as handle:
        manifest: dict[str, Any] = json.load(handle)

    targets = model.state_dict()
    device = _device(targets)
    device_index = device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()

    loaded: set[str] = set()
    seen_storage: dict[tuple[int, int], str] = {}
    unique_ranges: list[tuple[int, int, str]] = []
    file_entries: list[
        tuple[str, int, int, list[tuple[str, torch.Tensor, int, int, bool]]]
    ] = []

    # Validate the entire generation before changing any parameter.
    for filename in manifest["files"]:
        path = os.path.join(cache_dir, filename)
        file_size = os.path.getsize(path)
        header_size, header = _read_header(path)
        data_start = 8 + header_size
        data_bytes = 0
        entries: list[tuple[str, torch.Tensor, int, int, bool]] = []
        for name, tensor_meta in header.items():
            if name == "__metadata__":
                continue
            if name in loaded:
                raise RuntimeError(f"duplicate weight-cache key '{name}'")
            target = targets.get(name)
            if target is None:
                raise RuntimeError(f"unexpected weight-cache key '{name}'")
            start, end = tensor_meta["data_offsets"]
            nbytes = end - start
            data_bytes = max(data_bytes, end)
            if start < 0 or end < start or data_start + end > file_size:
                raise RuntimeError(f"invalid weight-cache offsets for '{name}'")
            if list(target.shape) != tensor_meta["shape"]:
                raise RuntimeError(
                    f"weight-cache shape mismatch for '{name}': "
                    f"cache={tensor_meta['shape']}, target={list(target.shape)}"
                )
            cache_dtype = SAFETENSORS_DTYPES.get(tensor_meta["dtype"])
            if cache_dtype is None or target.dtype != cache_dtype:
                raise RuntimeError(
                    f"weight-cache dtype mismatch for '{name}': "
                    f"cache={tensor_meta['dtype']}, target={target.dtype}"
                )
            if target.nbytes != nbytes or not target.is_contiguous():
                raise RuntimeError(
                    f"weight-cache storage mismatch for '{name}': "
                    f"cache_bytes={nbytes}, target_bytes={target.nbytes}, "
                    f"contiguous={target.is_contiguous()}"
                )
            if target.device != device:
                raise RuntimeError(
                    f"weight-cache device mismatch for '{name}': "
                    f"expected={device}, target={target.device}"
                )

            storage = (target.data_ptr(), nbytes)
            alias = storage in seen_storage
            if not alias and nbytes:
                seen_storage[storage] = name
                unique_ranges.append((storage[0], storage[0] + nbytes, name))
            entries.append((name, target, start, nbytes, alias))
            loaded.add(name)
        file_entries.append((path, data_start, data_bytes, entries))

    missing = set(targets) - loaded
    if missing:
        raise RuntimeError(
            f"weight-cache incomplete: {len(missing)} params not restored "
            f"(e.g. {list(missing)[:3]})"
        )
    unique_ranges.sort()
    for previous, current in zip(unique_ranges, unique_ranges[1:]):
        if current[0] < previous[1]:
            raise RuntimeError(
                "weight-cache targets overlap unexpectedly: "
                f"'{previous[2]}' and '{current[2]}'"
            )

    framework = get_framework_op("pytorch")
    load_library_func(framework)
    threads = _positive_env("VLLM_WEIGHT_CACHE_THREADS", _DEFAULT_THREADS)
    chunk_bytes = (
        _positive_env("VLLM_WEIGHT_CACHE_CHUNK_MB", _DEFAULT_CHUNK_MB) * 1024 * 1024
    )
    bounce_kb = _positive_env("VLLM_WEIGHT_CACHE_BBUF_KB", _DEFAULT_BOUNCE_KB)
    reclaim_window = _nonnegative_env("VLLM_WEIGHT_CACHE_RECLAIM_WINDOW_SHARDS", 0)
    reader = fstcpp.nogds_file_reader(False, bounce_kb, threads, True, device_index)
    logger.info(
        "DGX staged cache transport enabled "
        "(threads=%d chunk_bytes=%d bounce_kb=%d reclaim_window_shards=%d)",
        threads,
        chunk_bytes,
        bounce_kb,
        reclaim_window,
    )

    started = time.perf_counter()
    copied_bytes = 0
    executor = ThreadPoolExecutor(max_workers=1) if reclaim_window >= 2 else None
    pending_reclaims: deque[Future[None]] = deque()
    try:
        for index, (path, data_start, data_bytes, entries) in enumerate(
            file_entries, start=1
        ):
            staging = torch.empty(data_bytes, dtype=torch.uint8, device=device)
            buffer = fstcpp.gds_device_buffer(staging.data_ptr(), data_bytes, True)
            fd = os.open(path, os.O_RDONLY)
            requests: list[int] = []
            submit_error: Exception | None = None
            try:
                for chunk_start in range(0, data_bytes, chunk_bytes):
                    chunk_size = min(chunk_bytes, data_bytes - chunk_start)
                    request = reader.submit_read(
                        fd,
                        buffer,
                        data_start + chunk_start,
                        chunk_size,
                        chunk_start,
                    )
                    if request < 0:
                        raise RuntimeError(
                            f"weight-cache submit_read failed for '{path}': {request}"
                        )
                    requests.append(request)
            except Exception as error:
                submit_error = error

            failed = [request for request in requests if reader.wait_read(request) == 0]
            if submit_error is not None or failed:
                os.close(fd)
                if submit_error is not None:
                    raise submit_error
                raise RuntimeError(
                    f"weight-cache direct read failed for {len(failed)} requests "
                    f"(e.g. {failed[:3]})"
                )

            # The shard is resident in CUDA staging. Choose no reclaim, strict
            # synchronous reclaim, or a bounded asynchronous reclaim window.
            if reclaim_window == 0:
                os.close(fd)
            elif reclaim_window == 1:
                _reclaim_and_close(fd)
            else:
                assert executor is not None
                pending_reclaims.append(executor.submit(_reclaim_and_close, fd))
                if len(pending_reclaims) >= reclaim_window:
                    pending_reclaims.popleft().result()

            with torch.no_grad():
                for _name, target, cache_offset, nbytes, alias in entries:
                    if alias or not nbytes:
                        continue
                    source = (
                        staging[cache_offset : cache_offset + nbytes]
                        .view(target.dtype)
                        .reshape(target.shape)
                    )
                    target.copy_(source, non_blocking=True)
            torch.accelerator.synchronize()
            copied_bytes += data_bytes
            del staging, buffer
            torch.accelerator.empty_cache()

            if index == 1 or index % 8 == 0 or index == len(file_entries):
                elapsed = time.perf_counter() - started
                logger.info(
                    "staged weight-cache restore: %d/%d shards, %.1f GiB in %.1f s "
                    "(%.2f GiB/s)",
                    index,
                    len(file_entries),
                    copied_bytes / 2**30,
                    elapsed,
                    copied_bytes / 2**30 / elapsed,
                )
        while pending_reclaims:
            pending_reclaims.popleft().result()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    torch.accelerator.synchronize()


def restore(model: nn.Module, cache_dir: str) -> None:
    try:
        _restore_impl(model, cache_dir)
    except BaseException as error:
        logger.exception("staged weight-cache restore failed")
        print(
            f"STAGED_WEIGHT_CACHE_FAILURE: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        if os.getenv("VLLM_WEIGHT_CACHE_FAIL_HARD", "0") == "1":
            raise SystemExit("staged weight-cache restore failed") from error
        raise
