# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path

import vllm.utils.system_utils as system_utils
from vllm.utils.system_utils import _maybe_force_spawn, unique_filepath


def test_unique_filepath():
    temp_dir = tempfile.mkdtemp()
    path_fn = lambda i: Path(temp_dir) / f"file_{i}.txt"
    paths = set()
    for i in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


def test_numa_bind_forces_spawn(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "--numa-bind"])
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_initialized_cuda_does_not_override_early_forkserver(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "forkserver")
    monkeypatch.setattr("sys.argv", ["vllm", "serve"])
    monkeypatch.setattr(system_utils, "is_in_ray_actor", lambda: False)
    monkeypatch.setattr(system_utils, "cuda_is_initialized", lambda: True)
    monkeypatch.setattr(system_utils, "in_wsl", lambda: False)

    _maybe_force_spawn()

    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "forkserver"
