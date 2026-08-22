# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

from vllm.entrypoints.openai.api_server_bootstrap import _start_engine_forkserver


def test_bootstrap_preloads_async_llm_before_starting_forkserver():
    with (
        patch("multiprocessing.set_start_method") as set_start_method,
        patch("multiprocessing.set_forkserver_preload") as set_preload,
        patch("multiprocessing.forkserver.ensure_running") as ensure_running,
    ):
        _start_engine_forkserver()

    set_start_method.assert_called_once_with("forkserver")
    set_preload.assert_called_once_with(["vllm.v1.engine.async_llm"])
    ensure_running.assert_called_once_with()
