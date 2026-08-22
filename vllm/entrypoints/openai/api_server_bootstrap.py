# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def _start_engine_forkserver() -> None:
    import multiprocessing
    import multiprocessing.forkserver as forkserver

    # Start before importing the API server's large parent-process graph.
    multiprocessing.set_start_method("forkserver")
    multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
    forkserver.ensure_running()


def main() -> None:
    """Start the API server after launching the EngineCore forkserver."""
    import os

    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        _start_engine_forkserver()

    from vllm.entrypoints.launchers.api_server.entry import main as api_main

    api_main()


if __name__ == "__main__":
    main()
