# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
from pathlib import Path
from unittest import mock

import pytest

MODULE_PATH = (
    Path(__file__).parents[3]
    / "vllm/model_executor/model_loader/_dgx_weight_cache_lifecycle.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_dgx_weight_cache_lifecycle", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
cache_lifecycle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache_lifecycle)


def _publish(_model: object, cache_dir: str) -> None:
    target = Path(cache_dir)
    target.mkdir(parents=True)
    (target / "shard.safetensors").write_bytes(b"weights")
    (target / "manifest.json").write_text(
        json.dumps({"complete": True, "files": ["shard.safetensors"]}),
        encoding="utf-8",
    )


def test_transactional_publish_preserves_incomplete_generation(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache-key"
    cache_dir.mkdir()
    (cache_dir / "old-partial").write_bytes(b"partial")

    cache_lifecycle.wrap_cache_save(_publish)(object(), str(cache_dir))

    assert cache_lifecycle.cache_is_complete(str(cache_dir))
    abandoned = list(cache_dir.parent.glob("cache-key.abandoned-*"))
    assert len(abandoned) == 1
    assert (abandoned[0] / "old-partial").is_file()
    assert list(cache_dir.parent.glob("cache-key.building-*")) == []


def test_complete_generation_is_never_overwritten(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache-key"
    _publish(object(), str(cache_dir))
    original = (cache_dir / "shard.safetensors").read_bytes()
    save = mock.Mock(side_effect=AssertionError("save should not run"))

    cache_lifecycle.wrap_cache_save(save)(object(), str(cache_dir))

    save.assert_not_called()
    assert (cache_dir / "shard.safetensors").read_bytes() == original


def test_failed_staging_save_does_not_touch_existing_directory(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache-key"
    cache_dir.mkdir()
    (cache_dir / "old-partial").write_bytes(b"partial")

    def fail(_model: object, staging_dir: str) -> None:
        Path(staging_dir).mkdir(parents=True)
        raise RuntimeError("injected save failure")

    with pytest.raises(RuntimeError, match="injected save failure"):
        cache_lifecycle.wrap_cache_save(fail)(object(), str(cache_dir))

    assert (cache_dir / "old-partial").is_file()
    assert list(cache_dir.parent.glob("cache-key.building-*")) == []


def test_build_only_publishes_atomic_sentinel_then_exits(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache-key"
    sentinel = tmp_path / "sentinel.json"
    environment = {
        "VLLM_WEIGHT_CACHE_BUILD_ONLY": "1",
        "VLLM_WEIGHT_CACHE_BUILD_SENTINEL": str(sentinel),
    }
    with (
        mock.patch.dict("os.environ", environment, clear=False),
        pytest.raises(SystemExit, match="transactional weight-cache build complete"),
    ):
        cache_lifecycle.wrap_cache_save(_publish)(object(), str(cache_dir))

    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    assert Path(payload["cache_dir"]) == cache_dir.resolve()
    assert cache_lifecycle.cache_is_complete(str(cache_dir))


def test_peer_failure_prevents_success(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache-key"
    with (
        mock.patch.object(cache_lifecycle, "_save_barrier", return_value=False),
        pytest.raises(RuntimeError, match="peer PP rank failed"),
    ):
        cache_lifecycle.wrap_cache_save(_publish)(object(), str(cache_dir))
