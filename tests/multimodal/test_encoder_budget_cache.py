# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from types import SimpleNamespace

from vllm.multimodal import encoder_budget


def test_mm_max_tokens_cache_round_trip_and_validation(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(encoder_budget.envs, "VLLM_CACHE_ROOT", str(tmp_path))
    config = SimpleNamespace(compute_hash=lambda: "config-hash")
    mm_counts = {"image": 1}
    fingerprint = encoder_budget._mm_max_tokens_cache_fingerprint(config, mm_counts)

    encoder_budget._save_mm_max_tokens_cache(fingerprint, mm_counts, {"image": 4096})
    assert encoder_budget._load_mm_max_tokens_cache(fingerprint, mm_counts) == {
        "image": 4096
    }
    assert encoder_budget._load_mm_max_tokens_cache(fingerprint, {"video": 1}) is None

    cache_path = Path(encoder_budget._mm_max_tokens_cache_path(fingerprint))
    cache_path.write_text("not-json")
    assert encoder_budget._load_mm_max_tokens_cache(fingerprint, mm_counts) is None
