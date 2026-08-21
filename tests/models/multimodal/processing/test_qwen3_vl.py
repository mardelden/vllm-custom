# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Qwen3-VL processor.

Covers the fix for num_frames-based timestamp calculation
(issue vllm-project/vllm#35909).
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder
from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def test_warmup_inputs_avoid_maximum_media_sizing() -> None:
    info = MagicMock()
    info.ctx.tokenizer = None
    info.parse_mm_data.side_effect = lambda data, validate: data
    builder = Qwen3VLDummyInputsBuilder(info)

    inputs = builder.get_warmup_processor_inputs(
        seq_len=262144,
        mm_counts={"image": 1, "video": 1},
        mm_options={},
    )

    assert inputs.mm_data_items["image"][0].size == (1, 1)
    video, _ = inputs.mm_data_items["video"][0]
    assert video.shape == (2, 1, 1, 3)
    info.get_image_size_with_most_features.assert_not_called()
    info.get_video_processor.assert_not_called()


def _build_video_mm_data(
    num_frames: int,
    width: int = 128,
    height: int = 128,
    original_fps: float = 30.0,
) -> dict[str, Any]:
    """Create synthetic video data with metadata indicating that
    HF processor should re-sample frames (do_sample_frames=True).

    ``total_num_frames`` is set equal to the ndarray frame count so
    that HF's ``sample_frames`` indices stay within bounds of the
    actual tensor that is passed."""
    video = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    metadata = {
        "fps": original_fps,
        "duration": num_frames / original_fps,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": True,
    }
    return {"video": [(video, metadata)]}


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize(
    "num_frames",
    [8, 16],
)
def test_processor_num_frames_timestamp(
    model_id: str,
    num_frames: int,
) -> None:
    """Regression test: using ``num_frames`` (without ``fps``) must not
    cause a timestamp / token-count mismatch.

    Before the fix, ``_get_video_second_idx`` ignored the explicit
    ``num_frames`` and fell back to an fps-based calculation, which
    produced a different number of timestamp entries and ultimately led
    to shape mismatches in downstream token construction.

    We deliberately choose ``num_frames`` values (8, 16) that differ
    from what the default fps-based path would compute (which clamps
    to ``min_frames=4`` for a short video at 30 fps), so this test
    would fail without the fix.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    mm_data = _build_video_mm_data(num_frames=num_frames)

    # Process with explicit num_frames (no fps) -- this is the path
    # that was broken before the fix.
    hf_mm_kwargs: dict[str, Any] = {"num_frames": num_frames}
    processed = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_mm_kwargs,
    )

    # Basic sanity: the processor must produce video tokens.
    token_ids = processed["prompt_token_ids"]
    assert len(token_ids) > 0, "Processor produced empty token list"

    # Verify that video placeholders were actually inserted.
    assert "mm_placeholders" in processed
    video_phs = processed["mm_placeholders"].get("video", [])
    assert len(video_phs) == 1, (
        f"Expected exactly 1 video placeholder, got {len(video_phs)}"
    )


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize("num_videos", [2, 4])
def test_processor_multi_video(
    model_id: str,
    num_videos: int,
) -> None:
    """Verify that multi-video processing produces correct placeholders.

    This exercises the token-level replacement path in
    ``_call_hf_processor`` which avoids the quadratic text-level
    prompt expansion.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": num_videos},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<|vision_start|><|video_pad|><|vision_end|>" * num_videos
    mm_data = {"video": [_build_video_mm_data(num_frames=8)["video"][0]] * num_videos}

    processed = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={"num_frames": 8},
    )

    token_ids = processed["prompt_token_ids"]
    assert len(token_ids) > 0

    video_phs = processed["mm_placeholders"].get("video", [])
    assert len(video_phs) == num_videos, (
        f"Expected {num_videos} video placeholders, got {len(video_phs)}"
    )

    # All placeholders should have the same length (same video params)
    # and must not overlap.
    lengths = {ph.length for ph in video_phs}
    assert len(lengths) == 1, f"Placeholder lengths differ: {lengths}"
    for i in range(1, len(video_phs)):
        prev_end = video_phs[i - 1].offset + video_phs[i - 1].length
        assert video_phs[i].offset >= prev_end, (
            f"Placeholder {i} overlaps with placeholder {i - 1}"
        )


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize(
    "hf_mm_kwargs",
    [{"num_frames": [8, 16]}, {"fps": [2.0, 4.0]}],
)
def test_processor_multi_video_list_kwargs(
    model_id: str,
    hf_mm_kwargs: dict[str, Any],
) -> None:
    """Regression test: a multi-video request with list-valued per-video
    ``mm_processor_kwargs`` (one ``fps``/``num_frames`` per video) must not
    crash.

    Before the fix, ``_call_hf_processor`` copied the whole kwargs to every
    video without slicing, so ``_get_video_second_idx`` received the list
    where a scalar was expected and raised ``TypeError``.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": 2},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = (
        "<|vision_start|><|video_pad|><|vision_end|>"
        "<|vision_start|><|video_pad|><|vision_end|>"
    )
    mm_data = {
        "video": [
            _build_video_mm_data(num_frames=16)["video"][0],
            _build_video_mm_data(num_frames=32)["video"][0],
        ]
    }

    processed = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_mm_kwargs,
    )

    video_phs = processed["mm_placeholders"].get("video", [])
    assert len(video_phs) == 2, (
        f"Expected exactly 2 video placeholders, got {len(video_phs)}"
    )
