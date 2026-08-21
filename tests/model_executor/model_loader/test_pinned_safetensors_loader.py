# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.weight_utils import (
    _pinned_safetensors_file_iterator,
    _PinnedReadGroup,
    _plan_pinned_read_groups,
    _SafetensorsHeaderEntry,
)


@pytest.mark.parametrize("prefetch", [False, True])
def test_pinned_safetensors_iterator_round_trip(tmp_path, prefetch):
    expected = {
        "a.uint8": torch.arange(7, dtype=torch.uint8),
        "b.float32": torch.arange(12, dtype=torch.float32).view(3, 4),
        "c.bfloat16": torch.arange(11, dtype=torch.bfloat16),
        # Deliberately larger than the reusable buffer to exercise the
        # bounded loader's one-tensor overflow path.
        "d.oversized": torch.arange(80, dtype=torch.int16),
    }
    path = tmp_path / "model.safetensors"
    save_file(expected, path)

    num_buffers = 3 if prefetch else 2
    loaded = {
        name: tensor.clone()
        for name, tensor in _pinned_safetensors_file_iterator(
            str(path),
            local_expert_ids=None,
            num_threads=2,
            chunk_size=13,
            buffer_size=64,
            gap_size=0,
            prefetch=prefetch,
            stage_buffers=[None] * num_buffers,
            stage_buffer_cursor=[0],
            pin_memory=False,
        )
    }

    assert loaded.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(loaded[name], tensor)


def test_plan_pinned_read_groups_respects_buffer_and_gap():
    entries = [
        _SafetensorsHeaderEntry("a", torch.uint8, (8,), 0, 8),
        _SafetensorsHeaderEntry("b", torch.uint8, (8,), 8, 16),
        _SafetensorsHeaderEntry("c", torch.uint8, (8,), 24, 32),
        _SafetensorsHeaderEntry("large", torch.uint8, (40,), 32, 72),
    ]

    assert _plan_pinned_read_groups(entries, buffer_size=24, gap_size=4) == [
        _PinnedReadGroup(0, 2, 0, 16),
        _PinnedReadGroup(2, 3, 24, 32),
        _PinnedReadGroup(3, 4, 32, 72),
    ]


@pytest.mark.parametrize(
    "field",
    [
        "safetensors_pinned_num_threads",
        "safetensors_pinned_chunk_size",
        "safetensors_pinned_buffer_size",
    ],
)
def test_pinned_safetensors_config_rejects_non_positive_values(field):
    with pytest.raises(ValueError):
        LoadConfig(**{field: 0})
