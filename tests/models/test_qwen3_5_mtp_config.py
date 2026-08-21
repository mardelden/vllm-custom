# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for Qwen3.5 text-only MTP speculative decoding."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models import qwen3_5_mtp
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


def _mtp_config(model_type: str) -> PretrainedConfig:
    return PretrainedConfig(
        model_type=model_type,
        architectures=["SomeArch"],
        mtp_num_hidden_layers=1,
    )


@pytest.mark.parametrize(
    "model_type,expected_arch",
    [
        ("qwen3_5", "Qwen3_5MTP"),
        ("qwen3_5_moe", "Qwen3_5MoeMTP"),
        # Text-only config variants must map to the same MTP architectures.
        ("qwen3_5_text", "Qwen3_5MTP"),
        ("qwen3_5_moe_text", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_recognizes_text_only_types(model_type, expected_arch):
    cfg = SpeculativeConfig.hf_config_override(_mtp_config(model_type))
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.architectures == [expected_arch]
    assert cfg.n_predict == 1


def test_fastsafetensors_filters_shared_qwen_mtp_weights(monkeypatch):
    monkeypatch.setattr(
        qwen3_5_mtp,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )

    assert Qwen3_5ForConditionalGeneration.fastsafetensors_weight_filter(
        "model.language_model.layers.0.weight"
    )
    assert not Qwen3_5ForConditionalGeneration.fastsafetensors_weight_filter(
        "mtp.layers.0.weight"
    )
    assert Qwen3_5MTP.fastsafetensors_weight_filter(None, "mtp.layers.0.weight")
    assert not Qwen3_5MTP.fastsafetensors_weight_filter(
        None, "model.language_model.embed_tokens.weight"
    )
    assert not Qwen3_5MTP.fastsafetensors_weight_filter(None, "lm_head.weight")


def test_qwen_mtp_tracks_weights_shared_after_loading(monkeypatch):
    monkeypatch.setattr(
        qwen3_5_mtp,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )

    class FakeWeightsLoader:
        def __init__(self, model):
            pass

        def load_weights(self, weights):
            return {name for name, _ in weights}

    monkeypatch.setattr(qwen3_5_mtp, "AutoWeightsLoader", FakeWeightsLoader)
    model = SimpleNamespace(
        named_parameters=lambda: iter(
            [
                ("model.embed_tokens.weight", None),
                ("lm_head.weight", None),
                ("model.layers.0.weight", None),
            ]
        )
    )

    loaded = Qwen3_5MTP.load_weights(
        model,
        [("mtp.layers.0.weight", object())],
    )

    assert loaded == {
        "model.layers.0.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    }


@pytest.mark.parametrize("method,expected_calls", [("mtp", 1), ("draft_model", 0)])
def test_draft_model_shares_only_mtp_target_weights(method, expected_calls):
    proposer = object.__new__(DraftModelProposer)
    proposer.speculative_config = SimpleNamespace(method=method)
    target = MagicMock()

    with (
        patch.object(
            SpecDecodeBaseProposer, "_maybe_share_embeddings"
        ) as share_embeddings,
        patch.object(SpecDecodeBaseProposer, "_maybe_share_lm_head") as share_lm_head,
    ):
        proposer._maybe_share_embeddings(target)
        proposer._maybe_share_lm_head(target)

    assert share_embeddings.call_count == expected_calls
    assert share_lm_head.call_count == expected_calls
