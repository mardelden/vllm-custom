# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the operator-declared reasoning-effort gate.

Chat templates fold unsupported effort values silently (GLM-5.3 upgrades
them to Max, Qwen3.5's template ignores them). With a declared vocabulary
the gate must reject such values before rendering so a client is never
misled into believing the setting was honored; without one it must be a
strict no-op (stock passthrough).
"""

import pytest

from vllm.entrypoints.reasoning_effort import (
    VLLM_REASONING_EFFORT_ACCEPTED,
    _accepted_efforts,
    validate_reasoning_effort,
)
from vllm.exceptions import VLLMValidationError


@pytest.fixture(autouse=True)
def _fresh_vocabulary(monkeypatch):
    monkeypatch.delenv(VLLM_REASONING_EFFORT_ACCEPTED, raising=False)
    _accepted_efforts.cache_clear()
    yield
    _accepted_efforts.cache_clear()


def _declare(monkeypatch, levels: str) -> None:
    monkeypatch.setenv(VLLM_REASONING_EFFORT_ACCEPTED, levels)
    _accepted_efforts.cache_clear()


def test_unset_vocabulary_is_stock_passthrough():
    validate_reasoning_effort({"reasoning_effort": "banana"})


def test_declared_level_passes(monkeypatch):
    _declare(monkeypatch, "low,high,max")
    validate_reasoning_effort({"reasoning_effort": "high"})


def test_undeclared_level_rejected_naming_vocabulary(monkeypatch):
    # Declared order is preserved in the message (the convention makes list
    # order cosmetic-but-visible); values are never remapped or aliased.
    _declare(monkeypatch, "max,low,high")
    with pytest.raises(VLLMValidationError, match="max, low, high") as exc_info:
        validate_reasoning_effort({"reasoning_effort": "minimal"})
    assert exc_info.value.parameter == "reasoning_effort"
    assert exc_info.value.value == "minimal"


def test_none_is_part_of_the_vocabulary(monkeypatch):
    # A model that cannot not-think omits "none"; asking to disable thinking
    # must fail loudly, not be silently ignored while the model thinks anyway.
    _declare(monkeypatch, "low,high,max")
    with pytest.raises(VLLMValidationError):
        validate_reasoning_effort({"reasoning_effort": "none"})
    _declare(monkeypatch, "none,low,medium,xhigh")
    validate_reasoning_effort({"reasoning_effort": "none"})


def test_no_effort_requested_passes(monkeypatch):
    _declare(monkeypatch, "low,high")
    validate_reasoning_effort({})
    validate_reasoning_effort(None)
    validate_reasoning_effort({"enable_thinking": True})


def test_vocabulary_parsing_tolerates_whitespace(monkeypatch):
    _declare(monkeypatch, " low , high ,max,low,")
    assert _accepted_efforts() == ("low", "high", "max")


def test_empty_setting_means_passthrough(monkeypatch):
    _declare(monkeypatch, "  ")
    validate_reasoning_effort({"reasoning_effort": "banana"})
