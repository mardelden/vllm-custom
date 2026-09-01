# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Honest reasoning-effort handling at the serving boundary.

Clients express reasoning depth as an effort level (the ``reasoning_effort``
request field, Responses-style ``reasoning.effort``, Anthropic
``output_config.effort``, or a raw ``chat_template_kwargs`` entry), but what
interprets the value is the chat template shipped inside the model artifact —
and templates commonly fold unknown levels instead of erroring. GLM-5.3's
template upgrades anything outside its vocabulary to Max; Qwen3.5's ignores
the value wholesale. At the call site, a silently remapped setting is
indistinguishable from an honored one.

The serving layer must either honor an effort level verbatim or reject the
request naming the levels this deployment actually distinguishes — never
remap, never let the template's fallback decide. The accepted vocabulary is
declared by the operator next to the model pin, not detected from the model
family: folding behavior is a property of the exact shipped artifact
(checkpoint copies of one family ship different templates), so detection is
confidently wrong exactly when it matters.

Set ``VLLM_REASONING_EFFORT_ACCEPTED`` to a comma-separated list of levels
the deployed artifact's template genuinely expresses (include ``none`` only
if the model can actually not-think). Unset means stock passthrough, so the
code ships safely before a deployment's vocabulary is filled in. Levels are
opaque, case-sensitive strings matched exactly — never remapped, clamped, or
aliased — and their declared order is preserved in error messages. This is
the vLLM rendering of the fleet's per-profile ``reasoning_efforts`` key.

Validation runs at the single point every client surface converges — the
merged ``chat_template_kwargs`` just before template rendering — so one check
covers every adapter, including the server's own
``--default-chat-template-kwargs``.
"""

import os
from functools import lru_cache
from typing import Any

from vllm.exceptions import VLLMValidationError

VLLM_REASONING_EFFORT_ACCEPTED = "VLLM_REASONING_EFFORT_ACCEPTED"


@lru_cache
def _accepted_efforts() -> tuple[str, ...] | None:
    setting = os.environ.get(VLLM_REASONING_EFFORT_ACCEPTED)
    if setting is None or not setting.strip():
        return None
    levels: list[str] = []
    for level in setting.split(","):
        level = level.strip()
        if level and level not in levels:
            levels.append(level)
    return tuple(levels)


def validate_reasoning_effort(chat_template_kwargs: dict[str, Any] | None) -> None:
    """Reject an effort level outside the operator-declared vocabulary.

    Raises:
        VLLMValidationError: if ``chat_template_kwargs["reasoning_effort"]``
            is set and not in the accepted set. No-op when the deployment
            declares no vocabulary.
    """
    accepted = _accepted_efforts()
    if accepted is None or not chat_template_kwargs:
        return
    effort = chat_template_kwargs.get("reasoning_effort")
    if effort is None or effort in accepted:
        return
    raise VLLMValidationError(
        f"reasoning_effort={effort!r} is not a level this deployment "
        f"distinguishes; the model's chat template would silently fold or "
        f"ignore it. Accepted values: {', '.join(accepted)}.",
        parameter="reasoning_effort",
        value=effort,
    )
