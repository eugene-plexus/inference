"""Unit tests for chat-template rendering + token-budget truncation.

These exercise the only code that enforces the model's context window for chat
input — the drop-oldest-non-system loop and the last-resort tail-truncate — so a
regression that drops the system turn or overflows the budget is caught.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eugene_plexus_inference.engine.chat_template import (
    DEFAULT_SYSTEM,
    build_prompt,
    ensure_system,
    render_prompt_ids,
)

if TYPE_CHECKING:
    from eugene_plexus_inference.engine.tokenizer import InferenceTokenizer


def test_ensure_system_prepends_default_when_absent() -> None:
    out = ensure_system([{"role": "user", "content": "hi"}])
    assert out[0] == {"role": "system", "content": DEFAULT_SYSTEM}
    assert out[1]["role"] == "user"


def test_ensure_system_noop_when_present() -> None:
    msgs = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]
    assert ensure_system(msgs) is msgs


def test_build_prompt_appends_generation_suffix() -> None:
    prompt = build_prompt([{"role": "user", "content": "hi"}])
    assert "<|user|>" in prompt
    assert prompt.endswith("<|assistant|>\n")


def test_render_fits_budget_and_keeps_system(text_tokenizer: InferenceTokenizer) -> None:
    # A long multi-turn conversation forces the drop-oldest loop.
    messages = [{"role": "system", "content": "you are a careful assistant"}]
    for i in range(20):
        messages.append({"role": "user", "content": f"question number {i} about something"})
        messages.append({"role": "assistant", "content": f"answer number {i} with detail"})
    # Budget fits the system turn + a few recent turns (forcing the drop-oldest
    # path) but is far below the full ~20-turn conversation (~1.6k tokens with
    # this tiny tokenizer, whose unmerged role markers are byte-fallback-heavy).
    budget = 150
    ids = render_prompt_ids(messages, text_tokenizer, max_prompt_tokens=budget)
    assert len(ids) <= budget
    # The system turn must survive drop-oldest truncation.
    assert "<|system|>" in text_tokenizer.decode(ids)


def test_render_tail_truncates_oversized_single_turn(text_tokenizer: InferenceTokenizer) -> None:
    # A single turn larger than the budget forces the last-resort tail-truncate.
    huge = " ".join(f"token{i}" for i in range(500))
    ids = render_prompt_ids(
        [{"role": "user", "content": huge}], text_tokenizer, max_prompt_tokens=16
    )
    assert 0 < len(ids) <= 16
