"""Render a list of chat messages into a single prompt string.

A ChatML-like layout with literal role markers, a blank line between turns, and
a trailing generation-prompt suffix that cues the assistant turn. The markers
are plain text (they are not registered as tokenizer special tokens), so a
model only follows the template if it was trained or fine-tuned on it; a
base/pretrained checkpoint simply continues the text. Truncation drops the
oldest non-system turns to fit a token budget while always keeping the system
message and reserving room for the reply.

This module is import-light (only ``re``) and takes the tokenizer as an
argument, so it lives happily in the control plane.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tokenizer import InferenceTokenizer

DEFAULT_SYSTEM = "You are a helpful assistant who answers clearly and concisely."
ROLE_MARKERS: dict[str, str] = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
GENERATION_SUFFIX = f"\n\n{ROLE_MARKERS['assistant']}\n"
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str) -> str:
    """Minimal normalization: strip control chars, unify newlines, rstrip lines."""
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def ensure_system(
    messages: list[dict[str, str]], system_fallback: str = DEFAULT_SYSTEM
) -> list[dict[str, str]]:
    """Guarantee exactly one leading system message, prepending a default if none."""
    if any(m.get("role") == "system" for m in messages):
        return messages
    return [{"role": "system", "content": system_fallback}, *messages]


def build_prompt(messages: list[dict[str, str]], *, add_generation_prompt: bool = True) -> str:
    """Render messages to a single prompt string. Empty/whitespace turns drop out."""
    segments: list[str] = []
    for m in messages:
        role = m.get("role") or "user"
        marker = ROLE_MARKERS.get(role, ROLE_MARKERS["user"])
        norm = normalize_text(m.get("content") or "")
        if not norm:
            continue
        segments.append(f"{marker}\n{norm}")
    prompt = "\n\n".join(segments)
    if add_generation_prompt:
        prompt += GENERATION_SUFFIX
    return prompt


def render_prompt_ids(
    messages: list[dict[str, str]],
    tokenizer: InferenceTokenizer,
    *,
    max_prompt_tokens: int,
    system_fallback: str = DEFAULT_SYSTEM,
) -> list[int]:
    """Render + tokenize messages, truncating to ``max_prompt_tokens`` by dropping
    the oldest non-system turns. As a last resort, tail-truncate the token ids."""
    messages = ensure_system(messages, system_fallback)
    ids = tokenizer.encode(build_prompt(messages), add_special_tokens=False)
    if len(ids) <= max_prompt_tokens:
        return ids

    system = [m for m in messages if m.get("role") == "system"]
    others = [m for m in messages if m.get("role") != "system"]
    for drop in range(len(others)):
        candidate = [*system, *others[drop:]]
        ids = tokenizer.encode(build_prompt(candidate), add_special_tokens=False)
        if len(ids) <= max_prompt_tokens:
            return ids

    # Even the last turn alone overflows — hard tail-truncate the ids.
    minimal = [*system[-1:], *others[-1:]]
    ids = tokenizer.encode(build_prompt(minimal), add_special_tokens=False)
    return ids[-max_prompt_tokens:]
