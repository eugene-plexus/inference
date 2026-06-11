"""Autoregressive decode loops over a loaded model.

No KV cache in v0.3: each step re-feeds the running sequence cropped to the
model's context window (``block_size``), so decode is O(n^2) in the generated
length — fine for the small local models this first cut targets, and a clean
place to add a cache later without changing these signatures. The streaming
variant decodes the whole id list each step and emits the new text *suffix*,
which is robust to byte-level BPE tokens that span multiple UTF-8 bytes.

torch is imported lazily inside the loops so this module imports without it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .sampling import SamplingParams, sample_next_token

if TYPE_CHECKING:
    import torch
    from torch import nn

    from .tokenizer import InferenceTokenizer


@dataclass
class GenerateResult:
    token_ids: list[int]
    finish_reason: str  # "stop" (EOS produced) | "length" (hit max_tokens)


def _next_logits(model: nn.Module, idx: torch.Tensor, block_size: int) -> torch.Tensor:
    """Forward the (cropped) sequence and return the last position's 1-D logits."""
    idx_cond = idx[:, -block_size:]
    logits, _ = model(idx_cond)
    return logits[0, -1, :]


def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    params: SamplingParams,
    eos_id: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> GenerateResult:
    """Decode up to ``params.max_tokens`` tokens from ``input_ids`` (shape [1, T])."""
    import torch

    generated: list[int] = []
    idx = input_ids
    finish = "length"
    with torch.no_grad():
        for _ in range(max(0, params.max_tokens)):
            logits = _next_logits(model, idx, block_size)
            tok = sample_next_token(logits, params, generated_ids=generated, generator=generator)
            # Stop before recording EOS so it counts as neither a content token
            # nor a usage token — and so this matches generate_stream exactly.
            if tok == eos_id:
                finish = "stop"
                break
            generated.append(tok)
            idx = torch.cat(
                [idx, torch.tensor([[tok]], device=idx.device, dtype=torch.long)], dim=1
            )
    return GenerateResult(token_ids=generated, finish_reason=finish)


def generate_stream(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    params: SamplingParams,
    eos_id: int,
    block_size: int,
    tokenizer: InferenceTokenizer,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[str, bool, str | None]]:
    """Yield ``(text_piece, done, finish_reason)`` tuples. The EOS token is not
    emitted as text; the terminal tuple carries ``done=True`` and the reason."""
    import torch

    generated: list[int] = []
    idx = input_ids
    prev_text = ""
    finish = "length"
    with torch.no_grad():
        for _ in range(max(0, params.max_tokens)):
            logits = _next_logits(model, idx, block_size)
            tok = sample_next_token(logits, params, generated_ids=generated, generator=generator)
            if tok == eos_id:
                finish = "stop"
                break
            generated.append(tok)
            idx = torch.cat(
                [idx, torch.tensor([[tok]], device=idx.device, dtype=torch.long)], dim=1
            )
            # Forward-only diff so we never re-emit already-streamed text. A
            # byte-level token can leave an incomplete multi-byte char rendered
            # as a trailing U+FFFD; hold it back until the next token completes
            # it rather than emitting a placeholder we'd have to correct.
            full = tokenizer.decode(generated)
            common = os.path.commonprefix([prev_text, full])
            piece = full[len(common) :]
            if piece.endswith("�"):
                piece = piece[:-1]
            if piece:
                prev_text = common + piece
                yield (piece, False, None)
    yield ("", True, finish)
