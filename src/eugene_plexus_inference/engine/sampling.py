"""Turn a next-token logit vector into a sampled token id.

Standard decoder sampling: optional repetition penalty, then either greedy
(``temperature <= 0``) or temperature-scaled top-k / top-p (nucleus) sampling.
The OpenAI-compatible request surface exposes ``temperature`` and ``top_p``;
``top_k`` and ``repetition_penalty`` are internal knobs with neutral defaults
(disabled) so behavior matches a vanilla OpenAI server unless tuned.

torch is imported lazily by callers; the functions here take and return torch
tensors / ints but the module itself has no import-time torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class SamplingParams:
    """Resolved per-request sampling configuration (request value, else config default)."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 disables top-k
    repetition_penalty: float = 1.0  # 1.0 disables
    max_tokens: int = 512


def _apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: list[int], penalty: float
) -> torch.Tensor:
    """CTRL-style repetition penalty (Keskar et al.): divide positive logits of
    already-generated tokens by ``penalty`` and multiply negative ones, pushing
    their probability down."""
    if penalty == 1.0 or not generated_ids:
        return logits
    import torch

    idx = torch.tensor(sorted(set(generated_ids)), device=logits.device, dtype=torch.long)
    selected = logits.index_select(0, idx)
    selected = torch.where(selected > 0, selected / penalty, selected * penalty)
    logits = logits.clone()
    logits.index_copy_(0, idx, selected)
    return logits


def sample_next_token(
    logits: torch.Tensor,
    params: SamplingParams,
    *,
    generated_ids: list[int],
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token id from a 1-D ``logits`` vector (shape ``[vocab]``).

    ``generator`` makes sampling reproducible when a seed is supplied; the
    greedy path ignores it. ``generated_ids`` feeds the repetition penalty.
    """
    import torch

    logits = _apply_repetition_penalty(logits, generated_ids, params.repetition_penalty)

    if params.temperature <= 0.0:
        return int(torch.argmax(logits).item())

    logits = logits / params.temperature

    # top-k: keep only the k highest logits.
    if params.top_k and params.top_k > 0:
        k = min(params.top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < kth, logits.new_full((), float("-inf")), logits)

    # top-p (nucleus): keep the smallest set whose cumulative prob >= top_p.
    if params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Mask everything strictly past the nucleus, always keeping the top token.
        remove = cum > params.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(token.item())
