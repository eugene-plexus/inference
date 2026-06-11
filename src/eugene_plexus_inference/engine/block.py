"""Pre-norm decoder transformer block (attention + feed-forward, learned residual scales)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .attention import RMSNorm
from .layers import MoDRouter


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block: x = x + attn(ln1(x))*s1; x = x + gate*ffn(ln2(x))*s2.

    `attn` and `ffn` are constructed by the caller (so the block is agnostic to GQA vs
    differential attention and dense vs MoE FFN). Residual scales are learned and
    initialized to 1.0 (standard). `mod` optionally soft-gates the FFN output.
    """

    def __init__(
        self,
        n_embd: int,
        *,
        attn: nn.Module,
        ffn: nn.Module,
        mod: MoDRouter | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        self.attn = attn
        self.ffn = ffn
        self.mod = mod
        self.attn_res_scale = nn.Parameter(torch.ones(1))
        self.ffn_res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x)) * self.attn_res_scale
        normed = self.ln2(x)
        ffn_out, aux = self.ffn(normed)
        if self.mod is not None:
            gates, mod_aux = self.mod(normed)
            ffn_out = gates.unsqueeze(-1) * ffn_out
            aux = aux + mod_aux
        x = x + ffn_out * self.ffn_res_scale
        return x, aux


def make_ffn_factory(
    n_embd: int,
    *,
    activation: str,
    ffn_type: str,
    n_experts: int,
    top_k: int,
    use_shared_expert: bool,
    moe_aux_weight: float,
) -> Callable[[], nn.Module]:
    """Return a zero-arg factory that builds a fresh FFN module per block."""
    from .layers import FeedForward, MoEFeedForward

    def factory() -> nn.Module:
        if ffn_type == "moe":
            return MoEFeedForward(
                n_embd,
                n_experts=n_experts,
                top_k=top_k,
                use_shared_expert=use_shared_expert,
                aux_loss_weight=moe_aux_weight,
            )
        return FeedForward(n_embd, activation=activation)

    return factory
