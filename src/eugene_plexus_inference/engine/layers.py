"""Feed-forward blocks: dense (SwiGLU / classic), generic sparse MoE, MoD router, BlockAttnRes.

All feed-forward modules return ``(output, aux_loss)`` so the transformer block can
accumulate auxiliary losses (MoE load-balancing, MoD capacity) uniformly. Sparse
routing is a plain learned top-k mixture-of-experts.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .attention import RMSNorm

_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "gelu": F.gelu,
    "elu": F.elu,
}


def _swiglu_hidden(n_embd: int) -> int:
    return int(2 * (4 * n_embd) / 3)


class FeedForward(nn.Module):
    """Dense FFN. SwiGLU (gated SiLU) or a classic two-layer MLP for other activations."""

    def __init__(self, n_embd: int, activation: str = "swiglu") -> None:
        super().__init__()
        self.activation = activation
        hidden = _swiglu_hidden(n_embd)
        if activation == "swiglu":
            self.w1 = nn.Linear(n_embd, hidden, bias=False)  # gate
            self.w3 = nn.Linear(n_embd, hidden, bias=False)  # up
            self.w2 = nn.Linear(hidden, n_embd, bias=False)  # down
        else:
            self.w1 = nn.Linear(n_embd, hidden, bias=False)
            self.w2 = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.activation == "swiglu":
            out = self.w2(F.silu(self.w1(x)) * self.w3(x))
        else:
            act = _ACTIVATIONS.get(self.activation, F.relu)
            out = self.w2(act(self.w1(x)))
        return out, x.new_zeros(())


class _Expert(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        hidden = _swiglu_hidden(n_embd)
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w3 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFeedForward(nn.Module):
    """Generic top-k sparse MoE with an optional always-on shared expert (DeepSeek-V2 style).

    Token-level routing with a Switch-Transformer load-balancing auxiliary loss. Experts
    are computed densely and combined by routing weights — correct and simple; sparse
    dispatch is a later optimization.
    """

    def __init__(
        self,
        n_embd: int,
        *,
        n_experts: int = 4,
        top_k: int = 2,
        use_shared_expert: bool = True,
        aux_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.use_shared = use_shared_expert
        self.shared = _Expert(n_embd) if use_shared_expert else None
        self.n_routed = max(1, n_experts - (1 if use_shared_expert else 0))
        self.experts = nn.ModuleList([_Expert(n_embd) for _ in range(self.n_routed)])
        self.gate = nn.Linear(n_embd, self.n_routed, bias=False)
        self.top_k = min(top_k, self.n_routed)
        self.aux_loss_weight = aux_loss_weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.shared(x) if self.shared is not None else torch.zeros_like(x)
        probs = F.softmax(self.gate(x), dim=-1)  # (B, T, n_routed)
        top_v, top_i = probs.topk(self.top_k, dim=-1)
        top_v = top_v / (top_v.sum(-1, keepdim=True) + 1e-9)
        weights = torch.zeros_like(probs).scatter(-1, top_i, top_v)  # (B, T, n_routed)
        expert_out = torch.stack([e(x) for e in self.experts], dim=-2)  # (B, T, n_routed, C)
        routed = (expert_out * weights.unsqueeze(-1)).sum(dim=-2)
        # Switch load-balancing aux loss: n_routed * sum(f_i * P_i).
        f = (weights > 0).float().mean(dim=(0, 1))
        p = probs.mean(dim=(0, 1))
        aux = self.aux_loss_weight * self.n_routed * (f * p).sum()
        return out + routed, aux


class MoDRouter(nn.Module):
    """Mixture-of-Depths: a per-token soft gate on the FFN output + capacity aux loss."""

    def __init__(
        self, n_embd: int, *, capacity_factor: float = 0.5, aux_loss_weight: float = 0.01
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(n_embd, 1, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = torch.sigmoid(self.gate(x).squeeze(-1))  # (B, T)
        aux = ((gates.mean() - self.capacity_factor) ** 2) * self.aux_loss_weight
        return gates, aux


class BlockAttnRes(nn.Module):
    """Cross-block attention residual: attend the final block over its group's depth states."""

    def __init__(self, n_embd: int, n_heads: int = 4) -> None:
        super().__init__()
        if n_embd % n_heads != 0:
            n_heads = 1
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.norm = RMSNorm(n_embd)
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, block_states: list[torch.Tensor]) -> torch.Tensor:
        x_final = block_states[-1]
        b, t, c = x_final.shape
        g = len(block_states)
        q = self.q_proj(self.norm(x_final)).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        stack = torch.stack(block_states, dim=1).reshape(b * g, t, c)
        k = self.k_proj(stack).view(b, g, t, self.n_heads, self.head_dim)
        v = self.v_proj(stack).view(b, g, t, self.n_heads, self.head_dim)
        k = k.permute(0, 3, 1, 2, 4).reshape(b, self.n_heads, g * t, self.head_dim)
        v = v.permute(0, 3, 1, 2, 4).reshape(b, self.n_heads, g * t, self.head_dim)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).reshape(b, t, c)
        return x_final + torch.tanh(self.gate) * self.o_proj(attn)
