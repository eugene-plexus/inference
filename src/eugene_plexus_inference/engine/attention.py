"""Attention: RoPE (with NTK / YaRN context extension), GQA, optional differential.

Rotary position embeddings, grouped-query attention over PyTorch's
scaled_dot_product_attention (causal), optional QK-norm, and an optional
differential-attention variant (Ye et al., arXiv:2410.05258). Attention is purely
causal with no additive state-dependent bias.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _ntk_base(base: float, head_dim: int, ntk_factor: float) -> float:
    """NTK-scaled RoPE base for context extension (ntk_factor >= 1)."""
    if ntk_factor <= 1.0:
        return base
    exponent = min(4.0, max(1.0, head_dim / max(2, head_dim - 2)))
    return base * (ntk_factor**exponent)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float,
    device: torch.device,
    *,
    ntk_factor: float = 1.0,
    yarn_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) each shape (seq_len, head_dim) for rotary embedding.

    NTK uniformly rescales the base frequency. YaRN (yarn_factor > 1) blends
    original and interpolated frequencies per-dimension with an mscale logit
    correction.
    """
    half = head_dim // 2
    idx = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    mscale = 1.0

    if yarn_factor > 1.0:
        freq_orig = 1.0 / (base ** (idx / head_dim))
        freq_inter = freq_orig / yarn_factor
        beta_fast, beta_slow = 32.0, 1.0

        def _corr_dim(num_rot: float) -> float:
            return (head_dim * math.log(seq_len / (num_rot * 2 * math.pi))) / (2 * math.log(base))

        low = max(0, math.floor(_corr_dim(beta_fast)))
        high = min(half - 1, math.ceil(_corr_dim(beta_slow)))
        ramp = torch.clamp(
            (torch.arange(half, dtype=torch.float32, device=device) - low) / max(1, high - low),
            0.0,
            1.0,
        )
        blend = 1.0 - ramp
        inv_freq = freq_inter * (1.0 - blend) + freq_orig * blend
        mscale = 0.1 * math.log(yarn_factor) + 1.0
    else:
        scaled_base = _ntk_base(base, head_dim, ntk_factor)
        inv_freq = 1.0 / (scaled_base ** (idx / head_dim))

    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return (emb.cos() * mscale), (emb.sin() * mscale)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to x of shape (B, T, n_head, head_dim)."""
    cos = cos.to(x.dtype)[None, :, None, :]
    sin = sin.to(x.dtype)[None, :, None, :]
    return x * cos + _rotate_half(x) * sin


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, T, hd) -> (B, n_kv * n_rep, T, hd) via zero-copy expand+reshape."""
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class GroupedQueryAttention(nn.Module):
    """Causal GQA with RoPE and optional QK-norm."""

    def __init__(
        self,
        *,
        n_embd: int,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        rope_base: float,
        block_size: int,
        use_qk_norm: bool = False,
        yarn_max_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.block_size = block_size
        self.yarn_max_factor = yarn_max_factor
        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=False)
        self.q_norm = RMSNorm(head_dim) if use_qk_norm else None
        self.k_norm = RMSNorm(head_dim) if use_qk_norm else None
        self._cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _rope(self, t: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ntk_factor = 1.0
        yarn_factor = 1.0
        if t > self.block_size:
            base = t / self.block_size
            if self.yarn_max_factor > 1.0:
                yarn_factor = min(base, self.yarn_max_factor)
            else:
                ntk_factor = base
        key = (t, device)
        cached = self._cache.get(key)
        if cached is None:
            cached = build_rope_cache(
                t,
                self.head_dim,
                self.rope_base,
                device,
                ntk_factor=ntk_factor,
                yarn_factor=yarn_factor,
            )
            self._cache[key] = cached
        return cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim)

        cos, sin = self._rope(t, x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.n_kv_head < self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = _repeat_kv(k, n_rep)
            v = _repeat_kv(v, n_rep)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)
        return self.o_proj(out)


class DifferentialAttention(nn.Module):
    """Differential attention: two softmax maps subtracted by a learned lambda.

    Requires even n_head / n_kv_head. Reference: Ye et al., arXiv:2410.05258.
    """

    def __init__(
        self,
        *,
        n_embd: int,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        rope_base: float,
        block_size: int,
        layer_idx: int,
        use_qk_norm: bool = False,
        yarn_max_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if n_head % 2 != 0 or n_kv_head % 2 != 0:
            raise ValueError("differential attention requires even n_head and n_kv_head")
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.block_size = block_size
        self.yarn_max_factor = yarn_max_factor
        self.n_diff = n_head // 2
        self.n_diff_kv = n_kv_head // 2
        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_diff * 2 * head_dim, n_embd, bias=False)
        self.q_norm = RMSNorm(head_dim) if use_qk_norm else None
        self.k_norm = RMSNorm(head_dim) if use_qk_norm else None
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(head_dim).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(head_dim).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(head_dim).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(head_dim).normal_(0, 0.1))
        self.subln = RMSNorm(2 * head_dim)
        self._cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _rope(self, t: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ntk_factor = 1.0
        yarn_factor = 1.0
        if t > self.block_size:
            base = t / self.block_size
            if self.yarn_max_factor > 1.0:
                yarn_factor = min(base, self.yarn_max_factor)
            else:
                ntk_factor = base
        key = (t, device)
        cached = self._cache.get(key)
        if cached is None:
            cached = build_rope_cache(
                t,
                self.head_dim,
                self.rope_base,
                device,
                ntk_factor=ntk_factor,
                yarn_factor=yarn_factor,
            )
            self._cache[key] = cached
        return cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim)

        cos, sin = self._rope(t, x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.view(b, t, self.n_diff, 2, self.head_dim)
        k = k.view(b, t, self.n_diff_kv, 2, self.head_dim)
        q1, q2 = q[..., 0, :].transpose(1, 2), q[..., 1, :].transpose(1, 2)
        k1, k2 = k[..., 0, :].transpose(1, 2), k[..., 1, :].transpose(1, 2)
        v = v.view(b, t, self.n_diff_kv, 2 * self.head_dim).transpose(1, 2)

        if self.n_diff_kv < self.n_diff:
            n_rep = self.n_diff // self.n_diff_kv
            k1, k2 = _repeat_kv(k1, n_rep), _repeat_kv(k2, n_rep)
            v = _repeat_kv(v, n_rep)
        v1, v2 = v[..., : self.head_dim], v[..., self.head_dim :]

        a11 = F.scaled_dot_product_attention(q1, k1, v1, is_causal=True)
        a12 = F.scaled_dot_product_attention(q1, k1, v2, is_causal=True)
        a21 = F.scaled_dot_product_attention(q2, k2, v1, is_causal=True)
        a22 = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
        attn1 = torch.cat([a11, a12], dim=-1)
        attn2 = torch.cat([a21, a22], dim=-1)

        lam = (
            torch.exp((self.lambda_q1 * self.lambda_k1).sum())
            - torch.exp((self.lambda_q2 * self.lambda_k2).sum())
            + self.lambda_init
        )
        attn = attn1 - lam * attn2
        attn = self.subln(attn) * (1.0 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(b, t, self.n_diff * 2 * self.head_dim)
        return self.o_proj(attn)
