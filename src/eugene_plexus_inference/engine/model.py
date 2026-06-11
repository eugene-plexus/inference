"""Generic decoder-only transformer assembled from an ArchitectureConfig.

Embedding -> N transformer blocks -> final RMSNorm -> (weight-tied) LM head. A standard
GQA + RoPE + RMSNorm + SwiGLU decoder with optional MoE, Mixture-of-Depths, differential
attention, and block-attention residual.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .._generated.common_models import ArchitectureConfig
from .attention import DifferentialAttention, GroupedQueryAttention, RMSNorm
from .block import TransformerBlock, make_ffn_factory
from .layers import BlockAttnRes, MoDRouter


def _val(enum_or_str: Any, default: str) -> str:
    if enum_or_str is None:
        return default
    return getattr(enum_or_str, "value", enum_or_str)


def num_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    params = (p for p in model.parameters() if p.requires_grad or not trainable_only)
    return sum(p.numel() for p in params)


class GPTModel(nn.Module):
    def __init__(self, arch: ArchitectureConfig) -> None:
        super().__init__()
        n_embd = arch.nEmbd
        n_head = arch.nHead
        n_kv_head = arch.nKvHead or n_head
        if n_embd % n_head != 0:
            raise ValueError(f"nEmbd ({n_embd}) must be divisible by nHead ({n_head})")
        if n_head % n_kv_head != 0:
            raise ValueError(f"nHead ({n_head}) must be divisible by nKvHead ({n_kv_head})")
        head_dim = n_embd // n_head
        self.vocab_size = arch.vocabSize
        self.block_size = arch.blockSize
        self.n_layer = arch.nLayer

        activation = _val(arch.activation, "swiglu")
        variant = _val(arch.attentionVariant, "gqa")
        use_qk_norm = bool(arch.useQkNorm)
        rope_base = float(arch.ropeBase or 500000)

        ctx = arch.contextExtension
        yarn_max_factor = (
            float(ctx.maxFactor)
            if ctx is not None and _val(ctx.mode, "none") == "yarn" and ctx.maxFactor
            else 1.0
        )

        ffn = arch.ffn
        ffn_type = _val(ffn.type, "dense") if ffn is not None else "dense"
        ffn_factory = make_ffn_factory(
            n_embd,
            activation=activation,
            ffn_type=ffn_type,
            n_experts=(ffn.nExperts if ffn and ffn.nExperts else 4),
            top_k=(ffn.topK if ffn and ffn.topK else 2),
            use_shared_expert=bool(ffn.useSharedExpert) if ffn else False,
            moe_aux_weight=(ffn.auxLossWeight if ffn and ffn.auxLossWeight is not None else 0.01),
        )

        mod_cfg = arch.mixtureOfDepths
        mod_enabled = bool(mod_cfg.enabled) if mod_cfg is not None else False

        def make_mod() -> MoDRouter | None:
            if not mod_enabled or mod_cfg is None:
                return None
            return MoDRouter(
                n_embd,
                capacity_factor=mod_cfg.capacityFactor or 0.5,
                aux_loss_weight=mod_cfg.auxLossWeight or 0.01,
            )

        def make_attn(layer_idx: int) -> nn.Module:
            if variant == "differential":
                return DifferentialAttention(
                    n_embd=n_embd,
                    n_head=n_head,
                    n_kv_head=n_kv_head,
                    head_dim=head_dim,
                    rope_base=rope_base,
                    block_size=self.block_size,
                    layer_idx=layer_idx,
                    use_qk_norm=use_qk_norm,
                    yarn_max_factor=yarn_max_factor,
                )
            return GroupedQueryAttention(
                n_embd=n_embd,
                n_head=n_head,
                n_kv_head=n_kv_head,
                head_dim=head_dim,
                rope_base=rope_base,
                block_size=self.block_size,
                use_qk_norm=use_qk_norm,
                yarn_max_factor=yarn_max_factor,
            )

        self.tok_emb = nn.Embedding(self.vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(n_embd, attn=make_attn(i), ffn=ffn_factory(), mod=make_mod())
                for i in range(self.n_layer)
            ]
        )

        self.group_size = arch.blockAttnResGroupSize or 0
        self.block_attn_res: nn.ModuleList | None = None
        if self.group_size > 0:
            n_groups = (self.n_layer + self.group_size - 1) // self.group_size
            self.block_attn_res = nn.ModuleList([BlockAttnRes(n_embd) for _ in range(n_groups)])

        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, self.vocab_size, bias=False)
        self.apply(self._init_weights)
        if arch.weightTying is None or arch.weightTying:
            self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _forward_blocks(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aux_total = x.new_zeros(())
        if self.block_attn_res is None:
            for block in self.blocks:
                x, aux = block(x)
                aux_total = aux_total + aux
            return x, aux_total

        states: list[torch.Tensor] = []
        group_idx = 0
        last = self.n_layer - 1
        for i, block in enumerate(self.blocks):
            x, aux = block(x)
            aux_total = aux_total + aux
            states.append(x)
            if len(states) == self.group_size or i == last:
                x = self.block_attn_res[group_idx](states)
                states = []
                group_idx += 1
        return x, aux_total

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.tok_emb(idx)
        x, aux = self._forward_blocks(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss: torch.Tensor | None = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, self.vocab_size), targets.reshape(-1))
            loss = ce + aux
        return logits, loss
