from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetric_int8_quant_dequant(
    x: torch.Tensor,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, float, float]:
    max_abs = x.abs().max().item()
    if max_abs == 0.0:
        scale = 1.0
    elif scale is None:
        scale = max_abs / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127)
    x_hat = scale * q
    clip_rate = (x.abs() / scale > 127).float().mean().item() if scale > 0 else 0.0
    return x_hat, scale, clip_rate


class GrokTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 1,
        n_heads: int = 1,
        mlp_hidden: int = 128,
        dropout: float = 0.0,
        max_seq_len: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            _GrokBlock(d_model, n_heads, self.d_head, mlp_hidden, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size - 1, bias=False)

        if n_heads > 0 and d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        precision_policy: str = "fp32",
        sketch_dim: int | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_emb(x) + self.pos_emb(positions)

        last_attn = None
        for layer in self.layers:
            h, attn = layer(
                h,
                precision_policy=precision_policy,
                sketch_dim=sketch_dim,
            )
            last_attn = attn

        h = self.norm(h)
        logits = self.head(h[:, -1, :])

        if return_attention:
            return logits, last_attn
        return logits, None


class _GrokBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, mlp_hidden: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_norm = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _apply_storage_policy(self, x: torch.Tensor, policy: str) -> torch.Tensor:
        if policy in ("fp32", "int8_qkv_dynamic", "int8_logits_dynamic"):
            return x
        elif policy in ("bf16_safe",):
            return x.to(torch.bfloat16).to(torch.float32)
        elif policy in ("fp16_safe",):
            return x.to(torch.float16).to(torch.float32)
        return x

    def forward(
        self,
        x: torch.Tensor,
        *,
        precision_policy: str = "fp32",
        sketch_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, _ = x.shape
        residual = x
        h = self.attn_norm(x)

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        if precision_policy in ("int8_qkv_dynamic",):
            q, _, _ = symmetric_int8_quant_dequant(q)
            k, _, _ = symmetric_int8_quant_dequant(k)
            v, _, _ = symmetric_int8_quant_dequant(v)
        else:
            q = self._apply_storage_policy(q, precision_policy)
            k = self._apply_storage_policy(k, precision_policy)
            v = self._apply_storage_policy(v, precision_policy)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if sketch_dim is not None and precision_policy.startswith("sketch"):
            S = torch.randn(self.d_head, sketch_dim, dtype=q.dtype, device=q.device) / math.sqrt(sketch_dim)
            q_s = q @ S
            k_s = k @ S
            logits = (q_s @ k_s.transpose(-1, -2)) / math.sqrt(self.d_head)
        else:
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)

        if precision_policy in ("int8_logits_dynamic",):
            logits, _, _ = symmetric_int8_quant_dequant(logits)

        weights = F.softmax(logits, dim=-1)
        attn_out = weights @ v

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = self.out_proj(attn_out)

        h = residual + self.dropout(attn_out)
        residual = h
        h = self.mlp_norm(h)
        h = residual + self.dropout(self.mlp(h))

        return h, weights
