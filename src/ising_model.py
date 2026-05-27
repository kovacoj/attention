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
    clip_rate = (x.abs() / scale > 127).float().mean().item()
    return x_hat, scale, clip_rate


class TinyAttentionMagnetizationRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 32,
        depth: int = 1,
        use_cls: bool = True,
        precision_policy: str = "fp32",
        sketch_dim: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.use_cls = use_cls
        self.precision_policy = precision_policy
        self.sketch_dim = sketch_dim

        self.input_proj = nn.Linear(input_dim, d_model)

        if use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.q_proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(depth)])
        self.k_proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(depth)])
        self.v_proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(depth)])
        self.out_proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])

        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        if sketch_dim is not None:
            self.register_buffer(
                "_sketch_matrix",
                torch.randn(d_model, sketch_dim) / math.sqrt(sketch_dim),
            )

    def _apply_storage_policy(self, x: torch.Tensor) -> torch.Tensor:
        if self.precision_policy == "fp32":
            return x
        elif self.precision_policy in ("bf16_safe", "bf16_safe_eval"):
            return x.to(torch.bfloat16).to(torch.float32)
        elif self.precision_policy in ("fp16_safe", "fp16_safe_eval"):
            return x.to(torch.float16).to(torch.float32)
        return x

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = q.shape[-1]

        if self.precision_policy in ("int8_qkv_dynamic", "int8_qkv_dynamic_eval"):
            q, _, _ = symmetric_int8_quant_dequant(q)
            k, _, _ = symmetric_int8_quant_dequant(k)
            v, _, _ = symmetric_int8_quant_dequant(v)
        else:
            q = self._apply_storage_policy(q)
            k = self._apply_storage_policy(k)
            v = self._apply_storage_policy(v)

        if self.sketch_dim is not None and self.precision_policy.startswith("sketch"):
            S = self._sketch_matrix.to(q.dtype)
            q_s = q @ S
            k_s = k @ S
            logits = (q_s @ k_s.transpose(-1, -2)) / math.sqrt(d_k)
        else:
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(d_k)

        if self.precision_policy in ("int8_logits_dynamic", "int8_logits_dynamic_eval"):
            logits, _, _ = symmetric_int8_quant_dequant(logits)

        weights = F.softmax(logits, dim=-1)
        attn_out = weights @ v
        attn_out = self.out_proj[layer_idx](attn_out)
        return attn_out, weights

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B = x.shape[0]
        h = self.input_proj(x)

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)

        last_weights = None
        for i in range(self.depth):
            q = self.q_proj[i](h)
            k = self.k_proj[i](h)
            v = self.v_proj[i](h)

            attn_out, last_weights = self._compute_attention(q, k, v, i)
            h = self.norms[i](h + attn_out)

        if self.use_cls:
            pooled = h[:, 0]
        else:
            pooled = h.mean(dim=1)

        y_hat = torch.sigmoid(self.readout(pooled)).squeeze(-1)
        return y_hat, last_weights
