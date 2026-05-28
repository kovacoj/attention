from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from precision_policies import fake_cast, fake_quant_symmetric_int8_ste


class GrokTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        d_head: int | None = None,
        d_mlp: int = 512,
        dropout: float = 0.0,
        max_seq_len: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head or (d_model // n_heads)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            _GrokBlock(d_model, n_heads, self.d_head, d_mlp, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size - 1, bias=False)

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        train_policy: str = "fp32",
        eval_policy: str | None = None,
        sketch_dim: int | None = None,
        resample_sketch: bool = False,
        return_attention: bool = False,
        force_uniform_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.training:
            policy = train_policy
        else:
            policy = eval_policy if eval_policy is not None else train_policy

        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_emb(x) + self.pos_emb(positions)

        last_attn = None
        for layer in self.layers:
            h, attn = layer(
                h,
                precision_policy=policy,
                sketch_dim=sketch_dim,
                resample_sketch=resample_sketch,
                force_uniform_attention=force_uniform_attention,
            )
            last_attn = attn

        h = self.norm(h)
        logits = self.head(h[:, -1, :])

        if return_attention:
            return logits, last_attn
        return logits, None


class _GrokBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_mlp: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.max_sketch_dim = 32

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.attn_norm = nn.LayerNorm(d_model)

        self.register_buffer("sketch_base", torch.randn(d_head, self.max_sketch_dim))

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _apply_qkv_policy(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        policy: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        clip_qkv = 0.0
        clip_logits = 0.0

        if policy.startswith("int8_qkv"):
            q, _, c1 = fake_quant_symmetric_int8_ste(q)
            k, _, c2 = fake_quant_symmetric_int8_ste(k)
            v, _, c3 = fake_quant_symmetric_int8_ste(v)
            clip_qkv = max(c1.item(), c2.item(), c3.item())
        elif policy.startswith("bf16"):
            q = fake_cast(q, torch.bfloat16)
            k = fake_cast(k, torch.bfloat16)
            v = fake_cast(v, torch.bfloat16)
        elif policy.startswith("fp16"):
            q = fake_cast(q, torch.float16)
            k = fake_cast(k, torch.float16)
            v = fake_cast(v, torch.float16)

        return q, k, v, clip_qkv, clip_logits

    def forward(
        self,
        x: torch.Tensor,
        *,
        precision_policy: str = "fp32",
        sketch_dim: int | None = None,
        resample_sketch: bool = False,
        force_uniform_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, _ = x.shape
        residual = x
        h = self.attn_norm(x)

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q, k, v, clip_qkv, clip_logits = self._apply_qkv_policy(q, k, v, precision_policy)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        is_sketch = precision_policy.startswith("sketch")

        if is_sketch and sketch_dim is not None:
            if sketch_dim > self.max_sketch_dim:
                raise ValueError(f"sketch_dim={sketch_dim} exceeds max_sketch_dim={self.max_sketch_dim}")
            if resample_sketch:
                S = torch.randn(
                    self.d_head, sketch_dim,
                    dtype=q.dtype, device=q.device,
                ) / math.sqrt(sketch_dim)
            else:
                S = self.sketch_base[:, :sketch_dim].to(dtype=q.dtype, device=q.device) / math.sqrt(sketch_dim)
            q_s = q @ S
            k_s = k @ S
            logits = (q_s @ k_s.transpose(-1, -2)) / math.sqrt(self.d_head)
        else:
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)

        if precision_policy.startswith("int8_logits"):
            logits, _, clip_logits = fake_quant_symmetric_int8_ste(logits)

        weights = F.softmax(logits, dim=-1)

        if force_uniform_attention:
            weights = torch.full_like(weights, 1.0 / T)

        attn_out = weights @ v

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = self.out_proj(attn_out)

        h = residual + self.dropout(attn_out)
        residual = h
        h = self.mlp_norm(h)
        h = residual + self.dropout(self.mlp(h))

        return h, weights
