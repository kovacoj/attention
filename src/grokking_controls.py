from __future__ import annotations

import math

import torch
import torch.nn as nn


class GrokMLP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        d_hidden: int = 512,
        n_hidden: int = 2,
        max_seq_len: int = 3,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layers = []
        in_dim = d_model * max_seq_len
        for i in range(n_hidden):
            out_dim = d_hidden if i < n_hidden - 1 else d_model
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_hidden - 1:
                layers.append(nn.GELU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(d_model, vocab_size - 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = h.reshape(B, -1)
        h = self.mlp(h)
        return self.head(h)


def eval_uniform_attention(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    base_policy: str = "fp32",
    sketch_dim: int | None = None,
) -> tuple[float, float]:
    from grokking_model import GrokTransformer

    if not isinstance(model, GrokTransformer):
        return float("nan"), float("nan")

    model.eval()
    with torch.no_grad():
        logits_normal, _ = model(
            x,
            train_policy=base_policy,
            eval_policy=base_policy,
            sketch_dim=sketch_dim,
        )
        logits_uniform, _ = model(
            x,
            train_policy=base_policy,
            eval_policy=base_policy,
            sketch_dim=sketch_dim,
            force_uniform_attention=True,
        )

    acc_normal = (logits_normal.argmax(dim=-1) == y).float().mean().item()
    acc_uniform = (logits_uniform.argmax(dim=-1) == y).float().mean().item()
    return acc_normal, acc_uniform
