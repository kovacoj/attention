from __future__ import annotations

import torch


def fake_cast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.to(dtype).to(torch.float32)


def fake_quant_symmetric_int8_ste(
    x: torch.Tensor,
    *,
    dim: int | tuple[int, ...] | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dim is None:
        max_abs = x.detach().abs().max().clamp_min(eps)
    else:
        max_abs = x.detach().abs().amax(dim=dim, keepdim=True).clamp_min(eps)
    scale = max_abs / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127)
    x_hat = scale * q
    x_ste = x + (x_hat - x).detach()
    with torch.no_grad():
        clip_frac = (x.detach().abs() / scale > 127).float().mean()
    return x_ste, scale, clip_frac
