from __future__ import annotations

import torch


def fourier_energy_ratio(
    embeddings: torch.Tensor,
    p: int,
) -> float:
    if embeddings.shape[0] < p:
        padded = torch.zeros(p, embeddings.shape[1], device=embeddings.device, dtype=embeddings.dtype)
        padded[:embeddings.shape[0]] = embeddings
        embeddings = padded
    E = embeddings[:p].float()
    F = torch.fft.fft(E, dim=0)
    energy_per_freq = F.norm(dim=1) ** 2
    total = energy_per_freq.sum().item()
    if total < 1e-30:
        return 0.0
    zero_mode = energy_per_freq[0].item()
    return 1.0 - zero_mode / total


def compute_fourier_metrics(
    model: torch.nn.Module,
    p: int,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    emb = model.token_emb.weight.detach()
    ratio = fourier_energy_ratio(emb, p)
    return {"fourier_energy_ratio": ratio}
