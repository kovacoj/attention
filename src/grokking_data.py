from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModularAdditionData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    p: int
    vocab_size: int


def make_modular_addition_data(
    p: int = 97,
    train_fraction: float = 0.4,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> ModularAdditionData:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    n = len(all_pairs)
    perm = torch.randperm(n, generator=gen)
    n_train = int(train_fraction * n)

    xs = torch.tensor([[a, b, p] for a, b in all_pairs], dtype=torch.long)
    ys = torch.tensor([(a + b) % p for a, b in all_pairs], dtype=torch.long)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_x = xs[train_idx].to(device)
    train_y = ys[train_idx].to(device)
    test_x = xs[test_idx].to(device)
    test_y = ys[test_idx].to(device)

    return ModularAdditionData(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        p=p,
        vocab_size=p + 1,
    )
