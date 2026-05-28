from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModularAdditionData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    p: int
    vocab_size: int


def make_modular_addition_data(
    p: int = 97,
    train_fraction: float = 0.4,
    val_fraction: float = 0.1,
    seed: int = 0,
    stratify_by_target: bool = True,
    device: torch.device | str = "cpu",
) -> ModularAdditionData:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    xs = torch.tensor([[a, b, p] for a, b in all_pairs], dtype=torch.long)
    ys = torch.tensor([(a + b) % p for a, b in all_pairs], dtype=torch.long)
    n = xs.shape[0]

    if stratify_by_target:
        train_idx, val_idx, test_idx = [], [], []
        for cls in range(p):
            mask = ys == cls
            cls_inds = mask.nonzero(as_tuple=True)[0]
            perm = cls_inds[torch.randperm(cls_inds.shape[0], generator=gen)]
            n_cls = perm.shape[0]
            n_tr = int(train_fraction * n_cls)
            n_va = int(val_fraction * n_cls)
            train_idx.append(perm[:n_tr])
            val_idx.append(perm[n_tr : n_tr + n_va])
            test_idx.append(perm[n_tr + n_va :])
        train_idx = torch.cat(train_idx)
        val_idx = torch.cat(val_idx)
        test_idx = torch.cat(test_idx)
    else:
        perm = torch.randperm(n, generator=gen)
        n_tr = int(train_fraction * n)
        n_va = int(val_fraction * n)
        train_idx = perm[:n_tr]
        val_idx = perm[n_tr : n_tr + n_va]
        test_idx = perm[n_tr + n_va :]

    dev = torch.device(device)
    return ModularAdditionData(
        train_x=xs[train_idx].to(dev),
        train_y=ys[train_idx].to(dev),
        val_x=xs[val_idx].to(dev),
        val_y=ys[val_idx].to(dev),
        test_x=xs[test_idx].to(dev),
        test_y=ys[test_idx].to(dev),
        p=p,
        vocab_size=p + 1,
    )
