from __future__ import annotations

import torch


def generate_ising_dataset(
    L: int,
    temperatures: list[float],
    samples_per_temperature: int,
    *,
    seed: int,
    device: torch.device,
    burn_in: int = 50,
    sample_spacing: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grids_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    temps_list: list[torch.Tensor] = []

    gen = torch.Generator(device="cpu").manual_seed(seed)

    for T in temperatures:
        beta = 1.0 / T
        grid = torch.ones(L, L, dtype=torch.float32, device=device)

        for _ in range(burn_in):
            grid = _checkerboard_sweep(grid, beta, L, gen, device)

        for _ in range(samples_per_temperature):
            for _ in range(sample_spacing):
                grid = _checkerboard_sweep(grid, beta, L, gen, device)
            grids_list.append(grid.clone())
            m = grid.mean()
            targets_list.append(m.abs().unsqueeze(0))
            temps_list.append(torch.tensor([T], dtype=torch.float32))

    grids = torch.stack(grids_list)
    targets = torch.cat(targets_list)
    temps = torch.cat(temps_list)
    return grids, targets, temps


def _checkerboard_sweep(
    grid: torch.Tensor,
    beta: float,
    L: int,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    rows = torch.arange(L, device=device)
    cols = torch.arange(L, device=device)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")

    for parity in (0, 1):
        mask = (row_grid + col_grid) % 2 == parity

        up = torch.roll(grid, 1, dims=0)
        down = torch.roll(grid, -1, dims=0)
        left = torch.roll(grid, 1, dims=1)
        right = torch.roll(grid, -1, dims=1)
        nn_sum = up + down + left + right

        dE = 2.0 * grid * nn_sum
        prob = (-beta * dE).exp().clamp(max=1.0)
        rand = torch.rand(L, L, generator=gen, device=device)
        flip = (dE <= 0) | (rand < prob)
        grid = torch.where(mask & flip, -grid, grid)

    return grid


def grids_to_tokens(grids: torch.Tensor) -> torch.Tensor:
    B, L, _ = grids.shape
    row_coords = torch.linspace(-1, 1, L, device=grids.device, dtype=grids.dtype)
    col_coords = torch.linspace(-1, 1, L, device=grids.device, dtype=grids.dtype)
    row_grid, col_grid = torch.meshgrid(row_coords, col_coords, indexing="ij")
    row_flat = row_grid.reshape(1, L * L, 1).expand(B, -1, -1)
    col_flat = col_grid.reshape(1, L * L, 1).expand(B, -1, -1)
    spin_flat = grids.reshape(B, L * L, 1)
    return torch.cat([spin_flat, row_flat, col_flat], dim=-1)
