from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ising_data import generate_ising_dataset, grids_to_tokens
from ising_model import TinyAttentionMagnetizationRegressor


@dataclass(frozen=True)
class IsingLearningResult:
    L: int
    n_tokens: int
    seed: int
    train_policy: str
    eval_policy: str
    sketch_dim: int | None
    epochs: int
    train_loss: float
    test_mse: float
    test_mae: float
    test_r2: float
    success: int
    attn_entropy_norm: float
    attn_mean_max_mass: float
    mean_abs_m_low_T: float
    mean_abs_m_high_T: float
    runtime_sec: float


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum().item()
    ss_tot = ((target - target.mean()) ** 2).sum().item()
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_attention_metrics(weights: torch.Tensor) -> tuple[float, float]:
    eps = 1e-300
    w = weights.clamp_min(eps)
    B, N, _ = w.shape
    entropy = -(w * w.log()).sum(dim=-1).mean().item()
    norm_entropy = entropy / math.log(N) if N > 1 else 0.0
    mean_max = w.max(dim=-1).values.mean().item()
    return norm_entropy, mean_max


def train_one(
    L: int,
    d_model: int,
    depth: int,
    epochs: int,
    batch_size: int,
    lr: float,
    train_policy: str,
    sketch_dim: int | None,
    seed: int,
    device: torch.device,
    temperatures: list[float],
    samples_per_temperature: int,
) -> tuple[TinyAttentionMagnetizationRegressor, dict, float]:
    torch.manual_seed(seed)
    grids, targets, temps = generate_ising_dataset(
        L, temperatures, samples_per_temperature, seed=seed, device=device
    )
    tokens = grids_to_tokens(grids).to(torch.float32)
    targets_f = targets.to(torch.float32)

    n_total = tokens.shape[0]
    n_train = int(0.8 * n_total)
    perm = torch.randperm(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_tokens = tokens[train_idx]
    train_targets = targets_f[train_idx]
    test_tokens = tokens[test_idx]
    test_targets = targets_f[test_idx]
    test_temps = temps[test_idx]

    low_mask = test_temps < 2.0
    high_mask = test_temps > 2.5

    model = TinyAttentionMagnetizationRegressor(
        d_model=d_model,
        depth=depth,
        use_cls=True,
        precision_policy=train_policy,
        sketch_dim=sketch_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    t0 = time.perf_counter()
    final_loss = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = epoch_perm[start : start + batch_size]
            bt = train_tokens[idx].to(device)
            by = train_targets[idx].to(device)
            pred, _ = model(bt)
            loss = loss_fn(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_loss = epoch_loss / max(n_batches, 1)

    elapsed = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        pred, weights = model(test_tokens.to(device))
        pred = pred.cpu()
        if weights is not None:
            weights = weights.cpu()

    test_mse = ((pred - test_targets) ** 2).mean().item()
    test_mae = (pred - test_targets).abs().mean().item()
    test_r2 = compute_r2(pred, test_targets)
    success = 1 if test_r2 >= 0.90 else 0

    attn_ent, attn_mass = 0.0, 1.0
    if weights is not None:
        attn_ent, attn_mass = compute_attention_metrics(weights)

    m_low = float(test_targets[low_mask].mean()) if low_mask.any() else 0.0
    m_high = float(test_targets[high_mask].mean()) if high_mask.any() else 0.0

    metrics = dict(
        train_loss=final_loss,
        test_mse=test_mse,
        test_mae=test_mae,
        test_r2=test_r2,
        success=success,
        attn_entropy_norm=attn_ent,
        attn_mean_max_mass=attn_mass,
        mean_abs_m_low_T=m_low,
        mean_abs_m_high_T=m_high,
    )
    return model, metrics, elapsed


def evaluate_with_policy(
    model: TinyAttentionMagnetizationRegressor,
    eval_policy: str,
    L: int,
    temperatures: list[float],
    samples_per_temperature: int,
    seed: int,
    device: torch.device,
) -> dict:
    grids, targets, temps = generate_ising_dataset(
        L, temperatures, samples_per_temperature, seed=seed + 10000, device=device
    )
    tokens = grids_to_tokens(grids).to(torch.float32)
    targets_f = targets.to(torch.float32)

    original_policy = model.precision_policy
    model.precision_policy = eval_policy
    model.eval()
    with torch.no_grad():
        pred, weights = model(tokens.to(device))
        pred = pred.cpu()
        if weights is not None:
            weights = weights.cpu()
    model.precision_policy = original_policy

    test_mse = ((pred - targets_f) ** 2).mean().item()
    test_mae = (pred - targets_f).abs().mean().item()
    test_r2 = compute_r2(pred, targets_f)
    success = 1 if test_r2 >= 0.90 else 0

    attn_ent, attn_mass = 0.0, 1.0
    if weights is not None:
        attn_ent, attn_mass = compute_attention_metrics(weights)

    low_mask = temps < 2.0
    high_mask = temps > 2.5
    m_low = float(targets_f[low_mask].mean()) if low_mask.any() else 0.0
    m_high = float(targets_f[high_mask].mean()) if high_mask.any() else 0.0

    return dict(
        test_mse=test_mse,
        test_mae=test_mae,
        test_r2=test_r2,
        success=success,
        attn_entropy_norm=attn_ent,
        attn_mean_max_mass=attn_mass,
        mean_abs_m_low_T=m_low,
        mean_abs_m_high_T=m_high,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ising learnability transition experiment")
    parser.add_argument("--output", type=Path, default=Path("results/ising_learning_transition.dev.csv"))
    parser.add_argument("--L", type=int, default=12)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    L = args.L
    temperatures = [1.5, 1.8, 2.1, 2.269, 2.5, 2.8, 3.2]
    samples_per_temperature = args.samples

    train_policies = [
        ("fp32", None),
        ("bf16_safe", None),
        ("fp16_safe", None),
        ("int8_qkv_dynamic", None),
        ("int8_logits_dynamic", None),
        ("sketch_4", 4),
        ("sketch_8", 8),
        ("sketch_16", 16),
    ]

    eval_policies = [
        ("bf16_safe_eval", None),
        ("fp16_safe_eval", None),
        ("int8_qkv_dynamic_eval", None),
        ("int8_logits_dynamic_eval", None),
        ("sketch_4_eval", 4),
        ("sketch_8_eval", 8),
        ("sketch_16_eval", 16),
    ]

    results: list[IsingLearningResult] = []

    for seed in args.seeds:
        for policy_name, sketch_dim in train_policies:
            print(f"Training: seed={seed} policy={policy_name}")
            model, metrics, elapsed = train_one(
                L=L,
                d_model=args.d_model,
                depth=args.depth,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                train_policy=policy_name,
                sketch_dim=sketch_dim,
                seed=seed,
                device=device,
                temperatures=temperatures,
                samples_per_temperature=samples_per_temperature,
            )
            results.append(
                IsingLearningResult(
                    L=L,
                    n_tokens=L * L,
                    seed=seed,
                    train_policy=policy_name,
                    eval_policy=policy_name,
                    sketch_dim=sketch_dim,
                    epochs=args.epochs,
                    train_loss=metrics["train_loss"],
                    test_mse=metrics["test_mse"],
                    test_mae=metrics["test_mae"],
                    test_r2=metrics["test_r2"],
                    success=metrics["success"],
                    attn_entropy_norm=metrics["attn_entropy_norm"],
                    attn_mean_max_mass=metrics["attn_mean_max_mass"],
                    mean_abs_m_low_T=metrics["mean_abs_m_low_T"],
                    mean_abs_m_high_T=metrics["mean_abs_m_high_T"],
                    runtime_sec=elapsed,
                )
            )

            if policy_name == "fp32":
                for eval_name, eval_sketch in eval_policies:
                    print(f"  Evaluating fp32 model under: {eval_name}")
                    ev = evaluate_with_policy(
                        model, eval_name, L, temperatures,
                        samples_per_temperature, seed, device,
                    )
                    results.append(
                        IsingLearningResult(
                            L=L,
                            n_tokens=L * L,
                            seed=seed,
                            train_policy="fp32",
                            eval_policy=eval_name,
                            sketch_dim=eval_sketch,
                            epochs=args.epochs,
                            train_loss=metrics["train_loss"],
                            test_mse=ev["test_mse"],
                            test_mae=ev["test_mae"],
                            test_r2=ev["test_r2"],
                            success=ev["success"],
                            attn_entropy_norm=ev["attn_entropy_norm"],
                            attn_mean_max_mass=ev["attn_mean_max_mass"],
                            mean_abs_m_low_T=ev["mean_abs_m_low_T"],
                            mean_abs_m_high_T=ev["mean_abs_m_high_T"],
                            runtime_sec=elapsed,
                        )
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"Wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
