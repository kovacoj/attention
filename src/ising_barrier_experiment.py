from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from barrier_diagnostics import compute_gradient_barrier_metrics
from ising_data import apply_mask, generate_ising_dataset, grids_to_tokens
from ising_model import TinyAttentionMagnetizationRegressor


@dataclass(frozen=True)
class BarrierDiagnosticRow:
    mask_prob: float
    seed: int
    epoch: int
    train_policy: str
    sketch_dim: int | None
    train_loss: float
    test_r2_all: float
    test_r2_near_tc: float
    test_mae_near_tc: float
    success_near_tc: int
    loss_ref: float
    loss_approx: float
    loss_gap: float
    grad_norm_ref: float
    grad_norm_approx: float
    grad_diff_norm: float
    grad_rel_error: float
    grad_cosine: float
    update_snr: float
    output_rel_error: float
    innovation_norm: float
    attn_entropy_norm: float
    attn_mean_max_mass: float


@dataclass(frozen=True)
class CorrectionSweepRow:
    mask_prob: float
    seed: int
    train_mode: str
    base_policy: str
    correction_period: int | None
    sketch_dim: int | None
    epochs: int
    test_r2_all: float
    test_r2_near_tc: float
    test_mae_near_tc: float
    success_near_tc: int
    attn_entropy_norm: float
    attn_mean_max_mass: float
    final_grad_cosine: float
    final_update_snr: float
    runtime_sec: float


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _compute_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum().item()
    ss_tot = ((target - target.mean()) ** 2).sum().item()
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _compute_attn_metrics(weights: torch.Tensor | None) -> tuple[float, float]:
    if weights is None:
        return 0.0, 1.0
    eps = 1e-300
    w = weights.clamp_min(eps)
    B, N, _ = w.shape
    entropy = -(w * w.log()).sum(dim=-1).mean().item()
    norm_entropy = entropy / math.log(N) if N > 1 else 0.0
    mean_max = w.max(dim=-1).values.mean().item()
    return norm_entropy, mean_max


def _build_data(
    L: int,
    temperatures: list[float],
    samples_per_temp: int,
    mask_prob: float,
    seed: int,
    device: torch.device,
):
    torch.manual_seed(seed)
    grids, targets, temps = generate_ising_dataset(
        L, temperatures, samples_per_temp, seed=seed, device=device
    )
    gen = torch.Generator(device="cpu").manual_seed(seed + 5000)
    if mask_prob < 1.0:
        masked_grids, masks = apply_mask(grids, mask_prob, generator=gen)
        tokens = grids_to_tokens(masked_grids, masks).to(torch.float32)
    else:
        tokens = grids_to_tokens(grids).to(torch.float32)
        masks = None
    return tokens, targets.to(torch.float32), temps


def _split_data(tokens, targets, temps, train_frac=0.8):
    n = tokens.shape[0]
    n_train = int(train_frac * n)
    perm = torch.randperm(n)
    return (
        tokens[perm[:n_train]], targets[perm[:n_train]], temps[perm[:n_train]],
        tokens[perm[n_train:]], targets[perm[n_train:]], temps[perm[n_train:]],
    )


def _eval_model(model, tokens, targets, temps, device):
    model.eval()
    with torch.no_grad():
        pred, weights = model(tokens.to(device))
        pred = pred.cpu()
        if weights is not None:
            weights = weights.cpu()
    r2_all = _compute_r2(pred, targets)
    near_tc = (temps >= 2.1) & (temps <= 2.5)
    mae_all = (pred - targets).abs().mean().item()
    if near_tc.any():
        r2_tc = _compute_r2(pred[near_tc], targets[near_tc])
        mae_tc = (pred[near_tc] - targets[near_tc]).abs().mean().item()
    else:
        r2_tc = r2_all
        mae_tc = mae_all
    success = 1 if r2_tc >= 0.80 or mae_tc <= 0.08 else 0
    ent, mass = _compute_attn_metrics(weights)
    return r2_all, r2_tc, mae_tc, success, ent, mass


def _train_epoch(model, tokens, targets, optimizer, loss_fn, batch_size, device, policy, sketch_dim, correction_period=None, batch_counter=None):
    model.train()
    n = tokens.shape[0]
    perm = torch.randperm(n)
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        bt = tokens[idx].to(device)
        by = targets[idx].to(device)

        use_correction = False
        if correction_period is not None and batch_counter is not None:
            if batch_counter[0] % correction_period == 0:
                use_correction = True

        if use_correction:
            orig_policy = model.precision_policy
            orig_sketch = model.sketch_dim
            model.precision_policy = "fp32"
            model.sketch_dim = None
        else:
            orig_policy = None
            orig_sketch = None

        pred, _ = model(bt)
        loss = loss_fn(pred, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

        if use_correction:
            model.precision_policy = orig_policy
            model.sketch_dim = orig_sketch

        if batch_counter is not None:
            batch_counter[0] += 1

    return epoch_loss / max(n_batches, 1)


def run_diagnostics(
    L, temperatures, samples_per_temp, epochs, batch_size, lr,
    d_model, depth, mask_probs, seeds, device,
):
    policies = [
        ("fp16_safe", None),
        ("bf16_safe", None),
        ("int8_qkv_dynamic", None),
        ("int8_logits_dynamic", None),
        ("sketch_4", 4),
        ("sketch_8", 8),
    ]

    rows: list[BarrierDiagnosticRow] = []
    for mask_prob in mask_probs:
        for seed in seeds:
            tokens, targets, temps = _build_data(
                L, temperatures, samples_per_temp, mask_prob, seed, device
            )
            tr_tok, tr_tgt, tr_tmp, te_tok, te_tgt, te_tmp = _split_data(
                tokens, targets, temps
            )

            probe_tok = tr_tok[:batch_size].to(device)
            probe_tgt = tr_tgt[:batch_size].to(device)

            for policy_name, sketch_dim in policies:
                input_dim = 4 if mask_prob < 1.0 else 3
                model = TinyAttentionMagnetizationRegressor(
                    input_dim=input_dim,
                    d_model=d_model,
                    depth=depth,
                    use_cls=True,
                    precision_policy=policy_name,
                    sketch_dim=sketch_dim,
                ).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                loss_fn = nn.MSELoss()

                print(f"[diag] mask={mask_prob:.1f} seed={seed} policy={policy_name}")
                for epoch in range(epochs):
                    model.precision_policy = policy_name
                    model.sketch_dim = sketch_dim
                    train_loss = _train_epoch(
                        model, tr_tok, tr_tgt, optimizer, loss_fn,
                        batch_size, device, policy_name, sketch_dim,
                    )

                    r2_all, r2_tc, mae_tc, success, ent, mass = _eval_model(
                        model, te_tok, te_tgt, te_tmp, device
                    )

                    barrier = compute_gradient_barrier_metrics(
                        model, probe_tok, probe_tgt,
                        reference_policy="fp32",
                        approximate_policy=policy_name,
                        sketch_dim=sketch_dim,
                        loss_fn=loss_fn,
                    )

                    rows.append(BarrierDiagnosticRow(
                        mask_prob=mask_prob, seed=seed, epoch=epoch,
                        train_policy=policy_name, sketch_dim=sketch_dim,
                        train_loss=train_loss,
                        test_r2_all=r2_all, test_r2_near_tc=r2_tc,
                        test_mae_near_tc=mae_tc, success_near_tc=success,
                        loss_ref=barrier.loss_ref, loss_approx=barrier.loss_approx,
                        loss_gap=barrier.loss_gap,
                        grad_norm_ref=barrier.grad_norm_ref,
                        grad_norm_approx=barrier.grad_norm_approx,
                        grad_diff_norm=barrier.grad_diff_norm,
                        grad_rel_error=barrier.grad_rel_error,
                        grad_cosine=barrier.grad_cosine,
                        update_snr=barrier.update_snr,
                        output_rel_error=barrier.output_rel_error,
                        innovation_norm=barrier.innovation_norm,
                        attn_entropy_norm=ent, attn_mean_max_mass=mass,
                    ))

    return rows


def run_correction_sweep(
    L, temperatures, samples_per_temp, epochs, batch_size, lr,
    d_model, depth, mask_probs, seeds, device, correction_period=5,
):
    modes = [
        ("fp32", "fp32", None, None),
        ("sketch_4", "sketch_4", 4, None),
        ("sketch_4_periodic_fp32_correction", "sketch_4", 4, correction_period),
        ("sketch_8", "sketch_8", 8, None),
        ("sketch_8_periodic_fp32_correction", "sketch_8", 8, correction_period),
        ("int8_logits_dynamic", "int8_logits_dynamic", None, None),
        ("int8_logits_periodic_fp32_correction", "int8_logits_dynamic", None, correction_period),
    ]

    rows: list[CorrectionSweepRow] = []
    for mask_prob in mask_probs:
        for seed in seeds:
            tokens, targets, temps = _build_data(
                L, temperatures, samples_per_temp, mask_prob, seed, device
            )
            tr_tok, tr_tgt, tr_tmp, te_tok, te_tgt, te_tmp = _split_data(
                tokens, targets, temps
            )
            probe_tok = tr_tok[:batch_size].to(device)
            probe_tgt = tr_tgt[:batch_size].to(device)

            for mode_name, base_policy, sketch_dim, corr_period in modes:
                input_dim = 4 if mask_prob < 1.0 else 3
                model = TinyAttentionMagnetizationRegressor(
                    input_dim=input_dim,
                    d_model=d_model,
                    depth=depth,
                    use_cls=True,
                    precision_policy=base_policy,
                    sketch_dim=sketch_dim,
                ).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                loss_fn = nn.MSELoss()

                print(f"[corr] mask={mask_prob:.1f} seed={seed} mode={mode_name}")
                batch_counter = [0]
                t0 = time.perf_counter()
                for epoch in range(epochs):
                    model.precision_policy = base_policy
                    model.sketch_dim = sketch_dim
                    _train_epoch(
                        model, tr_tok, tr_tgt, optimizer, loss_fn,
                        batch_size, device, base_policy, sketch_dim,
                        correction_period=corr_period,
                        batch_counter=batch_counter,
                    )
                elapsed = time.perf_counter() - t0

                r2_all, r2_tc, mae_tc, success, ent, mass = _eval_model(
                    model, te_tok, te_tgt, te_tmp, device
                )

                barrier = compute_gradient_barrier_metrics(
                    model, probe_tok, probe_tgt,
                    reference_policy="fp32",
                    approximate_policy=base_policy,
                    sketch_dim=sketch_dim,
                    loss_fn=loss_fn,
                )

                rows.append(CorrectionSweepRow(
                    mask_prob=mask_prob, seed=seed, train_mode=mode_name,
                    base_policy=base_policy, correction_period=corr_period,
                    sketch_dim=sketch_dim, epochs=epochs,
                    test_r2_all=r2_all, test_r2_near_tc=r2_tc,
                    test_mae_near_tc=mae_tc, success_near_tc=success,
                    attn_entropy_norm=ent, attn_mean_max_mass=mass,
                    final_grad_cosine=barrier.grad_cosine,
                    final_update_snr=barrier.update_snr,
                    runtime_sec=elapsed,
                ))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Ising training-barrier diagnostics")
    parser.add_argument("--output-diagnostics", type=Path, default=Path("results/ising_barrier_diagnostics.dev.csv"))
    parser.add_argument("--output-corrections", type=Path, default=Path("results/ising_correction_sweep.dev.csv"))
    parser.add_argument("--L", type=int, default=12)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mask-probs", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    parser.add_argument("--correction-period", type=int, default=5)
    args = parser.parse_args()

    device = _choose_device(args.device)
    temperatures = [1.5, 1.8, 2.1, 2.269, 2.5, 2.8, 3.2]

    diag_rows = run_diagnostics(
        args.L, temperatures, args.samples, args.epochs, args.batch_size,
        args.lr, args.d_model, args.depth, args.mask_probs, args.seeds, device,
    )
    args.output_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    with args.output_diagnostics.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(diag_rows[0]).keys()))
        writer.writeheader()
        for r in diag_rows:
            writer.writerow(asdict(r))
    print(f"Wrote {len(diag_rows)} diagnostic rows to {args.output_diagnostics}")

    corr_rows = run_correction_sweep(
        args.L, temperatures, args.samples, args.epochs, args.batch_size,
        args.lr, args.d_model, args.depth, args.mask_probs, args.seeds, device,
        args.correction_period,
    )
    args.output_corrections.parent.mkdir(parents=True, exist_ok=True)
    with args.output_corrections.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(corr_rows[0]).keys()))
        writer.writeheader()
        for r in corr_rows:
            writer.writerow(asdict(r))
    print(f"Wrote {len(corr_rows)} correction rows to {args.output_corrections}")


if __name__ == "__main__":
    main()
