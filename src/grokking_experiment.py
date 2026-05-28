from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from barrier_diagnostics import compute_gradient_barrier_metrics
from grokking_controls import GrokMLP, eval_uniform_attention
from grokking_data import make_modular_addition_data
from grokking_fourier import compute_fourier_metrics
from grokking_model import GrokTransformer
from precision_policies import innovation_prob_kl

POLICY_TABLE = [
    ("fp32", "fp32", None, False, None),
    ("bf16_safe", "bf16_safe", None, False, None),
    ("fp16_safe", "fp16_safe", None, False, None),
    ("int8_logits_dynamic", "int8_logits_dynamic", None, False, None),
    ("int8_qkv_dynamic", "int8_qkv_dynamic", None, False, None),
    ("sketch_4_fixed", "sketch_4", 4, False, None),
    ("sketch_8_fixed", "sketch_8", 8, False, None),
    ("sketch_16_fixed", "sketch_16", 16, False, None),
    ("sketch_32_fixed", "sketch_32", 32, False, None),
    ("sketch_4_resampled", "sketch_4", 4, True, None),
    ("sketch_8_resampled", "sketch_8", 8, True, None),
    ("sketch_16_resampled", "sketch_16", 16, True, None),
    ("sketch_32_resampled", "sketch_32", 32, True, None),
    ("sketch_4_periodic_fp32_correction", "sketch_4", 4, False, 10),
    ("int8_logits_periodic_fp32_correction", "int8_logits_dynamic", None, False, 10),
    ("fp32_then_sketch_after_mem", "sketch_4", 4, False, None),
    ("sketch_then_fp32_after_mem", "sketch_4", 4, False, None),
]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def _attn_metrics(weights: torch.Tensor | None) -> tuple[float, float]:
    if weights is None:
        return float("nan"), float("nan")
    eps = 1e-300
    w = weights.clamp_min(eps)
    B, H, T, _ = w.shape
    entropy = -(w * w.log()).sum(dim=-1).mean().item()
    norm_entropy = entropy / math.log(T) if T > 1 else 0.0
    mean_max = w.max(dim=-1).values.mean().item()
    return norm_entropy, mean_max


@dataclass(frozen=True)
class CurveRow:
    run_id: str
    seed: int
    split_seed: int
    p: int
    train_fraction: float
    val_fraction: float
    architecture: str
    d_model: int
    n_layers: int
    n_heads: int
    d_mlp: int
    optimizer: str
    lr: float
    weight_decay: float
    batch_size: str
    train_policy: str
    eval_policy: str
    base_policy: str
    sketch_dim: int | None
    fixed_sketch: bool
    resample_sketch: bool
    correction_period: int | None
    step: int
    train_loss: float
    val_loss: float
    test_loss: float
    train_acc: float
    val_acc: float
    test_acc: float
    memorization_reached: int
    grokking_reached: int
    memorization_step: int
    grokking_step: int
    grokking_delay: int
    attn_entropy_eq: float
    attn_maxmass_eq: float
    uniform_attn_eval_drop: float
    fourier_energy_ratio: float
    max_logit: float
    mean_logit_abs: float
    runtime_sec: float
    git_commit: str


@dataclass(frozen=True)
class GradRow:
    run_id: str
    seed: int
    step: int
    probe_split: str
    ref_policy: str
    approx_policy: str
    sketch_dim: int | None
    resample_sketch: bool
    grad_norm_ref: float
    grad_norm_approx: float
    grad_cos: float
    grad_relerr: float
    update_snr: float
    innovation_l2: float
    innovation_prob_kl: float
    loss_ref: float
    loss_approx: float
    clip_fraction_qkv: float
    clip_fraction_logits: float
    theta_norm: float


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _logit_stats(logits: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        mx = logits.float().abs().max().item()
        mn = logits.float().abs().mean().item()
    return mx, mn


def _is_phase_switch_policy(policy_name: str) -> tuple[bool, str]:
    if policy_name == "fp32_then_sketch_after_mem":
        return True, "fp32_to_sketch"
    if policy_name == "sketch_then_fp32_after_mem":
        return True, "sketch_to_fp32"
    return False, ""


def run_one(
    p: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    policy_name: str,
    base_policy: str,
    sketch_dim: int | None,
    resample_sketch: bool,
    correction_period: int | None,
    steps: int,
    batch_size: str,
    eval_every: int,
    diagnostics_every: int,
    lr: float,
    weight_decay: float,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_mlp: int,
    device: torch.device,
    architecture: str = "GrokTransformer",
    git_commit: str = "",
) -> tuple[list[CurveRow], list[GradRow]]:
    torch.manual_seed(seed)
    data = make_modular_addition_data(p, train_fraction, val_fraction, seed=seed, device="cpu")

    d_head = d_model // n_heads
    use_mlp = architecture == "MLP"

    if use_mlp:
        model = GrokMLP(
            vocab_size=data.vocab_size,
            d_model=d_model,
            d_hidden=d_mlp,
            n_hidden=2,
            max_seq_len=3,
        ).to(device)
    else:
        model = GrokTransformer(
            vocab_size=data.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_head=d_head,
            d_mlp=d_mlp,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    n_train = data.train_x.shape[0]
    full_batch = batch_size == "full"
    bsz = n_train if full_batch else int(batch_size)

    mem_step = -1
    grok_step = -1
    mem_reached = 0
    grok_reached = 0
    uniform_drop = float("nan")

    curve_rows: list[CurveRow] = []
    grad_rows: list[GradRow] = []
    t0 = time.perf_counter()
    batch_counter = 0

    run_id = f"{policy_name}_s{seed}"
    is_phase_switch, switch_direction = _is_phase_switch_policy(policy_name)
    phase_switched = False

    for step in range(steps):
        model.train()

        if full_batch:
            bx = data.train_x.to(device)
            by = data.train_y.to(device)
        else:
            idx = torch.randint(0, n_train, (bsz,), device="cpu")
            bx = data.train_x[idx].to(device)
            by = data.train_y[idx].to(device)

        use_correction = (
            correction_period is not None
            and batch_counter % correction_period == 0
        )

        if is_phase_switch and not phase_switched and mem_reached:
            phase_switched = True

        if is_phase_switch:
            if switch_direction == "fp32_to_sketch":
                cur_policy = "fp32" if not phase_switched else base_policy
                cur_sketch = None if not phase_switched else sketch_dim
            else:
                cur_policy = base_policy if not phase_switched else "fp32"
                cur_sketch = sketch_dim if not phase_switched else None
        elif use_correction:
            cur_policy = "fp32"
            cur_sketch = None
        else:
            cur_policy = base_policy
            cur_sketch = sketch_dim

        if use_mlp:
            logits = model(bx)
        else:
            logits, _ = model(
                bx,
                train_policy=cur_policy,
                sketch_dim=cur_sketch,
                resample_sketch=resample_sketch,
            )

        loss = loss_fn(logits, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_counter += 1

        if (step + 1) % eval_every != 0 and step != 0:
            continue

        model.eval()
        with torch.no_grad():
            if use_mlp:
                tr_logits = model(data.train_x.to(device))
                va_logits = model(data.val_x.to(device))
                te_logits = model(data.test_x.to(device))
                tr_attn = te_attn = None
            else:
                tr_logits, tr_attn = model(
                    data.train_x.to(device),
                    train_policy=base_policy,
                    sketch_dim=sketch_dim,
                    resample_sketch=resample_sketch,
                    return_attention=True,
                )
                va_logits, _ = model(
                    data.val_x.to(device),
                    train_policy=base_policy,
                    sketch_dim=sketch_dim,
                    resample_sketch=resample_sketch,
                )
                te_logits, te_attn = model(
                    data.test_x.to(device),
                    train_policy=base_policy,
                    sketch_dim=sketch_dim,
                    resample_sketch=resample_sketch,
                    return_attention=True,
                )

            tr_loss = loss_fn(tr_logits, data.train_y.to(device)).item()
            va_loss = loss_fn(va_logits, data.val_y.to(device)).item()
            te_loss = loss_fn(te_logits, data.test_y.to(device)).item()
            tr_acc = _accuracy(tr_logits, data.train_y.to(device))
            va_acc = _accuracy(va_logits, data.val_y.to(device))
            te_acc = _accuracy(te_logits, data.test_y.to(device))

        if tr_acc >= 0.99 and mem_reached == 0:
            mem_step = step + 1
            mem_reached = 1
        if te_acc >= 0.95 and grok_reached == 0:
            grok_step = step + 1
            grok_reached = 1

        delay = -1
        if grok_reached and mem_reached:
            delay = grok_step - mem_step

        ent_eq, mass_eq = _attn_metrics(te_attn)

        if math.isnan(uniform_drop) and grok_reached and not use_mlp:
            acc_n, acc_u = eval_uniform_attention(
                model,
                data.test_x.to(device),
                data.test_y.to(device),
                base_policy=base_policy,
                sketch_dim=sketch_dim,
            )
            uniform_drop = acc_n - acc_u

        fourier_metrics = compute_fourier_metrics(model, p, device)
        fourier_ratio = fourier_metrics.get("fourier_energy_ratio", float("nan"))

        max_logit, mean_logit_abs = _logit_stats(te_logits)

        elapsed = time.perf_counter() - t0

        curve_rows.append(CurveRow(
            run_id=run_id, seed=seed, split_seed=seed,
            p=p, train_fraction=train_fraction, val_fraction=val_fraction,
            architecture=architecture, d_model=d_model,
            n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp,
            optimizer="AdamW", lr=lr, weight_decay=weight_decay,
            batch_size=batch_size,
            train_policy=policy_name, eval_policy=base_policy,
            base_policy=base_policy, sketch_dim=sketch_dim,
            fixed_sketch=sketch_dim is not None and not resample_sketch,
            resample_sketch=resample_sketch,
            correction_period=correction_period,
            step=step + 1,
            train_loss=tr_loss, val_loss=va_loss, test_loss=te_loss,
            train_acc=tr_acc, val_acc=va_acc, test_acc=te_acc,
            memorization_reached=mem_reached, grokking_reached=grok_reached,
            memorization_step=mem_step, grokking_step=grok_step,
            grokking_delay=delay,
            attn_entropy_eq=ent_eq, attn_maxmass_eq=mass_eq,
            uniform_attn_eval_drop=uniform_drop,
            fourier_energy_ratio=fourier_ratio,
            max_logit=max_logit, mean_logit_abs=mean_logit_abs,
            runtime_sec=elapsed, git_commit=git_commit,
        ))

        if (step + 1) % diagnostics_every == 0 or step + 1 == steps:
            if use_mlp:
                continue
            probe_idx = torch.randint(0, n_train, (min(bsz, n_train),), device="cpu")
            probe_x = data.train_x[probe_idx].to(device)
            probe_y = data.train_y[probe_idx].to(device)
            try:
                barrier = compute_gradient_barrier_metrics(
                    model, probe_x, probe_y,
                    reference_policy="fp32",
                    approximate_policy=base_policy,
                    sketch_dim=sketch_dim,
                    loss_fn=loss_fn,
                )

                with torch.no_grad():
                    approx_logits, _ = model(
                        probe_x,
                        train_policy=base_policy,
                        sketch_dim=sketch_dim,
                        resample_sketch=resample_sketch,
                    )
                inn_kl = innovation_prob_kl(
                    logits_ref=model(probe_x, train_policy="fp32")[0].detach(),
                    logits_approx=approx_logits.detach(),
                )

                g_flat = torch.cat([p.data.detach().reshape(-1) for p in model.parameters()])
                theta_norm = g_flat.norm().item()

                grad_rows.append(GradRow(
                    run_id=run_id, seed=seed, step=step + 1,
                    probe_split="train", ref_policy="fp32",
                    approx_policy=base_policy, sketch_dim=sketch_dim,
                    resample_sketch=resample_sketch,
                    grad_norm_ref=barrier.grad_norm_ref,
                    grad_norm_approx=barrier.grad_norm_approx,
                    grad_cos=barrier.grad_cosine,
                    grad_relerr=barrier.grad_rel_error,
                    update_snr=barrier.update_snr,
                    innovation_l2=barrier.innovation_norm,
                    innovation_prob_kl=inn_kl,
                    loss_ref=barrier.loss_ref,
                    loss_approx=barrier.loss_approx,
                    clip_fraction_qkv=float("nan"),
                    clip_fraction_logits=float("nan"),
                    theta_norm=theta_norm,
                ))
            except Exception:
                pass

    return curve_rows, grad_rows


def monotonicity_audit(
    model: nn.Module,
    p: int,
    sketch_dims: Sequence[int],
    n_draws: int = 50,
    device: torch.device | str = "cpu",
) -> dict[int, dict[str, float]]:
    from grokking_model import GrokTransformer

    if not isinstance(model, GrokTransformer):
        return {}

    results: dict[int, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        x_probe = torch.arange(p, device=device).unsqueeze(1).expand(p, 2)

        q = model.layers[0].q_proj(model.layers[0].attn_norm(model.token_emb(x_probe) + model.pos_emb(torch.arange(2, device=device).unsqueeze(0).expand(p, -1))))
        k = model.layers[0].k_proj(model.layers[0].attn_norm(model.token_emb(x_probe) + model.pos_emb(torch.arange(2, device=device).unsqueeze(0).expand(p, -1))))

        B, T, _ = q.shape
        q = q.view(B, T, model.n_heads, model.d_head)[:, 0, :, :]
        k = k.view(B, T, model.n_heads, model.d_head)[:, 0, :, :]

        exact_logits = (q @ k.transpose(-1, -2)) / math.sqrt(model.d_head)

        for sd in sketch_dims:
            mses = []
            for _ in range(n_draws):
                S = torch.randn(model.d_head, sd, device=device) / math.sqrt(sd)
                q_s = q @ S
                k_s = k @ S
                approx_logits = (q_s @ k_s.transpose(-1, -2)) / math.sqrt(model.d_head)
                mse = (exact_logits - approx_logits).float().pow(2).mean().item()
                mses.append(mse)
            results[sd] = {
                "mean_mse": sum(mses) / len(mses),
                "min_mse": min(mses),
                "max_mse": max(mses),
            }
    return results


def write_csv(rows: list, path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def main() -> None:
    parser = argparse.ArgumentParser(description="Grokking transition experiment")
    parser.add_argument("--output-curves", type=Path, default=Path("results/grokking_curves.dev.csv"))
    parser.add_argument("--output-barrier", type=Path, default=Path("results/grokking_barrier.dev.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/grokking_summary.dev.csv"))
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=str, default="full")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--diagnostics-every", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--d-mlp", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--policies", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--search-baseline", action="store_true")
    parser.add_argument("--audit-monotonicity", action="store_true")
    parser.add_argument("--architecture", type=str, default="GrokTransformer", choices=["GrokTransformer", "MLP"])
    args = parser.parse_args()

    device = _device(args.device)
    git_commit = _git_commit()

    if args.search_baseline:
        print("[search] Searching for fp32 grokking cell...")
        lrs = [3e-4, 1e-3, 3e-3]
        wds = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
        fractions = [0.2, 0.3, 0.4, 0.5]
        depths = [1, 2]
        best = None
        for depth in depths:
            for frac in fractions:
                for lr in lrs:
                    for wd in wds:
                        print(f"  depth={depth} frac={frac} lr={lr} wd={wd}")
                        c, _ = run_one(
                            p=args.p, train_fraction=frac, val_fraction=args.val_fraction,
                            seed=0, policy_name="fp32", base_policy="fp32",
                            sketch_dim=None, resample_sketch=False, correction_period=None,
                            steps=min(args.steps, 20000), batch_size=args.batch_size,
                            eval_every=500, diagnostics_every=100000,
                            lr=lr, weight_decay=wd,
                            d_model=args.d_model, n_layers=depth,
                            n_heads=args.n_heads, d_mlp=args.d_mlp,
                            device=device, git_commit=git_commit,
                        )
                        if c:
                            last = c[-1]
                            if last.grokking_reached:
                                print(f"  -> GROKKED at step {last.grokking_step}")
                                if best is None or last.grokking_step < best.grokking_step:
                                    best = last
                            else:
                                print(f"  -> test_acc={last.test_acc:.3f}")
        if best:
            print(f"[search] Best cell: depth={best.n_layers} frac={best.train_fraction} lr={best.lr} wd={best.weight_decay} grok_step={best.grokking_step}")
        else:
            print("[search] No fp32 cell found that groks within horizon.")
        return

    if args.audit_monotonicity:
        print("[audit] Running monotonicity audit...")
        torch.manual_seed(0)
        data = make_modular_addition_data(args.p, args.train_fraction, args.val_fraction, seed=0, device=str(device))
        model = GrokTransformer(
            vocab_size=data.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_head=args.d_model // args.n_heads,
            d_mlp=args.d_mlp,
        ).to(device)
        results = monotonicity_audit(model, args.p, [4, 8, 16, 32], device=device)
        for sd, stats in results.items():
            print(f"  sketch_dim={sd}: mean_mse={stats['mean_mse']:.6f}")
        mses = [results[sd]["mean_mse"] for sd in sorted(results)]
        monotone = all(a >= b for a, b in zip(mses, mses[1:]))
        print(f"[audit] Monotonicity: {'PASS' if monotone else 'FAIL (discuss in report)'}")
        return

    if args.policies:
        policy_map = {p[0]: p for p in POLICY_TABLE}
        selected = [policy_map[name] for name in args.policies if name in policy_map]
    else:
        selected = POLICY_TABLE

    all_curves: list[CurveRow] = []
    all_grads: list[GradRow] = []

    for seed in args.seeds:
        for entry in selected:
            pol_name, base_pol, sk_dim, resample, corr_per = entry
            print(f"[grok] p={args.p} frac={args.train_fraction} seed={seed} policy={pol_name} arch={args.architecture}")
            c, g = run_one(
                p=args.p, train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                seed=seed, policy_name=pol_name, base_policy=base_pol,
                sketch_dim=sk_dim, resample_sketch=resample, correction_period=corr_per,
                steps=args.steps, batch_size=args.batch_size,
                eval_every=args.eval_every, diagnostics_every=args.diagnostics_every,
                lr=args.lr, weight_decay=args.weight_decay,
                d_model=args.d_model, n_layers=args.n_layers,
                n_heads=args.n_heads, d_mlp=args.d_mlp,
                device=device, architecture=args.architecture,
                git_commit=git_commit,
            )
            all_curves.extend(c)
            all_grads.extend(g)

    write_csv(all_curves, args.output_curves)
    print(f"Wrote {len(all_curves)} curve rows to {args.output_curves}")
    write_csv(all_grads, args.output_barrier)
    print(f"Wrote {len(all_grads)} gradient rows to {args.output_barrier}")

    from grokking_summary import build_summary
    summary = build_summary(all_curves, all_grads)
    write_csv(summary, args.output_summary)
    print(f"Wrote {len(summary)} summary rows to {args.output_summary}")


if __name__ == "__main__":
    main()
