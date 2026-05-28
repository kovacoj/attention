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
from grokking_data import make_modular_addition_data
from grokking_model import GrokTransformer

POLICY_TABLE = [
    ("fp32", "fp32", None, None),
    ("bf16_safe", "bf16_safe", None, None),
    ("fp16_safe", "fp16_safe", None, None),
    ("int8_qkv_dynamic", "int8_qkv_dynamic", None, None),
    ("int8_logits_dynamic", "int8_logits_dynamic", None, None),
    ("sketch_4", "sketch_4", 4, None),
    ("sketch_8", "sketch_8", 8, None),
    ("sketch_16", "sketch_16", 16, None),
    ("sketch_32", "sketch_32", 32, None),
    ("sketch_4_periodic_fp32_correction", "sketch_4", 4, 10),
    ("int8_logits_periodic_fp32_correction", "int8_logits_dynamic", None, 10),
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
    grad_norm_ref: float
    grad_norm_approx: float
    grad_cos: float
    grad_relerr: float
    update_snr: float
    innovation_norm: float
    loss_ref: float
    loss_approx: float
    classifier_logit_norm_ref: float
    classifier_logit_norm_approx: float
    attention_logit_norm_ref: float
    attention_logit_norm_approx: float
    classifier_entropy_ref: float
    classifier_entropy_approx: float
    clip_fraction_qkv: float
    clip_fraction_logits: float
    theta_norm: float
    theta_grad_cos: float


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _classifier_metrics(model: GrokTransformer, x: torch.Tensor) -> dict:
    with torch.no_grad():
        logits, _ = model(x, precision_policy="fp32")
    norms = logits.norm(dim=-1).mean().item()
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.log().clamp_min(-1e30)).sum(dim=-1).mean().item()
    return {"logit_norm": norms, "entropy": entropy}


def _theta_cos(model: nn.Module, grad_vec: torch.Tensor) -> float:
    parts = []
    for p in model.parameters():
        parts.append(p.data.detach().reshape(-1))
    theta = torch.cat(parts)
    eps = 1e-12
    cos = torch.dot(theta, grad_vec).item() / (theta.norm().item() * grad_vec.norm().item() + eps)
    return cos


def run_one(
    p: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    policy_name: str,
    base_policy: str,
    sketch_dim: int | None,
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
    eval_policies: Sequence[str] = (),
    git_commit: str = "",
) -> tuple[list[CurveRow], list[GradRow]]:
    torch.manual_seed(seed)
    data = make_modular_addition_data(p, train_fraction, val_fraction, seed=seed, device="cpu")

    d_head = d_model // n_heads
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
        cur_policy = "fp32" if use_correction else base_policy
        cur_sketch = None if use_correction else sketch_dim

        logits, _ = model(bx, precision_policy=cur_policy, sketch_dim=cur_sketch)
        loss = loss_fn(logits, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_counter += 1

        if (step + 1) % eval_every != 0 and step != 0:
            continue

        model.eval()
        with torch.no_grad():
            tr_logits, tr_attn = model(
                data.train_x.to(device),
                precision_policy=base_policy,
                sketch_dim=sketch_dim,
                return_attention=True,
            )
            va_logits, _ = model(
                data.val_x.to(device),
                precision_policy=base_policy,
                sketch_dim=sketch_dim,
            )
            te_logits, te_attn = model(
                data.test_x.to(device),
                precision_policy=base_policy,
                sketch_dim=sketch_dim,
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

        if math.isnan(uniform_drop) and grok_reached:
            with torch.no_grad():
                te_u, _ = model(
                    data.test_x.to(device),
                    precision_policy=base_policy,
                    sketch_dim=sketch_dim,
                    force_uniform_attention=True,
                )
            te_u_acc = _accuracy(te_u, data.test_y.to(device))
            uniform_drop = te_acc - te_u_acc

        for evp in eval_policies:
            if evp == base_policy:
                continue
            with torch.no_grad():
                te_ev, _ = model(
                    data.test_x.to(device),
                    train_policy=base_policy,
                    eval_policy=evp,
                    sketch_dim=sketch_dim,
                )
                _ = _accuracy(te_ev, data.test_y.to(device))

        elapsed = time.perf_counter() - t0

        curve_rows.append(CurveRow(
            run_id=run_id, seed=seed, split_seed=seed,
            p=p, train_fraction=train_fraction, val_fraction=val_fraction,
            architecture="GrokTransformer", d_model=d_model,
            n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp,
            optimizer="AdamW", lr=lr, weight_decay=weight_decay,
            batch_size=batch_size,
            train_policy=policy_name, eval_policy=base_policy,
            base_policy=base_policy, sketch_dim=sketch_dim,
            fixed_sketch=sketch_dim is not None and not policy_name.startswith("resampled"),
            resample_sketch=policy_name.startswith("resampled"),
            correction_period=correction_period,
            step=step + 1,
            train_loss=tr_loss, val_loss=va_loss, test_loss=te_loss,
            train_acc=tr_acc, val_acc=va_acc, test_acc=te_acc,
            memorization_reached=mem_reached, grokking_reached=grok_reached,
            memorization_step=mem_step, grokking_step=grok_step,
            grokking_delay=delay,
            attn_entropy_eq=ent_eq, attn_maxmass_eq=mass_eq,
            uniform_attn_eval_drop=uniform_drop,
            runtime_sec=elapsed, git_commit=git_commit,
        ))

        if (step + 1) % diagnostics_every == 0 or step + 1 == steps:
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
                cm_ref = _classifier_metrics(model, probe_x)
                cm_approx: dict = {}
                with torch.no_grad():
                    al, _ = model(probe_x, precision_policy=base_policy, sketch_dim=sketch_dim)
                al_norm = al.detach().norm(dim=-1).mean().item()
                al_probs = torch.softmax(al.detach(), dim=-1)
                al_entropy = -(al_probs * al_probs.log().clamp_min(-1e30)).sum(dim=-1).mean().item()

                g_flat = torch.cat([p.data.detach().reshape(-1) for p in model.parameters()])
                theta_norm = g_flat.norm().item()

                grad_vec = torch.cat([
                    p.grad.detach().reshape(-1) if p.grad is not None
                    else torch.zeros_like(p.data).reshape(-1)
                    for p in model.parameters()
                ])

                grad_rows.append(GradRow(
                    run_id=run_id, seed=seed, step=step + 1,
                    probe_split="train", ref_policy="fp32",
                    approx_policy=base_policy, sketch_dim=sketch_dim,
                    grad_norm_ref=barrier.grad_norm_ref,
                    grad_norm_approx=barrier.grad_norm_approx,
                    grad_cos=barrier.grad_cosine,
                    grad_relerr=barrier.grad_rel_error,
                    update_snr=barrier.update_snr,
                    innovation_norm=barrier.innovation_norm,
                    loss_ref=barrier.loss_ref,
                    loss_approx=barrier.loss_approx,
                    classifier_logit_norm_ref=cm_ref.get("logit_norm", float("nan")),
                    classifier_logit_norm_approx=cm_approx.get("logit_norm", float("nan")),
                    attention_logit_norm_ref=float("nan"),
                    attention_logit_norm_approx=al_norm,
                    classifier_entropy_ref=cm_ref.get("entropy", float("nan")),
                    classifier_entropy_approx=cm_approx.get("entropy", float("nan")),
                    clip_fraction_qkv=float("nan"),
                    clip_fraction_logits=float("nan"),
                    theta_norm=theta_norm,
                    theta_grad_cos=_theta_cos(model, grad_vec),
                ))
            except Exception:
                pass

    return curve_rows, grad_rows


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
    parser.add_argument("--output-gradbarrier", type=Path, default=Path("results/grokking_gradbarrier.dev.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/grokking_summary.dev.csv"))
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train-fraction", type=float, default=0.4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=str, default="full")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--diagnostics-every", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--policies", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--search-baseline", action="store_true")
    args = parser.parse_args()

    device = _device(args.device)
    git_commit = _git_commit()

    if args.search_baseline:
        print("[search] Searching for fp32 grokking cell...")
        lrs = [3e-4, 1e-3, 3e-3]
        wds = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
        fractions = [0.2, 0.3, 0.4]
        best = None
        for frac in fractions:
            for lr in lrs:
                for wd in wds:
                    print(f"  frac={frac} lr={lr} wd={wd}")
                    c, _ = run_one(
                        p=args.p, train_fraction=frac, val_fraction=args.val_fraction,
                        seed=0, policy_name="fp32", base_policy="fp32",
                        sketch_dim=None, correction_period=None,
                        steps=min(args.steps, 20000), batch_size=args.batch_size,
                        eval_every=500, diagnostics_every=100000,
                        lr=lr, weight_decay=wd,
                        d_model=args.d_model, n_layers=args.n_layers,
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
            print(f"[search] Best cell: frac={best.train_fraction} lr={best.lr} wd={best.weight_decay} grok_step={best.grokking_step}")
        else:
            print("[search] No fp32 cell found that groks within horizon.")
        return

    if args.policies:
        policy_map = {p[0]: p for p in POLICY_TABLE}
        selected = [policy_map[name] for name in args.policies if name in policy_map]
    else:
        selected = POLICY_TABLE

    all_curves: list[CurveRow] = []
    all_grads: list[GradRow] = []

    for seed in args.seeds:
        for pol_name, base_pol, sk_dim, corr_per in selected:
            print(f"[grok] p={args.p} frac={args.train_fraction} seed={seed} policy={pol_name}")
            c, g = run_one(
                p=args.p, train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                seed=seed, policy_name=pol_name, base_policy=base_pol,
                sketch_dim=sk_dim, correction_period=corr_per,
                steps=args.steps, batch_size=args.batch_size,
                eval_every=args.eval_every, diagnostics_every=args.diagnostics_every,
                lr=args.lr, weight_decay=args.weight_decay,
                d_model=args.d_model, n_layers=args.n_layers,
                n_heads=args.n_heads, d_mlp=args.d_mlp,
                device=device, git_commit=git_commit,
            )
            all_curves.extend(c)
            all_grads.extend(g)

    write_csv(all_curves, args.output_curves)
    print(f"Wrote {len(all_curves)} curve rows to {args.output_curves}")
    write_csv(all_grads, args.output_gradbarrier)
    print(f"Wrote {len(all_grads)} gradient rows to {args.output_gradbarrier}")

    from grokking_summary import build_summary
    summary = build_summary(all_curves, all_grads)
    write_csv(summary, args.output_summary)
    print(f"Wrote {len(summary)} summary rows to {args.output_summary}")


if __name__ == "__main__":
    main()
