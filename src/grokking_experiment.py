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
from grokking_data import make_modular_addition_data
from grokking_model import GrokTransformer


@dataclass(frozen=True)
class GrokkingRow:
    seed: int
    p: int
    train_fraction: float
    policy: str
    base_policy: str
    sketch_dim: int | None
    correction_period: int | None
    step: int
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float
    memorization_reached: int
    grokking_reached: int
    memorization_step: int
    grokking_step: int
    grokking_delay: int
    attn_entropy_norm: float
    attn_mean_max_mass: float
    grad_cosine_vs_fp32: float
    grad_rel_error_vs_fp32: float
    update_snr_vs_fp32: float
    runtime_sec: float


POLICIES = [
    ("fp32", "fp32", None, None),
    ("fp16_safe", "fp16_safe", None, None),
    ("bf16_safe", "bf16_safe", None, None),
    ("int8_qkv_dynamic", "int8_qkv_dynamic", None, None),
    ("int8_logits_dynamic", "int8_logits_dynamic", None, None),
    ("sketch_4", "sketch_4", 4, None),
    ("sketch_8", "sketch_8", 8, None),
    ("sketch_16", "sketch_16", 16, None),
    ("sketch_4_periodic_fp32_correction", "sketch_4", 4, 10),
    ("int8_logits_periodic_fp32_correction", "int8_logits_dynamic", None, 10),
]


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == targets).float().mean().item()


def _attn_metrics(weights: torch.Tensor | None) -> tuple[float, float]:
    if weights is None:
        return 0.0, 1.0
    eps = 1e-300
    w = weights.clamp_min(eps)
    B, H, T, _ = w.shape
    entropy = -(w * w.log()).sum(dim=-1).mean().item()
    norm_entropy = entropy / math.log(T) if T > 1 else 0.0
    mean_max = w.max(dim=-1).values.mean().item()
    return norm_entropy, mean_max


def _flatten_grads(parameters) -> torch.Tensor:
    parts = []
    for p in parameters:
        if p.grad is not None:
            parts.append(p.grad.detach().reshape(-1))
        else:
            parts.append(torch.zeros_like(p.data).reshape(-1))
    return torch.cat(parts)


def run_one(
    p: int,
    train_fraction: float,
    seed: int,
    policy_name: str,
    base_policy: str,
    sketch_dim: int | None,
    correction_period: int | None,
    steps: int,
    batch_size: int,
    eval_every: int,
    lr: float,
    weight_decay: float,
    d_model: int,
    n_layers: int,
    n_heads: int,
    mlp_hidden: int,
    device: torch.device,
) -> list[GrokkingRow]:
    torch.manual_seed(seed)
    data = make_modular_addition_data(p, train_fraction, seed=seed, device="cpu")

    model = GrokTransformer(
        vocab_size=data.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_hidden=mlp_hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    n_train = data.train_x.shape[0]
    memorization_step = -1
    grokking_step = -1
    mem_reached = 0
    grok_reached = 0

    rows: list[GrokkingRow] = []
    t0 = time.perf_counter()
    batch_counter = [0]

    for step in range(steps):
        model.train()

        idx = torch.randint(0, n_train, (batch_size,), device="cpu")
        bx = data.train_x[idx].to(device)
        by = data.train_y[idx].to(device)

        use_correction = False
        if correction_period is not None and batch_counter[0] % correction_period == 0:
            use_correction = True

        cur_policy = "fp32" if use_correction else base_policy
        cur_sketch = None if use_correction else sketch_dim

        logits, _ = model(bx, precision_policy=cur_policy, sketch_dim=cur_sketch)
        loss = loss_fn(logits, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_counter[0] += 1

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
            te_logits, te_attn = model(
                data.test_x.to(device),
                precision_policy=base_policy,
                sketch_dim=sketch_dim,
                return_attention=True,
            )
            tr_loss = loss_fn(tr_logits, data.train_y.to(device)).item()
            te_loss = loss_fn(te_logits, data.test_y.to(device)).item()
            tr_acc = _accuracy(tr_logits, data.train_y.to(device))
            te_acc = _accuracy(te_logits, data.test_y.to(device))

        if tr_acc >= 0.99 and mem_reached == 0:
            memorization_step = step + 1
            mem_reached = 1
        if te_acc >= 0.95 and grok_reached == 0:
            grokking_step = step + 1
            grok_reached = 1

        delay = -1
        if grok_reached and mem_reached:
            delay = grokking_step - memorization_step

        ent, mass = _attn_metrics(te_attn)

        grad_cos = -2.0
        grad_rel = -2.0
        grad_snr = -2.0
        if step + 1 == steps or (step + 1) % (eval_every * 5) == 0:
            probe_idx = torch.randint(0, n_train, (min(batch_size, n_train),), device="cpu")
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
                grad_cos = barrier.grad_cosine
                grad_rel = barrier.grad_rel_error
                grad_snr = barrier.update_snr
            except Exception:
                pass

        elapsed = time.perf_counter() - t0

        rows.append(GrokkingRow(
            seed=seed, p=p, train_fraction=train_fraction,
            policy=policy_name, base_policy=base_policy,
            sketch_dim=sketch_dim, correction_period=correction_period,
            step=step + 1,
            train_loss=tr_loss, test_loss=te_loss,
            train_acc=tr_acc, test_acc=te_acc,
            memorization_reached=mem_reached, grokking_reached=grok_reached,
            memorization_step=memorization_step, grokking_step=grokking_step,
            grokking_delay=delay,
            attn_entropy_norm=ent, attn_mean_max_mass=mass,
            grad_cosine_vs_fp32=grad_cos,
            grad_rel_error_vs_fp32=grad_rel,
            update_snr_vs_fp32=grad_snr,
            runtime_sec=elapsed,
        ))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Grokking transition experiment")
    parser.add_argument("--output", type=Path, default=Path("results/grokking_transition.dev.csv"))
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train-fraction", type=float, default=0.4)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--policies", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = _choose_device(args.device)

    if args.policies:
        policy_map = {p[0]: p for p in POLICIES}
        selected = [policy_map[name] for name in args.policies if name in policy_map]
    else:
        selected = POLICIES

    all_rows: list[GrokkingRow] = []
    for seed in args.seeds:
        for pol_name, base_pol, sk_dim, corr_per in selected:
            print(f"[grok] p={args.p} frac={args.train_fraction} seed={seed} policy={pol_name}")
            rows = run_one(
                p=args.p,
                train_fraction=args.train_fraction,
                seed=seed,
                policy_name=pol_name,
                base_policy=base_pol,
                sketch_dim=sk_dim,
                correction_period=corr_per,
                steps=args.steps,
                batch_size=args.batch_size,
                eval_every=args.eval_every,
                lr=args.lr,
                weight_decay=args.weight_decay,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                mlp_hidden=args.mlp_hidden,
                device=device,
            )
            all_rows.extend(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(all_rows[0]).keys()))
        writer.writeheader()
        for r in all_rows:
            writer.writerow(asdict(r))
    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
