from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from attention import (
    PrecisionConfig,
    attention_components,
    default_precision_policies,
)


@dataclass(frozen=True)
class TemperatureResult:
    n: int
    d: int
    seed: int
    beta: float
    policy: str
    normalized_entropy: float
    mean_max_mass: float
    rel_err_logits_hs: float
    row_err_probs_l1_mean: float
    rel_err_output_hs: float
    argmax_flip_rate: float
    quant_clip_rate: float
    ref_logits_hs: float
    ref_output_hs: float
    runtime_ms: float


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def symmetric_int8_quant_dequant(
    x: torch.Tensor,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, float, float]:
    max_abs = x.abs().max().item()
    if max_abs == 0.0:
        scale = 1.0
    elif scale is None:
        scale = max_abs / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127)
    x_hat = scale * q
    clipping_rate = (x.abs() / scale > 127).float().mean().item()
    return x_hat, scale, clipping_rate


def compute_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    n = Q.shape[0]
    d = Q.shape[1]
    L_ref = beta * (Q @ K.T) / math.sqrt(d)
    P_ref = torch.softmax(L_ref, dim=-1)
    A_ref = P_ref @ V
    ref_logits_hs = torch.norm(L_ref, p="fro").item()
    ref_output_hs = torch.norm(A_ref, p="fro").item()
    return L_ref, P_ref, A_ref, ref_logits_hs, ref_output_hs


def compute_entropy_metrics(P_ref: torch.Tensor) -> tuple[float, float]:
    n = P_ref.shape[0]
    eps = 1e-300
    P_clamped = P_ref.clamp_min(eps)
    row_entropy = -(P_ref * P_clamped.log()).sum(dim=-1)
    normalized_entropy = row_entropy.mean().item() / math.log(n)
    mean_max_mass = P_ref.max(dim=-1).values.mean().item()
    return normalized_entropy, mean_max_mass


def compute_errors(
    L_ref: torch.Tensor,
    P_ref: torch.Tensor,
    A_ref: torch.Tensor,
    L_hat: torch.Tensor | None,
    P_hat: torch.Tensor,
    A_hat: torch.Tensor,
) -> tuple[float, float, float, float]:
    rel_err_logits = 0.0
    if L_hat is not None:
        denom = torch.norm(L_ref, p="fro").item()
        if denom > 0:
            rel_err_logits = torch.norm(L_ref - L_hat, p="fro").item() / denom
    row_err_probs = (P_ref - P_hat).abs().sum(dim=-1).mean().item()
    denom_out = torch.norm(A_ref, p="fro").item()
    rel_err_output = 0.0
    if denom_out > 0:
        rel_err_output = torch.norm(A_ref - A_hat, p="fro").item() / denom_out
    flip = (P_ref.argmax(dim=-1) != P_hat.argmax(dim=-1)).float().mean().item()
    return rel_err_logits, row_err_probs, rel_err_output, flip


def build_temperature_policies() -> dict[str, PrecisionConfig | None]:
    policies: dict[str, PrecisionConfig | None] = {}
    for p in default_precision_policies():
        if p.label == "fp32_reference":
            continue
        policies[p.label] = p
    policies["int8_qkv_dynamic"] = None
    policies["int8_logits_dynamic"] = None
    policies["int8_logits_static_beta1"] = None
    return policies


def run_temperature_experiment(
    n: int,
    d: int,
    seeds: list[int],
    betas: list[float],
    output: Path,
    device: torch.device,
) -> None:
    results: list[TemperatureResult] = []
    policy_map = build_temperature_policies()

    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        Q = torch.randn(n, d, dtype=torch.float64, device=device, generator=generator)
        K = torch.randn(n, d, dtype=torch.float64, device=device, generator=generator)
        V = torch.randn(n, d, dtype=torch.float64, device=device, generator=generator)

        static_scale: float | None = None
        d_model = d
        L_beta1 = (Q @ K.T) / math.sqrt(d_model)
        max_abs_L1 = L_beta1.abs().max().item()
        if max_abs_L1 > 0:
            static_scale = max_abs_L1 / 127.0
        else:
            static_scale = 1.0

        for beta in betas:
            L_ref, P_ref, A_ref, ref_logits_hs, ref_output_hs = compute_reference(
                Q, K, V, beta
            )
            normalized_entropy, mean_max_mass = compute_entropy_metrics(P_ref)

            for policy_name, policy_cfg in policy_map.items():
                t0 = time.perf_counter()
                quant_clip_rate = 0.0

                if policy_cfg is not None:
                    logits_hat, P_hat, A_hat = attention_components(
                        Q, K, V, precision=policy_cfg, beta=beta
                    )
                    L_hat = logits_hat.to(torch.float64)
                    P_hat = P_hat.to(torch.float64)
                    A_hat = A_hat.to(torch.float64)
                elif policy_name == "int8_qkv_dynamic":
                    Q_hat, _, clip_q = symmetric_int8_quant_dequant(Q)
                    K_hat, _, clip_k = symmetric_int8_quant_dequant(K)
                    V_hat, _, clip_v = symmetric_int8_quant_dequant(V)
                    quant_clip_rate = max(clip_q, clip_k, clip_v)
                    L_hat = beta * (Q_hat @ K_hat.T) / math.sqrt(d_model)
                    P_hat = torch.softmax(L_hat.to(torch.float64), dim=-1)
                    A_hat = P_hat @ V_hat
                elif policy_name == "int8_logits_dynamic":
                    L_raw = beta * (Q @ K.T) / math.sqrt(d_model)
                    L_hat, _, clip_l = symmetric_int8_quant_dequant(L_raw)
                    quant_clip_rate = clip_l
                    P_hat = torch.softmax(L_hat.to(torch.float64), dim=-1)
                    A_hat = P_hat @ V
                elif policy_name == "int8_logits_static_beta1":
                    L_raw = beta * (Q @ K.T) / math.sqrt(d_model)
                    L_hat, _, clip_l = symmetric_int8_quant_dequant(
                        L_raw, scale=static_scale
                    )
                    quant_clip_rate = clip_l
                    P_hat = torch.softmax(L_hat.to(torch.float64), dim=-1)
                    A_hat = P_hat @ V
                else:
                    continue

                elapsed = (time.perf_counter() - t0) * 1000.0

                rel_err_logits, row_err_probs, rel_err_output, flip = compute_errors(
                    L_ref, P_ref, A_ref, L_hat, P_hat, A_hat
                )

                results.append(
                    TemperatureResult(
                        n=n,
                        d=d,
                        seed=seed,
                        beta=beta,
                        policy=policy_name,
                        normalized_entropy=normalized_entropy,
                        mean_max_mass=mean_max_mass,
                        rel_err_logits_hs=rel_err_logits,
                        row_err_probs_l1_mean=row_err_probs,
                        rel_err_output_hs=rel_err_output,
                        argmax_flip_rate=flip,
                        quant_clip_rate=quant_clip_rate,
                        ref_logits_hs=ref_logits_hs,
                        ref_output_hs=ref_output_hs,
                        runtime_ms=elapsed,
                    )
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"Wrote {len(results)} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temperature/entropy transition experiment for attention precision"
    )
    parser.add_argument("--output", type=Path, default=Path("results/attention_temperature_sweep.dev.csv"))
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    run_temperature_experiment(
        n=args.n,
        d=args.d,
        seeds=args.seeds,
        betas=args.betas,
        output=args.output,
        device=device,
    )


if __name__ == "__main__":
    main()
