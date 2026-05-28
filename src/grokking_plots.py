from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _plot_learning_curves(df: pd.DataFrame, outdir: Path) -> None:
    for policy in df["train_policy"].unique():
        sub = df[df["train_policy"] == policy]
        fig, ax = plt.subplots(figsize=(8, 5))
        for seed in sub["seed"].unique():
            s = sub[sub["seed"] == seed].sort_values("step")
            ax.plot(s["step"], s["train_acc"], alpha=0.5, color="blue")
            ax.plot(s["step"], s["test_acc"], alpha=0.5, color="red")
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Learning curves: {policy}")
        ax.legend(["train", "test"])
        fig.tight_layout()
        fig.savefig(outdir / f"curves_{policy}.png", dpi=150)
        plt.close(fig)


def _plot_survival(summary: pd.DataFrame, outdir: Path) -> None:
    if "t_grok_median" not in summary.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    policies = summary.sort_values("t_grok_median")
    x = range(len(policies))
    bars = ax.barh(x, policies["t_grok_median"].fillna(0), xerr=policies["censored_fraction"])
    ax.set_yticks(x)
    ax.set_yticklabels(policies["policy"])
    ax.set_xlabel("Median grokking step")
    ax.set_title("Grokking delay by policy")
    fig.tight_layout()
    fig.savefig(outdir / "survival_grok_step.png", dpi=150)
    plt.close(fig)


def _plot_grok_rate_heatmap(summary: pd.DataFrame, outdir: Path) -> None:
    if "sketch_dim" not in summary.columns or "grok_rate" not in summary.columns:
        return
    sketch_policies = summary[summary["sketch_dim"].notna()]
    if sketch_policies.empty:
        return
    pivot = sketch_policies.pivot_table(
        index="policy", columns="sketch_dim", values="grok_rate", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Grok rate by policy x sketch dim")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "heatmap_grok_rate.png", dpi=150)
    plt.close(fig)


def _plot_grad_cos_vs_accuracy(summary: pd.DataFrame, outdir: Path) -> None:
    if "grad_cos_final_mean" not in summary.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    valid = summary.dropna(subset=["grad_cos_final_mean", "final_test_acc_mean"])
    ax.scatter(valid["grad_cos_final_mean"], valid["final_test_acc_mean"])
    for _, row in valid.iterrows():
        ax.annotate(row["policy"], (row["grad_cos_final_mean"], row["final_test_acc_mean"]),
                    fontsize=7, alpha=0.8)
    ax.set_xlabel("Gradient cosine (final, mean)")
    ax.set_ylabel("Final test accuracy (mean)")
    ax.set_title("Gradient fidelity vs generalization")
    fig.tight_layout()
    fig.savefig(outdir / "grad_cos_vs_accuracy.png", dpi=150)
    plt.close(fig)


def _plot_fourier_energy(df: pd.DataFrame, outdir: Path) -> None:
    if "fourier_energy_ratio" not in df.columns:
        return
    valid = df.dropna(subset=["fourier_energy_ratio"])
    if valid.empty:
        return
    for policy in valid["train_policy"].unique():
        sub = valid[valid["train_policy"] == policy]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        for seed in sub["seed"].unique():
            s = sub[sub["seed"] == seed].sort_values("step")
            ax1.plot(s["step"], s["fourier_energy_ratio"], alpha=0.5, color="green")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Fourier energy ratio", color="green")
        ax1.set_title(f"Fourier progress: {policy}")
        fig.tight_layout()
        fig.savefig(outdir / f"fourier_{policy}.png", dpi=150)
        plt.close(fig)


def _plot_uniform_drop(summary: pd.DataFrame, outdir: Path) -> None:
    if "uniform_attn_drop_mean" not in summary.columns:
        return
    valid = summary.dropna(subset=["uniform_attn_drop_mean"])
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(valid)), valid["uniform_attn_drop_mean"])
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels(valid["policy"])
    ax.set_xlabel("Uniform attention eval drop")
    ax.set_title("Attention routing importance")
    fig.tight_layout()
    fig.savefig(outdir / "uniform_attn_drop.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grokking plots")
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--barrier", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    curves_df = pd.read_csv(args.curves) if args.curves.exists() else pd.DataFrame()
    barrier_df = pd.read_csv(args.barrier) if args.barrier.exists() else pd.DataFrame()
    summary_df = pd.read_csv(args.summary) if args.summary.exists() else pd.DataFrame()

    if not curves_df.empty:
        _plot_learning_curves(curves_df, args.outdir)
        _plot_fourier_energy(curves_df, args.outdir)

    if not summary_df.empty:
        _plot_survival(summary_df, args.outdir)
        _plot_grok_rate_heatmap(summary_df, args.outdir)
        _plot_grad_cos_vs_accuracy(summary_df, args.outdir)
        _plot_uniform_drop(summary_df, args.outdir)

    print(f"Plots saved to {args.outdir}")


if __name__ == "__main__":
    main()
