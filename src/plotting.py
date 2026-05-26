from __future__ import annotations

from pathlib import Path
import csv
import shutil
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _ensure_styles(styles_dir: Path) -> None:
    # Prefer presentation style if LaTeX is available, otherwise use notebook style
    if not styles_dir.exists():
        return
    has_pdflatex = shutil.which("pdflatex") is not None or shutil.which("latex") is not None
    try:
        if has_pdflatex and (styles_dir / "presentation.mplstyle").exists():
            plt.style.use(str(styles_dir / "presentation.mplstyle"))
        elif (styles_dir / "notebook.mplstyle").exists():
            plt.style.use(str(styles_dir / "notebook.mplstyle"))
    except Exception:
        # If style application fails, fall back to default rc
        pass
    # Apply a few consistent tweaks to improve publication aesthetics
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 3.2),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "legend.frameon": False,
            "legend.fontsize": "small",
            "font.family": "serif",
        }
    )


def _get_field(row: dict, candidates: list[str], default=None):
    for c in candidates:
        if c in row and row[c] != "":
            return row[c]
    return default


def read_results_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def _group_means(rows: list[dict], key_field: str, metric_fields: list[str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        key = row[key_field]
        entry = grouped.setdefault(key, {metric: [] for metric in metric_fields})
        for metric in metric_fields:
            value = row.get(metric, "")
            if value != "":
                entry[metric].append(float(value))

    means: dict[str, dict[str, float]] = {}
    for key, metrics in grouped.items():
        means[key] = {
            metric: float(np.mean(values)) if values else float("nan")
            for metric, values in metrics.items()
        }
    return means


def plot_rf_error_vs_m(csv_path: Path, out_path: Path, styles_dir: Path | None = None, source_filter: str | None = None) -> None:
    """Plot random-feature rf and fp output errors vs feature dimension m.

    The CSV format can be either the compact summarizer format with columns
    [case, source, n, d, m, E_rf_out, E_fp_out, E_tot_out, ref_out, ms]
    or the full dataclass CSV produced by the experiments (with names like
    data_source, rf_err_output_hs, fp_err_output_hs, total_err_output_hs,
    ref_output_hs, runtime_ms).
    """
    if styles_dir is None:
        styles_dir = Path(__file__).resolve().parents[1] / "styles"
    _ensure_styles(styles_dir)

    rows = read_results_csv(csv_path)
    if not rows:
        raise RuntimeError(f"No rows in {csv_path}")

    # Group by (case, source) (optionally filter by source)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        case = _get_field(r, ["case", "label"], "")
        source = _get_field(r, ["source", "data_source"], "")
        if case is None or case == "":
            continue
        if source_filter is not None and source is not None and source_filter != source:
            continue
        key = (case.strip(), source.strip())
        groups.setdefault(key, []).append(r)

    # Explicit figsize/dpi to ensure consistent rendering across environments
    fig, ax = plt.subplots(figsize=plt.rcParams.get("figure.figsize"), dpi=plt.rcParams.get("figure.dpi"))
    colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", None)
    for i, ((case, source), items) in enumerate(sorted(groups.items())):
        m_vals = np.array([int(_get_field(it, ["m"], 0)) for it in items])
        rf_err = np.array([float(_get_field(it, ["E_rf_out", "rf_err_output_hs", "rf_err_output_hs"], 0.0)) for it in items])
        fp_err = np.array([float(_get_field(it, ["E_fp_out", "fp_err_output_hs", "fp_err_output_hs"], 0.0)) for it in items])

        unique_m = np.unique(m_vals)
        rf_mean = [rf_err[m_vals == um].mean() if (m_vals == um).any() else np.nan for um in unique_m]
        fp_mean = [fp_err[m_vals == um].mean() if (m_vals == um).any() else np.nan for um in unique_m]

        color = None if colors is None else colors[i % len(colors)]
        ax.loglog(unique_m, rf_mean, marker="o", label=f"{case} (rf)", color=color)
        ax.loglog(unique_m, fp_mean, marker="s", label=f"{case} (fp)", color=color, linestyle="--")

    ax.set_xlabel("Random-feature dimension $m$")
    ax.set_ylabel("HS error (output)")
    ax.set_title(f"Random-feature errors vs $m$ — {csv_path.name}")
    ax.grid(True, which="both", ls=":")
    # Place legend to the right of the axes to avoid overlapping the curves
    try:
        ax.legend(fontsize="small", ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    except Exception:
        ax.legend(fontsize="small", ncol=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=plt.rcParams.get("savefig.dpi"))


def plot_precision_policy_map(
    precision_csv: Path,
    residual_csv: Path,
    out_path: Path,
    styles_dir: Path | None = None,
) -> None:
    if styles_dir is None:
        styles_dir = Path(__file__).resolve().parents[1] / "styles"
    _ensure_styles(styles_dir)

    precision_rows = read_results_csv(precision_csv)
    residual_rows = read_results_csv(residual_csv)
    if not precision_rows or not residual_rows:
        raise RuntimeError("Need non-empty precision and residual CSVs for plotting.")

    precision_metrics = [
        "rel_err_logits_hs",
        "row_err_probs_l1_mean",
        "rel_err_output_hs",
        "softmax_amp_ratio",
    ]
    residual_metrics = ["rel_err_state_hs"]
    precision_means = _group_means(precision_rows, "case", precision_metrics)
    residual_means = _group_means(residual_rows, "case", residual_metrics)

    policy_order = [
        "fp32_reference",
        "bf16_safe",
        "fp16_safe",
        "bf16_low_logit",
        "fp16_low_logit",
        "bf16_low_softmax",
        "fp16_low_softmax",
        "bf16_low_value",
        "fp16_low_value",
    ]
    policy_labels = [
        "fp32 ref",
        "bf16 safe",
        "fp16 safe",
        "bf16 logits",
        "fp16 logits",
        "bf16 softmax",
        "fp16 softmax",
        "bf16 value",
        "fp16 value",
    ]
    metric_labels = [r"$E_L$", r"$E_P$", r"$E_A$", r"$\rho_{\mathrm{softmax}}$", r"$E_{\mathrm{depth}}$"]

    def rows_for_source(source: str) -> np.ndarray:
        matrix = np.full((len(policy_order), len(metric_labels)), np.nan, dtype=float)
        for i, policy in enumerate(policy_order):
            rvals = [float(row["rel_err_state_hs"]) for row in residual_rows if row["case"] == policy and row.get("data_source") == source]
            for j, metric in enumerate(precision_metrics):
                vals = [float(row[metric]) for row in precision_rows if row["case"] == policy and row.get("data_source") == source]
                if vals:
                    matrix[i, j] = float(np.mean(vals))
            if rvals:
                matrix[i, 4] = float(np.mean(rvals))
        return matrix

    sources = []
    for source in ["gaussian", "transformer"]:
        if any(row.get("data_source") == source for row in precision_rows) and any(
            row.get("data_source") == source for row in residual_rows
        ):
            sources.append(source)
    if not sources:
        raise RuntimeError("No overlapping data sources found between precision and residual CSVs.")

    fig, axes = plt.subplots(1, len(sources), figsize=(5.3 * len(sources), 4.5), dpi=plt.rcParams.get("figure.dpi"))
    if len(sources) == 1:
        axes = [axes]

    for ax, source in zip(axes, sources):
        raw = rows_for_source(source)
        safe = np.where(raw > 0.0, raw, np.nan)
        display = np.log10(safe)
        im = ax.imshow(display, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(metric_labels)), metric_labels)
        ax.set_yticks(range(len(policy_labels)), policy_labels)
        ax.set_title(source.title())
        for i in range(display.shape[0]):
            for j in range(display.shape[1]):
                value = raw[i, j]
                if np.isnan(value):
                    text = "-"
                else:
                    text = f"{value:.1e}" if value < 100 else f"{value:.1e}"
                ax.text(j, i, text, ha="center", va="center", color="white", fontsize=7)

    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label(r"$\log_{10}$ metric")
    fig.suptitle("Precision-placement map")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=plt.rcParams.get("savefig.dpi"))


def _main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 2:
        print("Usage: python -m src.plotting <input.csv> <output.pdf> [styles_dir] [source]")
        return 2
    command = argv[0]
    if command == "plot_rf_error_vs_m":
        csv_path = Path(argv[1])
        out_path = Path(argv[2])
        styles_dir = Path(argv[3]) if len(argv) >= 4 else None
        source_filter = argv[4] if len(argv) >= 5 else None
        plot_rf_error_vs_m(csv_path, out_path, styles_dir, source_filter)
    elif command == "plot_precision_policy_map":
        precision_csv = Path(argv[1])
        residual_csv = Path(argv[2])
        out_path = Path(argv[3])
        styles_dir = Path(argv[4]) if len(argv) >= 5 else None
        plot_precision_policy_map(precision_csv, residual_csv, out_path, styles_dir)
    else:
        csv_path = Path(argv[0])
        out_path = Path(argv[1])
        styles_dir = Path(argv[2]) if len(argv) >= 3 else None
        source_filter = argv[3] if len(argv) >= 4 else None
        plot_rf_error_vs_m(csv_path, out_path, styles_dir, source_filter)
    print(f"Wrote plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
