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
