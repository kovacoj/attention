from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import choose_device, run_sweep, summarize_results, write_results
from random_feature_experiment import (
    ActivationSourceConfig,
    run_random_feature_sweep,
    summarize_random_feature_results,
)
from plotting import plot_rf_error_vs_m


def _parse_int_list(raw: str) -> list[int]:
    return [int(p.strip()) for p in raw.split(",") if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mixed-precision attention approximation experiments.",
    )
    parser.add_argument(
        "--experiment",
        choices=("sketch", "random-features"),
        default="sketch",
    )
    parser.add_argument("--sequence-lengths", default="64,128,256")
    parser.add_argument("--d-models", default="32,64")
    parser.add_argument("--sketch-dims", default="8,16,24,32")
    parser.add_argument("--feature-dims", default="32,64,128,256")
    parser.add_argument(
        "--data-source",
        choices=("gaussian", "low-rank", "transformer"),
        default="gaussian",
    )
    parser.add_argument("--intrinsic-rank", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=1.0e-2)
    parser.add_argument("--transformer-model", default="distilbert-base-uncased")
    parser.add_argument(
        "--transformer-text",
        default=(
            "Transformers combine sequence modeling with learned attention, "
            "which makes them a natural target for studying approximation and finite precision."
        ),
    )
    parser.add_argument("--transformer-layer", type=int, default=0)
    parser.add_argument("--transformer-head", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default=None,
        help="Scratch output path; defaults depend on the selected experiment.",
    )
    args = parser.parse_args()

    device = choose_device(args.device)
    output = args.output
    if output is None:
        if args.experiment == "random-features":
            output = "results/random_feature_sweep.dev.csv"
        else:
            output = "results/attention_sweep.dev.csv"

    if args.experiment == "random-features":
        source_config = ActivationSourceConfig(
            source=args.data_source,
            intrinsic_rank=args.intrinsic_rank,
            noise_std=args.noise_std,
            transformer_model=args.transformer_model,
            transformer_text=args.transformer_text,
            transformer_layer=args.transformer_layer,
            transformer_head=args.transformer_head,
        )
        results = run_random_feature_sweep(
            _parse_int_list(args.sequence_lengths),
            _parse_int_list(args.d_models),
            _parse_int_list(args.feature_dims),
            _parse_int_list(args.seeds),
            device=device,
            source_config=source_config,
        )
        summary = summarize_random_feature_results(results)
        # Optionally generate a publication-quality plot if results were written
        try:
            out_path = Path(output)
            if out_path.exists():
                figures_dir = Path("figures")
                figures_dir.mkdir(exist_ok=True)
                plot_path = figures_dir / f"rf_errors_{out_path.stem}.pdf"
                plot_rf_error_vs_m(out_path, plot_path)
                print(f"Wrote plot to {plot_path}")
        except Exception as exc:  # keep the experiment robust
            print(f"Plot generation skipped or failed: {exc}")
    else:
        results = run_sweep(
            _parse_int_list(args.sequence_lengths),
            _parse_int_list(args.d_models),
            _parse_int_list(args.sketch_dims),
            _parse_int_list(args.seeds),
            device=device,
        )
        summary = summarize_results(results)

    write_results(results, Path(output))
    print(f"Wrote {len(results)} rows to {output} on device={device}.")
    print()
    print(summary)


if __name__ == "__main__":
    main()
