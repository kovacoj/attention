from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment import choose_device, run_sweep, summarize_results, write_results


def _parse_int_list(raw: str) -> list[int]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece:
            values.append(int(piece))
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run mixed-precision sketched attention experiments.",
    )
    parser.add_argument("--sequence-lengths", default="64,128,256", help="Comma-separated sequence lengths.")
    parser.add_argument("--d-models", default="32,64", help="Comma-separated feature dimensions.")
    parser.add_argument("--sketch-dims", default="8,16,24,32", help="Comma-separated sketch dimensions.")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated random seeds.")
    parser.add_argument("--device", default="auto", help="Torch device name, e.g. auto, cpu, cuda.")
    parser.add_argument(
        "--output",
        default="results/attention_sweep.csv",
        help="CSV path for experiment results.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sequence_lengths = _parse_int_list(args.sequence_lengths)
    d_models = _parse_int_list(args.d_models)
    sketch_dims = _parse_int_list(args.sketch_dims)
    seeds = _parse_int_list(args.seeds)

    device = choose_device(args.device)
    results = run_sweep(
        sequence_lengths,
        d_models,
        sketch_dims,
        seeds,
        device=device,
    )
    write_results(results, Path(args.output))
    print(f"Wrote {len(results)} rows to {args.output} on device={device}.")
    print()
    print(summarize_results(results))


if __name__ == "__main__":
    main()
