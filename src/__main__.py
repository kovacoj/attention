from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import choose_device, run_sweep, summarize_results, write_results


def _parse_int_list(raw: str) -> list[int]:
    return [int(p.strip()) for p in raw.split(",") if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mixed-precision sketched attention experiments.",
    )
    parser.add_argument("--sequence-lengths", default="64,128,256")
    parser.add_argument("--d-models", default="32,64")
    parser.add_argument("--sketch-dims", default="8,16,24,32")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/attention_sweep.csv")
    args = parser.parse_args()

    device = choose_device(args.device)
    results = run_sweep(
        _parse_int_list(args.sequence_lengths),
        _parse_int_list(args.d_models),
        _parse_int_list(args.sketch_dims),
        _parse_int_list(args.seeds),
        device=device,
    )
    write_results(results, Path(args.output))
    print(f"Wrote {len(results)} rows to {args.output} on device={device}.")
    print()
    print(summarize_results(results))


main()
