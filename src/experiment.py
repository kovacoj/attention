from __future__ import annotations

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from attention import PrecisionConfig, attention_components, default_precisions, gaussian_sketch


@dataclass(frozen=True)
class AttentionCase:
    label: str
    precision: PrecisionConfig
    sketch_dim: int | None = None


@dataclass(frozen=True)
class ExperimentResult:
    sequence_length: int
    d_model: int
    seed: int
    case: str
    storage_dtype: str
    accumulation_dtype: str
    softmax_dtype: str
    sketch_dim: int | None
    sketch_flop_ratio: float
    logits_rel_frob: float
    weights_row_l1: float
    output_rel_frob: float
    runtime_ms: float


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def build_cases(sketch_dims: list[int]) -> list[AttentionCase]:
    cases = [AttentionCase(label="reference_fp64", precision=PrecisionConfig(
        label="reference_fp64",
        storage_dtype=torch.float64,
        accumulation_dtype=torch.float64,
        softmax_dtype=torch.float64,
    ))]

    for precision in default_precisions():
        cases.append(AttentionCase(label=precision.label, precision=precision))
        for sketch_dim in sketch_dims:
            cases.append(
                AttentionCase(
                    label=f"{precision.label}_sketch_{sketch_dim}",
                    precision=precision,
                    sketch_dim=sketch_dim,
                )
            )

    return cases


def _generator_for(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device.type).manual_seed(seed)


def _relative_frobenius_error(reference: torch.Tensor, approximate: torch.Tensor) -> float:
    numerator = torch.linalg.norm((approximate - reference).to(torch.float64))
    denominator = torch.linalg.norm(reference.to(torch.float64))
    if denominator.item() == 0.0:
        return numerator.item()
    return (numerator / denominator).item()


def _rowwise_l1_error(reference: torch.Tensor, approximate: torch.Tensor) -> float:
    return (approximate.to(torch.float64) - reference.to(torch.float64)).abs().sum(dim=-1).mean().item()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_case(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    reference_logits: torch.Tensor,
    reference_weights: torch.Tensor,
    reference_output: torch.Tensor,
    *,
    case: AttentionCase,
    sketch_seed: int,
    device: torch.device,
) -> ExperimentResult:
    sketch = None
    if case.sketch_dim is not None:
        sketch = gaussian_sketch(
            q.shape[-1],
            case.sketch_dim,
            device=device,
            generator=_generator_for(device, sketch_seed),
        )

    _synchronize(device)
    start = time.perf_counter()
    logits, weights, output = attention_components(
        q,
        k,
        v,
        precision=case.precision,
        sketch=sketch,
    )
    _synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    sketch_flop_ratio = 1.0
    if case.sketch_dim is not None:
        sketch_flop_ratio = case.sketch_dim / q.shape[-1]

    return ExperimentResult(
        sequence_length=q.shape[0],
        d_model=q.shape[1],
        seed=sketch_seed,
        case=case.label,
        storage_dtype=str(case.precision.storage_dtype).replace("torch.", ""),
        accumulation_dtype=str(case.precision.accumulation_dtype).replace("torch.", ""),
        softmax_dtype=str(case.precision.softmax_dtype).replace("torch.", ""),
        sketch_dim=case.sketch_dim,
        sketch_flop_ratio=sketch_flop_ratio,
        logits_rel_frob=_relative_frobenius_error(reference_logits, logits),
        weights_row_l1=_rowwise_l1_error(reference_weights, weights),
        output_rel_frob=_relative_frobenius_error(reference_output, output),
        runtime_ms=runtime_ms,
    )


def run_single_experiment(
    sequence_length: int,
    d_model: int,
    *,
    seed: int,
    device: torch.device,
    sketch_dims: list[int],
) -> list[ExperimentResult]:
    data_generator = _generator_for(device, seed)
    q = torch.randn(sequence_length, d_model, dtype=torch.float64, device=device, generator=data_generator)
    k = torch.randn(sequence_length, d_model, dtype=torch.float64, device=device, generator=data_generator)
    v = torch.randn(sequence_length, d_model, dtype=torch.float64, device=device, generator=data_generator)

    reference_precision = PrecisionConfig(
        label="reference_fp64",
        storage_dtype=torch.float64,
        accumulation_dtype=torch.float64,
        softmax_dtype=torch.float64,
    )
    reference_logits, reference_weights, reference_output = attention_components(
        q,
        k,
        v,
        precision=reference_precision,
    )

    results = []
    for case in build_cases(sketch_dims):
        results.append(
            _run_case(
                q,
                k,
                v,
                reference_logits,
                reference_weights,
                reference_output,
                case=case,
                sketch_seed=seed,
                device=device,
            )
        )
    return results


def run_sweep(
    sequence_lengths: list[int],
    d_models: list[int],
    sketch_dims: list[int],
    seeds: list[int],
    *,
    device: torch.device,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for sequence_length in sequence_lengths:
        for d_model in d_models:
            for seed in seeds:
                results.extend(
                    run_single_experiment(
                        sequence_length,
                        d_model,
                        seed=seed,
                        device=device,
                        sketch_dims=sketch_dims,
                    )
                )
    return results


def write_results(results: list[ExperimentResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def summarize_results(results: list[ExperimentResult]) -> str:
    lines = [
        "case, n, d, sketch_dim, logits_rel_frob, weights_row_l1, output_rel_frob, runtime_ms",
    ]
    for result in results:
        lines.append(
            f"{result.case}, {result.sequence_length}, {result.d_model}, {result.sketch_dim}, "
            f"{result.logits_rel_frob:.4e}, {result.weights_row_l1:.4e}, "
            f"{result.output_rel_frob:.4e}, {result.runtime_ms:.2f}"
        )
    return "\n".join(lines)
