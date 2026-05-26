from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from attention import (
    PrecisionConfig,
    attention_components,
    default_sketch_precisions,
    gaussian_sketch,
)


@dataclass(frozen=True)
class AttentionCase:
    label: str
    precision: PrecisionConfig
    sketch_dim: int | None = None


@dataclass(frozen=True)
class ExperimentResult:
    n: int
    d: int
    s: int | None
    seed: int
    case: str
    sketch_err_logits_hs: float
    sketch_err_output_hs: float
    fp_err_logits_hs: float
    fp_err_output_hs: float
    total_err_logits_hs: float
    total_err_output_hs: float
    ref_logits_hs: float
    ref_output_hs: float
    runtime_ms: float


def build_cases(sketch_dims: list[int]) -> list[AttentionCase]:
    cases = []
    for precision in default_sketch_precisions():
        for sketch_dim in sketch_dims:
            cases.append(
                AttentionCase(
                    label=f"{precision.label}_sk{sketch_dim}",
                    precision=precision,
                    sketch_dim=sketch_dim,
                )
            )
    return cases


def _generator_for(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device.type).manual_seed(seed)


def _hs_norm(x: torch.Tensor) -> float:
    return torch.linalg.norm(x.to(torch.float64)).item()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_case(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    reference_logits: torch.Tensor,
    reference_output: torch.Tensor,
    sketched_logits: torch.Tensor | None,
    sketched_output: torch.Tensor | None,
    *,
    case: AttentionCase,
    sketch: torch.Tensor | None,
    seed: int,
    device: torch.device,
) -> ExperimentResult:
    _synchronize(device)
    start = time.perf_counter()
    logits, _, output = attention_components(
        q,
        k,
        v,
        precision=case.precision,
        sketch=sketch,
    )
    _synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    sketch_err_logits = 0.0
    sketch_err_output = 0.0
    if sketched_logits is not None and sketched_output is not None:
        sketch_err_logits = _hs_norm(reference_logits - sketched_logits)
        sketch_err_output = _hs_norm(reference_output - sketched_output)

    fp_err_logits = 0.0
    fp_err_output = 0.0
    if sketched_logits is not None and sketched_output is not None:
        fp_err_logits = _hs_norm(sketched_logits - logits)
        fp_err_output = _hs_norm(sketched_output - output)

    total_err_logits = _hs_norm(reference_logits - logits)
    total_err_output = _hs_norm(reference_output - output)

    return ExperimentResult(
        n=q.shape[0],
        d=q.shape[1],
        s=case.sketch_dim,
        seed=seed,
        case=case.label,
        sketch_err_logits_hs=sketch_err_logits,
        sketch_err_output_hs=sketch_err_output,
        fp_err_logits_hs=fp_err_logits,
        fp_err_output_hs=fp_err_output,
        total_err_logits_hs=total_err_logits,
        total_err_output_hs=total_err_output,
        ref_logits_hs=_hs_norm(reference_logits),
        ref_output_hs=_hs_norm(reference_output),
        runtime_ms=runtime_ms,
    )


def run_single_experiment(
    n: int,
    d: int,
    *,
    seed: int,
    device: torch.device,
    sketch_dims: list[int],
) -> list[ExperimentResult]:
    data_seed = seed
    sketch_seed = 10_000 + 97 * seed

    gen = _generator_for(device, data_seed)
    q = torch.randn(n, d, dtype=torch.float64, device=device, generator=gen)
    k = torch.randn(n, d, dtype=torch.float64, device=device, generator=gen)
    v = torch.randn(n, d, dtype=torch.float64, device=device, generator=gen)

    fp64 = PrecisionConfig(
        label="reference_fp64",
        storage_dtype=torch.float64,
        accumulation_dtype=torch.float64,
        softmax_dtype=torch.float64,
    )
    reference_logits, _, reference_output = attention_components(
        q, k, v, precision=fp64,
    )

    sketch_cache: dict[int, torch.Tensor] = {}
    sketched_logits_cache: dict[int, torch.Tensor] = {}
    sketched_output_cache: dict[int, torch.Tensor] = {}
    for s in sketch_dims:
        sketch = gaussian_sketch(
            d, s, device=device, generator=_generator_for(device, sketch_seed + s),
        )
        sl, _, so = attention_components(q, k, v, precision=fp64, sketch=sketch)
        sketch_cache[s] = sketch
        sketched_logits_cache[s] = sl
        sketched_output_cache[s] = so

    results = []
    for case in build_cases(sketch_dims):
        s = case.sketch_dim
        sketched_l = sketched_logits_cache.get(s)
        sketched_o = sketched_output_cache.get(s)
        sketch = sketch_cache.get(s)
        results.append(
            _run_case(
                q,
                k,
                v,
                reference_logits,
                reference_output,
                sketched_l,
                sketched_o,
                case=case,
                sketch=sketch,
                seed=seed,
                device=device,
            )
        )

    return results


def run_sweep(
    ns: list[int],
    ds: list[int],
    sketch_dims: list[int],
    seeds: list[int],
    *,
    device: torch.device,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for n in ns:
        for d in ds:
            for seed in seeds:
                results.extend(
                    run_single_experiment(
                        n, d, seed=seed, device=device, sketch_dims=sketch_dims,
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
        "case, n, d, s, E_sk, E_fp, E_tot, ref, ms",
    ]
    for r in results:
        lines.append(
            f"{r.case}, {r.n}, {r.d}, {r.s}, "
            f"{r.sketch_err_logits_hs:.4e}, {r.fp_err_logits_hs:.4e}, "
            f"{r.total_err_logits_hs:.4e}, {r.ref_logits_hs:.4e}, "
            f"{r.runtime_ms:.2f}"
        )
    return "\n".join(lines)
