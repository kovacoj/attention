from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from attention import PrecisionConfig, attention_components, default_precision_policies


@dataclass(frozen=True)
class AttentionCase:
    label: str
    precision: PrecisionConfig


@dataclass(frozen=True)
class ExperimentResult:
    n: int
    d: int
    seed: int
    case: str
    storage_dtype: str
    qk_accum_dtype: str
    logits_dtype: str
    softmax_dtype: str
    pv_accum_dtype: str
    output_dtype: str
    rel_err_logits_hs: float
    row_err_probs_l1_mean: float
    rel_err_output_hs: float
    row_err_logits_linf_mean: float
    softmax_amp_ratio: float
    ref_logits_hs: float
    ref_output_hs: float
    runtime_ms: float


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def build_cases() -> list[AttentionCase]:
    return [
        AttentionCase(label=precision.label, precision=precision)
        for precision in default_precision_policies()
    ]


def _generator_for(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device.type).manual_seed(seed)


def _hs_norm(x: torch.Tensor) -> float:
    return torch.linalg.norm(x.to(torch.float64)).item()


def _mean_row_l1(x: torch.Tensor) -> float:
    return x.to(torch.float64).abs().sum(dim=-1).mean().item()


def _mean_row_linf(x: torch.Tensor) -> float:
    return x.to(torch.float64).abs().amax(dim=-1).mean().item()


def _rel_hs_error(reference: torch.Tensor, approx: torch.Tensor) -> float:
    denom = _hs_norm(reference)
    if denom == 0.0:
        return 0.0
    return _hs_norm(reference - approx) / denom


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


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
    seed: int,
    device: torch.device,
) -> ExperimentResult:
    _synchronize(device)
    start = time.perf_counter()
    logits, weights, output = attention_components(
        q,
        k,
        v,
        precision=case.precision,
    )
    _synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    rel_err_logits = _rel_hs_error(reference_logits, logits)
    rel_err_output = _rel_hs_error(reference_output, output)
    row_err_probs = _mean_row_l1(reference_weights - weights)
    row_err_logits = _mean_row_linf(reference_logits - logits)
    softmax_amp_ratio = row_err_probs / row_err_logits if row_err_logits > 0.0 else 0.0

    logits_dtype = case.precision.logits_dtype or case.precision.accumulation_dtype
    pv_accum_dtype = case.precision.pv_accumulation_dtype or case.precision.accumulation_dtype
    output_dtype = case.precision.output_dtype or pv_accum_dtype

    return ExperimentResult(
        n=q.shape[0],
        d=q.shape[1],
        seed=seed,
        case=case.label,
        storage_dtype=_dtype_name(case.precision.storage_dtype),
        qk_accum_dtype=_dtype_name(case.precision.accumulation_dtype),
        logits_dtype=_dtype_name(logits_dtype),
        softmax_dtype=_dtype_name(case.precision.softmax_dtype),
        pv_accum_dtype=_dtype_name(pv_accum_dtype),
        output_dtype=_dtype_name(output_dtype),
        rel_err_logits_hs=rel_err_logits,
        row_err_probs_l1_mean=row_err_probs,
        rel_err_output_hs=rel_err_output,
        row_err_logits_linf_mean=row_err_logits,
        softmax_amp_ratio=softmax_amp_ratio,
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
) -> list[ExperimentResult]:
    gen = _generator_for(device, seed)
    q = torch.randn(n, d, dtype=torch.float64, device=device, generator=gen)
    k = torch.randn(n, d, dtype=torch.float64, device=device, generator=gen)
    v = torch.randn(n, d, dtype=torch.float64, device=device, generator=gen)

    fp64 = PrecisionConfig(
        label="reference_fp64",
        storage_dtype=torch.float64,
        accumulation_dtype=torch.float64,
        softmax_dtype=torch.float64,
    )
    reference_logits, reference_weights, reference_output = attention_components(
        q, k, v, precision=fp64,
    )

    results = []
    for case in build_cases():
        results.append(
            _run_case(
                q,
                k,
                v,
                reference_logits,
                reference_weights,
                reference_output,
                case=case,
                seed=seed,
                device=device,
            )
        )

    return results


def run_sweep(
    ns: list[int],
    ds: list[int],
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
                        n, d, seed=seed, device=device,
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
        "case, n, d, E_L, E_P, E_A, rho_softmax, ms",
    ]
    for r in results:
        lines.append(
            f"{r.case}, {r.n}, {r.d}, "
            f"{r.rel_err_logits_hs:.4e}, {r.row_err_probs_l1_mean:.4e}, "
            f"{r.rel_err_output_hs:.4e}, {r.softmax_amp_ratio:.4e}, "
            f"{r.runtime_ms:.2f}"
        )
    return "\n".join(lines)
