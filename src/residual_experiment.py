from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from attention import PrecisionConfig, attention_components, default_precision_policies


@dataclass(frozen=True)
class ResidualCase:
    label: str
    precision: PrecisionConfig


@dataclass(frozen=True)
class ResidualExperimentResult:
    n: int
    d: int
    depth: int
    residual_scale: float
    seed: int
    case: str
    storage_dtype: str
    qk_accum_dtype: str
    logits_dtype: str
    softmax_dtype: str
    pv_accum_dtype: str
    output_dtype: str
    local_rel_err_logits_hs_mean: float
    local_row_err_probs_l1_mean: float
    local_rel_err_output_hs_mean: float
    local_row_err_logits_linf_mean: float
    local_softmax_amp_ratio_mean: float
    rel_depth_err_l2: float
    ref_final_l2: float
    runtime_ms: float


def build_cases() -> list[ResidualCase]:
    return [
        ResidualCase(label=precision.label, precision=precision)
        for precision in default_precision_policies()
    ]


def _reference_precision() -> PrecisionConfig:
    return PrecisionConfig(
        label="reference_fp64",
        storage_dtype=torch.float64,
        accumulation_dtype=torch.float64,
        softmax_dtype=torch.float64,
    )


def _generator_for(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device.type).manual_seed(seed)


def _hs_norm(x: torch.Tensor) -> float:
    return torch.linalg.norm(x.to(torch.float64)).item()


def _rel_hs_error(reference: torch.Tensor, approx: torch.Tensor) -> float:
    denom = _hs_norm(reference)
    if denom == 0.0:
        return 0.0
    return _hs_norm(reference - approx) / denom


def _mean_row_l1(x: torch.Tensor) -> float:
    return x.to(torch.float64).abs().sum(dim=-1).mean().item()


def _mean_row_linf(x: torch.Tensor) -> float:
    return x.to(torch.float64).abs().amax(dim=-1).mean().item()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample_stack(
    n: int,
    d: int,
    depth: int,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    gen = _generator_for(device, seed)
    x0 = torch.randn(n, d, dtype=torch.float64, device=device, generator=gen)
    scale = 1.0 / math.sqrt(d)
    layer_weights = []
    for _ in range(depth):
        w_q = torch.randn(d, d, dtype=torch.float64, device=device, generator=gen) * scale
        w_k = torch.randn(d, d, dtype=torch.float64, device=device, generator=gen) * scale
        w_v = torch.randn(d, d, dtype=torch.float64, device=device, generator=gen) * scale
        layer_weights.append((w_q, w_k, w_v))
    return x0, layer_weights


def _project_qkv(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return x @ w_q, x @ w_k, x @ w_v


def _run_case(
    x0: torch.Tensor,
    layer_weights: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    case: ResidualCase,
    depth: int,
    residual_scale: float,
    seed: int,
    device: torch.device,
) -> ResidualExperimentResult:
    reference_precision = _reference_precision()
    x_reference = x0.clone()
    x_approx = x0.clone()

    local_rel_err_logits: list[float] = []
    local_row_err_probs: list[float] = []
    local_rel_err_output: list[float] = []
    local_row_err_logits: list[float] = []
    local_softmax_amp: list[float] = []

    _synchronize(device)
    start = time.perf_counter()
    for w_q, w_k, w_v in layer_weights:
        q_ref, k_ref, v_ref = _project_qkv(x_reference, w_q, w_k, w_v)
        reference_logits, reference_weights, reference_output = attention_components(
            q_ref,
            k_ref,
            v_ref,
            precision=reference_precision,
        )
        local_logits, local_weights, local_output = attention_components(
            q_ref,
            k_ref,
            v_ref,
            precision=case.precision,
        )

        row_err_logits = _mean_row_linf(reference_logits - local_logits)
        row_err_probs = _mean_row_l1(reference_weights - local_weights)
        local_rel_err_logits.append(_rel_hs_error(reference_logits, local_logits))
        local_row_err_probs.append(row_err_probs)
        local_rel_err_output.append(_rel_hs_error(reference_output, local_output))
        local_row_err_logits.append(row_err_logits)
        local_softmax_amp.append(
            row_err_probs / row_err_logits if row_err_logits > 0.0 else 0.0
        )

        q_approx, k_approx, v_approx = _project_qkv(x_approx, w_q, w_k, w_v)
        _, _, approx_output = attention_components(
            q_approx,
            k_approx,
            v_approx,
            precision=case.precision,
        )

        # Keep the residual accumulator in fp64 so E_depth isolates attention-policy error.
        x_reference = x_reference + residual_scale * reference_output
        x_approx = x_approx + residual_scale * approx_output.to(torch.float64)

    _synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    logits_dtype = case.precision.logits_dtype or case.precision.accumulation_dtype
    pv_accum_dtype = case.precision.pv_accumulation_dtype or case.precision.accumulation_dtype
    output_dtype = case.precision.output_dtype or pv_accum_dtype

    return ResidualExperimentResult(
        n=x0.shape[0],
        d=x0.shape[1],
        depth=depth,
        residual_scale=residual_scale,
        seed=seed,
        case=case.label,
        storage_dtype=_dtype_name(case.precision.storage_dtype),
        qk_accum_dtype=_dtype_name(case.precision.accumulation_dtype),
        logits_dtype=_dtype_name(logits_dtype),
        softmax_dtype=_dtype_name(case.precision.softmax_dtype),
        pv_accum_dtype=_dtype_name(pv_accum_dtype),
        output_dtype=_dtype_name(output_dtype),
        local_rel_err_logits_hs_mean=_mean(local_rel_err_logits),
        local_row_err_probs_l1_mean=_mean(local_row_err_probs),
        local_rel_err_output_hs_mean=_mean(local_rel_err_output),
        local_row_err_logits_linf_mean=_mean(local_row_err_logits),
        local_softmax_amp_ratio_mean=_mean(local_softmax_amp),
        rel_depth_err_l2=_rel_hs_error(x_reference, x_approx),
        ref_final_l2=_hs_norm(x_reference),
        runtime_ms=runtime_ms,
    )


def run_single_experiment(
    n: int,
    d: int,
    depth: int,
    *,
    seed: int,
    device: torch.device,
    residual_scale: float,
) -> list[ResidualExperimentResult]:
    x0, layer_weights = _sample_stack(n, d, depth, seed=seed, device=device)
    return [
        _run_case(
            x0,
            layer_weights,
            case=case,
            depth=depth,
            residual_scale=residual_scale,
            seed=seed,
            device=device,
        )
        for case in build_cases()
    ]


def run_sweep(
    ns: list[int],
    ds: list[int],
    depths: list[int],
    seeds: list[int],
    *,
    device: torch.device,
    residual_scale: float | None,
    terminal_time: float,
) -> list[ResidualExperimentResult]:
    results: list[ResidualExperimentResult] = []
    for n in ns:
        for d in ds:
            for depth in depths:
                step_scale = residual_scale if residual_scale is not None else terminal_time / depth
                for seed in seeds:
                    results.extend(
                        run_single_experiment(
                            n,
                            d,
                            depth,
                            seed=seed,
                            device=device,
                            residual_scale=step_scale,
                        )
                    )
    return results


def write_results(results: list[ResidualExperimentResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def summarize_results(results: list[ResidualExperimentResult]) -> str:
    lines = [
        "case, n, d, depth, h, E_L, E_P, E_A, rho_softmax, E_depth, ms",
    ]
    for result in results:
        lines.append(
            f"{result.case}, {result.n}, {result.d}, {result.depth}, "
            f"{result.residual_scale:.4e}, {result.local_rel_err_logits_hs_mean:.4e}, "
            f"{result.local_row_err_probs_l1_mean:.4e}, {result.local_rel_err_output_hs_mean:.4e}, "
            f"{result.local_softmax_amp_ratio_mean:.4e}, {result.rel_depth_err_l2:.4e}, "
            f"{result.runtime_ms:.2f}"
        )
    return "\n".join(lines)
