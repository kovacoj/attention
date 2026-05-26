from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from attention import PrecisionConfig, attention_components, default_precision_policies
from jax_backend import attention_components_jax
from random_feature_experiment import ActivationSourceConfig, load_qkv


@dataclass(frozen=True)
class AttentionCase:
    label: str
    precision: PrecisionConfig


@dataclass(frozen=True)
class ExperimentResult:
    data_source: str
    n: int
    d: int
    seed: int
    case: str
    intrinsic_rank: int | None
    noise_std: float | None
    transformer_model: str | None
    transformer_layer: int | None
    transformer_head: int | None
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


def _hs_norm_np(x: np.ndarray) -> float:
    return float(np.linalg.norm(x.astype(np.float64)))


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
    source_config: ActivationSourceConfig,
    seed: int,
    device: torch.device,
    backend: str,
) -> ExperimentResult:
    if backend == "jax":
        start = time.perf_counter()
        logits_np, weights_np, output_np = attention_components_jax(q, k, v, precision=case.precision)
        runtime_ms = (time.perf_counter() - start) * 1000.0

        ref_logits_np = np.asarray(reference_logits.detach().cpu(), dtype=np.float64)
        ref_weights_np = np.asarray(reference_weights.detach().cpu(), dtype=np.float64)
        ref_output_np = np.asarray(reference_output.detach().cpu(), dtype=np.float64)
        rel_err_logits = _hs_norm_np(ref_logits_np - logits_np) / _hs_norm_np(ref_logits_np)
        rel_err_output = _hs_norm_np(ref_output_np - output_np) / _hs_norm_np(ref_output_np)
        row_err_probs = float(np.mean(np.sum(np.abs(ref_weights_np - weights_np), axis=-1)))
        row_err_logits = float(np.mean(np.max(np.abs(ref_logits_np - logits_np), axis=-1)))
    else:
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

    transformer_model = None
    transformer_layer = None
    transformer_head = None
    intrinsic_rank = None
    noise_std = None
    if source_config.source == "transformer":
        transformer_model = source_config.transformer_model
        transformer_layer = source_config.transformer_layer
        transformer_head = source_config.transformer_head
    if source_config.source == "low-rank":
        intrinsic_rank = source_config.intrinsic_rank
        noise_std = source_config.noise_std

    return ExperimentResult(
        data_source=source_config.source,
        n=q.shape[0],
        d=q.shape[1],
        seed=seed,
        case=case.label,
        intrinsic_rank=intrinsic_rank,
        noise_std=noise_std,
        transformer_model=transformer_model,
        transformer_layer=transformer_layer,
        transformer_head=transformer_head,
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
    source_config: ActivationSourceConfig,
    backend: str,
) -> list[ExperimentResult]:
    q, k, v = load_qkv(n, d, source_config=source_config, seed=seed, device=device)
    q = q.to(torch.float64)
    k = k.to(torch.float64)
    v = v.to(torch.float64)

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
                source_config=source_config,
                seed=seed,
                device=device,
                backend=backend,
            )
        )

    return results


def run_sweep(
    ns: list[int],
    ds: list[int],
    seeds: list[int],
    *,
    device: torch.device,
    source_config: ActivationSourceConfig,
    backend: str = "torch",
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    d_values = ds if source_config.source != "transformer" else [0]
    for n in ns:
        for d in d_values:
            for seed in seeds:
                results.extend(
                    run_single_experiment(
                        n,
                        d,
                        seed=seed,
                        device=device,
                        source_config=source_config,
                        backend=backend,
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
        "case, source, n, d, E_L, E_P, E_A, rho_softmax, ms",
    ]
    for r in results:
        lines.append(
            f"{r.case}, {r.data_source}, {r.n}, {r.d}, "
            f"{r.rel_err_logits_hs:.4e}, {r.row_err_probs_l1_mean:.4e}, "
            f"{r.rel_err_output_hs:.4e}, {r.softmax_amp_ratio:.4e}, "
            f"{r.runtime_ms:.2f}"
        )
    return "\n".join(lines)
