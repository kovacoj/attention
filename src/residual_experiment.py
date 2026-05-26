from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from attention import PrecisionConfig, attention_components, default_precision_policies
from jax_backend import apply_residual_stack_jax
from random_feature_experiment import ActivationSourceConfig, load_qkv


@dataclass(frozen=True)
class ResidualResult:
    data_source: str
    n: int
    d: int
    depth: int
    residual_scale: float
    seed: int
    case: str
    intrinsic_rank: int | None
    noise_std: float | None
    transformer_model: str | None
    transformer_layer: int | None
    transformer_head: int | None
    storage_dtype: str
    rel_err_state_hs: float
    ref_state_hs: float
    runtime_ms: float


def _generator_for(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device.type).manual_seed(seed)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hs_norm(x: torch.Tensor) -> float:
    return torch.linalg.norm(x.to(torch.float64)).item()


def _hs_norm_np(x: np.ndarray) -> float:
    return float(np.linalg.norm(x.astype(np.float64)))


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _apply_residual_stack(
    state0: torch.Tensor,
    *,
    depth: int,
    residual_scale: float,
    precision: PrecisionConfig,
) -> torch.Tensor:
    state = state0
    residual_dtype = precision.output_dtype or precision.accumulation_dtype
    if residual_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        residual_dtype = torch.float32

    for _ in range(depth):
        logits, weights, attn_out = attention_components(
            state,
            state,
            state,
            precision=precision,
        )
        del logits, weights
        state = (
            state.to(residual_dtype) + residual_scale * attn_out.to(residual_dtype)
        ).to(precision.storage_dtype)

    return state


def run_residual_stack_sweep(
    ns: list[int],
    ds: list[int],
    depths: list[int],
    seeds: list[int],
    *,
    source_config: ActivationSourceConfig,
    residual_scale: float | None,
    terminal_time: float,
    device: torch.device,
    backend: str = "torch",
) -> list[ResidualResult]:
    results: list[ResidualResult] = []
    policies = default_precision_policies()
    reference_policy = PrecisionConfig(
        label="reference_fp64",
        storage_dtype=torch.float64,
        accumulation_dtype=torch.float64,
        softmax_dtype=torch.float64,
        logits_dtype=torch.float64,
        pv_accumulation_dtype=torch.float64,
        output_dtype=torch.float64,
    )

    for n in ns:
        d_values = ds if source_config.source != "transformer" else [0]
        for d in d_values:
            for seed in seeds:
                q, _, _ = load_qkv(n, d, source_config=source_config, seed=seed, device=device)
                state0 = q.to(torch.float64)
                for depth in depths:
                    step = residual_scale if residual_scale is not None else terminal_time / depth
                    if backend == "jax":
                        reference_state_np = apply_residual_stack_jax(
                            state0,
                            depth=depth,
                            residual_scale=step,
                            precision=reference_policy,
                        )
                        ref_norm = _hs_norm_np(reference_state_np)
                    else:
                        reference_state = _apply_residual_stack(
                            state0,
                            depth=depth,
                            residual_scale=step,
                            precision=reference_policy,
                        )
                        ref_norm = _hs_norm(reference_state)
                    for policy in policies:
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
                        if backend == "jax":
                            start = time.perf_counter()
                            approx_state_np = apply_residual_stack_jax(
                                state0,
                                depth=depth,
                                residual_scale=step,
                                precision=policy,
                            )
                            runtime_ms = (time.perf_counter() - start) * 1000.0
                            rel_err = 0.0 if ref_norm == 0.0 else _hs_norm_np(reference_state_np - approx_state_np) / ref_norm
                        else:
                            _synchronize(device)
                            start = time.perf_counter()
                            approx_state = _apply_residual_stack(
                                state0,
                                depth=depth,
                                residual_scale=step,
                                precision=policy,
                            )
                            _synchronize(device)
                            runtime_ms = (time.perf_counter() - start) * 1000.0
                            rel_err = 0.0 if ref_norm == 0.0 else _hs_norm(reference_state - approx_state) / ref_norm
                        results.append(
                            ResidualResult(
                                data_source=source_config.source,
                                n=n,
                                d=state0.shape[1],
                                depth=depth,
                                residual_scale=step,
                                seed=seed,
                                case=policy.label,
                                intrinsic_rank=intrinsic_rank,
                                noise_std=noise_std,
                                transformer_model=transformer_model,
                                transformer_layer=transformer_layer,
                                transformer_head=transformer_head,
                                storage_dtype=_dtype_name(policy.storage_dtype),
                                rel_err_state_hs=rel_err,
                                ref_state_hs=ref_norm,
                                runtime_ms=runtime_ms,
                            )
                        )

    return results


def run_sweep(
    ns: list[int],
    ds: list[int],
    depths: list[int],
    seeds: list[int],
    *,
    source_config: ActivationSourceConfig,
    residual_scale: float | None,
    terminal_time: float,
    device: torch.device,
    backend: str = "torch",
) -> list[ResidualResult]:
    return run_residual_stack_sweep(
        ns,
        ds,
        depths,
        seeds,
        source_config=source_config,
        residual_scale=residual_scale,
        terminal_time=terminal_time,
        device=device,
        backend=backend,
    )


def write_results(results: list[ResidualResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def summarize_results(results: list[ResidualResult]) -> str:
    lines = ["case, source, n, d, depth, h, E_depth, ms"]
    for r in results:
        lines.append(
            f"{r.case}, {r.data_source}, {r.n}, {r.d}, {r.depth}, {r.residual_scale:.4e}, "
            f"{r.rel_err_state_hs:.4e}, {r.runtime_ms:.2f}"
        )
    return "\n".join(lines)
