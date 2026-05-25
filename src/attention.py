from __future__ import annotations

import math
from dataclasses import dataclass

import torch


DTYPE_BY_NAME = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class PrecisionConfig:
    label: str
    storage_dtype: torch.dtype
    accumulation_dtype: torch.dtype
    softmax_dtype: torch.dtype


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return DTYPE_BY_NAME[name]
    except KeyError as exc:
        supported = ", ".join(sorted(DTYPE_BY_NAME))
        raise ValueError(f"Unsupported dtype '{name}'. Expected one of: {supported}") from exc


def gaussian_sketch(
    d_model: int,
    sketch_dim: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    sketch = torch.randn(
        d_model,
        sketch_dim,
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    return sketch / math.sqrt(sketch_dim)


def attention_components(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    precision: PrecisionConfig,
    sketch: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute attention with optional random projection sketching.

    `storage_dtype` models the precision used to store the projected tensors.
    `accumulation_dtype` models the precision used for matrix products.
    `softmax_dtype` lets us separate naive low-precision softmax from mixed-precision
    accumulation, which is one of the main effects we want to compare.
    """

    d_model = q.shape[-1]

    q_storage = q.to(precision.storage_dtype)
    k_storage = k.to(precision.storage_dtype)
    v_storage = v.to(precision.storage_dtype)

    if sketch is not None:
        sketch_storage = sketch.to(precision.storage_dtype)
        q_storage = q_storage @ sketch_storage
        k_storage = k_storage @ sketch_storage

    q_compute = q_storage.to(precision.accumulation_dtype)
    k_compute = k_storage.to(precision.accumulation_dtype)
    v_compute = v_storage.to(precision.accumulation_dtype)

    logits = (q_compute @ k_compute.transpose(-1, -2)) / math.sqrt(d_model)
    weights = torch.softmax(logits.to(precision.softmax_dtype), dim=-1)
    output = weights.to(precision.accumulation_dtype) @ v_compute
    return logits, weights, output


def default_precisions() -> list[PrecisionConfig]:
    return [
        PrecisionConfig(
            label="full_fp32",
            storage_dtype=torch.float32,
            accumulation_dtype=torch.float32,
            softmax_dtype=torch.float32,
        ),
        PrecisionConfig(
            label="full_fp16",
            storage_dtype=torch.float16,
            accumulation_dtype=torch.float16,
            softmax_dtype=torch.float16,
        ),
        PrecisionConfig(
            label="mixed_fp16",
            storage_dtype=torch.float16,
            accumulation_dtype=torch.float32,
            softmax_dtype=torch.float32,
        ),
    ]
