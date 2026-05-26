from __future__ import annotations

import math

import numpy as np
import torch

from attention import PrecisionConfig


def _jax_modules():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError(
            "The JAX backend requires the 'jax' package. Install JAX separately before using --backend jax."
        ) from exc

    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _jax_dtype(dtype: torch.dtype):
    _, jnp = _jax_modules()
    mapping = {
        torch.float64: jnp.float64,
        torch.float32: jnp.float32,
        torch.float16: jnp.float16,
        torch.bfloat16: jnp.bfloat16,
    }
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return jnp.float16
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype for JAX backend: {dtype}") from exc


def _matmul(a, b, *, preferred_dtype):
    jax, jnp = _jax_modules()
    try:
        return jnp.matmul(
            a,
            b,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=preferred_dtype,
        )
    except TypeError:
        return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def attention_components_jax(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    precision: PrecisionConfig,
):
    _, jnp = _jax_modules()

    q_np = np.asarray(q.detach().cpu(), dtype=np.float64)
    k_np = np.asarray(k.detach().cpu(), dtype=np.float64)
    v_np = np.asarray(v.detach().cpu(), dtype=np.float64)

    q_storage = jnp.asarray(q_np, dtype=_jax_dtype(precision.storage_dtype))
    k_storage = jnp.asarray(k_np, dtype=_jax_dtype(precision.storage_dtype))
    v_storage = jnp.asarray(v_np, dtype=_jax_dtype(precision.storage_dtype))

    qk_accum_dtype = _jax_dtype(precision.accumulation_dtype)
    logits_dtype = _jax_dtype(precision.logits_dtype or precision.accumulation_dtype)
    softmax_dtype = _jax_dtype(precision.softmax_dtype)
    pv_accum_dtype = _jax_dtype(precision.pv_accumulation_dtype or precision.accumulation_dtype)
    output_dtype = _jax_dtype(precision.output_dtype or precision.pv_accumulation_dtype or precision.accumulation_dtype)

    q_compute = q_storage.astype(qk_accum_dtype)
    k_compute = k_storage.astype(qk_accum_dtype)
    logits = _matmul(q_compute, jnp.swapaxes(k_compute, -1, -2), preferred_dtype=qk_accum_dtype) / math.sqrt(q_np.shape[-1])
    logits = logits.astype(logits_dtype)
    weights = jax.nn.softmax(logits.astype(softmax_dtype), axis=-1)

    v_compute = v_storage.astype(pv_accum_dtype)
    output = _matmul(weights.astype(pv_accum_dtype), v_compute, preferred_dtype=pv_accum_dtype).astype(output_dtype)

    return np.asarray(logits, dtype=np.float64), np.asarray(weights, dtype=np.float64), np.asarray(output, dtype=np.float64)


def apply_residual_stack_jax(
    state0: torch.Tensor,
    *,
    depth: int,
    residual_scale: float,
    precision: PrecisionConfig,
):
    _, jnp = _jax_modules()

    state_np = np.asarray(state0.detach().cpu(), dtype=np.float64)
    state = jnp.asarray(state_np, dtype=_jax_dtype(precision.storage_dtype))
    residual_dtype = _jax_dtype(precision.output_dtype or precision.accumulation_dtype)

    for _ in range(depth):
        state_torch = torch.from_numpy(np.asarray(state, dtype=np.float64))
        _, _, attn_out = attention_components_jax(
            state_torch,
            state_torch,
            state_torch,
            precision=precision,
        )
        state = (state.astype(residual_dtype) + residual_scale * jnp.asarray(attn_out, dtype=residual_dtype)).astype(
            _jax_dtype(precision.storage_dtype)
        )

    return np.asarray(state, dtype=np.float64)
