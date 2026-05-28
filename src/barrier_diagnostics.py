from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GradientBarrierMetrics:
    loss_ref: float
    loss_approx: float
    loss_gap: float
    grad_norm_ref: float
    grad_norm_approx: float
    grad_diff_norm: float
    grad_rel_error: float
    grad_cosine: float
    update_snr: float
    output_rel_error: float
    innovation_norm: float


def _flatten_grads(parameters) -> torch.Tensor:
    parts = []
    for p in parameters:
        if p.grad is not None:
            parts.append(p.grad.detach().reshape(-1))
        else:
            parts.append(torch.zeros_like(p.data).reshape(-1))
    return torch.cat(parts)


def compute_gradient_barrier_metrics(
    model: nn.Module,
    batch_tokens: torch.Tensor,
    batch_targets: torch.Tensor,
    *,
    reference_policy: str = "fp32",
    approximate_policy: str = "fp16_safe",
    sketch_dim: int | None = None,
    loss_fn: nn.Module | None = None,
) -> GradientBarrierMetrics:
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    eps = 1e-12

    model.zero_grad()
    pred_ref, _ = model(batch_tokens, train_policy=reference_policy, sketch_dim=sketch_dim)
    loss_ref = loss_fn(pred_ref, batch_targets)
    loss_ref.backward()
    g_ref = _flatten_grads(model.parameters())
    grad_norm_ref = g_ref.norm().item()
    model.zero_grad()

    with torch.no_grad():
        pred_ref_det = pred_ref.detach()

    model.zero_grad()
    pred_approx, _ = model(batch_tokens, train_policy=approximate_policy, sketch_dim=sketch_dim)
    loss_approx = loss_fn(pred_approx, batch_targets)
    loss_approx.backward()
    g_approx = _flatten_grads(model.parameters())
    grad_norm_approx = g_approx.norm().item()
    model.zero_grad()

    grad_diff = g_approx - g_ref
    grad_diff_norm = grad_diff.norm().item()
    grad_rel_error = grad_diff_norm / (grad_norm_ref + eps)
    dot = torch.dot(g_ref, g_approx).item()
    grad_cosine = dot / (grad_norm_ref * grad_norm_approx + eps)
    update_snr = grad_norm_ref / (grad_diff_norm + eps)

    with torch.no_grad():
        output_diff = pred_approx.detach() - pred_ref_det
        innovation_norm = output_diff.norm().item()
        output_rel_error = innovation_norm / (pred_ref_det.norm().item() + eps)

    loss_gap = loss_approx.item() - loss_ref.item()

    return GradientBarrierMetrics(
        loss_ref=loss_ref.item(),
        loss_approx=loss_approx.item(),
        loss_gap=loss_gap,
        grad_norm_ref=grad_norm_ref,
        grad_norm_approx=grad_norm_approx,
        grad_diff_norm=grad_diff_norm,
        grad_rel_error=grad_rel_error,
        grad_cosine=grad_cosine,
        update_snr=update_snr,
        output_rel_error=output_rel_error,
        innovation_norm=innovation_norm,
    )
