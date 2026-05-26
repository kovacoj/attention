from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from attention import PrecisionConfig, default_precisions


@dataclass(frozen=True)
class ActivationSourceConfig:
    source: str
    intrinsic_rank: int
    noise_std: float
    transformer_model: str
    transformer_text: str
    transformer_layer: int
    transformer_head: int


@dataclass(frozen=True)
class RandomFeatureCase:
    label: str
    precision: PrecisionConfig
    feature_dim: int


@dataclass(frozen=True)
class RandomFeatureExperimentResult:
    data_source: str
    n: int
    d: int
    m: int
    seed: int
    case: str
    intrinsic_rank: int | None
    noise_std: float | None
    transformer_model: str | None
    transformer_layer: int | None
    transformer_head: int | None
    rf_err_kernel_hs: float
    rf_err_weights_hs: float
    rf_err_output_hs: float
    fp_err_kernel_hs: float
    fp_err_weights_hs: float
    fp_err_output_hs: float
    total_err_kernel_hs: float
    total_err_weights_hs: float
    total_err_output_hs: float
    ref_kernel_hs: float
    ref_output_hs: float
    runtime_ms: float


def build_random_feature_cases(feature_dims: list[int]) -> list[RandomFeatureCase]:
    cases = []
    for precision in default_precisions():
        for feature_dim in feature_dims:
            cases.append(
                RandomFeatureCase(
                    label=f"{precision.label}_rf{feature_dim}",
                    precision=precision,
                    feature_dim=feature_dim,
                )
            )
    return cases


def orthogonal_random_feature_map(
    d_model: int,
    feature_dim: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build a block-orthogonal Gaussian projection matrix.

    This is closer to the positive orthogonal random features used in Performer
    than a plain iid Gaussian matrix while remaining small and self-contained.
    """

    columns: list[torch.Tensor] = []
    remaining = feature_dim
    while remaining > 0:
        block_width = min(d_model, remaining)
        gaussian_block = torch.randn(
            d_model,
            d_model,
            dtype=torch.float64,
            device=device,
            generator=generator,
        )
        orthogonal_block, _ = torch.linalg.qr(gaussian_block, mode="reduced")
        scale_block = torch.linalg.norm(gaussian_block, dim=0, keepdim=True)
        columns.append(orthogonal_block[:, :block_width] * scale_block[:, :block_width])
        remaining -= block_width
    return torch.cat(columns, dim=1)


def _generator_for(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device.type).manual_seed(seed)


def _hs_norm(x: torch.Tensor) -> float:
    return torch.linalg.norm(x.to(torch.float64)).item()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reference_precision() -> PrecisionConfig:
    return PrecisionConfig(
        label="reference_fp64",
        storage_dtype=torch.float64,
        accumulation_dtype=torch.float64,
        softmax_dtype=torch.float64,
    )


def _compute_dtype(precision: PrecisionConfig) -> torch.dtype:
    if precision.accumulation_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.float32
    return precision.accumulation_dtype


def _normalization_dtype(precision: PrecisionConfig) -> torch.dtype:
    if precision.softmax_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.float32
    return precision.softmax_dtype


def exact_softmax_attention_components(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    kernel = torch.exp(logits)
    weights = torch.softmax(logits, dim=-1)
    output = weights @ v
    return kernel, weights, output


def _performer_positive_features(
    x: torch.Tensor,
    feature_map: torch.Tensor,
    *,
    is_query: bool,
) -> torch.Tensor:
    projected = x @ feature_map
    squared_norm = torch.sum(x * x, dim=-1, keepdim=True) / 2.0
    if is_query:
        stabilizer = projected.amax(dim=-1, keepdim=True)
    else:
        stabilizer = projected.amax()
    features = torch.exp(projected - squared_norm - stabilizer)
    return (features + 1.0e-6) / math.sqrt(feature_map.shape[-1])


def random_feature_attention_components(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    precision: PrecisionConfig,
    feature_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate softmax attention with stabilized positive orthogonal features.

    The inputs and random feature matrix are quantized to `storage_dtype` first,
    then all arithmetic is carried out in `accumulation_dtype` (or float32 for fp8
    storage cases) so the prototype exposes storage, projection, and accumulation
    effects in one implementation path.
    """

    compute_dtype = _compute_dtype(precision)
    normalization_dtype = _normalization_dtype(precision)
    scale = q.shape[-1] ** -0.25

    q_compute = q.to(precision.storage_dtype).to(compute_dtype) * scale
    k_compute = k.to(precision.storage_dtype).to(compute_dtype) * scale
    v_compute = v.to(precision.storage_dtype).to(compute_dtype)
    feature_compute = feature_map.to(precision.storage_dtype).to(compute_dtype)

    q_features = _performer_positive_features(q_compute, feature_compute, is_query=True)
    k_features = _performer_positive_features(k_compute, feature_compute, is_query=False)
    kernel = q_features @ k_features.transpose(-1, -2)

    normalized_kernel = kernel.to(normalization_dtype)
    tiny = torch.finfo(normalized_kernel.dtype).tiny
    weights = normalized_kernel / normalized_kernel.sum(dim=-1, keepdim=True).clamp_min(tiny)
    output = weights.to(compute_dtype) @ v_compute
    return kernel, weights, output


def _gaussian_qkv(
    n: int,
    d: int,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = _generator_for(device, seed)
    q = torch.randn(n, d, dtype=torch.float64, device=device, generator=gen)
    k = torch.randn(n, d, dtype=torch.float64, device=device, generator=gen)
    v = torch.randn(n, d, dtype=torch.float64, device=device, generator=gen)
    return q, k, v


def _low_rank_qkv(
    n: int,
    d: int,
    *,
    rank: int,
    noise_std: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = _generator_for(device, seed)
    z = torch.randn(n, rank, dtype=torch.float64, device=device, generator=gen)
    a_q = torch.randn(rank, d, dtype=torch.float64, device=device, generator=gen) / math.sqrt(rank)
    a_k = torch.randn(rank, d, dtype=torch.float64, device=device, generator=gen) / math.sqrt(rank)
    a_v = torch.randn(rank, d, dtype=torch.float64, device=device, generator=gen) / math.sqrt(rank)
    q = z @ a_q
    k = z @ a_k
    v = z @ a_v
    if noise_std > 0.0:
        q = q + noise_std * torch.randn(n, d, dtype=torch.float64, device=device, generator=gen)
        k = k + noise_std * torch.randn(n, d, dtype=torch.float64, device=device, generator=gen)
        v = v + noise_std * torch.randn(n, d, dtype=torch.float64, device=device, generator=gen)
    return q, k, v


def _transformer_qkv(
    max_tokens: int,
    *,
    source_config: ActivationSourceConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformer activation source requires the 'transformers' package. "
            "Install project dependencies with uv sync first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(source_config.transformer_model)
    model = AutoModel.from_pretrained(source_config.transformer_model)
    model.eval()
    model.to(device)

    encoded = tokenizer(
        source_config.transformer_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, return_dict=True)

    if hasattr(model, "distilbert"):
        distilbert_model = model.distilbert
        layers = distilbert_model.transformer.layer
        attention = layers[source_config.transformer_layer].attention
        hidden_input = outputs.hidden_states[source_config.transformer_layer][0].to(torch.float64)
        q_full = attention.q_lin(hidden_input.to(attention.q_lin.weight.dtype)).to(torch.float64)
        k_full = attention.k_lin(hidden_input.to(attention.k_lin.weight.dtype)).to(torch.float64)
        v_full = attention.v_lin(hidden_input.to(attention.v_lin.weight.dtype)).to(torch.float64)
        head_count = attention.n_heads
        head_dim = attention.dim // head_count
    elif hasattr(model, "transformer") and hasattr(model, "embeddings"):
        layers = model.transformer.layer
        attention = layers[source_config.transformer_layer].attention
        hidden_input = outputs.hidden_states[source_config.transformer_layer][0].to(torch.float64)
        q_full = attention.q_lin(hidden_input.to(attention.q_lin.weight.dtype)).to(torch.float64)
        k_full = attention.k_lin(hidden_input.to(attention.k_lin.weight.dtype)).to(torch.float64)
        v_full = attention.v_lin(hidden_input.to(attention.v_lin.weight.dtype)).to(torch.float64)
        head_count = attention.n_heads
        head_dim = attention.dim // head_count
    elif hasattr(model, "bert"):
        bert_model = model.bert
        layers = bert_model.encoder.layer
        attention = layers[source_config.transformer_layer].attention.self
        hidden_input = outputs.hidden_states[source_config.transformer_layer][0].to(torch.float64)
        q_full = attention.query(hidden_input.to(attention.query.weight.dtype)).to(torch.float64)
        k_full = attention.key(hidden_input.to(attention.key.weight.dtype)).to(torch.float64)
        v_full = attention.value(hidden_input.to(attention.value.weight.dtype)).to(torch.float64)
        head_count = attention.num_attention_heads
        head_dim = attention.attention_head_size
    elif hasattr(model, "encoder") and hasattr(model, "embeddings"):
        layers = model.encoder.layer
        attention = layers[source_config.transformer_layer].attention.self
        hidden_input = outputs.hidden_states[source_config.transformer_layer][0].to(torch.float64)
        q_full = attention.query(hidden_input.to(attention.query.weight.dtype)).to(torch.float64)
        k_full = attention.key(hidden_input.to(attention.key.weight.dtype)).to(torch.float64)
        v_full = attention.value(hidden_input.to(attention.value.weight.dtype)).to(torch.float64)
        head_count = attention.num_attention_heads
        head_dim = attention.attention_head_size
    else:
        raise RuntimeError(
            "Transformer activation source currently supports DistilBERT and BERT-style models only."
        )

    if source_config.transformer_head >= head_count:
        raise ValueError(
            f"Requested head {source_config.transformer_head}, but model only has {head_count} heads."
        )

    sequence_length = q_full.shape[0]
    q = q_full.reshape(sequence_length, head_count, head_dim)[:, source_config.transformer_head, :]
    k = k_full.reshape(sequence_length, head_count, head_dim)[:, source_config.transformer_head, :]
    v = v_full.reshape(sequence_length, head_count, head_dim)[:, source_config.transformer_head, :]
    return q, k, v


def load_qkv(
    n: int,
    d: int,
    *,
    source_config: ActivationSourceConfig,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if source_config.source == "gaussian":
        return _gaussian_qkv(n, d, seed=seed, device=device)
    if source_config.source == "low-rank":
        return _low_rank_qkv(
            n,
            d,
            rank=source_config.intrinsic_rank,
            noise_std=source_config.noise_std,
            seed=seed,
            device=device,
        )
    if source_config.source == "transformer":
        return _transformer_qkv(n, source_config=source_config, device=device)
    raise ValueError(f"Unsupported data source '{source_config.source}'.")


def _run_case(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    reference_kernel: torch.Tensor,
    reference_weights: torch.Tensor,
    reference_output: torch.Tensor,
    rf_kernel: torch.Tensor,
    rf_weights: torch.Tensor,
    rf_output: torch.Tensor,
    *,
    case: RandomFeatureCase,
    feature_map: torch.Tensor,
    source_config: ActivationSourceConfig,
    seed: int,
    device: torch.device,
) -> RandomFeatureExperimentResult:
    _synchronize(device)
    start = time.perf_counter()
    kernel, weights, output = random_feature_attention_components(
        q,
        k,
        v,
        precision=case.precision,
        feature_map=feature_map,
    )
    _synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0

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

    return RandomFeatureExperimentResult(
        data_source=source_config.source,
        n=q.shape[0],
        d=q.shape[1],
        m=case.feature_dim,
        seed=seed,
        case=case.label,
        intrinsic_rank=intrinsic_rank,
        noise_std=noise_std,
        transformer_model=transformer_model,
        transformer_layer=transformer_layer,
        transformer_head=transformer_head,
        rf_err_kernel_hs=_hs_norm(reference_kernel - rf_kernel),
        rf_err_weights_hs=_hs_norm(reference_weights - rf_weights),
        rf_err_output_hs=_hs_norm(reference_output - rf_output),
        fp_err_kernel_hs=_hs_norm(rf_kernel - kernel),
        fp_err_weights_hs=_hs_norm(rf_weights - weights),
        fp_err_output_hs=_hs_norm(rf_output - output),
        total_err_kernel_hs=_hs_norm(reference_kernel - kernel),
        total_err_weights_hs=_hs_norm(reference_weights - weights),
        total_err_output_hs=_hs_norm(reference_output - output),
        ref_kernel_hs=_hs_norm(reference_kernel),
        ref_output_hs=_hs_norm(reference_output),
        runtime_ms=runtime_ms,
    )


def run_single_random_feature_experiment(
    n: int,
    d: int,
    *,
    seed: int,
    device: torch.device,
    feature_dims: list[int],
    source_config: ActivationSourceConfig,
) -> list[RandomFeatureExperimentResult]:
    feature_seed = 20_000 + 131 * seed

    q, k, v = load_qkv(n, d, source_config=source_config, seed=seed, device=device)
    q = q.to(torch.float64)
    k = k.to(torch.float64)
    v = v.to(torch.float64)

    reference_kernel, reference_weights, reference_output = exact_softmax_attention_components(q, k, v)

    fp64 = _reference_precision()
    feature_cache: dict[int, torch.Tensor] = {}
    rf_kernel_cache: dict[int, torch.Tensor] = {}
    rf_weights_cache: dict[int, torch.Tensor] = {}
    rf_output_cache: dict[int, torch.Tensor] = {}
    for m in feature_dims:
        feature_map = orthogonal_random_feature_map(
            q.shape[1],
            m,
            device=device,
            generator=_generator_for(device, feature_seed + m),
        )
        kernel, weights, output = random_feature_attention_components(
            q,
            k,
            v,
            precision=fp64,
            feature_map=feature_map,
        )
        feature_cache[m] = feature_map
        rf_kernel_cache[m] = kernel
        rf_weights_cache[m] = weights
        rf_output_cache[m] = output

    results = []
    for case in build_random_feature_cases(feature_dims):
        m = case.feature_dim
        results.append(
            _run_case(
                q,
                k,
                v,
                reference_kernel,
                reference_weights,
                reference_output,
                rf_kernel_cache[m],
                rf_weights_cache[m],
                rf_output_cache[m],
                case=case,
                feature_map=feature_cache[m],
                source_config=source_config,
                seed=seed,
                device=device,
            )
        )

    return results


def run_random_feature_sweep(
    ns: list[int],
    ds: list[int],
    feature_dims: list[int],
    seeds: list[int],
    *,
    device: torch.device,
    source_config: ActivationSourceConfig,
) -> list[RandomFeatureExperimentResult]:
    results: list[RandomFeatureExperimentResult] = []
    d_values = ds if source_config.source != "transformer" else [0]
    for n in ns:
        for d in d_values:
            for seed in seeds:
                results.extend(
                    run_single_random_feature_experiment(
                        n,
                        d,
                        seed=seed,
                        device=device,
                        feature_dims=feature_dims,
                        source_config=source_config,
                    )
                )
    return results


def summarize_random_feature_results(results: list[RandomFeatureExperimentResult]) -> str:
    lines = [
        "case, source, n, d, m, E_rf_out, E_fp_out, E_tot_out, ref_out, ms",
    ]
    for result in results:
        lines.append(
            f"{result.case}, {result.data_source}, {result.n}, {result.d}, {result.m}, "
            f"{result.rf_err_output_hs:.4e}, {result.fp_err_output_hs:.4e}, "
            f"{result.total_err_output_hs:.4e}, {result.ref_output_hs:.4e}, "
            f"{result.runtime_ms:.2f}"
        )
    return "\n".join(lines)
