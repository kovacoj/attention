# Term Paper Proposal

## Title

Precision Placement in Scaled Dot-Product Attention

## Motivation

The current sketching-centered version of the project is mathematically clean, but its empirical regime is weak: on generic Gaussian data, the sketching error is so large that the experiment mostly studies rounding around an already poor approximation. A stronger direction is to study mixed precision directly inside scaled dot-product attention and ask which subcomputations can safely run in low precision.

For a single attention block,

```math
A(Q,K,V) = P V,
\qquad
P = \operatorname{softmax}_{\mathrm{row}}(L),
\qquad
L = \frac{QK^T}{\sqrt d}.
```

This pipeline contains several numerically distinct stages: storage of `Q,K,V`, the bilinear logit product `QK^T`, logit storage/quantization, rowwise softmax, the final value product `PV`, and output storage. The main question is not just whether `fp16` or `bf16` works globally, but where low precision is safe and where higher precision is still needed.

## Main Question

Which parts of scaled dot-product attention can be computed or stored in low precision without causing unacceptable error in the logits, the attention weights, or the final output?

More concretely, I want to compare precision policies of the form

```math
\pi = (\text{storage},\; \text{matmul accumulation},\; \text{logit storage},\; \text{softmax reduction},\; \text{value accumulation},\; \text{output storage}).
```

## Scope and Non-Goals

The paper will study deterministic forward kernels, not full training.

In scope:

- a single attention forward pass,
- an optional attention-only residual stack,
- controlled precision sweeps on synthetic or fixed activations.

Out of scope:

- optimizer state,
- FP32 master weights,
- loss scaling,
- end-to-end training stability claims.

Those training ideas are historically important in mixed-precision training, but they are not the main object of study here. The goal is forward-pass precision placement, not general mixed-precision training.

## Proposed Approach

I will model exact attention by

```math
L = \frac{QK^T}{\sqrt d},
\qquad
P = \operatorname{softmax}_{\mathrm{row}}(L),
\qquad
A = PV.
```

For each precision policy `\pi`, the implementation produces

```math
\widehat L_\pi,
\qquad
\widehat P_\pi,
\qquad
\widehat A_\pi.
```

The paper will compare policies such as:

- `fp32_reference`: fp32 storage and fp32 compute throughout,
- `bf16_safe`: bf16 storage for `Q,K,V`, fp32 accumulation for `QK^T` and `PV`, fp32 logits, fp32 softmax,
- `fp16_safe`: fp16 storage for `Q,K,V`, fp32 accumulation for `QK^T` and `PV`, fp32 logits, fp32 softmax,
- `low_logit`: low-precision storage for `Q,K,V`, fp32 accumulation for `QK^T`, then quantized logits before softmax,
- `low_softmax`: low-precision storage and logits, with the rowwise softmax itself also evaluated in low precision,
- `low_value`: low-precision storage for `V` and low-precision output storage, with fp32 logit and softmax stages.

The key point is that these are forward-kernel policies, not blanket dtype casts.

## Mathematical Perspective

The analysis will separate local perturbations by stage:

```math
\widehat A = A + E_{\mathrm{storage}} + E_{\mathrm{matmul}} + E_{\mathrm{logit}} + E_{\mathrm{softmax}} + E_{\mathrm{value}}.
```

This viewpoint is closer to numerical linear algebra and perturbation analysis than to training heuristics. The most important question is whether the softmax stage damps or amplifies a perturbation already present in the logits.

To keep the measurements interpretable, the norms will be fixed in advance.

### Primary metrics

Let `\ell_i` and `p_i` denote the `i`-th rows of `L` and `P`.

1. Relative logit error:

```math
E_L := \frac{\lVert L - \widehat L \rVert_{\mathrm{HS}}}{\lVert L \rVert_{\mathrm{HS}}}.
```

2. Mean rowwise softmax error:

```math
E_P := \frac1n \sum_{i=1}^n \lVert p_i - \widehat p_i \rVert_1.
```

3. Relative attention-output error:

```math
E_A := \frac{\lVert A - \widehat A \rVert_{\mathrm{HS}}}{\lVert A \rVert_{\mathrm{HS}}}.
```

4. Softmax amplification ratio:

```math
\rho_{\mathrm{softmax}}
=
\frac{\frac1n \sum_{i=1}^n \lVert p_i - \widehat p_i \rVert_1}{\frac1n \sum_{i=1}^n \lVert \ell_i - \widehat \ell_i \rVert_\infty}.
```

This ratio is not a universal constant; it is an empirical diagnostic that tests whether a given regime tends to damp or amplify rowwise logit perturbations.

### Optional depth-propagation metric

For an attention-only residual stack

```math
x_{\ell+1} = x_\ell + h F_\ell(x_\ell),
```

I will also measure the final representation error

```math
E_{\mathrm{depth}} := \frac{\lVert x_L - \widehat x_L \rVert_2}{\lVert x_L \rVert_2}.
```

This allows a clean local-to-global stability story without needing to study full training.

## Experimental Backend

JAX is a good experimental harness for this project because it exposes precision controls on dot products and matmuls, and it makes it easy to request higher-precision references by enabling X64.

However, JAX should be treated as an experimental backend, not as a semantic guarantee of every precision policy. In particular:

- `precision=` does not by itself fully determine hardware accumulation behavior,
- `preferred_element_type=` is a request to the backend, not a paper-level proof of exact accumulation semantics,
- the actual kernel behavior can depend on the device and backend.

Therefore each experiment must report its backend context explicitly:

- device type,
- JAX backend,
- input storage dtype,
- requested accumulation/result dtype,
- whether X64 was enabled for the reference.

Reference computations will use JAX X64 mode explicitly.

## FP8 Policy

FP8 experiments will only be reported if the quantization model is explicit.

I will not use vague language such as “fp8 if available or simulated.” Instead, an FP8 experiment must specify:

- the format, for example `E4M3` or `E5M2`,
- the scaling rule, for example symmetric per-tensor scaling,
- the rounding rule, for example round-to-nearest,
- the overflow rule, for example saturation to the largest finite value,
- the dequantization point, for example dequantize to fp32 before accumulation.

If native hardware FP8 is not available, the baseline FP8 study will use a software quantization model with one fixed format and one fixed scaling rule. Without that, FP8 conclusions are too ambiguous to be useful.

## Numerical Experiments

The experiments will be split into two regimes.

### Small-scale truth benchmarking

This regime is used for exact references in float64 or X64.

Typical settings:

- sequence length `n \in \{64, 128, 256\}`,
- feature dimension `d \in \{32, 64, 128\}`,
- precisions: fp32, bf16, fp16, and optional FP8 under a fixed software model,
- synthetic Gaussian data and optionally fixed transformer activations.

Measured quantities:

- `E_L`,
- `E_P`,
- `E_A`,
- `\rho_{\mathrm{softmax}}`,
- runtime.

### Larger-scale trend runs

This regime is used for stress tests and qualitative scaling trends when exact float64 references become expensive. In that case, the paper will distinguish clearly between:

- metrics that compare to an exact high-precision reference,
- metrics that compare policies against a practical fp32 baseline.

The point is to avoid pretending that a full exact reference is always available in the large-scale regime.

## Expected Outcome

I expect the paper to show the following qualitative picture.

- Low-precision storage of `Q,K,V` is often acceptable when the two matrix products accumulate in fp32.
- Logit quantization and low-precision softmax are the more sensitive locations.
- bf16 is typically safer than fp16 when dynamic range is the main issue.
- FP8 can only be discussed meaningfully under an explicit quantization model.
- A precision policy is more informative than a single global dtype label.

This is a stronger and more believable contribution than claiming that a sketch approximation plus mixed precision is good in a regime where the sketch itself is already poor.

## Relation to the Current Repo

The current sketching experiments are still useful as a negative baseline and as background motivation, but they should no longer be the main claim of the paper.

The main narrative should become:

1. exact attention as the reference pipeline,
2. explicit precision placement across the forward pass,
3. local perturbation diagnostics at the logits, softmax weights, and output,
4. optional depth-propagation experiments for residual stacks.

## Provisional Structure

1. Introduction and motivation
2. Attention pipeline and precision-policy model
3. Perturbation metrics and stability questions
4. Numerical experiments
5. Discussion of safe and unsafe precision locations
6. Conclusion

## Short Summary

The proposed paper studies precision placement in scaled dot-product attention rather than low-rank or sketch-based approximation as the main contribution. The central idea is to treat mixed precision as a structured policy over storage, accumulation, reduction, and output stages, and to evaluate that policy with clearly specified norms and explicitly defined low-precision models. JAX is useful here as an experimental harness, but the paper's claims will remain tied to measured backend behavior rather than assumed from the API alone.
