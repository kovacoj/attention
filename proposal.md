# Term Paper Proposal

## Title

Mixed-Precision Randomized Sketching for Scaled Dot-Product Attention

## Motivation

This project connects my diploma thesis topic, mixed-precision arithmetic and randomized numerical linear algebra (randNLA), with the machine learning themes developed in the lecture notes. The most natural deep learning target is the attention mechanism used in transformers. Attention is mathematically structured, computationally expensive, and closely related to matrix approximation, dimension reduction, and kernel-style constructions discussed in the course.

In particular, scaled dot-product attention requires forming the matrix

```math
A(Q,K,V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V,
```

where the main cost comes from the matrix product `QK^T` and the storage of the attention matrix. This makes attention a good setting for studying both randomized low-dimensional approximations and finite-precision effects.

## Main Question

How do randomized sketching methods and mixed-precision arithmetic interact when approximating scaled dot-product attention?

More concretely, I want to study whether a randomized sketch of the query and key matrices can reduce the attention cost while preserving accuracy, and whether mixed precision remains reliable once the sketching error is already present.

## Proposed Approach

I will focus on a single attention block rather than the full transformer network. This keeps the project mathematically clean and suitable for a short paper.

The main approximation will be a random projection sketch:

```math
Q \mapsto QS, \qquad K \mapsto KS,
```

where `S \in \mathbb{R}^{d \times s}` is a random sketching matrix with `s < d`. Then the full logit matrix

```math
QK^T
```

is approximated by

```math
QSS^T K^T.
```

This leads to an approximate attention operator

```math
\widetilde{A}(Q,K,V) = \operatorname{softmax}\left(\frac{QSS^T K^T}{\sqrt{d}}\right)V.
```

I will compare:

- full attention in high precision as a reference,
- sketched attention in standard precision,
- sketched attention in low precision,
- mixed-precision variants with low-precision products and higher-precision accumulation.

## Mathematical Perspective

The project is related to several topics from the lecture notes:

- `Lecture3`: dimension reduction and low-rank approximation,
- `Lecture9`: kernel and Gram-matrix viewpoints,
- `Lecture10`: transformer attention and sequence models,
- `Lecture11`: computational aspects and optimization.

The paper will use the following mathematical viewpoint:

1. The attention logits `QK^T` form a structured matrix whose approximation can be studied by randNLA tools.
2. The total approximation error can be decomposed into sketching error and finite-precision error.
3. The numerical experiments will test when mixed precision is acceptable relative to the approximation error already introduced by sketching.

## Numerical Experiments

The experiments will be small but systematic.

Typical settings:

- sequence length `n \in \{256, 512, 1024\}`,
- feature dimension `d \in \{64, 128\}`,
- sketch size `s < d`, varied across several values,
- precisions: `float64` reference, `float32`, `float16`, and possibly `bfloat16` if available.

Measured quantities:

- relative Frobenius error of the logit matrix,
- rowwise error of the attention weights after softmax,
- relative output error in the attention result,
- runtime and memory usage.

The main empirical goal is to identify regimes in which randomized sketching provides a useful reduction in cost and mixed precision adds little extra error.

## Expected Outcome

I expect the paper to show that:

- randomized sketching can reduce the dimension of the attention computation with controlled loss of accuracy,
- mixed precision with careful accumulation is significantly more stable than naive low-precision computation,
- for moderate sketch sizes, the sketching error may dominate the rounding error, making mixed precision practically attractive.

## Scope of the Paper

The project is intended as a short research-style term paper of about five pages. For this reason, I will not study full transformer training. Instead, I will analyze one mathematically transparent component: scaled dot-product attention.

This keeps the project aligned with the abstract mathematical orientation of the course while still including concrete numerical experiments.

## Provisional Structure

1. Introduction and motivation
2. Mathematical formulation of attention and its sketch approximation
3. Mixed-precision model and error discussion
4. Numerical experiments
5. Conclusion

## Short Summary

The proposed term paper studies a mathematically focused approximation problem for transformers: randomized sketching and mixed-precision computation for scaled dot-product attention. The goal is to combine ideas from randNLA, numerical linear algebra, and modern deep learning in a form that is both theoretically motivated and experimentally verifiable.
