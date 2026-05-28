# RandNLA Sketching as Regularization: Research Notes

## Core hypothesis

RandNLA feature sketching in attention may function as an implicit regularizer, analogous to dropout, providing both computational savings **and** generalization benefits. The mechanism is different from dropout, but the effect—structured noise injection during training that suppresses overfitting—could be similar.

---

## 1. Structural analogy to dropout

| Property | Dropout | RandNLA sketch |
|----------|---------|----------------|
| Where applied | Activations / hidden units | Attention logits (QK^T → (QS)(KS)^T) |
| Mechanism | Zero out random units | Project Q,K onto random low-d subspace |
| Noise type | Binary mask (on/off) | Structured rank reduction |
| Train/eval gap | Yes — must scale at eval | Yes — can eval with full or sketched attention |
| Gradient effect | Blocks gradient through dropped units | Distorts gradient direction (measurable via cosine) |
| Regularization signal | "Don't rely on any one unit" | "Don't rely on fine-grained attention structure" |
| Computational benefit | Minor (sparse ops) | Major (O(ns) vs O(nd) for s ≪ d) |

Key difference: dropout destroys information; sketching compresses it. Sketching retains a low-rank approximation of the attention structure rather than randomly discarding entries. This means the regularization is **spectral** rather than **spatial**.

## 2. Why sketching might regularize

### 2a. Implicit rank constraint

Sketching QK^T ≈ (QS)(KS)^T forces the logit matrix to have rank ≤ s. During training, the model cannot learn attention patterns that require full-rank structure. This is a soft inductive bias toward low-rank attention, which:

- Prevents the model from memorizing idiosyncratic pairwise relationships
- Pushes the model toward algorithmic/circuit solutions that are inherently low-rank (e.g., modular addition uses circular Fourier structure, which is low-rank in the token embedding space)

### 2b. Noise injection in gradient signal

The sketch distorts the gradient: g_approx ≠ g_fp32. When this distortion is moderate (update SNR > 1 but not >> 1), it acts like gradient noise, which:

- Helps escape sharp minima associated with memorization
- Flattens the loss landscape around the generalizing circuit
- Is consistent with the "gradient noise as regularizer" literature (Smith & Le, 2018; Neelakantan et al., 2015)

Our gradient diagnostics already measure this: grad_cosine ≈ 0.5–0.7 for partial/unreliable policies, ≈ 1.0 for preserved policies.

### 2c. Analogy to Monte Carlo dropout

If the sketch is resampled each forward pass (stochastic sketch), it becomes directly analogous to Monte Carlo dropout:

- Each training step sees a different random projection
- The model must learn representations robust to projection variation
- At eval, averaging over multiple sketches gives a variance-reduced estimate

This is explicitly different from fixed-sketch training, where the same projection is used throughout. Our pilot data already hints at this: resampled sketches occasionally succeeded where fixed sketches failed.

## 3. Experimental tests

### Test 1: Sketch-dimension sweep as regularization sweep

If sketching regularizes, there should be an optimal sketch dimension s* that balances:

- Too small s: underfitting / capacity barrier (our sketch_16 result)
- Too large s: no regularization benefit, just wasted compute
- Intermediate s: best generalization (like optimal dropout rate)

**Experiment**: Sweep s ∈ {2, 4, 6, 8, 10, 12, 16, 20, 24, 32} with 5 seeds each. Plot grokking rate and final test accuracy vs s. Look for an inverted-U curve.

### Test 2: Compare sketch regularization to dropout regularization

Train the same model with:
- (a) No regularization
- (b) Dropout p ∈ {0.1, 0.3, 0.5}
- (c) Fixed sketch s ∈ {4, 8, 16}
- (d) Resampled sketch s ∈ {4, 8, 16}

If sketching regularizes like dropout, conditions (b) and (c)/(d) should show similar grokking acceleration. If the mechanism is different, they may complement each other.

### Test 3: Combine sketch + dropout

If the mechanisms are orthogonal (spectral vs spatial), combining them should give additive or super-additive benefits. If they overlap, combining them should show diminishing returns.

**Experiment**: 2×2 grid {no-sketch, sketch-8} × {no-dropout, dropout-0.3}.

### Test 4: Double descent check

If sketching acts as explicit rank regularization, it might eliminate the double-descent peak at the interpolation threshold. This would be strong evidence for the regularization interpretation.

**Experiment**: Vary train fraction near interpolation boundary (p=97 → n=9409, so interpolation at train_fraction ≈ 1.0). Compare test accuracy curves with and without sketching.

### Test 5: Train/eval sketch mismatch as probe

Train with sketch s_train, eval with sketch s_eval. If the model has learned a representation adapted to the sketch bandwidth, then:

- s_eval = s_train: best performance
- s_eval > s_train: slight improvement (more attention resolution available)
- s_eval < s_train: degradation (representation relies on bandwidth that's no longer there)

This is analogous to the dropout train/eval mismatch studies.

## 4. Theoretical framing

### 4a. Connection to explicit rank regularization

Sketching QK^T through S ∈ R^{d×s} is equivalent to adding a hard rank-s constraint on the logit matrix. This is related to:

- Nuclear norm regularization (soft rank penalty)
- Low-rank factorization of attention (Ye et al., 2024)
- Linear attention (Katharopoulos et al., 2020) — extreme case s=d

The sketch is a **randomized** rank constraint rather than a learned one. The randomness provides regularization; the rank constraint provides computational savings.

### 4b. Connection to noise regularization literature

The gradient distortion from sketching can be formalized:

```
g_sketch = g_true + η_sketch
```

where η_sketch has structured covariance determined by the sketch matrix S. This is different from:

- Gaussian gradient noise (SGD): isotropic, uncorrelated
- Dropout gradient noise: block-structured (zeroed units)
- Quantization gradient noise: bounded, STE-approximated

The sketch noise η_sketch has rank structure related to S. This could be analyzed via the perturbation bounds we already have (Proposition 1).

### 4c. Formal regularizer equivalence?

Open question: Can we derive an explicit regularizer R(θ) such that:

```
∇_θ L_sketch ≈ ∇_θ (L_fp32 + λ R(θ))
```

for some λ that depends on sketch dimension s? If so, sketching is equivalent to an explicit spectral regularizer. This would be a strong theoretical result.

Candidate: R(θ) might penalize the nuclear norm of QK^T, or the Frobenius norm of attention off-diagonal blocks, or something related to the spectral gap of the attention matrix.

## 5. Practical implications

If RandNLA sketching regularizes:

1. **Free lunch**: The computational savings from sketching come with generalization benefits, not just speed. This is already claimed by Performer/Linfram papers but without the regularization framing.

2. **Optimal sketch dimension**: There exists an optimal s* that is not the largest feasible s. Current practice uses s as large as accuracy allows; the regularization view suggests using s as small as generalization allows.

3. **Sketch schedule**: Like dropout rate scheduling, one could schedule the sketch dimension — start with small s (strong regularization) and increase s over training (anneal the regularization).

4. **Complementarity with dropout**: If spectral and spatial regularization are orthogonal, practitioners could use both simultaneously for compound benefit.

5. **Interpretability**: The sketch dimension s becomes a tunable knob on the "attention resolution" spectrum, analogous to how dropout rate p is a knob on "representation redundancy."

## 6. Risks and caveats

- The regularization effect may only appear for specific task architectures (small algorithmic datasets). Large-scale LLMs may not benefit because they already have strong implicit regularization from scale.
- Our pilot data shows sketch_16 failing entirely — the capacity barrier may dominate any regularization benefit for large s relative to d.
- Resampled sketches and fixed sketches may have qualitatively different regularization properties. This must be disentangled experimentally.
- The analogy to dropout is suggestive but not exact. Dropout's theoretical analysis (Baldi & Sadowski, 2013) relies on the binary mask structure; sketching has a different algebraic structure.
- Overclaiming risk: "RandNLA regularizes" is not yet supported by rigorous evidence. The correct claim at this stage is: "RandNLA sketching is consistent with a regularization effect in our pilot, but more seeds and controlled comparisons are needed."

## 7. Key references to connect

- **Dropout**: Srivastava et al. (2014), Baldi & Sadowski (2013) — theory of dropout as regularization
- **Gradient noise as regularizer**: Smith & Le (2018), Neelakantan et al. (2015) — SGD noise, quantization noise
- **Low-rank attention**: Ye et al. (2024), Katharopoulos et al. (2020) — computational motivation
- **RandNLA theory**: Mahoney (2016), Drineas & Mahoney (2016) — sketching guarantees
- **Grokking**: Power et al. (2022) — delayed generalization in algorithmic tasks
- **Mixed precision training**: Micikevicius et al. (2018) — fp32 master weights, loss scaling
- **8-bit training**: Sun et al. (2019), Kamp et al. (2023) — stochastic rounding, enhanced loss scaling
