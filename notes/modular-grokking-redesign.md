# Modular Arithmetic Grokking Under Mixed Precision and Sketched Attention

## Project Status: Research Notes (as of 2026-05-28)

---

## 1. Central Question

Do mixed precision, quantization, and RandNLA-sketch attention delay, prevent, preserve, or accelerate the grokking/generalization transition on modular arithmetic?

**Key reframing**: "Success" does not mean "all low-precision policies beat fp32." Success means finding a **reproducible phase boundary** where safe policies grok reliably while aggressive policies exhibit delayed grokking, lower grokking probability, or outright failure within the training horizon. Failures are informative results if they are reproducible and explained.

---

## 2. Why Modular Arithmetic (Not Ising)

Ising was dropped because:
- Unmasked Ising is too easy (all policies succeed → negative control only)
- Masked Ising is suggestive but borderline (R² near threshold, small seeds)
- Kalman diagnostics were layered on top without a clean causal story
- Modular arithmetic naturally separates memorization from generalization
- The mechanistic interpretability literature (Nanda et al. 2023) gives us progress measures beyond accuracy: circuit formation phases, Fourier structure, cleanup

Modular arithmetic is sufficient. The task family already contains easy cells, hard cells, and mechanistically richer variants (mod-add-3, mod-mul, mod-poly).

---

## 3. Current Findings (Pilot, 2 seeds)

Working baseline: p=97, train_frac=0.3, d_model=64, n_layers=1, n_heads=2, d_mlp=256, lr=1e-3, wd=1.0, full-batch, 6000 steps.

| Policy | Test acc | Train acc | Grok rate | Outcome |
|--------|----------|-----------|-----------|---------|
| fp32 | 0.992 | 1.000 | 1.00 | preserved |
| bf16_safe | 0.990 | 1.000 | 1.00 | preserved |
| fp16_safe | 0.695 | 1.000 | 0.50 | partial |
| int8_logits_dyn | 0.519 | 0.567 | 0.50 | partial |
| int8_qkv_dyn | 0.513 | 0.512 | 0.50 | partial |
| sketch_4 | 0.998 | 1.000 | 1.00 | preserved |
| sketch_8 | 0.979 | 1.000 | 1.00 | preserved |
| sketch_16 | 0.354 | 0.359 | 0.00 | failed |

**Three regimes**: preserved (grok reliably), partial/unreliable (succeed on some seeds, fail on others — near barrier boundary), failed (cannot even fit training set — capacity barrier).

The partial regime is the most scientifically interesting. It suggests these policies sit near a barrier boundary where small changes in seed or hyperparameters tip the outcome.

---

## 4. RandNLA Sketching as Regularization Hypothesis

### 4a. Core idea

RandNLA feature sketching in attention may function as an implicit regularizer, analogous to dropout. Both inject structured noise during training that suppresses overfitting. The mechanism differs, but the effect could be similar.

### 4b. Structural analogy to dropout

| Property | Dropout | RandNLA sketch |
|----------|---------|----------------|
| Where applied | Activations / hidden units | Attention logits (QK^T → (QS)(KS)^T) |
| Mechanism | Zero out random units | Project Q,K onto random low-d subspace |
| Noise type | Binary mask (on/off) | Structured rank reduction |
| Train/eval gap | Yes — must scale at eval | Yes — can eval with full or sketched attention |
| Gradient effect | Blocks gradient through dropped units | Distorts gradient direction (measurable via cosine) |
| Regularization signal | "Don't rely on any one unit" | "Don't rely on fine-grained attention structure" |
| Computational benefit | Minor (sparse ops) | Major (O(ns) vs O(nd) for s ≪ d) |

**Key difference**: Dropout destroys information; sketching compresses it. The regularization is **spectral** rather than **spatial**.

### 4c. Why sketching might regularize

1. **Implicit rank constraint**: QK^T ≈ (QS)(KS)^T forces rank ≤ s. The model cannot learn attention patterns requiring full-rank structure. This pushes toward algorithmic/circuit solutions that are inherently low-rank (modular addition uses circular Fourier structure, which is low-rank in token embedding space).

2. **Gradient distortion as noise**: g_approx ≠ g_fp32. When update SNR > 1 but not >> 1, this acts like gradient noise that helps escape sharp memorization minima and flatten the landscape around the generalizing circuit.

3. **Resampled sketches = MC dropout**: If the sketch is resampled each forward pass, each training step sees a different random projection. The model must learn representations robust to projection variation. This is directly analogous to Monte Carlo dropout.

### 4d. Formal regularizer equivalence (open question)

Can we derive an explicit regularizer R(θ) such that:
```
∇_θ L_sketch ≈ ∇_θ (L_fp32 + λ R(θ))
```
for some λ(s)? If yes, sketching is equivalent to an explicit spectral regularizer — a publishable result.

Candidate R: nuclear norm of QK^T, or Frobenius norm of attention off-diagonal blocks, or spectral gap of the attention matrix.

---

## 5. Literature-Backed Redesign Plan

### 5a. Why the current pilot is not yet decisive

- Only 2 seeds per policy → pilot, not conclusive
- No weight-decay sweep → grokking is known to exist in a narrow "Goldilocks" wd regime
- No MLP/uniform-attention controls → cannot claim attention-specific barriers
- No QAT/PTQ separation → different questions conflated
- No logit-collapse diagnostics → the Prieto et al. (2025) mechanism is directly testable
- No harder modular variants → easy cells may not expose the boundary
- No correction schedule beyond fixed period 10 → filtering/EF literature suggests richer schedules

### 5b. Required controls

| Control | Purpose | Implementation |
|---------|---------|----------------|
| MLP baseline | Test attention-specificity | 2-3 layer MLP on concatenated embeddings, parameter-matched |
| Uniform attention | Test whether attention routing matters | Force weights = 1/T at eval |
| Frozen random attention | Test whether any non-uniform pattern helps | Initialize attention weights randomly, freeze |
| QAT vs PTQ separation | QAT = training-dynamics intervention; PTQ = post-learning robustness test | QAT: train with fake quant. PTQ: train fp32, quantize eval |
| Fixed vs resampled sketch | Fixed = approximation bias; resampled = stochastic optimization noise | Separate policy families, never conflate |

### 5c. Key independent variables

1. **Train fraction** (strongest lever on delay): {0.20, 0.30, 0.40}
2. **Weight decay** (narrow grokking regime): {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1}
3. **Approximation strength**: sketch ratio s/d or quantization bits
4. **Correction period** K: {16, 64, 256} + optional error-feedback variant
5. **Task difficulty**: mod-add-2 (primary), mod-add-3 (secondary), mod-mul (tertiary)

### 5d. Key observables (beyond accuracy)

From the grokking/MI literature:
- t_mem, t_grok, delay, censoring
- Weight L2 norm trajectory
- Fourier spectrum of embeddings (Nanda et al. 2023)

From the numerical stability literature (Prieto et al. 2025):
- Max logit, mean |logit| over time
- Attention entropy over time
- Softmax collapse indicators (overflow, underflow, NaN)
- Gradient alignment with logit-scaling direction

From the mixed-precision/QAT literature:
- Gradient scaler value and overflow count (fp16)
- Clip fraction per quantized tensor
- Second-moment stats in optimizer

From our own diagnostics:
- grad_cosine_exact_vs_policy at current weights
- grad_rel_error, update SNR
- Update cosine after optimizer preconditioning
- Correction flag per step

---

## 6. Experimental Families

| Family | Purpose | Policies |
|--------|---------|----------|
| Exact baselines | Phase diagram without approximation | fp32, bf16_safe, fp16_safe, adamw_state_int8 |
| Unsafe precision stress | Test softmax/logit fragility directly | fp16_low_logit, fp16_low_softmax, int8_post_logits_static, int8_post_logits_dynamic |
| QAT families | Training adapts to quantization noise | int8_qat_qkv_ste, int8_qat_logits_ste, int8_qat_qkv_lsq |
| Fixed sketch families | Isolate approximation bias | sketch_fixed_ratio_{1/2, 1/4, 1/8} |
| Resampled sketch families | Explicit stepwise stochastic perturbation | sketch_resampled_ratio_{1/2, 1/4, 1/8} |
| Hybrid families | Probe quantization + sketch interaction | int8_qat_qkv + sketch_fixed_1/4, bf16_safe + sketch_fixed_1/8 |
| Correction families | Predictor-corrector / iterative refinement | periodic_fp32_correct_K_{16,64,256}, optional error_feedback_update |
| Controls | Separate attention-specific from generic effects | mlp_concat, uniform_attention, frozen_random_attention |

---

## 7. CSV Schemas

### curves.csv (one row per eval step per run)
```
run_id, seed, task, p, train_frac, architecture, policy, wd, lr, step,
train_loss, train_acc, val_acc, test_acc,
weight_l2, max_logit, mean_logit_abs,
attention_entropy, repr_spec_entropy,
grad_scaler, overflow_count, nan_count, correction_flag
```

### gradprobe.csv (one row per probe step per run)
```
run_id, step, probe_split,
grad_cosine_exact_vs_policy, grad_rel_error, grad_sign_agreement,
update_cosine_exact_vs_policy,
qkv_clip_frac, logit_clip_frac,
sketch_ratio, sketch_seed_mode, correction_flag
```

### summary.csv (one row per run)
```
run_id, seed, task, p, train_frac, policy, wd, lr,
grok_success, censor_flag, t_mem, t_grok, delay,
final_test_acc, best_test_acc, best_step,
max_logit_at_mem, repr_entropy_at_mem, grad_cos_at_mem
```

Robust event definitions: t_mem = first step where train_acc ≥ 0.99 for 3 consecutive evals. t_grok = first step where test_acc ≥ 0.95 for 3 consecutive evals. Right-censor runs that never grok.

---

## 8. Key Implementation Details

### Fixed sketches
- Sample one sketch matrix per layer and head at run initialization
- Normalize explicitly
- Store as non-trainable registered buffer
- Log sketch_seed separately for reproducibility

### Resampled sketches
- Redraw only at step boundaries, never within partial forward/backward
- This cleanly separates "approximation bias" from "optimization noise"

### Gradient probes
- At current weights of each approximate-policy run, compute BOTH:
  - gradient from the approximate forward/backward
  - gradient from an exact fp32 forward/backward on the same probe batch
- This local comparison measures distortion in the learning signal induced by precision/sketching, not cumulative divergence

### FP16 policies
- Must use GradScaler and FP32 master weights (per Micikevicius et al. 2017)
- Never compare against "pure fp16 everywhere" as the only fp16 baseline

### Safe vs unsafe precision
- Safe: softmax/logsumexp/exp/norm reductions in FP32 (per PyTorch AMP autocast rules)
- Unsafe: explicit stress tests with low-precision softmax/logit — label clearly

### Correction schedules
- First version: every K steps, turn off fake quant + sketching for one exact fp32 update on same batch
- Do not mix exact and approximate branches within one step initially
- Optional v2: error-feedback variant carrying quantization/sketching residuals in optimizer memory

---

## 9. Highest-Value Plots

1. **Learning curves**: train/test acc vs step, panels for easiest and hardest wd cells
2. **Delay histograms / violin plots**: delay by policy at fixed (task, p, train_frac, wd)
3. **Phase diagrams**: heatmap of grokking success rate and median delay over wd × approximation_strength, faceted by train fraction
4. **Gradient fidelity plots**: grad_cosine_exact_vs_policy vs time; scatter of grad_cosine at t_mem vs eventual delay
5. **Logit-collapse plots**: max_logit, mean|logit|, attention entropy, clip fraction over time for failing policies
6. **Correction ablations**: same boundary cell with K ∈ {16, 64, 256}
7. **Control comparison plots**: transformer, MLP, uniform-attention baselines on same modular cell

---

## 10. Recommended Staged Schedule

### Phase 1: Baseline grid (Week 1-2)
- Reproduce exact fp32 grokking grid over (train_frac × wd)
- Add bf16_safe and fp16_safe baselines
- Confirm: fp32 reliably groks on at least one cell

### Phase 2: Stress tests + diagnostics (Week 2-3)
- Add unsafe softmax/logit precision stress tests
- Add gradient probes and logit-collapse diagnostics
- Identify candidate boundary cells

### Phase 3: Approximation families (Week 3-4)
- Add QAT families (STE + optional LSQ)
- Add fixed-sketch and resampled-sketch families
- Run boundary-cell sweeps over wd

### Phase 4: Controls + harder tasks (Week 4-5)
- Add MLP, uniform-attention, frozen-random-attention controls
- Add mod-add-3 task
- If mod-add-2 too easy, shift focus to harder cells

### Phase 5: Corrections + report (Week 5-6)
- Add periodic fp32 correction and optional error-feedback
- Generate all plots and report tables
- Write up findings honestly

---

## 11. Acceptance Criteria

The project revision is successful only if ALL of the following hold:

- [ ] Modular arithmetic is the sole primary benchmark in the paper
- [ ] At least one harder modular variant (mod-add-3) is implemented
- [ ] MLP and uniform/frozen-attention controls are included
- [ ] QAT and PTQ are reported as separate experiment families
- [ ] Fixed-sketch and resampled-sketch are separate policy families
- [ ] FP32 reference path exists and is used for gradient probes
- [ ] FP16 policies use GradScaler + FP32 master weights
- [ ] Safe policies keep softmax/logsumexp/exp/norm in FP32
- [ ] At least one boundary cell shows real separation between fp32-safe and an approximate policy
- [ ] Delay and censoring are reported, not only final accuracy
- [ ] Curves, summary, and grad-probe CSVs are all generated
- [ ] Logit-scale and entropy diagnostics exist for failure cases
- [ ] Report explains whether observed barrier is or is not attention-specific
- [ ] Preprint-based claims are labeled as suggestive where appropriate
- [ ] Conclusion states clearly: true training-time barrier found, delay boundary found, or mostly robustness

---

## 12. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| fp32 does not grok reliably on any cell | Medium (depends on arch/hp) | Search wider arch/hp space before adding approximation; start with literature-reported cells |
| All approximate policies also grok (too easy) | Medium | Move to harder cells: lower train_frac, mod-add-3, finer wd sweep near fp32 boundary |
| Modular addition groks without learned attention | Medium-high (MLP literature) | Include controls; reframe result as "training-barrier in modular arithmetic models" not "attention-specific barrier" |
| sketch_16 failure was capacity, not training barrier | High (we already know this) | Report as capacity barrier; focus analysis on partial/unreliable regime where training dynamics matter |
| Resampled vs fixed sketch differences are noise | Medium | Use enough seeds; log sketch_seed; run paired statistical tests |
| Overclaiming on regularization analogy | Medium | Keep all claims as "consistent with" or "suggestive of" until formal equivalence is proven |
| Literature is 2025-2026 preprints, not settled | High | Anchor strongest claims in Power et al. 2022, Nanda et al. 2023, Micikevicius et al. 2017; label preprint claims cautiously |

---

## 13. Key References

### Grokking & mechanistic interpretability
- Power et al. (2022) — original grokking benchmark
- Nanda et al. (2023) — Fourier circuit reverse-engineering of modular-addition transformers
- Gromov (2023) — MLP grokking on modular arithmetic
- Li et al. (2024) — multi-input modular addition and Fourier/margin analysis
- Furuta et al. (2024) — modular polynomial grokking
- Minegishi et al. (2024) — weight norms, sparsity, grokking tickets

### Numerical stability & grokking
- Prieto et al. (2025) — logit scaling, gradient alignment, softmax collapse in delayed grokking
- Khanh et al. (2026) — delay laws, norm separation, first-passage models
- Khanh et al. (2026) — spectral entropy as grokking leading indicator

### Mixed-precision training
- Micikevicius et al. (2017) — FP32 master weights, loss scaling
- Kalamkar et al. (2019) — BF16 study
- PyTorch AMP docs — autocast rules for softmax/exp/norm
- Wortsman et al. (2023) — large-scale low-precision stability

### Quantization & QAT
- Jacob et al. (2017) — foundational QAT
- Bengio et al. (2013) — STE origin
- Yin et al. (2019) — STE theory and instability
- Esser et al. (2019) — LSQ learned step sizes
- StableQAT (2026) — smoother bounded surrogates for ultra-low-bit
- QPyTorch (2019) — low-precision simulation
- Dettmers et al. (2021) — 8-bit optimizer states
- NVIDIA Transformer Engine — FP8 scaling, delayed amax, stochastic rounding

### RandNLA & efficient attention
- Halko, Martinsson, Tropp (2009) — randomized SVD
- Mahoney (2016) — RandNLA foundations
- Wang et al. (2020) — Linformer
- Xiong et al. (2021) — Nyströmformer
- Choromanski et al. (2021) — Performer
- Dao et al. (2022) — FlashAttention, online softmax
- Kalamkar et al. (2019) — BF16 study

### Correction/filtering analogies
- KGD / OKF — Kalman gradient descent, optimal Kalman filter
- EF-SGD — error-feedback SGD for compressed updates
- Mixed-precision iterative refinement — cheap step + expensive correction
- Fitzgibbon & Felix (2025) — stochastic rounding bias caveat
