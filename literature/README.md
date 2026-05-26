# Literature

This folder contains the current working set of papers for the report on precision placement and approximation in scaled dot-product attention.

## Directly useful now

- `vaswani-et-al2017-attention-is-all-you-need.pdf`
  Ashish Vaswani et al., *Attention Is All You Need*.
  Why useful: the canonical source for the exact scaled dot-product attention pipeline.

- `micikevicius2018-mixed-precision-training.pdf`
  Paulius Micikevicius et al., *Mixed Precision Training*.
  Why useful: the most important practical reference for the current project. The key lessons are low-precision storage, fp32 accumulation for dot products, and fp32 reductions for numerically sensitive operations such as softmax and normalization.

- `1710.03740v3.pdf`
  Same paper in arXiv form.
  Why useful: convenient for exact wording and direct citation.

- `2507.03312v3.pdf`
  Alexander Gräfe and Sebastian Trimpe, *MPX: Mixed Precision Training for JAX*.
  Why useful: not mainly a theory paper, but very useful for implementation strategy. It supports treating JAX as an experimental harness with explicit casting wrappers and selective full precision for reductions such as sums, means, and softmax.

## Approximation baselines

- `wang-et-al2020-linformer.pdf`
  Sinong Wang et al., *Linformer: Self-Attention with Linear Complexity*.
  Why useful: low-rank attention baseline.

- `xiong-et-al2021-nystromformer.pdf`
  Yunyang Xiong et al., *Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention*.
  Why useful: Nyström-style approximation baseline.

- `choromanski-et-al2020-performer.pdf`
  Krzysztof Choromanski et al., *Rethinking Attention with Performers*.
  Why useful: random-feature softmax approximation baseline.

## RandNLA background

- `halko-martinsson-tropp2011-randomized-matrix-decompositions.pdf`
  Nathan Halko, Per-Gunnar Martinsson, Joel A. Tropp, *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions*.
  Why useful: general sketching language and approximation background.

- `mahoney2016-randomized-linear-algebra-notes.pdf`
  Michael W. Mahoney, *Lecture Notes on Randomized Linear Algebra*.
  Why useful: broader conceptual background for sketching and low-rank approximation.

## Statistical-physics / message-passing references

- `1102.1182v1.pdf`
  Aurelien Decelle, Florent Krzakala, Cristopher Moore, and Lenka Zdeborová, *Phase transition in the detection of modules in sparse networks*.
  Why useful: concise reference for easy / hard / impossible inference regimes and message-passing phase transitions.

- `1109.3041v2.pdf`
  Aurelien Decelle, Florent Krzakala, Cristopher Moore, and Lenka Zdeborová, *Asymptotic analysis of the stochastic block model for modular networks and its algorithmic applications*.
  Why useful: longer treatment of the same viewpoint. Useful as conceptual guidance if the attention experiments begin to show threshold-like behavior.

- `120518985.pdf`
  Marek Jankola, *Optimizing Initialization in Graph Dynamics: from Ferromagnetism to Opinion Consensus*.
  Why useful: indirect but helpful. It gives a modern message-passing / RSB / algorithmic-hardness perspective from the same statistical-physics tradition.

## Broad reviews / peripheral references

- `1903.10563v2.pdf`
  Giuseppe Carleo et al., *Machine learning and the physical sciences*.
  Why useful: broad context only. Not a core attention citation, but relevant for the general interface between ML, statistical physics, and algorithmic phase transitions.

- `Lectures_on_Geometric_Anatomy_of_Theoretical_Physics.pdf`
  General background; not currently central to the repo.

- `Statistical physics of inference thresholds and algorithms.pdf`
  Closely aligned in spirit with the Decelle / Krzakala / Zdeborová line of work.

## Current takeaways

- The strongest immediate theory / implementation pairing is still `MicikeviciusEtAl2018` plus `MPX 2025`.
- JAX is attractive as a precision-policy laboratory, not because it automatically resolves mixed-precision semantics.
- The Decelle / Krzakala / Zdeborová papers are useful mainly as conceptual guidance for future theory: if our precision-placement experiments show sharp regime changes, we should describe them as phase-diagram behavior rather than only as average error curves.

## Suggested reading order

1. Micikevicius et al.
2. MPX for the JAX implementation angle.
3. Vaswani et al.
4. Performer / Linformer / Nyströmformer for approximation baselines.
5. Decelle et al. for easy / hard regime language.

## Current angle for the report

Right now, the cleanest literature backbone is:

- mixed precision from Micikevicius et al.,
- exact attention from Vaswani et al.,
- optional JAX implementation pragmatics from MPX,
- approximation baselines from Performer / Linformer / Nyströmformer.

That is a better fit for the current precision-placement direction than the older sketching-first framing.
