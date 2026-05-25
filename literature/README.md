# Literature

This folder contains a small working set of papers for the report on mixed-precision randomized sketching for scaled dot-product attention.

## Canonical transformer reference

- `vaswani-et-al2017-attention-is-all-you-need.pdf`
  Ashish Vaswani et al., *Attention Is All You Need*.
  Why useful: the canonical source for the transformer architecture and the standard scaled dot-product attention formula studied in the report.

## Core mixed-precision reference

- `micikevicius2018-mixed-precision-training.pdf`
  Paulius Micikevicius et al., *Mixed Precision Training*.
  Why useful: the standard deep learning reference for fp16 storage, fp32 accumulation, and loss-scaling. This is the baseline motivation for the mixed-precision side of the report.

## Core randNLA references

- `halko-martinsson-tropp2011-randomized-matrix-decompositions.pdf`
  Nathan Halko, Per-Gunnar Martinsson, Joel A. Tropp, *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions*.
  Why useful: the main survey/reference for randomized low-rank approximation and sketching.

- `mahoney2016-randomized-linear-algebra-notes.pdf`
  Michael W. Mahoney, *Lecture Notes on Randomized Linear Algebra*.
  Why useful: broader randNLA background, especially for framing sketching ideas and error tradeoffs at a more conceptual level.

## Efficient-attention references

- `wang-et-al2020-linformer.pdf`
  Sinong Wang et al., *Linformer: Self-Attention with Linear Complexity*.
  Why useful: low-rank approximation viewpoint on attention. This is especially relevant if we want to compare our feature-sketch experiment with low-rank attention approximations.

- `xiong-et-al2021-nystromformer.pdf`
  Yunyang Xiong et al., *Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention*.
  Why useful: directly uses a classical matrix approximation method for attention, close in spirit to randNLA.

- `choromanski-et-al2020-performer.pdf`
  Krzysztof Choromanski et al., *Rethinking Attention with Performers*.
  Why useful: random-feature approximation of softmax attention. This is highly relevant if the report shifts from plain Gaussian projection of features toward kernelized attention approximations.

## Suggested reading order

1. Micikevicius et al. for the mixed-precision baseline.
2. Halko--Martinsson--Tropp for sketching language and randomized approximation tools.
3. Linformer and Nyströmformer for attention approximations closest to matrix sketching.
4. Performer if we want to connect the project more explicitly to kernel methods and random features.

## Current angle for the report

Right now, the strongest literature alignment is:

- mixed precision from Micikevicius et al.,
- randomized approximation from Halko--Martinsson--Tropp,
- attention approximation from Linformer or Nyströmformer.

That combination is probably the cleanest backbone for a short mathematical report.
