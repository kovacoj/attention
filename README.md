# Precision Placement in Attention

Small research repo for a term paper on precision placement, approximation, and finite-precision effects in transformer attention.

The main direction is now exact scaled dot-product attention under explicit precision policies. The older sketching and random-feature experiments remain in the repo as comparison baselines.

Install dependencies with `uv sync`.

## Run Experiments

```sh
# Default: exact-attention precision-placement sweep
uv run python -m src

# Exact-attention precision policies
uv run python -m src --experiment precision-placement --output results/precision_policy_sweep.csv
uv run python -m src --experiment precision-placement --data-source transformer --transformer-model sshleifer/tiny-distilbert-base-cased --transformer-layer 0 --transformer-head 0 --output results/precision_policy_transformer.csv

# Optional: the same exact-attention sweep through the JAX scaffold
uv run python -m src --experiment precision-placement --backend jax --output results/precision_policy_sweep.jax.csv

# Attention-only residual stack with configurable residual scale h
uv run python -m src --experiment residual-stack --depths 4,8,16 --terminal-time 1.0
uv run python -m src --experiment residual-stack --depths 8 --residual-scale 0.125
uv run python -m src --experiment residual-stack --data-source transformer --depths 2,4 --transformer-model sshleifer/tiny-distilbert-base-cased --transformer-layer 0 --transformer-head 0

# Attention entropy transition with quantization policies
python3 src/temperature_experiment.py --output results/attention_temperature_sweep.dev.csv

# Ising magnetization learnability transition
python3 src/ising_experiment.py --output results/ising_learning_transition.dev.csv

# Sketching baseline
uv run python -m src --experiment sketch --output results/attention_sweep.csv

# Random-feature baseline
uv run python -m src --experiment random-features --feature-dims 32,64,128
uv run python -m src --experiment random-features --data-source low-rank --intrinsic-rank 8 --noise-std 1e-2
uv run python -m src --experiment random-features --data-source transformer --sequence-lengths 64 --feature-dims 32,64 --transformer-model distilbert-base-uncased
uv run python -m src --experiment random-features --data-source transformer --sequence-lengths 16 --feature-dims 8 --transformer-model sshleifer/tiny-distilbert-base-cased

# Example: generate publication-quality plots using the bundled styles
uv run python -m src --experiment random-features --feature-dims 32,64,128 --output results/random_feature_sweep.dev.csv
uv run python -m src.plotting results/random_feature_sweep.dev.csv report/figures/rf_errors.pdf
```

## Build Report

```sh
make report
```

`make report` builds the PDF through the repo's Docker/TeX setup.

The report source lives in `report/`, experiment code in `src/`, and committed CSV outputs in `results/`.

Current experiment tracks:

- `precision-placement`: exact attention with explicit storage / accumulation / logit / softmax / value policies.
- `residual-stack`: repeated self-attention residual steps for simple depth-propagation experiments.
- `temperature`: inverse-temperature sweep across the attention entropy transition, including int8 quantization policies.
- `ising`: Ising magnetization learnability-transition experiment with precision/sketch training policies.
- `sketch`: the older sketch-based baseline retained for comparison.
- `random-features`: Performer-style approximation experiments.

JAX is currently an optional backend scaffold rather than a required dependency. The main verified sweeps in this repo run in PyTorch, while `--backend jax` is intended as the next experiment path once JAX is installed locally.
