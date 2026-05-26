# Precision Placement in Attention

Small research repo for a term paper on finite-precision effects in transformer attention.

The main direction is now exact scaled dot-product attention under explicit precision policies. The older sketching and random-feature experiments remain in the repo as comparison baselines.

Install dependencies with `uv sync`.

## Run Experiments

```sh
# Default: exact-attention precision-placement sweep
uv run python -m src

# Exact-attention precision policies
uv run python -m src --experiment precision-placement --output results/precision_policy_sweep.csv

# Attention-only residual stack with configurable residual scale h
uv run python -m src --experiment residual-stack --depths 4,8,16 --terminal-time 1.0
uv run python -m src --experiment residual-stack --depths 8 --residual-scale 0.125

# Sketching baseline
uv run python -m src --experiment sketch --output results/attention_sweep.csv

# Random-feature baseline
uv run python -m src --experiment random-features --feature-dims 32,64,128
uv run python -m src --experiment random-features --data-source low-rank --intrinsic-rank 8 --noise-std 1e-2
uv run python -m src --experiment random-features --data-source transformer --sequence-lengths 64 --feature-dims 32,64 --transformer-model distilbert-base-uncased
uv run python -m src --experiment random-features --data-source transformer --sequence-lengths 16 --feature-dims 8 --transformer-model sshleifer/tiny-distilbert-base-cased

# Example: generate publication-quality plots using the bundled styles
uv run python -m src --experiment random-features --feature-dims 32,64,128 --output results/random_feature_sweep.dev.csv
uv run python -m src.plotting results/random_feature_sweep.dev.csv figures/rf_errors.pdf
```

## Build Report

```sh
make report
```

`make report` builds the PDF through the repo's Docker/TeX setup.

The report source lives in `report/`, experiment code in `src/`, and committed CSV outputs in `results/`.
