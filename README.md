# Mixed-Precision Randomized Attention

Small research repo for a term paper on approximation and finite-precision effects in transformer attention.

Install dependencies with `uv sync`.

## Run Experiments

```sh
uv run python -m src
uv run python -m src --output results/attention_sweep.csv
uv run python -m src --experiment random-features --feature-dims 32,64,128
uv run python -m src --experiment random-features --data-source low-rank --intrinsic-rank 8 --noise-std 1e-2
uv run python -m src --experiment random-features --data-source transformer --sequence-lengths 64 --feature-dims 32,64 --transformer-model distilbert-base-uncased
uv run python -m src --experiment random-features --data-source transformer --sequence-lengths 16 --feature-dims 8 --transformer-model sshleifer/tiny-distilbert-base-cased

# Example: generate publication-quality plots using the bundled styles
uv run python -m src --experiment random-features --feature-dims 32,64,128 --output results/random_feature_sweep.dev.csv
python -m src.plotting plot_rf_error_vs_m results/random_feature_sweep.dev.csv figures/rf_errors.pdf
```

## Build Report

```sh
make report
```

`make report` builds the PDF through the repo's Docker/TeX setup.

The report source lives in `report/`, experiment code in `src/`, and committed CSV outputs in `results/`.
