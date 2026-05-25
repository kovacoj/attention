# Mixed-Precision Randomized Attention

Small research repo for a course report on:

`Mixed-Precision Randomized Sketching for Scaled Dot-Product Attention`

The project combines three pieces:

- PyTorch experiments for sketched attention under different precision modes,
- a TeXtured-based LaTeX report template adapted from thesis format to report format,
- a small literature bundle with core references for mixed precision, randNLA, and efficient attention.

## Layout

- `src/attention.py`: attention and random sketching primitives
- `src/experiment.py`: experiment sweeps and metrics
- `scripts/attention_sweep.py`: runnable CLI script
- `report/`: LaTeX source for the report
- `literature/`: curated PDFs and notes
- `.github/workflows/report-pages.yml`: PDF build and GitHub Pages publishing

## Run the experiments

```bash
.venv/bin/python scripts/attention_sweep.py
```

Example:

```bash
.venv/bin/python scripts/attention_sweep.py \
  --sequence-lengths 256 \
  --d-models 64 \
  --sketch-dims 16,32 \
  --seeds 0,1
```

Results are written to `results/attention_sweep.csv` by default.

## Build the report

The repo includes a Dockerized LaTeX build so the report does not depend on the host TeX installation.

```bash
make containers-build
make report
```

Interactive shell:

```bash
make latex-shell
```

## Current report question

The core empirical question is:

> When randomized sketching already introduces a controlled approximation, how much additional error does low precision add, and when does mixed precision recover most of that loss?
