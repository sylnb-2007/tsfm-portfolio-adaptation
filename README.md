# Zero-shot is enough: cost-aware evaluation of TSFM adaptation on U.S. portfolios

Code and saved result matrices for the paper *“Zero-shot is enough: a cost-aware forecast evaluation of adapting a time series foundation model to U.S. portfolio returns.”*

The study asks whether lightweight adaptation of a time series foundation model (TSFM) improves cross-sectional forecasts of U.S. portfolio returns once the comparison is made against (i) the model's own zero-shot output, (ii) an information-matched OLS baseline, and (iii) a portfolio that pays trading costs.

## At a glance

| Component | Specification |
|---|---|
| Backbone | Chronos-2 (119.5M parameters), globally frozen |
| Adaptation | Strict LoRA, 589,824 trainable parameters (0.49%) |
| Panel | 100 U.S. size–book-to-market portfolios (Kenneth R. French Data Library), daily, value-weighted |
| Evaluation window | 2015-01 to 2026-09, walk-forward, leakage-aware execution (signal at close t, trade at close t+1) |
| Costs | 10 basis points per unit of turnover |
| Inference | Newey–West t-statistics, 21-day block bootstrap, paired IC tests, Deflated Sharpe Ratio |

Key numbers reported in the paper: RankIC 0.0191 (t = 3.63) at h = 5 and 0.0248 (t = 2.76) at h = 21 for the adapted model; paired adaptation gain +0.0010 (t = 0.12); h = 5 net return −0.20 basis points per day with an 8.8 basis point break-even cost and a Deflated Sharpe Ratio of 0.248.

## Repository layout

```
code/          analysis scripts, 23 saved prediction matrices (*_pred.npz), result JSONs
figures/       Figures 1–5 in PDF, PNG (300 dpi) and SVG
styles/        matplotlib style used for the figures
scripts/       data download helper
docs/          data description and step-by-step reproduction guide
tools/         manuscript build helpers (docx generation, citation cross-check)
```

## Quick start

```bash
# 1. environment
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. data (not redistributed; downloaded from the public source)
python scripts/download_data.py

# 3. regenerate Figures 1-5 from the saved prediction matrices
python code/m6_figures.py

# 4. recompute the main statistics
python code/stats_m5.py
python code/m6_revision_stats.py
```

Steps 3–4 need only `numpy`, `pandas`, `scipy`, `matplotlib` and the saved matrices in `code/`. Full retraining (steps in `docs/REPRODUCE.md`, sections 5.2–5.4) additionally requires PyTorch, `transformers`, the `chronos-forecasting` package, a CUDA-capable GPU with about 8 GB of memory, and a local copy of the `amazon/chronos-2` weights.

## Data

The raw portfolio files are not included. `scripts/download_data.py` fetches the three daily 10×10 portfolio archives from the Kenneth R. French Data Library and unpacks them into `data/`. See `docs/DATA.md` for file names, section offsets used by the reader, and archive checksums.

## Caveats worth knowing before reusing the code

- `code/tsfm_mv_h5_pred.npz` and `code/tsfm_mv_h21_pred.npz` are legacy names: they hold **univariate batched** forecasts of shape `(N, 1, L)`, not joint cross-series input. The paper states the corrected interpretation.
- The C3 mechanism contrast (`c3_group_mode.py`) uses a **single-split** training prototype (train 2001–2014, test 2015–2026), not the yearly incremental protocol.
- The three French panels overlap in constituents and are not 300 independent series.
- `chronos-2` pretraining data are not public, so pretraining overlap cannot be fully excluded.

## License

Code: MIT (`LICENSE`). Figures, documentation and result matrices: CC BY 4.0 (`LICENSE-CC-BY-4.0.md`).

## Citation

If you use this code, please cite the article and this archive. The archived release is available at <https://doi.org/10.5281/zenodo.24162932>; machine-readable metadata are in `CITATION.cff`.
