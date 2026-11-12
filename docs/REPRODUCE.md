# Reproduction guide

This guide reproduces every number reported in the paper from public data and the saved prediction matrices. Figures 1–5 and the main statistics can be regenerated **without retraining**; full retraining of the adapted model requires a GPU.

## 1. Environment

| Item | Value used for the reported results |
|---|---|
| OS | Windows 11 / AMD64 (Linux is fine for steps 3–5) |
| Python | 3.14.0 |
| CPU / RAM | 32 logical cores / 15.8 GiB |
| GPU | NVIDIA RTX 5060 Laptop, ~8.5 GB |
| torch | 2.11.0+cu128 |
| transformers | 5.22.1 |
| chronos | `amazon/chronos-2` (Hugging Face) |
| other | numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.1, scipy 1.18.1, lightgbm 4.7.0, catboost 1.2.10 |

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

Full retraining additionally needs a CUDA build of PyTorch and the `amazon/chronos-2` weights in the Hugging Face cache. If a CPU-only torch build is loaded by mistake, `torch.cuda.is_available()` returns `False` and the training steps will be very slow.

## 2. Data

```bash
python scripts/download_data.py
```

This writes the three CSVs into `data/`. See `docs/DATA.md` for section offsets and checksums.

## 3. Figures from saved matrices (no training)

```bash
python code/m6_figures.py
```

Reads `code/*_pred.npz` and `code/*.json`; writes `figures/fig1…fig5` in `.pdf`, `.png` (300 dpi) and `.svg`. Figure 2 recomputes annual RankIC from the prediction matrices rather than reading a stored field.

## 4. Statistics from saved matrices (no training)

```bash
python code/stats_m5.py                 # h = 5 main statistics
python code/m6_revision_stats.py        # adaptation vs zero-shot paired tests, execution-day sensitivity
python code/m6_label_check.py           # earliest-tradable label check
```

Expected outputs: `code/stats_m5.json`, `code/m6_revision_stats.json`, `code/m6_label_check.json`.

## 5. Full pipeline (requires GPU)

Run from the repository root. `data_pipeline.py` sanity check:

```bash
python code/data_pipeline.py
```

Expected: 6,726 days × 100 portfolios on the main panel, 26 walk-forward splits, the first three splits pass the purge check, and the mean absolute random IC is about 0.089.

### 5.1 Information-matched OLS baseline

```bash
# main panel, h = 5, evaluation window from 2015
BS_ALIGNED=1 BS_H=5 BS_START_YEAR=2015 python code/baselines.py data/100_Portfolios_10x10_Daily.csv ols

# main panel, h = 21, full window from 2002 (window-sensitivity check)
BS_ALIGNED=1 BS_H=21 BS_START_YEAR=2002 python code/baselines.py data/100_Portfolios_10x10_Daily.csv ols

# OP and INV panels, h = 5
BS_ALIGNED=1 BS_H=5 BS_START_YEAR=2002 python code/baselines.py data/100_Portfolios_ME_OP_10x10_daily.csv ols
BS_ALIGNED=1 BS_H=5 BS_START_YEAR=2002 python code/baselines.py data/100_Portfolios_ME_INV_10x10_daily.csv ols
```

On PowerShell, set the same variables with `$env:BS_H="5"` before the call.

### 5.2 Zero-shot Chronos-2

```bash
python code/tsfm_eval.py
```

Note: `tsfm_mv_h5_pred.npz` and `tsfm_mv_h21_pred.npz` are legacy file names; they contain **univariate batched** forecasts of shape `(N, 1, L)`, not joint cross-series input.

### 5.3 Yearly adaptation

```bash
# incremental protocol, h = 5 (main result)
BS_H=5 BS_YEARLY_RESET=0 BS_LORA_STRIDE=10 BS_LORA_EPOCHS=1 python code/train_lora_yearly.py

# reset protocol, h = 5
BS_H=5 BS_YEARLY_RESET=1 BS_LORA_STRIDE=10 BS_LORA_EPOCHS=1 python code/train_lora_yearly.py

# incremental, h = 21
BS_H=21 BS_YEARLY_RESET=0 BS_LORA_STRIDE=10 BS_LORA_EPOCHS=1 python code/train_lora_yearly.py

# OP and INV panels
BS_H=5 BS_YEARLY_RESET=0 BS_PANEL=data/100_Portfolios_ME_OP_10x10_daily.csv BS_SECTION=23 BS_TAG=_OP python code/train_lora_yearly.py
BS_H=5 BS_YEARLY_RESET=0 BS_PANEL=data/100_Portfolios_ME_INV_10x10_daily.csv BS_SECTION=17 BS_TAG=_INV python code/train_lora_yearly.py
```

### 5.4 Mechanism controls and robustness

```bash
python code/c3_group_mode.py      # pooled vs series-isolated, parameter-matched
python code/c3_permutation.py     # 500 sequence-identity permutations
python code/c4_panels.py          # replication across the three panels
python code/c6_crisis.py          # crisis subperiods
```

The C3 solo matrix comes from a single-split prototype (train 2001–2014, test 2015–2026):

```bash
BS_GROUP_MODE=solo BS_H=5 BS_LORA_STRIDE=5 python code/train_lora.py
```

This single-split design is stated in Section 4.6 and Section 5.5 of the paper; it is not the yearly incremental protocol.

## 6. Mapping from results to claims

| Paper claim | Script | Stored output |
|---|---|---|
| Absolute RankIC at h = 5 and h = 21 | `stats_m5.py` | `stats_m5.json`, `stats_m5_h21.json` |
| Adaptation adds nothing (paired tests) | `m6_revision_stats.py` | `m6_revision_stats.json` |
| Signal does not survive costs (break-even, DSR) | `stats_m5.py` | `stats_m5.json` |
| Mechanism is not cross-series attention | `c3_group_mode.py`, `c3_permutation.py` | `c3_group_mode.json`, `c3_permutation.json` |
| No replication on OP/INV panels | `c4_panels.py` | `c4_panels.json` |
| Crisis-period concentration | `c6_crisis.py` | `c6_crisis.json` |
| Execution-day sensitivity | `m6_revision_stats.py`, `m6_label_check.py` | `m6_revision_stats.json`, `m6_label_check.json` |

## 7. Known caveats

- `chronos-2` pretraining data are not public; pretraining overlap with the evaluation window cannot be fully excluded.
- The three French panels overlap in constituents; they are not independent samples.
- LoRA training uses a single seed per configuration in the yearly protocol; the C3 contrast uses a single split.
