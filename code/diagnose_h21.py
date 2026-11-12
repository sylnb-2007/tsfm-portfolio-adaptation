"""Diagnose the h=21 divergence between predictive IC and portfolio P&L.

Compares:
1. IC against the label used for training (r_{t+1..t+21})
2. IC against the tradeable return (r_{t+2..t+22}) implied by execution at close t+1
3. long-short returns on rebalance dates under both definitions
4. whether predictions mainly track volatility rather than direction
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
from data_pipeline import forward_labels, rank_ic, read_section, trailing_features  # noqa: E402

H = 21
K = 20


def load_pred(path: str, panel: pd.DataFrame) -> pd.DataFrame:
    z = np.load(path, allow_pickle=False)
    dates = pd.to_datetime(z["dates"])
    cols = pd.Index(z["columns"])
    return pd.DataFrame(z["values"], index=dates, columns=cols).reindex(columns=panel.columns)


def long_short(pred_row: pd.Series, target_row: pd.Series, k: int = K) -> float:
    ok = pred_row.notna() & target_row.notna()
    p, r = pred_row[ok], target_row[ok]
    if len(p) < 2 * k:
        return np.nan
    order = p.sort_values()
    return float(r[order.index[-k:]].mean() - r[order.index[:k]].mean())


def main():
    panel = read_section("data/100_Portfolios_10x10_Daily.csv", 19).loc["2000-01-01":]
    pred = load_pred("code/tsfm_mv_h21_pred.npz", panel)

    logret = np.log1p(panel)
    label = forward_labels(panel, H)
    tradeable = np.expm1(sum(logret.shift(-k) for k in range(2, H + 2)))
    vol = trailing_features(panel)["vol_21"]

    ic_label = rank_ic(pred, label.loc[pred.index])
    ic_tradable = rank_ic(pred, tradeable.loc[pred.index])

    reb_dates = list(pred.index)[::H]
    ls_label = pd.Series({t: long_short(pred.loc[t], label.loc[t]) for t in reb_dates}).dropna()
    ls_tradable = pd.Series({t: long_short(pred.loc[t], tradeable.loc[t]) for t in reb_dates}).dropna()

    # does the prediction rank volatility instead of returns?
    vol_corr = []
    for t in reb_dates[:200]:
        p = pred.loc[t]
        v = vol.loc[t] if t in vol.index else None
        if v is None:
            continue
        ok = p.notna() & v.notna()
        if ok.sum() > 20:
            vol_corr.append(p[ok].rank().corr(v[ok].rank()))
    vol_corr = pd.Series(vol_corr)

    out = {
        "pred_dates": int(pred.shape[0]),
        "rebalances": int(len(reb_dates)),
        "ic_vs_training_label": round(float(ic_label.mean()), 4),
        "ic_vs_tradeable_return": round(float(ic_tradable.mean()), 4),
        "ls_mean_bps_using_label": round(float(ls_label.mean() * 1e4), 2),
        "ls_mean_bps_using_tradeable": round(float(ls_tradable.mean() * 1e4), 2),
        "ls_t_using_label": round(float(ls_label.mean() / (ls_label.std(ddof=1) / np.sqrt(len(ls_label)))), 2),
        "ls_t_using_tradeable": round(float(ls_tradable.mean() / (ls_tradable.std(ddof=1) / np.sqrt(len(ls_tradable)))), 2),
        "pred_vs_vol_rank_corr_mean": round(float(vol_corr.mean()), 3),
        "pred_cross_sectional_std_mean": round(float(pred.std(axis=1).mean()), 5),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
