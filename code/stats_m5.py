"""M5 formal validation: IC significance, DM tests, block bootstrap, Deflated Sharpe."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
from data_pipeline import forward_labels, rank_ic, read_section  # noqa: E402

import os

H = int(os.environ.get("BS_H", "5"))
K = 20
COST_BPS = 10.0
BLOCK = 21
N_BOOT = 2000
MODELS_BY_H = {
    5: {
        "lora_inc": "code/tsfm_lora_yearly_5_inc_pred.npz",
        "lora_reset": "code/tsfm_lora_yearly_5_reset_pred.npz",
        "zero_shot_uni": "code/tsfm_mv_h5_pred.npz",
        "zero_shot_group_rank": "code/tsfm_group_rank_h5_pred.npz",
        "ols_aligned": "code/baseline_ols_aligned_h5_pred.npz",
    },
    21: {
        "lora_inc": "code/tsfm_lora_yearly_21_inc_pred.npz",
        "lora_single": "code/tsfm_lora_21_pred.npz",
        "zero_shot_uni": "code/tsfm_mv_h21_pred.npz",
        "ols_aligned": "code/baseline_ols_aligned_h21_pred.npz",
    },
}
MODELS = MODELS_BY_H[H]


def load(path: str, panel: pd.DataFrame) -> pd.DataFrame:
    z = np.load(path, allow_pickle=False)
    return pd.DataFrame(z["values"], index=pd.to_datetime(z["dates"]), columns=pd.Index(z["columns"])).reindex(
        columns=panel.columns
    )


def nw_tstat(x: pd.Series, lag: int) -> float:
    x = x.dropna().to_numpy()
    n = len(x)
    mu = x.mean()
    e = x - mu
    var = (e @ e) / n
    for l in range(1, lag + 1):
        cov = (e[l:] @ e[:-l]) / n
        var += 2 * (1 - l / (lag + 1)) * cov
    se = np.sqrt(max(var, 1e-18) / n)
    return float(mu / se)


def block_bootstrap_ci(x: pd.Series, block: int = BLOCK, n_boot: int = N_BOOT, seed: int = 7) -> tuple[float, float]:
    x = x.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block), size=(n_boot, n_blocks))
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = (starts[i][:, None] + np.arange(block)[None, :]).ravel()
        idx = idx[idx < n][:n]
        means[i] = x[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def daily_net(pred: pd.DataFrame, ret: pd.DataFrame, scheme: str, cost_bps: float) -> pd.Series:
    dates = list(pred.index)
    rows, w_prev = [], None
    for i in range(0, len(dates), H):
        p = pred.loc[dates[i]]
        valid = p.dropna()
        if len(valid) < 2 * K:
            continue
        order = valid.sort_values()
        w = pd.Series(0.0, index=pred.columns)
        if scheme == "binary":
            w.loc[order.index[-K:]] = 1.0 / K
            w.loc[order.index[:K]] = -1.0 / K
        else:
            r = valid.rank() - valid.rank().mean()
            w.loc[valid.index] = (r / r.abs().sum()) * 2.0
        base = pd.Series(0.0, index=w.index) if w_prev is None else w_prev
        turnover = float((w - base).abs().sum())
        w_prev = w
        cost = cost_bps / 1e4 * turnover / H
        for d in dates[i + 2:i + 2 + H]:
            if d in ret.index:
                rows.append((d, float((w * ret.loc[d].fillna(0.0)).sum()) - cost))
    s = pd.Series({d: v for d, v in rows})
    return s.sort_index()


def mean_turnover(pred: pd.DataFrame) -> float:
    dates = list(pred.index)
    to, w_prev = [], None
    for i in range(0, len(dates), H):
        p = pred.loc[dates[i]]
        valid = p.dropna()
        if len(valid) < 2 * K:
            continue
        r = valid.rank() - valid.rank().mean()
        w = pd.Series(0.0, index=pred.columns)
        w.loc[valid.index] = (r / r.abs().sum()) * 2.0
        base = pd.Series(0.0, index=w.index) if w_prev is None else w_prev
        to.append(float((w - base).abs().sum()))
        w_prev = w
    return float(np.mean(to))


def deflated_sharpe(sr: float, n_trials: int, sr_var: float, skew: float, kurt: float) -> float:
    from math import sqrt
    from scipy.stats import norm

    sr0 = sqrt(max(sr_var, 1e-12)) * ((1 - np.euler_gamma) * norm.ppf(1 - 1 / n_trials)
                                       + np.euler_gamma * norm.ppf(1 - 1 / (n_trials * np.e)) )
    denom = sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr0) / denom))


def main():
    panel = read_section("data/100_Portfolios_10x10_Daily.csv", 19).loc["2000-01-01":]
    label = forward_labels(panel, H)
    ret = panel.shift(-1)
    common = None
    preds = {}
    for name, path in MODELS.items():
        p = load(path, panel)
        preds[name] = p
        common = p.index if common is None else common.intersection(p.index)
    common = common.sort_values()
    print("common dates", len(common))

    ic = {name: rank_ic(p.loc[common], label.loc[common]) for name, p in preds.items()}
    nets = {name: daily_net(p.loc[common], ret, "rank", COST_BPS) for name, p in preds.items()}

    rows = []
    for name in MODELS:
        s = ic[name]
        n = nets[name]
        sr = float(np.sqrt(252) * n.mean() / n.std(ddof=1))
        lo, hi = block_bootstrap_ci(s)
        rows.append(
            {
                "model": name,
                "ic_mean": round(float(s.mean()), 4),
                "ic_nw_t": round(nw_tstat(s, H), 2),
                "ic_ci95": [round(lo, 4), round(hi, 4)],
                "net_sharpe": round(sr, 3),
                "net_daily_bps": round(float(n.mean() * 1e4), 2),
                "turnover": round(mean_turnover(preds[name].loc[common]), 3),
            }
        )

    base = "ols_aligned"
    diff_rows = []
    for name in MODELS:
        if name == base:
            continue
        d = (ic[name] - ic[base]).dropna()
        lo, hi = block_bootstrap_ci(d)
        diff_rows.append(
            {
                "vs": f"{name} - {base}",
                "mean_diff": round(float(d.mean()), 4),
                "nw_t": round(nw_tstat(d, H), 2),
                "ci95": [round(lo, 4), round(hi, 4)],
            }
        )

    n = nets["lora_inc"]
    sr = float(np.sqrt(252) * n.mean() / n.std(ddof=1))
    srs = [float(np.sqrt(252) * x.mean() / x.std(ddof=1)) for x in nets.values()]
    dsr = deflated_sharpe(sr, len(MODELS), float(np.var(srs, ddof=1)), float(n.skew()), float(n.kurtosis() + 3))

    out = {"h": H, "table": rows, "paired_vs_ols": diff_rows, "deflated_sharpe_lora_inc": round(dsr, 3),
           "n_trials": len(MODELS)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    path = "code/stats_m5.json" if H == 5 else f"code/stats_m5_h{H}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()