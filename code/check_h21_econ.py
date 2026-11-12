"""Diagnose the h=21 economic anomaly: OLS shows insignificant IC but positive net Sharpe."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
from data_pipeline import forward_labels, rank_ic, read_section  # noqa: E402

H = 21
K = 20
MODELS = {
    "lora_inc": "code/tsfm_lora_yearly_21_inc_pred.npz",
    "ols_aligned": "code/baseline_ols_aligned_h21_pred.npz",
}


def load(path, panel):
    z = np.load(path, allow_pickle=False)
    return pd.DataFrame(z["values"], index=pd.to_datetime(z["dates"]), columns=pd.Index(z["columns"]))[
        panel.columns
    ]


def weights(p, scheme="rank"):
    valid = p.dropna()
    if len(valid) < 2 * K:
        return None
    w = pd.Series(0.0, index=p.index)
    if scheme == "rank":
        r = valid.rank() - valid.rank().mean()
        w.loc[valid.index] = (r / r.abs().sum()) * 2.0
    else:
        order = valid.sort_values()
        w.loc[order.index[-K:]] = 1.0 / K
        w.loc[order.index[:K]] = -1.0 / K
    return w


def main():
    panel = read_section("data/100_Portfolios_10x10_Daily.csv", 19).loc["2000-01-01":]
    label = forward_labels(panel, H)
    ret = panel.shift(-1)
    for name, path in MODELS.items():
        pred = load(path, panel)
        for tag, sl in (("full", pred.index), ("2015+", pred.loc["2015":].index)):
            pred_w = pred.loc[sl]
            dates = list(pred_w.index)
            gross, net10 = [], []
            w_prev, n_chg = None, []
            for i in range(0, len(dates), H):
                w = weights(pred_w.loc[dates[i]])
                if w is None:
                    continue
                if w_prev is not None:
                    n_chg.append(float((w - w_prev).abs().sum()))
                turnover = 0.0 if w_prev is None else float((w - w_prev).abs().sum())
                w_prev = w
                gross_day = 0.0
                for d in dates[i + 2:i + 2 + H]:
                    if d in ret.index:
                        gross_day += float((w * ret.loc[d].fillna(0.0)).sum())
                gross.append(gross_day / H)
                net10.append(gross_day / H - 10.0 / 1e4 * turnover / H)
            g = pd.Series(gross)
            n = pd.Series(net10)
            ic = rank_ic(pred_w, label.loc[pred_w.index])
            print(
                f"{name} [{tag}]: IC {ic.mean():.4f} | gross {g.mean()*1e4:.2f}bps SR {np.sqrt(252/H)*g.mean()/g.std():.3f}"
                f" | net10 {n.mean()*1e4:.2f}bps SR {np.sqrt(252/H)*n.mean()/n.std():.3f}"
                f" | turnover {np.mean(n_chg):.3f}"
            )


if __name__ == "__main__":
    main()
