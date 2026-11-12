"""C6: crisis-period robustness of the main conclusions (h=5 and h=21)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
from data_pipeline import forward_labels, rank_ic, read_section  # noqa: E402
from stats_m5 import daily_net, load, nw_tstat  # noqa: E402

PERIODS = {
    "2015-2019": ("2015", "2019"),
    "2020 COVID": ("2020", "2020"),
    "2021": ("2021", "2021"),
    "2022": ("2022", "2022"),
    "2023-2026": ("2023", "2026"),
}
MODELS_H5 = {
    "lora_inc": "code/tsfm_lora_yearly_5_inc_pred.npz",
    "zero_shot_uni": "code/tsfm_mv_h5_pred.npz",
    "ols_aligned": "code/baseline_ols_aligned_h5_pred.npz",
}
MODELS_H21 = {
    "lora_inc": "code/tsfm_lora_yearly_21_inc_pred.npz",
    "ols_aligned": "code/baseline_ols_aligned_h21_pred.npz",
}


def block(h: int, models: dict, panel: pd.DataFrame, out: dict):
    label = forward_labels(panel, h)
    ret = panel.shift(-1)
    for name, path in models.items():
        pred = load(path, panel)
        row = {}
        for tag, (a, b) in PERIODS.items():
            p = pred.loc[a:b]
            lab = label.loc[p.index]
            ic = rank_ic(p, lab)
            net = daily_net(p, ret, "rank", 10.0)
            row[tag] = {
                "ic": round(float(ic.mean()), 4),
                "nw_t": round(nw_tstat(ic, h), 2),
                "net_bps": round(float(net.mean() * 1e4), 2),
                "net_sharpe": round(float(np.sqrt(252) * net.mean() / net.std(ddof=1)), 3),
                "n_dates": int(len(p)),
            }
        out[f"h{h}"][name] = row


def main():
    panel = read_section("data/100_Portfolios_10x10_Daily.csv", 19).loc["2000-01-01":]
    out = {"h5": {}, "h21": {}}
    block(5, MODELS_H5, panel, out)
    block(21, MODELS_H21, panel, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open("code/c6_crisis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
