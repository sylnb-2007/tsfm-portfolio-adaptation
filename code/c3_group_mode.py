"""C3 control 1: parameter-matched control (same LoRA adapter, cross-series attention disabled)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
from data_pipeline import forward_labels, rank_ic, read_section  # noqa: E402
from stats_m5 import nw_tstat  # noqa: E402

H = 5
PATHS = {"pooled": "code/tsfm_lora_5_pred.npz", "solo": "code/tsfm_lora_5_solo_pred.npz"}


def load(path, panel):
    z = np.load(path, allow_pickle=False)
    return pd.DataFrame(z["values"], index=pd.to_datetime(z["dates"]), columns=pd.Index(z["columns"]))[
        panel.columns
    ]


def main():
    panel = read_section("data/100_Portfolios_10x10_Daily.csv", 19).loc["2000-01-01":]
    label = forward_labels(panel, H)
    preds = {k: load(v, panel) for k, v in PATHS.items()}

    ics = {k: rank_ic(p, label.loc[p.index]) for k, p in preds.items()}
    diff = (ics["pooled"] - ics["solo"]).dropna()

    # cross-sectional rank correlation of the two prediction matrices, averaged over dates
    rcorr = [
        preds["pooled"].loc[d].rank().corr(preds["solo"].loc[d].rank())
        for d in preds["pooled"].index
    ]

    out = {
        "h": H,
        "ic_pooled": round(float(ics["pooled"].mean()), 4),
        "ic_solo": round(float(ics["solo"].mean()), 4),
        "ic_diff_pooled_minus_solo": round(float(diff.mean()), 4),
        "ic_diff_nw_t": round(nw_tstat(diff, H), 2),
        "pred_rank_corr_mean": round(float(np.mean(rcorr)), 4),
        "pred_abs_diff_mean": round(float(np.abs(preds["pooled"].to_numpy() - preds["solo"].to_numpy()).mean()), 6),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open("code/c3_group_mode.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
