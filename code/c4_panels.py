"""C4: reproduce the h=5 incremental LoRA result on Size-OP and Size-INV panels."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
from data_pipeline import forward_labels, rank_ic, read_section  # noqa: E402
from stats_m5 import daily_net, load, nw_tstat  # noqa: E402

H = 5
COST = 10.0
PANELS = {
    "BM": ("data/100_Portfolios_10x10_Daily.csv", 19),
    "OP": ("data/100_Portfolios_ME_OP_10x10_daily.csv", 23),
    "INV": ("data/100_Portfolios_ME_INV_10x10_daily.csv", 17),
}
TAG = {"BM": "", "OP": "_OP", "INV": "_INV"}


def main():
    rows = []
    for name, (path, section) in PANELS.items():
        panel = read_section(path, section).loc["2000-01-01":]
        label = forward_labels(panel, H)
        ret = panel.shift(-1)
        tag = TAG[name]
        lora = load(f"code/tsfm_lora_yearly_5_inc{tag}_pred.npz", panel)
        ols = load(f"code/baseline_ols_aligned_h5{tag}_pred.npz", panel)
        common = lora.index.intersection(ols.index).sort_values()
        ic_l = rank_ic(lora.loc[common], label.loc[common])
        ic_o = rank_ic(ols.loc[common], label.loc[common])
        diff = (ic_l - ic_o).dropna()
        net = daily_net(lora.loc[common], ret, "rank", COST)
        rows.append(
            {
                "panel": name,
                "n_dates": int(len(common)),
                "lora_ic": round(float(ic_l.mean()), 4),
                "lora_nw_t": round(nw_tstat(ic_l, H), 2),
                "ols_ic": round(float(ic_o.mean()), 4),
                "ols_nw_t": round(nw_tstat(ic_o, H), 2),
                "diff": round(float(diff.mean()), 4),
                "diff_nw_t": round(nw_tstat(diff, H), 2),
                "net_bps": round(float(net.mean() * 1e4), 2),
                "net_sharpe": round(float(np.sqrt(252) * net.mean() / net.std(ddof=1)), 3),
            }
        )
        print(f"{name}: done", flush=True)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    with open("code/c4_panels.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
