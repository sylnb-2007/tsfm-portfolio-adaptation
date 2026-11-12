"""M6 revision: RankIC against the earliest-tradable label (r_{t+2}..r_{t+1+h})."""

import json
import sys

sys.path.insert(0, "code")
from data_pipeline import rank_ic, read_section  # noqa: E402
from stats_m5 import load  # noqa: E402


def forward_labels_shift(panel, h, start=2):
    # cumulative simple return from t+start to t+start+h-1
    import numpy as np
    r = panel.shift(-start)
    out = (1 + r).rolling(h).apply(np.prod, raw=True) - 1
    return out.shift(-(h - 1))


def main():
    panel = read_section("data/100_Portfolios_10x10_Daily.csv", 19).loc["2000-01-01":]
    res = {}
    for h, path in [(5, "code/tsfm_lora_yearly_5_inc_pred.npz"),
                    (21, "code/tsfm_lora_yearly_21_inc_pred.npz")]:
        pred = load(path, panel)
        lab = forward_labels_shift(panel, h)
        ic = rank_ic(pred, lab)
        res[f"h{h}"] = {"earliest_tradable_label_ic": round(float(ic.mean()), 4)}
    print(json.dumps(res, indent=2))
    with open("code/m6_label_check.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
