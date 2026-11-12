"""Year-by-year IC decomposition to probe pretraining leakage.

If a model's pretraining corpus covered the test years, performance is expected to be
stronger inside that coverage and weaker after it.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
from data_pipeline import forward_labels, rank_ic, read_section  # noqa: E402

H = 5
FILES = {
    "lora_rank": "code/tsfm_lora_5_pred.npz",
    "zero_shot_group_rank": "code/tsfm_group_rank_h5_pred.npz",
    "zero_shot_univariate": "code/tsfm_mv_h5_pred.npz",
}


def load(path: str, panel: pd.DataFrame) -> pd.DataFrame:
    z = np.load(path, allow_pickle=False)
    return pd.DataFrame(z["values"], index=pd.to_datetime(z["dates"]), columns=pd.Index(z["columns"])).reindex(
        columns=panel.columns
    )


def main():
    panel = read_section("data/100_Portfolios_10x10_Daily.csv", 19).loc["2000-01-01":]
    label = forward_labels(panel, H)
    table = {}
    for name, path in FILES.items():
        pred = load(path, panel)
        ic = rank_ic(pred, label.loc[pred.index])
        by_year = ic.groupby(ic.index.year).agg(["mean", "count"])
        table[name] = by_year["mean"].round(4)
    out = pd.DataFrame(table)
    out["n_dates"] = by_year["count"]
    print(out.to_string())
    pre = out.loc[out.index <= 2024].mean(numeric_only=True)
    post = out.loc[out.index >= 2025].mean(numeric_only=True)
    print("\npre-2025 vs 2025+ IC mean:")
    print(pd.DataFrame({"pre": pre, "post": post}).round(4).to_string())


if __name__ == "__main__":
    main()