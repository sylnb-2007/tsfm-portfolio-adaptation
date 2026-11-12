"""Fair MV vs UV comparison restricted to the dates both share (IC only).

Economic (portfolio) metrics are intentionally excluded: UV predictions are sparse
(every 10th trading day), while the backtest assumes contiguous holding days.
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
from data_pipeline import forward_labels, rank_ic, read_section  # noqa: E402

H = 5


def load_pred(path: str, panel: pd.DataFrame) -> pd.DataFrame:
    z = np.load(path, allow_pickle=False)
    return pd.DataFrame(z["values"], index=pd.to_datetime(z["dates"]), columns=pd.Index(z["columns"])).reindex(
        columns=panel.columns
    )


def main():
    panel = read_section("data/100_Portfolios_10x10_Daily.csv", 19).loc["2000-01-01":]
    mv = load_pred("code/tsfm_mv_h5_pred.npz", panel)
    uv = load_pred("code/tsfm_uv_h5_pred.npz", panel)
    common = mv.index.intersection(uv.index)
    label = forward_labels(panel, H).loc[common]

    ic_mv = rank_ic(mv.loc[common], label)
    ic_uv = rank_ic(uv.loc[common], label)
    paired = pd.concat([ic_mv.rename("mv"), ic_uv.rename("uv")], axis=1).dropna()
    diff = paired["mv"] - paired["uv"]

    out = {
        "common_dates": int(len(common)),
        "ic_mv": round(float(paired["mv"].mean()), 4),
        "ic_uv": round(float(paired["uv"].mean()), 4),
        "ic_mv_minus_uv": round(float(diff.mean()), 4),
        "paired_t": round(float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))), 2),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()