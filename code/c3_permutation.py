"""C3 control 2: series-identity permutation test.

The fitted model is permutation-equivariant across series slots (all series share one group
and the same frozen backbone), so feeding series histories in a permuted slot order yields
exactly the permuted prediction matrix. Pairing that permuted matrix with the *original*
labels therefore reproduces the effect of an identity permutation test at evaluation level.

Null: prediction-label pairing carries no cross-sectional information.
Pass criterion for C3: observed IC lies far outside the permutation null (p < 0.01).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
from data_pipeline import forward_labels, rank_ic, read_section  # noqa: E402

H = 5
N_PERM = 500
MODEL = "code/tsfm_lora_5_pred.npz"


def load(path: str, panel: pd.DataFrame) -> pd.DataFrame:
    z = np.load(path, allow_pickle=False)
    return pd.DataFrame(z["values"], index=pd.to_datetime(z["dates"]), columns=pd.Index(z["columns"]))[
        panel.columns
    ]


def main():
    panel = read_section("data/100_Portfolios_10x10_Daily.csv", 19).loc["2000-01-01":]
    pred = load(MODEL, panel)
    label = forward_labels(panel, H).loc[pred.index]
    observed = float(rank_ic(pred, label).mean())

    rng = np.random.default_rng(11)
    n_series = pred.shape[1]
    null = []
    for _ in range(N_PERM):
        perm = rng.permutation(n_series)
        shuffled = pd.DataFrame(pred.to_numpy()[:, perm], index=pred.index, columns=pred.columns)
        null.append(float(rank_ic(shuffled, label).mean()))
    null = np.array(null)

    p_value = float((1 + (null >= observed).sum()) / (N_PERM + 1))
    out = {
        "h": H,
        "n_permutations": N_PERM,
        "observed_ic": round(observed, 4),
        "null_mean": round(float(null.mean()), 4),
        "null_sd": round(float(null.std(ddof=1)), 4),
        "null_q99": round(float(np.percentile(null, 99)), 4),
        "p_value": round(p_value, 4),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open("code/c3_permutation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
