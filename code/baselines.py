"""M4 Phase B: linear and tree baselines on the French 100-portfolio panel.

Evaluation follows M3:
- expanding window, yearly test blocks, purge >= 21 days
- label: 5-day cumulative return starting at t+1
- portfolio: signal at close t, execution at close t+1, returns from t+2, h-day non-overlapping holding
- two weight schemes: binary top/bottom-k, and rank-weighted (turnover reduction option)
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "code")
from data_pipeline import (  # noqa: E402
    annualised_sharpe,
    forward_labels,
    read_section,
    rank_ic,
    trailing_features,
    walk_forward,
)

H = int(os.environ.get("BS_H", "5"))
PURGE = 21
K = 20
START_YEAR = int(os.environ.get("BS_START_YEAR", "2002"))
N_TREES = int(os.environ.get("BS_TREES", "150"))
ALIGNED = os.environ.get("BS_ALIGNED", "0") == "1"
TAG = os.environ.get("BS_TAG", "")


def build_aligned_features(panel: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Raw return windows ending at t, matching the information a TSFM sees: 60 + 252 lags + realised vol."""
    lr = np.log1p(panel.to_numpy(dtype=np.float64))
    T, N = lr.shape
    F = 60 + 252 + 1
    X = np.full((T, N, F), np.nan, dtype=np.float32)
    for j in range(60):
        lag = 59 - j
        if lag == 0:
            X[:, :, j] = lr
        else:
            X[lag:, :, j] = lr[: T - lag]
    for j in range(252):
        lag = 251 - j
        if lag == 0:
            X[:, :, 60 + j] = lr
        else:
            X[lag:, :, 60 + j] = lr[: T - lag]
    X[:, :, 312] = np.log1p(panel).rolling(21).std().to_numpy(dtype=np.float32)
    names = [f"lag_{j}" for j in range(60 + 252)] + ["vol_21"]
    return X, names


def build_xy(panel: pd.DataFrame, aligned: bool = False):
    if aligned:
        X3, names = build_aligned_features(panel)
    else:
        feats = trailing_features(panel)
        names = list(feats)
        X3 = np.stack([feats[k].to_numpy() for k in names], axis=-1)
    label = forward_labels(panel, H).to_numpy()
    return X3, label, names


def portfolio_weights(pred_row: np.ndarray, scheme: str, k: int, columns: pd.Index) -> tuple[pd.Series | None, float]:
    valid = np.where(~np.isnan(pred_row))[0]
    if len(valid) < 2 * k:
        return None, np.nan
    p = pd.Series(pred_row[valid], index=columns[valid])
    order = p.sort_values()
    w = pd.Series(0.0, index=columns)
    if scheme == "binary":
        w.loc[order.index[-k:]] = 1.0 / k
        w.loc[order.index[:k]] = -1.0 / k
    else:  # rank-weighted: demeaned cross-sectional rank, gross exposure normalised to 2
        r = p.rank()
        r = r - r.mean()
        gross = r.abs().sum()
        if gross == 0:
            return None, np.nan
        w.loc[p.index] = (r / gross) * 2.0
    return w, float(w.abs().sum())


def backtest(pred: pd.DataFrame, ret: pd.DataFrame, scheme: str, cost_bps: float, k: int = K, h: int | None = None) -> dict:
    h = H if h is None else h
    dates = list(pred.index)
    rows, turnovers = [], []
    w_prev = None
    for i in range(0, len(dates), h):
        w, gross_expo = portfolio_weights(pred.loc[dates[i]].to_numpy(), scheme, k, pred.columns)
        if w is None:
            continue
        base = pd.Series(0.0, index=w.index) if w_prev is None else w_prev
        turnover = float((w - base).abs().sum())
        turnovers.append(turnover)
        w_prev = w
        total_cost = cost_bps / 1e4 * turnover
        for d in dates[i + 2: i + 2 + h]:
            if d not in ret.index:
                continue
            rows.append((d, float((w * ret.loc[d].fillna(0.0)).sum()), total_cost / h))
    if not rows:
        return {"error": "no tradable days"}
    df = pd.DataFrame(rows, columns=["date", "gross", "cost"]).set_index("date")
    df["net"] = df["gross"] - df["cost"]
    return {
        "days": int(df.shape[0]),
        "turnover_per_rebalance": round(float(np.mean(turnovers)), 3) if turnovers else np.nan,
        "gross_sharpe": round(annualised_sharpe(df["gross"]), 3),
        "net_sharpe": round(annualised_sharpe(df["net"]), 3),
        "daily_gross_bps": round(float(df["gross"].mean() * 1e4), 2),
        "daily_cost_bps": round(float(df["cost"].mean() * 1e4), 2),
    }


def build_model(model_name: str):
    """Return an unfitted estimator. Fitting happens once per walk-forward split."""
    if model_name == "ols":
        return LinearRegression()
    if model_name == "ridge":
        return Ridge(alpha=10.0)
    if model_name == "pcr":
        from sklearn.pipeline import make_pipeline

        return make_pipeline(StandardScaler(), PCA(n_components=4, random_state=0), Ridge(alpha=10.0))
    if model_name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=N_TREES, learning_rate=0.05, num_leaves=31, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=8,
        )
    if model_name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=N_TREES, depth=6, learning_rate=0.05, verbose=False,
            thread_count=8, allow_writing_files=False,
        )
    raise ValueError(model_name)


def run(model_name: str, path: str, header_line: int = 19) -> dict:
    t0 = time.time()
    panel = read_section(path, header_line).loc["2000-01-01":]
    X3, L, names = build_xy(panel, aligned=ALIGNED)
    next_day = panel.shift(-1)
    dates = panel.index

    pred_rows, pred_dates, ics = [], [], []
    for train_end, test_start, test_end in walk_forward(dates, START_YEAR, max(PURGE, H)):
        train_idx = range(0, train_end + 1)
        blocks_x, blocks_y = [], []
        for i in train_idx:
            x = X3[i]
            y = L[i]
            ok = ~np.isnan(x).any(axis=1) & ~np.isnan(y)
            if ok.sum():
                blocks_x.append(x[ok])
                blocks_y.append(y[ok])
        if not blocks_x:
            continue
        Xtr, ytr = np.vstack(blocks_x), np.concatenate(blocks_y)
        model = build_model(model_name).fit(Xtr, ytr)

        blk_pred, blk_dates = [], []
        for i in range(test_start, test_end + 1):
            x = X3[i]
            ok = ~np.isnan(x).any(axis=1)
            if ok.sum() < 2 * K:
                continue
            r = np.full(panel.shape[1], np.nan)
            r[ok] = model.predict(x[ok])
            blk_pred.append(r)
            blk_dates.append(dates[i])
        pred = pd.DataFrame(np.vstack(blk_pred), index=blk_dates, columns=panel.columns)
        label = pd.DataFrame(L, index=dates, columns=panel.columns).loc[pred.index]
        ic = rank_ic(pred, label)
        ics.append(ic)
        pred_rows.append(pred)

    all_pred = pd.concat(pred_rows)
    all_ic = pd.concat(ics)
    np.savez_compressed(
        f"code/baseline_{model_name}_{'aligned' if ALIGNED else 'raw'}_h{H}{TAG}_pred.npz",
        values=all_pred.to_numpy(),
        dates=np.array([d.strftime("%Y-%m-%d") for d in all_pred.index]),
        columns=np.array(all_pred.columns.tolist()),
    )
    out = {
        "model": model_name,
        "n_features": len(names),
        "ic_mean": round(float(all_ic.mean()), 4),
        "ic_std": round(float(all_ic.std(ddof=1)), 4),
        "icir": round(float(all_ic.mean() / all_ic.std(ddof=1)), 3),
        "ic_days": int(all_ic.shape[0]),
        "schemes": {},
    }
    for scheme in ("binary", "rank"):
        out["schemes"][scheme] = {
            f"{c}bps": backtest(all_pred, next_day, scheme, float(c)) for c in (5, 10, 20)
        }
    out["seconds"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/100_Portfolios_10x10_Daily.csv"
    models = sys.argv[2:] or ["ols", "ridge", "pcr", "lightgbm", "catboost"]
    results = [run(m, path) for m in models]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    pd.DataFrame(results).to_json("code/baseline_results.json", orient="records", indent=2)
