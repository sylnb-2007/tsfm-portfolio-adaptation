"""M4 Phase A: French 100-portfolio daily panel, leakage guards and evaluation primitives.

Design rules (from M3):
- decision at close t, execution at close t+1, label = return from t+1 to t+1+h
- expanding-window walk-forward, yearly test blocks
- purge gap of max(21, h) trading days between train end and test start
- all preprocessing fitted inside the training window only
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSING = (-99.99, -999.0)


def read_section(path: str, header_line: int) -> pd.DataFrame:
    """Read one block of a Ken French CSV. header_line is 1-indexed."""
    header = None
    dates = []
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if i == header_line:
                header = [c.strip() for c in line.rstrip("\n").split(",")]
                continue
            if i > header_line and header is not None:
                first = line.split(",", 1)[0].strip()
                if not (first.isdigit() and len(first) == 8):
                    break
                dates.append(first)
                rows.append([c.strip() for c in line.rstrip("\n").split(",")])

    values = np.array([[float(c) if c else np.nan for c in r[1:]] for r in rows], dtype=float)
    values[np.isin(values, MISSING)] = np.nan
    df = pd.DataFrame(values / 100.0, index=pd.to_datetime(dates, format="%Y%m%d"), columns=header[1:])
    return df.sort_index()


def forward_labels(returns: pd.DataFrame, h: int) -> pd.DataFrame:
    """Cumulative return over t+1..t+h (execution lag of one day)."""
    logret = np.log1p(returns)
    total = sum(logret.shift(-k) for k in range(1, h + 1))
    return np.expm1(total)


def trailing_features(returns: pd.DataFrame, windows=(5, 21, 252)) -> dict[str, pd.DataFrame]:
    logret = np.log1p(returns)
    feats = {f"mom_{w}": np.expm1(logret.rolling(w).sum()) for w in windows}
    feats["vol_21"] = logret.rolling(21).std()
    return feats


def walk_forward(dates: pd.DatetimeIndex, start_year: int, purge: int):
    """Yield (train_end_pos, test_start_pos, test_end_pos) for each evaluation year."""
    years = sorted({d.year for d in dates})
    for y in years:
        if y < start_year:
            continue
        test_pos = np.where(dates.year == y)[0]
        if len(test_pos) == 0:
            continue
        test_start, test_end = int(test_pos[0]), int(test_pos[-1])
        train_end = test_start - purge - 1
        if train_end <= 0:
            continue
        yield train_end, test_start, test_end


def rank_ic(pred: pd.DataFrame, label: pd.DataFrame) -> pd.Series:
    """Per-date Spearman correlation across the portfolio cross-section."""
    out = {}
    common = pred.index.intersection(label.index)
    for t in common:
        p = pred.loc[t]
        r = label.loc[t]
        ok = p.notna() & r.notna()
        if ok.sum() >= 20:
            out[t] = p[ok].rank().corr(r[ok].rank())
    return pd.Series(out).dropna()


def annualised_sharpe(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    sd = x.std(ddof=1)
    return float(np.sqrt(252) * x.mean() / sd) if sd and sd > 0 else np.nan


def holding_backtest(pred: pd.DataFrame, daily_ret: pd.DataFrame, h: int = 5, k: int = 20, cost_bps: float = 10.0) -> dict:
    """Non-overlapping h-day holding. Signal at close t, execution at close t+1, returns counted from t+2."""
    dates = list(pred.index)
    rows, turnovers = [], []
    w_prev = None
    for i in range(0, len(dates), h):
        t = dates[i]
        p = pred.loc[t]
        ok = p.notna()
        if ok.sum() < 2 * k:
            continue
        order = p[ok].sort_values()
        w = pd.Series(0.0, index=pred.columns)
        w[order.index[-k:]] = 1.0 / k
        w[order.index[:k]] = -1.0 / k
        base = pd.Series(0.0, index=w.index) if w_prev is None else w_prev
        turnover = float((w - base).abs().sum())
        turnovers.append(turnover)
        w_prev = w
        total_cost = cost_bps / 1e4 * turnover
        for d in dates[i + 2: i + 2 + h]:
            if d not in daily_ret.index:
                continue
            gross = float((w * daily_ret.loc[d].fillna(0.0)).sum())
            rows.append((d, gross, total_cost / h))
    if not rows:
        return {"error": "no tradable days"}
    df = pd.DataFrame(rows, columns=["date", "gross", "cost"]).set_index("date")
    df["net"] = df["gross"] - df["cost"]
    return {
        "n_rebalances": len(turnovers),
        "n_holding_days": int(df.shape[0]),
        "avg_turnover_per_rebalance": round(float(np.mean(turnovers)), 3),
        "gross_sharpe": round(annualised_sharpe(df["gross"]), 3),
        "net_sharpe": round(annualised_sharpe(df["net"]), 3),
        "avg_daily_gross_bps": round(float(df["gross"].mean() * 1e4), 2),
        "avg_daily_cost_bps": round(float(df["cost"].mean() * 1e4), 2),
    }


def pipeline_check(path: str, header_line: int = 19, start: str = "2000-01-01") -> dict:
    panel = read_section(path, header_line)
    panel = panel.loc[start:]
    clean = panel.dropna(how="all")
    nan_rate = float(clean.isna().mean().mean())

    h = 5
    label = forward_labels(panel, h)
    feats = trailing_features(panel)

    splits = list(walk_forward(panel.index, start_year=2001, purge=max(21, h)))

    # leakage assertion: label at train_end must not overlap the test block
    checks = []
    for train_end, test_start, _ in splits[:3]:
        checks.append(test_start - train_end > max(21, h))

    rng = np.random.default_rng(7)
    first = splits[0]
    train_end, test_start, test_end = first
    test_dates = panel.index[test_start:test_end + 1]
    random_pred = pd.DataFrame(
        rng.standard_normal((len(test_dates), panel.shape[1]), dtype=np.float32),
        index=test_dates, columns=panel.columns,
    )
    ic_random = rank_ic(random_pred, label)

    return {
        "panel_shape": tuple(panel.shape),
        "date_range": (str(panel.index[0].date()), str(panel.index[-1].date())),
        "nan_rate_after_dropna_rows": round(nan_rate, 4),
        "n_splits": len(splits),
        "first_split": (str(panel.index[train_end].date()), str(panel.index[test_start].date()), str(panel.index[test_end].date())),
        "purge_ok_first3": checks,
        "random_ic_mean": round(float(ic_random.mean()), 4),
        "random_ic_abs_mean": round(float(ic_random.abs().mean()), 4),
        "random_ic_obs": int(ic_random.shape[0]),
    }


def ridge_smoke(path: str, header_line: int = 19, start: str = "2000-01-01", year: int = 2010, h: int = 5) -> dict:
    """End-to-end functional check: fit pooled Ridge in-sample, predict one test year, then trade."""
    from sklearn.linear_model import Ridge

    panel = read_section(path, header_line).loc[start:]
    label = forward_labels(panel, h)
    feats = trailing_features(panel)
    next_day = panel.shift(-1)

    names = list(feats)
    X3 = np.stack([feats[k].to_numpy() for k in names], axis=-1)  # (T, N, F)
    L = label.to_numpy()

    splits = list(walk_forward(panel.index, start_year=year, purge=max(21, h)))
    train_end, test_start, test_end = splits[0]
    dates = panel.index

    train_idx = list(range(0, train_end + 1))
    test_idx = list(range(test_start, test_end + 1))

    # z-score statistics fitted on training window only
    blocks = [X3[i] for i in train_idx if not np.isnan(X3[i]).all()]
    stacked = np.vstack(blocks)
    mu = np.nanmean(stacked, axis=0)
    sd = np.nanstd(stacked, axis=0)
    sd[sd == 0] = np.nan

    rows_x, rows_y = [], []
    for i in train_idx:
        x = (X3[i] - mu) / sd
        y = L[i]
        ok = ~np.isnan(x).any(axis=1) & ~np.isnan(y)
        if ok.sum() == 0:
            continue
        rows_x.append(x[ok])
        rows_y.append(y[ok])
    if not rows_x:
        return {"error": "no usable training rows"}
    model = Ridge(alpha=1.0).fit(np.vstack(rows_x), np.concatenate(rows_y))

    pred = []
    pred_index = []
    for i in test_idx:
        x = (X3[i] - mu) / sd
        ok = ~np.isnan(x).any(axis=1)
        if ok.sum() == 0:
            continue
        row = np.full(panel.shape[1], np.nan)
        row[ok] = model.predict(x[ok])
        pred.append(row)
        pred_index.append(dates[i])
    pred = pd.DataFrame(np.vstack(pred), index=pred_index, columns=panel.columns)

    ic = rank_ic(pred, label.loc[pred.index])
    backtests = {
        f"{c}bps": holding_backtest(pred, next_day, h=h, k=20, cost_bps=float(c))
        for c in (5, 10, 20)
    }

    return {
        "test_year": int(year),
        "train_end": str(dates[train_end].date()),
        "test_start": str(dates[test_start].date()),
        "test_end": str(dates[test_end].date()),
        "n_features": len(names),
        "n_train_rows": int(sum(len(r) for r in rows_y)),
        "ic_mean": round(float(ic.mean()), 4),
        "ic_obs": int(ic.shape[0]),
        "backtest": backtests,
    }


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/100_Portfolios_10x10_Daily.csv"
    if len(sys.argv) > 2 and sys.argv[2] == "smoke":
        print(json.dumps(ridge_smoke(path), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(pipeline_check(path), ensure_ascii=False, indent=2))
