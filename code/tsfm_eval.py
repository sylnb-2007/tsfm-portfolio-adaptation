"""M4: Chronos-2 zero-shot evaluation on the French 100-portfolio panel.

Modes:
- mv: all 100 series fed as one group (cross-sectional information sharing)
- uv: each series forecast alone on a subsampled set of dates (reference, slow)

Protocol follows M3: walk-forward, purge >= max(21, h), signal at close t, execution at t+1,
returns counted from t+2, h-day non-overlapping holding, turnover-based cost.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
from baselines import backtest  # noqa: E402
from data_pipeline import forward_labels, rank_ic, read_section, walk_forward  # noqa: E402

H = int(os.environ.get("BS_H", "5"))
START_YEAR = int(os.environ.get("BS_START_YEAR", "2015"))
CONTEXT = int(os.environ.get("BS_CONTEXT", "252"))
MODE = os.environ.get("BS_TSFM_MODE", "mv")
NORM = os.environ.get("BS_NORM", "none")
UV_STRIDE = int(os.environ.get("BS_UV_STRIDE", "5"))
MODEL = "amazon/chronos-2"
MEDIAN_INDEX = 10  # 21 quantile levels -> index 10 is the median


def transform_context(ctx: np.ndarray, mode: str) -> np.ndarray:
    """Module A: make the cross-section comparable before joint modelling.

    ctx: (N, L). z-score removes each series' own level/scale; rank removes
    cross-series scale differences while preserving the ordering at each date.
    """
    if mode == "none":
        return ctx
    out = ctx.astype(np.float32)
    if mode in ("zscore", "both"):
        mu = out.mean(axis=1, keepdims=True)
        sd = out.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        out = (out - mu) / sd
    if mode in ("rank", "both"):
        ranks = pd.DataFrame(out).rank(axis=0).to_numpy()  # rank across series at each time step
        out = (ranks / (ranks.shape[0] + 1.0)) * 2.0 - 1.0
    return out


def load_pipe():
    import torch
    from chronos import Chronos2Pipeline

    pipe = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda", dtype=torch.float32)
    return pipe


def predict_group(pipe, context: np.ndarray) -> np.ndarray:
    """True cross-sectional input: all N series are variates of ONE group, shape (1, N, L)."""
    arr = context[None, :, :].astype(np.float32)
    res = pipe.predict(arr, prediction_length=H)  # list of 1 tensor, (N, 21, H)
    med = res[0][:, MEDIAN_INDEX, :].cpu().numpy()  # (N, H)
    return med.sum(axis=1)


def predict_batch(pipe, context: np.ndarray) -> np.ndarray:
    """Batched independent series (N, 1, L). Verified numerically identical to univariate."""
    arr = context[:, None, :].astype(np.float32)
    res = pipe.predict(arr, prediction_length=H)  # list of N tensors, each (1, 21, H)
    med = np.vstack([r[0, MEDIAN_INDEX, :].cpu().numpy() for r in res])  # (N, H)
    return med.sum(axis=1)


def predict_cross(pipe, context: np.ndarray) -> np.ndarray:
    """Batched series forced into one shared group via cross_learning=True."""
    arr = context[:, None, :].astype(np.float32)
    res = pipe.predict(arr, prediction_length=H, cross_learning=True)
    med = np.vstack([r[0, MEDIAN_INDEX, :].cpu().numpy() for r in res])
    return med.sum(axis=1)


def predict_uv(pipe, context: np.ndarray) -> np.ndarray:
    preds = np.full(context.shape[0], np.nan, dtype=np.float32)
    for j in range(context.shape[0]):
        arr = context[j][None, None, :].astype(np.float32)
        res = pipe.predict(arr, prediction_length=H)  # list of 1 tensor (1, 21, H)
        preds[j] = float(res[0][0, MEDIAN_INDEX, :].cpu().numpy().sum())
    return preds


def main():
    t0 = time.time()
    panel = read_section("data/100_Portfolios_10x10_Daily.csv", 19).loc["2000-01-01":]
    logret = np.log1p(panel).to_numpy(dtype=np.float64)
    dates = panel.index
    next_day = panel.shift(-1)

    pipe = load_pipe()
    print("model loaded", round(time.time() - t0, 1), "s", flush=True)

    pred_rows, pred_dates = [], []
    for train_end, test_start, test_end in walk_forward(dates, START_YEAR, max(21, H)):
        positions = range(test_start, test_end + 1)
        if MODE == "uv":
            positions = list(positions)[::UV_STRIDE]
        for i in positions:
            lo = max(0, i - CONTEXT + 1)
            ctx = transform_context(logret[lo:i + 1].T, NORM)
            if ctx.shape[1] < 32:
                continue
            if MODE == "group":
                p = predict_group(pipe, ctx)
            elif MODE == "cross":
                p = predict_cross(pipe, ctx)
            elif MODE == "batch":
                p = predict_batch(pipe, ctx)
            else:
                p = predict_uv(pipe, ctx)
            pred_rows.append(p)
            pred_dates.append(dates[i])

    pred = pd.DataFrame(np.vstack(pred_rows), index=pred_dates, columns=panel.columns)
    np.savez_compressed(
        f"code/tsfm_{MODE}_{NORM}_h{H}_pred.npz",
        values=pred.to_numpy(),
        dates=np.array([d.strftime("%Y-%m-%d") for d in pred.index]),
        columns=np.array(pred.columns.tolist()),
    )
    labels = forward_labels(panel, H).loc[pred.index]
    ic = rank_ic(pred, labels)

    out = {
        "mode": MODE,
        "norm": NORM,
        "context": CONTEXT,
        "h": H,
        "start_year": START_YEAR,
        "n_pred_dates": int(pred.shape[0]),
        "ic_mean": round(float(ic.mean()), 4),
        "ic_std": round(float(ic.std(ddof=1)), 4),
        "icir": round(float(ic.mean() / ic.std(ddof=1)), 3),
        "ic_days": int(ic.shape[0]),
        "schemes": {
            scheme: {f"{c}bps": backtest(pred, next_day, scheme, float(c)) for c in (5, 10, 20)}
            for scheme in ("binary", "rank")
        },
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(f"code/tsfm_{MODE}_{NORM}_h{H}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
