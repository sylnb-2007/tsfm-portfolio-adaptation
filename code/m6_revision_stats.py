"""M6 revision: paired tests vs zero-shot, execution-lag sensitivity, leakage check.

Reads only saved prediction matrices; no retraining.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
from data_pipeline import forward_labels, rank_ic, read_section  # noqa: E402
from stats_m5 import block_bootstrap_ci, daily_net, deflated_sharpe, load, nw_tstat  # noqa: E402

K = 20


def net_series(pred, panel, h, ret_shift, cost_bps=10.0):
    """ret_shift=True -> main (r_{t+3} first); False -> earliest tradable (r_{t+2})."""
    dates = list(pred.index)
    ret = panel.shift(-1) if ret_shift else panel
    rows, w_prev = [], None
    for i in range(0, len(dates), h):
        p = pred.loc[dates[i]]
        valid = p.dropna()
        if len(valid) < 2 * K:
            continue
        w = pd.Series(0.0, index=pred.columns)
        r = valid.rank() - valid.rank().mean()
        w.loc[valid.index] = (r / r.abs().sum()) * 2.0
        base = pd.Series(0.0, index=w.index) if w_prev is None else w_prev
        turnover = float((w - base).abs().sum())
        w_prev = w
        for d in dates[i + 2:i + 2 + h]:
            if d in ret.index:
                rows.append((d, float((w * ret.loc[d].fillna(0.0)).sum()) - cost_bps / 1e4 * turnover / h))
    return pd.Series({d: v for d, v in rows}).sort_index()


def econ_block(pred, panel, h, ret_shift):
    net = net_series(pred, panel, h, ret_shift, 10.0)
    gross = net_series(pred, panel, h, ret_shift, 0.0)
    sr = float(np.sqrt(252) * net.mean() / net.std(ddof=1))
    # break-even: gross/day = c * turnover/h
    dates = list(pred.index)
    tos, w_prev = [], None
    for i in range(0, len(dates), h):
        p = pred.loc[dates[i]]
        valid = p.dropna()
        if len(valid) < 2 * K:
            continue
        r = valid.rank() - valid.rank().mean()
        w = (r / r.abs().sum()) * 2.0
        base = pd.Series(0.0, index=w.index) if w_prev is None else w_prev
        tos.append(float((w - base).abs().sum()))
        w_prev = w
    to = float(np.mean(tos))
    be = float(gross.mean() * 1e4 * h / to) if to > 0 else float("nan")
    return {"net_bps": round(float(net.mean() * 1e4), 2), "net_sharpe": round(sr, 3),
            "gross_bps": round(float(gross.mean() * 1e4), 2), "turnover": round(to, 3),
            "break_even_bps": round(be, 1), "_net": net}


def main():
    panel = read_section("data/100_Portfolios_10x10_Daily.csv", 19).loc["2000-01-01":]
    out = {}

    # --- paired IC vs zero-shot ---
    for h, pairs in [
        (5, [("lora_inc", "code/tsfm_lora_yearly_5_inc_pred.npz"),
             ("zs_uni", "code/tsfm_mv_h5_pred.npz"),
             ("zs_group", "code/tsfm_group_rank_h5_pred.npz")]),
        (21, [("lora_inc", "code/tsfm_lora_yearly_21_inc_pred.npz"),
              ("zs_uni", "code/tsfm_mv_h21_pred.npz")]),
    ]:
        label = forward_labels(panel, h)
        preds = {n: load(p, panel) for n, p in pairs}
        common = preds["lora_inc"].index
        for n in preds:
            common = common.intersection(preds[n].index)
        common = common.sort_values()
        ics = {n: rank_ic(p.loc[common], label.loc[common]) for n, p in preds.items()}
        res = {}
        for n in ics:
            if n == "lora_inc":
                continue
            d = (ics["lora_inc"] - ics[n]).dropna()
            lo, hi = block_bootstrap_ci(d)
            res[f"lora_inc - {n}"] = {"mean_diff": round(float(d.mean()), 4),
                                      "nw_t": round(nw_tstat(d, h), 2),
                                      "ci95": [round(lo, 4), round(hi, 4)]}
        out[f"paired_vs_zeroshot_h{h}"] = res

    # --- execution-lag sensitivity (Table 9) ---
    for h, path, n_trials in [(5, "code/tsfm_lora_yearly_5_inc_pred.npz", 5),
                              (21, "code/tsfm_lora_yearly_21_inc_pred.npz", 4)]:
        pred = load(path, panel)
        row = {}
        nets = {}
        for name, rs in [("main", True), ("earliest", False)]:
            eb = econ_block(pred, panel, h, rs)
            nets[name] = eb.pop("_net")
            row[name] = eb
        # DSR under each convention: sr_var across the model set under that convention
        model_paths = ([("code/tsfm_lora_yearly_5_inc_pred.npz"), ("code/tsfm_lora_yearly_5_reset_pred.npz"),
                        ("code/tsfm_mv_h5_pred.npz"), ("code/tsfm_group_rank_h5_pred.npz"),
                        ("code/baseline_ols_aligned_h5_pred.npz")] if h == 5 else
                       [("code/tsfm_lora_yearly_21_inc_pred.npz"), ("code/tsfm_lora_21_pred.npz"),
                        ("code/tsfm_mv_h21_pred.npz"), ("code/baseline_ols_aligned_h21_pred.npz")])
        for name, rs in [("main", True), ("earliest", False)]:
            srs = []
            for mp in model_paths:
                ns = net_series(load(mp, panel), panel, h, rs, 10.0)
                srs.append(float(np.sqrt(252) * ns.mean() / ns.std(ddof=1)))
            n = nets[name]
            sr = float(np.sqrt(252) * n.mean() / n.std(ddof=1))
            row[name]["dsr"] = round(deflated_sharpe(sr, n_trials, float(np.var(srs, ddof=1)),
                                                     float(n.skew()), float(n.kurtosis() + 3)), 3)
        out[f"exec_lag_h{h}"] = row

    # --- leakage symptom check: IC before/after 2025 ---
    for h, path in [(5, "code/tsfm_lora_yearly_5_inc_pred.npz"),
                    (21, "code/tsfm_lora_yearly_21_inc_pred.npz")]:
        label = forward_labels(panel, h)
        pred = load(path, panel)
        ic = rank_ic(pred, label)
        pre = ic.loc[ic.index < "2025-01-01"]
        post = ic.loc[ic.index >= "2025-01-01"]
        out[f"leakage_check_h{h}"] = {"pre_2025_ic": round(float(pre.mean()), 4),
                                      "post_2025_ic": round(float(post.mean()), 4)}

    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open("code/m6_revision_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
