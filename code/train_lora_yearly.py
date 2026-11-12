"""Yearly walk-forward LoRA retraining, aligned with the baseline protocol.

For every test year Y: reset LoRA adapters, train on all data with labels ending before
January 1 of Y, then predict year Y only. Predictions are concatenated and evaluated once.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

sys.path.insert(0, "code")
from baselines import backtest  # noqa: E402
from data_pipeline import forward_labels, rank_ic, read_section  # noqa: E402
from train_lora import LoRALinear, build_targets, inject_lora, rank_transform  # noqa: E402

H = int(os.environ.get("BS_H", "5"))
PANEL = os.environ.get("BS_PANEL", "data/100_Portfolios_10x10_Daily.csv")
SECTION = int(os.environ.get("BS_SECTION", "19"))
TAG = os.environ.get("BS_TAG", "")
CONTEXT = int(os.environ.get("BS_CONTEXT", "252"))
FIRST_TEST_YEAR = int(os.environ.get("BS_FIRST_TEST_YEAR", "2015"))
STRIDE = int(os.environ.get("BS_LORA_STRIDE", "10"))
EPOCHS = int(os.environ.get("BS_LORA_EPOCHS", "1"))
LR = float(os.environ.get("BS_LORA_LR", "1e-3"))
RESET = os.environ.get("BS_YEARLY_RESET", "1") == "1"
MEDIAN_INDEX = 10


def reset_lora(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            torch.nn.init.kaiming_uniform_(module.A, a=math.sqrt(5))
            module.B.data.zero_()


def load_pipe():
    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", dtype=torch.float32)


def main():
    t0 = time.time()
    panel = read_section(PANEL, SECTION).loc["2000-01-01":]
    logret = pd.DataFrame(np.log1p(panel)).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
    dates = panel.index
    n_dates, n_series = logret.shape

    pipe = load_pipe()
    model = pipe.model
    model.requires_grad_(False)  # strict LoRA: freeze all backbone weights before injecting adapters
    n_lora = inject_lora(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in trainable)
    total_params = sum(p.numel() for p in model.parameters())
    n_patches = math.ceil(H / pipe.model_output_patch_size)
    gids = torch.zeros(n_series, dtype=torch.long, device="cuda")
    print(f"lora layers {n_lora}, trainable tensors {len(trainable)}, trainable params {trainable_params}/{total_params} ({trainable_params/total_params*100:.2f}%)", flush=True)

    pred_rows, pred_dates, year_log = [], [], []
    test_years = sorted({d.year for d in dates if d.year >= FIRST_TEST_YEAR})

    for year in test_years:
        if RESET:
            reset_lora(model)
        opt = torch.optim.AdamW(trainable, lr=LR)
        train_indices = [
            i for i in range(CONTEXT - 1, n_dates)
            if dates[i].year < year and i + H < n_dates and dates[i + H].year < year
        ][::STRIDE]
        model.train()
        for epoch in range(EPOCHS):
            for step, i in enumerate(train_indices):
                ctx = rank_transform(logret[i - CONTEXT + 1:i + 1].T)
                tgt_raw = build_targets(logret, i, H)
                mask = torch.tensor(~np.isnan(tgt_raw), dtype=torch.float32, device="cuda")
                tgt = torch.tensor(np.nan_to_num(tgt_raw), dtype=torch.float32, device="cuda")
                out = model(context=torch.tensor(ctx, device="cuda"), group_ids=gids, num_output_patches=n_patches)
                pred = out.quantile_preds[:, MEDIAN_INDEX, :H].sum(dim=1)
                loss = ((pred - tgt) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        year_rows = []
        with torch.no_grad():
            for i in range(n_dates):
                if dates[i].year != year or i < CONTEXT - 1:
                    continue
                ctx = rank_transform(logret[i - CONTEXT + 1:i + 1].T)
                out = model(context=torch.tensor(ctx, device="cuda"), group_ids=gids, num_output_patches=n_patches)
                year_rows.append(out.quantile_preds[:, MEDIAN_INDEX, :H].sum(dim=1).cpu().numpy())
                pred_dates.append(dates[i])
        pred_rows.extend(year_rows)
        year_log.append({"year": year, "n_train": len(train_indices), "n_test": len(year_rows)})
        print(f"year {year}: train {len(train_indices)} test {len(year_rows)}", flush=True)

    pred = pd.DataFrame(np.vstack(pred_rows), index=pred_dates, columns=panel.columns)
    np.savez_compressed(
        f"code/tsfm_lora_yearly_{H}_{'reset' if RESET else 'inc'}{TAG}_pred.npz",
        values=pred.to_numpy(),
        dates=np.array([d.strftime("%Y-%m-%d") for d in pred.index]),
        columns=np.array(pred.columns.tolist()),
    )
    label = forward_labels(panel, H).loc[pred.index]
    ic = rank_ic(pred, label)
    by_year = ic.groupby(ic.index.year).mean().round(4).to_dict()
    next_day = panel.shift(-1)

    out = {
        "mode": "lora_yearly",
        "panel": PANEL,
        "section_line": SECTION,
        "reset_each_year": RESET,
        "h": H,
        "first_test_year": FIRST_TEST_YEAR,
        "stride": STRIDE,
        "epochs": EPOCHS,
        "n_pred_dates": int(pred.shape[0]),
        "ic_mean": round(float(ic.mean()), 4),
        "ic_std": round(float(ic.std(ddof=1)), 4),
        "icir": round(float(ic.mean() / ic.std(ddof=1)), 3),
        "ic_by_year": by_year,
        "schemes": {
            scheme: {f"{c}bps": backtest(pred, next_day, scheme, float(c)) for c in (5, 10, 20)}
            for scheme in ("binary", "rank")
        },
        "year_log": year_log,
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps({k: v for k, v in out.items() if k != "year_log"}, ensure_ascii=False, indent=2))
    with open(f"code/tsfm_lora_yearly_{H}_{'reset' if RESET else 'inc'}{TAG}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()