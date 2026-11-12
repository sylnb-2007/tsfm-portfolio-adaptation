"""Module B: grouped LoRA adaptation of Chronos-2 on the cross-sectional panel.

Approach:
- manual LoRA on the T5 self-attention q/v projections (no peft wrapper, because the
  model forward uses a custom signature: context, group_ids, num_output_patches)
- input uses Module A rank normalisation
- target: h-day forward log return per series
- train on 2001-2014, evaluate 2015-2026 (single split for the M4 prototype)
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
import torch.nn as nn
from chronos import Chronos2Pipeline

sys.path.insert(0, "code")
from baselines import backtest  # noqa: E402
from data_pipeline import forward_labels, rank_ic, read_section, trailing_features  # noqa: E402

H = int(os.environ.get("BS_H", "5"))
CONTEXT = int(os.environ.get("BS_CONTEXT", "252"))
TRAIN_END_YEAR = int(os.environ.get("BS_TRAIN_END", "2014"))
TEST_START_YEAR = int(os.environ.get("BS_START_YEAR", "2015"))
EPOCHS = int(os.environ.get("BS_LORA_EPOCHS", "1"))
STRIDE = int(os.environ.get("BS_LORA_STRIDE", "5"))
LR = float(os.environ.get("BS_LORA_LR", "1e-3"))
RANK_R = 8
GROUP_MODE = os.environ.get("BS_GROUP_MODE", "pooled")  # pooled = one group (cross-series), solo = isolated series
SUFFIX = "" if GROUP_MODE == "pooled" else f"_{GROUP_MODE}"
MEDIAN_INDEX = 10


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def inject_lora(model: nn.Module) -> int:
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in ("q", "v"):
                setattr(module, child_name, LoRALinear(child))
                count += 1
    return count


def rank_transform(ctx: np.ndarray) -> np.ndarray:
    ranks = pd.DataFrame(ctx).rank(axis=0).to_numpy()
    return ((ranks / (ranks.shape[0] + 1.0)) * 2.0 - 1.0).astype(np.float32)


def build_targets(logret: np.ndarray, t: int, h: int) -> np.ndarray:
    end = t + h + 1
    return logret[t + 1:end].sum(axis=0) if end <= logret.shape[0] else np.full(logret.shape[1], np.nan)


def main():
    t0 = time.time()
    panel = read_section("data/100_Portfolios_10x10_Daily.csv", 19).loc["2000-01-01":]
    logret = pd.DataFrame(np.log1p(panel)).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
    dates = panel.index
    n_dates, n_series = logret.shape

    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", dtype=torch.float32)
    model = pipe.model
    model.requires_grad_(False)  # strict LoRA: freeze all backbone weights before injecting adapters
    n_lora = inject_lora(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in trainable)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"lora layers {n_lora}, trainable tensors {len(trainable)}, trainable params {trainable_params}/{total_params} ({trainable_params/total_params*100:.2f}%)", flush=True)

    opt = torch.optim.AdamW(trainable, lr=LR)
    n_patches = math.ceil(H / pipe.model_output_patch_size)

    # ---- training ----
    train_indices = [
        i for i in range(CONTEXT - 1, n_dates)
        if dates[i].year <= TRAIN_END_YEAR and i + H < n_dates and dates[i + H].year <= TRAIN_END_YEAR
    ][::STRIDE]
    print(f"training samples {len(train_indices)}", flush=True)

    gids = (
        torch.zeros(n_series, dtype=torch.long, device="cuda")
        if GROUP_MODE == "pooled"
        else torch.arange(n_series, dtype=torch.long, device="cuda")
    )
    for epoch in range(EPOCHS):
        running = 0.0
        for step, i in enumerate(train_indices):
            ctx = rank_transform(logret[i - CONTEXT + 1:i + 1].T)
            tgt_raw = build_targets(logret, i, H)
            mask = torch.tensor(~np.isnan(tgt_raw), dtype=torch.float32, device="cuda")
            tgt = torch.tensor(np.nan_to_num(tgt_raw), dtype=torch.float32, device="cuda")
            out = model(context=torch.tensor(ctx, device="cuda"), group_ids=gids, num_output_patches=n_patches)
            pred = out.quantile_preds[:, MEDIAN_INDEX, :H].sum(dim=1)
            sq = (pred - tgt) ** 2
            loss = (sq * mask).sum() / mask.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            if (step + 1) % 100 == 0:
                print(f"epoch {epoch} step {step+1}/{len(train_indices)} loss {running/100:.5f}", flush=True)
                running = 0.0
    print("training done", round(time.time() - t0, 1), "s", flush=True)

    # ---- evaluation ----
    pred_rows, pred_dates = [], []
    model.eval()
    with torch.no_grad():
        for i in range(n_dates):
            if dates[i].year < TEST_START_YEAR or i < CONTEXT - 1:
                continue
            ctx = rank_transform(logret[i - CONTEXT + 1:i + 1].T)
            out = model(context=torch.tensor(ctx, device="cuda"), group_ids=gids, num_output_patches=n_patches)
            pred_rows.append(out.quantile_preds[:, MEDIAN_INDEX, :H].sum(dim=1).cpu().numpy())
            pred_dates.append(dates[i])

    pred = pd.DataFrame(np.vstack(pred_rows), index=pred_dates, columns=panel.columns)
    np.savez_compressed(
        f"code/tsfm_lora_{H}{SUFFIX}_pred.npz",
        values=pred.to_numpy(),
        dates=np.array([d.strftime("%Y-%m-%d") for d in pred.index]),
        columns=np.array(pred.columns.tolist()),
    )
    label = forward_labels(panel, H).loc[pred.index]
    ic = rank_ic(pred, label)
    next_day = panel.shift(-1)

    out = {
        "mode": "lora_group_rank",
        "group_mode": GROUP_MODE,
        "h": H,
        "train_end_year": TRAIN_END_YEAR,
        "test_start_year": TEST_START_YEAR,
        "epochs": EPOCHS,
        "train_stride": STRIDE,
        "n_train_samples": len(train_indices),
        "n_pred_dates": int(pred.shape[0]),
        "ic_mean": round(float(ic.mean()), 4),
        "ic_std": round(float(ic.std(ddof=1)), 4),
        "icir": round(float(ic.mean() / ic.std(ddof=1)), 3),
        "schemes": {
            scheme: {f"{c}bps": backtest(pred, next_day, scheme, float(c)) for c in (5, 10, 20)}
            for scheme in ("binary", "rank")
        },
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(f"code/tsfm_lora_{H}{SUFFIX}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()