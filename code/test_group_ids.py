"""Check semantics of group_ids for C3 control design.

Question: does group_ids=arange(N) disable cross-series attention while keeping the same
trainable parameter count (same LoRA adapter)?
"""

from __future__ import annotations

import numpy as np
import torch
from chronos import Chronos2Pipeline

N, L, H = 100, 252, 5

pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", dtype=torch.float32)
ctx = torch.tensor(np.random.default_rng(0).standard_normal((N, L)), dtype=torch.float32, device="cuda")

with torch.no_grad():
    same = pipe.model(context=ctx, group_ids=torch.zeros(N, dtype=torch.long, device="cuda"), num_output_patches=1)
    solo = pipe.model(context=ctx, group_ids=torch.arange(N, device="cuda"), num_output_patches=1)

a = same.quantile_preds[:, 10, :H].cpu().numpy()
b = solo.quantile_preds[:, 10, :H].cpu().numpy()
print("group0 vs groupsolo max_abs_diff", float(np.abs(a - b).max()))
print("group0 vs groupsolo mean_abs_diff", float(np.abs(a - b).mean()))
print("corr", float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
