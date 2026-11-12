"""Feasibility test: feed all 100 portfolios as variates of ONE group, shape (1, N, L)."""

import time

import numpy as np
import torch
from chronos import Chronos2Pipeline

N, L, H = 100, 252, 5

pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", dtype=torch.float32)
rng = np.random.default_rng(0)

for trial in range(3):
    ctx = rng.standard_normal((1, N, L)).astype(np.float32)
    t0 = time.time()
    out = pipe.predict(ctx, prediction_length=H)
    dt = time.time() - t0
    shapes = [tuple(o.shape) for o in out[:2]]
    print(f"trial {trial}: {dt:.2f}s, n_out={len(out)}, shapes={shapes}", flush=True)

print("peak GPU MB", round(torch.cuda.max_memory_allocated() / 2**20, 1))