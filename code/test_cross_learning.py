"""Check whether cross_learning=True changes predictions for (N,1,L) inputs."""

import time

import numpy as np
import torch
from chronos import Chronos2Pipeline

N, L, H = 100, 252, 5
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", dtype=torch.float32)
ctx = np.random.default_rng(0).standard_normal((N, 1, L)).astype(np.float32)

t0 = time.time()
base = pipe.predict(ctx, prediction_length=H)
t1 = time.time()
cross = pipe.predict(ctx, prediction_length=H, cross_learning=True)
t2 = time.time()

a = torch.stack([r[0, 10, :] for r in base]).cpu().numpy()
b = torch.stack([r[0, 10, :] for r in cross]).cpu().numpy()
print("baseline s", round(t1 - t0, 2), "cross s", round(t2 - t1, 2))
print("max_abs_diff", float(np.abs(a - b).max()))
print("mean_abs_diff", float(np.abs(a - b).mean()))
