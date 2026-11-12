"""Inspect the differentiable forward: tensor shapes for training with module B."""

import math

import numpy as np
import torch
from chronos import Chronos2Pipeline

pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", dtype=torch.float32)
model = pipe.model
print("context_length", pipe.model_context_length)
print("output_patch_size", pipe.model_output_patch_size)

N, L, H = 100, 252, 5
ctx = torch.randn(N, L, device="cuda", dtype=torch.float32)
gids = torch.zeros(N, dtype=torch.long, device="cuda")
n_patches = math.ceil(H / pipe.model_output_patch_size)

out = model(context=ctx, group_ids=gids, num_output_patches=n_patches)
print("out type", type(out).__name__)
print("keys", [k for k in dir(out) if not k.startswith("_")][:12])
print("quantile_preds", tuple(out.quantile_preds.shape), out.quantile_preds.dtype)
