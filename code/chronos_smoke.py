"""M4: Chronos-2 availability smoke test (single series + grouped cross-section)."""

import time

import numpy as np
import torch
from chronos import Chronos2Pipeline

MODEL = "amazon/chronos-2"


def describe(x) -> str:
    if isinstance(x, torch.Tensor):
        return f"Tensor{tuple(x.shape)} dtype={x.dtype} device={x.device}"
    if isinstance(x, np.ndarray):
        return f"ndarray{x.shape}"
    if isinstance(x, (list, tuple)):
        inner = describe(x[0]) if len(x) else "empty"
        return f"{type(x).__name__}[{len(x)}] each {inner}"
    return f"{type(x).__name__}"


def main():
    t0 = time.time()
    pipe = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda", dtype=torch.float32)
    print("loaded", MODEL, "in", round(time.time() - t0, 1), "s")
    print("device", next(pipe.model.parameters()).device)

    rng = np.random.default_rng(0)
    single = rng.standard_normal((1, 1, 256)).astype(np.float32)
    t1 = time.time()
    out = pipe.predict(single, prediction_length=5)
    print("single-series output:", describe(out), "in", round(time.time() - t1, 2), "s")

    group = rng.standard_normal((100, 1, 128)).astype(np.float32)
    t2 = time.time()
    out_group = pipe.predict(group, prediction_length=5)
    print("group(100 series) output:", describe(out_group), "in", round(time.time() - t2, 2), "s")

    peak = torch.cuda.max_memory_allocated() / 2**20
    print("peak GPU memory MB", round(peak, 1))


if __name__ == "__main__":
    main()
