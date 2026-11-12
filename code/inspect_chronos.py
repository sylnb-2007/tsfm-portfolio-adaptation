import torch
from chronos import Chronos2Pipeline

pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", dtype=torch.float32)
model = pipe.model
print("class", type(model).__name__)
names = [n for n, _ in model.named_modules()]
print("n_modules", len(names))
for n in names[:40]:
    print(n)
print("--- linear-ish layers containing q/k/v/proj ---")
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and any(t in n for t in ("q", "k", "v", "proj", "o.")):
        print(n, m.in_features, m.out_features)
