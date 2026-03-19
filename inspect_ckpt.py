import torch
import sys

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
model_keys = list(ckpt["model"].keys())
for k in model_keys:
    print(k, ckpt["model"][k].shape)
