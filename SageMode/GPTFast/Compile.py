import torch
import torch.nn as nn

def torch_compile_model(model:nn.Module) -> nn.Module:
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    return model

